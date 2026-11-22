"""Эксперимент 1: наборы описаний, их ансамбли (выбор вложенно), text-anchored голова, калибровка температуры.
LOOO по 40 предметам; все выборы — только по train-объектам фолда (вложенный 5-fold по объектам)."""
import argparse
import itertools
import json
import math
import time
from collections import Counter, defaultdict

import numpy as np

from common import (CACHE, HERE, LIGHTS, MATS, MODELS, VIEWS, Standardizer, class_weights, fit_logreg, inner_folds, l2n, load_bank,
                    predict_logreg, report, softmax, M_frame)

TB = dict(np.load(HERE / "text_bank.npz"))
SET_KEYS = sorted({k.split("|")[1] + "|" + k.split("|")[2] for k in TB if "|per|" not in k})


def zs_logits(F, skey, tags=MODELS, views=VIEWS, mode="mean"):
    """Логиты zero-shot (косинусы), усреднённые по моделям и видам. mode=mean: эмбеддинг класса = среднее описаний;
    mode=lse: logsumexp по отдельным описаниям класса (temp 0.01)."""
    out = []
    for tag in tags:
        if mode == "mean":
            T = TB[f"{tag}|{skey}"]
            for v in views:
                out.append(F[f"{tag}_{v}"] @ T.T)
        else:
            for v in views:
                E = F[f"{tag}_{v}"]
                cols = []
                for mt in MATS:
                    Pm = TB[f"{tag}|{skey}|per|{mt}"]
                    s = E @ Pm.T / 0.01
                    m = s.max(1, keepdims=True)
                    cols.append((m[:, 0] + np.log(np.exp(s - m).mean(1))) * 0.01)
                out.append(np.stack(cols, 1))
    return np.mean(out, 0)


def head_fit_predict(X, y, w, tr, te, wd=1e-2, offset=None, anchor_W=None, anchor=0.0):
    st = Standardizer(X[tr])
    Xs = st(X)
    if offset is None:
        W, b = fit_logreg(Xs[tr], y[tr], w, wd=wd, W0=anchor_W, anchor=anchor)
        return predict_logreg(Xs[te], W, b)
    W, b = fit_logreg_offset(Xs[tr], y[tr], w, offset[tr], wd=wd)
    return softmax(Xs[te] @ W + b + offset[te])


def fit_logreg_offset(X, y, w, off, wd=1e-2, epochs=300):
    import torch
    from common import DEV
    Xt = torch.tensor(X, device=DEV, dtype=torch.float32)
    yt = torch.tensor(y, device=DEV)
    wt = torch.tensor(w, device=DEV, dtype=torch.float32)
    ot = torch.tensor(off, device=DEV, dtype=torch.float32)
    W = torch.zeros(X.shape[1], 6, device=DEV, requires_grad=True)
    b = torch.zeros(6, device=DEV, requires_grad=True)
    opt = torch.optim.LBFGS([W, b], lr=1, max_iter=epochs, history_size=20, line_search_fn="strong_wolfe")

    def closure():
        opt.zero_grad()
        loss = (torch.nn.functional.cross_entropy(Xt @ W + b + ot, yt, reduction="none") * wt).sum() / wt.sum() + wd * (W ** 2).sum()
        loss.backward()
        return loss

    opt.step(closure)
    return W.detach().cpu().numpy(), b.detach().cpu().numpy()


class Runner:
    def __init__(self):
        self.F, self.meta, self.oid, self.src, self.mat, self.light = load_bank()
        self.real = self.src == "real_train"
        self.idx = np.where(self.real)[0]
        self.objs = sorted(set(self.oid[self.real]))
        self.X = (l2n(self.F["clipL_c14"]) * math.sqrt(768 / 64)).astype(np.float32)
        self.ZL = {}
        self._inner_cache = {}

    def logits(self, skey, mode="mean", views=VIEWS, tags=MODELS):
        k = (skey, mode, tuple(views), tuple(tags))
        if k not in self.ZL:
            self.ZL[k] = zs_logits(self.F, skey, tags, views, mode)
        return self.ZL[k]

    def inner_oof(self, train_objs, Xkey="clipL_c14", wd=1e-2, seed=0):
        """Inner-OOF вероятности головы для train-объектов (5-fold по объектам)."""
        key = (tuple(train_objs), Xkey, wd, seed)
        if key in self._inner_cache:
            return self._inner_cache[key]
        X = self.X if Xkey == "clipL_c14" else self.feat(Xkey)
        P = np.zeros((len(self.mat), 6))
        for g in inner_folds(train_objs, 5, seed):
            a = np.where(np.isin(self.oid, train_objs) & ~np.isin(self.oid, g) & self.real)[0]
            b = np.where(np.isin(self.oid, g) & self.real)[0]
            P[b] = head_fit_predict(X, self.mat, class_weights(a, self.oid, self.src, self.mat), a, b, wd=wd)
        self._inner_cache[key] = P
        return P

    def feat(self, key):
        parts = []
        for k in key.split("+"):
            parts.append(l2n(self.F[k]) * math.sqrt(768 / 64))
        return np.concatenate(parts, 1).astype(np.float32)

    def head_full(self, train_objs, te, Xkey="clipL_c14", wd=1e-2):
        X = self.X if Xkey == "clipL_c14" else self.feat(Xkey)
        tr = np.where(np.isin(self.oid, train_objs) & self.real)[0]
        return head_fit_predict(X, self.mat, class_weights(tr, self.oid, self.src, self.mat), tr, te, wd=wd)

    def run(self, method, name, **kw):
        t0 = time.time()
        P = np.zeros((len(self.mat), 6))
        picks = []
        for o in self.objs:
            train_objs = [q for q in self.objs if q != o]
            te = np.where((self.oid == o) & self.real)[0]
            P[te], pk = method(self, train_objs, te, **kw)
            picks.append(pk)
        r = report(name, P, self.mat, self.oid, self.idx, self.light)
        cnt = Counter(json.dumps(p, sort_keys=True, default=str) for p in picks)
        print(f"      выбор: {cnt.most_common(4)}  ({time.time() - t0:.0f}s)", flush=True)
        r["picks"] = cnt.most_common(6)
        return P, r


W_GRID = [0.0, 0.25, 0.4, 0.5, 0.6, 0.75, 1.0]
T_GRID = [0.005, 0.01, 0.02]


def fuse(Plin, Z, w):
    return softmax((1 - w) * np.log(Plin + 1e-9) + w * np.log(Z + 1e-9))


def m_base(R, train_objs, te, skeys=("rich|floor",), mode="mean", temp_grid=(0.01,), w_grid=W_GRID, views=VIEWS):
    """Голова ⊕ zero-shot (среднее логитов по наборам skeys); w и temp — вложенно."""
    tr = np.where(np.isin(R.oid, train_objs) & R.real)[0]
    Pin = R.inner_oof(train_objs)
    L = np.mean([R.logits(s, mode, views) for s in skeys], 0)
    best = max(((w, t) for w in w_grid for t in temp_grid), key=lambda wt: M_frame(fuse(Pin, softmax(L / wt[1]), wt[0]), R.mat, tr))
    w, t = best
    Pte = R.head_full(train_objs, te)
    return fuse(Pte, softmax(L[te] / t), w), {"w": w, "t": t}


def m_select_sets(R, train_objs, te, cands, mode="mean", k=3, temp=0.01, greedy=True):
    """Вложенный выбор подмножества наборов описаний: жадно добавляем набор, максимизирующий M (fused с inner-OOF головой)."""
    tr = np.where(np.isin(R.oid, train_objs) & R.real)[0]
    Pin = R.inner_oof(train_objs)

    def score(sel, w):
        L = np.mean([R.logits(s, mode) for s in sel], 0)
        return M_frame(fuse(Pin, softmax(L / temp), w), R.mat, tr)

    sel = []
    best_s, best_w = -1, 0.5
    for _ in range(k):
        cand_best = None
        for c in cands:
            if c in sel:
                continue
            for w in W_GRID:
                s = score(sel + [c], w)
                if cand_best is None or s > cand_best[0]:
                    cand_best = (s, c, w)
        if cand_best is None or cand_best[0] < best_s - 1e-9:
            break
        best_s, c, best_w = cand_best
        sel.append(c)
    L = np.mean([R.logits(s, mode) for s in sel], 0)
    Pte = R.head_full(train_objs, te)
    return fuse(Pte, softmax(L[te] / temp), best_w), {"sets": tuple(sel), "w": best_w}


def m_anchored(R, train_objs, te, skeys=("rich|floor",), mode="mean", alpha_grid=(50, 100, 200), wd_grid=(1e-2, 3e-2, 1e-1)):
    """Text-anchored голова: логиты = alpha * zs_logits + X W + b; W с L2; alpha, wd — вложенно."""
    tr = np.where(np.isin(R.oid, train_objs) & R.real)[0]
    L = np.mean([R.logits(s, mode) for s in skeys], 0)
    best, best_s = None, -1
    for alpha in alpha_grid:
        for wd in wd_grid:
            P = np.zeros((len(R.mat), 6))
            for g in inner_folds(train_objs, 5, 0):
                a = np.where(np.isin(R.oid, train_objs) & ~np.isin(R.oid, g) & R.real)[0]
                b = np.where(np.isin(R.oid, g) & R.real)[0]
                P[b] = head_fit_predict(R.X, R.mat, class_weights(a, R.oid, R.src, R.mat), a, b, wd=wd, offset=alpha * L)
            s = M_frame(P, R.mat, tr)
            if s > best_s:
                best, best_s = (alpha, wd), s
    alpha, wd = best
    P = head_fit_predict(R.X, R.mat, class_weights(tr, R.oid, R.src, R.mat), tr, te, wd=wd, offset=alpha * L)
    return P, {"alpha": alpha, "wd": wd}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", default="all")
    a = ap.parse_args()
    R = Runner()
    results = {}
    print(f"наборов описаний: {len(SET_KEYS)}: {SET_KEYS}")
    if a.part in ("all", "zs"):
        print("\n== 1. Zero-shot по наборам (без обучения; mean / lse по описаниям) ==")
        rows = []
        for s in SET_KEYS:
            for mode in ("mean", "lse"):
                Z = softmax(R.logits(s, mode) / 0.01)
                r = report(f"zs {s} {mode}", Z, R.mat, R.oid, R.idx, R.light, show_wrong=False)
                rows.append((r["M"], s, mode))
                results[f"zs|{s}|{mode}"] = r
        rows.sort(reverse=True)
        print("  топ-10:", rows[:10])
        print("\n== 1b. Zero-shot: ансамбль всех общих наборов ==")
        gen = [s for s in SET_KEYS if not s.startswith("rich") and not s.startswith("generic_syn")]
        for mode in ("mean", "lse"):
            L = np.mean([R.logits(s, mode) for s in gen], 0)
            results[f"zs|ALLGEN|{mode}"] = report(f"zs все общие наборы ({len(gen)}) {mode}", softmax(L / 0.01), R.mat, R.oid, R.idx, R.light)
        for tname in ("floor", "plain", "robot", "light"):
            L = np.mean([R.logits(s, "mean") for s in gen if s.endswith("|" + tname)], 0)
            results[f"zs|ALLGEN|{tname}"] = report(f"zs общие наборы, шаблон {tname}", softmax(L / 0.01), R.mat, R.oid, R.idx, R.light, show_wrong=False)
        for sname in ("names", "waste", "surface", "packaging", "shape_free"):
            L = np.mean([R.logits(s, "mean") for s in gen if s.startswith(sname + "|")], 0)
            results[f"zs|{sname}|alltemps"] = report(f"zs словарь {sname}, все шаблоны", softmax(L / 0.01), R.mat, R.oid, R.idx, R.light, show_wrong=False)
        print("\n== 1c. Zero-shot: по видам (все общие наборы) ==")
        for v in VIEWS:
            L = np.mean([R.logits(s, "mean", views=[v]) for s in gen], 0)
            results[f"zs|ALLGEN|view_{v}"] = report(f"zs общие, вид {v}", softmax(L / 0.01), R.mat, R.oid, R.idx, R.light, show_wrong=False)
        for tag in MODELS:
            L = np.mean([R.logits(s, "mean", tags=[tag]) for s in gen], 0)
            results[f"zs|ALLGEN|{tag}"] = report(f"zs общие, модель {tag}", softmax(L / 0.01), R.mat, R.oid, R.idx, R.light, show_wrong=False)
    if a.part in ("all", "fuse"):
        print("\n== 2. Голова ⊕ zero-shot (вес/температура вложенно) ==")
        _, results["base_rich"] = R.run(m_base, "база: rich|floor, w вложенно", skeys=("rich|floor",))
        _, results["base_rich_T"] = R.run(m_base, "база: rich|floor, w+T вложенно", skeys=("rich|floor",), temp_grid=T_GRID)
        _, results["base_gsyn"] = R.run(m_base, "база: generic_syn|floor", skeys=("generic_syn|floor",))
        gen = [s for s in SET_KEYS if not s.startswith("rich") and not s.startswith("generic_syn")]
        _, results["ens_allgen"] = R.run(m_base, "голова ⊕ zs(все общие наборы)", skeys=tuple(gen))
        _, results["ens_allgen_T"] = R.run(m_base, "голова ⊕ zs(все общие), w+T", skeys=tuple(gen), temp_grid=T_GRID)
        _, results["ens_allgen_lse"] = R.run(m_base, "голова ⊕ zs(все общие, lse)", skeys=tuple(gen), mode="lse")
        for sname in ("names", "waste", "surface", "packaging", "shape_free"):
            ss = tuple(s for s in gen if s.startswith(sname + "|"))
            _, results[f"ens_{sname}"] = R.run(m_base, f"голова ⊕ zs({sname}, все шаблоны)", skeys=ss)
        _, results["sel_greedy3"] = R.run(m_select_sets, "голова ⊕ zs(жадный выбор ≤3 из общих)", cands=gen, k=3)
        _, results["sel_greedy5"] = R.run(m_select_sets, "голова ⊕ zs(жадный выбор ≤5 из общих)", cands=gen, k=5)
        _, results["sel_greedy3_all"] = R.run(m_select_sets, "голова ⊕ zs(жадный ≤3 из всех, вкл. rich)", cands=SET_KEYS, k=3)
    if a.part in ("all", "anchor"):
        print("\n== 3. Text-anchored голова (zs-логиты как смещение, W с L2) ==")
        gen = [s for s in SET_KEYS if not s.startswith("rich") and not s.startswith("generic_syn")]
        _, results["anch_rich"] = R.run(m_anchored, "anchored: rich|floor", skeys=("rich|floor",))
        _, results["anch_allgen"] = R.run(m_anchored, "anchored: все общие", skeys=tuple(gen))
    json.dump(results, open(HERE / f"exp1_{a.part}.json", "w"), indent=1, default=str)


if __name__ == "__main__":
    main()
