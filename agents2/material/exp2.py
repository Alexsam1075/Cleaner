"""Эксперимент 2: мультивидовые головы / TTA, калиброванный zero-shot (bias+T), ансамбль голов на разных признаках,
стекинг по inner-OOF, глубина как «дыры». LOOO по 40 предметам, выбор гиперпараметров вложенно."""
import argparse
import json
import math
import time
from collections import Counter

import numpy as np
import torch

from common import (DEV, HERE, MATS, MODELS, VIEWS, Standardizer, class_weights, fit_logreg, inner_folds, l2n, load_bank, predict_logreg, report,
                    softmax, M_frame)
from exp1 import Runner, SET_KEYS, W_GRID, fuse, head_fit_predict, m_base

GEN = [s for s in SET_KEYS if not s.startswith("rich") and not s.startswith("generic_syn")]


def fit_bias_temp(L, y, w, tr, wd=1e-3):
    """Калибровка zero-shot: logits = L/T + b (7 параметров), обучается на train-кадрах."""
    Lt = torch.tensor(L[tr], device=DEV, dtype=torch.float32)
    yt = torch.tensor(y[tr], device=DEV)
    wt = torch.tensor(w, device=DEV, dtype=torch.float32)
    s = torch.tensor(math.log(100.0), device=DEV, requires_grad=True)
    b = torch.zeros(6, device=DEV, requires_grad=True)
    opt = torch.optim.LBFGS([s, b], lr=1, max_iter=200, line_search_fn="strong_wolfe")

    def closure():
        opt.zero_grad()
        loss = (torch.nn.functional.cross_entropy(Lt * torch.exp(s) + b, yt, reduction="none") * wt).sum() / wt.sum() + wd * (b ** 2).sum()
        loss.backward()
        return loss

    opt.step(closure)
    return float(torch.exp(s)), b.detach().cpu().numpy()


class Runner2(Runner):
    def __init__(self):
        super().__init__()
        self._view_cache = {}

    def Xview(self, tag, v):
        k = (tag, v)
        if k not in self._view_cache:
            self._view_cache[k] = (l2n(self.F[f"{tag}_{v}"]) * math.sqrt(768 / 64)).astype(np.float32)
        return self._view_cache[k]

    def head_multiview(self, train_objs, te, tag="clipL", train_views=("c14",), test_views=("c14",), wd=1e-2, inner=False, seed=0):
        """Одна голова, обученная на кадрах во всех train_views (как аугментация), предсказание — среднее log-p по test_views.
        inner=True: возвращает inner-OOF для train_objs (5-fold по объектам)."""
        if inner:
            P = np.zeros((len(self.mat), 6))
            for g in inner_folds(train_objs, 5, seed):
                a = [q for q in train_objs if q not in set(g)]
                b = np.where(np.isin(self.oid, g) & self.real)[0]
                P[b] = self.head_multiview(a, b, tag, train_views, test_views, wd)
            return P
        tr = np.where(np.isin(self.oid, train_objs) & self.real)[0]
        Xtr = np.concatenate([self.Xview(tag, v)[tr] for v in train_views])
        ytr = np.concatenate([self.mat[tr]] * len(train_views))
        w = np.concatenate([class_weights(tr, self.oid, self.src, self.mat)] * len(train_views)) / len(train_views)
        st = Standardizer(Xtr)
        W, b = fit_logreg(st(Xtr), ytr, w, wd=wd)
        lp = np.mean([np.log(predict_logreg(st(self.Xview(tag, v)[te]), W, b) + 1e-9) for v in test_views], 0)
        return softmax(lp)


def m_multiview(R, train_objs, te, configs, skeys=("rich|floor",), temp=0.01, wd=1e-2):
    """Выбор (вложенно) конфигурации видов для головы из списка configs = [(train_views, test_views), ...] и веса zs."""
    tr = np.where(np.isin(R.oid, train_objs) & R.real)[0]
    L = np.mean([R.logits(s, "mean") for s in skeys], 0)
    Z = softmax(L / temp)
    best, best_s = None, -1
    for cfg in configs:
        Pin = R.head_multiview(train_objs, None, "clipL", cfg[0], cfg[1], wd, inner=True)
        for w in W_GRID:
            s = M_frame(fuse(Pin, Z, w), R.mat, tr)
            if s > best_s:
                best, best_s = (cfg, w), s
    cfg, w = best
    Pte = R.head_multiview(train_objs, te, "clipL", cfg[0], cfg[1], wd)
    return fuse(Pte, Z[te], w), {"cfg": cfg, "w": w}


def m_fixed_multiview(R, train_objs, te, cfg, skeys=("rich|floor",), temp=0.01, wd=1e-2, tag="clipL"):
    tr = np.where(np.isin(R.oid, train_objs) & R.real)[0]
    L = np.mean([R.logits(s, "mean") for s in skeys], 0)
    Z = softmax(L / temp)
    Pin = R.head_multiview(train_objs, None, tag, cfg[0], cfg[1], wd, inner=True)
    w = max(W_GRID, key=lambda w: M_frame(fuse(Pin, Z, w), R.mat, tr))
    Pte = R.head_multiview(train_objs, te, tag, cfg[0], cfg[1], wd)
    return fuse(Pte, Z[te], w), {"w": w}


def m_calib_zs(R, train_objs, te, skeys=("rich|floor",), use_head=True, wd_b=1e-3, cfg=(("c14",), ("c14",))):
    """Zero-shot с обученными T и смещениями классов (7 параметров, на train-кадрах), ⊕ голова с весом по inner-OOF."""
    tr = np.where(np.isin(R.oid, train_objs) & R.real)[0]
    L = np.mean([R.logits(s, "mean") for s in skeys], 0)
    # inner-OOF калибровки для честного подбора w
    Zin = np.zeros((len(R.mat), 6))
    for g in inner_folds(train_objs, 5, 0):
        a = np.where(np.isin(R.oid, train_objs) & ~np.isin(R.oid, g) & R.real)[0]
        b = np.where(np.isin(R.oid, g) & R.real)[0]
        T, bias = fit_bias_temp(L, R.mat, class_weights(a, R.oid, R.src, R.mat), a, wd_b)
        Zin[b] = softmax(L[b] * T + bias)
    T, bias = fit_bias_temp(L, R.mat, class_weights(tr, R.oid, R.src, R.mat), tr, wd_b)
    Zte = softmax(L[te] * T + bias)
    if not use_head:
        return Zte, {"T": round(1 / T, 4), "bias": np.round(bias, 2).tolist()}
    Pin = R.head_multiview(train_objs, None, "clipL", cfg[0], cfg[1], inner=True)
    w = max(W_GRID, key=lambda w: M_frame(fuse(Pin, Zin, w), R.mat, tr))
    Pte = R.head_multiview(train_objs, te, "clipL", cfg[0], cfg[1])
    return fuse(Pte, Zte, w), {"w": w, "T": round(1 / T, 4)}


def m_stack(R, train_objs, te, members, skeys=("rich|floor",), temp=0.01, wd_stack=1e-1):
    """Стекинг: признаки = log-p inner-OOF от нескольких голов (разные признаки/виды) + log zs; стекер — логрег с L2.
    Для отложенного предмета члены обучаются на всех train_objs."""
    tr = np.where(np.isin(R.oid, train_objs) & R.real)[0]
    L = np.mean([R.logits(s, "mean") for s in skeys], 0)
    Z = softmax(L / temp)
    feats_in, feats_te = [np.log(Z + 1e-9)], [np.log(Z[te] + 1e-9)]
    for tag, cfg in members:
        feats_in.append(np.log(R.head_multiview(train_objs, None, tag, cfg[0], cfg[1], inner=True) + 1e-9))
        feats_te.append(np.log(R.head_multiview(train_objs, te, tag, cfg[0], cfg[1]) + 1e-9))
    Fin = np.concatenate(feats_in, 1)
    Fte = np.concatenate(feats_te, 1)
    W, b = fit_logreg(Fin[tr], R.mat[tr], class_weights(tr, R.oid, R.src, R.mat), wd=wd_stack)
    return predict_logreg(Fte, W, b), {"n": len(members)}


def m_avg_heads(R, train_objs, te, members, skeys=("rich|floor",), temp=0.01):
    """Среднее log-p нескольких голов (равные веса) ⊕ zs (вес вложенно)."""
    tr = np.where(np.isin(R.oid, train_objs) & R.real)[0]
    L = np.mean([R.logits(s, "mean") for s in skeys], 0)
    Z = softmax(L / temp)
    Pin = softmax(np.mean([np.log(R.head_multiview(train_objs, None, tag, cfg[0], cfg[1], inner=True) + 1e-9) for tag, cfg in members], 0))
    w = max(W_GRID, key=lambda w: M_frame(fuse(Pin, Z, w), R.mat, tr))
    Pte = softmax(np.mean([np.log(R.head_multiview(train_objs, te, tag, cfg[0], cfg[1]) + 1e-9) for tag, cfg in members], 0))
    return fuse(Pte, Z[te], w), {"w": w}


def m_depth_holes(R, train_objs, te, skeys=("rich|floor",), temp=0.01, lam_grid=(0.0, 0.5, 1.0, 2.0)):
    """База + подсказка по дырам глубины: log P(glass), log P(metal) += λ·z(holes). λ вложенно."""
    gk = list(R.F["geo_keys"])
    h = np.log1p(R.F["geo"][:, gk.index("holes_inner")] * 20) + np.log1p(R.F["geo"][:, gk.index("holes_edge")] * 5)
    tr = np.where(np.isin(R.oid, train_objs) & R.real)[0]
    L = np.mean([R.logits(s, "mean") for s in skeys], 0)
    Z = softmax(L / temp)
    Pin = R.inner_oof(train_objs)
    mu, sd = h[tr].mean(), h[tr].std() + 1e-6
    z = (h - mu) / sd

    def adj(P, lam):
        Lg = np.log(P + 1e-9)
        Lg[:, MATS.index("glass")] += lam * z
        Lg[:, MATS.index("metal")] += lam * z
        return softmax(Lg)

    best = max(((w, lam) for w in W_GRID for lam in lam_grid), key=lambda t: M_frame(adj(fuse(Pin, Z, t[0]), t[1]), R.mat, tr))
    w, lam = best
    Pte = R.head_full(train_objs, te)
    return adj(fuse(Pte, Z[te], w), lam)[te] if False else adj(fuse(Pte, Z[te], w), lam), {"w": w, "lam": lam}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", default="all")
    a = ap.parse_args()
    R = Runner2()
    results = {}
    if a.part in ("all", "views"):
        print("== 1. Мультивидовые головы CLIP-L (train views → test views), ⊕ zs rich|floor ==")
        cfgs = {"c14→c14": (("c14",), ("c14",)), "c20→c20": (("c20",), ("c20",)), "m10→m10": (("m10",), ("m10",)),
                "c14+c20→c14+c20": (("c14", "c20"), ("c14", "c20")), "c14+c20→c14": (("c14", "c20"), ("c14",)),
                "all4→all4": (tuple(VIEWS), tuple(VIEWS)), "all4→c14+c20": (tuple(VIEWS), ("c14", "c20")), "c14+m10→c14+m10": (("c14", "m10"), ("c14", "m10"))}
        for name, cfg in cfgs.items():
            _, results[f"mv_{name}"] = R.run(m_fixed_multiview, f"голова {name} ⊕ zs", cfg=cfg)
        _, results["mv_select"] = R.run(m_multiview, "голова: выбор конфигурации видов вложенно ⊕ zs", configs=list(cfgs.values()))
        print("\n== 1b. SigLIP-головы ==")
        for name in ("c14→c14", "c14+c20→c14+c20"):
            _, results[f"mv_siglip_{name}"] = R.run(m_fixed_multiview, f"siglip голова {name} ⊕ zs", cfg=cfgs[name], tag="siglip")
        print("\n== 1c. Среднее нескольких голов ⊕ zs ==")
        _, results["avg_clip_c14_c20"] = R.run(m_avg_heads, "avg[clip c14, clip c20] ⊕ zs", members=[("clipL", cfgs["c14→c14"]), ("clipL", cfgs["c20→c20"])])
        _, results["avg_clip_siglip"] = R.run(m_avg_heads, "avg[clip c14, siglip c14] ⊕ zs", members=[("clipL", cfgs["c14→c14"]), ("siglip", cfgs["c14→c14"])])
        _, results["avg_4"] = R.run(m_avg_heads, "avg[clip c14, clip c20, siglip c14, siglip c20] ⊕ zs",
                                    members=[("clipL", cfgs["c14→c14"]), ("clipL", cfgs["c20→c20"]), ("siglip", cfgs["c14→c14"]), ("siglip", cfgs["c20→c20"])])
    if a.part in ("all", "calib"):
        print("\n== 2. Калиброванный zero-shot (T + смещения классов, обучены на train) ==")
        _, results["czs_rich_only"] = R.run(m_calib_zs, "калибр. zs rich|floor (без головы)", skeys=("rich|floor",), use_head=False)
        _, results["czs_gen_only"] = R.run(m_calib_zs, "калибр. zs все общие (без головы)", skeys=tuple(GEN), use_head=False)
        _, results["czs_rich"] = R.run(m_calib_zs, "калибр. zs rich|floor ⊕ голова", skeys=("rich|floor",))
        _, results["czs_gen"] = R.run(m_calib_zs, "калибр. zs все общие ⊕ голова", skeys=tuple(GEN))
        _, results["czs_gen_mv"] = R.run(m_calib_zs, "калибр. zs все общие ⊕ голова c14+c20", skeys=tuple(GEN), cfg=(("c14", "c20"), ("c14", "c20")))
    if a.part in ("all", "stack"):
        print("\n== 3. Стекинг / глубина ==")
        c14 = (("c14",), ("c14",))
        c20 = (("c20",), ("c20",))
        _, results["stack_2"] = R.run(m_stack, "стек[zs, clip c14, clip c20]", members=[("clipL", c14), ("clipL", c20)])
        _, results["stack_4"] = R.run(m_stack, "стек[zs, clip c14, clip c20, siglip c14, siglip c20]", members=[("clipL", c14), ("clipL", c20), ("siglip", c14), ("siglip", c20)])
        _, results["stack_4_wd1"] = R.run(m_stack, "стек[...4] wd=1", members=[("clipL", c14), ("clipL", c20), ("siglip", c14), ("siglip", c20)], wd_stack=1.0)
        _, results["depth_holes"] = R.run(m_depth_holes, "база + дыры глубины (glass/metal)")
    json.dump(results, open(HERE / f"exp2_{a.part}.json", "w"), indent=1, default=str)


if __name__ == "__main__":
    main()
