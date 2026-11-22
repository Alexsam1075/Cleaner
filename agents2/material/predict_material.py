"""MaterialClassifier — классификатор материала (metal, wood, plastic, glass, paper, mineral).

Метод: линейная голова (логрегрессия, L2) на CLIP-L эмбеддинге вырезки с контекстом (c14) ⊕ zero-shot CLIP-L + SigLIP
с общими словарями описаний материалов, где сходство с классом = logsumexp по отдельным формулировкам (lse, τ=0.01),
усреднённое по видам c14 и c20 и по двум моделям. Вес zero-shot w подбирается внутренним 5-fold CV по train-объектам.

Совместимость с full_pipeline: fit(train_object_ids) учится по кэшу shared_cache (features2.npz, meta.json),
predict(clipL_c14_emb, embs_by_(tag,view)) принимает эмбеддинги вырезок {("clipL"|"siglip", "m10"|"m14"|"c14"|"c20"): (N,768)}
(L2-нормированные, как в Encoders.embed) и возвращает (N,6) вероятности в порядке MATS.

Текстовые эмбеддинги берутся из text_bank.npz (строится text_bank.py; при отсутствии — считаются через open_clip).
Для интегратора: comp_material(ctx) — совместимая с agents2/integrator/eval_folds.comp_material функция."""
import json
import math
import os
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
CACHE = HERE / "shared_cache"
MATS = ["metal", "wood", "plastic", "glass", "paper", "mineral"]
MODELS = ["clipL", "siglip"]
VIEWS = ["m10", "m14", "c14", "c20"]

DEFAULT_CFG = {
    "head_views": ["c14"],          # виды CLIP-L для головы (обучение = объединение, предсказание = среднее log-p)
    "head_tag": "clipL",
    "wd": 1e-2,
    "zs_sets": ["rich|floor"],      # наборы описаний из text_bank (словарь|шаблон)
    "zs_mode": "lse",               # lse | mean
    "zs_tau": 0.01,                 # температура lse внутри класса
    "zs_views": ["c14", "c20"],
    "zs_tags": ["clipL", "siglip"],
    "temp": 0.01,                   # температура softmax zero-shot
    "w_grid": [0.0, 0.25, 0.4, 0.5, 0.6, 0.75, 1.0],
    "inner_k": 5,
    "inner_seed": 0,
}


def softmax(z):
    z = z - z.max(-1, keepdims=True)
    p = np.exp(z)
    return p / p.sum(-1, keepdims=True)


def l2n(X):
    return X / (np.linalg.norm(X, axis=-1, keepdims=True) + 1e-8)


def _fit_logreg(X, y, w, wd=1e-2, epochs=300, n_cls=6):
    import torch
    torch.set_num_threads(int(os.environ.get("MAT_THREADS", "8")))
    dev = "cuda" if (torch.cuda.is_available() and os.environ.get("MAT_GPU", "0") == "1") else "cpu"
    Xt = torch.tensor(X, device=dev, dtype=torch.float32)
    yt = torch.tensor(y, device=dev)
    wt = torch.tensor(w, device=dev, dtype=torch.float32)
    W = torch.zeros(X.shape[1], n_cls, device=dev, requires_grad=True)
    b = torch.zeros(n_cls, device=dev, requires_grad=True)
    opt = torch.optim.LBFGS([W, b], lr=1, max_iter=epochs, history_size=20, line_search_fn="strong_wolfe")

    def closure():
        opt.zero_grad()
        loss = (torch.nn.functional.cross_entropy(Xt @ W + b, yt, reduction="none") * wt).sum() / wt.sum() + wd * (W ** 2).sum()
        loss.backward()
        return loss

    opt.step(closure)
    return W.detach().cpu().numpy(), b.detach().cpu().numpy()


def class_weights(oid, mat):
    """Каждый физический предмет весит одинаково, каждый материал весит одинаково."""
    from collections import Counter, defaultdict
    cnt = Counter(oid)
    w = np.array([1.0 / cnt[o] for o in oid])
    per = defaultdict(float)
    for j in range(len(oid)):
        per[mat[j]] += w[j]
    w = np.array([w[j] / per[mat[j]] for j in range(len(oid))])
    return w / w.mean()


def M_frame(P, mat):
    pred = P.argmax(1)
    return float(np.mean([(pred[mat == c] == c).mean() for c in range(6) if (mat == c).any()]))


class TextBank:
    """Текстовые эмбеддинги по наборам: cls[(tag, set)] (6,d) и per[(tag, set, mat)] (n_prompts, d)."""

    def __init__(self, path=HERE / "text_bank.npz"):
        if not Path(path).exists():
            import subprocess
            import sys
            subprocess.check_call([sys.executable, str(HERE / "text_bank.py")])
        self.T = dict(np.load(path))

    def cls(self, tag, skey):
        return self.T[f"{tag}|{skey}"]

    def per(self, tag, skey, mt):
        return self.T[f"{tag}|{skey}|per|{mt}"]


class ZeroShot:
    def __init__(self, bank, cfg):
        self.bank, self.cfg = bank, cfg

    def logits(self, embs):
        """embs: {(tag, view): (N,768)} → (N,6) усреднённые сходства."""
        c = self.cfg
        out = []
        for skey in c["zs_sets"]:
            for tag in c["zs_tags"]:
                for v in c["zs_views"]:
                    E = l2n(np.asarray(embs[(tag, v)], np.float32))
                    if c["zs_mode"] == "mean":
                        out.append(E @ self.bank.cls(tag, skey).T)
                    else:
                        cols = []
                        for mt in MATS:
                            s = E @ self.bank.per(tag, skey, mt).T / c["zs_tau"]
                            m = s.max(1, keepdims=True)
                            cols.append((m[:, 0] + np.log(np.exp(s - m).mean(1))) * c["zs_tau"])
                        out.append(np.stack(cols, 1))
        return np.mean(out, 0)

    def probs(self, embs):
        return softmax(self.logits(embs) / self.cfg["temp"])


class Head:
    def __init__(self, cfg):
        self.cfg = cfg

    @staticmethod
    def prep(E):
        return (l2n(np.asarray(E, np.float32)) * math.sqrt(768 / 64)).astype(np.float32)

    def fit(self, embs_by_view, y, oid):
        """embs_by_view: {view: (N,768)} для train-кадров."""
        vs = self.cfg["head_views"]
        X = np.concatenate([self.prep(embs_by_view[v]) for v in vs])
        yy = np.concatenate([y] * len(vs))
        w = np.concatenate([class_weights(oid, y)] * len(vs)) / len(vs)
        self.mu, self.sd = X.mean(0), X.std(0) + 1e-6
        self.W, self.b = _fit_logreg((X - self.mu) / self.sd, yy, w, wd=self.cfg["wd"])
        return self

    def predict(self, embs_by_view):
        lp = [np.log(softmax(((self.prep(embs_by_view[v]) - self.mu) / self.sd) @ self.W + self.b) + 1e-9) for v in self.cfg["head_views"]]
        return softmax(np.mean(lp, 0))


def fuse(P_head, Z, w):
    return softmax((1 - w) * np.log(P_head + 1e-9) + w * np.log(Z + 1e-9))


class MaterialClassifier:
    def __init__(self, cfg=None, bank=None, F=None, meta=None):
        self.cfg = dict(DEFAULT_CFG)
        if cfg:
            self.cfg.update(cfg)
        self.bank = bank or TextBank()
        self.zs = ZeroShot(self.bank, self.cfg)
        if F is None:
            F = dict(np.load(CACHE / "features2.npz"))
            meta = json.load(open(CACHE / "meta.json"))
        self.F, self.meta = F, meta
        self.oid = np.array([m["oid"] for m in meta])
        self.src = np.array([m["src"] for m in meta])
        self.mat = np.array([MATS.index(m["mat"]) for m in meta])
        self.real = self.src == "real_train"
        self._Z = None

    def _train_embs(self, idx, views):
        return {v: self.F[f"{self.cfg['head_tag']}_{v}"][idx] for v in views}

    def _bank_Z(self):
        if self._Z is None:
            self._Z = self.zs.probs({(t, v): self.F[f"{t}_{v}"] for t in MODELS for v in VIEWS})
        return self._Z

    def fit(self, train_object_ids, verbose=False):
        train_objs = sorted(set(train_object_ids))
        tr = np.where(np.isin(self.oid, train_objs) & self.real)[0]
        assert set(self.oid[tr]) == set(train_objs), "материал: не все train-объекты есть в real_train"
        c = self.cfg
        vs = c["head_views"]
        # вложенный подбор веса zero-shot по inner-OOF головы (k-fold по объектам)
        rs = np.random.default_rng(c["inner_seed"])
        Pin = np.zeros((len(self.mat), 6))
        for g in np.array_split(rs.permutation(train_objs), c["inner_k"]):
            a = np.where(np.isin(self.oid, train_objs) & ~np.isin(self.oid, g) & self.real)[0]
            b = np.where(np.isin(self.oid, g) & self.real)[0]
            Pin[b] = Head(c).fit(self._train_embs(a, vs), self.mat[a], self.oid[a]).predict(self._train_embs(b, vs))
        Z = self._bank_Z()
        self.w = max(c["w_grid"], key=lambda w: M_frame(fuse(Pin[tr], Z[tr], w), self.mat[tr]))
        self.inner_M = M_frame(fuse(Pin[tr], Z[tr], self.w), self.mat[tr])
        self.head = Head(c).fit(self._train_embs(tr, vs), self.mat[tr], self.oid[tr])
        self.train_objs = train_objs
        if verbose:
            print(f"MaterialClassifier.fit: {len(train_objs)} объектов, {len(tr)} кадров, w_zs={self.w}, inner M={self.inner_M:.3f}")
        return self

    def predict(self, clipL_c14_emb, embs_by_tag_view=None, return_parts=False):
        """clipL_c14_emb: (N,768); embs_by_tag_view: {(tag, view): (N,768)} (нужны виды из cfg: c14, c20 обеих моделей).
        Возвращает (N,6) вероятности в порядке MATS."""
        clipL_c14_emb = np.asarray(clipL_c14_emb, np.float32).reshape(-1, 768)
        if len(clipL_c14_emb) == 0:
            return np.zeros((0, 6))
        embs = dict(embs_by_tag_view or {})
        embs.setdefault(("clipL", "c14"), clipL_c14_emb)
        hv = {v: embs[(self.cfg["head_tag"], v)] for v in self.cfg["head_views"]}
        Ph = self.head.predict(hv)
        Z = self.zs.probs(embs)
        P = fuse(Ph, Z, self.w)
        if return_parts:
            return P, Ph, Z
        return P


def comp_material(ctx):
    """Для agents2/integrator/pipeline_v2.py --external material_v2=<этот файл>:comp_material"""
    clf = MaterialClassifier().fit(ctx["train_objs"])
    ctx["w_zs"] = clf.w
    embs = ctx["embs"]
    if len(embs[("clipL", "c14")]) == 0:
        return np.zeros((0, 6))
    return clf.predict(embs[("clipL", "c14")], embs)


if __name__ == "__main__":
    meta = json.load(open(CACHE / "meta.json"))
    objs = sorted({m["oid"] for m in meta if m["src"] == "real_train"})
    clf = MaterialClassifier().fit(objs, verbose=True)
    F = clf.F
    P = clf.predict(F["clipL_c14"][:5], {(t, v): F[f"{t}_{v}"][:5] for t in MODELS for v in VIEWS})
    print(np.round(P, 3), [MATS[i] for i in P.argmax(1)], [m["mat"] for m in meta[:5]])
