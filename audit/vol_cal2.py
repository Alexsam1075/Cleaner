"""Калибратор объёма v2 = признаки volume/ (fit_calibrator) + zero-shot «заполнение» (CLIP-L+SigLIP: hollow, bag,
empty/full container, dense, pile, crushed…) + IRLS под метрику V. Проверка: LOOO по 40 предметам, GT-маски, истинный класс.
Класс VolCal2: fit(rows_train, embs_train) / predict(rows, embs, material) — для конвейера."""
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path("/home/cleaner/Cleaner")
sys.path.insert(0, str(ROOT))
from evaluation.volume_parts_cv import EST_KEYS, KNOTS_ML, _class_heights, _features, apply_calibrator, fit_calibrator  # noqa: E402

C = ROOT / "audit/cache"
L3 = math.log(3)
ZS_TAGS = ("clipL", "siglip")
ZS_VIEWS = ("m10", "c14", "c20")
FILL = ["hollow", "bag", "empty_bottle", "full_bottle", "empty_can", "solid", "metal_block", "rock", "paper_light", "foam_light", "crushed", "intact", "powder", "screws", "liquid", "transparent"]


def zs_fill(embs):
    T = np.load(ROOT / "agents/take/cache/zs_text.npz")
    names = list(T["concepts"])
    cols = [names.index(c) for c in FILL]
    out = []
    for tag in ZS_TAGS:
        acc = None
        for v in ZS_VIEWS:
            S = embs[(tag, v)] @ T[f"{tag}_concepts"].T
            S = S - S.mean(1, keepdims=True)
            acc = S if acc is None else acc + S
        out.append((acc / len(ZS_VIEWS))[:, cols])
    return np.concatenate(out, 1) * 20.0


def _ridge(X, y, w, a):
    mu = np.average(X, 0, weights=w)
    sd = np.sqrt(np.average((X - mu) ** 2, 0, weights=w))
    sd[sd < 1e-9] = 1
    Z = (X - mu) / sd
    ym = np.average(y, weights=w)
    coef = np.linalg.solve(Z.T @ (Z * w[:, None]) + a * np.eye(Z.shape[1]), Z.T @ (w * (y - ym)))
    return lambda Xn: ym + ((Xn - mu) / sd) @ coef


def _irls(X, y, w, a, floor=0.2, iters=8):
    m = _ridge(X, y, w, a)
    for _ in range(iters):
        r = np.abs(m(X) - y)
        m = _ridge(X, y, w * np.where(r < L3, 1 / np.maximum(r, floor), 0.1 / L3), a)
    return m


class VolCal2:
    def __init__(self, use_zs=True, irls=True, alphas=(3, 10, 30, 100, 300)):
        self.use_zs, self.irls, self.alphas = use_zs, irls, alphas

    def _X(self, rows, embs, mats=None):
        mats = rows["material"].values if mats is None else mats
        Xb = np.vstack([_features(r, r["form"], m, self.heights, list(EST_KEYS), list(KNOTS_ML)) for r, m in zip(rows.to_dict("records"), mats)])
        return np.c_[Xb, zs_fill(embs)] if self.use_zs else Xb

    def fit(self, rows, embs):
        self.heights = _class_heights(rows)
        X = self._X(rows, embs)
        y = np.log(rows["volume_ml"].values)
        ids = rows["object_id"].values
        cnt = defaultdict(int)
        for o in ids:
            cnt[o] += 1
        w = np.array([1.0 / cnt[o] for o in ids])
        fitter = _irls if self.irls else _ridge
        objs = np.array(sorted(set(ids)))
        np.random.default_rng(7).shuffle(objs)
        best = None
        for a in self.alphas:
            p = np.zeros(len(y))
            for g in np.array_split(objs, 5):
                te = np.isin(ids, g)
                p[te] = fitter(X[~te], y[~te], w[~te], a)(X[te])
            s = np.sum(w * np.maximum(0, 1 - np.abs(p - y) / L3)) / w.sum()
            if best is None or s > best[0]:
                best = (s, a)
        self.alpha = best[1]
        self.model = fitter(X, y, w, self.alpha)
        return self

    def predict(self, rows, embs, mats=None):
        return np.clip(np.exp(self.model(self._X(rows, embs, mats))), 1.0, 20000.0)


def main():
    rows = pd.read_csv(ROOT / "reports/volume_parts_gt_rows.csv")
    rows = rows[rows.split == "real"].reset_index(drop=True)
    F2 = dict(np.load(C / "features2.npz"))
    meta = json.load(open(C / "meta.json"))
    key = {(m["frame"], m["oid"]): i for i, m in enumerate(meta) if m["src"] == "real_train"}
    ridx = np.array([key[(f, o)] for f, o in zip(rows.image_id, rows.object_id)])
    embs_all = {(t, v): F2[f"{t}_{v}"][ridx] for t in ZS_TAGS for v in ZS_VIEWS}
    g = rows.volume_ml.values
    ids = rows.object_id.values
    rng = np.random.default_rng(5)
    noisy = rows.material.values.copy()
    flip = rng.random(len(rows)) < 0.08
    MATS = ["glass", "metal", "mineral", "paper", "plastic", "wood"]
    noisy[flip] = [rng.choice([m for m in MATS if m != x]) for x in rows.material.values[flip]]

    def V(p):
        return np.clip(1 - np.abs(np.log(np.clip(p, 1e-3, None) / g)) / L3, 0, 1)

    def rep(name, p):
        v = V(p)
        b = "  ".join(f"{lo:g}-{hi:g}:{v[(g >= lo) & (g < hi)].mean():.2f}" for lo, hi in ((0, 50), (50, 200), (200, 1000), (1000, 1e9)))
        print(f"  {name:44s} V={v.mean():.4f}  bulk {v[rows.form == 'bulk'].mean():.2f} solid {v[rows.form == 'solid'].mean():.2f} | {b}", flush=True)

    print("== LOOO по 40 предметам (GT-маски); второй столбец — с 8% неверного материала ==")
    for name, cfg in (("A текущий fit_calibrator", None), ("B ridge + zs-заполнение", dict(use_zs=True, irls=False)),
                      ("C IRLS без zs", dict(use_zs=False, irls=True)), ("D IRLS + zs-заполнение", dict(use_zs=True, irls=True))):
        p, pn = np.zeros(len(g)), np.zeros(len(g))
        for o in sorted(set(ids)):
            tr, te = ids != o, ids == o
            e_tr = {k: v[tr] for k, v in embs_all.items()}
            e_te = {k: v[te] for k, v in embs_all.items()}
            if cfg is None:
                cal = fit_calibrator(rows[tr])
                p[te] = [apply_calibrator(cal, r, r["form"], r["material"]) for r in rows[te].to_dict("records")]
                pn[te] = [apply_calibrator(cal, r, r["form"], m) for r, m in zip(rows[te].to_dict("records"), noisy[te])]
            else:
                m = VolCal2(**cfg).fit(rows[tr], e_tr)
                p[te] = m.predict(rows[te], e_te)
                pn[te] = m.predict(rows[te], e_te, noisy[te])
        rep(name, p)
        rep(name + " [материал 8% шум]", pn)


if __name__ == "__main__":
    main()
