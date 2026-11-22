"""Модель take (вариант C0 агента): ridge по log(массы) на [log калибр. объёма, геометрия, класс one-hot,
zero-shot «тяжёлый/лёгкий» концепты CLIP-L+SigLIP]; порог по log-массе — внутренний 5-fold CV по объектам
под сбалансированную точность. Обучение: real_train-аннотации train-объектов (GT-маски, кэш audit/cache).
LOOO агента: S 0.870 (ловит 6 из 7 тяжёлых)."""
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path("/home/cleaner/Cleaner")
C = ROOT / "audit/cache"
TAKE_CACHE = ROOT / "agents/take/cache"
MATS = ["metal", "wood", "plastic", "glass", "paper", "mineral"]
FORMS = ["bulk", "solid"]
GEO_COLS = ["h95", "hmax", "h_med", "ext1", "ext2", "valid", "holes_edge", "holes_inner", "rough"]
LOG_COLS = ["vol", "vol_thr", "foot", "surf"]
ZS_TAGS = ("clipL", "siglip")
ZS_VIEWS = ("m10", "c14", "c20")
PARTS_COLS = ["columns_ml", "parts_ml", "hull_ml", "footprint_ml", "valid_frac", "truncated", "n_parts", "p95_h_m", "footprint_m2",
              "fp_area_0", "fp_area_20", "fp_area_40", "fp_area_80", "fp_area_160", "min_h_m"]


def _ridge(X, y, w, a):
    mu = np.average(X, 0, weights=w)
    sd = np.sqrt(np.average((X - mu) ** 2, 0, weights=w))
    sd[sd < 1e-9] = 1
    Z = (X - mu) / sd
    ym = np.average(y, weights=w)
    coef = np.linalg.solve(Z.T @ (Z * w[:, None]) + a * np.eye(Z.shape[1]), Z.T @ (w * (y - ym)))
    return lambda Xn: ym + ((Xn - mu) / sd) @ coef


def zs_concepts(embs_by_key):
    """embs_by_key[(tag, view)] -> (N, d). Возвращает (N, n_concepts): центрированные сходства, среднее по видам."""
    T = np.load(TAKE_CACHE / "zs_text.npz")
    out = []
    for tag in ZS_TAGS:
        acc = None
        for v in ZS_VIEWS:
            S = embs_by_key[(tag, v)] @ T[f"{tag}_concepts"].T
            S = S - S.mean(1, keepdims=True)
            acc = S if acc is None else acc + S
        out.append(acc / len(ZS_VIEWS))
    return np.concatenate(out, 1)


def geo_feats(geo_dicts):
    g = pd.DataFrame(geo_dicts)
    lv = np.log(np.maximum(g[LOG_COLS].values.astype(float), 1.0))
    return np.c_[lv, g[GEO_COLS].values.astype(float)]


def onehot(forms, mats):
    return np.c_[np.array([[f == x for x in FORMS] for f in forms], float), np.array([[m == x for x in MATS] for m in mats], float)]


class TakeModel:
    def __init__(self, alpha=10.0, zs_scale=20.0, inner_folds=5, seed=0):
        self.alpha, self.zs_scale, self.inner_folds, self.seed = alpha, zs_scale, inner_folds, seed

    def _X(self, log_vol, geo_dicts, forms, mats, embs_by_key):
        return np.c_[np.asarray(log_vol)[:, None], geo_feats(geo_dicts), onehot(forms, mats), zs_concepts(embs_by_key) * self.zs_scale]

    def fit(self, train_objs):
        import sys
        sys.path.insert(0, str(ROOT))
        from evaluation.volume_parts_cv import apply_calibrator, fit_calibrator
        F = dict(np.load(C / "features.npz"))
        F2 = dict(np.load(C / "features2.npz"))
        meta = json.load(open(C / "meta.json"))
        gk = list(F["geo_keys"])
        rows = pd.read_csv(ROOT / "reports/volume_parts_gt_rows.csv")
        rows = rows[rows.split == "real"]
        key = {(r.image_id, r.object_id): i for i, r in rows.reset_index().iterrows()}
        idx = [i for i, m in enumerate(meta) if m["src"] == "real_train" and m["oid"] in set(train_objs)]
        tr_rows = rows.iloc[[key[(meta[i]["frame"], meta[i]["oid"])] for i in idx]].copy()
        cal = fit_calibrator(tr_rows)
        self.cal = cal
        log_vol = np.log([apply_calibrator(cal, r, r["form"], r["material"]) for r in tr_rows.to_dict("records")])
        geo = [dict(zip(gk, F["geo"][i])) for i in idx]
        forms = [meta[i]["form"] for i in idx]
        mats = [meta[i]["mat"] for i in idx]
        embs = {(t, v): F2[f"{t}_{v}"][idx] for t in ZS_TAGS for v in ZS_VIEWS}
        X = self._X(log_vol, geo, forms, mats, embs)
        y = np.log([meta[i]["mass"] for i in idx])
        oid = np.array([meta[i]["oid"] for i in idx])
        cnt = defaultdict(int)
        for o in oid:
            cnt[o] += 1
        w = np.array([1.0 / cnt[o] for o in oid])
        heavy = np.array([meta[i]["mass"] > 500 for i in idx])
        # порог — внутренний grouped CV
        objs = np.array(sorted(set(oid)))
        np.random.default_rng(self.seed).shuffle(objs)
        oof = np.zeros(len(y))
        for g in np.array_split(objs, self.inner_folds):
            te = np.isin(oid, g)
            oof[te] = _ridge(X[~te], y[~te], w[~te], self.alpha)(X[te])
        best = (-1, math.log(500))
        for t in np.unique(np.r_[oof, math.log(500)]):
            p = oof > t
            s = 0.5 * (np.sum(w[heavy] * p[heavy]) / w[heavy].sum() + np.sum(w[~heavy] * ~p[~heavy]) / w[~heavy].sum())
            if s > best[0]:
                best = (s, float(t))
        self.thr_logmass = best[1]
        self.inner_bal_acc = best[0]
        self.model = _ridge(X, y, w, self.alpha)
        return self

    def predict(self, log_vol, geo_dicts, forms, mats, embs_by_key):
        """Возвращает (take 0/1, log-масса)."""
        lm = self.model(self._X(log_vol, geo_dicts, forms, mats, embs_by_key))
        return (lm <= self.thr_logmass).astype(int), lm
