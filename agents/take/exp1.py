"""Experiment 1: baselines + log-mass ridge regressors with different feature sets (nested LOOO)."""
from __future__ import annotations

import json
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, "/home/cleaner/Cleaner/agents/take")
from take_data import HEAVY, MATS, FORMS, CACHE, build_table, load_feats
from take_lib import (LOG500, fmt, object_weights, permutation_check, ridge_fit, ridge_predict, run_loo, summarize)
from evaluation.volume_parts_cv import apply_calibrator, fit_calibrator

df = build_table()
F = load_feats()
PARTS_COLS = ["columns_ml", "parts_ml", "hull_ml", "footprint_ml", "valid_frac", "truncated", "n_parts", "p95_h_m",
              "footprint_m2", "fp_area_0", "fp_area_20", "fp_area_40", "fp_area_80", "fp_area_160", "min_h_m"]
ZS_COLS = [c for c in df.columns if c.startswith("zs_")]
GEO_COLS = [c for c in df.columns if c.startswith("g_")]


def cal_volume(tr: pd.DataFrame, te: pd.DataFrame, mat_col="mat") -> np.ndarray:
    """Project calibrator fitted on real training rows -> calibrated volume for te rows."""
    r = tr[tr.real]
    rows = r[PARTS_COLS].copy()
    rows["object_id"], rows["form"], rows["material"], rows["volume_ml"] = r.oid.values, r.form.values, r[mat_col].values, r.vol.values
    cal = fit_calibrator(rows)
    return np.array([apply_calibrator(cal, x, x["form"], x[mat_col]) for x in te.to_dict("records")])


def class_density(tr: pd.DataFrame) -> dict:
    r = tr[tr.real].groupby("oid").first()
    d = {}
    for (f, m), g in r.groupby(["form", "mat"]):
        d[(f, m)] = float(np.exp(np.mean(np.log(g.mass / g.vol))))
    for f, g in r.groupby("form"):
        d[(f, "*")] = float(np.exp(np.mean(np.log(g.mass / g.vol))))
    d[("*", "*")] = float(np.exp(np.mean(np.log(r.mass / r.vol))))
    return d


def lookup_density(d, f, m):
    return d.get((f, m), d.get((f, "*"), d[("*", "*")]))


# ------------------------------------------------------------------ score functions (heavy score)

def s_volume(tr, te):
    return np.log(cal_volume(tr, te)), None


def s_mass_density(tr, te):
    v = cal_volume(tr, te)
    d = class_density(tr)
    return np.log(v * np.array([lookup_density(d, f, m) for f, m in zip(te.form, te.mat)])), None


def onehot(d: pd.DataFrame, mat_col="mat"):
    X = [np.array([(d.form == f).values for f in FORMS], float).T,
         np.array([(d[mat_col] == m).values for m in MATS], float).T]
    return np.concatenate(X, 1)


class EmbPCA:
    def __init__(self, keys, n):
        self.keys, self.n = keys, n

    def fit(self, idx):
        X = np.concatenate([F[k][idx] for k in self.keys], 1)
        self.mu = X.mean(0)
        U, S, Vt = np.linalg.svd(X - self.mu, full_matrices=False)
        self.V = Vt[: self.n].T
        self.scale = S[: self.n] / np.sqrt(len(idx))
        return self

    def transform(self, idx):
        X = np.concatenate([F[k][idx] for k in self.keys], 1)
        return ((X - self.mu) @ self.V) / self.scale


def make_mass_ridge(feature_fn, alpha=10.0, synth_w=0.5, inner_alpha=False):
    """Weighted ridge on log(mass); heavy score = predicted log mass."""
    def score(tr, te):
        Xtr, Xte = feature_fn(tr, te)
        y = np.log(tr.mass.values)
        w = object_weights(tr.oid.values, tr.src.values, synth_w)
        a = alpha
        if inner_alpha:
            # inner grouped 5-fold on training objects for alpha by weighted RMSE
            objs = sorted(tr.oid.unique())
            rng = np.random.default_rng(0)
            perm = list(rng.permutation(objs))
            best = None
            for cand in (1, 3, 10, 30, 100, 300):
                err = 0.0
                for k in range(5):
                    g = perm[k::5]
                    mtr = ~tr.oid.isin(g).values
                    m = ridge_fit(Xtr[mtr], y[mtr], w[mtr], cand)
                    p = ridge_predict(m, Xtr[~mtr])
                    err += np.sum(w[~mtr] * (p - y[~mtr]) ** 2)
                if best is None or err < best[0]:
                    best = (err, cand)
            a = best[1]
        m = ridge_fit(Xtr, y, w, a)
        return ridge_predict(m, Xte), a
    return score


def feats_geo(tr, te, mat_col="mat"):
    """Only features available for synth too: g_* geometry + class one-hot."""
    def f(d):
        g = d[GEO_COLS].values.astype(float)
        lv = np.log(np.maximum(d[["g_vol", "g_vol_thr", "g_foot", "g_surf"]].values, 1.0))
        return np.concatenate([lv, g[:, [GEO_COLS.index(c) for c in ["g_h95", "g_hmax", "g_h_med", "g_ext1", "g_ext2", "g_valid", "g_holes_edge", "g_holes_inner", "g_rough"]]], onehot(d, mat_col)], 1)
    return f(tr), f(te)


def feats_geo_cal(tr, te, mat_col="mat"):
    """Real-only: parts geometry + nested calibrated volume + class one-hot."""
    tr = tr[tr.real]
    vtr = np.log(cal_volume(tr, tr, mat_col))  # in-sample for train rows (cheap); test is out-of-sample
    vte = np.log(cal_volume(tr, te, mat_col))
    def f(d, v):
        lv = np.log(np.maximum(d[["parts_ml", "columns_ml", "hull_ml", "footprint_ml", "g_vol"]].values.astype(float), 1.0))
        extra = d[["p95_h_m", "valid_frac", "g_h95", "g_ext1", "g_ext2", "g_holes_inner", "g_rough"]].values.astype(float)
        return np.concatenate([v[:, None], lv, extra, onehot(d, mat_col)], 1)
    return f(tr, vtr), f(te, vte)


def with_zs(base, zs_cols=ZS_COLS, scale=20.0):
    def f(tr, te):
        a, b = base(tr, te)
        trr = tr[tr.real] if a.shape[0] == tr.real.sum() else tr
        return (np.concatenate([a, trr[zs_cols].values * scale], 1), np.concatenate([b, te[zs_cols].values * scale], 1))
    return f


def with_pca(base, keys, n=16):
    def f(tr, te):
        a, b = base(tr, te)
        trr = tr[tr.real] if a.shape[0] == tr.real.sum() else tr
        p = EmbPCA(keys, n).fit(trr.idx.values)
        return (np.concatenate([a, p.transform(trr.idx.values)], 1), np.concatenate([b, p.transform(te.idx.values)], 1))
    return f


def real_only(fn):
    def f(tr, te):
        return fn(tr[tr.real], te)
    return f


VARIANTS = {
    "A0 cal volume <= T": (s_volume, False, LOG500),
    "A1 vol x class density": (s_mass_density, False, LOG500),
    "B0 ridge geo+class (real)": (make_mass_ridge(feats_geo, inner_alpha=True), False, LOG500),
    "B1 ridge geo+class (real+synth)": (make_mass_ridge(feats_geo, inner_alpha=True), True, LOG500),
    "B2 ridge geo_cal+class (real)": (make_mass_ridge(feats_geo_cal, inner_alpha=True), False, LOG500),
    "C0 ridge geo_cal+class+zs": (make_mass_ridge(with_zs(feats_geo_cal), inner_alpha=True), False, LOG500),
    "C1 ridge geo+class+zs (r+s)": (make_mass_ridge(with_zs(feats_geo), inner_alpha=True), True, LOG500),
    "C2 ridge geo_cal+class+clipL pca16": (make_mass_ridge(with_pca(feats_geo_cal, ["clipL_c14"], 16), inner_alpha=True), False, LOG500),
    "C3 ridge geo+class+clipL pca16 (r+s)": (make_mass_ridge(with_pca(feats_geo, ["clipL_c14"], 16), inner_alpha=True), True, LOG500),
    "C4 ridge geo_cal+class+zs+pca16": (make_mass_ridge(with_pca(with_zs(feats_geo_cal), ["clipL_c14"], 16), inner_alpha=True), False, LOG500),
    "C5 ridge geo+class+zs+dinoL pca16 (r+s)": (make_mass_ridge(with_pca(with_zs(feats_geo), ["dinoL_m10"], 16), inner_alpha=True), True, LOG500),
}


def main():
    which = sys.argv[1:] or list(VARIANTS)
    real = df[df.real].reset_index(drop=True)
    results = {}
    for name in which:
        fn, use_synth, dt = VARIANTS[name]
        t = time.time()
        score, pred, ths = run_loo(df, fn, use_synth, inner_threshold=True, default_t=dt, inner_folds=None)
        r = summarize(real, pred)
        r["thresholds"] = [float(x) for x in ths]
        # fixed-threshold (500 g) variant for mass-type scores
        pred_fixed = score > dt
        r["S_fixed"] = summarize(real, pred_fixed)["S"]
        results[name] = r
        print(fmt(name, r), f"| S@fixed={r['S_fixed']:.3f} | {time.time()-t:.0f}s", flush=True)
        np.save(CACHE / f"score_{name.split()[0]}.npy", score)
    json.dump(results, open(CACHE / "exp1_results.json", "w"), indent=1)


if __name__ == "__main__":
    main()
