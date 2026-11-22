"""Exp7: residual models on top of the IRLS reference (stacked honestly: residual model fit on inner-OOF
residuals of the training objects), with shape-descriptor kNN / small GBM / ridge; plus 'sequential knots on
height' (thin objects)."""
import sys
import warnings

warnings.filterwarnings("ignore")
sys.path.insert(0, "/home/cleaner/Cleaner/agents2/volume")
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.neighbors import KNeighborsRegressor

from common import *
from vmodels import *

df = load_rows()
y = np.log(df.volume_ml.values)
ids = df.object_id.values
ref = ["columns_ml", "parts_ml", "hull_ml", "footprint_ml", "gi_env_ml"]
X_ref = design(df, ref, inter="gi_env_ml", knots=(50, 300))
w_all = w_of(ids)

df["log_h"] = np.log(np.maximum(df.p95_h_m, 0.004))
df["log_env"] = np.log(np.maximum(df.gi_env_ml, 1))
df["aspect"] = np.log(np.maximum(df.inp_L_m, 1e-3) / np.maximum(df.inp_W_m, 1e-3))
df["h_over_w"] = np.log(np.maximum(df.p95_h_m, 0.004) / np.maximum(df.inp_W_m, 1e-3))
df["hidden"] = np.log(np.maximum(df.inp_sym_cells_m2, 1e-5) / np.maximum(df.inp_cells_m2, 1e-5))
df["env_over_col"] = np.log(np.maximum(df.gi_env_ml, 1) / np.maximum(df.i0_colmax_ml, 1))
df["vh_over_env"] = np.log(np.maximum(df.vhc_ml, 1) / np.maximum(df.gi_env_ml, 1))
df["hull_over_env"] = np.log(np.maximum(df.hull_ml, 1) / np.maximum(df.gi_env_ml, 1))
df["bulk"] = (df.form == "bulk").astype(float)
S_COLS = ["log_h", "log_env", "aspect", "h_over_w", "hidden", "env_over_col", "vh_over_env", "hull_over_env", "fill", "gap",
          "valid_frac", "hole_inner_frac", "hole_edge_frac", "cos_tilt", "bulk"]
S = df[S_COLS].to_numpy(float)


def inner_oof_resid(Xt, yt, it, wt, g):
    f = inner_folds(it, 5, 1)
    p = np.zeros(len(yt))
    for k in range(5):
        p[f == k] = fit_irls(Xt[f != k], yt[f != k], wt[f != k], g)(Xt[f == k])
    return yt - p


def stacked(resid_fitter, name, shrink_grid=(0.0, 0.5, 1.0)):
    """Outer LOOO. Inside: pick IRLS g by inner CV; compute inner-OOF residuals; fit residual model on
    (S, resid); choose shrink in {0,.5,1} by a second inner CV on the residual model; predict."""
    pred = np.zeros(len(y))
    for o in sorted(set(ids)):
        tr = ids != o
        Xt, yt, it, wt, St = X_ref[tr], y[tr], ids[tr], w_all[tr], S[tr]
        f = inner_folds(it, 5, 0)
        best = None
        for g in IRLS_GRID:
            p = np.zeros(len(yt))
            for k in range(5):
                p[f == k] = fit_irls(Xt[f != k], yt[f != k], wt[f != k], g)(Xt[f == k])
            s = np.sum(wt * vscore(p, yt)) / wt.sum()
            if best is None or s > best[0]:
                best = (s, g)
        g = best[1]
        r = inner_oof_resid(Xt, yt, it, wt, g)
        # choose shrink by grouped CV of the residual model on the OOF residuals
        f2 = inner_folds(it, 5, 2)
        rp = np.zeros(len(yt))
        for k in range(5):
            rp[f2 == k] = resid_fitter(St[f2 != k], r[f2 != k], wt[f2 != k])(St[f2 == k])
        base_p = yt - r
        bests = None
        for sh in shrink_grid:
            s = np.sum(wt * vscore(base_p + sh * rp, yt)) / wt.sum()
            if bests is None or s > bests[0]:
                bests = (s, sh)
        sh = bests[1]
        m = fit_irls(Xt, yt, wt, g)
        rm = resid_fitter(St, r, wt)
        pred[~tr] = m(X_ref[~tr]) + sh * rm(S[~tr])
    return report(name, pred, df), pred


def knn_fitter(k):
    def fit(St, r, wt):
        mu, sd = St.mean(0), St.std(0) + 1e-9
        m = KNeighborsRegressor(n_neighbors=k, weights="distance").fit((St - mu) / sd, r, )
        return lambda Sn: m.predict((Sn - mu) / sd)
    return fit


def gbm_fitter(n, depth=2, leaf=15):
    def fit(St, r, wt):
        m = GradientBoostingRegressor(n_estimators=n, max_depth=depth, learning_rate=0.05, subsample=0.8,
                                      min_samples_leaf=leaf, loss="huber", random_state=0).fit(St, r, sample_weight=wt)
        return m.predict
    return fit


def ridge_fitter(a):
    def fit(St, r, wt):
        return ridge(St, r, wt, a)
    return fit


p_ref = nested_looo(X_ref, y, ids, fit_irls, IRLS_GRID)
report("ref IRLS", p_ref, df)
P = {}
for k in (10, 25, 50):
    P[f"knn{k}"] = stacked(knn_fitter(k), f"stack kNN-resid k={k} (shape descr.)")[1]
for n in (50, 150):
    P[f"gbm{n}"] = stacked(gbm_fitter(n), f"stack GBM-resid n={n}")[1]
for a in (10, 100):
    P[f"ridge{a}"] = stacked(ridge_fitter(a), f"stack ridge-resid a={a}")[1]
report("mean of all stacks", np.mean(list(P.values()), 0), df)
np.save(HERE + "/cache_exp7_preds.npy", P, allow_pickle=True)
