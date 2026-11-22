"""Exp10: trust silhouette/hull candidates more when depth is missing: (1-valid_frac) x log(candidate/main)
interactions; depth-hole fraction interactions. Nested LOOO + 5-fold x 3 seeds for the survivors."""
import sys
import warnings

warnings.filterwarnings("ignore")
sys.path.insert(0, "/home/cleaner/Cleaner/agents2/volume")
import numpy as np

from common import *
from vmodels import *

df = load_rows()
y = np.log(df.volume_ml.values)
ids = df.object_id.values
ref = ["columns_ml", "parts_ml", "hull_ml", "footprint_ml", "gi_env_ml"]
X_ref = design(df, ref, inter="gi_env_ml", knots=(50, 300))
miss = 1.0 - df.valid_frac.values
hole = df.hole_inner_frac.values
lenv = np.log(np.maximum(df.gi_env_ml.values, 1))
ex = {
    "miss*log(vhc/env)": miss * (np.log(np.maximum(df.vhc_ml.values, 1)) - lenv),
    "miss*log(hull/env)": miss * (np.log(np.maximum(df.hull_ml.values, 1)) - lenv),
    "miss*log(fp/env)": miss * (np.log(np.maximum(df.footprint_ml.values, 1)) - lenv),
    "miss*log(symbox/env)": miss * (np.log(np.maximum(df.inp_sym_box_ml.values, 1)) - lenv),
    "hole*log(hull/env)": hole * (np.log(np.maximum(df.hull_ml.values, 1)) - lenv),
    "miss": miss,
    "miss*lenv": miss * lenv,
}


def kfold_oof(X, seed, n=5):
    pred = np.zeros(len(y))
    w = w_of(ids)
    f = inner_folds(ids, n, 100 + seed)
    for k in range(n):
        tr = f != k
        Xt, yt, it, wt = X[tr], y[tr], ids[tr], w[tr]
        fi = inner_folds(it, 5, 0)
        best = None
        for g in IRLS_GRID:
            p = np.zeros(len(yt))
            for kk in range(5):
                p[fi == kk] = fit_irls(Xt[fi != kk], yt[fi != kk], wt[fi != kk], g)(Xt[fi == kk])
            s = np.sum(wt * vscore(p, yt)) / wt.sum()
            if best is None or s > best[0]:
                best = (s, g)
        pred[~tr] = fit_irls(Xt, yt, wt, best[1])(X[~tr])
    return pred


def kf3(X, name):
    vs = [vscore(kfold_oof(X, s), y).mean() for s in range(3)]
    print(f"   5fold x3 {name:40s} {np.mean(vs):.4f} ± {np.std(vs):.4f}", flush=True)


report("ref IRLS", nested_looo(X_ref, y, ids, fit_irls, IRLS_GRID), df)
kf3(X_ref, "ref")
for name, e in ex.items():
    X = design(df, ref, inter="gi_env_ml", knots=(50, 300), extra=e)
    v = report("IRLS + " + name, nested_looo(X, y, ids, fit_irls, IRLS_GRID), df)
    if v > 0.778:
        kf3(X, name)
for combo in (["miss*log(vhc/env)", "miss*log(hull/env)"], ["miss", "miss*log(hull/env)"], ["miss*log(hull/env)", "hole*log(hull/env)"]):
    X = design(df, ref, inter="gi_env_ml", knots=(50, 300), extra=np.c_[[ex[c] for c in combo]].T)
    v = report("IRLS + " + "+".join(combo), nested_looo(X, y, ids, fit_irls, IRLS_GRID), df)
    if v > 0.778:
        kf3(X, "+".join(combo))
