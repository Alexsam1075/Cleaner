"""Exp16: thin-object handling (p95 height < 30 mm): soft indicator + interactions, separate expert,
and area x class-thickness candidate as an extra feature. Nested LOOO."""
import math
import sys
import warnings

warnings.filterwarnings("ignore")
sys.path.insert(0, "/home/cleaner/Cleaner/agents2/volume")
import numpy as np
import pandas as pd

from common import *
from vmodels import *

df = load_rows()
y = np.log(df.volume_ml.values)
ids = df.object_id.values
ref = ["columns_ml", "parts_ml", "hull_ml", "footprint_ml", "gi_env_ml"]
X_ref = design(df, ref, inter="gi_env_ml", knots=(50, 300))
h = df.p95_h_m.values
thin = (h < 0.03).astype(float)
soft = 1.0 / (1.0 + np.exp((h - 0.03) / 0.005))
lenv = np.log(np.maximum(df.gi_env_ml.values, 1))
lfp = np.log(np.maximum(df.footprint_m2.values, 1e-5))
print("thin frames:", int(thin.sum()), "objects:", df.object_id[thin > 0].nunique())

report("ref IRLS", nested_looo(X_ref, y, ids, fit_irls, IRLS_GRID), df)
for name, e in {"thin": thin[:, None], "thin, thin*lenv": np.c_[thin, thin * lenv], "soft, soft*lenv": np.c_[soft, soft * lenv],
                "thin*lfp, thin": np.c_[thin * lfp, thin], "log h": np.log(np.maximum(h, 0.004))[:, None]}.items():
    X = design(df, ref, inter="gi_env_ml", knots=(50, 300), extra=e)
    report("IRLS + " + name, nested_looo(X, y, ids, fit_irls, IRLS_GRID), df)

# separate expert for thin frames (class thickness learned via material one-hot + log footprint), nested
pred = np.zeros(len(y))
w = w_of(ids)
Xthin = design(df, ["footprint_ml", "columns_ml", "gi_env_ml"], inter="columns_ml", knots=(), extra=lfp)
for o in sorted(set(ids)):
    tr = ids != o
    # main model on all rows
    f = inner_folds(ids[tr], 5, 0)
    best = None
    for g in IRLS_GRID:
        p = np.zeros(tr.sum())
        for k in range(5):
            p[f == k] = fit_irls(X_ref[tr][f != k], y[tr][f != k], w[tr][f != k], g)(X_ref[tr][f == k])
        s = np.sum(w[tr] * vscore(p, y[tr])) / w[tr].sum()
        if best is None or s > best[0]:
            best = (s, g)
    m = fit_irls(X_ref[tr], y[tr], w[tr], best[1])
    te = ids == o
    pm = m(X_ref[te])
    trt = tr & (thin > 0)
    if thin[te].any() and trt.sum() > 20:
        mt = irls(Xthin[trt], y[trt], w[trt], 30.0, floor=0.2)
        pt = mt(Xthin[te])
        pred[te] = np.where(thin[te] > 0, 0.5 * (pm + pt), pm)
    else:
        pred[te] = pm
report("thin expert blended 50/50 on thin frames", pred, df)
