"""Exp11: loss shaping. (a) IRLS with wider floor/alpha grid and 'cap' variants; (b) object-level training rows
(median log-features per object) vs frame rows; (c) two-stage: object-level fit then frame-level refinement;
(d) alpha chosen by LOO-objects inside (40-1 inner folds) instead of 5-fold."""
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

report("ref IRLS", nested_looo(X_ref, y, ids, fit_irls, IRLS_GRID), df)
WIDE = [(a, fl) for a in (1, 3, 10, 30, 100) for fl in (0.05, 0.1, 0.2, 0.3, 0.5)]
report("IRLS wide grid", nested_looo(X_ref, y, ids, fit_irls, WIDE), df)


def fit_irls_out(X, yt, w, g):
    """Outlier weight (beyond ln3) as a hyper-parameter: 0 = ignore gross outliers completely."""
    a, fl, ow = g
    m = ridge(X, yt, w, a)
    for _ in range(8):
        r = np.abs(m(X) - yt)
        m = ridge(X, yt, w * np.where(r < L3, 1.0 / np.maximum(r, fl), ow / L3), a)
    return m


report("IRLS outlier-weight grid", nested_looo(X_ref, y, ids, fit_irls_out, [(a, 0.2, ow) for a in (3, 10, 30, 100) for ow in (0.0, 0.1, 0.3, 1.0)]), df)
report("IRLS a in {3,10,30} inner=8 folds", nested_looo(X_ref, y, ids, fit_irls, [(a, fl) for a in (3, 10, 30) for fl in (0.1, 0.2, 0.3)], inner=8), df)

# (b) object-level rows
d_obj = df.groupby("object_id").agg({**{c: "median" for c in ref}, "form": "first", "material": "first", "volume_ml": "first"}).reset_index()
Xo = design(d_obj, ref, inter="gi_env_ml", knots=(50, 300))
yo = np.log(d_obj.volume_ml.values)
ido = d_obj.object_id.values
# map back to frames
pos = {o: i for i, o in enumerate(ido)}
pred = np.zeros(len(y))
wo = np.ones(len(yo))
for o in sorted(set(ids)):
    tr = ido != o
    f = inner_folds(ido[tr], 5, 0)
    best = None
    for g in IRLS_GRID:
        p = np.zeros(tr.sum())
        for k in range(5):
            p[f == k] = fit_irls(Xo[tr][f != k], yo[tr][f != k], wo[tr][f != k], g)(Xo[tr][f == k])
        s = vscore(p, yo[tr]).mean()
        if best is None or s > best[0]:
            best = (s, g)
    m = fit_irls(Xo[tr], yo[tr], wo[tr], best[1])
    pred[ids == o] = m(X_ref[ids == o])
report("object-level rows (median feats) -> frames", pred, df)

# (c) frame rows but with the object-level model as an offset (residual IRLS on frames, strong reg)
pred2 = np.zeros(len(y))
for o in sorted(set(ids)):
    tr = ido != o
    m = fit_irls(Xo[tr], yo[tr], wo[tr], (10, 0.2))
    trf = ids != o
    off = m(X_ref)
    r = y - off
    m2 = irls(X_ref[trf], r[trf], w_of(ids[trf]), 100.0, floor=0.2)
    pred2[~trf] = off[~trf] + m2(X_ref[~trf])
report("object-level fit + frame residual (a=100)", pred2, df)
