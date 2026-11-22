"""Exp9: (F) coarser material groups for the volume calibrator (fewer class parameters);
(J) series-level feature pooling (median of log candidates over the frame series) before calibration
vs median of per-frame predictions; combined with the series protocol used by the pipeline."""
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
fnum = df.image_id.str[2:].astype(int).values
ref = ["columns_ml", "parts_ml", "hull_ml", "footprint_ml", "gi_env_ml"]


def series_ids(gap=3):
    sid = np.zeros(len(df), int)
    k = 0
    for o in sorted(set(ids)):
        ii = np.where(ids == o)[0]
        ii = ii[np.argsort(fnum[ii])]
        sid[ii[0]] = k
        for a, b in zip(ii[:-1], ii[1:]):
            if fnum[b] - fnum[a] > gap:
                k += 1
            sid[b] = k
        k += 1
    return sid


SID = series_ids(3)


def series_median(pred, sid=SID):
    out = pred.copy()
    for s in set(sid):
        out[sid == s] = np.median(pred[sid == s])
    return out


def design_g(d, cols, inter, knots, groups):
    """Like design() but material replaced by a coarse group label."""
    X = np.stack([np.log(np.maximum(d[c].to_numpy(float), 1.0)) for c in cols], 1)
    form = (d.form == "bulk").to_numpy().astype(float)[:, None]
    gl = sorted(set(groups.values()))
    mat = np.stack([(d.material.map(groups) == g).to_numpy() for g in gl], 1).astype(float)
    main = X[:, cols.index(inter)][:, None]
    parts = [X, form, mat, form * mat, main * mat, main * form] + [np.maximum(0, main - math.log(t)) for t in knots]
    return np.hstack(parts)


X_ref = design(df, ref, inter="gi_env_ml", knots=(50, 300))
p_ref = nested_looo(X_ref, y, ids, fit_irls, IRLS_GRID)
report("ref IRLS", p_ref, df)
report("ref IRLS | series median", series_median(p_ref), df)

GROUPS = {
    "dense/light": {"glass": "dense", "metal": "dense", "mineral": "dense", "paper": "light", "plastic": "light", "wood": "light"},
    "3 groups": {"glass": "hard", "metal": "hard", "mineral": "hard", "wood": "hard", "paper": "soft", "plastic": "plastic"},
    "wood/other": {"glass": "o", "metal": "o", "mineral": "o", "paper": "o", "plastic": "o", "wood": "wood"},
    "plastic/wood/other": {"glass": "o", "metal": "o", "mineral": "o", "paper": "o", "plastic": "plastic", "wood": "wood"},
}
P = {"ref": p_ref}
for name, gmap in GROUPS.items():
    X = design_g(df, ref, "gi_env_ml", (50, 300), gmap)
    P[name] = nested_looo(X, y, ids, fit_irls, IRLS_GRID)
    report(f"IRLS mat groups {name}", P[name], df)
    report(f"IRLS mat groups {name} | series median", series_median(P[name]), df)
# full material + coarse group both (group gets the shared part, ridge shrinks the fine part)
gmap = GROUPS["dense/light"]
grp = np.stack([(df.material.map(gmap) == g).to_numpy() for g in ("dense", "light")], 1).astype(float)
main = np.log(np.maximum(df.gi_env_ml.values, 1))[:, None]
X = design(df, ref, inter="gi_env_ml", knots=(50, 300), extra=np.c_[grp, grp * main])
P["full+dense/light"] = nested_looo(X, y, ids, fit_irls, IRLS_GRID)
report("IRLS full material + dense/light", P["full+dense/light"], df)
ens = np.mean([P[k] for k in P], 0)
report("ensemble ref + group variants", ens, df)
report("ensemble ref + group variants | series median", series_median(ens), df)

# (J) series-level pooled features
dp = df.copy()
for c in ref:
    dp[c] = np.exp(pd.Series(np.log(np.maximum(df[c].values, 1))).groupby(SID).transform("median").values)
Xp = design(dp, ref, inter="gi_env_ml", knots=(50, 300))
p_pool = nested_looo(Xp, y, ids, fit_irls, IRLS_GRID)
report("IRLS on series-pooled candidates", p_pool, df)
report("IRLS on series-pooled candidates | series median", series_median(p_pool), df)
# train on per-frame rows, predict on pooled features
p_mixed = nested_looo(X_ref, y, ids, fit_irls, IRLS_GRID, Xeval=Xp)
report("IRLS train frames / predict pooled", p_mixed, df)
# pooled with mean instead of median
dp2 = df.copy()
for c in ref:
    dp2[c] = np.exp(pd.Series(np.log(np.maximum(df[c].values, 1))).groupby(SID).transform("mean").values)
report("IRLS on series-pooled (mean) candidates", nested_looo(design(dp2, ref, inter="gi_env_ml", knots=(50, 300)), y, ids, fit_irls, IRLS_GRID), df)
np.save(HERE + "/cache_exp9_preds.npy", {**P, "pool": p_pool}, allow_pickle=True)
