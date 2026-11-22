"""Exp4: design variants (knots, interactions), series-median aggregation, honest ensembles."""
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


def series_median(pred, gap=3):
    """Median of log-predictions over the frame series of the same object (gap<=3), as the pipeline does."""
    out = pred.copy()
    for o in set(ids):
        ii = np.where(ids == o)[0]
        ii = ii[np.argsort(fnum[ii])]
        groups, cur = [], [ii[0]]
        for a, b in zip(ii[:-1], ii[1:]):
            if fnum[b] - fnum[a] <= gap:
                cur.append(b)
            else:
                groups.append(cur)
                cur = [b]
        groups.append(cur)
        for g in groups:
            out[g] = np.median(pred[g])
    return out


def design2(d, cols, inter, knots, inter_mat=True, inter_form=True, cls_inter=True, extra=None):
    X = np.stack([np.log(np.maximum(d[c].to_numpy(float), 1.0)) if c.endswith("_ml") else d[c].to_numpy(float) for c in cols], 1)
    form = (d.form == "bulk").to_numpy().astype(float)[:, None]
    mat = np.stack([(d.material == m).to_numpy() for m in MATS], 1).astype(float)
    parts = [X, form, mat]
    if cls_inter:
        parts.append(form * mat)
    main = X[:, cols.index(inter)][:, None]
    if inter_mat:
        parts.append(main * mat)
    if inter_form:
        parts.append(main * form)
    for t in knots:
        parts.append(np.maximum(0, main[:, 0] - math.log(t))[:, None])
    if extra is not None:
        parts.append(np.asarray(extra, float).reshape(len(d), -1))
    return np.hstack(parts)


import math  # noqa

P = {}
X = design2(df, ref, "gi_env_ml", (50, 300))
P["ref"] = nested_looo(X, y, ids, fit_irls, IRLS_GRID)
report("ref IRLS", P["ref"], df)
report("ref IRLS + series median", series_median(P["ref"]), df)
report("ref IRLS + object median (ceiling for aggregation)", series_median(P["ref"], gap=10000), df)

for knots in ((50, 300, 1000), (30, 100, 300, 1000), (100, 1000), (50, 300, 2000), ()):
    X = design2(df, ref, "gi_env_ml", knots)
    P[f"knots{knots}"] = nested_looo(X, y, ids, fit_irls, IRLS_GRID)
    report(f"IRLS knots={knots}", P[f"knots{knots}"], df)

for im, ifm, ci in ((False, True, True), (True, False, True), (False, False, True), (False, False, False), (True, True, False)):
    X = design2(df, ref, "gi_env_ml", (50, 300), inter_mat=im, inter_form=ifm, cls_inter=ci)
    P[f"int{im}{ifm}{ci}"] = nested_looo(X, y, ids, fit_irls, IRLS_GRID)
    report(f"IRLS inter_mat={im} inter_form={ifm} form*mat={ci}", P[f"int{im}{ifm}{ci}"], df)

# different main candidates for the interaction/knots
for main in ("parts_ml", "columns_ml", "footprint_ml", "hull_ml"):
    X = design2(df, ref, main, (50, 300))
    P[f"main{main}"] = nested_looo(X, y, ids, fit_irls, IRLS_GRID)
    report(f"IRLS main={main}", P[f"main{main}"], df)

# honest ensembles (no selection): all variants / all knot variants
ens = np.mean([P[k] for k in P], 0)
report("ensemble all variants", ens, df)
report("ensemble all variants + series median", series_median(ens), df)
ens2 = np.mean([P[k] for k in P if k.startswith("main") or k == "ref"], 0)
report("ensemble main-candidate variants", ens2, df)
report("ensemble main-candidate variants + series median", series_median(ens2), df)
np.save(HERE + "/cache_exp4_preds.npy", P, allow_pickle=True)
