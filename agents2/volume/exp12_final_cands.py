"""Exp12: candidate final configurations, evaluated per-frame and with series aggregation (gap<=3),
clean material and 8% noisy material. Also tests predicting on series-pooled features."""
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
base = ["columns_ml", "parts_ml", "hull_ml", "footprint_ml"]

rng = np.random.default_rng(5)
noisy = df.copy()
flip = rng.random(len(df)) < 0.08
noisy.loc[flip, "material"] = [rng.choice([m for m in MATS if m != x]) for x in df.material[flip]]


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


def smed(pred):
    out = pred.copy()
    for s in set(SID):
        out[SID == s] = np.median(pred[SID == s])
    return out


def run(name, cols, inter, knots=(50, 300), extra_cols=None, fitter=fit_irls, grid=IRLS_GRID):
    ex = df[extra_cols].to_numpy(float) if extra_cols else None
    X = design(df, cols, inter=inter, knots=knots, extra=ex)
    Xn = design(noisy, cols, inter=inter, knots=knots, extra=ex)
    p = nested_looo(X, y, ids, fitter, grid)
    pn = nested_looo(X, y, ids, fitter, grid, Xeval=Xn)
    report(f"{name} | frame", p, df)
    report(f"{name} | frame, mat 8% noise", pn, df)
    report(f"{name} | series median", smed(p), df)
    report(f"{name} | series median, mat 8% noise", smed(pn), df)
    return p, pn


R = {}
R["ref5 IRLS"] = run("ref5 IRLS", ref, "gi_env_ml")
R["ref5 ridge"] = run("ref5 ridge", ref, "gi_env_ml", fitter=fit_ridge, grid=RIDGE_GRID)
R["base4 IRLS"] = run("base4 IRLS", base, "parts_ml")
R["base4 IRLS main=columns"] = run("base4 IRLS main=columns", base, "columns_ml")
R["ref5 IRLS + cos_tilt"] = run("ref5 IRLS + cos_tilt", ref, "gi_env_ml", extra_cols=["cos_tilt"])
R["ref5 IRLS + fill+hole_edge"] = run("ref5 IRLS + fill+hole_edge", ref, "gi_env_ml", extra_cols=["hole_edge_frac", "fill"])
ens = np.mean([R[k][0] for k in ("ref5 IRLS", "ref5 ridge", "base4 IRLS", "base4 IRLS main=columns")], 0)
ensn = np.mean([R[k][1] for k in ("ref5 IRLS", "ref5 ridge", "base4 IRLS", "base4 IRLS main=columns")], 0)
report("ens4 | frame", ens, df)
report("ens4 | frame, mat 8% noise", ensn, df)
report("ens4 | series median", smed(ens), df)
report("ens4 | series median, mat 8% noise", smed(ensn), df)
np.save(HERE + "/cache_exp12.npy", {k: v[0] for k, v in R.items()}, allow_pickle=True)
