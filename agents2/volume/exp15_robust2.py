"""Exp15: robustness of the few candidate improvements: 5-fold grouped x 10 seeds, and paired per-object
LOOO comparison (how many objects gain/lose vs ref)."""
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


sets = {"ref": None, "ref+fill": ["fill"], "ref+fill+hole_edge": ["hole_edge_frac", "fill"], "ref+hole_edge": ["hole_edge_frac"],
        "ref+cos_tilt": ["cos_tilt"], "ref+fill+hole_edge+cos_tilt": ["hole_edge_frac", "fill", "cos_tilt"]}
res = {}
for name, ex in sets.items():
    X = design(df, ref, inter="gi_env_ml", knots=(50, 300), extra=df[ex].to_numpy(float) if ex else None)
    res[name] = np.array([vscore(kfold_oof(X, s), y).mean() for s in range(10)])
    print(f"5fold x10 {name:30s} {res[name].mean():.4f} ± {res[name].std():.4f}  min {res[name].min():.4f} max {res[name].max():.4f}", flush=True)
for name in sets:
    if name != "ref":
        d = res[name] - res["ref"]
        print(f"  paired vs ref {name:30s} mean {d.mean():+.4f}  wins {int((d > 0).sum())}/10")

print("\n== LOOO paired per-object ==")
X0 = design(df, ref, inter="gi_env_ml", knots=(50, 300))
p0 = nested_looo(X0, y, ids, fit_irls, IRLS_GRID)
v0 = pd.Series(vscore(p0, y)).groupby(ids).mean()
for name, ex in sets.items():
    if name == "ref":
        continue
    X = design(df, ref, inter="gi_env_ml", knots=(50, 300), extra=df[ex].to_numpy(float))
    p = nested_looo(X, y, ids, fit_irls, IRLS_GRID)
    v1 = pd.Series(vscore(p, y)).groupby(ids).mean()
    d = v1 - v0
    print(f"{name:30s} V={vscore(p, y).mean():.4f} obj gain>0.02: {int((d > 0.02).sum())} loss<-0.02: {int((d < -0.02).sum())}  biggest: {d.abs().sort_values(ascending=False).head(4).round(3).to_dict()}")
