"""Exp14: what can be squeezed from the pipeline geometry dict alone (estimate_volume keys), since
VolumeModel.predict only receives that dict: fp_area ratios (silhouette-elongation proxy), p95 height, valid_frac,
n_parts; and ensembles of main-candidate variants. LOOO frame + series median + 5-fold x3."""
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
base = ["columns_ml", "parts_ml", "hull_ml", "footprint_ml"]
df["fp_ratio40"] = np.log(np.maximum(df.fp_area_0, 1e-5) / np.maximum(df.fp_area_40, 1e-5))
df["fp_ratio160"] = np.log(np.maximum(df.fp_area_0, 1e-5) / np.maximum(df.fp_area_160, 1e-5))
df["log_h"] = np.log(np.maximum(df.p95_h_m, 0.004))
df["log_fp"] = np.log(np.maximum(df.footprint_m2, 1e-5))
df["trunc"] = df.truncated.astype(float)
df["nparts"] = np.minimum(df.n_parts, 4).astype(float)
df["geomean_ml"] = np.exp(np.mean([np.log(np.maximum(df[c].values, 1)) for c in base], 0))


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


def run(name, cols, inter, extra=None, kf=True):
    X = design(df, cols, inter=inter, knots=(50, 300), extra=df[extra].to_numpy(float) if extra else None)
    p = nested_looo(X, y, ids, fit_irls, IRLS_GRID)
    v = report(f"{name} | frame", p, df)
    report(f"{name} | series", smed(p), df)
    if kf:
        vs = [vscore(kfold_oof(X, s), y).mean() for s in range(3)]
        print(f"   5fold x3: {np.mean(vs):.4f} ± {np.std(vs):.4f}", flush=True)
    return p


P = {}
P["b4 parts"] = run("base4 main=parts", base, "parts_ml")
P["b4 cols"] = run("base4 main=columns", base, "columns_ml")
P["b4 geomean"] = run("base4+geomean main=geomean", base + ["geomean_ml"], "geomean_ml")
for ex in (["fp_ratio40"], ["fp_ratio160"], ["fp_ratio40", "log_h"], ["valid_frac", "trunc", "nparts"], ["fp_ratio40", "valid_frac"]):
    P["b4 cols+" + "+".join(ex)] = run("base4 main=columns + " + "+".join(ex), base, "columns_ml", ex)
ens = np.mean([P["b4 parts"], P["b4 cols"], P["b4 geomean"]], 0)
report("ens(parts, cols, geomean) | frame", ens, df)
report("ens(parts, cols, geomean) | series", smed(ens), df)
np.save(HERE + "/cache_exp14.npy", P, allow_pickle=True)
