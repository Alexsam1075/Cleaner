"""Exp5: variance-reduction (bagging over objects, multi-seed inner CV), empirical-Bayes material offsets,
and a repeated 5-fold x 3 seeds check of the few promising feature sets (to separate signal from LOOO noise)."""
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


def fit_bag(X, yt, w, g, n_bags=15, seed=0):
    a, fl = g
    uo = np.array(sorted(set(_ids_tr)))  # set by caller via global
    rng = np.random.default_rng(seed)
    ms = []
    for _ in range(n_bags):
        pick = rng.choice(uo, len(uo), replace=True)
        cnt = pd.Series(pick).value_counts()
        wb = w * np.array([cnt.get(o, 0) for o in _ids_tr])
        keep = wb > 0
        ms.append(irls(X[keep], yt[keep], wb[keep], a, floor=fl))
    return lambda Xn: np.mean([m(Xn) for m in ms], 0)


def nested_bag(X, grid, n_bags=15):
    global _ids_tr
    pred = np.zeros(len(y))
    w = w_of(ids)
    for o in sorted(set(ids)):
        tr = ids != o
        Xt, yt, it, wt = X[tr], y[tr], ids[tr], w[tr]
        f = inner_folds(it, 5, 0)
        best = None
        for g in grid:
            p = np.zeros(len(yt))
            for k in range(5):
                p[f == k] = fit_irls(Xt[f != k], yt[f != k], wt[f != k], g)(Xt[f == k])
            s = np.sum(wt * vscore(p, yt)) / wt.sum()
            if best is None or s > best[0]:
                best = (s, g)
        _ids_tr = it
        pred[~tr] = fit_bag(Xt, yt, wt, best[1], n_bags)(X[~tr])
    return pred


p_ref = nested_looo(X_ref, y, ids, fit_irls, IRLS_GRID)
report("ref IRLS", p_ref, df)
p_bag = nested_bag(X_ref, IRLS_GRID, 15)
report("ref IRLS bagged x15", p_bag, df)
# multi-seed inner CV: average the models chosen under 3 inner-fold seeds
ps = [nested_looo(X_ref, y, ids, fit_irls, IRLS_GRID, seed=s) for s in (0, 1, 2)]
report("ref IRLS, mean over 3 inner seeds", np.mean(ps, 0), df)


# ---- empirical Bayes material offsets on top of a material-free model
def design_nomat(d, cols, inter, knots):
    X = np.stack([np.log(np.maximum(d[c].to_numpy(float), 1.0)) for c in cols], 1)
    form = (d.form == "bulk").to_numpy().astype(float)[:, None]
    main = X[:, cols.index(inter)][:, None]
    parts = [X, form, main * form] + [np.maximum(0, main - math.log(t)) for t in knots]
    return np.hstack(parts)


import math  # noqa

X_nm = design_nomat(df, ref, "gi_env_ml", (50, 300))
mats = df.material.values
forms = df.form.values


def fit_eb(X, yt, w, g, mats_tr, forms_tr):
    a, fl, k = g
    m = irls(X, yt, w, a, floor=fl)
    r = yt - m(X)
    off = {}
    for fm in ("bulk", "solid"):
        for mt in MATS:
            s = (mats_tr == mt) & (forms_tr == fm)
            # n_obj-weighted: weights already 1/frames, so sum(w) = n_objects in class
            off[(fm, mt)] = float(np.sum(w[s] * r[s]) / (np.sum(w[s]) + k)) if s.any() else 0.0
    return lambda Xn, fm_n, mt_n: m(Xn) + np.array([off.get((f_, m_), 0.0) for f_, m_ in zip(fm_n, mt_n)])


def nested_eb(grid):
    pred = np.zeros(len(y))
    w = w_of(ids)
    for o in sorted(set(ids)):
        tr = ids != o
        Xt, yt, it, wt = X_nm[tr], y[tr], ids[tr], w[tr]
        f = inner_folds(it, 5, 0)
        best = None
        for g in grid:
            p = np.zeros(len(yt))
            for kf in range(5):
                a_, b_ = f != kf, f == kf
                p[b_] = fit_eb(Xt[a_], yt[a_], wt[a_], g, mats[tr][a_], forms[tr][a_])(Xt[b_], forms[tr][b_], mats[tr][b_])
            s = np.sum(wt * vscore(p, yt)) / wt.sum()
            if best is None or s > best[0]:
                best = (s, g)
        pred[~tr] = fit_eb(Xt, yt, wt, best[1], mats[tr], forms[tr])(X_nm[~tr], forms[~tr], mats[~tr])
    return pred


p_eb = nested_eb([(a, 0.2, k) for a in (3, 10, 30) for k in (0.5, 1, 2, 4, 1e9)])
report("no-material IRLS + EB class offsets", p_eb, df)
report("mean(ref IRLS, EB)", 0.5 * (p_ref + p_eb), df)

# ---- repeated grouped 5-fold x 3 seeds for candidate sets
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


print("\n== 5-fold grouped x 3 seeds (V mean ± sd over seeds) ==")
sets = {"ref": None, "ref+fill": ["fill"], "ref+fill+hole_edge": ["hole_edge_frac", "fill"], "ref+gap+fill": ["gap", "fill"],
        "ref+cos_tilt": ["cos_tilt"], "ref+valid_frac": ["valid_frac"]}
for name, ex in sets.items():
    X = design(df, ref, inter="gi_env_ml", knots=(50, 300), extra=df[ex].to_numpy(float) if ex else None)
    vs = [vscore(kfold_oof(X, s), y).mean() for s in range(3)]
    print(f"  {name:24s} {np.mean(vs):.4f} ± {np.std(vs):.4f}   {[round(v, 4) for v in vs]}", flush=True)
