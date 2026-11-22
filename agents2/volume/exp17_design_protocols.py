"""Exp17: the previous fit_calibrator design beats mine on the 3 pipeline fold splits (30 train objects) although
it loses in LOOO (39 train objects). Compare design elements under 3 protocols: LOOO, pipeline fold splits,
5-fold x 10 seeds. Designs: old (volume_parts_cv._features), mine, hybrids."""
import json
import math
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path("/home/cleaner/Cleaner")
HERE = ROOT / "agents2/volume"
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT))
from evaluation.volume_parts_cv import EST_KEYS, _class_heights, _features as old_features  # noqa: E402
from predict_volume import VolumeModel, _apply, _irls, _ridge  # noqa: E402
from volume_cv import series_ids, series_median, vscore  # noqa: E402

MATS = ["glass", "metal", "mineral", "paper", "plastic", "wood"]
rows = VolumeModel.training_rows()
y = np.log(rows.volume_ml.values)
ids = rows.object_id.values
cnt = Counter(ids)
w_all = np.array([1 / cnt[o] for o in ids])
sid = series_ids(rows)
FOLDS = {1: ROOT / "configs/object_id_split.json", 2: ROOT / "checkpoints/score83_cv_fold2_noval18/configs/object_id_split.json",
         3: ROOT / "checkpoints/score83_cv_fold3_noval18/configs/object_id_split.json"}
heights = _class_heights(rows)


def D_old(d, keys=EST_KEYS, knots=(50, 300)):
    return np.vstack([old_features(r, r["form"], r["material"], heights, list(keys), list(knots)) for r in d.to_dict("records")])


def D_mine(d, cols, inter, knots=(50, 300), extra_old=False, inter_form=True, knots_cols=None, inter_mat=True):
    X = np.stack([np.log(np.maximum(d[c].to_numpy(float), 1.0)) for c in cols], 1)
    form = (d.form == "bulk").to_numpy().astype(float)[:, None]
    mat = np.stack([(d.material == m).to_numpy() for m in MATS], 1).astype(float)
    main = X[:, cols.index(inter)][:, None]
    parts = [X, form, mat, form * mat]
    if inter_mat:
        parts.append(main * mat)
    if inter_form:
        parts.append(main * form)
    for kc in (knots_cols or [inter]):
        mc = X[:, cols.index(kc)]
        parts += [np.maximum(0, mc - math.log(t))[:, None] for t in knots]
    if extra_old:
        parts.append(np.c_[d.valid_frac.values.astype(float), d.truncated.values.astype(float), np.minimum(d.n_parts.values, 4).astype(float), (d.n_parts.values == 0).astype(float)])
    return np.hstack(parts)


B4 = ["columns_ml", "parts_ml", "hull_ml", "footprint_ml"]
R5 = B4 + ["gi_env_ml"]
DESIGNS = {
    "old (parts main, knots parts+cols, +valid/trunc/nparts)": D_old(rows),
    "old + gi_env key": D_old(rows, EST_KEYS + ["gi_env_ml"]),
    "mine b4 parts": D_mine(rows, B4, "parts_ml"),
    "mine b4 parts + old extras": D_mine(rows, B4, "parts_ml", extra_old=True),
    "mine b4 parts, knots parts+cols": D_mine(rows, B4, "parts_ml", knots_cols=["parts_ml", "columns_ml"]),
    "mine b4 parts, no main*form": D_mine(rows, B4, "parts_ml", inter_form=False),
    "mine b4 parts, no main*form, +extras, knots p+c": D_mine(rows, B4, "parts_ml", inter_form=False, extra_old=True, knots_cols=["parts_ml", "columns_ml"]),
    "mine r5 env": D_mine(rows, R5, "gi_env_ml"),
    "mine r5 env + old extras": D_mine(rows, R5, "gi_env_ml", extra_old=True),
    "mine r5 env, no main*form, +extras, knots env+cols": D_mine(rows, R5, "gi_env_ml", inter_form=False, extra_old=True, knots_cols=["gi_env_ml", "columns_ml"]),
    "mine r5 parts main": D_mine(rows, R5, "parts_ml"),
    "mine r5 parts main, +extras, knots p+c": D_mine(rows, R5, "parts_ml", extra_old=True, knots_cols=["parts_ml", "columns_ml"]),
}


def fit(X, yt, w, it, kind, seed=0):
    uo = np.array(sorted(set(it)))
    np.random.default_rng(seed).shuffle(uo)
    fo = {o: k % 5 for k, o in enumerate(uo)}
    f = np.array([fo[o] for o in it])
    grid = [(a, fl) for a in (1, 3, 10, 30, 100) for fl in (0.1, 0.2, 0.3)] if kind == "irls" else [(a,) for a in (0.3, 1, 3, 10, 30, 100)]
    best = None
    for g in grid:
        p = np.zeros(len(yt))
        for k in range(5):
            m = _irls(X[f != k], yt[f != k], w[f != k], *g) if kind == "irls" else _ridge(X[f != k], yt[f != k], w[f != k], g[0])
            p[f == k] = _apply(m, X[f == k])
        s = np.sum(w * np.maximum(0, 1 - np.abs(p - yt) / math.log(3))) / w.sum()
        if best is None or s > best[0]:
            best = (s, g)
    return _irls(X, yt, w, *best[1]) if kind == "irls" else _ridge(X, yt, w, best[1][0])


def looo(X, kind):
    p = np.zeros(len(y))
    for o in sorted(set(ids)):
        tr = ids != o
        p[~tr] = _apply(fit(X[tr], y[tr], w_all[tr], ids[tr], kind), X[~tr])
    return p


def foldsplits(X, kind):
    p = np.full(len(y), np.nan)
    for fold, sp in FOLDS.items():
        split = json.load(open(sp))
        tr = rows.object_id.isin(set(split["train_object_ids"])).values
        te = rows.image_id.isin(set(split["validation_real_frames"])).values & ~tr
        p[te] = _apply(fit(X[tr], y[tr], w_all[tr], ids[tr], kind), X[te])
    return p


def kfold(X, kind, seed):
    p = np.zeros(len(y))
    uo = np.array(sorted(set(ids)))
    np.random.default_rng(100 + seed).shuffle(uo)
    fo = {o: k % 5 for k, o in enumerate(uo)}
    f = np.array([fo[o] for o in ids])
    for k in range(5):
        tr = f != k
        p[~tr] = _apply(fit(X[tr], y[tr], w_all[tr], ids[tr], kind), X[~tr])
    return p


def V(p, m=None):
    m = np.ones(len(y), bool) if m is None else m
    return vscore(np.exp(p[m]), rows.volume_ml.values[m]).mean()


if __name__ == "__main__":
  print(f"{'design':58s} {'fit':5s} | LOOO frame  series | folds frame  series | 5fx10 mean±sd")
  for name, X in DESIGNS.items():
    for kind in ("ridge", "irls"):
        pl = looo(X, kind)
        pf = foldsplits(X, kind)
        m = ~np.isnan(pf)
        ks = [V(kfold(X, kind, s)) for s in range(10)]
        print(f"{name:58s} {kind:5s} | {V(pl):.4f} {V(np.log(series_median(np.exp(pl), sid))):.4f} | {V(pf, m):.4f} "
              f"{vscore(series_median(np.exp(pf[m]), sid[m]), rows.volume_ml.values[m]).mean():.4f} | {np.mean(ks):.4f}±{np.std(ks):.4f}", flush=True)
