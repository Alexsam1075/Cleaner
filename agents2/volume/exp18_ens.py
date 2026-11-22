"""Exp18: honest 2-model ensemble (mine r5 env IRLS + old design IRLS), all 3 protocols, clean and noisy material."""
import json
import math
import sys
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path("/home/cleaner/Cleaner")
HERE = ROOT / "agents2/volume"
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT))
from exp17_design_protocols import D_mine, D_old, EST_KEYS, FOLDS, R5, B4, V, fit, foldsplits, ids, kfold, looo, rows, sid, w_all, y  # noqa: E402
from predict_volume import MATS, _apply  # noqa: E402
from volume_cv import series_median, vscore  # noqa: E402

rng = np.random.default_rng(5)
noisy = rows.copy()
flip = rng.random(len(rows)) < 0.08
noisy.loc[flip, "material"] = [rng.choice([m for m in MATS if m != x]) for x in rows.material[flip]]

designs = {"r5env": (D_mine(rows, R5, "gi_env_ml"), D_mine(noisy, R5, "gi_env_ml")),
           "old": (D_old(rows), D_old(noisy)),
           "b4pc": (D_mine(rows, B4, "parts_ml", knots_cols=["parts_ml", "columns_ml"]), D_mine(noisy, B4, "parts_ml", knots_cols=["parts_ml", "columns_ml"]))}


def looo2(X, Xn):
    p, pn = np.zeros(len(y)), np.zeros(len(y))
    for o in sorted(set(ids)):
        tr = ids != o
        m = fit(X[tr], y[tr], w_all[tr], ids[tr], "irls")
        p[~tr] = _apply(m, X[~tr])
        pn[~tr] = _apply(m, Xn[~tr])
    return p, pn


def folds2(X, Xn):
    p, pn = np.full(len(y), np.nan), np.full(len(y), np.nan)
    for fold, sp in FOLDS.items():
        split = json.load(open(sp))
        tr = rows.object_id.isin(set(split["train_object_ids"])).values
        te = rows.image_id.isin(set(split["validation_real_frames"])).values & ~tr
        m = fit(X[tr], y[tr], w_all[tr], ids[tr], "irls")
        p[te] = _apply(m, X[te])
        pn[te] = _apply(m, Xn[te])
    return p, pn


res = {}
for k, (X, Xn) in designs.items():
    res[k] = {"looo": looo2(X, Xn), "folds": folds2(X, Xn), "kf": [kfold(X, "irls", s) for s in range(10)]}
    print("done", k, flush=True)


def show(name, L, F, K):
    p, pn = L
    pf, pfn = F
    m = ~np.isnan(pf)
    g = rows.volume_ml.values
    print(f"{name:28s} LOOO {V(p):.4f} (noisy {V(pn):.4f}) series {vscore(series_median(np.exp(p), sid), g).mean():.4f} (noisy {vscore(series_median(np.exp(pn), sid), g).mean():.4f})"
          f" | folds {V(pf, m):.4f} (noisy {V(pfn, m):.4f}) series {vscore(series_median(np.exp(pf[m]), sid[m]), g[m]).mean():.4f}"
          f" | 5fx10 {np.mean([V(k) for k in K]):.4f}±{np.std([V(k) for k in K]):.4f}", flush=True)


for k in designs:
    show(k, res[k]["looo"], res[k]["folds"], res[k]["kf"])
for combo in (("r5env", "old"), ("r5env", "b4pc"), ("r5env", "old", "b4pc"), ("old", "b4pc")):
    L = tuple(np.mean([res[k]["looo"][i] for k in combo], 0) for i in range(2))
    F = tuple(np.mean([res[k]["folds"][i] for k in combo], 0) for i in range(2))
    K = [np.mean([res[k]["kf"][s] for k in combo], 0) for s in range(10)]
    show("ens " + "+".join(combo), L, F, K)
