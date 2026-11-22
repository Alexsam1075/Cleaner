"""Same 3 fold splits (30 train objects), GT masks vs YOLO masks, several calibrators, to separate
'training-set size variance' from 'mask effect'. Also averaged ridge+IRLS ensembles."""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path("/home/cleaner/Cleaner")
HERE = ROOT / "agents2/volume"
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT))
from evaluation.volume_parts_cv import apply_calibrator, fit_calibrator  # noqa: E402
from predict_volume import BASE, REF, VolumeModel, _apply, _design, _irls, _ridge  # noqa: E402
from volume_cv import breakdown, series_ids, series_median, vscore  # noqa: E402
from collections import Counter  # noqa: E402

FOLDS = {1: ROOT / "configs/object_id_split.json", 2: ROOT / "checkpoints/score83_cv_fold2_noval18/configs/object_id_split.json",
         3: ROOT / "checkpoints/score83_cv_fold3_noval18/configs/object_id_split.json"}
gt = VolumeModel.training_rows()
yo = pd.read_csv(HERE / "yolo_check_rows.csv")


def fit_variant(rows, cols, kind, alpha_grid=(1, 3, 10, 30, 100)):
    X = _design(rows, cols)
    y = np.log(rows.volume_ml.values)
    ids = rows.object_id.values
    cnt = Counter(ids)
    w = np.array([1 / cnt[o] for o in ids])
    uo = np.array(sorted(set(ids)))
    np.random.default_rng(0).shuffle(uo)
    fo = {o: k % 5 for k, o in enumerate(uo)}
    f = np.array([fo[o] for o in ids])
    grid = [(a, 0.2) for a in alpha_grid] if kind == "irls" else [(a, None) for a in alpha_grid]
    best = None
    for g in grid:
        p = np.zeros(len(y))
        for k in range(5):
            m = _irls(X[f != k], y[f != k], w[f != k], *g) if kind == "irls" else _ridge(X[f != k], y[f != k], w[f != k], g[0])
            p[f == k] = _apply(m, X[f == k])
        s = np.sum(w * np.maximum(0, 1 - np.abs(p - y) / np.log(3))) / w.sum()
        if best is None or s > best[0]:
            best = (s, g)
    m = _irls(X, y, w, *best[1]) if kind == "irls" else _ridge(X, y, w, best[1][0])
    return lambda rows_te: np.exp(_apply(m, _design(rows_te, cols)))


def evaluate(src, name):
    out = {}
    preds = {k: np.zeros(len(src)) for k in ("old", "ridge5", "irls5", "irls4", "ridge4")}
    for fold, sp in FOLDS.items():
        split = json.load(open(sp))
        tr_objs = set(split["train_object_ids"])
        val_frames = set(split["validation_real_frames"])
        tr = gt[gt.object_id.isin(tr_objs)]
        te = (src.image_id.isin(val_frames) & ~src.object_id.isin(tr_objs)).values
        cal = fit_calibrator(tr)
        preds["old"][te] = [apply_calibrator(cal, r, r["form"], r["material"]) for r in src[te].to_dict("records")]
        preds["ridge5"][te] = fit_variant(tr, REF, "ridge")(src[te])
        preds["irls5"][te] = fit_variant(tr, REF, "irls")(src[te])
        preds["irls4"][te] = fit_variant(tr, BASE, "irls")(src[te])
        preds["ridge4"][te] = fit_variant(tr, BASE, "ridge")(src[te])
    used = np.zeros(len(src), bool)
    for fold, sp in FOLDS.items():
        split = json.load(open(sp))
        used |= (src.image_id.isin(set(split["validation_real_frames"])) & ~src.object_id.isin(set(split["train_object_ids"]))).values
    s = src[used].reset_index(drop=True)
    sid = series_ids(s)
    preds = {k: v[used] for k, v in preds.items()}
    preds["mean(ridge5,irls5)"] = np.exp(0.5 * (np.log(preds["ridge5"]) + np.log(preds["irls5"])))
    preds["mean(all4)"] = np.exp(np.mean([np.log(preds[k]) for k in ("ridge5", "irls5", "irls4", "ridge4")], 0))
    print(f"== {name}: {len(s)} rows, {s.object_id.nunique()} objects ==")
    for k, p in preds.items():
        b = breakdown(s, p)
        bs = breakdown(s, series_median(p, sid))
        perfold = []
        for fold, sp in FOLDS.items():
            split = json.load(open(sp))
            m = s.image_id.isin(set(split["validation_real_frames"])).values
            perfold.append(vscore(p[m], s.volume_ml.values[m]).mean())
        print(f"  {k:20s} frame V={b['V']:.4f} series V={bs['V']:.4f}  bulk {b['by_form']['bulk']:.3f} solid {b['by_form']['solid']:.3f}  folds {[round(v, 3) for v in perfold]}")


evaluate(gt, "GT masks, fold splits")
evaluate(yo, "YOLO masks, fold splits")
