"""Shared data/eval helpers for agents2/volume. Real-train frames only, GT masks, cached geometry."""
import collections
import math
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, "/home/cleaner/Cleaner")
ROOT = "/home/cleaner/Cleaner"
HERE = ROOT + "/agents2/volume"
L3 = math.log(3.0)
MATS = ["glass", "metal", "mineral", "paper", "plastic", "wood"]
BINS = [(0, 50), (50, 200), (200, 1000), (1000, 1e9)]


def load_rows() -> pd.DataFrame:
    r = pd.read_csv(ROOT + "/reports/volume_parts_gt_rows.csv")
    f = pd.read_csv(ROOT + "/agents/volume/cache/feat_real.csv")
    f2 = pd.read_csv(ROOT + "/agents/volume/cache/feat2_real.csv")
    f3 = pd.read_csv(ROOT + "/agents/volume/cache/feat3_real.csv")
    a = pd.read_csv(ROOT + "/audit/cache/algo_volume_rows.csv")
    assert (r.image_id.values == f.frame.values).all() and (r.image_id.values == a.frame.values).all()
    f2 = f2.rename(columns={c: c + "_ml" for c in f2.columns if c.endswith(("colmax", "colmean"))})
    parts = [r, f.drop(columns=[c for c in f.columns if c in r.columns]),
             f2.drop(columns=["split", "frame", "object_id"]), f3.drop(columns=["split", "frame", "object_id"]),
             a[["gap", "fill", "sil_W", "sil_H", "L", "W", "H", "colvol", "foot"]].rename(columns={"L": "aL", "W": "aW", "H": "aH"})]
    df = pd.concat(parts, axis=1).copy()
    return df


def vscore(pred_log, y_log):
    return np.maximum(0.0, 1.0 - np.abs(np.asarray(pred_log) - np.asarray(y_log)) / L3)


def w_of(ids):
    c = collections.Counter(ids)
    return np.array([1.0 / c[o] for o in ids])


def inner_folds(ids, n=5, seed=0):
    ut = np.array(sorted(set(ids)))
    np.random.default_rng(seed).shuffle(ut)
    fo = {u: k % n for k, u in enumerate(ut)}
    return np.array([fo[u] for u in ids])


def report(name, pred_log, df, extra=True):
    y = np.log(df.volume_ml.values)
    v = vscore(pred_log, y)
    g = df.volume_ml.values
    bins = [v[(g >= lo) & (g < hi)].mean() for lo, hi in BINS]
    s = f"{name:52s} V={v.mean():.4f} bulk {v[df.form.values == 'bulk'].mean():.3f} solid {v[df.form.values == 'solid'].mean():.3f} | " + " ".join(f"{b:.3f}" for b in bins)
    if extra:
        # per-object mean then averaged (robust to frame-count imbalance)
        po = pd.Series(v).groupby(df.object_id.values).mean()
        s += f" | obj-mean {po.mean():.3f} sd {po.std():.3f}"
    print(s, flush=True)
    return v.mean()
