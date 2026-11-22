"""Reproducible nested-LOOO evaluation of VolumeModel on the 40 real-train objects (GT masks).

Reports V overall / by volume bin / by form / by material for:
  (a) true class, per frame;  (a') true class, series median (consecutive frames of one object, gap<=3);
  (b) 8% wrong input material (per frame and series median);
  plus the base4-only variant (what the model falls back to without gi_env_ml) and the previous reference
  (evaluation.volume_parts_cv ridge) under the identical protocol.
Usage: ~/venv/bin/python volume_cv.py  [--out volume_cv_report.json]
"""
import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, "/home/cleaner/Cleaner")
from predict_volume import MATS, VolumeModel  # noqa: E402

L3 = math.log(3.0)
BINS = [(0, 50), (50, 200), (200, 1000), (1000, 1e9)]


def vscore(pred_ml, gt_ml):
    p, g = np.asarray(pred_ml, float), np.asarray(gt_ml, float)
    return np.where(p > 0, np.maximum(0.0, 1.0 - np.abs(np.log(np.maximum(p, 1e-9) / g)) / L3), 0.0)


def series_ids(rows, gap=3):
    fnum = rows["image_id"].str[2:].astype(int).values
    ids = rows["object_id"].values
    sid = np.zeros(len(rows), int)
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


def series_median(pred, sid):
    out = np.asarray(pred, float).copy()
    for s in set(sid):
        out[sid == s] = VolumeModel.aggregate_series(out[sid == s])
    return out


def breakdown(rows, pred):
    v = vscore(pred, rows["volume_ml"])
    g = rows["volume_ml"].values
    out = {"V": float(v.mean()),
           "by_bin": {f"{lo:g}-{hi:g}": float(v[(g >= lo) & (g < hi)].mean()) for lo, hi in BINS},
           "by_form": {f: float(v[rows["form"].values == f].mean()) for f in ("bulk", "solid")},
           "by_material": {m: float(v[rows["material"].values == m].mean()) for m in MATS if (rows["material"].values == m).any()},
           "per_object_mean": float(pd.Series(v).groupby(rows["object_id"].values).mean().mean())}
    return out


def fmt(name, b):
    bins = " ".join(f"{k}:{v:.3f}" for k, v in b["by_bin"].items())
    forms = " ".join(f"{k}:{v:.3f}" for k, v in b["by_form"].items())
    return f"{name:46s} V={b['V']:.4f} | {bins} | {forms} | obj-mean {b['per_object_mean']:.3f}"


def looo(rows, noisy_mat, variant_force=None):
    ids = rows["object_id"].values
    p, pn = np.zeros(len(rows)), np.zeros(len(rows))
    for o in sorted(set(ids)):
        te = ids == o
        vm = VolumeModel().fit(rows=rows[~te])
        if variant_force:
            vm.models = {variant_force: vm.models[variant_force]}
        for i in np.where(te)[0]:
            r = rows.iloc[i].to_dict()
            if variant_force == "base4":
                r.pop("gi_env_ml", None)
            p[i] = vm.predict(r, r["form"], r["material"])
            pn[i] = vm.predict(r, r["form"], noisy_mat[i])
    return p, pn


def looo_old_ref(rows):
    from evaluation.volume_parts_cv import apply_calibrator, fit_calibrator
    ids = rows["object_id"].values
    p = np.zeros(len(rows))
    for o in sorted(set(ids)):
        te = ids == o
        cal = fit_calibrator(rows[~te])
        p[te] = [apply_calibrator(cal, r, r["form"], r["material"]) for r in rows[te].to_dict("records")]
    return p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(HERE / "volume_cv_report.json"))
    args = ap.parse_args()
    rows = VolumeModel.training_rows()
    assert rows["object_id"].nunique() == 40 and len(rows) == 214
    rng = np.random.default_rng(5)
    flip = rng.random(len(rows)) < 0.08
    noisy = rows["material"].values.copy()
    noisy[flip] = [rng.choice([m for m in MATS if m != x]) for x in rows["material"].values[flip]]
    sid = series_ids(rows)
    rep = {}
    print("== nested LOOO over 40 objects, GT masks, real frames only ==")
    p, pn = looo(rows, noisy)
    for name, pred in (("VolumeModel (ref5) | frame, true class", p), ("VolumeModel (ref5) | frame, 8% wrong material", pn),
                       ("VolumeModel (ref5) | series median, true class", series_median(p, sid)),
                       ("VolumeModel (ref5) | series median, 8% wrong material", series_median(pn, sid))):
        rep[name] = breakdown(rows, pred)
        print(fmt(name, rep[name]), flush=True)
    pb, pbn = looo(rows, noisy, "base4")
    for name, pred in (("VolumeModel (base4 fallback) | frame, true class", pb), ("VolumeModel (base4 fallback) | frame, 8% wrong", pbn),
                       ("VolumeModel (base4 fallback) | series median", series_median(pb, sid))):
        rep[name] = breakdown(rows, pred)
        print(fmt(name, rep[name]), flush=True)
    po = looo_old_ref(rows)
    for name, pred in (("previous ridge (volume_parts_cv) | frame", po), ("previous ridge (volume_parts_cv) | series median", series_median(po, sid))):
        rep[name] = breakdown(rows, pred)
        print(fmt(name, rep[name]), flush=True)
    print("materials:", {k: round(v, 3) for k, v in rep["VolumeModel (ref5) | frame, true class"]["by_material"].items()})
    rows.assign(pred_ref5=p, pred_ref5_noisy=pn, pred_base4=pb, pred_old=po, series_id=sid).to_csv(HERE / "volume_cv_oof.csv", index=False)
    Path(args.out).write_text(json.dumps(rep, indent=1))


if __name__ == "__main__":
    main()
