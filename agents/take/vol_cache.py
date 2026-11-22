"""Precompute nested calibrated volumes: V[(o, q)] = calibrated volume of the frames of object q with
the project calibrator fitted on all real objects except {o, q}. Both true material and the noisy
(material-classifier OOF argmax) material are stored. Runs once (~1600 calibrator fits)."""
from __future__ import annotations

import pickle
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, "/home/cleaner/Cleaner/agents/take")
from take_data import CACHE, MATS, SHARED, build_table
from evaluation.volume_parts_cv import apply_calibrator, fit_calibrator

PARTS_COLS = ["columns_ml", "parts_ml", "hull_ml", "footprint_ml", "valid_frac", "truncated", "n_parts", "p95_h_m",
              "footprint_m2", "fp_area_0", "fp_area_20", "fp_area_40", "fp_area_80", "fp_area_160", "min_h_m"]


def main():
    df = build_table()
    real = df[df.real].reset_index(drop=True)
    fused = np.load(SHARED / "fused_generic_syn.npy")  # material OOF probs, ~90% acc on real frames
    real["mat_noisy"] = [MATS[i] for i in fused[real.idx.values].argmax(1)]
    print("noisy material acc:", (real.mat_noisy == real.mat).mean())
    rows = real[PARTS_COLS].copy()
    rows["object_id"], rows["form"], rows["material"], rows["volume_ml"] = real.oid.values, real.form.values, real.mat.values, real.vol.values
    objs = sorted(real.oid.unique())
    V = {}
    t = time.time()
    for o in objs:
        for q in objs:
            tr = rows[~rows.object_id.isin({o, q})]
            cal = fit_calibrator(tr)
            te_idx = np.where(real.oid.values == q)[0]
            recs = rows.iloc[te_idx].to_dict("records")
            V[(o, q)] = {
                "idx": te_idx,
                "true": np.array([apply_calibrator(cal, r, r["form"], r["material"]) for r in recs]),
                "noisy": np.array([apply_calibrator(cal, r, r["form"], m) for r, m in zip(recs, real.mat_noisy.values[te_idx])]),
                "alpha": cal["alpha"],
            }
        print(o, f"{time.time()-t:.0f}s", flush=True)
    pickle.dump({"V": V, "mat_noisy": real.mat_noisy.values, "objs": objs}, open(CACHE / "vol_cache.pkl", "wb"))


if __name__ == "__main__":
    main()
