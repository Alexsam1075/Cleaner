import json

import numpy as np
import pandas as pd

from evaluation.volume_parts_cv import apply_calibrator, fit_calibrator, group_folds, volume_score

FORMS = ["bulk", "solid"]
MATERIALS = ["plastic", "metal", "glass", "paper", "wood", "mineral"]


def _rows(n_obj=20, per=5, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for k in range(n_obj):
        true = float(np.exp(rng.uniform(np.log(10), np.log(3000))))
        form, mat = FORMS[k % 2], MATERIALS[k % 6]
        for _ in range(per):
            g = true * float(np.exp(rng.normal(0.3, 0.2)))
            rows.append({"object_id": f"o{k:02d}", "form": form, "material": mat, "volume_ml": true,
                         "columns_ml": g, "parts_ml": g * 0.9, "hull_ml": g * 1.3, "footprint_ml": g * 1.6,
                         "valid_frac": 0.9, "truncated": False, "n_parts": 1, "p95_h_m": 0.04,
                         "footprint_m2": 0.004})
    return pd.DataFrame(rows)


def test_no_object_leak():
    ids = _rows()["object_id"].to_numpy()
    folds = group_folds(ids, n_folds=5, seed=0)
    assert len(folds) == 5
    for tr, va in folds:
        assert not set(ids[tr]) & set(ids[va])
    assert sorted(np.concatenate([va for _, va in folds])) == list(range(len(ids)))


def test_calibrator_roundtrip():
    rows = _rows()
    cal = fit_calibrator(rows)
    cal2 = json.loads(json.dumps(cal))
    r = rows.iloc[3].to_dict()
    assert apply_calibrator(cal, r, r["form"], r["material"]) == apply_calibrator(cal2, r, r["form"], r["material"])


def test_calibrator_beats_raw_on_biased_rows():
    rows = _rows()
    cal = fit_calibrator(rows)
    pred = [apply_calibrator(cal, r, r["form"], r["material"]) for r in rows.to_dict("records")]
    assert volume_score(pred, rows["volume_ml"]) > volume_score(rows["columns_ml"], rows["volume_ml"])


def test_volume_score_matches_metric():
    assert volume_score([100.0], [100.0]) == 1.0
    assert volume_score([300.0], [100.0]) == 0.0
    assert volume_score([0.0], [100.0]) == 0.0
