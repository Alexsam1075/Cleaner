import math

import numpy as np
import pandas as pd
import pytest

from evaluation.volume_parts_cv import fit_calibrator
from tests.synth_scene import CAMERA, PLANE, render
from tests.test_volume_cv import _rows
from volume import VolumeEstimator
from volume.submission import rewrite_submission

BOX = {"kind": "box", "center_uv": (0.5, 0.0), "size": (0.10, 0.05, 0.04), "yaw": 0.3}
DENS = {("solid", "plastic"): 0.279, ("solid", "metal"): 0.82, ("bulk", "plastic"): 0.571}


@pytest.fixture(scope="module")
def scene():
    depth, masks = render(CAMERA, PLANE, [BOX])
    return depth, masks[0]


@pytest.fixture(scope="module")
def est():
    return VolumeEstimator(fit_calibrator(_rows()), DENS, CAMERA)


def test_estimate_returns_volume_and_mass(est, scene):
    depth, mask = scene
    r = est.estimate(depth, mask, PLANE, "solid", "plastic")
    assert r.volume_ml > 0
    assert r.mass_g == pytest.approx(r.volume_ml * 0.279)
    assert "parts_ml" in r.geometry


def test_one_hot_probs_equal_hard_material(est, scene):
    depth, mask = scene
    hard = est.estimate(depth, mask, PLANE, "solid", "metal")
    soft = est.estimate(depth, mask, PLANE, "solid", material_probs={"metal": 1.0})
    assert soft.volume_ml == pytest.approx(hard.volume_ml)


def test_mixed_probs_give_log_mean(est, scene):
    depth, mask = scene
    a = est.estimate(depth, mask, PLANE, "solid", "metal").volume_ml
    b = est.estimate(depth, mask, PLANE, "solid", "plastic").volume_ml
    mix = est.estimate(depth, mask, PLANE, "solid", material_probs={"metal": 0.5, "plastic": 0.5})
    assert mix.volume_ml == pytest.approx(math.sqrt(a * b))


def test_unknown_material_rejected(est, scene):
    depth, mask = scene
    with pytest.raises(ValueError):
        est.estimate(depth, mask, PLANE, "solid", "unobtainium")


def test_rewrite_submission_swaps_only_volume_and_mass(est, scene):
    depth, mask = scene
    df = pd.DataFrame({"image_id": ["r_a", "r_b"],
                       "prediction": ["10.0 20.0 30.0 40.0 0.90000 solid plastic 123.00 45.00 1", "none"]})
    frames = {"r_a": (depth, PLANE)}
    out = rewrite_submission(df, est, lambda iid: frames[iid], lambda iid, box: mask)
    tok = out.loc[0, "prediction"].split()
    assert tok[:7] == ["10.0", "20.0", "30.0", "40.0", "0.90000", "solid", "plastic"] and tok[9] == "1"
    assert float(tok[7]) != 123.0 and float(tok[8]) == pytest.approx(float(tok[7]) * 0.279, rel=1e-3)
    assert out.loc[1, "prediction"] == "none"
