import numpy as np

from geometry.volume_parts import estimate_volume
from tests.synth_scene import CAMERA, PLANE, render

BOX = {"kind": "box", "center_uv": (0.5, 0.0), "size": (0.10, 0.05, 0.04), "yaw": 0.3}
KEYS = ("columns_ml", "parts_ml", "hull_ml", "footprint_ml")


def test_no_valid_depth_fallback():
    depth, masks = render(CAMERA, PLANE, [BOX])
    depth0 = depth.copy()
    depth0[masks[0]] = 0
    r = estimate_volume(depth0, masks[0], PLANE, CAMERA, "solid", class_height_m=0.08)
    assert r["valid_frac"] == 0
    assert r["footprint_ml"] > 1 and r["parts_ml"] >= 1


def test_truncated_flag():
    edge_box = dict(BOX, center_uv=(0.5, 0.39), yaw=0.0)
    depth, masks = render(CAMERA, PLANE, [edge_box])
    m = masks[0]
    assert m.any() and (m[:, 0].any() or m[:, -1].any())
    assert estimate_volume(depth, m, PLANE, CAMERA, "solid")["truncated"] is True


def test_all_keys_positive():
    depth, masks = render(CAMERA, PLANE, [BOX])
    r = estimate_volume(depth, masks[0], PLANE, CAMERA, "solid")
    assert all(r[k] >= 1.0 for k in KEYS)
    assert abs(r["parts_ml"] / 200.0 - 1) < 0.2


def test_loose_mask_noisy_depth_parts_volume():
    import cv2
    depth, masks = render(CAMERA, PLANE, [BOX], noise_mm=3.0)
    tight = masks[0]
    loose = cv2.dilate(tight.astype(np.uint8), np.ones((21, 21), np.uint8)).astype(bool)
    v_tight = estimate_volume(depth, tight, PLANE, CAMERA, "solid")["parts_ml"]
    v_loose = estimate_volume(depth, loose, PLANE, CAMERA, "solid")["parts_ml"]
    assert v_loose < 1.2 * v_tight


def test_no_depth_footprint_corrected_for_height():
    from geometry.volume_parts import corrected_footprint_m2
    depth, masks = render(CAMERA, PLANE, [BOX])
    depth0 = depth.copy()
    depth0[masks[0]] = 0
    est = estimate_volume(depth0, masks[0], PLANE, CAMERA, "solid", class_height_m=0.04)
    assert abs(corrected_footprint_m2(est, 0.04) / 0.005 - 1) < 0.4
    assert abs(est["footprint_ml"] / 200.0 - 1) < 0.5
