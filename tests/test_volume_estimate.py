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
