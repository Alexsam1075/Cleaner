import cv2
import numpy as np

from geometry.volume_parts import clean_cloud, columns_volume_ml, refine_plane
from tests.synth_scene import CAMERA, PLANE, render

BOX = {"kind": "box", "center_uv": (0.5, 0.0), "size": (0.10, 0.05, 0.04), "yaw": 0.3}
TINY = {"kind": "box", "center_uv": (0.5, 0.0), "size": (0.03, 0.02, 0.008), "yaw": 0.0}


def test_box_heights():
    depth, masks = render(CAMERA, PLANE, [BOX])
    c = clean_cloud(depth, masks[0], PLANE, CAMERA)
    assert abs(float(np.quantile(c.h, 0.98)) - 0.040) < 0.004


def test_tiny_object_survives():
    depth, masks = render(CAMERA, PLANE, [TINY])
    assert len(clean_cloud(depth, masks[0], PLANE, CAMERA).h) > 50


def test_plane_sign():
    depth, masks = render(CAMERA, PLANE, [BOX])
    c = clean_cloud(depth, masks[0], -PLANE, CAMERA)
    assert (c.h >= 0).all()
    assert abs(float(np.quantile(c.h, 0.98)) - 0.040) < 0.004


def test_loose_mask():
    depth, masks = render(CAMERA, PLANE, [BOX])
    tight = masks[0]
    loose = cv2.dilate(tight.astype(np.uint8), np.ones((21, 21), np.uint8)).astype(bool)
    v_tight = columns_volume_ml(clean_cloud(depth, tight, PLANE, CAMERA))
    v_loose = columns_volume_ml(clean_cloud(depth, loose, PLANE, CAMERA))
    assert v_loose < 1.2 * v_tight


def test_refine_plane_recovers_offset():
    depth, masks = render(CAMERA, PLANE, [BOX])
    shifted = PLANE.copy()
    shifted[3] += 0.005
    got = refine_plane(depth, masks[0], shifted, CAMERA)
    true = PLANE / np.linalg.norm(PLANE[:3])
    assert abs(float(got[3]) - float(true[3])) < 0.0015
