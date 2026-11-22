import numpy as np

from geometry.volume_parts import clean_cloud, parts_volume_ml, split_parts
from tests.synth_scene import CAMERA, PLANE, render


def _cloud(shapes):
    depth, masks = render(CAMERA, PLANE, shapes)
    mask = np.logical_or.reduce(masks)
    return clean_cloud(depth, mask, PLANE, CAMERA)


def test_lying_cylinder_volume():
    c = _cloud([{"kind": "cylinder_lying", "center_uv": (0.5, 0.0), "radius": 0.035, "length": 0.20, "yaw": 0.4}])
    v, fits = parts_volume_ml(c, "solid")
    assert abs(v / 769.7 - 1) < 0.15


def test_standing_cylinder_hidden_back():
    c = _cloud([{"kind": "cylinder_standing", "center_uv": (0.5, 0.0), "radius": 0.040, "length": 0.12}])
    v, fits = parts_volume_ml(c, "solid")
    assert abs(v / 603.2 - 1) < 0.15


def test_bottle_two_parts():
    c = _cloud([
        {"kind": "cylinder_lying", "center_uv": (0.5, 0.0), "radius": 0.040, "length": 0.18, "yaw": np.pi / 2},
        {"kind": "cylinder_lying", "center_uv": (0.5, 0.12), "radius": 0.013, "length": 0.06, "yaw": np.pi / 2},
    ])
    assert len(split_parts(c)) == 2


def test_dome_pile():
    c = _cloud([{"kind": "dome", "center_uv": (0.5, 0.0), "radius": 0.060, "height": 0.040}])
    v, fits = parts_volume_ml(c, "bulk")
    assert abs(v / 259.7 - 1) < 0.20
