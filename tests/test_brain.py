import math

import numpy as np

from polygon.brain import Brain
from polygon.fastrt import FastCam


def _det(x, y, take=1):
    return {"world": np.array([x, y, 0.02]), "top": 0.05, "take": take, "material": "plastic", "volume": 100.0,
            "mass": 30.0, "score": 0.9}


CAM = FastCam.look_at((0.33, 0, 0.25), (1.2, 0, -0.25), W=1280, H=720, fov_deg=70)


def test_explore_drives_forward_not_spinning():
    b = Brain()
    x, y, yaw = -0.3, -1.2, 0.0
    for _ in range(200):
        v, w = b.act(x, y, yaw)
        yaw += w * 0.05
        x += v * math.cos(yaw) * 0.05
        y += v * math.sin(yaw) * 0.05
    assert b.mode == "explore"
    assert math.hypot(x + 0.3, y + 1.2) > 1.5          # travelled, did not circle in place


def test_turns_toward_target():
    b = Brain()
    b.observe([_det(1.0, 0.5)], CAM)
    v, w = b.act(0.0, 0.0, 0.0)
    assert b.mode == "target" and w > 0 and v > 0


def test_repeated_sightings_merge_and_false_positive_fades():
    b = Brain()
    b.observe([_det(0.9, 0.0)], CAM)
    b.observe([_det(0.95, 0.02)], CAM)
    assert len(b.memory) == 1 and b.memory[0]["hits"] == 2
    b.observe([], CAM)
    b.observe([], CAM)
    assert len(b.memory) == 0


def test_avoid_object_in_lane_pushes_aside():
    b = Brain()
    b.observe([_det(0.6, 0.05, take=0)], CAM)
    v, w = b.act(0.0, 0.0, 0.0)
    assert abs(w) > 0.3
