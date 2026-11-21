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


def _drive(b, x, y, yaw, steps, dt=0.05):
    import polygon.brain as br
    t0 = br.time.time
    clock = [0.0]
    br.time.time = lambda: clock[0]
    try:
        for _ in range(steps):
            v, w = b.act(x, y, yaw, dt)
            yaw += w * dt
            x += v * math.cos(yaw) * dt
            y += v * math.sin(yaw) * dt
            b.passed(x + 0.42 * math.cos(yaw), y + 0.42 * math.sin(yaw), x, y, yaw)
            clock[0] += dt
    finally:
        br.time.time = t0
    return x, y, yaw


def test_target_beside_the_body_does_not_trap_the_robot():
    b = Brain()
    b.memory.append({"x": 0.1, "y": 0.36, "top": 0.05, "take": 1, "material": "plastic", "volume": 50.0,
                     "mass": 10.0, "score": 0.9, "hits": 3, "misses": 0, "seen": 0.0})
    _drive(b, 0.0, 0.0, 0.0, 600)
    assert b.mode == "explore"
    assert b.wp >= 2                                    # swept lanes instead of spinning on the spot


def test_unreachable_target_is_abandoned():
    b = Brain()
    b.memory.append({"x": 0.0, "y": 0.0, "top": 0.05, "take": 1, "material": "plastic", "volume": 50.0,
                     "mass": 10.0, "score": 0.9, "hits": 3, "misses": 0, "seen": 0.0})
    _drive(b, 5.0, 5.0, 0.0, 10)                         # pursuit starts
    b.memory[0]["x"] = 100.0                             # target can never be reached
    b.memory[0]["y"] = 100.0
    _drive(b, 5.0, 5.0, math.pi, 400)
    assert all(m["x"] != 100.0 for m in b.memory)
