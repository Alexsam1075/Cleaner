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


def test_lidar_and_collision_with_wall():
    from polygon.world import Obstacle, lidar, robot_collides
    wall = Obstacle("wall", [], boxes=[(1.0, 0.0, 0.05, 1.0, 0.0)])
    r = lidar([wall], 0.0, 0.0, 0.0)
    assert abs(r[0] - 0.95) < 1e-3                       # straight ahead hits the wall face at x=0.95
    assert robot_collides([wall], 0.7, 0.0, 0.0) and not robot_collides([wall], 0.3, 0.0, 0.0)


def test_brain_turns_away_from_wall_ahead():
    from polygon.world import Obstacle, lidar
    wall = Obstacle("wall", [], boxes=[(0.8, 0.0, 0.05, 2.0, 0.0)])
    b = Brain()
    b.memory.append({"x": 2.0, "y": 0.3, "top": 0.05, "take": 1, "material": "plastic", "volume": 50.0,
                     "mass": 10.0, "score": 0.9, "hits": 3, "misses": 0, "seen": 1e12})
    v, w = b.act(0.0, 0.3, 0.0, 0.05, lidar([wall], 0.0, 0.3, 0.0))
    assert abs(w) > 0.8 and v < 0.15                    # target is behind the wall: turn, do not ram it


def test_detections_on_walls_are_ignored():
    b = Brain()
    b.blocked = lambda x, y: x > 1.0
    b.observe([_det(1.2, 0.0), _det(0.8, 0.0)], CAM)
    assert len(b.memory) == 1 and abs(b.memory[0]["x"] - 0.8) < 1e-6


def test_robot_gets_around_a_pillar_in_its_lane():
    import polygon.brain as br
    from polygon.world import Obstacle, lidar, robot_collides
    pillar = Obstacle("pillar", [], boxes=[(1.2, 0.0, 0.2, 0.2, 0.0)])
    b = Brain()
    b.memory.append({"x": 2.6, "y": 0.0, "top": 0.05, "take": 1, "material": "plastic", "volume": 50.0,
                     "mass": 10.0, "score": 0.9, "hits": 3, "misses": 0, "seen": 1e12})
    clock = [0.0]
    t0 = br.time.time
    br.time.time = lambda: clock[0]
    x, y, yaw, dt = 0.0, 0.0, 0.0, 0.05
    try:
        for _ in range(900):
            v, w = b.act(x, y, yaw, dt, lidar([pillar], x, y, yaw))
            nyaw = yaw + w * dt
            nx, ny = x + v * math.cos(nyaw) * dt, y + v * math.sin(nyaw) * dt
            if not robot_collides([pillar], nx, ny, nyaw):
                x, y, yaw = nx, ny, nyaw
            b.passed(x + 0.42 * math.cos(yaw), y + 0.42 * math.sin(yaw), x, y, yaw)
            clock[0] += dt
            if not b.memory:
                break
    finally:
        br.time.time = t0
    assert not b.memory                                 # reached the trash behind the pillar


def test_bump_backs_off_then_turns():
    import polygon.brain as br
    b = Brain()
    clock = [0.0]
    t0 = br.time.time
    br.time.time = lambda: clock[0]
    try:
        b.bump(-0.9)
        far = np.full(72, 4.0)
        v, w = b.act(0.0, 0.0, 0.0, 0.05, far)
        assert v < 0
        clock[0] = 1.0
        b.act(0.0, 0.0, 0.0, 0.05, far)
        clock[0] = 1.05
        v, w = b.act(0.0, 0.0, 0.0, 0.05, far)
        assert w > 0                                    # turns away from the side it was turning into
    finally:
        br.time.time = t0
