"""Robot autopilot driven by the model's detections.

* World memory: detections are merged by floor position (repeated sightings make an object sure);
  an object is forgotten when the robot has driven over it, or when its place is in the camera's
  field of view but the model stopped seeing it (false positives fade out).
* Target: the nearest model-TAKE object, cheaper if it needs less turning; pure-pursuit steering,
  turning on the spot only when the target is far off the heading.
* Avoid: model-AVOID objects push the robot sideways and slow it down when they are in its lane.
* Explore: with nothing to collect, the robot sweeps the arena in straight lanes (lawn-mower).
"""
from __future__ import annotations

import math

ARENA = (-0.6, 4.2, -1.5, 1.5)       # x0, x1, y0, y1
LANE = 0.55


def _wrap(a):
    return (a + math.pi) % (2 * math.pi) - math.pi


class Brain:
    def __init__(self):
        self.memory = []              # dicts: x, y, top, take, material, volume, mass, score, hits, misses
        self.mode = "explore"
        self.target = None
        self.lanes = self._lanes()
        self.wp = 0

    def _lanes(self):
        x0, x1, y0, y1 = ARENA
        pts, y, flip = [], y0 + 0.3, False
        while y <= y1 - 0.3 + 1e-6:
            pts += [(x1 - 0.3, y), (x0 + 0.3, y)] if flip else [(x0 + 0.3, y), (x1 - 0.3, y)]
            y += LANE
            flip = not flip
        return pts

    def reset(self):
        self.__init__()

    # ---------------------------------------------------------------- perception
    def observe(self, dets, cam):
        matched = set()
        for d in dets:
            x, y = float(d["world"][0]), float(d["world"][1])
            best, bd = None, 0.14
            for i, m in enumerate(self.memory):
                dist = math.hypot(m["x"] - x, m["y"] - y)
                if dist < bd:
                    best, bd = i, dist
            info = {k: d[k] for k in ("take", "material", "volume", "mass", "score")}
            if best is None:
                self.memory.append({"x": x, "y": y, "top": d["top"], "hits": 1, "misses": 0, **info})
                matched.add(len(self.memory) - 1)
            else:
                m = self.memory[best]
                k = m["hits"]
                m["x"], m["y"] = (m["x"] * k + x) / (k + 1), (m["y"] * k + y) / (k + 1)
                m.update(info)
                m["top"], m["hits"], m["misses"] = d["top"], k + 1, 0
                matched.add(best)
        keep = []
        for i, m in enumerate(self.memory):
            if i not in matched and self._in_view(cam, m):
                m["misses"] += 1
            if m["misses"] < 2:
                keep.append(m)
        self.memory = keep

    @staticmethod
    def _in_view(cam, m):
        (px, py), z = cam.project((m["x"], m["y"], 0.02))
        return 0.3 < z < 1.4 and 40 < px < cam.W - 40 and 40 < py < cam.H - 40

    def passed(self, fx, fy):
        """Forget objects under the robot's intake (collected, or found to be nothing)."""
        self.memory = [m for m in self.memory if math.hypot(m["x"] - fx, m["y"] - fy) > 0.2 or not m["take"]]

    # ---------------------------------------------------------------- control
    def act(self, x, y, yaw):
        c, s = math.cos(yaw), math.sin(yaw)
        best = None
        for m in self.memory:
            if not m["take"]:
                continue
            dx, dy = m["x"] - x, m["y"] - y
            ang = _wrap(math.atan2(dy, dx) - yaw)
            cost = math.hypot(dx, dy) + 0.35 * abs(ang) + (0.3 if m["hits"] < 2 else 0.0)
            if best is None or cost < best[0]:
                best = (cost, m)
        if best is not None:
            self.mode, self.target = "target", best[1]
            goal = (best[1]["x"], best[1]["y"])
        else:
            self.mode, self.target = "explore", None
            goal = self.lanes[self.wp % len(self.lanes)]
            if math.hypot(goal[0] - x, goal[1] - y) < 0.3:
                self.wp += 1
                goal = self.lanes[self.wp % len(self.lanes)]
        dx, dy = goal[0] - x, goal[1] - y
        err = _wrap(math.atan2(dy, dx) - yaw)
        dist = math.hypot(dx, dy)
        if abs(err) > 0.9:                      # far off the heading: turn on the spot
            v, w = 0.04, 1.5 * (1 if err > 0 else -1)
        else:
            v = min(0.38, 0.12 + 0.5 * dist) * (1 - 0.5 * abs(err))
            w = max(-1.6, min(1.6, 2.4 * err))
        for m in self.memory:                   # keep out of the lane of model-AVOID objects
            if m["take"]:
                continue
            fx, fy = m["x"] - x, m["y"] - y
            fwd, lat = c * fx + s * fy, -s * fx + c * fy
            if 0.0 < fwd < 1.0 and abs(lat) < 0.5:
                push = (1 - fwd / 1.0) * (1 - abs(lat) / 0.5)
                w += (-1.0 if lat > 0 else 1.0) * 2.2 * push
                v *= 1 - 0.6 * push
        return v, max(-1.8, min(1.8, w))
