"""Static world of the polygon: walls, furniture and boxes, with 2D footprints for collisions and lidar."""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from .fastrt import Mat, Prim, Scene, rot_z

WALLS = (-1.3, 4.9, -2.0, 2.0)        # inner faces: x0, x1, y0, y1
ROBOT_HX, ROBOT_HY = 0.38, 0.33        # body + tracks half sizes for collisions


@dataclass
class Obstacle:
    kind: str
    prims: list
    circles: list = field(default_factory=list)   # (x, y, r)
    boxes: list = field(default_factory=list)     # (x, y, hx, hy, yaw)


def _box(x, y, hx, hy, hz, yaw, mat, z0=0.0, r=0.01):
    return Prim("rbox", (hx, hy, hz, r), (x, y, z0 + hz), rot=rot_z(yaw), mat=mat, bound=math.sqrt(hx * hx + hy * hy + hz * hz) + 0.01)


def add_walls(sc: Scene) -> list[Obstacle]:
    x0, x1, y0, y1 = WALLS
    paint = sc.mat(Mat((0.74, 0.74, 0.71), roughness=0.85, texture="fbm", tex_scale=6, tex_amp=0.04))
    base = sc.mat(Mat((0.12, 0.12, 0.13), roughness=0.5))
    out = []
    t, h = 0.06, 0.45
    for (cx, cy, hx, hy) in (((x0 + x1) / 2, y1 + t, (x1 - x0) / 2 + 2 * t, t), ((x0 + x1) / 2, y0 - t, (x1 - x0) / 2 + 2 * t, t),
                             (x1 + t, (y0 + y1) / 2, t, (y1 - y0) / 2), (x0 - t, (y0 + y1) / 2, t, (y1 - y0) / 2)):
        prims = [Prim("rbox", (hx, hy, h / 2, 0.004), (cx, cy, h / 2), mat=paint, bound=math.hypot(hx, hy) + h),
                 Prim("rbox", (hx + (0.012 if hy == t else 0), hy + (0.012 if hx == t else 0), 0.035, 0.003), (cx, cy, 0.035),
                      mat=base, bound=math.hypot(hx, hy) + 0.05)]
        for p in prims:
            sc.add(p)
        out.append(Obstacle("wall", prims, boxes=[(cx, cy, hx, hy, 0.0)]))
    return out


def random_obstacle(sc: Scene, rng, x, y) -> Obstacle:
    kind = str(rng.choice(["boxes", "table", "chair", "bucket", "pillar"], p=[0.35, 0.2, 0.2, 0.15, 0.1]))
    yaw = float(rng.uniform(0, math.pi))
    if kind == "boxes":
        card = sc.mat(Mat((0.42, 0.27, 0.14), roughness=0.9, texture="paper", tex_scale=40))
        hx, hy = rng.uniform(0.15, 0.25), rng.uniform(0.12, 0.2)
        h1, h2 = rng.uniform(0.12, 0.2), rng.uniform(0.08, 0.15)
        prims = [_box(x, y, hx, hy, h1, yaw, card), _box(x, y, hx * 0.8, hy * 0.85, h2, yaw + 0.15, card, z0=2 * h1)]
        ob = Obstacle(kind, prims, boxes=[(x, y, hx, hy, yaw)])
    elif kind == "table":
        metal = sc.mat(Mat((0.35, 0.35, 0.37), roughness=0.35, metallic=1.0))
        wood = sc.mat(Mat((0.55, 0.42, 0.28), (0.3, 0.2, 0.12), 0.5, texture="wood", tex_scale=30))
        hx, hy, H = 0.6, 0.35, 0.74
        c, s = math.cos(yaw), math.sin(yaw)
        legs, prims = [], [_box(x, y, hx, hy, 0.015, yaw, wood, z0=H - 0.03, r=0.006)]
        for lx, ly in ((hx - 0.05, hy - 0.05), (-hx + 0.05, hy - 0.05), (hx - 0.05, -hy + 0.05), (-hx + 0.05, -hy + 0.05)):
            px, py = x + c * lx - s * ly, y + s * lx + c * ly
            prims.append(Prim("rcyl", (0.022, H / 2, 0.004), (px, py, H / 2), mat=metal, bound=H / 2 + 0.03))
            legs.append((px, py, 0.03))
        ob = Obstacle(kind, prims, circles=legs)
    elif kind == "chair":
        plastic = sc.mat(Mat((0.08, 0.3, 0.32), roughness=0.4, clearcoat=0.5))
        metal = sc.mat(Mat((0.3, 0.3, 0.32), roughness=0.3, metallic=1.0))
        hs, H = 0.22, 0.45
        c, s = math.cos(yaw), math.sin(yaw)
        prims = [_box(x, y, hs, hs, 0.02, yaw, plastic, z0=H, r=0.015),
                 _box(x - c * hs, y - s * hs, 0.015, hs, 0.2, yaw, plastic, z0=H + 0.04, r=0.01)]
        for lx, ly in ((hs - 0.03, hs - 0.03), (-hs + 0.03, hs - 0.03), (hs - 0.03, -hs + 0.03), (-hs + 0.03, -hs + 0.03)):
            prims.append(Prim("rcyl", (0.014, H / 2, 0.003), (x + c * lx - s * ly, y + s * lx + c * ly, H / 2), mat=metal, bound=H / 2 + 0.02))
        ob = Obstacle(kind, prims, boxes=[(x, y, hs, hs, yaw)])
    elif kind == "bucket":
        pl = sc.mat(Mat((0.75, 0.12, 0.06), roughness=0.35, clearcoat=0.6))
        r, H = rng.uniform(0.13, 0.18), rng.uniform(0.25, 0.35)
        prims = [Prim("rcyl", (r, H / 2, 0.01), (x, y, H / 2), mat=pl, bound=H / 2 + r)]
        ob = Obstacle(kind, prims, circles=[(x, y, r)])
    else:
        conc = sc.mat(Mat((0.62, 0.62, 0.6), roughness=0.9, texture="fbm", tex_scale=12, tex_amp=0.08))
        prims = [_box(x, y, 0.15, 0.15, 0.6, yaw, conc, r=0.01)]
        ob = Obstacle(kind, prims, boxes=[(x, y, 0.15, 0.15, yaw)])
    for p in prims:
        sc.add(p)
    return ob


def _box_hit(px, py, b, margin=0.0):
    x, y, hx, hy, yaw = b
    c, s = math.cos(yaw), math.sin(yaw)
    dx, dy = px - x, py - y
    lx, ly = c * dx + s * dy, -s * dx + c * dy
    return abs(lx) <= hx + margin and abs(ly) <= hy + margin


def robot_collides(obstacles: list[Obstacle], x, y, yaw) -> bool:
    """Robot rectangle vs obstacle footprints (sampled boundary + centre test)."""
    c, s = math.cos(yaw), math.sin(yaw)
    pts = [(lx, ly) for lx in np.linspace(-ROBOT_HX, ROBOT_HX, 7) for ly in (-ROBOT_HY, ROBOT_HY)] + \
          [(lx, ly) for lx in (-ROBOT_HX, ROBOT_HX) for ly in np.linspace(-ROBOT_HY, ROBOT_HY, 5)]
    world = [(x + c * lx - s * ly, y + s * lx + c * ly) for lx, ly in pts]
    for ob in obstacles:
        for (cx, cy, r) in ob.circles:
            lx, ly = c * (cx - x) + s * (cy - y), -s * (cx - x) + c * (cy - y)
            if abs(lx) <= ROBOT_HX + r and abs(ly) <= ROBOT_HY + r:
                return True
        for b in ob.boxes:
            if any(_box_hit(px, py, b) for px, py in world):
                return True
            bx, by, bhx, bhy, byaw = b
            cb, sb = math.cos(byaw), math.sin(byaw)
            for lx, ly in ((bhx, bhy), (-bhx, bhy), (bhx, -bhy), (-bhx, -bhy)):
                qx, qy = bx + cb * lx - sb * ly, by + sb * lx + cb * ly
                rx, ry = c * (qx - x) + s * (qy - y), -s * (qx - x) + c * (qy - y)
                if abs(rx) <= ROBOT_HX and abs(ry) <= ROBOT_HY:
                    return True
    return False


def lidar(obstacles: list[Obstacle], x, y, yaw, n=72, rmax=4.0):
    """2D lidar from the robot centre: ranges (n,) in metres, angle k*2pi/n relative to the heading."""
    ang = yaw + np.arange(n) * 2 * np.pi / n
    dx, dy = np.cos(ang), np.sin(ang)
    best = np.full(n, rmax)
    for ob in obstacles:
        for (cx, cy, r) in ob.circles:
            ox, oy = x - cx, y - cy
            b = ox * dx + oy * dy
            cc = ox * ox + oy * oy - r * r
            disc = b * b - cc
            t = -b - np.sqrt(np.maximum(disc, 0))
            ok = (disc > 0) & (t > 0)
            best = np.where(ok & (t < best), t, best)
        for (bx, by, hx, hy, byaw) in ob.boxes:
            cb, sb = math.cos(byaw), math.sin(byaw)
            ox, oy = cb * (x - bx) + sb * (y - by), -sb * (x - bx) + cb * (y - by)
            ldx, ldy = cb * dx + sb * dy, -sb * dx + cb * dy
            with np.errstate(divide="ignore", invalid="ignore"):
                t1x, t2x = (-hx - ox) / ldx, (hx - ox) / ldx
                t1y, t2y = (-hy - oy) / ldy, (hy - oy) / ldy
            tmin = np.maximum(np.minimum(t1x, t2x), np.minimum(t1y, t2y))
            tmax = np.minimum(np.maximum(t1x, t2x), np.maximum(t1y, t2y))
            ok = (tmax >= np.maximum(tmin, 0)) & (tmin > 0)
            best = np.where(ok & (tmin < best), tmin, best)
    return best


def free_spot(obstacles, x, y, r) -> bool:
    """True if a disc of radius r at (x, y) is clear of all footprints and inside the walls."""
    x0, x1, y0, y1 = WALLS
    if not (x0 + r < x < x1 - r and y0 + r < y < y1 - r):
        return False
    for ob in obstacles:
        if ob.kind == "wall":
            continue
        for (cx, cy, cr) in ob.circles:
            if math.hypot(x - cx, y - cy) < r + cr:
                return False
        for b in ob.boxes:
            if _box_hit(x, y, b, margin=r):
                return False
    return True
