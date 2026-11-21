"""Robot and procedural trash as primitive tables for the real-time renderer (polygon.fastrt)."""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .fastrt import LAY_X, Mat, Prim, Scene, rot_x, rot_y, rot_z, transform, volume_ml

# ---------------------------------------------------------------- robot
WHITE, RED, BLACK = (0.78, 0.78, 0.77), (0.83, 0.055, 0.04), (0.018, 0.018, 0.02)
HX, HY, HZ = 0.31, 0.24, 0.20
BODY_Z = 0.17 + HZ
TOP = BODY_Z + HZ
CAM_POS = (HX + 0.045, 0.0, 0.25)
CAM_PITCH_DEG = 30.0


def add_robot(sc: Scene, x: float, y: float, yaw: float, brush_angle: float = 0.0):
    m_body = sc.mat(Mat(WHITE, RED, 0.35, texture="robot", clearcoat=0.5))
    m_black = sc.mat(Mat(BLACK, roughness=0.6, texture="fbm", tex_scale=200, tex_amp=0.3))
    m_rubber = sc.mat(Mat(BLACK, roughness=0.85, texture="fbm", tex_scale=300, tex_amp=0.4))
    m_red = sc.mat(Mat(RED, roughness=0.4, clearcoat=0.5))
    m_glossy = sc.mat(Mat((0.04, 0.045, 0.05), roughness=0.15, clearcoat=1.0))
    m_bristle = sc.mat(Mat((0.03, 0.03, 0.035), roughness=0.75, texture="bristle"))
    m_white = sc.mat(Mat(WHITE, roughness=0.45))
    m_orange = sc.mat(Mat((0.9, 0.18, 0.05), roughness=0.4))
    parts = [
        Prim("rbox", (HX, HY, HZ, 0.035), (0, 0, BODY_Z), mat=m_body, bound=0.48),
        Prim("rbox", (HX + 0.02, HY + 0.015, 0.05, 0.04), (0, 0, 0.12), mat=m_black, bound=0.42),
        Prim("rbox", (0.014, 0.10, 0.028, 0.008), (HX + 0.012, 0, 0.25), mat=m_red, bound=0.11),
        Prim("rbox", (0.004, 0.045, 0.012, 0.004), (HX + 0.027, 0, 0.25), mat=m_glossy, bound=0.05),
        Prim("rcyl", (0.045, 0.03, 0.006), (0.12, 0, TOP + 0.03), mat=m_glossy, bound=0.06),
        Prim("rbox", (0.13, 0.09, 0.006, 0.004), (-0.06, 0.02, TOP + 0.006), mat=m_glossy, bound=0.17),
        Prim("rcyl", (0.022, 0.016, 0.008), (-0.21, -0.15, TOP + 0.016), mat=m_red, bound=0.03),
        Prim("rbox", (0.06, 0.17, 0.09, 0.012), (-HX - 0.06, 0, 0.12), mat=m_white, bound=0.21),
    ]
    for wx in (-0.2, 0.2):
        for s in (-1, 1):
            parts.append(Prim("rcyl", (0.085, 0.03, 0.012), (wx, s * (HY + 0.03), 0.085), rot=rot_x(math.pi / 2),
                              mat=m_rubber, bound=0.095))
    for s in (-1, 1):
        bx, by = HX + 0.10, s * 0.15
        parts.append(Prim("rcyl", (0.16, 0.022, 0.01), (bx, by, 0.025), rot=rot_z(brush_angle * s), mat=m_bristle,
                          disp=(0.004, 35.0, 3.0), flags=4, bound=0.18))
        parts.append(Prim("rcyl", (0.04, 0.025, 0.006), (bx, by, 0.065), mat=m_glossy, bound=0.05))
        parts.append(Prim("capsule", (HX - 0.05 - bx, by * 0.6 - by, 0.13 - 0.08, 0, 0, 0, 0.018), (bx, by, 0.08),
                          mat=m_black, bound=0.2))
    for ry in (-0.08, 0.08):
        parts.append(Prim("torus", (0.03, 0.009), (-HX - 0.122, ry, 0.11), rot=rot_y(math.pi / 2), mat=m_orange, bound=0.045))
    for p in transform(parts, (x, y, 0.0), yaw):
        sc.add(p)


def robot_camera(x, y, yaw):
    c, s = math.cos(yaw), math.sin(yaw)
    cx, cy, cz = CAM_POS
    eye = np.array([x + c * cx - s * cy, y + s * cx + c * cy, cz])
    p = math.radians(CAM_PITCH_DEG)
    return eye, eye + np.array([c * math.cos(p), s * math.cos(p), -math.sin(p)])


# ---------------------------------------------------------------- trash
DENSITY = {
    ("bulk", "metal"): (0.714, 2.25), ("bulk", "mineral"): (1.5, 1.778), ("bulk", "paper"): (0.014, 0.05),
    ("bulk", "plastic"): (0.077, 1.2), ("bulk", "wood"): (0.583, 0.8), ("bulk", "glass"): (1.0, 1.4),
    ("solid", "glass"): (0.9, 1.233), ("solid", "metal"): (0.085, 1.556), ("solid", "mineral"): (1.414, 2.0),
    ("solid", "paper"): (0.061, 0.889), ("solid", "plastic"): (0.042, 1.0), ("solid", "wood"): (0.789, 0.875),
}
MATERIALS = ["plastic", "metal", "glass", "paper", "wood", "mineral"]
SHAPES = {
    "plastic": [("bottle", "solid", 3), ("box", "solid", 2), ("crumpled", "bulk", 2), ("block", "solid", 1), ("pile", "bulk", 1), ("can", "solid", 1)],
    "metal": [("can", "solid", 4), ("block", "solid", 2), ("pile", "bulk", 2)],
    "glass": [("bottle", "solid", 5), ("pile", "bulk", 1)],
    "paper": [("box", "solid", 3), ("crumpled", "bulk", 3), ("pile", "bulk", 1)],
    "wood": [("block", "solid", 3), ("pile", "bulk", 3)],
    "mineral": [("rock", "solid", 4), ("pile", "bulk", 3)],
}


def _hsv(rng, sat=(0.5, 0.95), val=(0.25, 0.85)):
    h, s, v = rng.uniform(0, 1), rng.uniform(*sat), rng.uniform(*val)
    i = int(h * 6) % 6
    f = h * 6 - int(h * 6)
    p_, q, t = v * (1 - s), v * (1 - f * s), v * (1 - (1 - f) * s)
    rgb = [(v, t, p_), (q, v, p_), (p_, v, t), (p_, q, v), (t, p_, v), (v, p_, q)][i]
    return tuple(float(c) ** 2.2 for c in rgb)


def trash_material(name, rng, kind=""):
    u = lambda a, b: float(rng.uniform(a, b))
    if name == "plastic":
        dark = rng.random() < 0.25
        col = (0.02, 0.02, 0.025) if dark else _hsv(rng)
        tex = "label" if kind == "bottle" and rng.random() < 0.6 else ("fbm" if rng.random() < 0.3 else "plain")
        return Mat(col, _hsv(rng, val=(0.6, 1.0)), u(0.2, 0.55), texture=tex,
                   tex_amp=u(0.02, 0.05) if tex == "label" else 0.12, tex_scale=25, clearcoat=float(rng.random() < 0.4))
    if name == "metal":
        if rng.random() < 0.6:
            g = u(0.75, 0.95)
            return Mat((g, g, g * 1.02), roughness=u(0.12, 0.35), metallic=1.0, texture="brushed")
        return Mat(_hsv(rng), (0.9, 0.9, 0.9), u(0.15, 0.3), metallic=0.7, texture="label", tex_amp=u(0.02, 0.04))
    if name == "glass":
        tints = [(0.82, 0.93, 0.82), (0.92, 0.8, 0.6), (0.97, 0.98, 0.98), (0.78, 0.9, 0.88)]
        return Mat((0.05, 0.06, 0.05), roughness=0.02, transmission=0.95, tint=tints[int(rng.integers(4))])
    if name == "paper":
        if rng.random() < 0.5:
            return Mat((0.42, 0.27, 0.14), roughness=0.9, texture="paper", tex_scale=60)
        g = u(0.7, 0.9)
        return Mat((g, g, g * 0.97), roughness=0.85, texture="paper", tex_scale=80)
    if name == "wood":
        b = (u(0.35, 0.6), u(0.2, 0.35), u(0.08, 0.15))
        return Mat(tuple(c ** 1.6 for c in b), tuple(c ** 1.6 * 0.55 for c in b), 0.7, texture="wood", tex_scale=u(40, 90))
    g, w = u(0.06, 0.22), u(0.85, 1.15)
    return Mat((g * w, g, g / w), (g * 0.35, g * 0.33, g * 0.3), 0.8, texture="stone", tex_scale=u(35, 90))


@dataclass
class Trash:
    kind: str
    form: str
    material: str
    prims: list            # in object frame (origin on the floor)
    mat: Mat
    radius: float
    height: float
    volume_ml: float = 0.0
    mass_g: float = 0.0
    filled: bool = False

    @property
    def take(self):
        return int(self.mass_g <= 500)

    def place(self, sc: Scene, x, y, yaw, obj_id):
        mi = sc.mat(self.mat)
        g = sc.new_group()
        for p in transform(self.prims, (x, y, 0.0), yaw):
            p.mat, p.obj = mi, obj_id
            if p.group == 0:
                p.group = g
            elif p.group > 0:
                p.group = g * 100 + p.group
            sc.add(p)


def _shape(kind, material, rng):
    u = lambda a, b: float(rng.uniform(a, b))
    if kind == "bottle":
        r, L = u(0.028, 0.045), u(0.14, 0.26)
        rn, ln = r * u(0.3, 0.42), u(0.035, 0.06)
        prims = [Prim("rcyl", (r, L / 2, r * 0.35), (0, 0, r), rot=LAY_X, group=0, bound=L / 2 + r),
                 Prim("rcyl", (rn, ln / 2, 0.003), (L / 2 + ln / 2, 0, r), rot=LAY_X, group=0, smooth=r * 0.6, bound=ln / 2 + rn + 0.01),
                 Prim("rcyl", (rn * 1.15, 0.008, 0.002), (L / 2 + ln, 0, r), rot=LAY_X, bound=rn * 1.2 + 0.01)]
        return prims, (L + 2 * ln) / 2 + 0.01, 2 * r
    if kind == "can":
        r, h = u(0.026, 0.034), u(0.09, 0.125)
        lying = rng.random() < 0.7
        crush = rng.random() < 0.4
        disp = (0.005, 50.0, u(0, 50)) if crush else (0.0, 1.0, 0.0)
        if lying:
            return [Prim("rcyl", (r, h / 2, 0.003), (0, 0, r), rot=LAY_X, disp=disp, bound=h / 2 + r)], h / 2 + 0.01, 2 * r
        return [Prim("rcyl", (r, h / 2, 0.003), (0, 0, h / 2), disp=disp, bound=h / 2 + r)], r + 0.01, h
    if kind in ("box", "block"):
        if kind == "box":
            b = (u(0.02, 0.075), u(0.015, 0.05), u(0.008, 0.045))
            rr = min(b) * 0.12
        else:
            b = (u(0.02, 0.09), u(0.015, 0.05), u(0.01, 0.035))
            rr = 0.002
        return [Prim("rbox", (*b, rr), (0, 0, b[2]), bound=math.sqrt(sum(v * v for v in b)) + 0.005)], math.hypot(b[0], b[1]) + 0.01, 2 * b[2]
    if kind == "rock":
        r = (u(0.02, 0.08), u(0.015, 0.06), u(0.01, 0.04))
        return [Prim("ellipsoid", r, (0, 0, r[2] * 0.85), disp=(min(r) * 0.32, 14.0, u(0, 100)), flags=2,
                     bound=max(r) * 1.35)], max(r[:2]) * 1.3 + 0.01, 2 * r[2]
    if kind == "crumpled":
        r = u(0.03, 0.075)
        return [Prim("ellipsoid", (r, r, r * 0.8), (0, 0, r * 0.8), disp=(r * 0.35, 9.0 / 1.0, u(0, 100)), flags=1,
                     bound=r * 1.45)], r * 1.4 + 0.01, r * 1.9
    # pile: many small pieces (one group, hard union)
    n = int(rng.integers(8, 30))
    R = u(0.04, 0.10)
    piece = {"wood": (0.012, 0.004, 0.003), "mineral": (0.008, 0.007, 0.005), "metal": (0.010, 0.003, 0.003),
             "plastic": (0.012, 0.008, 0.002), "paper": (0.015, 0.012, 0.001), "glass": (0.008, 0.006, 0.003)}[material]
    prims, hmax = [], 0.0
    for _ in range(n):
        a, rad = rng.uniform(0, 2 * np.pi), R * math.sqrt(rng.uniform(0, 1))
        sz = np.array(piece) * rng.uniform(0.6, 1.6)
        h = sz[2] + max(0.0, 1 - rad / R) * R * 0.35 * rng.uniform(0.3, 1)
        hmax = max(hmax, h + sz[2])
        prims.append(Prim("ellipsoid", tuple(sz), (rad * math.cos(a), rad * math.sin(a), h), rot=rot_z(rng.uniform(0, np.pi)),
                          group=0, bound=float(sz.max()) * 1.05))
    return prims, R + 0.025, hmax


def random_trash(rng, material=None) -> Trash:
    material = material or str(rng.choice(MATERIALS, p=[0.34, 0.14, 0.1, 0.16, 0.13, 0.13]))
    opts = SHAPES[material]
    w = np.array([o[2] for o in opts], float)
    kind, form, _ = opts[int(rng.choice(len(opts), p=w / w.sum()))]
    prims, rad, h = _shape(kind, material, rng)
    t = Trash(kind, form, material, prims, trash_material(material, rng, kind), rad, h)
    sc = Scene()
    sc.mat(t.mat)
    t.place(sc, 0.0, 0.0, 0.0, 0)
    t.volume_ml = volume_ml(sc, (0.0, 0.0), rad + 0.02, h + 0.02)
    lo, hi = DENSITY[(form, material)]
    t.filled = kind in ("bottle", "can") and material in ("plastic", "glass", "metal") and rng.random() < 0.3
    rho = rng.uniform(0.85, 1.1) if t.filled else float(np.exp(rng.uniform(np.log(lo), np.log(hi))))
    t.mass_g = t.volume_ml * rho
    return t
