"""Procedural trash with known properties: form, material, exact volume, mass, take.

Every object is built in a local frame whose origin lies on the floor under the object (z up),
so placing it is just a floor position and a yaw. The volume is integrated from the SDF on a
voxel grid, so it is exact up to the grid step (< 1 %).
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import torch

from . import sdf as S
from .render import DEV, Material, Part

def _u(rng, *a):
    v = rng.uniform(*a)
    return float(v) if np.ndim(v) == 0 else v


MATERIALS = ["plastic", "metal", "glass", "paper", "wood", "mineral"]

# density ranges (g/ml) per form x material, from datasets/raw/densities.csv (min, max)
DENSITY = {
    ("bulk", "metal"): (0.714, 2.25), ("bulk", "mineral"): (1.5, 1.778), ("bulk", "paper"): (0.014, 0.05),
    ("bulk", "plastic"): (0.077, 1.2), ("bulk", "wood"): (0.583, 0.8), ("bulk", "glass"): (1.0, 1.4),
    ("solid", "glass"): (0.9, 1.233), ("solid", "metal"): (0.085, 1.556), ("solid", "mineral"): (1.414, 2.0),
    ("solid", "paper"): (0.061, 0.889), ("solid", "plastic"): (0.042, 1.0), ("solid", "wood"): (0.789, 0.875),
}


@dataclass
class TrashObject:
    kind: str
    form: str
    material: str
    parts: list[Part]          # local-frame parts (pos is the offset inside the object)
    radius: float              # footprint radius used for placement
    volume_ml: float
    mass_g: float
    filled: bool = False

    @property
    def take(self) -> int:
        return int(self.mass_g <= 500.0)

    def placed(self, x: float, y: float, yaw: float, obj_id: int) -> list[Part]:
        out = []
        c, s = math.cos(yaw), math.sin(yaw)
        for p in self.parts:
            px, py, pz = p.pos
            out.append(Part(p.sdf, p.material, p.radius, (x + c * px - s * py, y + s * px + c * py, pz),
                            p.yaw + yaw, obj_id, p.no_shadow))
        return out


# ---------------------------------------------------------------- materials

def _rand_color(rng, sat=(0.5, 0.95), val=(0.25, 0.85)):
    h = _u(rng, 0, 1)
    s, v = _u(rng, *sat), _u(rng, *val)
    i = int(h * 6) % 6
    f = h * 6 - int(h * 6)
    p_, q, t = v * (1 - s), v * (1 - f * s), v * (1 - (1 - f) * s)
    rgb = [(v, t, p_), (q, v, p_), (p_, v, t), (p_, q, v), (t, p_, v), (v, p_, q)][i]
    return tuple(float(c) ** 2.2 for c in rgb)


def material_for(name: str, rng: np.random.Generator, kind: str = "") -> Material:
    if name == "plastic":
        dark = rng.random() < 0.25
        col = (0.02, 0.02, 0.025) if dark else _rand_color(rng)
        tex = "label" if kind == "bottle" and rng.random() < 0.6 else ("fbm" if rng.random() < 0.3 else "plain")
        return Material(col, float(_u(rng, 0.2, 0.55)), texture=tex, color2=_rand_color(rng, val=(0.6, 1.0)),
                        tex_amp=float(_u(rng, 0.02, 0.05)) if tex == "label" else 0.12, tex_scale=25,
                        clearcoat=float(rng.random() < 0.4))
    if name == "metal":
        if rng.random() < 0.6:        # bare aluminium / steel
            g = _u(rng, 0.75, 0.95)
            return Material((g, g, g * 1.02), float(_u(rng, 0.12, 0.35)), metallic=1.0, texture="brushed")
        return Material(_rand_color(rng), float(_u(rng, 0.15, 0.3)), metallic=0.7, texture="label",
                        color2=(0.9, 0.9, 0.9), tex_amp=float(_u(rng, 0.02, 0.04)))
    if name == "glass":
        tints = [(0.82, 0.93, 0.82), (0.92, 0.8, 0.6), (0.97, 0.98, 0.98), (0.78, 0.9, 0.88)]
        tint = tints[int(rng.integers(len(tints)))]
        return Material((0.05, 0.06, 0.05), 0.02, transmission=0.95, ior=1.5, tint=tint)
    if name == "paper":
        if rng.random() < 0.5:
            return Material((0.42, 0.27, 0.14), 0.9, texture="paper", tex_scale=60)          # cardboard
        g = _u(rng, 0.7, 0.9)
        return Material((g, g, g * 0.97), 0.85, texture="paper", tex_scale=80)
    if name == "wood":
        base = (_u(rng, 0.35, 0.6), _u(rng, 0.2, 0.35), _u(rng, 0.08, 0.15))
        return Material(tuple(c ** 1.6 for c in base), 0.7, texture="wood",
                        color2=tuple(c ** 1.6 * 0.55 for c in base), tex_scale=float(_u(rng, 40, 90)))
    if name == "mineral":
        g = _u(rng, 0.06, 0.22)
        warm = _u(rng, 0.85, 1.15)
        c1 = (g * warm, g, g / warm)
        return Material(c1, 0.8, texture="stone", color2=(g * 0.35, g * 0.33, g * 0.3), tex_scale=float(_u(rng, 35, 90)))
    raise ValueError(name)


# ---------------------------------------------------------------- volume

def sdf_volume_ml(parts: list[Part], radius: float, height: float, step: float | None = None) -> float:
    """Volume of the union of parts (in the object's local frame) by voxel integration."""
    step = step or max(radius, height) / 90
    xs = torch.arange(-radius, radius, step, device=DEV) + step / 2
    zs = torch.arange(0, height, step, device=DEV) + step / 2
    total = 0
    for z0 in range(0, len(zs), 16):
        zz = zs[z0:z0 + 16]
        g = torch.stack(torch.meshgrid(xs, xs, zz, indexing="ij"), -1).reshape(-1, 3)
        d = torch.full((len(g),), 1e3, device=DEV)
        for p in parts:
            d = torch.minimum(d, p.sdf(p.to_local(g)).to(d.dtype))
        total += int((d < 0).sum())
    return total * step ** 3 * 1e6


# ---------------------------------------------------------------- shapes (local frame, resting on z=0)

def _bottle(rng, mat):
    r = _u(rng, 0.028, 0.045)
    L = _u(rng, 0.14, 0.26)
    rn, ln = r * _u(rng, 0.3, 0.42), _u(rng, 0.035, 0.06)
    wall = 0.0

    def f(p):
        q = p - torch.tensor([0.0, 0.0, r], device=p.device)
        body = S.round_cylinder_z(S.lying(q), r, L / 2, r * 0.35)
        neck = S.round_cylinder_z(S.lying(q - torch.tensor([L / 2 + ln / 2, 0, 0], device=p.device)), rn, ln / 2, 0.003)
        cap = S.round_cylinder_z(S.lying(q - torch.tensor([L / 2 + ln, 0, 0], device=p.device)), rn * 1.15, 0.008, 0.002)
        return torch.minimum(S.smin(body, neck, r * 0.6), cap)
    return f, (L + 2 * ln) / 2 + 0.01, 2 * r


def _can(rng, mat):
    r = _u(rng, 0.026, 0.034)
    h = _u(rng, 0.09, 0.125)
    crush = rng.random() < 0.4
    lying_ = rng.random() < 0.7

    def f(p):
        q = p - torch.tensor([0.0, 0.0, r if lying_ else h / 2], device=p.device)
        if lying_:
            q = S.lying(q)
        if crush:
            q = q * torch.tensor([1.0, 1.6, 1.0], device=p.device)
        d = S.round_cylinder_z(q, r, h / 2, 0.003)
        if crush:
            d = d / 1.6 + 0.004 * S.fbm(p * 60, 3)
        return d
    rad = (h / 2 if lying_ else r) + 0.01
    return f, max(rad, r) + 0.01, (2 * r if lying_ else h)


def _box(rng, mat):
    b = np.array([_u(rng, 0.02, 0.075), _u(rng, 0.015, 0.05), _u(rng, 0.008, 0.045)])
    rr = min(b) * 0.12

    def f(p):
        q = p - torch.tensor([0.0, 0.0, b[2]], device=p.device)
        return S.round_box(q, b, rr)
    return f, float(np.hypot(b[0], b[1])) + 0.01, float(2 * b[2])


def _block(rng, mat):
    b = np.array([_u(rng, 0.02, 0.09), _u(rng, 0.015, 0.05), _u(rng, 0.01, 0.035)])

    def f(p):
        q = p - torch.tensor([0.0, 0.0, b[2]], device=p.device)
        return S.round_box(q, b, 0.002)
    return f, float(np.hypot(b[0], b[1])) + 0.01, float(2 * b[2])


def _rock(rng, mat):
    r = np.array([_u(rng, 0.02, 0.08), _u(rng, 0.015, 0.06), _u(rng, 0.01, 0.04)])
    seed = _u(rng, 0, 100)

    def f(p):
        q = p - torch.tensor([0.0, 0.0, r[2] * 0.85], device=p.device)
        d = S.ellipsoid(q, r) + min(r) * 0.32 * S.fbm(q * 14 + seed, 5) + 0.0015 * S.value_noise(q * 220)
        return torch.maximum(d, -p[..., 2])          # flat where it rests on the floor
    return f, float(max(r[:2]) * 1.3) + 0.01, float(2 * r[2])


def _crumpled(rng, mat):
    r = _u(rng, 0.03, 0.075)
    seed = _u(rng, 0, 100)

    def f(p):
        q = (p - torch.tensor([0.0, 0.0, r * 0.8], device=p.device)) * torch.tensor([1.0, 1.0, 1.25], device=p.device)
        n = S.fbm(q * 9 + seed, 4)
        return (q.norm(dim=-1) - r) / 1.25 + r * 0.35 * (n.abs() - 0.25)
    return f, r * 1.4 + 0.01, r * 1.9


def _pile(rng, mat, material):
    """Bulk: many small pieces (wood chips, gravel, screws, plastic scraps) in a heap."""
    n = int(rng.integers(8, 30))
    R = _u(rng, 0.04, 0.10)
    piece = {"wood": (0.012, 0.004, 0.003), "mineral": (0.008, 0.007, 0.005), "metal": (0.010, 0.003, 0.003),
             "plastic": (0.012, 0.008, 0.002), "paper": (0.015, 0.012, 0.001), "glass": (0.008, 0.006, 0.003)}[material]
    ang = _u(rng, 0, 2 * np.pi, n)
    rad = R * np.sqrt(_u(rng, 0, 1, n))
    centers = np.stack([rad * np.cos(ang), rad * np.sin(ang)], 1)
    sizes = np.array(piece)[None] * _u(rng, 0.6, 1.6, (n, 1))
    heights = sizes[:, 2] + np.maximum(0, (1 - rad / R)) * R * 0.35 * _u(rng, 0.3, 1, n)
    yaws = _u(rng, 0, np.pi, n)
    C = torch.tensor(np.c_[centers, heights], device=DEV, dtype=torch.float32)
    Sz = torch.tensor(sizes, device=DEV, dtype=torch.float32)
    cy, sy = torch.tensor(np.cos(yaws), device=DEV, dtype=torch.float32), torch.tensor(np.sin(yaws), device=DEV, dtype=torch.float32)

    def f(p):
        q = p[:, None, :] - C[None]
        x = cy * q[..., 0] + sy * q[..., 1]
        y = -sy * q[..., 0] + cy * q[..., 1]
        q2 = torch.stack([x, y, q[..., 2]], -1)
        k0 = (q2 / Sz).norm(dim=-1)
        k1 = (q2 / (Sz * Sz)).norm(dim=-1)
        return (k0 * (k0 - 1) / k1.clamp(min=1e-9)).min(dim=1).values
    return f, R + 0.025, float(heights.max() + sizes[:, 2].max())


SHAPES = {
    "plastic": [("bottle", "solid", 3), ("box", "solid", 2), ("crumpled", "bulk", 2), ("block", "solid", 1), ("pile", "bulk", 1), ("can", "solid", 1)],
    "metal": [("can", "solid", 4), ("block", "solid", 2), ("pile", "bulk", 2)],
    "glass": [("bottle", "solid", 5), ("pile", "bulk", 1)],
    "paper": [("box", "solid", 3), ("crumpled", "bulk", 3), ("pile", "bulk", 1)],
    "wood": [("block", "solid", 3), ("pile", "bulk", 3)],
    "mineral": [("rock", "solid", 4), ("pile", "bulk", 3)],
}


def random_trash(rng: np.random.Generator, material: str | None = None) -> TrashObject:
    material = material or str(rng.choice(MATERIALS, p=[0.34, 0.14, 0.1, 0.16, 0.13, 0.13]))
    opts = SHAPES[material]
    w = np.array([o[2] for o in opts], float)
    kind, form, _ = opts[int(rng.choice(len(opts), p=w / w.sum()))]
    mat = material_for(material, rng, kind)
    if kind == "pile":
        f, rad, h = _pile(rng, mat, material)
    else:
        f, rad, h = {"bottle": _bottle, "can": _can, "box": _box, "block": _block, "rock": _rock,
                     "crumpled": _crumpled}[kind](rng, mat)
    part = Part(f, mat, radius=float(math.hypot(rad, h)) + 0.01)
    vol = sdf_volume_ml([part], rad + 0.01, h + 0.02)
    lo, hi = DENSITY[(form, material)]
    filled = kind in ("bottle", "can") and material in ("plastic", "glass", "metal") and rng.random() < 0.3
    if filled:
        rho = _u(rng, 0.85, 1.1)
    else:
        rho = float(np.exp(_u(rng, np.log(lo), np.log(hi))))
    return TrashObject(kind, form, material, [part], rad, vol, vol * rho, filled)
