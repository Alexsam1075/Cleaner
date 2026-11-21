"""Signed distance functions and procedural noise in PyTorch (GPU).

World frame of a dataset frame: origin at the camera foot on the floor, x forward along the floor,
y to the side, z up (toward the camera). Units are metres. Every SDF takes points (N, 3) in the
part's local frame and returns distances (N,).
"""
from __future__ import annotations

import math

import torch


# ---------------------------------------------------------------- noise

def _hash(p: torch.Tensor) -> torch.Tensor:
    """Pseudo-random value in [0, 1) for integer lattice points (N, 3)."""
    h = p[..., 0] * 127.1 + p[..., 1] * 311.7 + p[..., 2] * 74.7
    return torch.frac(torch.sin(h) * 43758.5453)


def value_noise(p: torch.Tensor) -> torch.Tensor:
    """Smooth 3D value noise in [-1, 1]."""
    i = torch.floor(p)
    f = p - i
    u = f * f * (3 - 2 * f)
    out = 0.0
    for dx in (0, 1):
        for dy in (0, 1):
            for dz in (0, 1):
                corner = i + torch.tensor([dx, dy, dz], device=p.device, dtype=p.dtype)
                w = ((u[..., 0] if dx else 1 - u[..., 0]) * (u[..., 1] if dy else 1 - u[..., 1])
                     * (u[..., 2] if dz else 1 - u[..., 2]))
                out = out + w * _hash(corner)
    return out * 2 - 1


def fbm(p: torch.Tensor, octaves: int = 4, lacunarity: float = 2.0, gain: float = 0.5) -> torch.Tensor:
    amp, total, norm = 1.0, 0.0, 0.0
    for _ in range(octaves):
        total = total + amp * value_noise(p)
        norm += amp
        p = p * lacunarity + 17.3
        amp *= gain
    return total / norm


# ---------------------------------------------------------------- primitives (local frame)

def sphere(p, r):
    return p.norm(dim=-1) - r


def box(p, b):
    b = torch.as_tensor(b, device=p.device, dtype=p.dtype)
    q = p.abs() - b
    return q.clamp(min=0).norm(dim=-1) + q.max(dim=-1).values.clamp(max=0)


def round_box(p, b, r):
    b = torch.as_tensor(b, device=p.device, dtype=p.dtype)
    return box(p, b - r) - r


def cylinder_z(p, r, h):
    """Vertical cylinder centred at the origin, radius r, half-height h."""
    d = torch.stack([p[..., :2].norm(dim=-1) - r, p[..., 2].abs() - h], dim=-1)
    return d.clamp(min=0).norm(dim=-1) + d.max(dim=-1).values.clamp(max=0)


def round_cylinder_z(p, r, h, rr):
    d = torch.stack([p[..., :2].norm(dim=-1) - (r - rr), p[..., 2].abs() - (h - rr)], dim=-1)
    return d.clamp(min=0).norm(dim=-1) + d.max(dim=-1).values.clamp(max=0) - rr


def capsule(p, a, b, r):
    a = torch.as_tensor(a, device=p.device, dtype=p.dtype)
    b = torch.as_tensor(b, device=p.device, dtype=p.dtype)
    pa, ba = p - a, b - a
    h = ((pa * ba).sum(-1) / (ba * ba).sum()).clamp(0, 1)
    return (pa - ba * h[..., None]).norm(dim=-1) - r


def ellipsoid(p, r):
    r = torch.as_tensor(r, device=p.device, dtype=p.dtype)
    k0 = (p / r).norm(dim=-1)
    k1 = (p / (r * r)).norm(dim=-1)
    return k0 * (k0 - 1) / k1.clamp(min=1e-9)


def torus_z(p, R, r):
    q = torch.stack([p[..., :2].norm(dim=-1) - R, p[..., 2]], dim=-1)
    return q.norm(dim=-1) - r


# ---------------------------------------------------------------- operators

def smin(a, b, k):
    h = (0.5 + 0.5 * (b - a) / k).clamp(0, 1)
    return b * (1 - h) + a * h - k * h * (1 - h)


def rot_z(p, a):
    c, s = math.cos(a), math.sin(a)
    return torch.stack([c * p[..., 0] + s * p[..., 1], -s * p[..., 0] + c * p[..., 1], p[..., 2]], dim=-1)


def rot_x(p, a):
    c, s = math.cos(a), math.sin(a)
    return torch.stack([p[..., 0], c * p[..., 1] + s * p[..., 2], -s * p[..., 1] + c * p[..., 2]], dim=-1)


def rot_y(p, a):
    c, s = math.cos(a), math.sin(a)
    return torch.stack([c * p[..., 0] - s * p[..., 2], p[..., 1], s * p[..., 0] + c * p[..., 2]], dim=-1)


def lying(p):
    """Map a vertical (z-axis) primitive onto the x axis: lay it on its side."""
    return torch.stack([p[..., 2], p[..., 1], -p[..., 0]], dim=-1)
