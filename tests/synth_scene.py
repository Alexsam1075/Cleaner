"""Test utility: render a synthetic RGB-D frame (depth + masks) of simple shapes on a floor.

Shapes are described as a heightfield over the floor plus vertical walls at the heightfield
border, which is how a camera looking down at the floor sees solid objects. Coordinates
``(u, v)`` are on the floor: ``u`` points forward from the camera, ``v`` to the side.
"""
from __future__ import annotations

import numpy as np

CAMERA = {
    "width": 1280, "height": 720,
    "fx": 911.6585693359375, "fy": 909.689697265625,
    "cx": 632.4738159179688, "cy": 345.352783203125,
}
PLANE = np.array([-0.002392542193282726, -0.8876295939720227, -0.4605518208051247, 0.2514167226874536])

STEP_M = 0.0004


def floor_frame(plane: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return (origin, e_forward, e_side, up) of the floor in camera coordinates."""
    norm = float(np.linalg.norm(plane[:3]))
    n, d = plane[:3] / norm, plane[3] / norm
    up = n if d > 0 else -n                      # points toward the camera
    origin = -d * n                              # foot of the camera on the floor
    fwd = np.array([0.0, 0.0, 1.0]) - up * up[2]
    fwd /= np.linalg.norm(fwd)
    side = np.cross(up, fwd)
    return origin, fwd, side, up


def _heightfield(shape: dict, uu: np.ndarray, vv: np.ndarray) -> np.ndarray:
    cu, cv = shape["center_uv"]
    yaw = shape.get("yaw", 0.0)
    du, dv = uu - cu, vv - cv
    a = np.cos(yaw) * du + np.sin(yaw) * dv      # along the shape axis
    b = -np.sin(yaw) * du + np.cos(yaw) * dv     # across
    h = np.full(uu.shape, np.nan)
    kind = shape["kind"]
    if kind == "box":
        L, W, H = shape["size"]
        inside = (np.abs(a) <= L / 2) & (np.abs(b) <= W / 2)
        h[inside] = H
    elif kind == "cylinder_lying":
        r, L = shape["radius"], shape["length"]
        inside = (np.abs(a) <= L / 2) & (np.abs(b) <= r)
        h[inside] = r + np.sqrt(np.maximum(r * r - b[inside] ** 2, 0.0))
    elif kind == "cylinder_standing":
        r, H = shape["radius"], shape["length"]
        inside = a * a + b * b <= r * r
        h[inside] = H
    elif kind == "dome":                         # spherical cap
        r, H = shape["radius"], shape["height"]
        R = (r * r + H * H) / (2 * H)
        rho2 = a * a + b * b
        inside = rho2 <= r * r
        h[inside] = np.sqrt(R * R - rho2[inside]) - (R - H)
    else:
        raise ValueError(kind)
    return h


def _bounds(shape: dict) -> float:
    if shape["kind"] == "box":
        return 0.5 * float(np.hypot(shape["size"][0], shape["size"][1])) + 0.002
    if shape["kind"] == "dome":
        return shape["radius"] + 0.002
    return 0.5 * float(np.hypot(shape["length"], 2 * shape["radius"])) + 0.002


def _surface_points(shape: dict) -> np.ndarray:
    ext = _bounds(shape)
    cu, cv = shape["center_uv"]
    us = np.arange(cu - ext, cu + ext, STEP_M)
    vs = np.arange(cv - ext, cv + ext, STEP_M)
    uu, vv = np.meshgrid(us, vs, indexing="ij")
    h = _heightfield(shape, uu, vv)
    ok = np.isfinite(h)
    pts = [np.stack([uu[ok], vv[ok], h[ok]], axis=1)]
    pad = np.pad(ok, 1, constant_values=False)
    border = ok & ~(pad[:-2, 1:-1] & pad[2:, 1:-1] & pad[1:-1, :-2] & pad[1:-1, 2:])
    for u, v, top in zip(uu[border], vv[border], h[border]):
        z = np.arange(0.0, top, STEP_M)
        pts.append(np.stack([np.full_like(z, u), np.full_like(z, v), z], axis=1))
    return np.concatenate(pts)


def render(camera: dict, plane: np.ndarray, shapes: list[dict], noise_mm: float = 1.0,
           seed: int = 0) -> tuple[np.ndarray, list[np.ndarray]]:
    """Return depth (H×W uint16, mm) and one boolean mask per shape."""
    H, W = int(camera["height"]), int(camera["width"])
    fx, fy, cx, cy = (float(camera[k]) for k in ("fx", "fy", "cx", "cy"))
    norm = float(np.linalg.norm(plane[:3]))
    n, d = plane[:3] / norm, plane[3] / norm
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float64)
    rx, ry = (xx - cx) / fx, (yy - cy) / fy
    denom = n[0] * rx + n[1] * ry + n[2]
    with np.errstate(divide="ignore", invalid="ignore"):
        t = -d / denom
    zbuf = np.where(t > 0, t, np.inf).ravel()
    ids = np.full(H * W, -1, dtype=np.int32)
    origin, fwd, side, up = floor_frame(plane)
    for k, shape in enumerate(shapes):
        p = _surface_points(shape)
        P = origin + p[:, :1] * fwd + p[:, 1:2] * side + p[:, 2:3] * up
        z = P[:, 2]
        px = np.round(P[:, 0] / z * fx + cx).astype(np.int64)
        py = np.round(P[:, 1] / z * fy + cy).astype(np.int64)
        ok = (z > 0) & (px >= 0) & (px < W) & (py >= 0) & (py < H)
        flat, z = py[ok] * W + px[ok], z[ok]
        before = zbuf.copy()
        np.minimum.at(zbuf, flat, z)
        won = zbuf[flat] < before[flat]
        ids[np.unique(flat[won])] = k
    rng = np.random.default_rng(seed)
    depth = zbuf.reshape(H, W) * 1000.0
    depth = depth + rng.normal(0.0, noise_mm, depth.shape)
    depth = np.where(np.isfinite(depth), np.clip(np.round(depth), 0, 65535), 0).astype(np.uint16)
    masks = [(ids == k).reshape(H, W) for k in range(len(shapes))]
    return depth, masks
