"""Object volume from RGB-D: clean the point cloud, split it into parts, fit a shape per part.

All lengths are metres unless the name says otherwise; volumes are millilitres.
Floor coordinates: ``u`` forward from the camera along the floor, ``v`` to the side,
``h`` height above the floor (always >= 0, pointing toward the camera).
"""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

MIN_DEPTH_MM, MAX_DEPTH_MM = 50.0, 8000.0


@dataclass
class ObjectCloud:
    uv: np.ndarray            # (N, 2) floor coordinates of object points
    h: np.ndarray             # (N,) height above the floor, >= 0
    footprint_uv: np.ndarray  # (K, 2) mask contour cast onto the floor
    valid_frac: float         # valid-depth pixels / mask pixels
    truncated: bool           # mask touches the image border


def _unit_plane(plane: np.ndarray) -> tuple[np.ndarray, float]:
    plane = np.asarray(plane, dtype=np.float64)
    norm = max(float(np.linalg.norm(plane[:3])), 1e-12)
    n, d = plane[:3] / norm, float(plane[3]) / norm
    if d < 0:                 # orient the normal toward the camera (origin on the + side)
        n, d = -n, -d
    return n, d


def _floor_axes(n: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    fwd = np.array([0.0, 0.0, 1.0]) - n * n[2]
    if np.linalg.norm(fwd) < 1e-6:
        fwd = np.array([1.0, 0.0, 0.0]) - n * n[0]
    fwd /= np.linalg.norm(fwd)
    return fwd, np.cross(n, fwd)


def _backproject(depth_mm: np.ndarray, sel: np.ndarray, camera: dict) -> np.ndarray:
    ys, xs = np.nonzero(sel)
    z = depth_mm[ys, xs].astype(np.float64) / 1000.0
    x = (xs - float(camera["cx"])) * z / float(camera["fx"])
    y = (ys - float(camera["cy"])) * z / float(camera["fy"])
    return np.stack([x, y, z], axis=1)


def _valid_depth(depth_mm: np.ndarray) -> np.ndarray:
    d = np.asarray(depth_mm, dtype=np.float64)
    return np.isfinite(d) & (d > MIN_DEPTH_MM) & (d < MAX_DEPTH_MM)


def refine_plane(depth_mm: np.ndarray, mask: np.ndarray, plane: np.ndarray, camera: dict,
                 ring_px: int = 25) -> np.ndarray:
    """Re-fit the floor plane on a ring of pixels around the mask (unit normal, toward camera)."""
    n0, d0 = _unit_plane(plane)
    m = np.asarray(mask, dtype=np.uint8)
    outer = cv2.dilate(m, np.ones((2 * ring_px + 1,) * 2, np.uint8))
    inner = cv2.dilate(m, np.ones((11, 11), np.uint8))
    ring = (outer > 0) & (inner == 0) & _valid_depth(depth_mm)
    if ring.sum() < 200:
        return np.concatenate([n0, [d0]])
    P = _backproject(depth_mm, ring, camera)
    for tol in (0.010, 0.004):
        dist = P @ n0 + d0
        inl = P[np.abs(dist) < tol]
        if len(inl) < 200:
            return np.concatenate([n0, [d0]])
        c = inl.mean(axis=0)
        n = np.linalg.svd(inl - c, full_matrices=False)[2][-1]
        if n @ n0 < 0:
            n = -n
        n0, d0 = n, -float(n @ c)
    n0, d0 = _unit_plane(np.concatenate([n0, [d0]]))
    return np.concatenate([n0, [d0]])


def _floor_hit(xs: np.ndarray, ys: np.ndarray, n: np.ndarray, d: float, camera: dict) -> np.ndarray:
    r = np.stack([(xs - float(camera["cx"])) / float(camera["fx"]),
                  (ys - float(camera["cy"])) / float(camera["fy"]),
                  np.ones(len(xs))], axis=1)
    denom = r @ n
    t = np.where(np.abs(denom) > 1e-9, -d / np.where(np.abs(denom) > 1e-9, denom, 1.0), np.nan)
    pts = r * t[:, None]
    return pts[np.isfinite(t) & (t > 0)]


def clean_cloud(depth_mm: np.ndarray, mask: np.ndarray, plane: np.ndarray, camera: dict,
                erode_px: int = 2, min_h_m: float = 0.003) -> ObjectCloud:
    """Object points above the floor with edge and floor noise removed."""
    mask = np.asarray(mask, dtype=bool)
    if depth_mm.shape != mask.shape:
        raise ValueError(f"depth/mask shape mismatch: {depth_mm.shape} vs {mask.shape}")
    n, d = _unit_plane(plane)
    fwd, side = _floor_axes(n)
    H, W = mask.shape
    n_mask = int(mask.sum())
    truncated = bool(n_mask and (mask[0].any() or mask[-1].any() or mask[:, 0].any() or mask[:, -1].any()))

    core = mask
    if erode_px > 0 and n_mask:
        er = cv2.erode(mask.astype(np.uint8), np.ones((2 * erode_px + 1,) * 2, np.uint8)) > 0
        if er.sum() >= 0.3 * n_mask:
            core = er
    valid = mask & _valid_depth(depth_mm)
    valid_frac = float(valid.sum() / max(n_mask, 1))

    P = _backproject(depth_mm, core & valid, camera)
    h = P @ n + d
    keep = h >= min_h_m
    P, h = P[keep], h[keep]
    flat = P - h[:, None] * n
    uv = np.stack([flat @ fwd, flat @ side], axis=1)

    footprint = np.zeros((0, 2))
    if n_mask:
        contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        if contours:
            pix = np.concatenate([c.reshape(-1, 2) for c in contours]).astype(np.float64)
            hits = _floor_hit(pix[:, 0], pix[:, 1], n, d, camera)
            footprint = np.stack([hits @ fwd, hits @ side], axis=1) if len(hits) else footprint
    return ObjectCloud(uv=uv, h=h, footprint_uv=footprint, valid_frac=valid_frac, truncated=truncated)


def columns_volume_ml(cloud: ObjectCloud, cell_m: float = 0.004) -> float:
    """Sum of per-cell maximum heights over the floor grid (2.5D column integral)."""
    if len(cloud.h) == 0:
        return 0.0
    ij = np.floor((cloud.uv - cloud.uv.min(axis=0)) / cell_m).astype(np.int64)
    key = ij[:, 0] * (int(ij[:, 1].max()) + 1) + ij[:, 1]
    _, inv = np.unique(key, return_inverse=True)
    top = np.zeros(int(inv.max()) + 1)
    np.maximum.at(top, inv, cloud.h)
    return float(top.sum() * cell_m * cell_m * 1e6)
