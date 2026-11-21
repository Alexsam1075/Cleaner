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


# ---------------------------------------------------------------- parts and shape fits

@dataclass
class Part:
    uv: np.ndarray                 # (N, 2) floor coordinates
    h: np.ndarray                  # (N,) heights
    s_range: tuple[float, float]   # extent along the object's main axis
    axis: np.ndarray               # (2,) unit main axis on the floor
    center: np.ndarray             # (2,) object centre used for s/t coordinates


@dataclass
class PartFit:
    kind: str
    volume_ml: float
    residual: float


SOLID_KINDS = ("box", "cylinder_lying", "cylinder_standing", "ellipsoid")
BULK_KINDS = ("dome", "ellipsoid", "box")


def _main_axis(uv: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    center = np.median(uv, axis=0)
    if len(uv) < 3:
        return np.array([1.0, 0.0]), center
    cov = np.cov((uv - center).T)
    w, v = np.linalg.eigh(cov)
    return v[:, int(np.argmax(w))], center


def split_parts(cloud: ObjectCloud, n_slices: int = 16, merge_tol: float = 0.30,
                min_slices: int = 2) -> list[Part]:
    """Cut the object along its main axis; join neighbouring slices with a similar profile.

    Profile = (width across the axis, p95 height). A new part starts where the width or the
    height jumps by more than ``merge_tol`` of the object's maximum width/height.
    """
    if len(cloud.h) == 0:
        return []
    axis, center = _main_axis(cloud.uv)
    perp = np.array([-axis[1], axis[0]])
    rel = cloud.uv - center
    s, t = rel @ axis, rel @ perp
    lo, hi = np.quantile(s, [0.005, 0.995])
    edges = np.linspace(lo, hi + 1e-9, n_slices + 1)
    idx = np.clip(np.searchsorted(edges, s, side="right") - 1, 0, n_slices - 1)
    prof = []
    for k in range(n_slices):
        sel = idx == k
        if sel.sum() >= 5:
            q = np.quantile(t[sel], [0.05, 0.95])
            prof.append((q[1] - q[0], float(np.quantile(cloud.h[sel], 0.95))))
        else:
            prof.append(None)
    known = [p for p in prof if p is not None]
    if not known:
        return [Part(cloud.uv, cloud.h, (float(lo), float(hi)), axis, center)]
    wmax = max(p[0] for p in known) or 1e-6
    hmax = max(p[1] for p in known) or 1e-6
    groups: list[list[int]] = []
    last = None
    for k, p in enumerate(prof):
        if p is None:                       # sparse slice: keep with the current part
            if groups:
                groups[-1].append(k)
            else:
                groups.append([k])
            continue
        if last is None or (abs(p[0] - last[0]) / wmax <= merge_tol and abs(p[1] - last[1]) / hmax <= merge_tol):
            if groups and last is not None:
                groups[-1].append(k)
            elif groups:
                groups[-1].append(k)
            else:
                groups.append([k])
        else:
            groups.append([k])
        last = p
    # fold too-short parts into the neighbour with the closer profile
    changed = True
    while changed and len(groups) > 1:
        changed = False
        for g_i, g in enumerate(groups):
            if len(g) < min_slices:
                j = g_i - 1 if g_i == len(groups) - 1 else (g_i + 1 if g_i == 0 else
                    (g_i - 1 if len(groups[g_i - 1]) >= len(groups[g_i + 1]) else g_i + 1))
                groups[j] = sorted(groups[j] + g)
                del groups[g_i]
                changed = True
                break
    parts = []
    for g in groups:
        sel = np.isin(idx, g)
        if sel.sum() == 0:
            continue
        parts.append(Part(cloud.uv[sel], cloud.h[sel], (float(edges[g[0]]), float(edges[g[-1] + 1])), axis, center))
    return parts


def _cap_height(rho: np.ndarray, r: float, H: float) -> np.ndarray:
    R = (r * r + H * H) / (2 * H)
    return np.where(rho <= r, np.sqrt(np.maximum(R * R - rho * rho, 0.0)) - (R - H), 0.0)


def fit_part(part: Part, kinds: tuple[str, ...], pad: float = 0.0) -> PartFit:
    """Fit each candidate shape to the part's height cells; ``pad`` = one cell width."""
    perp = np.array([-part.axis[1], part.axis[0]])
    rel = part.uv - part.center
    s, t = rel @ part.axis, rel @ perp
    L = max(part.s_range[1] - part.s_range[0] + pad, 1e-4)
    s0 = 0.5 * (part.s_range[0] + part.s_range[1])
    tq = np.quantile(t, [0.005, 0.995])
    w = max(float(tq[1] - tq[0]) + pad, 1e-4)
    t0 = float(tq.mean())
    H = max(float(np.quantile(part.h, 0.95)), 1e-4)
    ds, dt = s - s0, t - t0
    a, b = L / 2, w / 2
    best = None
    for kind in kinds:
        if kind == "box":
            pred = np.full_like(part.h, H)
            vol = L * w * H
            area = L * w
        elif kind == "cylinder_lying":
            r = 0.5 * max(H, 1e-4)
            pred = r + np.sqrt(np.maximum(r * r - dt * dt, 0.0))
            vol = np.pi * r * r * L
            area = L * 2 * r
        elif kind == "cylinder_standing":
            r = 0.25 * (L + w)
            pred = np.where(ds * ds + dt * dt <= r * r, H, 0.0)
            vol = np.pi * r * r * H
            area = np.pi * r * r
        elif kind == "ellipsoid":            # full ellipsoid lying on the floor
            c = H / 2
            q = 1 - (ds / a) ** 2 - (dt / b) ** 2
            pred = c + c * np.sqrt(np.maximum(q, 0.0))
            vol = 4.0 / 3.0 * np.pi * a * b * c
            area = np.pi * a * b
        elif kind == "dome":                 # elliptic spherical cap
            r = np.sqrt(a * b)
            rho = np.sqrt((ds / a) ** 2 + (dt / b) ** 2) * r
            pred = _cap_height(rho, r, H)
            vol = np.pi * H * (3 * a * b + H * H) / 6.0
            area = np.pi * a * b
        else:
            raise ValueError(kind)
        area_obs = max(len(part.h) * pad * pad, 1e-8) if pad > 0 else area
        res = float(np.median(np.abs(part.h - pred))) + H * abs(area - area_obs) / area_obs
        fit = PartFit(kind, float(vol * 1e6), res)
        if best is None or res < best.residual:
            best = fit
    return best


def heightmap(cloud: ObjectCloud, cell_m: float = 0.004) -> ObjectCloud:
    """Top surface: one point per floor cell (cell centre, max height). Walls collapse into edges."""
    if len(cloud.h) == 0:
        return cloud
    ij = np.floor(cloud.uv / cell_m).astype(np.int64)
    keys, inv = np.unique(ij, axis=0, return_inverse=True)
    inv = inv.ravel()
    top = np.zeros(len(keys))
    np.maximum.at(top, inv, cloud.h)
    return ObjectCloud(uv=(keys + 0.5) * cell_m, h=top, footprint_uv=cloud.footprint_uv,
                       valid_frac=cloud.valid_frac, truncated=cloud.truncated)


def complete_symmetric(hm: ObjectCloud, cell_m: float = 0.004) -> ObjectCloud:
    """Fill the side hidden from the camera by mirroring near-side cells across the ridge line.

    Ridge = cells within 10% of the top height. Its direction comes from PCA; a round ridge
    (no clear direction) uses the side axis ``v``. Only cells on the camera side (-u) of the
    ridge are mirrored, and only into empty cells.
    """
    if len(hm.h) < 3:
        return hm
    hi = hm.uv[hm.h >= 0.9 * hm.h.max()]
    p0 = hi.mean(axis=0)
    direction = np.array([0.0, 1.0])
    if len(hi) >= 3:
        w, v = np.linalg.eigh(np.cov((hi - p0).T))
        if w[1] > 2.0 * max(w[0], 1e-12):
            direction = v[:, 1]
    normal = np.array([-direction[1], direction[0]])
    if normal[0] > 0:
        normal = -normal                       # point toward the camera (-u)
    if abs(normal[0]) < 0.3:                   # ridge points at the camera: both sides seen
        return hm
    dist = (hm.uv - p0) @ normal
    near = dist > cell_m / 2
    mirrored = hm.uv[near] - 2.0 * dist[near, None] * normal
    occupied = {tuple(k) for k in np.floor(hm.uv / cell_m).astype(np.int64)}
    keys = np.floor(mirrored / cell_m).astype(np.int64)
    add_uv, add_h, seen = [], [], set()
    for k, hh in zip(map(tuple, keys), hm.h[near]):
        if k in occupied or k in seen:
            continue
        seen.add(k)
        add_uv.append((np.asarray(k, dtype=np.float64) + 0.5) * cell_m)
        add_h.append(hh)
    if not add_uv:
        return hm
    return ObjectCloud(uv=np.concatenate([hm.uv, np.asarray(add_uv)]), h=np.concatenate([hm.h, add_h]),
                       footprint_uv=hm.footprint_uv, valid_frac=hm.valid_frac, truncated=hm.truncated)


def parts_volume_ml(cloud: ObjectCloud, form: str, cell_m: float = 0.004) -> tuple[float, list[PartFit]]:
    kinds = BULK_KINDS if form == "bulk" else SOLID_KINDS
    hm = complete_symmetric(heightmap(cloud, cell_m), cell_m)
    fits = [fit_part(p, kinds, pad=cell_m) for p in split_parts(hm)]
    return float(sum(f.volume_ml for f in fits)), fits


# ---------------------------------------------------------------- single entry point

def _area_2d(pts: np.ndarray) -> float:
    if len(pts) < 3:
        return 0.0
    from scipy.spatial import ConvexHull, QhullError
    try:
        return float(ConvexHull(pts).volume)
    except QhullError:
        return 0.0


def _hull_ml(hm: ObjectCloud) -> float:
    if len(hm.h) < 4:
        return 0.0
    from scipy.spatial import ConvexHull, QhullError
    top = np.column_stack([hm.uv, hm.h])
    base = np.column_stack([hm.uv, np.zeros(len(hm.h))])
    try:
        return float(ConvexHull(np.vstack([top, base])).volume * 1e6)
    except QhullError:
        return 0.0


def estimate_volume(depth_mm: np.ndarray, mask: np.ndarray, plane: np.ndarray, camera: dict,
                    form: str, class_height_m: float | None = None, cell_m: float = 0.004,
                    erode_px: int = 2, min_h_m: float = 0.003) -> dict:
    """Several geometric volume estimates (ml, each >= 1) plus diagnostics for one object."""
    cloud = clean_cloud(depth_mm, mask, plane, camera, erode_px=erode_px, min_h_m=min_h_m)
    footprint_m2 = _area_2d(cloud.footprint_uv)
    out = {"valid_frac": cloud.valid_frac, "truncated": cloud.truncated, "footprint_m2": footprint_m2}
    if len(cloud.h) < 20:
        h = class_height_m if class_height_m else 0.05
        v = max(footprint_m2 * h * 0.6 * 1e6, 1.0)
        out.update(columns_ml=v, parts_ml=v, hull_ml=v, footprint_ml=v, n_parts=0, p95_h_m=0.0)
        return out
    p95 = float(np.quantile(cloud.h, 0.95))
    hm = complete_symmetric(heightmap(cloud, cell_m), cell_m)
    parts_ml, fits = parts_volume_ml(cloud, form, cell_m)
    out.update(
        columns_ml=max(columns_volume_ml(cloud, cell_m), 1.0),
        parts_ml=max(parts_ml, 1.0),
        hull_ml=max(_hull_ml(hm), 1.0),
        footprint_ml=max(footprint_m2 * p95 * 1e6, 1.0),
        n_parts=len(fits),
        p95_h_m=p95,
    )
    return out
