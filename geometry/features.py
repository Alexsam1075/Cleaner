from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

try:
    import cv2
except ImportError:  # pragma: no cover - Pillow fallback keeps inference portable
    cv2 = None


FEATURE_NAMES = [
    "valid_depth_fraction",
    "log_depth_median_m",
    "log_depth_spread_m",
    "height_mean_m",
    "height_median_m",
    "height_p95_m",
    "height_max_m",
    "log_footprint_mm2",
    "log_width_mm",
    "log_length_mm",
    "height_coverage_fraction",
    "log_visible_surface_cm2",
    "log_geometric_volume_ml",
    "height_roughness_mm",
    "mask_area_log_px",
    "bbox_aspect_ratio",
    "plane_residual_median_m",
    "valid_point_count_log",
]


def load_camera(camera_path: str | Path) -> dict[str, Any]:
    return json.loads(Path(camera_path).read_text(encoding="utf-8"))


def load_plane(meta_path: str | Path) -> np.ndarray:
    meta = json.loads(Path(meta_path).read_text(encoding="utf-8"))
    plane = np.asarray(meta["floor_plane"], dtype=np.float64)
    if plane.shape != (4,) or not np.isfinite(plane).all():
        raise ValueError(f"Invalid floor_plane in {meta_path}")
    return plane


def _resize(array: np.ndarray, size: int, nearest: bool = False) -> np.ndarray:
    if cv2 is not None:
        interp = cv2.INTER_NEAREST if nearest else cv2.INTER_AREA
        return cv2.resize(array, (size, size), interpolation=interp)
    from PIL import Image
    mode = Image.Resampling.NEAREST if nearest else Image.Resampling.BILINEAR
    return np.asarray(Image.fromarray(array.astype(np.float32)).resize((size, size), mode))


def _crop_bounds(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    ys, xs = np.where(mask)
    if len(xs) == 0:
        return None
    pad_x = max(2, int((xs.max() - xs.min() + 1) * 0.08))
    pad_y = max(2, int((ys.max() - ys.min() + 1) * 0.08))
    return (max(0, int(xs.min()) - pad_x), max(0, int(ys.min()) - pad_y),
            min(mask.shape[1], int(xs.max()) + pad_x + 1),
            min(mask.shape[0], int(ys.max()) + pad_y + 1))


def geometry_input(depth_mm: np.ndarray, mask: np.ndarray, plane: np.ndarray,
                   camera: dict[str, Any], size: int = 64) -> np.ndarray:
    """Return depth, valid-depth, and floor-relative-height crop channels."""
    mask = np.asarray(mask, dtype=bool)
    bounds = _crop_bounds(mask)
    if bounds is None:
        return np.zeros((3, size, size), dtype=np.float32)
    x1, y1, x2, y2 = bounds
    z = np.asarray(depth_mm[y1:y2, x1:x2], dtype=np.float32) / 1000.0
    m = mask[y1:y2, x1:x2]
    valid = m & np.isfinite(z) & (z > 0.05) & (z < 8.0)
    yy, xx = np.mgrid[y1:y2, x1:x2].astype(np.float32)
    X = (xx - float(camera["cx"])) * z / float(camera["fx"])
    Y = (yy - float(camera["cy"])) * z / float(camera["fy"])
    signed = (plane[0] * X + plane[1] * Y + plane[2] * z + plane[3]) / max(float(np.linalg.norm(plane[:3])), 1e-8)
    if valid.any():
        direction = 1.0 if float(np.median(signed[valid])) >= 0 else -1.0
        height = np.maximum(direction * signed, 0.0)
    else:
        height = np.zeros_like(z)
    depth_channel = np.where(valid, np.clip(z / 4.0, 0.0, 1.0), 0.0)
    height_channel = np.where(valid, np.clip(height / 0.4, 0.0, 1.0), 0.0)
    valid_channel = valid.astype(np.float32)
    channels = [
        _resize(depth_channel, size),
        _resize(valid_channel, size, nearest=True),
        _resize(height_channel, size),
    ]
    return np.stack(channels).astype(np.float32)


def compute_geometry_features(depth_mm: np.ndarray, mask: np.ndarray, plane: np.ndarray,
                              camera: dict[str, Any], cell_m: float = 0.004) -> tuple[np.ndarray, dict[str, float | None]]:
    """Measure observed RGB-D geometry; invalid depth stays explicitly missing."""
    mask = np.asarray(mask, dtype=bool)
    if mask.ndim != 2 or depth_mm.shape != mask.shape:
        raise ValueError(f"Depth/mask shape mismatch: {depth_mm.shape} vs {mask.shape}")
    bounds = _crop_bounds(mask)
    if bounds is None:
        empty = np.zeros(len(FEATURE_NAMES), dtype=np.float32)
        return empty, {"geometric_volume_ml": None, "valid_depth_fraction": 0.0}
    x1, y1, x2, y2 = bounds
    mc = mask[y1:y2, x1:x2]
    dc = np.asarray(depth_mm[y1:y2, x1:x2], dtype=np.float32)
    valid = mc & np.isfinite(dc) & (dc > 50.0) & (dc < 8000.0)
    n_mask = int(mc.sum())
    n_valid = int(valid.sum())
    if n_valid == 0:
        values = np.zeros(len(FEATURE_NAMES), dtype=np.float32)
        values[0] = 0.0
        values[14] = np.log1p(n_mask)
        ys, xs = np.where(mc)
        bw, bh = max(1, xs.max() - xs.min() + 1), max(1, ys.max() - ys.min() + 1)
        values[15] = np.log(max(bw / bh, 1e-6))
        return values, {"geometric_volume_ml": None, "valid_depth_fraction": 0.0}

    yy, xx = np.mgrid[y1:y2, x1:x2].astype(np.float64)
    z = dc / 1000.0
    X = (xx - float(camera["cx"])) * z / float(camera["fx"])
    Y = (yy - float(camera["cy"])) * z / float(camera["fy"])
    plane = np.asarray(plane, dtype=np.float64)
    normal = plane[:3] / max(float(np.linalg.norm(plane[:3])), 1e-10)
    residual = X * normal[0] + Y * normal[1] + z * normal[2] + plane[3] / max(float(np.linalg.norm(plane[:3])), 1e-10)
    direction = 1.0 if float(np.median(residual[valid])) >= 0.0 else -1.0
    signed_height = direction * residual
    heights = np.clip(signed_height[valid], 0.0, 1.0)
    xyz = np.stack((X[valid], Y[valid], z[valid]), axis=1)
    n_valid = int(len(heights))

    ref = np.array([0.0, 0.0, 1.0])
    e1 = np.cross(normal, ref)
    if np.linalg.norm(e1) < 1e-6:
        e1 = np.cross(normal, np.array([1.0, 0.0, 0.0]))
    e1 /= max(float(np.linalg.norm(e1)), 1e-10)
    e2 = np.cross(normal, e1)
    e2 /= max(float(np.linalg.norm(e2)), 1e-10)
    projected = xyz - heights[:, None] * normal[None, :]
    uv = np.stack((projected @ e1, projected @ e2), axis=1)
    origin = uv.min(axis=0)
    ij = np.floor((uv - origin) / max(cell_m, 1e-5)).astype(np.int64)
    iy_width = int(ij[:, 1].max()) + 1
    cell_ids = ij[:, 0] * max(iy_width, 1) + ij[:, 1]
    _, inverse = np.unique(cell_ids, return_inverse=True)
    cell_heights = np.zeros(int(inverse.max()) + 1, dtype=np.float64)
    np.maximum.at(cell_heights, inverse, heights)
    geometric_volume_ml = float(cell_heights.sum() * (cell_m ** 2) * 1_000_000.0)
    footprint_area_m2 = float(len(cell_heights) * (cell_m ** 2))
    extents = np.quantile(uv, 0.95, axis=0) - np.quantile(uv, 0.05, axis=0)

    # Approximate visible surface area from adjacent valid 3D samples only.
    pts = np.stack((X, Y, z), axis=-1)
    cell_ok = valid[:-1, :-1] & valid[:-1, 1:] & valid[1:, :-1] & valid[1:, 1:]
    if cell_ok.any():
        dx = pts[:-1, 1:] - pts[:-1, :-1]
        dy = pts[1:, :-1] - pts[:-1, :-1]
        area = np.linalg.norm(np.cross(dx, dy), axis=-1)
        visible_surface_m2 = float(np.sum(area[cell_ok]))
    else:
        visible_surface_m2 = 0.0

    depth_values = z[valid]
    hq = np.quantile(heights, [0.5, 0.95])
    ys, xs = np.where(mc)
    bw, bh = max(1, xs.max() - xs.min() + 1), max(1, ys.max() - ys.min() + 1)
    bbox_aspect = float(np.log(max(bw / bh, 1e-6)))
    positive = heights[heights > 0.002]
    roughness = float(np.std(positive)) if len(positive) else 0.0
    values = np.asarray([
        n_valid / max(n_mask, 1),
        np.log1p(float(np.median(depth_values))),
        np.log1p(float(np.std(depth_values))),
        float(np.mean(heights)),
        float(hq[0]),
        float(hq[1]),
        float(np.max(heights)),
        np.log1p(footprint_area_m2 * 1_000_000.0),
        np.log1p(max(float(extents[0]), 0.0) * 1000.0),
        np.log1p(max(float(extents[1]), 0.0) * 1000.0),
        float(np.mean(heights > 0.002)),
        np.log1p(visible_surface_m2 * 10000.0),
        np.log1p(max(geometric_volume_ml, 0.0)),
        roughness * 1000.0,
        np.log1p(n_mask),
        bbox_aspect,
        float(np.median(residual[valid])),
        np.log1p(n_valid),
    ], dtype=np.float32)
    values = np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)
    return values, {
        "geometric_volume_ml": geometric_volume_ml if geometric_volume_ml > 0 else None,
        "valid_depth_fraction": float(n_valid / max(n_mask, 1)),
        "footprint_area_m2": footprint_area_m2,
        "visible_surface_m2": visible_surface_m2,
        "height_p95_m": float(hq[1]),
        "valid_points": n_valid,
    }

