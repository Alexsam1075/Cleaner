"""Experimental observations from an aligned RGB-D object mask."""

import cv2
import numpy as np


def observe(rgb, depth, mask, plane, camera, geometry, box_fallback):
    ys, xs = np.nonzero(mask)
    if not xs.size:
        return {"mask_source": "empty"}
    region = np.s_[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
    cropped_mask = mask[region]
    pixels = rgb[ys, xs].astype(np.float32) / 255
    gray = cv2.cvtColor(rgb[region], cv2.COLOR_RGB2GRAY).astype(np.float32) / 255
    interior = cv2.erode(cropped_mask.astype(np.uint8), np.ones((3, 3), np.uint8),
                         borderType=cv2.BORDER_CONSTANT, borderValue=0).astype(bool)
    laplacian = cv2.Laplacian(gray, cv2.CV_32F)
    highlights = (pixels.max(axis=1) >= 0.95) & (np.ptp(pixels, axis=1) <= 0.12)
    z = depth[ys, xs].astype(np.float64) / 1000
    valid = np.isfinite(z) & (z > 0.05) & (z < 8.0)
    count, _, stats, _ = cv2.connectedComponentsWithStats(cropped_mask.astype(np.uint8))
    areas = stats[1:count, cv2.CC_STAT_AREA]
    observation = {
        "mask_source": "box" if box_fallback else "segmentation",
        "luminance_std": float(gray[cropped_mask].std()),
        "texture_laplacian_std": float(laplacian[interior].std()) if interior.any() else None,
        "bright_low_chroma_fraction": float(highlights.mean()),
        "highlight_invalid_depth_fraction": float((~valid[highlights]).mean()) if highlights.any() else None,
        "visible_mask_components": None if box_fallback else int(np.sum(areas >= max(9, xs.size * 0.01))),
        "geometric_volume_ml": geometry.get("geometric_volume_ml"),
        "valid_depth_fraction": float(valid.mean()),
        "floor_geometry": None,
    }
    normal_length = np.linalg.norm(plane[:3])
    if valid.sum() < 3 or not np.isfinite(normal_length) or normal_length <= 1e-8:
        return observation
    normal = plane[:3] / normal_length
    offset = plane[3] / normal_length
    if offset < 0:
        normal, offset = -normal, -offset
    axis = np.cross(normal, [0., 0., 1.])
    if np.linalg.norm(axis) < 1e-6:
        axis = np.cross(normal, [1., 0., 0.])
    axis /= np.linalg.norm(axis)
    forward = np.cross(normal, axis)
    z = z[valid]
    xyz = np.column_stack(((xs[valid] - camera["cx"]) * z / camera["fx"],
                           (ys[valid] - camera["cy"]) * z / camera["fy"], z))
    signed_height = xyz @ normal + offset
    uv = xyz @ np.stack((axis, forward), axis=1)
    center = np.median(uv, axis=0)
    centered = uv - center
    _, directions = np.linalg.eigh(centered.T @ centered)
    bounds = np.quantile(centered @ directions, [0.05, 0.95], axis=0)
    width, length = sorted(bounds[1] - bounds[0])
    observation["floor_geometry"] = {
        "center_uv_m": center.tolist(),
        "normal_camera": normal.tolist(),
        "offset_m": float(offset),
        "u_axis_camera": axis.tolist(),
        "v_axis_camera": forward.tolist(),
        "width_p05_p95_m": float(width),
        "length_p05_p95_m": float(length),
        "height_mean_m": float(np.maximum(signed_height, 0).mean()),
        "height_p95_m": float(np.quantile(np.maximum(signed_height, 0), 0.95)),
        "below_plane_fraction": float(np.mean(signed_height < -0.005)),
    }
    return observation
