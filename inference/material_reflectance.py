from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import numpy as np


# Experimental metal cue calibrated from RGB highlight and depth dropout.
# It is deliberately opt-in because the development folds are small.
MIN_DEPTH_HOLE_FRACTION = 0.25
MIN_BRIGHT_LOW_SAT_FRACTION = 0.05


def apply_reflective_metal_override(
    image_path: str | Path,
    predictions: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Override material only when a detection has strong RGB glare and depth holes.

    Other fields, including geometry, score, volume, mass and take, are copied
    unchanged. Detections without a valid segmentation mask are left untouched.
    """
    if not predictions:
        return predictions

    image_path = Path(image_path)
    bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    depth = cv2.imread(str(image_path.parent / "depth.png"), cv2.IMREAD_UNCHANGED)
    if bgr is None or depth is None:
        raise FileNotFoundError(f"RGB/depth pair missing for {image_path}")
    if depth.ndim == 3 and depth.shape[2] == 1:
        depth = depth[:, :, 0]
    if depth.shape != bgr.shape[:2]:
        raise ValueError(f"RGB/depth shape mismatch for {image_path}: {bgr.shape[:2]} vs {depth.shape}")

    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    bright_low_sat = (hsv[..., 2] >= 238) & (hsv[..., 1] <= 50)
    valid_depth = np.isfinite(depth) & (depth > 50) & (depth < 8000)

    updated = [dict(prediction) for prediction in predictions]
    for prediction in updated:
        mask = prediction.get("mask")
        if mask is None:
            continue
        mask = np.asarray(mask, dtype=bool)
        if mask.shape != depth.shape:
            continue
        object_pixels = int(mask.sum())
        if object_pixels == 0:
            continue

        hole_fraction = float((mask & ~valid_depth).sum()) / object_pixels
        glare_fraction = float((mask & bright_low_sat).sum()) / object_pixels
        if (hole_fraction >= MIN_DEPTH_HOLE_FRACTION and
                glare_fraction >= MIN_BRIGHT_LOW_SAT_FRACTION):
            prediction["material"] = "metal"

    return updated
