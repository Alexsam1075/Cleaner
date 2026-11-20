from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import numpy as np


def draw_predictions(rgb: np.ndarray, predictions: list[dict[str, Any]]) -> np.ndarray:
    out = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR).copy()
    palette = {"bulk": (35, 190, 255), "solid": (255, 150, 40)}
    overlay = out.copy()
    for pred in predictions:
        mask = pred.get("mask")
        color = palette.get(pred.get("form"), (50, 230, 50))
        if mask is not None and mask.shape == out.shape[:2]:
            overlay[np.asarray(mask, dtype=bool)] = color
    out = cv2.addWeighted(out, 0.72, overlay, 0.28, 0)
    for pred in predictions:
        x1, y1, x2, y2 = [int(round(x)) for x in pred["box"]]
        color = palette.get(pred.get("form"), (50, 230, 50))
        cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
        geometry = pred.get("geometry", {})
        raw_volume = geometry.get("geometric_volume_ml")
        raw_text = "NA" if raw_volume is None else f"{raw_volume:.0f}ml"
        text = (f"{pred.get('form')} {pred.get('material')} "
                f"V={pred.get('volume_ml', 0):.0f}ml geom={raw_text} "
                f"M={pred.get('mass_g', 0):.0f}g take={pred.get('take', 0)}")
        baseline = min(max(y1 - 8, 18), out.shape[0] - 6)
        cv2.putText(out, text, (x1, baseline), cv2.FONT_HERSHEY_SIMPLEX,
                    0.42, color, 2, cv2.LINE_AA)
    return out


def save_overlay(image_path: str | Path, predictions: list[dict[str, Any]],
                 output_path: str | Path) -> Path:
    image_path, output_path = Path(image_path), Path(output_path)
    bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise FileNotFoundError(image_path)
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    rendered = draw_predictions(rgb, predictions)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(output_path), rendered):
        raise OSError(f"Could not write visualization: {output_path}")
    return output_path

