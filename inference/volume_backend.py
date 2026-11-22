"""Parts-based geometric volume for one detection, plus mass from the class density table."""
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

try:  # package import (predict.py / validate.py)
    from ..evaluation.volume_parts_cv import apply_calibrator
    from ..geometry.volume_parts import estimate_volume, refine_plane
except ImportError:  # top-level import (tests, scripts run from the repo root)
    from evaluation.volume_parts_cv import apply_calibrator
    from geometry.volume_parts import estimate_volume, refine_plane


def load_densities(path: str | Path) -> dict[tuple[str, str], float]:
    with open(path, newline="", encoding="utf-8") as f:
        return {(r["form"], r["material"]): float(r["density_g_per_ml"]) for r in csv.DictReader(f)}


def density_for(densities: dict[tuple[str, str], float], form: str, material: str) -> float:
    """Class density; unknown class falls back to the median of the same form, then of all."""
    if (form, material) in densities:
        return densities[(form, material)]
    same = [v for (f, _), v in densities.items() if f == form]
    return float(np.median(same if same else list(densities.values())))


def load_calibrator(path: str | Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def parts_volume_for_detection(depth_mm: np.ndarray, mask: np.ndarray, plane: np.ndarray, camera: dict,
                               form: str, material: str, calibrator: dict,
                               densities: dict[tuple[str, str], float]) -> tuple[float, float]:
    """Return (volume_ml, mass_g) for one predicted mask using the calibrated parts geometry."""
    refined = refine_plane(depth_mm, mask, plane, camera)
    est = estimate_volume(depth_mm, mask, refined, camera, form)
    volume_ml = apply_calibrator(calibrator, est, form, material)
    return volume_ml, volume_ml * density_for(densities, form, material)
