"""Calibrated parts-based volume for one object mask; the material is an input, not a guess."""
from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

try:  # package import
    from ..evaluation.volume_parts_cv import FORMS, MATERIALS, apply_calibrator
    from ..geometry.volume_parts import estimate_volume, refine_plane
    from ..inference.volume_backend import density_for, load_densities
except ImportError:  # run from the repo root
    from evaluation.volume_parts_cv import FORMS, MATERIALS, apply_calibrator
    from geometry.volume_parts import estimate_volume, refine_plane
    from inference.volume_backend import density_for, load_densities


@dataclass
class VolumeResult:
    volume_ml: float
    mass_g: float
    density_g_per_ml: float
    geometry: dict = field(default_factory=dict)   # raw geometric candidates and diagnostics


class VolumeEstimator:
    """Volume (ml) and mass (g) of one object from depth + mask + floor plane + form + material.

    The calibration depends on the material: with the true material grouped-OOF V is 0.764,
    so a better material classifier upstream directly improves the volume. When only class
    probabilities are known, pass ``material_probs``: the volume is the probability-weighted
    geometric mean over materials and the mass uses the expected density.
    """

    def __init__(self, calibrator: dict, densities: dict[tuple[str, str], float], camera: dict):
        self.calibrator, self.densities, self.camera = calibrator, densities, camera

    @classmethod
    def load(cls, calibrator_path: str | Path, densities_path: str | Path, camera: dict | str | Path):
        cal = json.loads(Path(calibrator_path).read_text(encoding="utf-8"))
        if not isinstance(camera, dict):
            camera = json.loads(Path(camera).read_text(encoding="utf-8"))
        return cls(cal, load_densities(densities_path), camera)

    def geometry(self, depth_mm: np.ndarray, mask: np.ndarray, plane: np.ndarray, form: str) -> dict:
        refined = refine_plane(depth_mm, mask, plane, self.camera)
        return estimate_volume(depth_mm, np.asarray(mask, bool), refined, self.camera, form)

    def estimate(self, depth_mm: np.ndarray, mask: np.ndarray, plane: np.ndarray, form: str,
                 material: str | None = None, material_probs: dict[str, float] | None = None,
                 geometry: dict | None = None) -> VolumeResult:
        if form not in FORMS:
            raise ValueError(f"unknown form {form!r}; expected one of {FORMS}")
        probs = self._material_weights(material, material_probs)
        geo = geometry if geometry is not None else self.geometry(depth_mm, mask, plane, form)
        log_v = sum(p * math.log(apply_calibrator(self.calibrator, geo, form, m)) for m, p in probs.items())
        density = sum(p * density_for(self.densities, form, m) for m, p in probs.items())
        volume = float(math.exp(log_v))
        return VolumeResult(volume, volume * density, density, geo)

    @staticmethod
    def _material_weights(material, material_probs) -> dict[str, float]:
        if material_probs:
            bad = [m for m in material_probs if m not in MATERIALS]
            if bad:
                raise ValueError(f"unknown materials {bad}; expected {MATERIALS}")
            total = float(sum(max(p, 0.0) for p in material_probs.values()))
            if total <= 0:
                raise ValueError("material_probs must have a positive sum")
            return {m: max(p, 0.0) / total for m, p in material_probs.items() if p > 0}
        if material not in MATERIALS:
            raise ValueError(f"unknown material {material!r}; expected one of {MATERIALS}")
        return {material: 1.0}
