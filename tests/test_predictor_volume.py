import pytest

from evaluation.volume_parts_cv import fit_calibrator
from inference.volume_backend import density_for, load_densities, parts_volume_for_detection
from tests.synth_scene import CAMERA, PLANE, render
from tests.test_volume_cv import _rows

DENSITIES_CSV = "datasets/raw/densities.csv"
BOX = {"kind": "box", "center_uv": (0.5, 0.0), "size": (0.10, 0.05, 0.04), "yaw": 0.3}


def test_parts_backend_volume_and_mass():
    depth, masks = render(CAMERA, PLANE, [BOX])
    cal = fit_calibrator(_rows())
    dens = load_densities(DENSITIES_CSV)
    volume_ml, mass_g = parts_volume_for_detection(depth, masks[0], PLANE, CAMERA, "solid", "plastic", cal, dens)
    assert volume_ml > 0
    assert mass_g == pytest.approx(volume_ml * density_for(dens, "solid", "plastic"))


def test_density_fallback_for_unknown_class():
    dens = load_densities(DENSITIES_CSV)
    assert density_for(dens, "solid", "unobtainium") > 0
