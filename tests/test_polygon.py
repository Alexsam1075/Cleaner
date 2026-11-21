import math

import numpy as np
import pytest
import torch

pytest.importorskip("torch")
from polygon import sdf as S
from polygon.render import DEV, Camera, Material, Part, march
from polygon.trash import random_trash, sdf_volume_ml
from tests.synth_scene import CAMERA, PLANE


def test_box_volume_exact():
    part = Part(lambda p: S.box(p - torch.tensor([0, 0, 0.02], device=p.device), (0.05, 0.03, 0.02)), Material(), 0.07)
    assert sdf_volume_ml([part], 0.07, 0.05) == pytest.approx(0.1 * 0.06 * 0.04 * 1e6, rel=0.02)


def test_ray_hits_box_top_at_right_depth():
    cam = Camera.from_plane(CAMERA, PLANE)
    top = 0.04
    part = Part(lambda p: S.box(p - torch.tensor([0, 0, top / 2], device=p.device), (0.05, 0.05, top / 2)),
                Material(), 0.08, pos=(0.5, 0.0, 0.0))
    target = torch.tensor([[0.5, 0.0, top]], device=DEV)
    pix, z = cam.project(target)
    o, d, zf = cam.rays(pix)
    t, hit, _ = march([part], o, d)
    assert bool(hit[0])
    assert float(t[0] * zf[0]) == pytest.approx(float(z[0]), abs=2e-3)


def test_random_trash_properties():
    rng = np.random.default_rng(3)
    for _ in range(6):
        ob = random_trash(rng)
        assert ob.volume_ml > 0.5 and ob.mass_g > 0
        assert ob.take == int(ob.mass_g <= 500)
        assert ob.form in ("bulk", "solid")
