"""predict_frame with stand-in models: the parts backend must not change the take decision."""
import json
import math
import sys
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
PKG = Path(__file__).resolve().parents[1].name
predictor = __import__(f"{PKG}.inference.predictor", fromlist=["predict_frame"])
geometry_features = __import__(f"{PKG}.geometry.features", fromlist=["FEATURE_NAMES"])

from evaluation.volume_parts_cv import fit_calibrator  # noqa: E402
from tests.synth_scene import CAMERA, PLANE, render  # noqa: E402
from tests.test_volume_cv import _rows  # noqa: E402

BIG = {"kind": "box", "center_uv": (0.5, 0.0), "size": (0.20, 0.10, 0.08), "yaw": 0.0}


class FakeDetector:
    def __init__(self, mask):
        ys, xs = np.nonzero(mask)
        box = torch.tensor([[xs.min(), ys.min(), xs.max(), ys.max()]], dtype=torch.float32)
        self.res = SimpleNamespace(
            boxes=SimpleNamespace(xyxy=box, conf=torch.tensor([0.9]), cls=torch.tensor([1]),
                                  __len__=lambda: 1),
            masks=SimpleNamespace(data=torch.from_numpy(mask[None].astype(np.float32))))

    def predict(self, **kwargs):
        boxes = self.res.boxes
        class B:  # object with len() for the "no detections" check
            xyxy, conf, cls = boxes.xyxy, boxes.conf, boxes.cls
            def __len__(self):
                return 1
        return [SimpleNamespace(boxes=B(), masks=self.res.masks)]


class FakeAttr(torch.nn.Module):
    def forward(self, rgb, geom, depth):
        out = {"form": torch.tensor([[0.0, 5.0]]), "material": torch.zeros(1, 6),
               "volume_log": torch.tensor([math.log1p(150.0)]), "mass_log": torch.tensor([math.log1p(100.0)]),
               "take_logit": torch.tensor([0.0])}
        out["material"][0, predictor.MATERIALS.index("metal")] = 5.0
        return out


def test_parts_backend_keeps_net_take(tmp_path):
    depth, masks = render(CAMERA, PLANE, [BIG])
    cv2.imwrite(str(tmp_path / "color.png"), np.full(depth.shape + (3,), 128, np.uint8))
    cv2.imwrite(str(tmp_path / "depth.png"), depth)
    (tmp_path / "meta.json").write_text(json.dumps({"floor_plane": PLANE.tolist()}))
    cal = fit_calibrator(_rows())
    cal["coef"] = [0.0] * len(cal["coef"])
    cal["intercept"] = math.log(5000.0)          # parts volume 5000 ml -> metal mass far above 500 g
    n = len(geometry_features.FEATURE_NAMES)
    ckpt = {"input_rgb_size": 64, "input_geometry_size": 32, "geometry_mean": [0.0] * n, "geometry_std": [1.0] * n}
    preds = predictor.predict_frame(tmp_path / "color.png", FakeDetector(masks[0]), FakeAttr(), ckpt,
                                    torch.device("cpu"), CAMERA, volume_backend="parts",
                                    volume_calibrator=cal, densities={("solid", "metal"): 0.82})
    assert len(preds) == 1
    assert abs(preds[0]["volume_ml"] - 5000.0) < 1.0
    assert preds[0]["take"] == 1                 # net mass 100 g decides inside the uncertain band
