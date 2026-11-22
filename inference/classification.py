from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch

from ..datasets.objects import FORMS, MATERIALS, RGB_MEAN, RGB_STD, _rgb_crop
from ..geometry.features import compute_geometry_features, geometry_input, load_plane


def reclassify_predictions(
    image_path: str | Path,
    predictions: list[dict[str, Any]],
    classifier: torch.nn.Module,
    checkpoint: dict[str, Any],
    camera: dict[str, Any],
    device: torch.device,
    tasks: tuple[str, ...] = ("material",),
    material_blend: float = 1.0,
    form_blend: float = 1.0,
) -> list[dict[str, Any]]:
    """Override requested classification fields, preserving all other predictions."""
    invalid_tasks = set(tasks) - {"form", "material"}
    if invalid_tasks:
        raise ValueError(f"Unsupported classification tasks: {sorted(invalid_tasks)}")
    if not 0.0 <= material_blend <= 1.0:
        raise ValueError("material_blend must be between 0 and 1")
    if not 0.0 <= form_blend <= 1.0:
        raise ValueError("form_blend must be between 0 and 1")
    if not predictions:
        return predictions
    image_path = Path(image_path)
    bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    depth = cv2.imread(str(image_path.parent / "depth.png"), cv2.IMREAD_UNCHANGED)
    if bgr is None or depth is None:
        raise FileNotFoundError(f"RGB/depth pair missing for {image_path}")
    if depth.ndim == 3 and depth.shape[2] == 1:
        depth = depth[:, :, 0]
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    plane = load_plane(image_path.parent / "meta.json")
    mean = torch.as_tensor(checkpoint["geometry_mean"], dtype=torch.float32, device=device)
    std = torch.as_tensor(checkpoint["geometry_std"], dtype=torch.float32, device=device)
    updated_predictions = [dict(prediction) for prediction in predictions]

    classifier.eval()
    with torch.no_grad():
        for prediction in updated_predictions:
            x1, y1, x2, y2 = map(float, prediction["box"])
            mask = prediction.get("mask")
            if mask is None or np.asarray(mask).shape != depth.shape or int(np.asarray(mask).sum()) == 0:
                mask = np.zeros(depth.shape, dtype=np.uint8)
                cv2.rectangle(mask, (max(0, int(x1)), max(0, int(y1))),
                              (min(depth.shape[1] - 1, int(x2)), min(depth.shape[0] - 1, int(y2))),
                              1, thickness=-1)
            else:
                mask = np.asarray(mask).astype(bool)

            bbox_xywh = [x1, y1, max(1.0, x2 - x1), max(1.0, y2 - y1)]
            rgb_crop = _rgb_crop(rgb, mask, bbox_xywh, int(checkpoint["input_rgb_size"]))
            geometry, _ = compute_geometry_features(
                depth, mask, plane, camera, float(checkpoint.get("geometry_cell_m", 0.004))
            )
            depth_map = geometry_input(
                depth, mask, plane, camera, int(checkpoint.get("geometry_size", 80))
            )
            normalized_geometry = (geometry - mean.cpu().numpy()) / np.maximum(std.cpu().numpy(), 1.0e-5)
            image = rgb_crop.astype(np.float32) / 255.0
            image = (image - RGB_MEAN) / RGB_STD
            rgb_tensor = torch.from_numpy(np.ascontiguousarray(image.transpose(2, 0, 1))) \
                .unsqueeze(0).float().to(device)
            geometry_tensor = torch.from_numpy(normalized_geometry).unsqueeze(0).float().to(device)
            depth_tensor = torch.from_numpy(depth_map).unsqueeze(0).float().to(device)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"):
                output = classifier(rgb_tensor, geometry_tensor, depth_tensor)
            form_probs = torch.softmax(output["form"].float(), dim=1)[0].cpu().numpy()
            material_probs = torch.softmax(output["material"].float(), dim=1)[0].cpu().numpy()
            if "form" in tasks:
                base_form_probs = np.asarray(prediction.get("form_probs", []), dtype=np.float32)
                if base_form_probs.shape == form_probs.shape and np.isfinite(base_form_probs).all() and base_form_probs.sum() > 0:
                    base_form_probs = base_form_probs / base_form_probs.sum()
                    form_probs = (1.0 - form_blend) * base_form_probs + form_blend * form_probs
                    form_probs = form_probs / max(float(form_probs.sum()), 1.0e-8)
                prediction["form"] = FORMS[int(np.argmax(form_probs))]
                prediction["form_probs"] = form_probs.tolist()
            if "material" in tasks:
                base_probs = np.asarray(prediction.get("material_probs", []), dtype=np.float32)
                if base_probs.shape == material_probs.shape and np.isfinite(base_probs).all() and base_probs.sum() > 0:
                    base_probs = base_probs / base_probs.sum()
                    material_probs = (1.0 - material_blend) * base_probs + material_blend * material_probs
                    material_probs = material_probs / max(float(material_probs.sum()), 1.0e-8)
                prediction["material"] = MATERIALS[int(np.argmax(material_probs))]
                prediction["material_probs"] = material_probs.tolist()
    return updated_predictions
