from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pandas as pd
import torch
from ultralytics import YOLO

from ..datasets.common import frame_id, load_coco
from ..datasets.objects import MATERIALS, RGB_MEAN, RGB_STD, _rgb_crop
from ..geometry.features import compute_geometry_features, geometry_input, load_camera, load_plane
from ..training.attributes import load_attribute_model


def _mask_for_detection(result: Any, index: int, height: int, width: int,
                         xyxy: tuple[float, float, float, float]) -> tuple[np.ndarray, bool]:
    if result.masks is not None and index < len(result.masks.data):
        source = result.masks.data[index].detach().float().cpu().numpy()
        mask = (source > 0.5).astype(np.uint8)
        if mask.shape != (height, width):
            mask = cv2.resize(mask, (width, height), interpolation=cv2.INTER_NEAREST)
        if int(mask.sum()) > 0:
            return mask.astype(bool), False
    x1, y1, x2, y2 = xyxy
    mask = np.zeros((height, width), dtype=np.uint8)
    cv2.rectangle(mask, (max(0, int(x1)), max(0, int(y1))),
                  (min(width - 1, int(x2)), min(height - 1, int(y2))), 1, thickness=-1)
    return mask.astype(bool), True


def _to_tensor(rgb_crop: np.ndarray, geometry: np.ndarray, geom_map: np.ndarray,
               device: torch.device, checkpoint: dict[str, Any]) -> tuple[torch.Tensor, ...]:
    image = rgb_crop.astype(np.float32) / 255.0
    image = (image - RGB_MEAN) / RGB_STD
    image_tensor = torch.from_numpy(np.ascontiguousarray(image.transpose(2, 0, 1))).unsqueeze(0).float().to(device)
    mean = torch.as_tensor(checkpoint["geometry_mean"], dtype=torch.float32, device=device)
    std = torch.as_tensor(checkpoint["geometry_std"], dtype=torch.float32, device=device)
    geom_tensor = torch.from_numpy(geometry).unsqueeze(0).float().to(device)
    geom_tensor = (geom_tensor - mean) / std
    map_tensor = torch.from_numpy(geom_map).unsqueeze(0).float().to(device)
    return image_tensor, geom_tensor, map_tensor


def predict_frame(image_path: str | Path, detector: YOLO,
                  attr_model: torch.nn.Module, checkpoint: dict[str, Any],
                  device: torch.device, camera: dict[str, Any],
                  conf: float = 0.08, imgsz: int = 640, max_det: int = 30,
                  take_threshold_g: float = 500.0) -> list[dict[str, Any]]:
    image_path = Path(image_path)
    bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    depth = cv2.imread(str(image_path.parent / "depth.png"), cv2.IMREAD_UNCHANGED)
    if bgr is None or depth is None:
        raise FileNotFoundError(f"RGB/depth pair missing for {image_path}")
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    height, width = depth.shape
    plane = load_plane(image_path.parent / "meta.json")
    results = detector.predict(source=str(image_path), imgsz=imgsz, conf=conf,
                               max_det=max_det, device=str(device), retina_masks=True,
                               verbose=False)
    if not results or results[0].boxes is None or len(results[0].boxes) == 0:
        return []
    result = results[0]
    boxes = result.boxes.xyxy.detach().float().cpu().numpy()
    scores = result.boxes.conf.detach().float().cpu().numpy()
    classes = result.boxes.cls.detach().long().cpu().numpy()
    predictions = []
    attr_model.eval()
    with torch.no_grad():
        for i, (xyxy, score, detector_form) in enumerate(zip(boxes, scores, classes)):
            x1, y1, x2, y2 = [float(v) for v in xyxy]
            mask, box_mask_fallback = _mask_for_detection(result, i, height, width, (x1, y1, x2, y2))
            bbox_xywh = [x1, y1, max(1.0, x2 - x1), max(1.0, y2 - y1)]
            rgb_crop = _rgb_crop(rgb, mask, bbox_xywh, int(checkpoint["input_rgb_size"]))
            geometry, geometry_meta = compute_geometry_features(
                depth, mask, plane, camera, float(checkpoint.get("geometry_cell_m", 0.004)))
            geom_map = geometry_input(depth, mask, plane, camera,
                                      int(checkpoint["input_geometry_size"]))
            rgb_t, geometry_t, depth_t = _to_tensor(rgb_crop, geometry, geom_map, device, checkpoint)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"):
                output = attr_model(rgb_t, geometry_t, depth_t)
            form_probs = torch.softmax(output["form"].float(), dim=1)[0].cpu().numpy()
            det_form = int(detector_form)
            combined_form = 0.65 * form_probs
            if det_form in (0, 1):
                combined_form[det_form] += 0.35
            form_idx = int(np.argmax(combined_form))
            material_probs = torch.softmax(output["material"].float(), dim=1)[0].cpu().numpy()
            material_idx = int(np.argmax(material_probs))
            volume_log = float(np.nan_to_num(output["volume_log"].float().item(), nan=0.0,
                                             posinf=np.log1p(20000.0), neginf=0.0))
            mass_log = float(np.nan_to_num(output["mass_log"].float().item(), nan=0.0,
                                           posinf=np.log1p(5000.0), neginf=0.0))
            volume_ml = float(np.clip(np.expm1(np.clip(volume_log, 0.0, np.log1p(20000.0))), 1.0, 20000.0))
            mass_g = float(np.clip(np.expm1(np.clip(mass_log, 0.0, np.log1p(5000.0))), 1.0, 5000.0))
            take_prob = float(torch.sigmoid(output["take_logit"].float()).item())
            take = int(take_prob >= 0.5)
            # Keep the physical boundary visible in the output when the take head is uncertain.
            if abs(take_prob - 0.5) < 0.08:
                take = int(mass_g <= take_threshold_g)
            predictions.append({
                "box": [x1, y1, x2, y2],
                "score": float(score * max(float(combined_form.max()), 0.05)),
                "form": ["bulk", "solid"][form_idx],
                "form_probs": combined_form.tolist(),
                "material": MATERIALS[material_idx],
                "material_probs": material_probs.tolist(),
                "volume_ml": volume_ml,
                "mass_g": mass_g,
                "take": take,
                "take_probability": take_prob,
                "geometry": geometry_meta,
                "mask": mask,
                "box_mask_fallback": box_mask_fallback,
            })
    return predictions


def prediction_cell(predictions: list[dict[str, Any]]) -> str:
    if not predictions:
        return "none"
    parts = []
    for obj in predictions[:30]:
        x1, y1, x2, y2 = obj["box"]
        parts.append(f"{x1:.1f} {y1:.1f} {x2:.1f} {y2:.1f} {obj['score']:.5f} "
                     f"{obj['form']} {obj['material']} {obj['volume_ml']:.2f} "
                     f"{obj['mass_g']:.2f} {int(obj['take'])}")
    return ";".join(parts)


def predict_ids(data_root: str | Path, image_ids: list[str], detector_weights: str | Path,
                attribute_weights: str | Path, output_csv: str | Path,
                conf: float = 0.08, imgsz: int = 640, device: str = "cuda:0",
                take_threshold_g: float = 500.0, visualize_dir: str | Path | None = None,
                max_visualizations: int = 20) -> Path:
    data_root = Path(data_root).resolve()
    output_csv = Path(output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    attr_model, checkpoint, device_obj = load_attribute_model(attribute_weights, device)
    detector = YOLO(str(detector_weights))
    camera = load_camera(data_root / "camera.json")
    rows = []
    visualized = 0
    for fid in image_ids:
        path = data_root / "real" / "test" / fid / "color.png"
        if not path.is_file():
            # Validation calls can use real/train frames.
            path = data_root / "real" / "train" / fid / "color.png"
        predictions = predict_frame(path, detector, attr_model, checkpoint, device_obj,
                                    camera, conf, imgsz, take_threshold_g=take_threshold_g)
        rows.append({"image_id": fid, "prediction": prediction_cell(predictions)})
        if visualize_dir is not None and visualized < max_visualizations:
            from ..evaluation.visualize import save_overlay
            save_overlay(path, predictions, Path(visualize_dir) / f"{fid}.png")
            visualized += 1
    pd.DataFrame(rows, columns=["image_id", "prediction"]).to_csv(output_csv, index=False)
    return output_csv


def sample_submission_ids(sample_path: str | Path) -> list[str]:
    frame = pd.read_csv(sample_path)
    if list(frame.columns) != ["image_id", "prediction"]:
        raise ValueError("sample_submission.csv columns must be image_id,prediction")
    ids = frame["image_id"].astype(str).tolist()
    if len(ids) != len(set(ids)):
        raise ValueError("sample_submission.csv has repeated image_id values")
    return ids

