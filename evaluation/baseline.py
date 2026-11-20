from __future__ import annotations

import csv
from collections import Counter
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from ultralytics import YOLO

from ..datasets.common import annotation_object_id, frame_id, image_path, load_coco
from ..datasets.objects import polygon_mask
from ..geometry.features import compute_geometry_features, load_camera, load_plane
from ..inference.predictor import _mask_for_detection


def baseline_statistics(data_root: str | Path, split: dict[str, Any]) -> dict[str, Any]:
    root = Path(data_root)
    coco = load_coco(root, "real_train")
    train_frames = set(split["train_real_frames"])
    material_counts = Counter()
    physical_volumes: dict[str, list[float]] = {}
    raw_geometry_by_object: dict[str, list[float]] = {}
    gt_volume_by_object: dict[str, list[float]] = {}
    camera = load_camera(root / "camera.json")
    for image in coco["images"]:
        if frame_id(image) not in train_frames:
            continue
        frame_path = image_path(root, image)
        depth = cv2.imread(str(frame_path.parent / "depth.png"), cv2.IMREAD_UNCHANGED)
        if depth is None:
            continue
        plane = load_plane(frame_path.parent / "meta.json")
        for ann in coco["_annotations_by_image"].get(int(image["id"]), []):
            attrs = ann["attributes"]
            material_counts[attrs["material"]] += 1
            oid = annotation_object_id(ann)
            physical_volumes.setdefault(oid, []).append(float(attrs["volume_ml"]))
            gt_volume_by_object.setdefault(oid, []).append(float(attrs["volume_ml"]))
            mask = polygon_mask(ann, int(image["height"]), int(image["width"]))
            _, geom = compute_geometry_features(depth, mask, plane, camera)
            raw = geom.get("geometric_volume_ml")
            if raw is not None and np.isfinite(raw) and float(raw) > 0:
                raw_geometry_by_object.setdefault(oid, []).append(float(raw))
    if not material_counts:
        raise ValueError("No real training annotations for baseline statistics")
    volumes = [float(value) for values in physical_volumes.values() for value in values]
    calibration_pairs = [(float(np.median(raw_values)),
                          float(np.median(gt_volume_by_object[oid])))
                         for oid, raw_values in raw_geometry_by_object.items()
                         if oid in gt_volume_by_object]
    if len(calibration_pairs) >= 3:
        x = np.log1p(np.asarray([pair[0] for pair in calibration_pairs], dtype=np.float64))
        y = np.log1p(np.asarray([pair[1] for pair in calibration_pairs], dtype=np.float64))
        slope, intercept = [float(v) for v in np.polyfit(x, y, 1)]
        slope = float(np.clip(slope, 0.1, 2.0))
    else:
        slope, intercept = 1.0, 0.0
    density_rows = {}
    density_path = root / "densities.csv"
    with density_path.open(newline="", encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            density_rows[f"{row['form']}|{row['material']}"] = float(row["density_g_per_ml"])
    return {"majority_material": material_counts.most_common(1)[0][0],
            "median_train_volume_ml": float(np.median(volumes)),
            "densities": density_rows,
            "material_counts": dict(material_counts),
            "calibration_object_count": len(calibration_pairs),
            "volume_calibration": {"method": "train-only linear fit in log1p volume space",
                                   "slope": slope, "intercept": intercept}}


def predict_baseline_frame(image_path_value: str | Path, detector: YOLO,
                           data_root: str | Path, camera: dict[str, Any],
                           stats: dict[str, Any], conf: float = 0.08,
                           imgsz: int = 640, device: str = "0",
                           max_det: int = 30, calibrate_volume: bool = False) -> list[dict[str, Any]]:
    path = Path(image_path_value)
    depth = cv2.imread(str(path.parent / "depth.png"), cv2.IMREAD_UNCHANGED)
    if depth is None:
        raise FileNotFoundError(path.parent / "depth.png")
    results = detector.predict(source=str(path), imgsz=imgsz, conf=conf, max_det=max_det,
                               device=device, retina_masks=True, verbose=False)
    if not results or results[0].boxes is None or len(results[0].boxes) == 0:
        return []
    result = results[0]
    boxes = result.boxes.xyxy.detach().float().cpu().numpy()
    scores = result.boxes.conf.detach().float().cpu().numpy()
    classes = result.boxes.cls.detach().long().cpu().numpy()
    plane = load_plane(path.parent / "meta.json")
    predictions = []
    for i, (xyxy, score, cls) in enumerate(zip(boxes, scores, classes)):
        x1, y1, x2, y2 = [float(v) for v in xyxy]
        mask, fallback = _mask_for_detection(result, i, depth.shape[0], depth.shape[1], (x1, y1, x2, y2))
        geometry, geom_meta = compute_geometry_features(depth, mask, plane, camera)
        volume = geom_meta["geometric_volume_ml"]
        if volume is None or not np.isfinite(volume) or volume <= 0:
            volume = stats["median_train_volume_ml"]
        elif calibrate_volume:
            calibration = stats["volume_calibration"]
            volume = float(np.expm1(calibration["slope"] * np.log1p(volume) + calibration["intercept"]))
            volume = float(np.clip(volume, 1.0, 20000.0))
        form = ["bulk", "solid"][int(cls)] if int(cls) in (0, 1) else "solid"
        material = stats["majority_material"]
        density = stats["densities"].get(f"{form}|{material}", 1.0)
        mass = float(volume) * float(density)
        predictions.append({
            "box": [x1, y1, x2, y2], "score": float(score),
            "form": form, "material": material,
            "volume_ml": float(volume), "mass_g": mass,
            "take": int(mass <= 500.0), "geometry": geom_meta,
            "mask": mask, "box_mask_fallback": fallback,
        })
    return predictions

