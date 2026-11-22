"""Grouped-CV test of RGB specular highlights and missing-depth cues for material.

The outer validation object IDs are never used for fitting, feature selection,
or threshold selection. Only the final official-score call reads their labels.
This experiment changes the material token of an existing submission and
leaves detection, volume, and take untouched.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import cv2
import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from torchvision.models import ResNet18_Weights, resnet18
from ultralytics import YOLO

if __package__ in (None, ""):
    project_dir = Path(__file__).resolve().parent.parent
    sys.path.insert(0, str(project_dir.parent))
    __package__ = project_dir.name + ".evaluation"

from ..datasets.common import frame_id, image_path, load_coco, resolve_data_root
from ..datasets.objects import MATERIALS, _rgb_crop, object_records, polygon_mask
from ..evaluation.original_metric import load_original_score, score_submission


def _read_depth(path: str | Path) -> np.ndarray | None:
    depth = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if depth is not None and depth.ndim == 3 and depth.shape[2] == 1:
        depth = depth[:, :, 0]
    return depth


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


FEATURE_VERSION = "glare_rgbd_v1"
RGB_MEAN = np.asarray([0.485, 0.456, 0.406], dtype=np.float32)
RGB_STD = np.asarray([0.229, 0.224, 0.225], dtype=np.float32)
TRAIN_CV_SEEDS = (17, 41, 73)
SYNTHETIC_OBJECT_WEIGHT = 0.25


def _largest_component_features(binary: np.ndarray, object_pixels: int,
                                min_area: int) -> tuple[float, float, float, float]:
    count, labels, stats, _ = cv2.connectedComponentsWithStats(
        binary.astype(np.uint8), connectivity=8
    )
    components = [i for i in range(1, count)
                  if int(stats[i, cv2.CC_STAT_AREA]) >= min_area]
    if not components:
        return 0.0, 0.0, 0.0, 0.0
    largest = max(components, key=lambda i: int(stats[i, cv2.CC_STAT_AREA]))
    area = int(stats[largest, cv2.CC_STAT_AREA])
    width = int(stats[largest, cv2.CC_STAT_WIDTH])
    height = int(stats[largest, cv2.CC_STAT_HEIGHT])
    component = (labels == largest).astype(np.uint8)
    contours, _ = cv2.findContours(component, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    perimeter = float(cv2.arcLength(contours[0], True)) if contours else 0.0
    circularity = 4.0 * math.pi * area / max(perimeter * perimeter, 1.0)
    return (math.log1p(len(components)), area / max(object_pixels, 1),
            math.log(max(width / max(height, 1), 1e-6)),
            float(np.clip(circularity, 0.0, 1.0)))


def _handcrafted_features(rgb: np.ndarray, mask: np.ndarray,
                          depth: np.ndarray) -> dict[str, float]:
    """Small, explicit RGB/depth indicators; raw depth holes stay unfilled."""
    m = np.asarray(mask, dtype=bool)
    if rgb.shape[:2] != m.shape or depth.shape[:2] != m.shape:
        raise ValueError(f"RGB/mask/depth shape mismatch: {rgb.shape}, {m.shape}, {depth.shape}")
    n = int(m.sum())
    if n == 0:
        return {name: 0.0 for name in _HAND_FEATURE_NAMES}

    pixels = rgb[m].astype(np.uint8)
    hsv = cv2.cvtColor(pixels.reshape(-1, 1, 3), cv2.COLOR_RGB2HSV).reshape(-1, 3)
    lab = cv2.cvtColor(pixels.reshape(-1, 1, 3), cv2.COLOR_RGB2LAB).reshape(-1, 3)
    vals: dict[str, float] = {}
    for idx, channel in enumerate(("r", "g", "b")):
        x = pixels[:, idx].astype(np.float32) / 255.0
        vals[f"color_{channel}_mean"] = float(x.mean())
        vals[f"color_{channel}_std"] = float(x.std())
    for idx, channel in ((1, "s"), (2, "v")):
        x = hsv[:, idx].astype(np.float32) / 255.0
        vals[f"color_hsv_{channel}_mean"] = float(x.mean())
        vals[f"color_hsv_{channel}_std"] = float(x.std())
        vals[f"color_hsv_{channel}_q90"] = float(np.quantile(x, 0.90))
    for idx, channel in enumerate(("l", "a", "b")):
        x = lab[:, idx].astype(np.float32) / 255.0
        vals[f"color_lab_{channel}_mean"] = float(x.mean())
        vals[f"color_lab_{channel}_std"] = float(x.std())

    # Fixed sensor-scale definition of a clipped, low-saturation highlight.
    # The neutral-white condition catches broad diffuse highlights as well.
    rgbf = rgb.astype(np.float32)
    vmax = rgbf.max(axis=2)
    vmin = rgbf.min(axis=2)
    hsv_full = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    highlight = m & (hsv_full[..., 2] >= 238) & (hsv_full[..., 1] <= 50)
    neutral = m & (vmin >= 220.0) & ((vmax - vmin) <= 32.0)
    hi_count = int(highlight.sum())
    neutral_count = int(neutral.sum())
    vals["glare_bright_low_sat_fraction"] = hi_count / n
    vals["glare_neutral_white_fraction"] = neutral_count / n
    vals["glare_value_q95"] = float(np.quantile(hsv[:, 2], 0.95) / 255.0)
    vals["glare_value_q99"] = float(np.quantile(hsv[:, 2], 0.99) / 255.0)
    min_area = max(3, int(n * 0.00005))
    comp = _largest_component_features(highlight, n, min_area)
    vals["glare_component_log_count"] = comp[0]
    vals["glare_largest_component_fraction"] = comp[1]
    vals["glare_largest_component_log_aspect"] = comp[2]
    vals["glare_largest_component_circularity"] = comp[3]

    d = np.asarray(depth, dtype=np.float32)
    valid = m & np.isfinite(d) & (d > 50.0) & (d < 8000.0)
    holes = m & ~valid
    depth_count = int(valid.sum())
    hole_count = int(holes.sum())
    vals["depth_valid_fraction"] = depth_count / n
    vals["depth_hole_fraction"] = hole_count / n
    vals["depth_high_sentinel_fraction"] = float((m & np.isfinite(d) & (d >= 8000.0)).sum()) / n
    vals["depth_low_invalid_fraction"] = float((m & np.isfinite(d) & (d <= 50.0)).sum()) / n
    if depth_count:
        z = d[valid] / 1000.0
        median = float(np.median(z))
        vals["depth_median_m"] = median
        vals["depth_mad_m"] = float(np.median(np.abs(z - median)))
        vals["depth_std_m"] = float(z.std())
        vals["depth_q10_m"] = float(np.quantile(z, 0.10))
        vals["depth_q90_m"] = float(np.quantile(z, 0.90))
    else:
        for name in ("depth_median_m", "depth_mad_m", "depth_std_m", "depth_q10_m", "depth_q90_m"):
            vals[name] = 0.0
    hole_comp = _largest_component_features(holes, n, min_area)
    vals["holes_component_log_count"] = hole_comp[0]
    vals["holes_largest_component_fraction"] = hole_comp[1]
    vals["holes_largest_component_log_aspect"] = hole_comp[2]
    vals["holes_largest_component_circularity"] = hole_comp[3]

    # Preserve the invalid-depth pattern and measure where RGB highlights
    # coincide with it or lie immediately beside it.
    if hi_count:
        vals["cross_glare_over_holes_fraction"] = float((highlight & holes).sum()) / hi_count
        vals["cross_holes_under_glare_fraction"] = float((highlight & holes).sum()) / max(hole_count, 1)
    else:
        vals["cross_glare_over_holes_fraction"] = 0.0
        vals["cross_holes_under_glare_fraction"] = 0.0
    near_hole = cv2.dilate(holes.astype(np.uint8), np.ones((5, 5), np.uint8), iterations=1).astype(bool)
    vals["cross_glare_near_holes_fraction"] = float((highlight & near_hole).sum()) / max(hi_count, 1)
    vals["cross_glare_near_holes_object_fraction"] = float((m & near_hole).sum()) / n

    # Missing/extra dictionary values are a programming error, not silently zeroed.
    return {name: float(vals[name]) for name in _HAND_FEATURE_NAMES}


_HAND_FEATURE_NAMES = [
    "color_r_mean", "color_r_std", "color_g_mean", "color_g_std", "color_b_mean", "color_b_std",
    "color_hsv_s_mean", "color_hsv_s_std", "color_hsv_s_q90",
    "color_hsv_v_mean", "color_hsv_v_std", "color_hsv_v_q90",
    "color_lab_l_mean", "color_lab_l_std", "color_lab_a_mean", "color_lab_a_std",
    "color_lab_b_mean", "color_lab_b_std",
    "glare_bright_low_sat_fraction", "glare_neutral_white_fraction", "glare_value_q95", "glare_value_q99",
    "glare_component_log_count", "glare_largest_component_fraction",
    "glare_largest_component_log_aspect", "glare_largest_component_circularity",
    "depth_valid_fraction", "depth_hole_fraction", "depth_high_sentinel_fraction",
    "depth_low_invalid_fraction", "depth_median_m", "depth_mad_m", "depth_std_m", "depth_q10_m", "depth_q90_m",
    "holes_component_log_count", "holes_largest_component_fraction",
    "holes_largest_component_log_aspect", "holes_largest_component_circularity",
    "cross_glare_over_holes_fraction", "cross_holes_under_glare_fraction",
    "cross_glare_near_holes_fraction", "cross_glare_near_holes_object_fraction",
]


def _extract_records(records: list[dict[str, Any]], batch_size: int,
                     device: torch.device) -> tuple[np.ndarray, np.ndarray, list[dict[str, Any]]]:
    crops: list[np.ndarray] = []
    hand_rows: list[list[float]] = []
    meta: list[dict[str, Any]] = []
    for index, record in enumerate(records):
        path = Path(record["image"])
        bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
        depth = _read_depth(path.parent / "depth.png")
        if bgr is None or depth is None:
            raise FileNotFoundError(f"RGB/depth pair missing: {path}")
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        mask = polygon_mask(record, depth.shape[0], depth.shape[1])
        crop = _rgb_crop(rgb, mask, record["bbox"], 224)
        handcrafted = _handcrafted_features(rgb, mask, depth)
        crops.append(crop)
        hand_rows.append([handcrafted[name] for name in _HAND_FEATURE_NAMES])
        meta.append({
            "frame_id": str(record["frame_id"]),
            "object_id": str(record["object_id"]),
            "material": str(record["material"]),
            "source": "synth" if str(record["coco_name"]).startswith("synth") else "real",
            "image": str(record["image"]),
        })
        if (index + 1) % 128 == 0:
            print(f"features: {index + 1}/{len(records)} annotations", flush=True)

    model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
    model.fc = torch.nn.Identity()
    model.eval().to(device)
    mean = torch.as_tensor(RGB_MEAN, dtype=torch.float32, device=device)[None, :, None, None]
    std = torch.as_tensor(RGB_STD, dtype=torch.float32, device=device)[None, :, None, None]
    embeddings: list[np.ndarray] = []
    with torch.inference_mode():
        for start in range(0, len(crops), batch_size):
            images = np.stack(crops[start:start + batch_size]).astype(np.float32) / 255.0
            tensor = torch.from_numpy(images.transpose(0, 3, 1, 2)).to(device)
            tensor = (tensor - mean) / std
            if device.type == "cuda":
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    features = model(tensor)
            else:
                features = model(tensor)
            embeddings.append(features.float().cpu().numpy())
    del model
    return np.concatenate(embeddings).astype(np.float32), np.asarray(hand_rows, dtype=np.float32), meta


def _features_for(embeddings: np.ndarray, hand: np.ndarray, mode: str) -> np.ndarray:
    names = _HAND_FEATURE_NAMES
    color_idx = [i for i, name in enumerate(names) if name.startswith("color_")]
    glare_idx = [i for i, name in enumerate(names) if name.startswith("glare_")]
    depth_idx = [i for i, name in enumerate(names)
                 if name.startswith("depth_") or name.startswith("holes_")]
    cross_idx = [i for i, name in enumerate(names) if name.startswith("cross_")]
    if mode == "resnet":
        pieces = [embeddings]
    elif mode == "rgbd_basic":
        pieces = [embeddings, hand[:, color_idx], hand[:, depth_idx]]
    elif mode == "rgb_glare":
        pieces = [embeddings, hand[:, color_idx], hand[:, glare_idx]]
    elif mode == "rgbd_glare":
        pieces = [embeddings, hand]
    elif mode == "handcrafted_rgbd_glare":
        pieces = [hand[:, color_idx + glare_idx + depth_idx + cross_idx]]
    else:
        raise ValueError(mode)
    return np.concatenate(pieces, axis=1).astype(np.float32)


def _make_model(kind: str) -> Pipeline:
    if kind == "svc_rbf":
        estimator = SVC(C=1.0, kernel="rbf", gamma="scale", class_weight=None)
    elif kind == "logistic":
        estimator = LogisticRegression(C=0.3, max_iter=3000, solver="lbfgs")
    else:
        raise ValueError(kind)
    return Pipeline([("scale", StandardScaler()), ("classifier", estimator)])


def _training_weights(rows: list[dict[str, Any]], indices: np.ndarray,
                      synth_weight: float) -> np.ndarray:
    selected = [rows[int(i)] for i in indices]
    ids_by_class: dict[str, set[str]] = defaultdict(set)
    samples_by_id_source: dict[tuple[str, str], int] = Counter()
    for row in selected:
        ids_by_class[row["material"]].add(row["object_id"])
        samples_by_id_source[(row["object_id"], row["source"])] += 1
    result = np.zeros(len(selected), dtype=np.float64)
    for local_idx, row in enumerate(selected):
        oid, source, material = row["object_id"], row["source"], row["material"]
        class_ids = max(1, len(ids_by_class[material]))
        real_n = samples_by_id_source[(oid, "real")]
        synth_n = samples_by_id_source[(oid, "synth")]
        real_mass = 1.0 if real_n else 0.0
        synthetic_mass = synth_weight if synth_n else 0.0
        denominator = real_mass + synthetic_mass
        source_mass = real_mass if source == "real" else synthetic_mass
        count = samples_by_id_source[(oid, source)]
        result[local_idx] = source_mass / max(count, 1) / class_ids
    if not np.isfinite(result).all() or (result <= 0).any():
        raise ValueError("Invalid class/object/source sample weights")
    return result


def _fit(model: Pipeline, x: np.ndarray, y: np.ndarray, weights: np.ndarray) -> Pipeline:
    model.fit(x, y, scale__sample_weight=weights, classifier__sample_weight=weights)
    return model


def _run_group_cv(embeddings: np.ndarray, hand: np.ndarray,
                  rows: list[dict[str, Any]], train_real_idx: np.ndarray,
                  all_train_idx: np.ndarray) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    material_by_id: dict[str, str] = {}
    for idx in train_real_idx:
        row = rows[int(idx)]
        previous = material_by_id.setdefault(row["object_id"], row["material"])
        if previous != row["material"]:
            raise ValueError(f"Physical object has inconsistent material: {row['object_id']}")
    object_ids = np.asarray(sorted(material_by_id))
    object_y = np.asarray([material_by_id[oid] for oid in object_ids])
    variants = ("resnet", "rgbd_basic", "rgb_glare", "rgbd_glare", "handcrafted_rgbd_glare")
    train_modes = ("real", "real_plus_synth")
    model_kinds = ("svc_rbf", "logistic")
    summaries: list[dict[str, Any]] = []
    detailed: dict[str, Any] = {"seeds": list(TRAIN_CV_SEEDS), "n_splits": 2, "configs": {}}

    for source_mode in train_modes:
        source_train = np.asarray([i for i in all_train_idx
                                   if source_mode == "real" or rows[int(i)]["source"] == "real" or
                                   rows[int(i)]["source"] == "synth"], dtype=int)
        if source_mode == "real":
            source_train = np.asarray([i for i in source_train if rows[int(i)]["source"] == "real"], dtype=int)
        for variant in variants:
            x_all = _features_for(embeddings, hand, variant)
            for kind in model_kinds:
                key = f"{source_mode}/{variant}/{kind}"
                per_seed: list[dict[str, Any]] = []
                for seed in TRAIN_CV_SEEDS:
                    splitter = StratifiedGroupKFold(n_splits=2, shuffle=True, random_state=seed)
                    oof_true: list[str] = []
                    oof_pred: list[str] = []
                    oof_groups: list[str] = []
                    for fold, (tr_obj_idx, va_obj_idx) in enumerate(
                            splitter.split(object_ids, object_y, object_ids), start=1):
                        tr_ids = set(object_ids[tr_obj_idx].tolist())
                        va_ids = set(object_ids[va_obj_idx].tolist())
                        fit_idx = np.asarray([i for i in source_train if rows[int(i)]["object_id"] in tr_ids], dtype=int)
                        if not fit_idx.size:
                            raise ValueError(f"No training records in {key}, seed={seed}, fold={fold}")
                        val_idx = np.asarray([i for i in train_real_idx if rows[int(i)]["object_id"] in va_ids], dtype=int)
                        if not val_idx.size:
                            raise ValueError(f"No held-out real records in {key}, seed={seed}, fold={fold}")
                        y_fit = np.asarray([rows[int(i)]["material"] for i in fit_idx])
                        weights = _training_weights(rows, fit_idx, SYNTHETIC_OBJECT_WEIGHT)
                        model = _fit(_make_model(kind), x_all[fit_idx], y_fit, weights)
                        predicted = model.predict(x_all[val_idx])
                        oof_true.extend(rows[int(i)]["material"] for i in val_idx)
                        oof_pred.extend(str(x) for x in predicted)
                        oof_groups.extend(rows[int(i)]["object_id"] for i in val_idx)
                    per_seed.append({
                        "seed": seed,
                        "frame_macro_material_accuracy": float(balanced_accuracy_score(oof_true, oof_pred)),
                        "frame_accuracy": float(accuracy_score(oof_true, oof_pred)),
                        "object_accuracy": _object_vote_accuracy(oof_true, oof_pred, oof_groups),
                        "per_material": {
                            material: {
                                "n": int(sum(actual == material for actual in oof_true)),
                                "correct": int(sum(actual == material and predicted == material
                                                   for actual, predicted in zip(oof_true, oof_pred))),
                                "accuracy": float(sum(actual == material and predicted == material
                                                      for actual, predicted in zip(oof_true, oof_pred)) /
                                                  max(1, sum(actual == material for actual in oof_true))),
                            }
                            for material in MATERIALS
                        },
                        "n_real_frames": len(oof_true),
                    })
                mean_m = float(np.mean([x["frame_macro_material_accuracy"] for x in per_seed]))
                summary = {
                    "key": key,
                    "train_source": source_mode,
                    "features": variant,
                    "classifier": kind,
                    "cv_mean_material_macro_accuracy": mean_m,
                    "cv_std_material_macro_accuracy": float(np.std([x["frame_macro_material_accuracy"] for x in per_seed], ddof=1)),
                    "cv_mean_frame_accuracy": float(np.mean([x["frame_accuracy"] for x in per_seed])),
                    "cv_mean_object_accuracy": float(np.mean([x["object_accuracy"] for x in per_seed])),
                    "cv_runs": per_seed,
                }
                summaries.append(summary)
                detailed["configs"][key] = summary
                print(f"CV {key}: M={mean_m:.4f} +/- {summary['cv_std_material_macro_accuracy']:.4f}; "
                      f"object={summary['cv_mean_object_accuracy']:.4f}", flush=True)
    return summaries, detailed


def _object_vote_accuracy(y_true: list[str], y_pred: list[str], groups: list[str]) -> float:
    per_object: dict[str, list[str]] = defaultdict(list)
    truth: dict[str, str] = {}
    for actual, predicted, oid in zip(y_true, y_pred, groups):
        truth[oid] = actual
        per_object[oid].append(predicted)
    hits = 0
    for oid, predictions in per_object.items():
        counts = Counter(predictions)
        majority = sorted(counts.items(), key=lambda item: (-item[1], item[0]))[0][0]
        hits += int(majority == truth[oid])
    return hits / max(len(per_object), 1)


def _val_detection_records(data_root: Path, val_frames: list[str], detector_path: Path,
                           conf: float, imgsz: int, max_det: int,
                           device: str) -> tuple[list[dict[str, Any]], dict[str, int]]:
    coco = load_coco(data_root, "real_train")
    image_by_frame = {frame_id(im): im for im in coco["images"]}
    detector = YOLO(str(detector_path))
    detections: list[dict[str, Any]] = []
    counts = {"frames": 0, "detections": 0, "box_mask_fallbacks": 0}
    for fid in val_frames:
        image = image_by_frame.get(str(fid))
        if image is None:
            raise KeyError(f"Validation frame missing from COCO labels: {fid}")
        path = image_path(data_root, image)
        bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
        depth = _read_depth(path.parent / "depth.png")
        if bgr is None or depth is None:
            raise FileNotFoundError(f"Validation RGB/depth pair missing: {path}")
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        results = detector.predict(source=str(path), imgsz=imgsz, conf=conf, max_det=max_det,
                                   device=device, retina_masks=True, verbose=False)
        counts["frames"] += 1
        if not results or results[0].boxes is None or len(results[0].boxes) == 0:
            continue
        result = results[0]
        boxes = result.boxes.xyxy.detach().float().cpu().numpy()
        for i, box in enumerate(boxes):
            xyxy = tuple(float(x) for x in box)
            mask, fallback = _mask_for_detection(result, i, depth.shape[0], depth.shape[1], xyxy)
            bbox_xywh = [xyxy[0], xyxy[1], max(1.0, xyxy[2] - xyxy[0]), max(1.0, xyxy[3] - xyxy[1])]
            detections.append({
                "image_id": str(fid),
                "box": [float(x) for x in xyxy],
                "crop": _rgb_crop(rgb, mask, bbox_xywh, 224),
                "hand": _handcrafted_features(rgb, mask, depth),
                "box_mask_fallback": bool(fallback),
            })
            counts["box_mask_fallbacks"] += int(fallback)
        counts["detections"] += int(len(boxes))
    return detections, counts


def _embed_crops(crops: list[np.ndarray], batch_size: int,
                 device: torch.device) -> np.ndarray:
    if not crops:
        return np.zeros((0, 512), dtype=np.float32)
    model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
    model.fc = torch.nn.Identity()
    model.eval().to(device)
    mean = torch.as_tensor(RGB_MEAN, dtype=torch.float32, device=device)[None, :, None, None]
    std = torch.as_tensor(RGB_STD, dtype=torch.float32, device=device)[None, :, None, None]
    out: list[np.ndarray] = []
    with torch.inference_mode():
        for start in range(0, len(crops), batch_size):
            images = np.stack(crops[start:start + batch_size]).astype(np.float32) / 255.0
            tensor = torch.from_numpy(images.transpose(0, 3, 1, 2)).to(device)
            tensor = (tensor - mean) / std
            if device.type == "cuda":
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    features = model(tensor)
            else:
                features = model(tensor)
            out.append(features.float().cpu().numpy())
    del model
    return np.concatenate(out).astype(np.float32)


def _load_predictions(csv_path: Path, metric) -> tuple[pd.DataFrame, dict[str, list[dict[str, Any]]], dict[str, list[str]]]:
    frame = pd.read_csv(csv_path)
    if list(frame.columns) != ["image_id", "prediction"]:
        raise ValueError(f"Unexpected columns in candidate submission: {csv_path}")
    parsed: dict[str, list[dict[str, Any]]] = {}
    raw_parts: dict[str, list[str]] = {}
    for row in frame.itertuples(index=False):
        fid = str(row.image_id)
        raw = str(row.prediction)
        parsed[fid] = metric._parse(raw, True, fid)
        raw_parts[fid] = [] if raw.strip().lower() in {"", "none"} else [x.strip() for x in raw.split(";")]
        if len(parsed[fid]) != len(raw_parts[fid]):
            raise ValueError(f"Parsed/raw object count mismatch for {fid}")
    return frame, parsed, raw_parts


def _iou(a: list[float] | tuple[float, ...], b: list[float] | tuple[float, ...]) -> float:
    x1 = max(float(a[0]), float(b[0]))
    y1 = max(float(a[1]), float(b[1]))
    x2 = min(float(a[2]), float(b[2]))
    y2 = min(float(a[3]), float(b[3]))
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area_a = max(0.0, float(a[2]) - float(a[0])) * max(0.0, float(a[3]) - float(a[1]))
    area_b = max(0.0, float(b[2]) - float(b[0])) * max(0.0, float(b[3]) - float(b[1]))
    return inter / max(area_a + area_b - inter, 1e-12)


def _material_confusion(state: dict[str, Any], metric) -> dict[str, Any]:
    per_class = {name: {"n": 0, "correct": 0} for name in MATERIALS}
    confusion = {name: {pred: 0 for pred in MATERIALS} for name in MATERIALS}
    gt_all, pred_all = state["ground_truth"], state["predictions"]
    for fid, targets in gt_all.items():
        matches = metric._match(targets, pred_all.get(fid, []))
        for target, found in zip(targets, matches):
            truth = str(target["material"])
            per_class[truth]["n"] += 1
            if found is not None:
                pred = str(found["material"])
                confusion[truth][pred] += 1
                per_class[truth]["correct"] += int(pred == truth)
    for values in per_class.values():
        values["accuracy"] = values["correct"] / max(values["n"], 1)
    return {"per_material": per_class, "confusion": confusion}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", default=None)
    parser.add_argument("--project-root", default=None)
    parser.add_argument("--output", default="checkpoints/material_glare_fold1")
    parser.add_argument("--candidate", default="logs/score83_candidates/C_joint/rgb_only/geometry_train_oofvoltake_340ml.csv")
    parser.add_argument("--split", default="configs/object_id_split.json")
    parser.add_argument("--detector", default="checkpoints/experiments/C_joint/detector.pt")
    parser.add_argument("--score-path", default="references/score.py")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--conf", type=float, default=0.08)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--max-det", type=int, default=30)
    args = parser.parse_args()

    project = Path(args.project_root).resolve() if args.project_root else Path(__file__).resolve().parent.parent
    output = (project / args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    data_root = resolve_data_root(args.data_root)
    split = json.loads((project / args.split).resolve().read_text(encoding="utf-8"))
    train_ids = set(map(str, split["train_object_ids"]))
    val_ids = set(map(str, split["validation_object_ids"]))
    if train_ids & val_ids:
        raise ValueError("Training/validation object IDs overlap")
    train_frames = set(map(str, split["train_real_frames"]))
    val_frames = list(map(str, split["validation_real_frames"]))
    synth_frames = set(map(str, split["train_synth_frames"]))
    real_records = object_records(data_root, "real_train", train_frames)
    synth_records = object_records(data_root, "synth_train", synth_frames)
    if any(str(row["object_id"]) not in train_ids for row in real_records + synth_records):
        raise ValueError("A validation or unknown object entered feature extraction")
    if {str(row["object_id"]) for row in real_records} != train_ids:
        raise ValueError("Real feature records do not cover the exact training object split")
    if set(map(str, split.get("train_synth_object_ids", []))) & val_ids:
        raise ValueError("Saved synthetic training IDs overlap outer validation IDs")
    records = real_records + synth_records
    record_meta = [{
        "frame_id": str(row["frame_id"]), "object_id": str(row["object_id"]),
        "material": str(row["material"]),
        "source": "synth" if str(row["coco_name"]).startswith("synth") else "real",
        "image": str(row["image"]),
    } for row in records]
    real_idx = np.asarray([i for i, row in enumerate(record_meta) if row["source"] == "real"], dtype=int)
    all_idx = np.arange(len(records), dtype=int)
    if not torch.cuda.is_available() and args.device.startswith("cuda"):
        raise RuntimeError("CUDA was requested but is unavailable")
    device = torch.device(args.device)
    print(json.dumps({
        "train_object_ids": len(train_ids), "validation_object_ids": len(val_ids),
        "real_train_records": len(real_records), "synth_train_records": len(synth_records),
        "validation_frames": len(val_frames), "device": str(device),
        "feature_version": FEATURE_VERSION,
    }), flush=True)

    feature_cache = output / "train_features.npz"
    feature_meta_path = output / "train_feature_rows.json"
    cached = False
    if feature_cache.is_file() and feature_meta_path.is_file():
        cache_meta = json.loads(feature_meta_path.read_text(encoding="utf-8"))
        if cache_meta.get("feature_version") == FEATURE_VERSION and cache_meta.get("rows") == record_meta:
            with np.load(feature_cache) as features:
                embeddings = features["resnet"].astype(np.float32)
                hand = features["handcrafted"].astype(np.float32)
            cached = embeddings.shape == (len(records), 512) and hand.shape == (len(records), len(_HAND_FEATURE_NAMES))
    if cached:
        print(f"loaded cached train features: {feature_cache}", flush=True)
    else:
        embeddings, hand, extracted_meta = _extract_records(records, args.batch_size, device)
        comparable_meta = [{key: row[key] for key in record_meta[0]} for row in extracted_meta]
        if comparable_meta != record_meta:
            raise AssertionError("Feature extraction metadata ordering changed")
        np.savez_compressed(feature_cache, resnet=embeddings, handcrafted=hand)
        feature_meta_path.write_text(json.dumps({
            "feature_version": FEATURE_VERSION, "hand_feature_names": _HAND_FEATURE_NAMES,
            "rows": record_meta,
        }, indent=2), encoding="utf-8")

    cv_summaries, cv_detail = _run_group_cv(embeddings, hand, record_meta, real_idx, all_idx)
    cv_detail["feature_version"] = FEATURE_VERSION
    cv_detail["outer_validation_ids_used_for_selection"] = False
    cv_detail["synthetic_weight_per_object_relative_to_real"] = SYNTHETIC_OBJECT_WEIGHT
    (output / "grouped_cv.json").write_text(json.dumps(cv_detail, indent=2), encoding="utf-8")
    selected = sorted(cv_summaries, key=lambda row: (
        -row["cv_mean_material_macro_accuracy"],
        row["cv_std_material_macro_accuracy"],
        row["train_source"] != "real",
        row["features"] != "resnet",
        row["classifier"],
    ))[0]
    print("selected by grouped train-only CV:", json.dumps(selected, sort_keys=True), flush=True)

    selected_sources = real_idx if selected["train_source"] == "real" else all_idx
    final_x = _features_for(embeddings, hand, selected["features"])
    final_y = np.asarray([record_meta[int(i)]["material"] for i in selected_sources])
    final_weights = _training_weights(record_meta, selected_sources, SYNTHETIC_OBJECT_WEIGHT)
    final_model = _fit(_make_model(selected["classifier"]), final_x[selected_sources], final_y, final_weights)
    model_path = output / "material_glare_model.joblib"
    joblib.dump({
        "model": final_model,
        "features": selected["features"],
        "classifier": selected["classifier"],
        "feature_version": FEATURE_VERSION,
        "hand_feature_names": _HAND_FEATURE_NAMES,
        "material_order": MATERIALS,
        "training_object_ids": sorted(train_ids),
        "training_source": selected["train_source"],
        "cv_summary": selected,
    }, model_path)

    candidate_path = (project / args.candidate).resolve()
    split_path = (project / args.split).resolve()
    detector_path = (project / args.detector).resolve()
    score_path = (project / args.score_path).resolve()
    for required in (candidate_path, split_path, detector_path, score_path):
        if not required.is_file():
            raise FileNotFoundError(required)
    metric = load_original_score(score_path)
    labels_path = project / "logs" / "validation" / "C_joint" / "rgb_only" / "val_labels.json"
    if not labels_path.is_file():
        raise FileNotFoundError(labels_path)
    baseline_metrics, _ = score_submission(score_path, labels_path, candidate_path)
    base_frame, base_parsed, raw_parts = _load_predictions(candidate_path, metric)
    detector_records, detector_counts = _val_detection_records(
        data_root, val_frames, detector_path, args.conf, args.imgsz, args.max_det, args.device
    )
    if set(base_parsed) != set(val_frames):
        raise ValueError("Candidate submission frame IDs differ from the saved outer validation fold")
    if set(str(x) for x in base_frame["image_id"]) != set(val_frames):
        raise ValueError("Candidate CSV rows differ from the outer validation frame IDs")

    val_embeddings = _embed_crops([row["crop"] for row in detector_records], args.batch_size, device)
    val_hand = np.asarray([[row["hand"][name] for name in _HAND_FEATURE_NAMES]
                           for row in detector_records], dtype=np.float32).reshape(-1, len(_HAND_FEATURE_NAMES))
    val_x = _features_for(val_embeddings, val_hand, selected["features"])
    val_materials = final_model.predict(val_x) if len(val_x) else np.asarray([], dtype=str)
    for row, pred in zip(detector_records, val_materials):
        row["material_prediction"] = str(pred)

    by_frame: dict[str, list[int]] = defaultdict(list)
    for i, row in enumerate(detector_records):
        by_frame[row["image_id"]].append(i)
    changed = 0
    unmatched = 0
    output_rows = []
    for row in base_frame.itertuples(index=False):
        fid = str(row.image_id)
        raw_objects = raw_parts[fid]
        parsed_objects = base_parsed[fid]
        available = set(by_frame.get(fid, []))
        changed_parts = list(raw_objects)
        for obj_idx, obj in enumerate(parsed_objects):
            if not available:
                unmatched += 1
                continue
            best_idx = max(available, key=lambda idx: _iou(obj["box"], detector_records[idx]["box"]))
            overlap = _iou(obj["box"], detector_records[best_idx]["box"])
            if overlap < 0.70:
                unmatched += 1
                continue
            available.remove(best_idx)
            tokens = changed_parts[obj_idx].split()
            if len(tokens) != 10:
                raise ValueError(f"Expected ten tokens in {fid}: {changed_parts[obj_idx]}")
            tokens[6] = detector_records[best_idx]["material_prediction"]
            changed_parts[obj_idx] = " ".join(tokens)
            changed += 1
        output_rows.append({
            "image_id": fid,
            "prediction": ";".join(changed_parts) if changed_parts else "none",
        })
    output_csv = output / "outer_material_only_submission.csv"
    pd.DataFrame(output_rows, columns=["image_id", "prediction"]).to_csv(output_csv, index=False)
    candidate_metrics, score_state = score_submission(score_path, labels_path, output_csv)
    confusion = _material_confusion(score_state, score_state["score_module"])
    report = {
        "experiment": "grouped_cv_resnet18_rgb_depth_glare_material",
        "feature_version": FEATURE_VERSION,
        "selected_model": selected,
        "training_object_ids": sorted(train_ids),
        "validation_object_ids": sorted(val_ids),
        "training_source_counts": dict(Counter(row["source"] for row in record_meta)),
        "validation_detector": detector_counts,
        "candidate_predictions_changed_material": changed,
        "candidate_predictions_unmatched_to_detector": unmatched,
        "baseline_score_components": baseline_metrics,
        "material_only_score_components": candidate_metrics,
        "material_classification_detail": confusion,
        "grouped_cv_path": str(output / "grouped_cv.json"),
        "trained_model_path": str(model_path),
        "output_submission": str(output_csv),
        "outer_validation_used_for_model_selection": False,
        "depth_valid_rule": "mask & finite & (depth_mm > 50) & (depth_mm < 8000)",
        "depth_holes_inpainted": False,
        "changed_submission_fields": ["material only"],
    }
    (output / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
