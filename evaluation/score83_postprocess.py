"""Evaluate a train-only density + predicted-geometry take policy.

This keeps the detector and RGB-D attribute predictions fixed, uses the
geometry-only submission produced by validate.py for depth volume, and fits
class densities exclusively from the exact training object IDs in the saved
split. It intentionally does not duplicate detections or tune on validation.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import pandas as pd
import cv2
import numpy as np

if __package__ in (None, ""):
    project_dir = Path(__file__).resolve().parent.parent
    sys.path.insert(0, str(project_dir.parent))
    __package__ = project_dir.name + ".evaluation"

from ..datasets.common import annotation_object_id, load_coco, metadata_path, resolve_data_root
from ..datasets.objects import _mask_from_record, object_records
from ..geometry.features import compute_geometry_features, load_camera, load_plane
from .original_metric import load_original_score, score_submission
from .train_only_volume_take import _best_threshold, _cross_validate, _training_rows


def _load_predictions(path: Path, metric) -> dict[str, list[dict]]:
    frame = pd.read_csv(path)
    if list(frame.columns) != ["image_id", "prediction"]:
        raise ValueError(f"Unexpected submission columns in {path}")
    return {
        str(row.image_id): metric._parse(str(row.prediction), True, str(row.image_id))
        for row in frame.itertuples(index=False)
    }


def _train_statistics(
    data_root: Path, split: dict, fit_volume_calibration: bool
) -> tuple[dict, dict, float, dict, float, int]:
    train_ids = set(map(str, split["train_object_ids"]))
    coco = load_coco(data_root, "real_train")
    camera = load_camera(data_root / "camera.json")
    records = object_records(data_root, "real_train", split["train_real_frames"])
    by_object: dict[str, tuple[str, str, float]] = {}
    for ann in coco["annotations"]:
        attrs = ann["attributes"]
        oid = annotation_object_id(ann)
        if oid not in train_ids:
            continue
        volume = float(attrs["volume_ml"])
        mass = float(attrs["mass_g"])
        if volume <= 0 or mass < 0:
            continue
        value = (str(attrs["form"]), str(attrs["material"]), mass / volume)
        previous = by_object.setdefault(oid, value)
        if previous != value:
            raise ValueError(f"Inconsistent density labels for object_id={oid}")
    if set(by_object) != train_ids:
        raise ValueError(f"Density fitting IDs differ from train split: missing={sorted(train_ids-set(by_object))}")

    by_class: dict[tuple[str, str], list[float]] = defaultdict(list)
    by_material: dict[str, list[float]] = defaultdict(list)
    all_values = []
    for form, material, density in by_object.values():
        by_class[(form, material)].append(density)
        by_material[material].append(density)
        all_values.append(density)
    class_mean = {key: sum(values) / len(values) for key, values in by_class.items()}
    material_mean = {key: sum(values) / len(values) for key, values in by_material.items()}
    global_mean = sum(all_values) / len(all_values)

    class_volume_scale: dict[tuple[str, str], float] = {}
    global_volume_scale = 1.0
    if fit_volume_calibration:
        # Fit only on training IDs and average repeated views per object first,
        # so objects with more frames cannot dominate the calibration.
        object_ratios: dict[str, list[float]] = defaultdict(list)
        object_classes: dict[str, tuple[str, str]] = {}
        for record in records:
            oid = str(record["object_id"])
            if oid not in train_ids:
                raise ValueError(f"Unexpected non-training object in calibration records: {oid}")
            target_volume = float(record["volume_ml"])
            if target_volume <= 0:
                continue
            depth_path = Path(record["image"]).parent / "depth.png"
            depth = cv2.imread(str(depth_path), cv2.IMREAD_UNCHANGED)
            if depth is None:
                raise FileNotFoundError(depth_path)
            if depth.ndim == 3:
                depth = depth[..., 0]
            mask = _mask_from_record(record, depth.shape)
            plane = load_plane(metadata_path(record["image"]))
            _, meta = compute_geometry_features(depth, mask, plane, camera, 0.004)
            observed = meta.get("geometric_volume_ml")
            if observed is None or float(observed) <= 0:
                continue
            object_ratios[oid].append(float(np.log(target_volume / float(observed))))
            object_classes[oid] = (str(record["form"]), str(record["material"]))

        log_ratio_by_class: dict[tuple[str, str], list[float]] = defaultdict(list)
        log_ratios = []
        for oid, values in object_ratios.items():
            ratio = float(np.mean(values))
            log_ratios.append(ratio)
            log_ratio_by_class[object_classes[oid]].append(ratio)
        global_log_bias = float(np.mean(log_ratios)) if log_ratios else 0.0
        global_volume_scale = float(np.exp(global_log_bias))
        shrink = 2.0
        for key, values in log_ratio_by_class.items():
            weight = len(values) / (len(values) + shrink)
            class_volume_scale[key] = float(np.exp(
                weight * float(np.mean(values)) + (1.0 - weight) * global_log_bias
            ))
    return class_mean, material_mean, global_mean, class_volume_scale, global_volume_scale, len(by_object)


def _serialize(predictions: list[dict]) -> str:
    cells = []
    for p in predictions[:30]:
        x1, y1, x2, y2 = map(float, p["box"])
        cells.append(
            f"{x1:.1f} {y1:.1f} {x2:.1f} {y2:.1f} {float(p['score']):.5f} "
            f"{p['form']} {p['material']} {float(p['volume']):.2f} "
            f"{float(p.get('mass', 0.0)):.2f} {int(p['take'])}"
        )
    return ";".join(cells) if cells else "none"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, help="validate.py output root")
    parser.add_argument("--experiment", required=True)
    parser.add_argument("--fusion", default="intermediate_geom")
    parser.add_argument("--data-root", default=None)
    parser.add_argument("--score-path", default=None)
    parser.add_argument("--threshold-g", type=float, default=500.0)
    parser.add_argument("--volume-calibration", choices=("none", "form_material"), default="none")
    parser.add_argument("--take-policy", choices=("mass", "train_only_oof_volume"), default="mass")
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parent.parent
    output_root = Path(args.output).resolve()
    data_root = resolve_data_root(args.data_root)
    config_split_path = output_root / "configs" / "object_id_split.json"
    log_split_path = output_root / "logs" / "object_id_split.json"
    split_path = log_split_path if log_split_path.is_file() else config_split_path
    if not split_path.is_file():
        raise FileNotFoundError(f"The exact saved object split is required: {split_path}")
    split = json.loads(split_path.read_text(encoding="utf-8"))
    train_ids = set(map(str, split["train_object_ids"]))
    val_ids = set(map(str, split["validation_object_ids"]))
    if train_ids & val_ids:
        raise ValueError("Training and validation object IDs overlap")

    result_dir = output_root / "logs" / "validation" / args.experiment / args.fusion
    labels_path = result_dir / "val_labels.json"
    attr_path = result_dir / "multimodal_submission.csv"
    geometry_path = result_dir / "baseline_logic_same_detector_submission.csv"
    for path in (labels_path, attr_path, geometry_path):
        if not path.is_file():
            raise FileNotFoundError(path)

    metric_path = Path(args.score_path).resolve() if args.score_path else project_root / "references" / "score.py"
    metric = load_original_score(metric_path)
    class_density, material_density, global_density, class_volume_scale, global_volume_scale, n_fit = \
        _train_statistics(data_root, split, args.volume_calibration == "form_material")
    take_threshold_ml = None
    take_cv_report = None
    take_train_balanced_accuracy = None
    if args.take_policy == "train_only_oof_volume":
        threshold_rows = _training_rows(data_root, split)
        take_cv_report, _ = _cross_validate(threshold_rows, folds=4, seed=17)
        take_threshold_ml, take_train_balanced_accuracy = _best_threshold(
            np.asarray([row["volume_ml"] for row in threshold_rows], dtype=np.float64),
            np.asarray([row["take"] for row in threshold_rows], dtype=np.int64),
        )
    attr = _load_predictions(attr_path, metric)
    geometry = _load_predictions(geometry_path, metric)
    if set(attr) != set(geometry):
        raise ValueError("Attribute and geometry submissions have different image IDs")

    rows = []
    matched = 0
    density_fallbacks = 0
    for image_id, predictions in attr.items():
        geometric_predictions = geometry[image_id]
        for p in predictions:
            best = max(geometric_predictions,
                       key=lambda g: metric._iou(p["box"], g["box"]), default=None)
            iou = metric._iou(p["box"], best["box"]) if best is not None else 0.0
            raw_geometric_volume = None
            if best is not None and iou >= 0.70 and float(best["volume"]) > 0:
                raw_geometric_volume = float(best["volume"])
                p["volume"] = raw_geometric_volume
                matched += 1
            key = (str(p["form"]), str(p["material"]))
            if args.volume_calibration == "form_material":
                p["volume"] *= class_volume_scale.get(key, global_volume_scale)
            density = class_density.get(key)
            if density is None:
                density = material_density.get(key[1], global_density)
                density_fallbacks += 1
            p["mass"] = float(p["volume"]) * density
            if args.take_policy == "train_only_oof_volume":
                if raw_geometric_volume is None or take_threshold_ml is None:
                    p["take"] = int(p["mass"] <= args.threshold_g)
                else:
                    p["take"] = int(raw_geometric_volume <= take_threshold_ml)
            else:
                p["take"] = int(p["mass"] <= args.threshold_g)
        rows.append({"image_id": image_id, "prediction": _serialize(predictions)})

    candidate_dir = output_root / "logs" / "score83_candidates" / args.experiment / args.fusion
    candidate_dir.mkdir(parents=True, exist_ok=True)
    take_suffix = (f"train_oof_volume_{take_threshold_ml:.2f}ml" if take_threshold_ml is not None
                   else f"train_density_take_{int(args.threshold_g)}g")
    candidate_csv = candidate_dir / f"geometry_{args.volume_calibration}_{take_suffix}.csv"
    pd.DataFrame(rows, columns=["image_id", "prediction"]).to_csv(candidate_csv, index=False)
    metrics, _ = score_submission(metric_path, labels_path, candidate_csv)
    report = {
        "experiment": args.experiment,
        "fusion": args.fusion,
        "policy": ("calibrated predicted-mask geometry volume + train-only OOF geometric take threshold"
                   if args.take_policy == "train_only_oof_volume" else
                   "predicted-mask geometry volume + training-object-only class density + fixed mass threshold"),
        "take_policy": args.take_policy,
        "take_threshold_ml": take_threshold_ml,
        "take_threshold_train_resubstitution_balanced_accuracy": take_train_balanced_accuracy,
        "take_threshold_grouped_oof": take_cv_report,
        "threshold_g": args.threshold_g,
        "volume_calibration": args.volume_calibration,
        "train_object_count": n_fit,
        "validation_object_count": len(val_ids),
        "density_by_form_material": {f"{k[0]}_{k[1]}": v for k, v in class_density.items()},
        "volume_scale_by_form_material": {f"{k[0]}_{k[1]}": v for k, v in class_volume_scale.items()},
        "global_volume_scale": global_volume_scale,
        "predictions_with_geometry_match": matched,
        "density_fallback_predictions": density_fallbacks,
        "metrics": metrics,
        "candidate_csv": str(candidate_csv),
    }
    report_path = candidate_csv.with_suffix(".json")
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
