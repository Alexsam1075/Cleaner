from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

if __package__ in (None, ""):
    project_dir = Path(__file__).resolve().parent.parent
    sys.path.insert(0, str(project_dir.parent))
    __package__ = f"{project_dir.name}.evaluation"

import numpy as np
import pandas as pd
import torch
from ultralytics import YOLO

from ..datasets.common import PROJECT_ROOT, resolve_data_root
from ..evaluation.original_metric import build_validation_labels, error_analysis, score_submission
from ..geometry.features import load_camera
from ..inference.classification import reclassify_predictions
from ..inference.predictor import prediction_cell, predict_frame
from ..training.classification import load_form_material_classifier
from ..training.attributes import load_attribute_model


FROZEN_FIELDS = ("box", "score", "volume_ml", "mass_g", "take")


def _write_submission(frame_ids: list[str], cells: list[str], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"image_id": frame_ids, "prediction": cells}).to_csv(path, index=False)


def _assert_frozen(before: list[dict[str, Any]], after: list[dict[str, Any]], context: str) -> None:
    if len(before) != len(after):
        raise AssertionError(f"{context}: classifier changed the number of predictions")
    for index, (old, new) in enumerate(zip(before, after)):
        for key in FROZEN_FIELDS:
            old_value, new_value = old[key], new[key]
            if isinstance(old_value, (list, tuple, np.ndarray)):
                same = np.array_equal(np.asarray(old_value), np.asarray(new_value))
            else:
                same = old_value == new_value
            if not same:
                raise AssertionError(f"{context}: prediction {index} changed frozen field {key}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate form/material classification while freezing detections, scores, volume, mass and take."
    )
    parser.add_argument("--run-root", required=True, help="Classifier training run root containing the saved split")
    parser.add_argument("--classifier", default=None)
    parser.add_argument("--detector", required=True)
    parser.add_argument("--detector-provenance", required=True,
                        help="detector_run.json from the matching detector training run")
    parser.add_argument("--attributes", required=True,
                        help="Existing attribute checkpoint used only to provide frozen base predictions")
    parser.add_argument("--data-root", default=None)
    parser.add_argument("--score-path", default=None)
    parser.add_argument("--output", default=None)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--conf", type=float, default=0.08)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--name", default="rgbd_resnet18")
    args = parser.parse_args()

    run_root = Path(args.run_root).resolve()
    data_root = resolve_data_root(args.data_root)
    split_path = run_root / "logs" / "object_id_split.json"
    if not split_path.is_file():
        split_path = run_root / "configs" / "object_id_split.json"
    if not split_path.is_file():
        raise FileNotFoundError(f"Exact classifier training split is required: {split_path}")
    split = json.loads(split_path.read_text(encoding="utf-8"))
    required = {"train_object_ids", "validation_object_ids", "validation_real_frames"}
    if not required.issubset(split):
        raise ValueError(f"Incomplete object-ID split: {split_path}")
    train_ids = set(map(str, split["train_object_ids"]))
    validation_ids = set(map(str, split["validation_object_ids"]))
    if train_ids & validation_ids:
        raise ValueError("Training and validation object IDs overlap")

    classifier_path = Path(args.classifier).resolve() if args.classifier else \
        run_root / "checkpoints" / "classification" / "form_material_resnet18.pt"
    classifier, classifier_checkpoint, device = load_form_material_classifier(classifier_path, args.device)
    classifier_ids = set(map(str, classifier_checkpoint.get("trained_on", {}).get("object_ids", [])))
    if not classifier_ids:
        raise ValueError("Classifier checkpoint lacks trained_on.object_ids")
    if classifier_ids & validation_ids:
        raise ValueError(f"Classifier saw validation object IDs: {sorted(classifier_ids & validation_ids)}")
    if classifier_ids != train_ids:
        raise ValueError(f"Classifier training IDs differ from exact run split: missing={sorted(train_ids-classifier_ids)}, "
                         f"unexpected={sorted(classifier_ids-train_ids)}")

    detector_path = Path(args.detector).resolve()
    detector_manifest_path = Path(args.detector_provenance).resolve()
    detector_metadata = json.loads(detector_manifest_path.read_text(encoding="utf-8"))
    detector_ids = set(map(str, detector_metadata.get("train_object_ids", [])))
    if not detector_ids:
        raise ValueError("Detector provenance lacks train_object_ids")
    if detector_ids & validation_ids:
        raise ValueError(f"Detector saw validation object IDs: {sorted(detector_ids & validation_ids)}")

    attr_path = Path(args.attributes).resolve()
    attribute_checkpoint = torch.load(attr_path, map_location="cpu", weights_only=False)
    attr_ids = set(map(str, attribute_checkpoint.get("trained_on", {}).get("object_ids", [])))
    if not attr_ids:
        raise ValueError("Attribute checkpoint lacks trained_on.object_ids")
    if attr_ids & validation_ids:
        raise ValueError(f"Base attribute model saw validation object IDs: {sorted(attr_ids & validation_ids)}")

    score_path = Path(args.score_path).resolve() if args.score_path else \
        (PROJECT_ROOT / "references" / "score.py" if (PROJECT_ROOT / "references" / "score.py").is_file()
         else PROJECT_ROOT.parent / "score.py")
    result_dir = Path(args.output).resolve() if args.output else \
        run_root / "logs" / "classification_validation" / args.name
    labels_path = build_validation_labels(data_root, set(split["validation_real_frames"]),
                                          result_dir / "val_labels.json")
    camera = load_camera(data_root / "camera.json")
    detector = YOLO(str(detector_path))
    attr_model, base_checkpoint, _ = load_attribute_model(attr_path, args.device)

    frame_ids = list(map(str, split["validation_real_frames"]))
    base_cells: list[str] = []
    candidates = {"form_only": [], "form_blend_25": [], "material_only": [],
                  "material_blend_50": [], "form_material": [], "form_blend_25_material": []}
    for frame_id in frame_ids:
        frame = data_root / "real" / "train" / frame_id / "color.png"
        base = predict_frame(frame, detector, attr_model, base_checkpoint, device, camera,
                             args.conf, args.imgsz)
        base_cells.append(prediction_cell(base))
        for name, tasks, blend in (("form_only", ("form",), 1.0),
                                   ("form_blend_25", ("form",), 0.25),
                                   ("material_only", ("material",), 1.0),
                                   ("material_blend_50", ("material",), 0.5),
                                   ("form_material", ("form", "material"), 1.0),
                                   ("form_blend_25_material", ("form", "material"), 1.0)):
            updated = reclassify_predictions(frame, base, classifier, classifier_checkpoint,
                                             camera, device, tasks=tasks,
                                             material_blend=blend,
                                             form_blend=0.25 if "blend_25" in name else 1.0)
            _assert_frozen(base, updated, f"{frame_id}/{name}")
            candidates[name].append(prediction_cell(updated))

    score_module = score_submission
    metrics: dict[str, Any] = {}
    details: dict[str, Any] = {}
    for name, cells in [("base", base_cells), *candidates.items()]:
        csv_path = result_dir / f"{name}_submission.csv"
        _write_submission(frame_ids, cells, csv_path)
        metric, loaded = score_module(score_path, labels_path, csv_path)
        metrics[name] = metric
        details[name] = error_analysis(loaded["ground_truth"], loaded["predictions"], loaded["score_module"])

    report = {
        "split": {"path": str(split_path), "seed": split.get("seed"),
                  "validation_fold": split.get("validation_fold"),
                  "validation_object_ids": sorted(validation_ids),
                  "validation_frames": len(frame_ids)},
        "classifier": str(classifier_path),
        "classifier_trained_ids": sorted(classifier_ids),
        "detector": str(detector_path),
        "detector_provenance": str(detector_manifest_path),
        "detector_trained_ids": sorted(detector_ids),
        "base_attributes": str(attr_path),
        "base_attribute_trained_ids": sorted(attr_ids),
        "score_path": str(score_path),
        "frozen_fields": list(FROZEN_FIELDS),
        "all_frozen_field_checks_passed": True,
        "metrics": metrics,
        "error_analysis": details,
    }
    result_dir.mkdir(parents=True, exist_ok=True)
    (result_dir / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    for name, row in metrics.items():
        print(f"{name}: D={row['D']:.4f} M={row['M']:.4f} V={row['V']:.4f} "
              f"S={row['S']:.4f} Score={row['Score']:.3f}")
    print(f"Classification-only validation report: {result_dir / 'report.json'}")


if __name__ == "__main__":
    main()
