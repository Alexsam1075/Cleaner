from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

if __package__ in (None, ""):
    project_dir = Path(__file__).resolve().parent
    sys.path.insert(0, str(project_dir.parent))
    __package__ = project_dir.name

import numpy as np
import pandas as pd
from ultralytics import YOLO

from .datasets.common import PROJECT_ROOT, resolve_data_root
from .datasets.splits import load_or_create_split
from .evaluation.baseline import baseline_statistics, predict_baseline_frame
from .evaluation.original_metric import build_validation_labels, error_analysis, score_submission
from .geometry.features import load_camera
from .inference.predictor import predict_frame, prediction_cell
from .training.attributes import load_attribute_model


def _save_csv(ids: list[str], cells: list[str], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"image_id": ids, "prediction": cells}).to_csv(path, index=False)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate on the held-out object_id validation split using the original score.py.")
    parser.add_argument("--data-root", default=None)
    parser.add_argument("--output", default=str(PROJECT_ROOT))
    parser.add_argument("--experiment", default="A_real")
    parser.add_argument("--fusion", default="intermediate_geom")
    parser.add_argument("--detector", default=None)
    parser.add_argument("--attributes", default=None)
    parser.add_argument("--score-path", default=None)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--conf", type=float, default=0.08)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--baseline-only", action="store_true")
    parser.add_argument("--visualize", action="store_true")
    args = parser.parse_args()

    output = Path(args.output).resolve()
    data_root = resolve_data_root(args.data_root)
    split_path = output / "configs" / "object_id_split.json"
    if not split_path.is_file():
        split_path = output / "logs" / "object_id_split.json"
    if not split_path.is_file():
        raise FileNotFoundError(f"Training split is missing: {split_path}")
    split = json.loads(split_path.read_text(encoding="utf-8"))
    required = {"seed", "group_folds", "validation_fold", "train_object_ids",
                "validation_object_ids", "validation_real_frames"}
    if not required.issubset(split):
        raise ValueError(f"Incomplete training split: {split_path}")
    if set(split["train_object_ids"]) & set(split["validation_object_ids"]):
        raise ValueError(f"Object IDs overlap in training split: {split_path}")
    val_ids = split["validation_real_frames"]
    if args.detector:
        detector_path = Path(args.detector).resolve()
    else:
        detector_path = output / "checkpoints" / "experiments" / args.experiment / "detector.pt"
    if args.attributes:
        attr_path = Path(args.attributes).resolve()
    else:
        attr_path = output / "checkpoints" / "experiments" / args.experiment / f"attributes_{args.fusion}.pt"
    score_path = Path(args.score_path).resolve() if args.score_path else \
        (output / "references" / "score.py" if (output / "references" / "score.py").is_file()
         else output.parent / "score.py")
    result_dir = output / "logs" / "validation" / args.experiment / args.fusion
    labels_path = build_validation_labels(data_root, set(val_ids), result_dir / "val_labels.json")
    camera = load_camera(data_root / "camera.json")
    detector = YOLO(str(detector_path))

    rows: dict[str, dict[str, float]] = {}
    details: dict[str, dict] = {}
    baseline = baseline_statistics(data_root, split)
    base_cells = []
    calibrated_cells = []
    for fid in val_ids:
        frame = data_root / "real" / "train" / fid / "color.png"
        pred = predict_baseline_frame(frame, detector, data_root, camera, baseline,
                                     args.conf, args.imgsz, args.device)
        base_cells.append(prediction_cell(pred))
        calibrated = []
        for obj in pred:
            item = dict(obj)
            raw_volume = item["geometry"].get("geometric_volume_ml")
            if raw_volume is not None and raw_volume > 0:
                params = baseline["volume_calibration"]
                item["volume_ml"] = float(np.expm1(params["slope"] * np.log1p(raw_volume) + params["intercept"]))
                item["volume_ml"] = float(np.clip(item["volume_ml"], 1.0, 20000.0))
            else:
                item["volume_ml"] = baseline["median_train_volume_ml"]
            density = baseline["densities"].get(f"{item['form']}|{item['material']}", 1.0)
            item["mass_g"] = float(item["volume_ml"] * density)
            item["take"] = int(item["mass_g"] <= 500.0)
            calibrated.append(item)
        calibrated_cells.append(prediction_cell(calibrated))

    for name, cells in (("baseline_logic_same_detector", base_cells),
                        ("geometry_calibrated_train_fit", calibrated_cells)):
        csv_path = result_dir / f"{name}_submission.csv"
        _save_csv(val_ids, cells, csv_path)
        metrics, loaded = score_submission(score_path, labels_path, csv_path)
        rows[name] = metrics
        details[name] = error_analysis(
            loaded["ground_truth"], loaded["predictions"], loaded["score_module"])

    if not args.baseline_only:
        if not attr_path.is_file():
            raise FileNotFoundError(f"Attribute checkpoint not found: {attr_path}")
        attr_model, checkpoint, device_obj = load_attribute_model(attr_path, args.device)
        pred_cells = []
        visual_dir = output / "visualizations" / "validation" / args.experiment / args.fusion if args.visualize else None
        for n, fid in enumerate(val_ids):
            frame = data_root / "real" / "train" / fid / "color.png"
            predictions = predict_frame(frame, detector, attr_model, checkpoint, device_obj,
                                        camera, args.conf, args.imgsz)
            pred_cells.append(prediction_cell(predictions))
            if visual_dir is not None and n < 24:
                from .evaluation.visualize import save_overlay
                save_overlay(frame, predictions, visual_dir / f"{fid}.png")
        pred_csv = result_dir / "multimodal_submission.csv"
        _save_csv(val_ids, pred_cells, pred_csv)
        metrics, loaded = score_submission(score_path, labels_path, pred_csv)
        rows["multimodal"] = metrics
        details["multimodal"] = error_analysis(
            loaded["ground_truth"], loaded["predictions"], loaded["score_module"])

    report = {
        "experiment": args.experiment,
        "fusion": args.fusion,
        "split": {"seed": split["seed"], "fold": split["validation_fold"],
                  "validation_object_ids": split["validation_object_ids"],
                  "validation_frames": len(val_ids), "validation_objects": details["baseline_logic_same_detector"]["validation_objects"],
                  "validation_materials": split["validation_materials"],
                  "validation_object_ids_by_material": split["validation_object_ids_by_material"]},
        "score_path": str(score_path),
        "detector": str(detector_path),
        "attributes": None if args.baseline_only else str(attr_path),
        "metrics": rows,
        "error_analysis": details,
        "baseline_training_stats": baseline,
    }
    result_dir.mkdir(parents=True, exist_ok=True)
    (result_dir / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    for name, component in rows.items():
        print(f"{name}: D={component['D']:.4f} M={component['M']:.4f} "
              f"V={component['V']:.4f} S={component['S']:.4f} Score={component['Score']:.3f}")
    print(f"Official score.py validation report: {result_dir / 'report.json'}")


if __name__ == "__main__":
    main()

