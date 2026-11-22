"""Fit an object-grouped, train-only geometric take threshold and score it.

The threshold is fitted from training annotations only. Validation output is
used only once for the official score; detections, materials and forms remain
from the saved model pipeline while predicted-mask geometry supplies volume.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import StratifiedGroupKFold

if __package__ in (None, ""):
    project_dir = Path(__file__).resolve().parent.parent
    sys.path.insert(0, str(project_dir.parent))
    __package__ = project_dir.name + ".evaluation"

from ..datasets.common import annotation_object_id, frame_id, load_coco, metadata_path, resolve_data_root
from ..datasets.objects import _mask_from_record, object_records
from ..evaluation.original_metric import load_original_score, score_submission
from ..geometry.features import compute_geometry_features, load_camera, load_plane


def _read_depth(path: str | Path) -> np.ndarray | None:
    depth = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if depth is not None and depth.ndim == 3 and depth.shape[2] == 1:
        depth = depth[:, :, 0]
    return depth


def _best_threshold(volumes: np.ndarray, take: np.ndarray) -> tuple[float, float]:
    unique = np.unique(np.asarray(volumes, dtype=np.float64))
    if len(unique) == 0:
        raise ValueError("Cannot fit a volume threshold without samples")
    candidates = [max(0.0, float(unique[0]) - 1e-6)]
    candidates.extend(float(a + (b - a) * 0.5) for a, b in zip(unique[:-1], unique[1:]))
    candidates.append(float(unique[-1]) + 1e-6)
    y = np.asarray(take, dtype=np.int64)
    best_t, best_score = candidates[0], -1.0
    for threshold in candidates:
        pred = (volumes <= threshold).astype(np.int64)
        score = float(balanced_accuracy_score(y, pred))
        if score > best_score or (score == best_score and threshold < best_t):
            best_t, best_score = float(threshold), score
    return best_t, best_score


def _training_rows(data_root: Path, split: dict[str, Any]) -> list[dict[str, Any]]:
    train_ids = set(map(str, split["train_object_ids"]))
    val_ids = set(map(str, split["validation_object_ids"]))
    if train_ids & val_ids:
        raise ValueError("Object-ID split overlaps")
    records = object_records(data_root, "real_train", set(map(str, split["train_real_frames"])))
    camera = load_camera(data_root / "camera.json")
    rows = []
    seen_ids: set[str] = set()
    for record in records:
        oid = str(record["object_id"])
        if oid not in train_ids or oid in val_ids:
            raise ValueError(f"Non-training object reached threshold fit: {oid}")
        seen_ids.add(oid)
        path = Path(record["image"])
        depth = _read_depth(path.parent / "depth.png")
        if depth is None:
            raise FileNotFoundError(path.parent / "depth.png")
        mask = _mask_from_record(record, depth.shape)
        _, meta = compute_geometry_features(
            depth, mask, load_plane(metadata_path(path)), camera, 0.004
        )
        volume = meta.get("geometric_volume_ml")
        if volume is None or not math.isfinite(float(volume)) or float(volume) <= 0:
            continue
        mass = float(record["mass_g"])
        rows.append({
            "frame_id": str(record["frame_id"]),
            "object_id": oid,
            "form": str(record["form"]),
            "material": str(record["material"]),
            "volume_ml": float(volume),
            "mass_g": mass,
            "take": int(mass <= 500.0),
        })
    if seen_ids != train_ids:
        raise ValueError(f"Training annotations do not cover split IDs: {sorted(train_ids-seen_ids)}")
    if not rows or len({row["take"] for row in rows}) != 2:
        raise ValueError("Training annotations must contain both take classes")
    return rows


def _cross_validate(rows: list[dict[str, Any]], folds: int = 4,
                    seed: int = 17) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    y = np.asarray([row["take"] for row in rows], dtype=np.int64)
    groups = np.asarray([row["object_id"] for row in rows])
    x = np.asarray([row["volume_ml"] for row in rows], dtype=np.float64)
    n_heavy = len({row["object_id"] for row in rows if row["take"] == 0})
    n_light = len({row["object_id"] for row in rows if row["take"] == 1})
    n_splits = min(folds, n_heavy, n_light)
    if n_splits < 2:
        raise ValueError(f"Not enough heavy/light object groups for CV: {n_heavy}/{n_light}")
    splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    oof_pred = np.full(len(rows), -1, dtype=np.int64)
    fold_reports = []
    for fold, (train_idx, val_idx) in enumerate(splitter.split(x, y, groups), start=1):
        if set(groups[train_idx]) & set(groups[val_idx]):
            raise AssertionError("A physical object crosses a CV boundary")
        threshold, fit_score = _best_threshold(x[train_idx], y[train_idx])
        oof_pred[val_idx] = (x[val_idx] <= threshold).astype(np.int64)
        fold_reports.append({
            "fold": fold,
            "train_objects": len(set(groups[train_idx])),
            "validation_objects": len(set(groups[val_idx])),
            "train_threshold_ml": threshold,
            "train_balanced_accuracy": fit_score,
            "validation_balanced_accuracy": float(balanced_accuracy_score(y[val_idx], oof_pred[val_idx])),
        })
    if (oof_pred < 0).any():
        raise AssertionError("Some training rows lack OOF predictions")
    overall = {
        "n_splits": n_splits,
        "seed": seed,
        "frame_oof_balanced_accuracy": float(balanced_accuracy_score(y, oof_pred)),
        "folds": fold_reports,
        "heavy_objects": n_heavy,
        "light_objects": n_light,
    }
    return overall, [{**row, "oof_take": int(pred)} for row, pred in zip(rows, oof_pred)]


def _read_csv(path: Path, metric) -> tuple[pd.DataFrame, dict[str, list[dict[str, Any]]], dict[str, list[str]]]:
    frame = pd.read_csv(path)
    if list(frame.columns) != ["image_id", "prediction"]:
        raise ValueError(f"Unexpected submission columns: {path}")
    parsed, raw = {}, {}
    for row in frame.itertuples(index=False):
        fid = str(row.image_id)
        cell = str(row.prediction)
        parsed[fid] = metric._parse(cell, True, fid)
        raw[fid] = [] if cell.strip().lower() in {"", "none"} else [part.strip() for part in cell.split(";")]
        if len(parsed[fid]) != len(raw[fid]):
            raise ValueError(f"Malformed prediction cell for {fid}")
    return frame, parsed, raw


def _candidate(attr_path: Path, geometry_path: Path, output_path: Path,
               threshold_ml: float, metric) -> tuple[int, int]:
    attr_frame, attr, attr_raw = _read_csv(attr_path, metric)
    _, geometry, _ = _read_csv(geometry_path, metric)
    if set(attr) != set(geometry):
        raise ValueError("Attribute/geometry submissions have different image IDs")
    rows = []
    matched = 0
    unmatched = 0
    for row in attr_frame.itertuples(index=False):
        fid = str(row.image_id)
        base_parts = attr_raw[fid]
        geoms = geometry[fid]
        output_parts = []
        for index, pred in enumerate(attr[fid]):
            best = max(geoms, key=lambda item: metric._iou(pred["box"], item["box"]), default=None)
            overlap = metric._iou(pred["box"], best["box"]) if best is not None else 0.0
            tokens = base_parts[index].split()
            if len(tokens) != 10:
                raise ValueError(f"Expected ten fields in {fid}: {base_parts[index]}")
            if best is not None and overlap >= 0.70 and float(best["volume"]) > 0:
                volume = float(best["volume"])
                tokens[7] = f"{volume:.2f}"
                tokens[9] = str(int(volume <= threshold_ml))
                matched += 1
            else:
                unmatched += 1
            output_parts.append(" ".join(tokens))
        rows.append({"image_id": fid, "prediction": ";".join(output_parts) if output_parts else "none"})
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows, columns=["image_id", "prediction"]).to_csv(output_path, index=False)
    return matched, unmatched


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, help="Training/validation output root containing the saved split")
    parser.add_argument("--experiment", default="C_joint")
    parser.add_argument("--fusion", default="rgb_only")
    parser.add_argument("--data-root", default=None)
    parser.add_argument("--score-path", default=None)
    parser.add_argument("--folds", type=int, default=4)
    parser.add_argument("--seed", type=int, default=17)
    args = parser.parse_args()

    project = Path(__file__).resolve().parent.parent
    output_root = Path(args.output).resolve()
    data_root = resolve_data_root(args.data_root)
    split_path = output_root / "configs" / "object_id_split.json"
    if not split_path.is_file():
        raise FileNotFoundError(split_path)
    split = json.loads(split_path.read_text(encoding="utf-8"))
    train_ids = set(map(str, split["train_object_ids"]))
    val_ids = set(map(str, split["validation_object_ids"]))
    if train_ids & val_ids:
        raise ValueError("Training and outer validation IDs overlap")

    rows = _training_rows(data_root, split)
    if {row["object_id"] for row in rows} != train_ids:
        raise AssertionError("Threshold fit did not cover the exact training IDs")
    cv_report, oof_rows = _cross_validate(rows, args.folds, args.seed)
    threshold, train_balanced = _best_threshold(
        np.asarray([row["volume_ml"] for row in rows], dtype=np.float64),
        np.asarray([row["take"] for row in rows], dtype=np.int64),
    )

    result_dir = output_root / "logs" / "validation" / args.experiment / args.fusion
    attr_path = result_dir / "multimodal_submission.csv"
    geometry_path = result_dir / "baseline_logic_same_detector_submission.csv"
    labels_path = result_dir / "val_labels.json"
    score_path = Path(args.score_path).resolve() if args.score_path else project / "references" / "score.py"
    for path in (attr_path, geometry_path, labels_path, score_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    metric = load_original_score(score_path)
    candidate_dir = output_root / "logs" / "score83_candidates" / args.experiment / args.fusion
    candidate_path = candidate_dir / f"geometry_train_only_oof_volume_take_{threshold:.2f}ml.csv"
    matched, unmatched = _candidate(attr_path, geometry_path, candidate_path, threshold, metric)
    metrics, _ = score_submission(score_path, labels_path, candidate_path)

    train_rows_path = candidate_dir / "volume_take_train_rows.csv"
    pd.DataFrame(oof_rows).to_csv(train_rows_path, index=False)
    report = {
        "experiment": "train_only_group_oof_volume_take_threshold",
        "train_object_ids": len(train_ids),
        "outer_validation_object_ids": len(val_ids),
        "training_frames_with_volume": len(rows),
        "final_threshold_ml": threshold,
        "training_resubstitution_balanced_accuracy": train_balanced,
        "grouped_oof": cv_report,
        "geometry_predictions_matched": matched,
        "geometry_predictions_unmatched": unmatched,
        "metrics": metrics,
        "candidate_csv": str(candidate_path),
        "fit_labels_source": "real training annotations from train object IDs only",
        "threshold_feature_source": "GT segmentation geometry in train; predicted-mask geometry at outer validation",
        "outer_validation_used_for_threshold_fit": False,
        "candidate_fields_changed": ["volume", "take"],
    }
    report_path = candidate_path.with_suffix(".json")
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
