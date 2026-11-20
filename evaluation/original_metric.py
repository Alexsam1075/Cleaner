from __future__ import annotations

import importlib.util
import json
import math
import statistics
from pathlib import Path
from typing import Any

import pandas as pd

from ..datasets.common import frame_id, load_coco


def load_original_score(score_path: str | Path):
    path = Path(score_path).resolve()
    spec = importlib.util.spec_from_file_location("astramis_original_score", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load the supplied score.py at {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def build_validation_labels(data_root: str | Path, frame_ids: set[str], output_path: str | Path) -> Path:
    coco = load_coco(data_root, "real_train")
    images = [image for image in coco["images"] if frame_id(image) in frame_ids]
    image_ids = {int(image["id"]) for image in images}
    data = {key: value for key, value in coco.items()
            if not key.startswith("_") and key not in {"images", "annotations"}}
    data["images"] = images
    data["annotations"] = [ann for ann in coco["annotations"] if int(ann["image_id"]) in image_ids]
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data), encoding="utf-8")
    return path


def score_submission(score_path: str | Path, labels_path: str | Path,
                     submission_path: str | Path) -> tuple[dict[str, float], dict[str, Any]]:
    metric = load_original_score(score_path)
    solution = metric.labels_to_solution(labels_path)
    submission = pd.read_csv(submission_path)
    total = metric.score(solution, submission, "image_id")
    gt = {str(image_id): metric._parse(cell, False, str(image_id))
          for image_id, cell in zip(solution["image_id"], solution["objects"])}
    pred_rows = dict(zip(submission["image_id"].astype(str), submission["prediction"]))
    pred = {image_id: metric._parse(pred_rows.get(image_id, "none"), True, image_id)
            for image_id in gt}
    labels = json.loads(Path(labels_path).read_text(encoding="utf-8"))
    image_names = {int(item["id"]): Path(item["file_name"]).parent.name for item in labels["images"]}
    truth_mass: dict[str, list[float]] = {}
    for ann in labels["annotations"]:
        truth_mass.setdefault(image_names[int(ann["image_id"])], []).append(float(ann["attributes"]["mass_g"]))
    for image_id, objects in gt.items():
        for obj, mass in zip(objects, truth_mass.get(image_id, [])):
            obj["mass_g"] = mass
    for image_id, objects in pred.items():
        raw = pred_rows.get(image_id, "none")
        parts = [] if str(raw).strip().lower() in {"", "none"} else str(raw).split(";")
        for obj, part in zip(objects, parts):
            tokens = part.split()
            if len(tokens) == 10:
                obj["mass_g"] = float(tokens[8])
    parts = metric.components(gt, pred)
    if abs(parts["Score"] - float(total)) > 1e-5:
        raise AssertionError("Original score.py total and component score do not agree")
    return parts, {"ground_truth": gt, "predictions": pred, "solution": solution, "score_module": metric}


def error_analysis(gt: dict[str, list[dict[str, Any]]], pred: dict[str, list[dict[str, Any]]],
                   metric: Any) -> dict[str, Any]:
    materials: dict[str, dict[str, int]] = {}
    forms: dict[str, dict[str, int]] = {}
    volume_bins = {"<50": [0, 0], "50-200": [0, 0], "200-1000": [0, 0], ">=1000": [0, 0]}
    mass_abs_errors: list[float] = []
    mass_error_by_material: dict[str, list[float]] = {}
    missing = 0
    for image_id, objects in gt.items():
        matches = metric._match(objects, pred.get(image_id, []))
        for target, found in zip(objects, matches):
            mat = target["material"]
            forms_name = target["form"]
            materials.setdefault(mat, {"n": 0, "correct": 0})
            forms.setdefault(forms_name, {"n": 0, "matched": 0, "correct_form": 0})
            materials[mat]["n"] += 1
            forms[forms_name]["n"] += 1
            volume = float(target["volume"])
            key = "<50" if volume < 50 else "50-200" if volume < 200 else "200-1000" if volume < 1000 else ">=1000"
            volume_bins[key][1] += 1
            if found is None:
                missing += 1
                continue
            materials[mat]["correct"] += int(found["material"] == mat)
            forms[forms_name]["matched"] += 1
            forms[forms_name]["correct_form"] += int(found["form"] == forms_name)
            if "mass_g" in target and "mass_g" in found:
                mass_error = abs(float(found["mass_g"]) - float(target["mass_g"]))
                mass_abs_errors.append(mass_error)
                mass_error_by_material.setdefault(mat, []).append(mass_error)
            ratio_error = abs(math.log(max(float(found["volume"]), 1e-9) / max(volume, 1e-9)))
            volume_bins[key][0] += int(ratio_error <= __import__("math").log(3.0))
    for values in materials.values():
        values["accuracy"] = values["correct"] / max(values["n"], 1)
    for values in forms.values():
        values["recall_at_iou50"] = values["matched"] / max(values["n"], 1)
        values["correct_form_given_match"] = values["correct_form"] / max(values["matched"], 1)
    mass_by_material = {material: {"n": len(values), "mae_g": float(sum(values) / len(values))}
                        for material, values in mass_error_by_material.items() if values}
    return {"materials": materials, "forms": forms,
            "volume_accuracy_within_3x": {k: (v[0] / v[1] if v[1] else None) for k, v in volume_bins.items()},
            "volume_bin_counts": {k: v[1] for k, v in volume_bins.items()},
            "mass_mae_g_on_matched_objects": float(sum(mass_abs_errors) / len(mass_abs_errors)) if mass_abs_errors else None,
            "mass_median_absolute_error_g": float(statistics.median(mass_abs_errors)) if mass_abs_errors else None,
            "mass_mae_by_material": mass_by_material,
            "unmatched_ground_truth_objects": missing,
            "validation_objects": sum(len(v) for v in gt.values())}

