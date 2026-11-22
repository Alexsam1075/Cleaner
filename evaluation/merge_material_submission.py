from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def _objects(cell: str, image_id: str) -> list[list[str]]:
    text = str(cell).strip()
    if text.lower() in {"", "none", "nan"}:
        return []
    result = []
    for index, part in enumerate(text.split(";")):
        tokens = part.split()
        if len(tokens) != 10:
            raise ValueError(f"Malformed prediction in {image_id}, object {index}: expected 10 fields")
        result.append(tokens)
    return result


def _assert_same_detection(base: list[str], classified: list[str], image_id: str, index: int,
                           box_tolerance: float, score_tolerance: float) -> None:
    if base[5] != classified[5]:
        raise AssertionError(f"Form label changed in {image_id}, object {index}")
    for field in range(4):
        if abs(float(base[field]) - float(classified[field])) > box_tolerance:
            raise AssertionError(f"Box changed in {image_id}, object {index}")
    if abs(float(base[4]) - float(classified[4])) > score_tolerance:
        raise AssertionError(f"Detection score changed in {image_id}, object {index}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Copy material labels from a classifier submission while preserving every other field."
    )
    parser.add_argument("--baseline", required=True, help="Submission whose non-material fields are retained")
    parser.add_argument("--classified", required=True, help="Same-frame submission with new material labels")
    parser.add_argument("--output", required=True)
    parser.add_argument("--box-tolerance", type=float, default=1.0e-3)
    parser.add_argument("--score-tolerance", type=float, default=1.0e-5)
    args = parser.parse_args()

    baseline = pd.read_csv(args.baseline, dtype={"image_id": str}, keep_default_na=False)
    classified = pd.read_csv(args.classified, dtype={"image_id": str}, keep_default_na=False)
    if list(baseline.columns) != ["image_id", "prediction"] or list(classified.columns) != ["image_id", "prediction"]:
        raise ValueError("Both CSVs must have columns image_id,prediction")
    if baseline.image_id.tolist() != classified.image_id.tolist():
        raise ValueError("Image IDs or row order differ between submissions")

    merged = []
    changed = 0
    object_count = 0
    for image_id, base_cell, classifier_cell in zip(baseline.image_id, baseline.prediction, classified.prediction):
        base_objects = _objects(base_cell, image_id)
        classified_objects = _objects(classifier_cell, image_id)
        if len(base_objects) != len(classified_objects):
            raise AssertionError(f"Object count changed in {image_id}")
        output_objects = []
        for index, (base, new) in enumerate(zip(base_objects, classified_objects)):
            _assert_same_detection(base, new, image_id, index, args.box_tolerance, args.score_tolerance)
            if new[6] not in {"metal", "wood", "plastic", "glass", "paper", "mineral"}:
                raise ValueError(f"Unknown material {new[6]!r} in {image_id}, object {index}")
            changed += int(base[6] != new[6])
            object_count += 1
            base[6] = new[6]
            output_objects.append(" ".join(base))
        merged.append(";".join(output_objects) if output_objects else "none")

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"image_id": baseline.image_id, "prediction": merged}).to_csv(output_path, index=False)
    print(json.dumps({"output": str(output_path.resolve()), "frames": len(merged),
                      "objects": object_count, "material_labels_changed": changed,
                      "preserved_fields": ["box", "score", "form", "volume_ml", "mass_g", "take"]}, indent=2))


if __name__ == "__main__":
    main()
