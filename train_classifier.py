from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

if __package__ in (None, ""):
    project_dir = Path(__file__).resolve().parent
    sys.path.insert(0, str(project_dir.parent))
    __package__ = project_dir.name

from .datasets.common import resolve_data_root
from .datasets.objects import object_records
from .training.classification import train_form_material_classifier


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train only the form/material classifier on the saved object-ID training split."
    )
    parser.add_argument("--output", required=True, help="Run output root containing configs/logs split copies")
    parser.add_argument("--data-root", default=None)
    parser.add_argument("--epochs", type=int, default=28)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--image-size", type=int, default=320)
    parser.add_argument("--geometry-size", type=int, default=80)
    parser.add_argument("--geometry-cell-m", type=float, default=0.004)
    parser.add_argument("--freeze-rgb-backbone", action="store_true",
                        help="Keep ImageNet ResNet-18 fixed and train only classification heads")
    args = parser.parse_args()

    output = Path(args.output).resolve()
    data_root = resolve_data_root(args.data_root)
    config_split = output / "configs" / "object_id_split.json"
    training_log_split = output / "logs" / "object_id_split.json"
    split_path = training_log_split if training_log_split.is_file() else config_split
    if not split_path.is_file():
        raise FileNotFoundError(f"Exact training object-ID split is required: {split_path}")
    split = json.loads(split_path.read_text(encoding="utf-8"))
    required = {"train_object_ids", "validation_object_ids", "train_real_frames", "train_synth_frames"}
    if not required.issubset(split):
        raise ValueError(f"Incomplete saved object-ID split: {split_path}")
    train_ids = set(map(str, split["train_object_ids"]))
    validation_ids = set(map(str, split["validation_object_ids"]))
    if train_ids & validation_ids:
        raise ValueError("Training and validation object IDs overlap")

    real_records = object_records(data_root, "real_train", set(map(str, split["train_real_frames"])))
    synth_records = object_records(data_root, "synth_train", set(map(str, split["train_synth_frames"])))
    records = real_records + synth_records
    record_ids = {str(record["object_id"]) for record in records}
    if not record_ids.issubset(train_ids) or record_ids & validation_ids:
        raise ValueError("Non-training IDs reached classification training")
    if record_ids != train_ids:
        raise ValueError(f"Classifier records do not cover exact train IDs: missing={sorted(train_ids-record_ids)}")

    checkpoint_path = output / "checkpoints" / "classification" / "form_material_resnet18.pt"
    log_path = output / "logs" / "classification" / "form_material_resnet18.jsonl"
    train_form_material_classifier(
        records,
        checkpoint_path,
        data_root,
        epochs=args.epochs,
        batch_size=args.batch,
        seed=args.seed,
        device=args.device,
        image_size=args.image_size,
        geometry_size=args.geometry_size,
        cell_m=args.geometry_cell_m,
        freeze_rgb_backbone=args.freeze_rgb_backbone,
        log_path=log_path,
    )
    manifest = {
        "split_path": str(split_path),
        "seed": split["seed"],
        "group_folds": split["group_folds"],
        "validation_fold": split["validation_fold"],
        "train_object_ids": sorted(train_ids),
        "validation_object_ids": sorted(validation_ids),
        "real_annotation_count": len(real_records),
        "synth_annotation_count": len(synth_records),
        "checkpoint": str(checkpoint_path),
        "tasks": ["form", "material"],
        "architecture": "resnet18_rgbd_form_material_v2",
        "input_rgb_size": args.image_size,
        "geometry_size": args.geometry_size,
        "synthetic_source_weight": 0.25,
        "freeze_rgb_backbone": args.freeze_rgb_backbone,
        "outer_validation_used_for_training": False,
    }
    manifest_path = output / "logs" / "classification" / "training_manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2), flush=True)


if __name__ == "__main__":
    main()
