from __future__ import annotations

import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Any

from sklearn.model_selection import StratifiedGroupKFold

from .common import annotation_object_id, frame_id, load_coco


def build_group_split(data_root: str | Path, seed: int = 0, folds: int = 4,
                      validation_fold: int = 1, empty_fraction: float = 0.2) -> dict[str, Any]:
    """Split real frames by physical object_id and exclude their synthetic views."""
    real = load_coco(data_root, "real_train")
    synth = load_coco(data_root, "synth_train")
    image_by_id = real["_image_by_id"]
    ann_by_img = real["_annotations_by_image"]
    group_to_label: dict[str, str] = {}
    group_frames: dict[str, list[str]] = defaultdict(list)
    for image_id, anns in ann_by_img.items():
        for ann in anns:
            oid = annotation_object_id(ann)
            material = str(ann["attributes"]["material"])
            if oid in group_to_label and group_to_label[oid] != material:
                raise ValueError(f"Physical object {oid} has conflicting material labels")
            group_to_label[oid] = material
            group_frames[oid].append(frame_id(image_by_id[image_id]))
    groups = sorted(group_to_label)
    if folds < 2 or not 0 <= validation_fold < folds:
        raise ValueError("validation_fold must be in [0, group_folds)")
    splitter = StratifiedGroupKFold(n_splits=folds, shuffle=True, random_state=seed)
    X = list(range(len(groups)))
    labels = [group_to_label[group] for group in groups]
    train_idx, val_idx = list(splitter.split(X, labels, groups))[validation_fold]
    val_groups = {groups[i] for i in val_idx}
    train_groups = set(groups) - val_groups

    annotated_frame_ids = set()
    for image_id, anns in ann_by_img.items():
        annotated_frame_ids.add(frame_id(image_by_id[image_id]))
    empty_ids = sorted(frame_id(image) for image in real["images"]
                       if frame_id(image) not in annotated_frame_ids)
    n_empty_val = min(len(empty_ids), round(len(empty_ids) * empty_fraction))
    empty_val = set(random.Random(seed).sample(empty_ids, n_empty_val)) if n_empty_val else set()

    train_real_frames: list[str] = []
    val_real_frames: list[str] = []
    for image in real["images"]:
        fid = frame_id(image)
        anns = ann_by_img.get(int(image["id"]), [])
        if not anns:
            (val_real_frames if fid in empty_val else train_real_frames).append(fid)
        elif all(annotation_object_id(ann) in val_groups for ann in anns):
            val_real_frames.append(fid)
        elif all(annotation_object_id(ann) in train_groups for ann in anns):
            train_real_frames.append(fid)
        else:
            raise ValueError(f"Frame {fid} contains objects assigned to both split sides")

    train_synth_frames: list[str] = []
    dropped_synth_frames: list[str] = []
    train_synth_object_ids: set[str] = set()
    for image in synth["images"]:
        anns = synth["_annotations_by_image"].get(int(image["id"]), [])
        ids = {annotation_object_id(ann) for ann in anns}
        fid = frame_id(image)
        if ids and ids <= train_groups:
            train_synth_frames.append(fid)
            train_synth_object_ids.update(ids)
        else:
            dropped_synth_frames.append(fid)

    if train_groups & val_groups or train_synth_object_ids & val_groups:
        raise AssertionError("object_id leakage across the grouped split")
    return {
        "seed": seed,
        "group_folds": folds,
        "validation_fold": validation_fold,
        "train_object_ids": sorted(train_groups),
        "validation_object_ids": sorted(val_groups),
        "train_real_frames": sorted(train_real_frames),
        "validation_real_frames": sorted(val_real_frames),
        "empty_validation_frames": sorted(empty_val),
        "train_synth_frames": sorted(train_synth_frames),
        "excluded_synth_frames_with_validation_objects": sorted(dropped_synth_frames),
        "train_synth_object_ids": sorted(train_synth_object_ids),
        "validation_materials": sorted({group_to_label[g] for g in val_groups}),
        "validation_object_ids_by_material": {
            material: sorted(g for g in val_groups if group_to_label[g] == material)
            for material in sorted({group_to_label[g] for g in val_groups})
        },
    }


def load_or_create_split(data_root: str | Path, output_path: str | Path,
                         seed: int = 0, folds: int = 4, validation_fold: int = 1,
                         empty_fraction: float = 0.2) -> dict[str, Any]:
    path = Path(output_path)
    if path.is_file():
        split = json.loads(path.read_text(encoding="utf-8"))
        if (split.get("seed") == seed and split.get("group_folds") == folds and
                split.get("validation_fold") == validation_fold and
                split.get("empty_validation_frames") is not None and
                "validation_object_ids_by_material" in split):
            return split
    split = build_group_split(data_root, seed, folds, validation_fold, empty_fraction)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(split, indent=2), encoding="utf-8")
    return split

