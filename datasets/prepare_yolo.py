from __future__ import annotations

import os
import shutil
from pathlib import Path
from typing import Any

import yaml

from .common import frame_id, image_path, load_coco

FORMS = ["bulk", "solid"]


def _link_or_copy(source: Path, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists() or target.is_symlink():
        try:
            if target.resolve() == source.resolve():
                return
        except OSError:
            pass
        target.unlink()
    try:
        os.link(source, target)
    except OSError:
        try:
            target.symlink_to(source)
        except OSError:
            shutil.copy2(source, target)


def _largest_polygon(ann: dict[str, Any]) -> list[float] | None:
    segments = ann.get("segmentation", [])
    if not isinstance(segments, list):
        return None
    valid = [segment for segment in segments if len(segment) >= 6 and len(segment) % 2 == 0]
    if not valid:
        return None
    # Matches the baseline's polygon treatment for rare multi-contour objects.
    def area(poly: list[float]) -> float:
        pts = list(zip(poly[::2], poly[1::2]))
        return abs(sum(pts[i][0] * pts[(i + 1) % len(pts)][1] -
                       pts[(i + 1) % len(pts)][0] * pts[i][1]
                       for i in range(len(pts))) / 2.0)
    return max(valid, key=area)


def _label_text(anns: list[dict[str, Any]], width: int, height: int) -> str:
    lines: list[str] = []
    for ann in anns:
        form = ann["attributes"]["form"]
        polygon = _largest_polygon(ann)
        if polygon is None:
            continue
        coords = []
        for x, y in zip(polygon[::2], polygon[1::2]):
            coords.extend((f"{min(max(float(x) / width, 0.0), 1.0):.6f}",
                           f"{min(max(float(y) / height, 0.0), 1.0):.6f}"))
        lines.append(str(FORMS.index(form)) + " " + " ".join(coords))
    return "\n".join(lines) + ("\n" if lines else "")


def prepare_yolo_dataset(data_root: str | Path, split: dict[str, Any], output_dir: str | Path,
                         train_real: bool = True, train_synth: bool = False,
                         validation: bool = True) -> Path:
    root = Path(output_dir).resolve()
    train_dir = root / "images" / "train"
    val_dir = root / "images" / "val"
    label_train = root / "labels" / "train"
    label_val = root / "labels" / "val"
    train_dir.mkdir(parents=True, exist_ok=True)
    label_train.mkdir(parents=True, exist_ok=True)
    if validation:
        val_dir.mkdir(parents=True, exist_ok=True)
        label_val.mkdir(parents=True, exist_ok=True)
    data_root = Path(data_root).resolve()
    frame_sets: list[tuple[str, list[str]]] = []
    if train_real:
        frame_sets.append(("real_train", list(split["train_real_frames"])))
    if train_synth:
        frame_sets.append(("synth_train", list(split["train_synth_frames"])))
    if not frame_sets:
        raise ValueError("At least one training source must be selected")

    for coco_name, selected in frame_sets:
        coco = load_coco(data_root, coco_name)
        selected_set = set(selected)
        for image in coco["images"]:
            fid = frame_id(image)
            if fid not in selected_set:
                continue
            source = image_path(data_root, image)
            target = train_dir / f"{fid}.png"
            _link_or_copy(source, target)
            anns = coco["_annotations_by_image"].get(int(image["id"]), [])
            (label_train / f"{fid}.txt").write_text(
                _label_text(anns, int(image["width"]), int(image["height"])), encoding="utf-8")

    if validation:
        real = load_coco(data_root, "real_train")
        selected_val = set(split["validation_real_frames"])
        for image in real["images"]:
            fid = frame_id(image)
            if fid not in selected_val:
                continue
            source = image_path(data_root, image)
            _link_or_copy(source, val_dir / f"{fid}.png")
            anns = real["_annotations_by_image"].get(int(image["id"]), [])
            (label_val / f"{fid}.txt").write_text(
                _label_text(anns, int(image["width"]), int(image["height"])), encoding="utf-8")

    yaml_path = root / "dataset.yaml"
    config = {"path": str(root), "train": "images/train",
              "val": "images/val" if validation else "images/train",
              "names": {i: name for i, name in enumerate(FORMS)}}
    yaml_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    return yaml_path

