from __future__ import annotations

import json
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = PROJECT_ROOT.parent


def resolve_data_root(value: str | Path | None = None) -> Path:
    if value is not None:
        root = Path(value).expanduser().resolve()
        if not root.exists():
            raise FileNotFoundError(f"Dataset root does not exist: {root}")
        return root
    candidates = [WORKSPACE_ROOT / "dataset", PROJECT_ROOT / "datasets" / "raw"]
    for root in candidates:
        if (root / "labels" / "real_train.json").is_file():
            return root.resolve()
    raise FileNotFoundError(
        "Astramis data not found. Pass --data-root or place it under "
        "<project>/datasets/raw."
    )


def load_json(path: str | Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def load_coco(data_root: str | Path, name: str) -> dict[str, Any]:
    path = Path(data_root) / "labels" / f"{name}.json"
    data = load_json(path)
    data["_image_by_id"] = {int(item["id"]): item for item in data["images"]}
    data["_annotations_by_image"] = {}
    for ann in data["annotations"]:
        data["_annotations_by_image"].setdefault(int(ann["image_id"]), []).append(ann)
    return data


def frame_id(image: dict[str, Any]) -> str:
    return Path(image["file_name"]).parent.name


def image_path(data_root: str | Path, image: dict[str, Any]) -> Path:
    return Path(data_root) / Path(image["file_name"])


def metadata_path(image_path_value: str | Path) -> Path:
    return Path(image_path_value).parent / "meta.json"


def annotation_object_id(ann: dict[str, Any]) -> str:
    value = ann.get("attributes", {}).get("object_id")
    if value is None:
        raise ValueError(f"Annotation {ann.get('id')} has no object_id")
    return str(value)

