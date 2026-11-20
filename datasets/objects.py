from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, WeightedRandomSampler

from ..geometry.features import compute_geometry_features, geometry_input, load_camera, load_plane
from .common import annotation_object_id, frame_id, image_path, load_coco, metadata_path

FORMS = ["bulk", "solid"]
MATERIALS = ["metal", "wood", "plastic", "glass", "paper", "mineral"]
RGB_MEAN = np.asarray([0.485, 0.456, 0.406], dtype=np.float32)
RGB_STD = np.asarray([0.229, 0.224, 0.225], dtype=np.float32)


def polygon_mask(annotation: dict[str, Any], height: int, width: int) -> np.ndarray:
    mask = np.zeros((height, width), dtype=np.uint8)
    segmentation = annotation.get("segmentation", [])
    if isinstance(segmentation, list):
        for polygon in segmentation:
            points = np.asarray(polygon, dtype=np.float32).reshape(-1, 2)
            if len(points) >= 3:
                points = np.rint(points).astype(np.int32)
                cv2.fillPoly(mask, [points], 1)
    elif isinstance(segmentation, dict):
        raise ValueError("RLE masks are not present in this dataset; polygon labels expected")
    return mask.astype(bool)


def object_records(data_root: str | Path, coco_name: str,
                   frame_ids: set[str] | list[str] | None = None) -> list[dict[str, Any]]:
    coco = load_coco(data_root, coco_name)
    allowed = None if frame_ids is None else set(frame_ids)
    records: list[dict[str, Any]] = []
    for image in coco["images"]:
        fid = frame_id(image)
        if allowed is not None and fid not in allowed:
            continue
        for ann in coco["_annotations_by_image"].get(int(image["id"]), []):
            attrs = ann["attributes"]
            form = attrs["form"]
            material = attrs["material"]
            if form not in FORMS or material not in MATERIALS:
                raise ValueError(f"Unexpected labels in annotation {ann['id']}: {form}/{material}")
            records.append({
                "coco_name": coco_name,
                "frame_id": fid,
                "image": str(image_path(data_root, image)),
                "bbox": [float(x) for x in ann["bbox"]],
                "segmentation": ann["segmentation"],
                "object_id": annotation_object_id(ann),
                "form": form,
                "material": material,
                "volume_ml": float(attrs["volume_ml"]),
                "mass_g": float(attrs["mass_g"]),
                "take": int(float(attrs["mass_g"]) <= 500.0),
            })
    return records


def _mask_from_record(record: dict[str, Any], shape: tuple[int, int]) -> np.ndarray:
    mask = np.zeros(shape, dtype=np.uint8)
    for polygon in record["segmentation"]:
        pts = np.asarray(polygon, dtype=np.float32).reshape(-1, 2)
        if len(pts) >= 3:
            cv2.fillPoly(mask, [np.rint(pts).astype(np.int32)], 1)
    return mask.astype(bool)


def _rgb_crop(rgb: np.ndarray, mask: np.ndarray, bbox: list[float], size: int) -> np.ndarray:
    x, y, w, h = bbox
    pad_x, pad_y = max(4, int(w * 0.10)), max(4, int(h * 0.10))
    x1, y1 = max(0, int(x) - pad_x), max(0, int(y) - pad_y)
    x2, y2 = min(rgb.shape[1], int(x + w) + pad_x), min(rgb.shape[0], int(y + h) + pad_y)
    crop = rgb[y1:y2, x1:x2].copy()
    m = mask[y1:y2, x1:x2]
    crop[~m] = 127
    return cv2.resize(crop, (size, size), interpolation=cv2.INTER_AREA)


class ObjectDataset(Dataset):
    """Object crops plus aligned RGB-D and floor-relative geometry features."""

    def __init__(self, records: list[dict[str, Any]], data_root: str | Path,
                 rgb_size: int = 224, geometry_size: int = 64,
                 cell_m: float = 0.004, augment: bool = False,
                 domain_randomized: bool = False) -> None:
        self.records = records
        self.data_root = Path(data_root)
        self.rgb_size = rgb_size
        self.geometry_size = geometry_size
        self.cell_m = cell_m
        self.augment = augment
        self.domain_randomized = domain_randomized
        self.camera = load_camera(self.data_root / "camera.json")
        self.rgb_crops: list[np.ndarray] = []
        self.geometry_maps: list[np.ndarray] = []
        self.features: list[np.ndarray] = []
        self.geometry_meta: list[dict[str, float | None]] = []
        self.forms: list[int] = []
        self.materials: list[int] = []
        self.volumes: list[float] = []
        self.masses: list[float] = []
        self.takes: list[int] = []
        self.object_ids: list[str] = []
        self.sources: list[str] = []
        for record in records:
            path = Path(record["image"])
            bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
            depth_path = path.parent / "depth.png"
            depth = cv2.imread(str(depth_path), cv2.IMREAD_UNCHANGED)
            if bgr is None or depth is None:
                raise FileNotFoundError(f"Cannot read RGB/depth pair for {record['frame_id']}")
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            mask = _mask_from_record(record, depth.shape)
            plane = load_plane(metadata_path(path))
            feature, meta = compute_geometry_features(depth, mask, plane, self.camera, self.cell_m)
            dmap = geometry_input(depth, mask, plane, self.camera, self.geometry_size)
            self.rgb_crops.append(_rgb_crop(rgb, mask, record["bbox"], self.rgb_size))
            self.geometry_maps.append(dmap)
            self.features.append(feature)
            self.geometry_meta.append(meta)
            self.forms.append(FORMS.index(record["form"]))
            self.materials.append(MATERIALS.index(record["material"]))
            self.volumes.append(float(record["volume_ml"]))
            self.masses.append(float(record["mass_g"]))
            self.takes.append(int(record["take"]))
            self.object_ids.append(str(record["object_id"]))
            self.sources.append("synth" if record["coco_name"].startswith("synth") else "real")
        if not self.records:
            raise ValueError("ObjectDataset has no annotated objects")

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        rgb = self.rgb_crops[index].copy()
        geom_map = self.geometry_maps[index].copy()
        if self.augment:
            if np.random.random() < 0.5:
                rgb = cv2.flip(rgb, 1)
                geom_map = geom_map[:, :, ::-1].copy()
            if np.random.random() < 0.8:
                alpha = float(np.random.uniform(0.78, 1.22))
                beta = float(np.random.uniform(-20.0, 20.0))
                rgb = np.clip(rgb.astype(np.float32) * alpha + beta, 0, 255).astype(np.uint8)
            if np.random.random() < 0.35:
                hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV).astype(np.float32)
                hsv[:, :, 0] = np.mod(hsv[:, :, 0] + np.random.uniform(-6, 6), 180)
                hsv[:, :, 1] *= np.random.uniform(0.8, 1.2)
                rgb = cv2.cvtColor(np.clip(hsv, 0, 255).astype(np.uint8), cv2.COLOR_HSV2RGB)
            if self.domain_randomized:
                if np.random.random() < 0.65:
                    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV).astype(np.float32)
                    hsv[:, :, 0] = np.mod(hsv[:, :, 0] + np.random.uniform(-14, 14), 180)
                    hsv[:, :, 1] *= np.random.uniform(0.55, 1.45)
                    hsv[:, :, 2] *= np.random.uniform(0.65, 1.35)
                    rgb = cv2.cvtColor(np.clip(hsv, 0, 255).astype(np.uint8), cv2.COLOR_HSV2RGB)
                if np.random.random() < 0.25:
                    noise = np.random.normal(0.0, np.random.uniform(2.0, 8.0), rgb.shape)
                    rgb = np.clip(rgb.astype(np.float32) + noise, 0, 255).astype(np.uint8)
        rgbf = rgb.astype(np.float32) / 255.0
        rgbf = (rgbf - RGB_MEAN) / RGB_STD
        return {
            "rgb": torch.from_numpy(np.ascontiguousarray(rgbf.transpose(2, 0, 1))).float(),
            "depth_map": torch.from_numpy(np.ascontiguousarray(geom_map)).float(),
            "geometry": torch.from_numpy(self.features[index].copy()).float(),
            "form": torch.tensor(self.forms[index], dtype=torch.long),
            "material": torch.tensor(self.materials[index], dtype=torch.long),
            "volume_log": torch.tensor(np.log1p(self.volumes[index]), dtype=torch.float32),
            "mass_log": torch.tensor(np.log1p(self.masses[index]), dtype=torch.float32),
            "take": torch.tensor(self.takes[index], dtype=torch.float32),
            "index": torch.tensor(index, dtype=torch.long),
        }

    def fit_geometry_scaler(self, indices: list[int] | None = None) -> tuple[np.ndarray, np.ndarray]:
        values = np.stack(self.features if indices is None else [self.features[i] for i in indices])
        mean = values.mean(axis=0).astype(np.float32)
        std = values.std(axis=0).astype(np.float32)
        std[std < 1e-5] = 1.0
        return mean, std

    def material_class_weights(self, indices: list[int] | None = None) -> torch.Tensor:
        ids = range(len(self)) if indices is None else indices
        counts = Counter(self.materials[i] for i in ids)
        weights = np.asarray([1.0 / np.sqrt(max(counts.get(i, 0), 1)) for i in range(len(MATERIALS))], dtype=np.float32)
        weights /= weights.mean()
        return torch.from_numpy(weights)

    def balanced_sampler(self, indices: list[int] | None = None,
                         generator: torch.Generator | None = None) -> WeightedRandomSampler:
        ids = list(range(len(self))) if indices is None else list(indices)
        group_counts = Counter(self.object_ids[i] for i in ids)
        material_counts = Counter(self.materials[i] for i in ids)
        sample_weights = []
        for i in ids:
            source_weight = 1.0 if self.sources[i] == "real" else 0.55
            sample_weights.append(source_weight / group_counts[self.object_ids[i]] / np.sqrt(material_counts[self.materials[i]]))
        return WeightedRandomSampler(torch.as_tensor(sample_weights, dtype=torch.double),
                                     num_samples=len(ids), replacement=True, generator=generator)

