from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ultralytics import YOLO

from ..datasets.prepare_yolo import prepare_yolo_dataset


def _train_part(model: YOLO, data_yaml: Path, project_dir: Path, name: str,
                epochs: int, image_size: int, batch: int, workers: int,
                device: str | int, validate: bool, domain_randomized: bool,
                seed: int) -> Path:
    common: dict[str, Any] = {
        "data": str(data_yaml),
        "epochs": int(epochs),
        "imgsz": int(image_size),
        "batch": int(batch),
        "workers": int(workers),
        "device": device,
        "project": str(project_dir),
        "name": name,
        "exist_ok": True,
        "seed": int(seed),
        "deterministic": True,
        "optimizer": "AdamW",
        "lr0": 0.001,
        "lrf": 0.01,
        "weight_decay": 0.0005,
        "patience": max(5, min(10, int(epochs) // 2)),
        "amp": True,
        "cache": False,
        "val": bool(validate),
        "save": True,
        "plots": True,
        "verbose": True,
        "close_mosaic": 4,
        "mosaic": 0.0 if domain_randomized else 0.2,
        "mixup": 0.0,
        "copy_paste": 0.0,
        "fliplr": 0.5,
        "flipud": 0.0,
    }
    if domain_randomized:
        common.update({"hsv_h": 0.03, "hsv_s": 0.75, "hsv_v": 0.45,
                       "degrees": 5.0, "translate": 0.10, "scale": 0.35,
                       "shear": 1.0, "perspective": 0.0002})
    model.train(**common)
    save_dir = project_dir / name
    best = save_dir / "weights" / "best.pt"
    last = save_dir / "weights" / "last.pt"
    return best if best.is_file() else last


def train_detector(data_root: str | Path, split: dict[str, Any],
                   experiment: str, output_root: str | Path,
                   model_name: str = "yolo11n-seg.pt", epochs: int = 18,
                   image_size: int = 640, batch: int = 8, workers: int = 4,
                   device: str | int = 0, seed: int = 0,
                   strategy: str | None = None,
                   domain_randomized: bool | None = None,
                   synth_pretrain_epochs: int | None = None) -> Path:
    output_root = Path(output_root).resolve()
    data_root = Path(data_root).resolve()
    exp_dir = output_root / "detector" / experiment
    exp_dir.mkdir(parents=True, exist_ok=True)
    strategy = strategy or experiment
    randomized = (experiment == "D_domain_randomized") if domain_randomized is None else domain_randomized

    if strategy in {"A_real", "real_only"}:
        yaml_path = prepare_yolo_dataset(data_root, split, exp_dir / "data_real",
                                        train_real=True, train_synth=False)
        weight = _train_part(YOLO(model_name), yaml_path, exp_dir, "real_only",
                             epochs, image_size, batch, workers, device, True, False, seed)
    elif strategy in {"B_synth_then_real", "synth_pretrain_real_finetune"}:
        synth_yaml = prepare_yolo_dataset(data_root, split, exp_dir / "data_synth",
                                          train_real=False, train_synth=True, validation=False)
        synth_model = YOLO(model_name)
        pretrain_epochs = int(synth_pretrain_epochs or max(5, epochs // 2))
        synth_weight = _train_part(synth_model, synth_yaml, exp_dir, "synth_pretrain",
                                   pretrain_epochs, image_size, batch, workers, device,
                                   False, False, seed)
        real_yaml = prepare_yolo_dataset(data_root, split, exp_dir / "data_real",
                                         train_real=True, train_synth=False)
        finetuned = YOLO(str(synth_weight))
        weight = _train_part(finetuned, real_yaml, exp_dir, "real_finetune",
                             epochs, image_size, batch, workers, device, True, False, seed)
    elif strategy in {"C_joint", "D_domain_randomized", "joint", "joint_domain_randomized"}:
        if strategy == "joint_domain_randomized":
            randomized = True
        yaml_path = prepare_yolo_dataset(data_root, split, exp_dir / "data_joint",
                                        train_real=True, train_synth=True)
        model = YOLO(model_name)
        weight = _train_part(model, yaml_path, exp_dir, "joint", epochs, image_size,
                             batch, workers, device, True, randomized, seed)
    else:
        raise ValueError(f"Unknown experiment: {experiment}")
    meta = {"experiment": experiment, "weights": str(weight), "seed": int(seed),
            "epochs_requested": epochs, "image_size": image_size,
            "train_object_ids": split["train_object_ids"],
            "validation_object_ids": split["validation_object_ids"],
            "train_synth_frames": len(split["train_synth_frames"])}
    (exp_dir / "detector_run.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return weight

