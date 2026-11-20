from __future__ import annotations

import json
import random
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from ..datasets.objects import MATERIALS, ObjectDataset
from ..geometry.features import FEATURE_NAMES
from ..models.multimodal import MultiModalAttributeModel


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def _device(value: str | None = None) -> torch.device:
    if value:
        if value == "cpu":
            return torch.device("cpu")
        if value in {"0", "cuda"}:
            value = "cuda:0"
        return torch.device(value)
    return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def _seed_worker(worker_id: int) -> None:
    worker_seed = torch.initial_seed() % (2 ** 32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def _read_checkpoint(path: str | Path, model: MultiModalAttributeModel,
                     device: torch.device) -> dict[str, Any]:
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state"])
    return checkpoint


def train_attribute_model(records: list[dict[str, Any]], output_path: str | Path,
                          fusion: str = "intermediate_geom", epochs: int = 28,
                          batch_size: int = 32, seed: int = 0, device: str | None = None,
                          image_size: int = 224, geometry_size: int = 64,
                          cell_m: float = 0.004, data_root: str | Path | None = None,
                          domain_randomized: bool = False,
                          initial_checkpoint: str | Path | None = None,
                          scaler_source_records: list[dict[str, Any]] | None = None,
                          log_path: str | Path | None = None) -> Path:
    if not records:
        raise ValueError("No object records selected for attribute training")
    seed_everything(seed)
    device_obj = _device(device)
    if data_root is None:
        data_root = Path(records[0]["image"]).parents[3]
    data = ObjectDataset(records, data_root, image_size, geometry_size, cell_m,
                         augment=True, domain_randomized=domain_randomized)

    scaler_records = scaler_source_records or records
    scaler_dataset = data if scaler_records is records else ObjectDataset(
        scaler_records, data_root, image_size, geometry_size, cell_m, augment=False)
    geom_mean, geom_std = scaler_dataset.fit_geometry_scaler()
    material_weights = data.material_class_weights().to(device_obj)

    model = MultiModalAttributeModel(len(data.features[0]), fusion=fusion,
                                     pretrained=initial_checkpoint is None).to(device_obj)
    if initial_checkpoint is not None:
        _read_checkpoint(initial_checkpoint, model, device_obj)
    loader_generator = torch.Generator()
    loader_generator.manual_seed(seed)
    sampler = data.balanced_sampler(generator=loader_generator)
    loader = DataLoader(data, batch_size=batch_size, sampler=sampler,
                        num_workers=4 if device_obj.type == "cuda" else 0,
                        pin_memory=device_obj.type == "cuda", drop_last=False,
                        worker_init_fn=_seed_worker, generator=loader_generator)
    head_params = list(model.trunk.parameters()) + list(model.form_head.parameters()) + \
        list(model.material_head.parameters()) + list(model.volume_head.parameters()) + \
        list(model.mass_head.parameters()) + list(model.take_head.parameters())
    other_params = [p for name, p in model.named_parameters()
                    if not any(name.startswith(key) for key in
                               ("trunk", "form_head", "material_head", "volume_head", "mass_head", "take_head"))]
    optimizer = torch.optim.AdamW([
        {"params": other_params, "lr": 2.0e-5},
        {"params": head_params, "lr": 3.0e-4},
    ], weight_decay=1.0e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(epochs, 1))
    amp_enabled = device_obj.type == "cuda"
    try:
        scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)
    except Exception:  # older torch compatibility
        scaler = torch.cuda.amp.GradScaler(enabled=amp_enabled)

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    log_file = Path(log_path) if log_path else output.with_suffix(".jsonl")
    log_file.parent.mkdir(parents=True, exist_ok=True)
    with log_file.open("a", encoding="utf-8") as log:
        for epoch in range(epochs):
            model.train()
            sums = {k: 0.0 for k in ("total", "form", "material", "volume", "mass", "take")}
            count = 0
            started = time.time()
            for batch in loader:
                rgb = batch["rgb"].to(device_obj, non_blocking=True)
                geometry = batch["geometry"].to(device_obj, non_blocking=True)
                geometry = (geometry - torch.as_tensor(geom_mean, device=device_obj)) / \
                    torch.as_tensor(geom_std, device=device_obj)
                depth_map = batch["depth_map"].to(device_obj, non_blocking=True)
                targets = {k: batch[k].to(device_obj, non_blocking=True)
                           for k in ("form", "material", "volume_log", "mass_log", "take")}
                optimizer.zero_grad(set_to_none=True)
                with torch.autocast(device_type=device_obj.type, dtype=torch.float16, enabled=amp_enabled):
                    outputs = model(rgb, geometry, depth_map)
                    losses = {
                        "form": F.cross_entropy(outputs["form"], targets["form"]),
                        "material": F.cross_entropy(outputs["material"], targets["material"],
                                                     weight=material_weights),
                        "volume": F.smooth_l1_loss(outputs["volume_log"], targets["volume_log"]),
                        "mass": F.smooth_l1_loss(outputs["mass_log"], targets["mass_log"]),
                        "take": F.binary_cross_entropy_with_logits(outputs["take_logit"], targets["take"]),
                    }
                    total = losses["form"] + losses["material"] + 0.7 * losses["volume"] + \
                        0.5 * losses["mass"] + 0.5 * losses["take"]
                scaler.scale(total).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                scaler.step(optimizer)
                scaler.update()
                n = int(rgb.shape[0])
                count += n
                sums["total"] += float(total.detach()) * n
                for key, loss in losses.items():
                    sums[key] += float(loss.detach()) * n
            scheduler.step()
            row = {"epoch": epoch + 1, "objects": count,
                   "seconds": round(time.time() - started, 2),
                   "lr_backbone": optimizer.param_groups[0]["lr"],
                   "lr_heads": optimizer.param_groups[1]["lr"],
                   **{f"loss_{key}": value / max(count, 1) for key, value in sums.items()}}
            log.write(json.dumps(row) + "\n")
            log.flush()

    torch.save({
        "model_state": model.state_dict(),
        "model_config": model.config(),
        "geometry_mean": geom_mean,
        "geometry_std": geom_std,
        "geometry_feature_names": FEATURE_NAMES,
        "forms": ["bulk", "solid"],
        "materials": MATERIALS,
        "input_rgb_size": image_size,
        "input_geometry_size": geometry_size,
        "seed": seed,
        "epochs": epochs,
        "trained_on": {"objects": len(records), "object_ids": sorted({r["object_id"] for r in records})},
    }, output)
    return output


def load_attribute_model(path: str | Path, device: str | None = None) -> tuple[MultiModalAttributeModel, dict[str, Any], torch.device]:
    device_obj = _device(device)
    checkpoint = torch.load(path, map_location=device_obj, weights_only=False)
    config = checkpoint["model_config"]
    model = MultiModalAttributeModel(config["geometry_features"],
                                     fusion=config["fusion"], pretrained=False)
    model.load_state_dict(checkpoint["model_state"])
    model.eval().to(device_obj)
    return model, checkpoint, device_obj

