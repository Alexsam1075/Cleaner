from __future__ import annotations

import json
import random
import time
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, WeightedRandomSampler

from ..datasets.objects import FORMS, MATERIALS, ObjectDataset
from ..models.classification import FormMaterialClassifier


def _seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def _classification_sampler(data: ObjectDataset, seed: int) -> WeightedRandomSampler:
    """Balance object labels and cap synthetic contribution per object/source."""
    source_counts = Counter(zip(data.object_ids, data.sources))
    sources_by_object: dict[str, set[str]] = {}
    object_labels: dict[str, tuple[int, int]] = {}
    for oid, form, material, source in zip(data.object_ids, data.forms, data.materials, data.sources):
        label = (form, material)
        previous = object_labels.setdefault(oid, label)
        if previous != label:
            raise ValueError(f"Object {oid} has inconsistent form/material labels")
        sources_by_object.setdefault(oid, set()).add(source)

    form_object_counts = Counter(form for form, _ in object_labels.values())
    material_object_counts = Counter(material for _, material in object_labels.values())
    weights = []
    for oid, form, material, source in zip(data.object_ids, data.forms, data.materials, data.sources):
        # For objects present in both domains, target 80:20 real:synth. Divide
        # each source's share across its frames so a long synthetic sequence
        # cannot outweigh the real observations of the same physical object.
        available_sources = sources_by_object[oid]
        source_factors = {name: (1.0 if name == "real" else 0.25) for name in available_sources}
        source_weight = source_factors[source] / sum(source_factors.values())
        class_weight = 1.0 / np.sqrt(max(form_object_counts[form], 1) * material_object_counts[material])
        weights.append(source_weight * class_weight / source_counts[(oid, source)])
    generator = torch.Generator().manual_seed(seed)
    return WeightedRandomSampler(torch.as_tensor(weights, dtype=torch.double),
                                 num_samples=len(data), replacement=True, generator=generator)


def train_form_material_classifier(
    records: list[dict[str, Any]],
    output_path: str | Path,
    data_root: str | Path,
    epochs: int = 28,
    batch_size: int = 32,
    seed: int = 0,
    device: str = "cuda:0",
    image_size: int = 320,
    geometry_size: int = 80,
    cell_m: float = 0.004,
    freeze_rgb_backbone: bool = False,
    log_path: str | Path | None = None,
) -> Path:
    if not records:
        raise ValueError("No training records were supplied to the classifier")
    _seed(seed)
    device_obj = torch.device("cuda:0" if device in {"0", "cuda"} else device)
    dataset = ObjectDataset(records, data_root, image_size, geometry_size, cell_m,
                            augment=True, domain_randomized=False)
    geometry_mean, geometry_std = dataset.fit_geometry_scaler()
    sampler = _classification_sampler(dataset, seed)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=4 if device_obj.type == "cuda" else 0,
        pin_memory=device_obj.type == "cuda",
        drop_last=False,
    )

    model = FormMaterialClassifier(len(dataset.features[0]), pretrained=True).to(device_obj)
    rgb_prefixes = ("rgb_stem.", "rgb_layer1.", "rgb_layer2.", "rgb_layer3.", "rgb_layer4.")
    if freeze_rgb_backbone:
        for name, parameter in model.named_parameters():
            if name.startswith(rgb_prefixes):
                parameter.requires_grad_(False)
    pretrained_params = [p for name, p in model.named_parameters()
                         if name.startswith(rgb_prefixes) and p.requires_grad]
    classifier_params = [p for name, p in model.named_parameters()
                         if not name.startswith(rgb_prefixes) and p.requires_grad]
    parameter_groups = []
    if pretrained_params:
        parameter_groups.append({"params": pretrained_params, "lr": 2.0e-5, "name": "rgb"})
    parameter_groups.append({"params": classifier_params, "lr": 3.0e-4, "name": "heads"})
    optimizer = torch.optim.AdamW(parameter_groups, weight_decay=1.0e-4)
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
    with log_file.open("w", encoding="utf-8") as log:
        for epoch in range(epochs):
            model.train()
            if freeze_rgb_backbone:
                for module in (model.rgb_stem, model.rgb_layer1, model.rgb_layer2,
                               model.rgb_layer3, model.rgb_layer4):
                    module.eval()
            sums = {"form": 0.0, "material": 0.0, "form_correct": 0, "material_correct": 0}
            count = 0
            started = time.time()
            for batch in loader:
                rgb = batch["rgb"].to(device_obj, non_blocking=True)
                geometry = batch["geometry"].to(device_obj, non_blocking=True)
                depth_map = batch["depth_map"].to(device_obj, non_blocking=True)
                geometry = (geometry - torch.as_tensor(geometry_mean, device=device_obj)) / \
                    torch.as_tensor(geometry_std, device=device_obj)
                form_target = batch["form"].to(device_obj, non_blocking=True)
                material_target = batch["material"].to(device_obj, non_blocking=True)
                optimizer.zero_grad(set_to_none=True)
                with torch.autocast(device_type=device_obj.type, dtype=torch.float16, enabled=amp_enabled):
                    result = model(rgb, geometry, depth_map)
                    form_loss = F.cross_entropy(result["form"], form_target)
                    material_loss = F.cross_entropy(result["material"], material_target)
                    total = form_loss + material_loss
                scaler.scale(total).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                scaler.step(optimizer)
                scaler.update()

                n = int(rgb.shape[0])
                count += n
                sums["form"] += float(form_loss.detach()) * n
                sums["material"] += float(material_loss.detach()) * n
                sums["form_correct"] += int((result["form"].argmax(1) == form_target).sum())
                sums["material_correct"] += int((result["material"].argmax(1) == material_target).sum())
            scheduler.step()
            row = {
                "epoch": epoch + 1,
                "samples": count,
                "seconds": round(time.time() - started, 2),
                "loss_form": sums["form"] / max(count, 1),
                "loss_material": sums["material"] / max(count, 1),
                "train_form_accuracy": sums["form_correct"] / max(count, 1),
                "train_material_accuracy": sums["material_correct"] / max(count, 1),
                "lr_pretrained_rgb": next((group["lr"] for group in optimizer.param_groups
                                             if group.get("name") == "rgb"), 0.0),
                "lr_classification_heads": next(group["lr"] for group in optimizer.param_groups
                                                  if group.get("name") == "heads"),
            }
            log.write(json.dumps(row) + "\n")
            log.flush()

    torch.save({
        "model_state": model.state_dict(),
        "model_config": model.config(),
        "geometry_mean": geometry_mean,
        "geometry_std": geometry_std,
        "forms": FORMS,
        "materials": MATERIALS,
        "input_rgb_size": image_size,
        "geometry_size": geometry_size,
        "geometry_cell_m": cell_m,
        "seed": seed,
        "epochs": epochs,
        "freeze_rgb_backbone": freeze_rgb_backbone,
        "trained_on": {
            "objects": len(records),
            "object_ids": sorted({str(record["object_id"]) for record in records}),
        },
        "tasks": ["form", "material"],
        "uses_depth_for_material": True,
        "inputs": ["rgb", "depth_m", "depth_validity", "height_above_floor", "shape_features"],
        "synthetic_source_weight": 0.25,
    }, output)
    return output


def load_form_material_classifier(
    path: str | Path, device: str = "cuda:0"
) -> tuple[FormMaterialClassifier, dict[str, Any], torch.device]:
    device_obj = torch.device("cuda:0" if device in {"0", "cuda"} else device)
    checkpoint = torch.load(path, map_location=device_obj, weights_only=False)
    config = checkpoint["model_config"]
    model = FormMaterialClassifier(config["geometry_features"], pretrained=False)
    model.load_state_dict(checkpoint["model_state"])
    model.eval().to(device_obj)
    return model, checkpoint, device_obj
