from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

if __package__ in (None, ""):
    project_dir = Path(__file__).resolve().parent
    sys.path.insert(0, str(project_dir.parent))
    __package__ = project_dir.name

from .datasets.common import PROJECT_ROOT, resolve_data_root
from .datasets.objects import object_records
from .datasets.splits import load_or_create_split
from .training.attributes import train_attribute_model
from .training.detector import train_detector


def _records(data_root: Path, coco_name: str, frame_ids: list[str]) -> list[dict]:
    return object_records(data_root, coco_name, set(frame_ids))


def main() -> None:
    default_config_path = PROJECT_ROOT / "configs" / "default.json"
    default_config = json.loads(default_config_path.read_text(encoding="utf-8"))
    parser = argparse.ArgumentParser(description="Train the Astramis RGB-D detector and attributes.")
    parser.add_argument("--config", default=str(default_config_path))
    parser.add_argument("--data-root", default=None)
    parser.add_argument("--output", default=str(PROJECT_ROOT))
    parser.add_argument("--experiments", nargs="+", default=None)
    parser.add_argument("--fusion", default=None,
                        choices=["rgb_only", "late", "intermediate", "intermediate_geom"])
    parser.add_argument("--fusion-ablation", action="store_true",
                        help="Also train RGB-only, RGB-D, and RGB-D+geometry attribute heads on the joint split.")
    parser.add_argument("--no-fusion-ablation", action="store_false", dest="fusion_ablation",
                        help="Skip the fusion ablation configured by default.")
    parser.set_defaults(fusion_ablation=default_config.get("run_fusion_ablation", False))
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--group-folds", type=int, default=None)
    parser.add_argument("--validation-fold", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--synth-pretrain-epochs", type=int, default=None)
    parser.add_argument("--attribute-epochs", type=int, default=None)
    parser.add_argument("--synth-attribute-pretrain-epochs", type=int, default=None)
    parser.add_argument("--batch", type=int, default=None)
    parser.add_argument("--attribute-batch", type=int, default=None)
    parser.add_argument("--image-size", type=int, default=None)
    parser.add_argument("--attribute-image-size", type=int, default=None)
    parser.add_argument("--geometry-size", type=int, default=None)
    parser.add_argument("--geometry-cell-m", type=float, default=None)
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--skip-detector", action="store_true")
    parser.add_argument("--detector-model", default=None)
    args = parser.parse_args()

    config_path = Path(args.config).resolve()
    default_config = json.loads(config_path.read_text(encoding="utf-8"))
    experiments = args.experiments or default_config["experiments"]
    seed = args.seed if args.seed is not None else int(default_config["seed"])
    group_folds = args.group_folds if args.group_folds is not None else int(default_config["group_folds"])
    validation_fold = args.validation_fold if args.validation_fold is not None else int(default_config["validation_fold"])
    device = args.device or default_config.get("device", "0")
    image_size = args.image_size or int(default_config["image_size"])
    batch = args.batch or int(default_config["detector_batch"])
    workers = args.workers if args.workers is not None else int(default_config["detector_workers"])
    attribute_batch = args.attribute_batch or int(default_config["attribute_batch"])
    attribute_image_size = args.attribute_image_size or int(default_config["attribute_image_size"])
    geometry_size = args.geometry_size or int(default_config["geometry_image_size"])
    geometry_cell_m = args.geometry_cell_m or float(default_config["geometry_cell_m"])
    detector_model = args.detector_model or default_config["detector_model"]
    primary_fusion = args.fusion or default_config["primary_fusion"]
    data_root = resolve_data_root(args.data_root)
    output_root = Path(args.output).resolve()
    for name in ("checkpoints", "logs", "datasets/yolo"):
        (output_root / name).mkdir(parents=True, exist_ok=True)
    split_path = output_root / "configs" / "object_id_split.json"
    split = load_or_create_split(data_root, split_path, seed, group_folds, validation_fold)
    (output_root / "logs" / "object_id_split.json").write_text(
        json.dumps(split, indent=2), encoding="utf-8")
    (output_root / "logs" / "split_summary.json").write_text(json.dumps({
        "seed": split["seed"], "group_folds": split["group_folds"],
        "validation_fold": split["validation_fold"],
        "train_object_ids": len(split["train_object_ids"]),
        "validation_object_ids": len(split["validation_object_ids"]),
        "train_real_frames": len(split["train_real_frames"]),
        "validation_real_frames": len(split["validation_real_frames"]),
        "train_synth_frames": len(split["train_synth_frames"]),
        "validation_materials": split["validation_materials"],
        "validation_object_ids_by_material": split["validation_object_ids_by_material"],
    }, indent=2), encoding="utf-8")

    real_records = _records(data_root, "real_train", split["train_real_frames"])
    synth_records = _records(data_root, "synth_train", split["train_synth_frames"])
    joint_records = real_records + synth_records
    exp_manifest = {"seed": seed, "split": str(split_path), "config": str(config_path), "experiments": {}}

    for experiment in experiments:
        print(f"\n=== {experiment} ===", flush=True)
        experiment_config_path = PROJECT_ROOT / "configs" / "experiments" / f"{experiment}.json"
        if not experiment_config_path.is_file():
            raise FileNotFoundError(f"Missing experiment configuration: {experiment_config_path}")
        run_config = json.loads(experiment_config_path.read_text(encoding="utf-8"))
        run_seed = seed if args.seed is not None else int(run_config.get("seed", seed))
        run_epochs = args.epochs if args.epochs is not None else int(run_config["detector_epochs"])
        run_synth_epochs = args.synth_pretrain_epochs if args.synth_pretrain_epochs is not None else \
            int(run_config.get("synth_pretrain_epochs", max(5, run_epochs // 2)))
        run_attribute_epochs = args.attribute_epochs if args.attribute_epochs is not None else \
            int(run_config["attribute_epochs"])
        run_synth_attribute_epochs = args.synth_attribute_pretrain_epochs if args.synth_attribute_pretrain_epochs is not None else \
            int(run_config.get("synth_attribute_pretrain_epochs", max(5, run_attribute_epochs // 2)))
        run_detector_batch = batch
        if args.batch is None:
            run_detector_batch = int(run_config.get("detector_batch", batch))
        run_attribute_batch = attribute_batch if args.attribute_batch is not None else \
            int(run_config.get("attribute_batch", attribute_batch))
        fusion = primary_fusion if args.fusion is not None else run_config.get("fusion", primary_fusion)
        if args.seed is None:
            run_seed = int(run_config.get("seed", seed))
        exp_dir = output_root / "checkpoints" / "experiments" / experiment
        exp_dir.mkdir(parents=True, exist_ok=True)
        log_dir = output_root / "logs" / experiment
        log_dir.mkdir(parents=True, exist_ok=True)
        resolved_config = {**default_config, **run_config,
                           "seed": run_seed, "detector_epochs": run_epochs,
                           "synth_pretrain_epochs": run_synth_epochs,
                           "attribute_epochs": run_attribute_epochs,
                           "synth_attribute_pretrain_epochs": run_synth_attribute_epochs,
                           "detector_batch": run_detector_batch,
                           "attribute_batch": run_attribute_batch,
                           "fusion": fusion, "device": device}
        (log_dir / "resolved_config.json").write_text(json.dumps(resolved_config, indent=2), encoding="utf-8")
        detector_path = exp_dir / "detector.pt"
        if not args.skip_detector:
            trained_detector = train_detector(
                data_root, split, experiment, output_root / "checkpoints" / "experiments",
                model_name=detector_model, epochs=run_epochs, image_size=image_size,
                batch=run_detector_batch, workers=workers, device=device, seed=run_seed,
                strategy=run_config["detector_strategy"],
                domain_randomized=bool(run_config.get("domain_randomized", False)),
                synth_pretrain_epochs=run_synth_epochs)
            detector_path.parent.mkdir(parents=True, exist_ok=True)
            detector_path.write_bytes(trained_detector.read_bytes())
        elif not detector_path.is_file():
            raise FileNotFoundError(f"--skip-detector requested, but missing {detector_path}")

        attr_path = exp_dir / f"attributes_{fusion}.pt"
        attribute_strategy = run_config["attribute_strategy"]
        if attribute_strategy == "real_only":
            records = real_records
            model_path = train_attribute_model(
                records, attr_path, fusion, run_attribute_epochs,
                run_attribute_batch, run_seed, device,
                data_root=data_root,
                image_size=attribute_image_size, geometry_size=geometry_size, cell_m=geometry_cell_m,
                domain_randomized=bool(run_config.get("domain_randomized", False)),
                log_path=log_dir / f"attributes_{fusion}.jsonl")
        elif attribute_strategy == "synth_pretrain_real_finetune":
            synth_pretrain = exp_dir / f"attributes_synth_pretrain_{fusion}.pt"
            train_attribute_model(
                synth_records, synth_pretrain, fusion, run_synth_attribute_epochs,
                run_attribute_batch, run_seed, device,
                data_root=data_root, scaler_source_records=joint_records,
                image_size=attribute_image_size, geometry_size=geometry_size, cell_m=geometry_cell_m,
                domain_randomized=bool(run_config.get("domain_randomized", False)),
                log_path=log_dir / f"attributes_synth_{fusion}.jsonl")
            model_path = train_attribute_model(
                real_records, attr_path, fusion, run_attribute_epochs,
                run_attribute_batch, run_seed, device,
                data_root=data_root, initial_checkpoint=synth_pretrain,
                scaler_source_records=joint_records,
                image_size=attribute_image_size, geometry_size=geometry_size, cell_m=geometry_cell_m,
                domain_randomized=bool(run_config.get("domain_randomized", False)),
                log_path=log_dir / f"attributes_finetune_{fusion}.jsonl")
        elif attribute_strategy == "joint":
            model_path = train_attribute_model(
                joint_records, attr_path, fusion, run_attribute_epochs,
                run_attribute_batch, run_seed, device,
                data_root=data_root,
                image_size=attribute_image_size, geometry_size=geometry_size, cell_m=geometry_cell_m,
                domain_randomized=bool(run_config.get("domain_randomized", False)),
                log_path=log_dir / f"attributes_{fusion}.jsonl")
        else:
            raise ValueError(f"Unsupported attribute strategy: {attribute_strategy}")
        exp_manifest["experiments"][experiment] = {
            "detector": str(detector_path), "attributes": str(model_path),
            "fusion": fusion, "resolved_config": str(log_dir / "resolved_config.json"),
            "real_train_objects": len(real_records), "synth_train_objects": len(synth_records),
        }
        if args.fusion_ablation and experiment == "C_joint":
            for fusion in ("rgb_only", "intermediate", "intermediate_geom", "late"):
                if fusion == primary_fusion:
                    model_path = attr_path
                else:
                    ablation_path = exp_dir / f"attributes_{fusion}.pt"
                    model_path = train_attribute_model(
                        joint_records, ablation_path, fusion, run_attribute_epochs,
                        run_attribute_batch, run_seed, device,
                        data_root=data_root,
                        image_size=attribute_image_size, geometry_size=geometry_size, cell_m=geometry_cell_m,
                        log_path=log_dir / f"attributes_{fusion}.jsonl")
                exp_manifest["experiments"][f"{experiment}_{fusion}"] = {
                    "detector": str(detector_path), "attributes": str(model_path),
                    "fusion": fusion, "real_train_objects": len(real_records),
                    "synth_train_objects": len(synth_records),
                }

    manifest_path = output_root / "logs" / "training_manifest.json"
    manifest_path.write_text(json.dumps(exp_manifest, indent=2), encoding="utf-8")
    print(f"Training manifest: {manifest_path}", flush=True)


if __name__ == "__main__":
    main()

