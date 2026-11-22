# Astramis RGB-D model

This directory contains an independent first-pass detector and object attribute pipeline. The original `baseline.ipynb`, `score.py`, label files, and source images remain untouched.

## Project deliverables

- [Final submission CSV](deliverables/submission_best.csv) — predictions for 137 test frames.
- [LiveCliner presentation](deliverables/LiveCliner.pptx) — project slides in PowerPoint format.


## Data findings

Run the audit from the workspace root (or `python -m datasets.audit` from this project directory):

```powershell
python -m astramis_model.datasets.audit
```

The audit writes `references/data_audit.md` and `references/data_audit.json`. Labels are read only from `real_train.json` and `synth_train.json`. Test image RGB-D files and sensor metadata are checked for availability; test labels are not loaded.

There are 263 real training frames with 214 objects and 49 empty frames, 600 synthetic frames with 678 objects, and 137 real test frames. The real training set has 40 physical `object_id` values. Every one of these IDs also appears in synthetic training with the same form, material, volume, and mass. The grouped split therefore excludes synthetic frames containing any validation object.

## Architecture

- **RGB detector and segmenter:** pretrained YOLO11n-seg, fine-tuned on the selected real/synthetic training condition. The scorer consumes boxes; instance masks are retained for geometry and overlays.
- **Object attribute model:** pretrained ResNet-18 image encoder, a floor-relative RGB-D geometry branch, and separate form, material, log-volume, log-mass, and take heads. Material loss uses training-set class weights; volume and mass use log targets with Smooth L1 loss.
- **Late fusion:** joins the RGB embedding with handcrafted geometry features. **Intermediate fusion:** joins spatial depth/height features with the RGB feature map; `intermediate_geom` also adds the geometry vector.
- **Geometry:** unprojects valid depth with `camera.json`, normalizes the per-frame `floor_plane`, estimates height above the floor, visible surface area, projected footprint, and a 4 mm cell height-sum volume. Invalid depth remains an explicit validity signal. Geometry is an observed-surface estimate, not true volume under occlusion.
- **Mass and take:** the neural heads are supervised from training labels. RGB-D does not directly measure material density. The density table is used only in the baseline comparator.

The model uses YOLO11 and ResNet-18 pretrained weights rather than training a large vision network from scratch. The AION Lab run uses a dedicated Astramis project directory and its own Python environment; it does not modify AION checkpoints or running services.

## Grouped validation and experiments

The default split is a deterministic four-fold `StratifiedGroupKFold` split over physical object IDs (seed 0, fold 1). It keeps all frames of each object on one side, includes all six materials in validation, and reserves a sample of empty frames for false-positive checks. Synthetic frames containing held-out object IDs are removed before pretraining or joint training.

The reproducible training matrix is:

| Experiment | Detector data | Attribute data |
| --- | --- | --- |
| `A_real` | Real only | Real only |
| `B_synth_then_real` | Synthetic pretraining, real fine-tuning | Synthetic pretraining, real fine-tuning |
| `C_joint` | Joint real + synthetic | Joint real + synthetic |
| `D_domain_randomized` | Joint data with stronger augmentation | Joint data with crop color randomization |

All conditions share the same object-ID split and model configuration. The default run also compares `rgb_only`, `late`, `intermediate`, and `intermediate_geom` heads with the joint training set; pass `--no-fusion-ablation` to skip that additional comparison. Validation uses the original, unchanged `score.py` and reports D, M, V, S, and Score. Error summaries include material, form, and volume ranges.

On AION Lab, create the isolated project environment once, then run the commands below from the project directory:

```bash
bash setup_aionlab.sh
source .venv/bin/activate
python train.py --data-root datasets/raw --device 0
python validate.py --experiment C_joint --fusion intermediate_geom --visualize
python validate.py --experiment A_real --fusion intermediate_geom
python validate.py --experiment B_synth_then_real --fusion intermediate_geom
python validate.py --experiment D_domain_randomized --fusion intermediate_geom
python validate.py --experiment C_joint --fusion rgb_only
python validate.py --experiment C_joint --fusion late
python validate.py --experiment C_joint --fusion intermediate
python evaluation/compare.py
python predict.py --experiment C_joint --fusion intermediate_geom --visualize
```

For a quicker smoke run, reduce `--epochs` and `--attribute-epochs`; such a run is a pipeline check, not a final model comparison. Model checkpoints and logs are written under `checkpoints/` and `logs/`. The generated test file is `submissions/submission.csv`, with the exact `image_id,prediction` columns and row order from `sample_submission.csv`.

## Reproducibility

- Fixed seed and saved object-ID split: `configs/object_id_split.json` (run copy: `logs/object_id_split.json`).
- Each run stores model weights, configuration, training manifest, and JSONL epoch logs.
- Test labels are never read. Validation labels are derived only from `real_train.json`.
- `score.py` is copied to `references/` for remote use and is never edited.
- `visualizations/` contains RGB overlays with predicted masks, boxes, labels, learned volume/mass, and observed geometric volume.
- `requirements-aionlab.txt` pins the torchvision version matching the existing AION Lab PyTorch 2.9.1+cu128 environment. No credentials are stored in this project.
- The raw dataset under `datasets/raw/` and uploaded `*.tar.gz` archives stay on AION Lab and are excluded from Git; restore them there from the provided dataset bundle when needed.

## Current status

Data inventory and architecture code are in place. The final experiment table and score comparison are added to `DEVLOG.md` after the AION Lab training and held-out validation complete. A successful local build or a staged checkpoint alone is not reported as a quality improvement.

