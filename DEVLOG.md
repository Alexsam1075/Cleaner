# Astramis development log

## 2026-11-20 — Initial audit and implementation

### Before

- The workspace contained the supplied baseline notebook, scorer, submission template, labels, and image folders, but no standalone model project or training scripts.
- The baseline detector was RGB-only YOLO11n-seg. It chose the most frequent material and derived mass from the density table. Sparse depth used a typical-volume fallback.
- The real/synthetic data split had not been checked for physical-object leakage.

### Audit findings

- Real train: 263 frames, 214 object instances, 40 physical object IDs, 49 empty frames.
- Synthetic train: 600 frames, 678 instances, 40 IDs.
- Real test: 137 frames; the submission template contains the same 137 IDs. No test annotation JSON is present.
- Every physical object ID in real train also occurs in synthetic train with matching attributes. Training excludes entire synthetic scenes if any held-out object appears.
- All frames are RGB 1280x720 with aligned 16-bit depth and frame metadata. The RealSense D435 camera intrinsics and per-frame floor plane are available.
- The local Windows machine has CPU-only PyTorch and limited memory. The user redirected training to AION Lab; the remote server was identified read-only as `aion-lab`, with a Tesla V100 16 GB and approximately 95 GB available system memory. The GPU already had other processes using about 5 GB, so this project uses a separate directory and bounded training batch without stopping those processes.

### Changes

- Added a new `astramis_model/` package with grouped splitting, RGB-D geometry, YOLO11 segmentation data preparation, multi-head attribute training, official-score validation, test submission generation, and visualization.
- Added RGB-only, late fusion, intermediate RGB-D fusion, and intermediate RGB-D plus geometry variants.
- Added real-only, synthetic-pretrain/real-finetune, joint-data, and domain-randomized experiment conditions.
- Added a train-only log-volume calibration comparator and kept the supplied `score.py` unchanged.

### Results

At the time of this initial implementation entry, no model result was available. AION Lab training, official-score validation, and test submission results are recorded below.

### Limitations to carry forward

- Forty physical IDs are a small validation population; rare materials have only a few distinct physical examples.
- The 4 mm cell height-sum is a visible-surface approximation and can underestimate hidden volume or be biased by depth holes.
- The neural mass head learns the provided labels; RGB-D does not directly measure material density.
- Baseline comparison uses the same RGB detector and grouped validation while restoring the notebook's dominant-material, geometry-volume, density-mass logic. It isolates attribute/fusion changes and is not a byte-for-byte reproduction of the notebook's original validation split.

## 2026-11-20 - AION Lab shared access

- Host role and health: AION Lab (`aion-lab`, SSH port 4107). The `alexsandrdev` key login remained available; a separate `cleaner` shell account was created for collaborators.
- Remote project path: `/home/alexsandrdev/Projects/Cleaner`. It is owned by `alexsandrdev:cleaner_shared`, mode `2770`, with default group read/write access. The existing `alexsandrdev` owner retains full access.
- The `cleaner` account has no sudo membership. ACLs allow traversal through the owner home and `Projects` without listing sibling projects. A symlink at `/home/cleaner/Cleaner` points to the shared directory.
- Health check: public-key SSH login succeeded; the account could create and remove a file in `Cleaner`; listing `/home/alexsandrdev/Projects` was denied as intended. The supplied `alexsandrdev` SSH profile and `sshd_config` were not changed.
- A separate shared-account key was installed. Its private part is kept outside the project in the Windows user's `.ssh` directory; no credential material is recorded here.
- The dataset archive upload was still in progress at the last check (about 844 MB). Its contents and file permissions were not changed during transfer.
- Revision: the Windows source workspace had no Git metadata when the shared account was configured; the AION Lab repository was initialized later under the remote `Cleaner` path.

## 2026-11-20 - AION Lab archive extraction

- Verified the uploaded code archive (`070b3194448ccf4643ead973829387804e8a5f546aee32ef4b8de2b420c3175f`) and dataset archive (`2d071609b68192261d401fdf5c68ead00062e60a781255cd7d9d04b54e32ae3c`) against the local copies.
- Extracted the project source into `/home/alexsandrdev/Projects/Cleaner` and the dataset into `datasets/raw`; the extracted dataset contains 3,604 files.
- Kept the transfer archives on AION Lab. The existing `.gitignore` excludes both `*.tar.gz` and `datasets/raw/`, so the source repository can remain small while the full data stays on the server.

## 2026-11-20 - Private repository bootstrap

- The initial source commit was `1c15ee5` (`Initial Astramis RGB-D model package`). The user then created the private repository `Alexsam1075/Cleaner`; its original README commit was preserved in the resulting history, with no force-push.
- `origin` is `git@github.com:Alexsam1075/Cleaner.git`. The source and docs were pushed to `main`; the latest source revision at the time of training was `3ecf9ef` (`Fix entrypoint package bootstrap`).
- The raw dataset, transfer archives, checkpoints, logs, visualizations, and submissions are excluded from Git. The 3,604 raw files, checkpoints, and training logs remain on AION Lab; copies of the experiment report, submission, and 24 test overlays are also in the Windows workspace. The shared `cleaner` account retains read/write access to the project and `.git`.

## 2026-11-20 - AION Lab training and validation

### Before

- The RGB-D model package and experiment configs existed, but no weights, official-score results, or test submission had been produced.
- Validation was not yet measured for real-only, synthetic-pretrained, joint, domain-randomized, or fusion-ablation variants.

### After

- Trained A (real only), B (synthetic pretraining then real fine-tuning), C (joint real and filtered synthetic), and D (joint with domain randomization), using seed 0 and a fixed split grouped by physical `object_id`.
- Trained RGB-only, late-fusion, intermediate RGB-D, and intermediate RGB-D plus geometry attribute heads on the C training split. All detector and attribute checkpoints were saved under `checkpoints/experiments/` on AION Lab.
- Evaluated all conditions through the supplied, unchanged `score.py`. The fixed held-out fold contains 10 physical IDs, 67 real frames, and 57 annotated objects. Synthetic frames containing validation IDs were excluded from training. No test labels were used.
- Expanded the data audit in `reports/experiment_report.md`: all 40 IDs overlap between real and synth with matching labels; no malformed train annotations or missing required files were found. The report records class counts, volume/mass quartiles and outliers, camera/floor-plane metadata, and depth validity coverage from a 30-map-per-split sample.

| Data strategy | D (mAP50) | M | V | S (`take`) | Score |
| --- | ---: | ---: | ---: | ---: | ---: |
| A — real only | 0.9562 | 0.4750 | 0.5081 | 0.5000 | 67.948 |
| B — synthetic then real | 0.9552 | 0.6167 | 0.5105 | 0.3646 | 68.774 |
| C — joint | 0.9344 | 0.6100 | 0.5379 | 0.3542 | 68.337 |
| D — domain randomized | 0.9696 | 0.5625 | 0.5513 | 0.3542 | 69.130 |

| C fusion head | D (mAP50) | M | V | S (`take`) | Score |
| --- | ---: | ---: | ---: | ---: | ---: |
| RGB only | 0.9714 | 0.7067 | 0.3040 | 0.4062 | 66.685 |
| Late RGB-D | 0.9714 | 0.5700 | 0.4943 | 0.3750 | 68.241 |
| Intermediate RGB-D | 0.9714 | 0.5833 | 0.3356 | 0.4062 | 65.007 |
| Intermediate RGB-D plus geometry | 0.9344 | 0.6100 | 0.5379 | 0.3542 | 68.337 |

- D had the highest observed strategy Score on this fold. B exceeded A by 0.826 points; that is an encouraging synthetic-data signal on this split, not proof of generalization. For fusion, intermediate RGB-D plus geometry and late fusion were nearly tied; RGB-only raised material accuracy while reducing volume accuracy.
- Training-fit geometric volume calibration did not improve the C baseline comparator on this fold: Score 64.808 versus 64.861 before calibration. The fitted value is retained as a comparison, not claimed as an improvement.
- Generated `submissions/submission.csv` with D weights. It has 137 data rows, matching the 137 IDs in `sample_submission.csv`. Saved 24 test-frame overlays under `visualizations/test/`; the C validation run also saved overlays for inspection.
- The full per-condition results and error breakdowns are in `reports/experiment_report.md` and `logs/validation/`. Training logs and checkpoints remain on AION Lab and are not committed.

### Limits and next work

- This is one grouped fold with only 10 held-out physical IDs; the highest Score is a model-selection result, not a stable estimate. More physical objects and grouped cross-validation are needed before deployment claims.
- The best observed model has S=0.3542, so `take` remains a weak area. The visible-depth volume estimate cannot recover hidden surfaces, and mass error is supplementary because the scorer evaluates `take`, not mass directly.
- Keep D as the current submission candidate, but compare it on additional object-ID folds and inspect take errors by mass range before allowing autonomous pickup decisions.

## 2026-11-21 - split integrity repair and roadmap to 100

An attempted fold-2 run was scored on fold 1 because the old remote validator regenerated the saved split with its default fold. The resulting 97.363 was invalid: all 10 reported validation IDs were in training. Its report and submissions are preserved under invalid_fold1_before_split_fix/ and must not be used as a result.

Fixed remote validate.py to load the training-log split, report its source, reject overlapping train/validation IDs, and verify detector and attribute training IDs. Added --no-detector-validation so training runs a fixed number of epochs and selects last.pt instead of using outer-fold mAP to select best.pt.

The clean C RGB-only 18-epoch fold-2 run scored 64.519 before postprocessing. A threshold fitted only from grouped OOF training-object geometry changed volume/take and reached 71.605 (D=.8282, M=.6319, V=.4934, S=.9000). detector.pt SHA-256 matches last.pt; the saved train and validation IDs are disjoint. Fold 2 has been used for exploratory analysis, so this is not an untouched final holdout.

The component roadmap is in reports/ROADMAP_SCORE_100_2026-11-21.md. Current largest gaps are volume (12.66 score points to perfect on this fold), material (7.36), and detection/form (6.87). The dataset has only 40 physical object IDs; more independently labeled objects, especially rare materials and heavy items, are required to support a credible 100-point claim.
Correction to the fold-2 entry above: applying the train-only form/material volume scale while keeping the same grouped-OOF take threshold scored 72.374 (D=.8282, M=.6319, V=.5242, S=.9000), a +0.769-point development-fold ablation. This must be verified on another object-ID fold before promotion. Details: reports/score83_fold2_noval18_calibrated_volume_oof_take.json.
Restored the earlier fold-1 80.974 candidate unchanged and rescored it with the official references/score.py: D=.9714, M=.7067, V=.6068, S=.8542. Candidate: reports/score80_candidate_fold1.csv; SHA-256 and split are in reports/score80_candidate_fold1.json. Keep it separate from new architecture runs; it is a development score, not a blind estimate.
