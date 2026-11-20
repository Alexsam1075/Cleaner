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

No model result is claimed yet. AION Lab package setup, training, official validation, and test submission generation are pending.

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

- Initialized branch `main` at `/home/alexsandrdev/Projects/Cleaner` and committed 40 source, config, reference, and documentation files as `1c15ee5` (`Initial Astramis RGB-D model package`). Commit author is the authenticated GitHub account `Alexsam1075`.
- The raw dataset and both transfer archives are ignored by Git. All 3,604 raw files remain on AION Lab. Group write access on `.git` lets the shared `cleaner` account use the repository.
- Set `origin` to `git@github.com:Alexsam1075/Cleaner.git`. The server SSH key authenticates as `Alexsam1075`, but GitHub reports that repository as not found. No GitHub API session or connected GitHub integration is available to create it, so the private online repository has not yet been created or pushed.

