# Cleaner score 83+: five independent code reviews

Date: 2026-11-21  
Target checked read-only: AION Lab `aion-lab`, `/home/alexsandrdev/Projects/Cleaner`, branch `main`, HEAD `78eac5c`.

Five reviewers covered data/training splits, detector/RGB-D, model and losses, geometry/take, and scorer integrity. They found the pre-existing untracked `AUDIT.md`, `audit/`, and `sim/`; none was modified. The reviewers did not train models or change project code.

## Current evidence

The best tracked server run is `D_domain_randomized`: `D=.9696, M=.5625, V=.5513, S=.3542`, official score `69.13`. The colleague's reported `79.40` is reproducible on the same 10-object fold, but it is a model-selected result, not independent validation. The recipe uses a geometry volume derived from ground-truth annotation masks, a class density table whose counts cover all 40 labeled objects including the held-out fold, and opposite-form duplicate detections at 5% confidence. The latter raises AP because the scorer computes AP separately by form; it does not improve form recognition and is kept out of quality claims.

The local `audit/analyze.py` diagnostic ran with the project's `.venv` and `ASTRAMIS_DATA=datasets/raw`. On 40 leave-one-object-out groups, raw geometry scored `V=.565`, `ridge + true form/material` scored `.696`, and the mass rule reached `S=.814` at 300 g. These are upper-bound diagnostics: the class labels are true, annotation masks are used, the density table includes all 40 objects, and the threshold is selected on the reported predictions. They do not establish the deployed pipeline's score.

## Reviewer findings

### 1. Data and training protocol

- `datasets/splits.py:34-77` correctly groups real and synthetic frames by physical `object_id` and excludes held-out IDs from synthetic training. The problem is that reported selection uses one four-way fold: 30 train IDs and 10 validation IDs. Rare materials have one validation object, so score changes are noisy and repeated tuning on the fold overfits.
- `datasets/objects.py:188-205` balances object and material, not the `take` class. There are 7 heavy (`take=0`) and 33 light objects. `training/attributes.py:131-134` uses unweighted BCE for take.
- `training/attributes.py:109-167` records train losses and saves the final epoch; it does not select a checkpoint by inner object-level validation. Augmentations alter RGB but do not model depth holes/glare.

### 2. Detection and RGB-D pipeline

- Detection is already strong (`D≈.97`); its remaining ceiling is at most roughly 1.2 total points. A larger YOLO model/input is a secondary ablation, not the main route to 83.
- Attribute training uses annotation polygons (`datasets/objects.py:120-124`), while inference uses detector masks or a box fallback (`inference/predictor.py:19-32,75-82`). This mask-domain gap can change crops and geometry.
- Material fares better on RGB alone in the existing fold (`M=.7067`) than in fused models (`.5625-.6100`), while depth/geometry helps volume. One shared representation (`models/multimodal.py:42,69,103`) makes one task's useful input a possible source of interference for another.
- Invalid depth is zeroed (`geometry/features.py:69-97`) and valid samples alone contribute to volume (`:112-154`); reflective/transparent surfaces therefore have biased volume. Preserve valid-depth fraction and test masked recovery on object-ID folds.
- The inference score multiplies detector confidence by the maximum form probability (`inference/predictor.py:85-106`), which can demote otherwise good boxes. Calibrate this only out of fold.

### 3. Model and optimization

- The shared RGB-D trunk feeds form, material, volume, mass, and take heads (`models/multimodal.py:42,69,103`). Existing modality ablations favor separate RGB material and RGB-D geometry paths.
- Low train losses do not show generalization: the C model reached material `.0157`, volume `.0619`, take `.0223`, yet there is no best-epoch selection. Evaluate task-specific validation metrics on inner physical-object groups.
- The independent take head is not physically tied to volume and density. Volume/mass losses also use fixed weights despite the benchmark evaluating volume and take rather than mass directly. Compare geometry-calibrated volume and class-weighted take on nested object-ID folds.

### 4. Geometry, volume, and take

- On the same outer fold, train-fit geometric calibration gives `V=.6049`, above the network volume head at `.5513`; errors concentrate below 50 ml and above 1000 ml (`logs/validation/D_domain_randomized/intermediate_geom/report.json`). Geometry counts visible points only (`geometry/features.py:100-204`), so holes and hidden surfaces need separate handling.
- The benchmark's `S` is balanced accuracy between heavy and light objects (`references/score.py:165-171`), but training uses a fixed `.5` threshold and only falls back to mass in an arbitrary `.08` window (`inference/predictor.py:99-103`). The network's `S=.3542` is below simple controls on this fold.
- A physics rule `estimated volume × train-only density` is promising; fit volume calibration, densities, and threshold from inner out-of-fold predictions, then use predicted form/material and predicted masks at inference. Never use true validation classes or a density table derived from held-out labels as if they were known.

### 5. Official score and validation integrity

- The score is `40·D + 20·M + 25·V + 15·S` (`references/score.py`). Missed detections also lose M/V/S, so any confidence change must preserve recall.
- `audit/postprocess.py:41-45` duplicates each detection with the opposite form at 5% confidence. This raises D through the scorer's per-form AP logic (`references/score.py:99-132`) without correcting form predictions. It is excluded from perception-quality results.
- The reported `.814` take and `.696` volume LOO numbers use ideal/true material and the global density file, and select the threshold across the same 40 predictions (`audit/analyze.py:28-60`). They are diagnostic ceilings only.
- Required evidence before promotion: grouped outer object-ID folds, inner-only model/threshold/calibration selection, densities fit on each train partition, inference-like predicted masks and classes, per-object errors, and all D/M/V/S components. Keep a separate untouched final fold.

## Highest-priority implementation sequence

1. Add an inner grouped-validation path with per-epoch/task metrics and best-checkpoint selection; keep current outer validation IDs unavailable to training, early stopping, and calibration.
2. Build honest out-of-fold material/form/volume/take predictions, recalculating density statistics per train partition. Test a separate RGB material path and geometry-based volume/take against the existing network.
3. Close the GT-mask/inference-mask gap, retain depth-validity/glare features, and test depth-hole recovery only within object masks.
4. Run the agreed multi-fold comparison and inspect physical-object errors. Promote a change only if it improves held-out quality and does not trade material/volume/take for artificial AP gains.

No review established a validated route to 83+. The first goal is to turn the same-fold `79.40` recipe into a clean end-to-end baseline; score 83+ remains an active target, not a promised result.
