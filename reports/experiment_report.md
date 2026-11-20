# Astramis experiment report

Metrics are computed by the supplied, unchanged `score.py` on the fixed validation frames grouped by physical `object_id`. No test annotations are used.

## Data audit

The remote data audit read only `real_train.json` and `synth_train.json`; no test-label file was opened. Both train sets contain the same 40 physical IDs, and form, material, volume, and mass agree for every shared ID.

| Label count | Real annotations | Synthetic annotations |
| --- | ---: | ---: |
| Frames | 263 | 600 |
| Object annotations | 214 | 678 |
| Empty frames | 49 | 0 |
| Form: bulk / solid | 98 / 116 | 236 / 442 |
| Material: plastic | 78 | 279 |
| Material: paper | 28 | 87 |
| Material: metal | 38 | 83 |
| Material: glass | 20 | 52 |
| Material: wood | 35 | 89 |
| Material: mineral | 15 | 88 |

Unique-ID counts are 14 bulk and 26 solid; material counts are plastic 17, paper 5, metal 5, wood 5, mineral 5, glass 3. Every object ID occurs in both real and synthetic data; matching attributes show that synthetic data are object-linked copies, so validation IDs must be filtered from synth training.

Volume and mass quartiles use linear interpolation; outliers use Tukey's 1.5×IQR rule. Annotation-level values repeat objects across frames, so the unique-ID view is less frame-biased.

| Data | Measure | Min–max | Q1 / median / Q3 | IQR | Outliers |
| --- | --- | ---: | ---: | ---: | ---: |
| Real annotations | Volume (ml) | 12–5,200 | 90 / 300 / 1,000 | 910 | 8 |
| Real annotations | Mass (g) | 5–1,000 | 60 / 150 / 370 | 310 | 19 |
| Synthetic annotations | Volume (ml) | 12–5,200 | 90 / 300 / 800 | 710 | 31 |
| Synthetic annotations | Mass (g) | 5–1,000 | 35 / 140 / 350 | 315 | 51 |
| Unique physical IDs | Volume (ml) | 12–5,200 | 90 / 300 / 800 | 710 | 2 |
| Unique physical IDs | Mass (g) | 5–1,000 | 46.25 / 145 / 355 | 308.75 | 3 |

All required files are present: real train 263 frame folders / 789 files, real test 137 / 411, and synthetic train 600 / 2,400. No missing attributes, invalid numeric labels, bad boxes, malformed or out-of-frame polygons, unknown classes, dangling image references, or conflicting per-ID labels were found. The label directory lists only the two train JSON files. `densities.csv` has 11 valid rows covering 40 objects.

All 2,600 color/depth/mask PNG headers are 1280×720 and use the expected types: RGB color, 16-bit grayscale depth, and 8-bit grayscale synthetic masks. No required PNGs are missing or have malformed headers. `camera.json` describes a RealSense D435 at 1280×720 with depth in millimeters aligned to color. All 1,000 metadata files have a valid normalized floor plane. The three floor-plane groups correspond to heights/tilts of 0.2514 m/27.42°, 0.2532 m/31.47°, and 0.2516 m/32.78°.

Depth pixels were decoded for a deterministic 30-frame sample from each split; these coverage statistics are not a full-set pixel scan.

| Split | Zero-pixel share: min / Q1 / median / Q3 / max | Median positive-depth coverage |
| --- | ---: | ---: |
| Real train | 5.70 / 7.27 / 9.34 / 10.38 / 12.14% | 90.66% |
| Real test | 5.87 / 7.38 / 8.60 / 11.31 / 13.40% | 91.40% |
| Synthetic train | 5.94 / 7.30 / 10.47 / 11.62 / 14.48% | 89.53% |

All 90 sampled maps decoded as 720×1280 `uint16`; none were all-zero. On sampled real frames, `meta.holes_share` matched the measured zero share within 0.0001. Sparse values above 10,000 mm and up to 65,535 mm occur; without a sensor-code specification they are not classified here as invalid. The geometry feature code currently accepts depth from 50 to below 8,000 mm and excludes out-of-range samples from geometry features.

## Validation split

- Seed 0, fold 1: 10 held-out physical IDs and 67 real frames (57 annotated objects).
- Held-out IDs by material: `{"glass": ["s38"], "metal": ["s45"], "mineral": ["s47", "s48"], "paper": ["s16"], "plastic": ["b03", "b10", "s11", "s18"], "wood": ["b16"]}`.
- The same fold was used for every experiment. Synthetic frames containing any held-out ID were excluded from pretraining and joint training.
- M is macro accuracy by represented material; S averages take accuracy for GT mass groups above and at/below 500 g.

## Data strategy results

D / M / V / S / Score are shown as fractions / points, with Score on the 0–100 scale.

| Condition | Baseline logic on same detector | Train-fit geometry calibration | RGB-D multimodal | Mass MAE (g) |
| --- | ---: | ---: | ---: | ---: |
| `A_real` | 0.931 / 0.167 / 0.607 / 0.500 / 63.25 | 0.931 / 0.167 / 0.606 / 0.500 / 63.22 | 0.956 / 0.475 / 0.508 / 0.500 / 67.95 | 157.6 |
| `B_synth_then_real` | 0.956 / 0.167 / 0.612 / 0.500 / 64.37 | 0.956 / 0.167 / 0.612 / 0.500 / 64.38 | 0.955 / 0.617 / 0.510 / 0.365 / 68.77 | 149.2 |
| `C_joint` | 0.971 / 0.167 / 0.607 / 0.500 / 64.86 | 0.971 / 0.167 / 0.605 / 0.500 / 64.81 | 0.934 / 0.610 / 0.538 / 0.354 / 68.34 | 152.2 |
| `D_domain_randomized` | 0.971 / 0.167 / 0.595 / 0.490 / 64.41 | 0.971 / 0.167 / 0.605 / 0.490 / 64.66 | 0.970 / 0.562 / 0.551 / 0.354 / 69.13 | 163.2 |

## Fusion ablation on joint training data

The RGB-only, late-fusion, intermediate RGB-D, and intermediate RGB-D plus geometry heads share the C_joint detector and the same object-ID split.

| Attribute fusion | D / M / V / S / Score | Matched-object mass MAE (g) |
| --- | ---: | ---: |
| `rgb_only` | 0.971 / 0.707 / 0.304 / 0.406 / 66.68 | 159.7 |
| `late` | 0.971 / 0.570 / 0.494 / 0.375 / 68.24 | 145.0 |
| `intermediate` | 0.971 / 0.583 / 0.336 / 0.406 / 65.01 | 175.1 |
| `intermediate_geom` | 0.934 / 0.610 / 0.538 / 0.354 / 68.34 | 152.2 |

## Error details

Per-material class accuracy and per-form matched recall are recorded in each validation `report.json`. Volume accuracy by GT range and mass MAE by material are also included there.

Highest observed data-strategy Score on this one grouped fold: `D_domain_randomized`.
Highest observed fusion Score on this one grouped fold: `intermediate_geom`.
These are selection results on a small validation population, not a final generalization estimate. Some represented rare materials are supported by one held-out physical object; use grouped cross-validation or more real objects before making a deployment claim.

## Baseline interpretation

The baseline comparator restores majority-material prediction, depth/floor-plane cell summation, median-volume fallback, and density-derived mass/take while using the same RGB detector as the associated model. The calibrated comparator fits a linear correction in log-volume space using training object IDs only. This controlled comparison is not an exact reproduction of the notebook's original split or epoch settings.

## Known limitations and next work

- Visible depth cannot recover hidden surfaces; report geometric and learned volume errors separately and collect more real object IDs.
- Mass MAE is supplementary because the provided scorer does not score mass directly; take remains scored.
- Masks are used by the geometry and visualization pipeline, while the supplied score measures bounding boxes rather than mask AP.
- A follow-up should run grouped cross-validation, inspect per-object repeated-frame errors, and only keep synthetic/domain-randomization paths if held-out Score and generation outputs both improve.
