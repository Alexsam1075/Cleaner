# Astramis experiment report

Metrics are computed by the supplied, unchanged `score.py` on the fixed validation frames grouped by physical `object_id`. No test annotations are used.

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
