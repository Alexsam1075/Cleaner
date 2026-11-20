# Astramis data audit

The report reads only `real_train.json` and `synth_train.json` for labels. Test data contributes RGB-D/meta availability and submission-ID checks; no test labels are read.

## Dataset inventory

- Real train: 263 frames, 214 objects, 40 physical object IDs, 49 empty frames.
- Synthetic train: 600 frames, 678 objects, 40 physical object IDs.
- Real test: 137 frames, 137 submission rows; IDs match: True.
- Actual RGB dimensions: real {'1280x720': 263}, synthetic {'1280x720': 600}; depth types/sizes: real {'1280x720:uint16': 263}, synthetic {'1280x720:uint16': 600}, test {'1280x720:uint16': 137}.

## Labels

- Real forms: `{'bulk': 98, 'solid': 116}`; materials: `{'glass': 20, 'metal': 38, 'mineral': 15, 'paper': 28, 'plastic': 78, 'wood': 35}`.
- Synthetic forms: `{'bulk': 236, 'solid': 442}`; materials: `{'glass': 52, 'metal': 83, 'mineral': 88, 'paper': 87, 'plastic': 279, 'wood': 89}`.
- Real volume (ml): `{'count': 214, 'min': 12.0, 'median': 300.0, 'max': 5200.0, 'q1': 90.0, 'q3': 1000.0, 'iqr_outliers': 8}`; mass (g): `{'count': 214, 'min': 5.0, 'median': 150.0, 'max': 1000.0, 'q1': 60.0, 'q3': 370.0, 'iqr_outliers': 19}`.
- Missing required real attributes: 0; invalid boxes: 0; invalid polygons: 0.
- Synthetic bbox/polygon extrema differ by >2 px in 3 annotations.

## RGB-D and leakage

- Camera: RealSense D435, 1280x720, fx=911.659, fy=909.690; depth units `mm` and aligned to color.
- Floor plane missing: real 0, synthetic 0, test 0.
- Depth zero-share: real `{'count': 263, 'min': 0.05681857638888889, 'median': 0.09188259548611111, 'max': 0.15767035590277778, 'q1': 0.07291395399305556, 'q3': 0.10216254340277778, 'iqr_outliers': 1}`, synthetic `{'count': 600, 'min': 0.059174262152777775, 'median': 0.1007058376736111, 'max': 0.16486762152777779, 'q1': 0.07499945746527778, 'q3': 0.11459906684027778, 'iqr_outliers': 0}`, test `{'count': 137, 'min': 0.05734809027777778, 'median': 0.0880837673611111, 'max': 0.18146918402777779, 'q1': 0.07423177083333334, 'q3': 0.10663519965277778, 'iqr_outliers': 1}`.
- Real metadata holes_share: `{'count': 263, 'min': 0.0568, 'median': 0.0919, 'max': 0.1577, 'q1': 0.07295, 'q3': 0.10214999999999999, 'iqr_outliers': 1}`; synthetic `holes_share` is absent. Real lighting/surface: `{'bright': 92, 'dim': 89, 'side': 82}` / `{'tile': 263}`; synthetic: `{'bright': 77, 'dim': 292, 'side': 231}` / `{'tile': 600}`.
- Real/synthetic object ID overlap: 40 of 40; attribute mismatches: 0.
- Therefore synthetic views must be filtered by the same held-out `object_id` split. The pipeline enforces this in `datasets/splits.py`.

## Quality flags

- Real missing/unreadable images: 0; metadata size mismatches: 0; object-ID label conflicts: 0.
- Synthetic missing/unreadable images: 0; metadata size mismatches: 0; object-ID label conflicts: 0.
- Test missing RGB/depth/meta: 0; test metadata missing floor planes: 0.

## Notes

Volume/mass IQR outlier counts are recorded in `data_audit.json`; they flag review candidates and do not cause samples to be deleted. Geometric volume is an observed-surface estimate, so hidden surfaces and depth holes remain a known source of bias.
