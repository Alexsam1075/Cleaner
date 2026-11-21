# Модуль объёма (`volume/`)

Считает объём (мл) и массу (г) предмета по глубине, маске, плоскости пола, **форме и материалу**.
Материал — вход модуля: чем точнее его даёт классификатор, тем точнее объём.

## Как устроено

1. Чистка облака точек: уточнение пола по кольцу вокруг маски, порог высоты 3σ шума пола (3–6 мм).
2. Карта высот 4 мм; невидимая с камеры сторона достраивается зеркально через линию гребня.
3. Деление на части вдоль главной оси, на каждую часть — фигура (коробка, цилиндр лёжа/стоя,
   эллипсоид, купол).
4. Несколько геометрических оценок → поправка в лог-шкале по форме и материалу с изломами
   на 50 и 300 мл (обучена на train-объектах, GroupKFold по `object_id`).
5. Без глубины (стекло): отпечаток маски на полу с учётом типичной высоты класса.
6. Масса = объём × плотность класса из `densities.csv`.

## Качество (V официального scorer)

| Проверка | V |
|---|---:|
| Старая геометрия, GT-маски, grouped OOF | 0.565 |
| Модуль, GT-маски и GT-материал, grouped OOF (5×3) | **0.764** |

Маски детектора (замена только объёма в готовых CSV, калибратор на train-объектах fold):

| Fold | Старый | Предсказанный материал | Верный материал |
|---|---:|---:|---:|
| 1 (`score83_rgb320_fold1`) | 0.595 | 0.691 | **0.718** |
| 2 (`score83_cv_fold2_noval18`) | 0.524 | 0.629 | **0.719** |
| 3 (`score83_cv_fold3_noval18`) | 0.603 | 0.649 | **0.787** |

V считает ненайденные детектором объекты как 0, поэтому верхняя граница зависит и от D.

## Использование из кода

```python
from volume import VolumeEstimator
est = VolumeEstimator.load("checkpoints/volume_parts/calibrator.json",
                           "datasets/raw/densities.csv", "datasets/raw/camera.json")
r = est.estimate(depth_mm, mask, floor_plane, form="solid", material="glass")
r.volume_ml, r.mass_g, r.geometry          # geometry — сырые оценки и диагностика
# если есть только вероятности материала:
r = est.estimate(depth_mm, mask, floor_plane, form="solid",
                 material_probs={"glass": 0.6, "plastic": 0.4})
```

Формы: `bulk`, `solid`. Материалы: `glass, metal, mineral, paper, plastic, wood`.
`floor_plane` — `meta.json["floor_plane"]` кадра.

## Пересчёт объёма в любом submission CSV

Берёт форму и материал из каждой строки, маски — повторным прогоном детектора (сопоставление
по IoU ≥ 0.9), меняет только объём и массу; box, score, form, material, take не трогает.

```bash
~/venv/bin/python -m volume.submission --submission in.csv --out out.csv \
    --detector checkpoints/experiments/C_joint/detector.pt --data datasets/raw
```

С разметкой валидации можно сразу посчитать метрику, а `--oracle-material` подставит верный
материал (верхняя граница объёма):

```bash
~/venv/bin/python -m volume.submission --submission in.csv --out out.csv --detector det.pt \
    --labels val_labels.json [--oracle-material]
```

Для честной валидации калибратор нужно обучить только на train-объектах fold:

```python
import json, pandas as pd
from evaluation.volume_parts_cv import fit_calibrator
split = json.load(open("<run>/logs/object_id_split.json"))
rows = pd.read_csv("reports/volume_parts_gt_rows.csv")
cal = fit_calibrator(rows[rows.object_id.isin(split["train_object_ids"])].reset_index(drop=True))
json.dump(cal, open("cal_fold.json", "w"))
```

`checkpoints/volume_parts/calibrator.json` обучен на всех 40 объектах — для теста, не для валидации.
Пересобрать строки и калибратор: `~/venv/bin/python -m evaluation.volume_parts_cv --data datasets/raw`.

## Файлы

- `volume/estimator.py` — `VolumeEstimator`, `VolumeResult`
- `volume/submission.py` — CLI пересчёта CSV
- `geometry/volume_parts.py` — геометрия (чистка, части, фигуры, отпечаток)
- `evaluation/volume_parts_cv.py` — калибратор и grouped CV
- `inference/volume_backend.py` — плотности
- `tests/` — 30 тестов: `~/venv/bin/python -m pytest tests -q`
- Ветка с полной историей: `volume-parts` (worktree `~/CleanerVolume`)
