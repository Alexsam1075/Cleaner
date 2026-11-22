# Как запустить модель (конвейер интегратора agents2/integrator)

Сервер AION Lab (`ssh cleaner`), проект `~/Cleaner`, Python `~/venv/bin/python`.
Все команды — из `~/Cleaner/audit` с переменной окружения:

```bash
cd ~/Cleaner/audit
export ASTRAMIS_DATA=/home/cleaner/Cleaner/datasets/raw
P=/home/cleaner/Cleaner/agents2/integrator
```

## Что внутри лучшего конвейера

| Блок | Реализация | Файл |
|---|---|---|
| Рамки, маски | YOLO11n-seg 640, обучен на 40 train-объектах (real+synth) | `audit/final/det_all/weights/last.pt` |
| Материал | голова CLIP-L c14 ⊕ zero-shot CLIP-L+SigLIP (rich, logsumexp по c14+c20) | `agents2/material/predict_material.py:comp_material` |
| Форма (bulk/solid) | FormModel2: голова CLIP-L c14+DINOv2+zs ⊕ zero-shot ensemble-prior, 3× inner CV, top-3 конфигураций, порог 0.5 | `agents2/form/predict_form.py` через `agents2/integrator/ext_form.py:comp_form` |
| Память формы по серии | pool_series: кластеры CLIP cos ≥ 0.9, окно ±10 кадров | `ext_form.py:series_pool` |
| Материал по серии | голосование при согласии ≥ 0.75 (токен `b`) | `pipeline_v2.py` |
| Объём | объём по частям + ridge-калибратор по GT-маскам train; медиана log-объёма по группе «тот же предмет» (`--agg` cos, `--agg-gap` разрыв кадров) | `audit/full_pipeline.py` |
| take | гибрид: metal/glass/mineral — объём ≤ T (T по train); остальные — масса = объём × плотность материала ≤ 500 г (`take_t1`) | `agents2/integrator/ext_take.py:take_t1` |
| Вторая рамка (опция `--two-box`) | та же рамка с противоположной формой, score = conf × p(вторая форма); влияет только на D | `eval_folds.assemble` |

Токены комбинации: `form_v2+series_pool+material_v2+b+take_t1`.

## 1. Кэши (делаются один раз, уже есть)

```bash
~/venv/bin/python $P/cache_text.py                 # text_mat.npz — текстовые эмбеддинги описаний материалов
~/venv/bin/python $P/test_cache.py                 # cache/test_dets.pkl — детекции+эмбеддинги real/test (детектор + CLIP/SigLIP, GPU)
~/venv/bin/python $P/form_feats.py --what 1 2 3 4 test   # cache/form_v2_*.npz — DINOv2/глубина/FastSAM для детекций (GPU)
```
Кэши фолдов `audit/cache/fold{1,2,3}_dets.pkl`, `fold4_dets_holdout4.pkl` — детекторы фолдов (`audit/full_pipeline.py`).
Train-признаки: `audit/cache/features*.npz`, `meta.json`.

## 2. Сабмит на тест

Общие аргументы:
```bash
EXT="material_v2=/home/cleaner/Cleaner/agents2/material/predict_material.py:comp_material \
     form_v2=$P/ext_form.py:comp_form \
     series_pool=$P/ext_form.py:series_pool \
     take_t1=$P/ext_take.py:take_t1"
COMBO="fvp_t1=form_v2+series_pool+material_v2+b+take_t1"
```

Две рамки (максимум Score, для лидерборда):
```bash
~/venv/bin/python $P/make_submission.py --variant fvp_t1 --external $EXT --combo "$COMBO" \
    --two-box 0.0001 --agg 0.85 --agg-gap 1000 \
    --out ~/Cleaner/publish/submission_twobox.csv
```

Одна рамка на объект (честный вариант):
```bash
~/venv/bin/python $P/make_submission.py --variant fvp_t1 --external $EXT --combo "$COMBO" \
    --agg 0.85 --agg-gap 1000 \
    --out ~/Cleaner/publish/submission_onebox.csv
```

Без памяти через весь ролик (прежняя настройка): `--agg 0.90 --agg-gap 3`.
Вариант без нового take: `--variant fvp --combo "fvp=form_v2+series_pool+material_v2+b"`.

Скрипт печатает число кадров/объектов, T, заметки компонентов и проверяет формат через `references/score.py`.
Порядок строк = `references/sample_submission.csv`.

## 3. Оценка на фолдах (official score.py, 4 фолда по object_id; ф1 завышен, ф4 — нетронутый holdout)

```bash
~/venv/bin/python $P/pipeline_v2.py --folds 1 2 3 4 --variants b --external $EXT --combo "$COMBO" \
    --only fvp_t1 --two-box 0.0001 --agg 0.85 --agg-gap 1000 --tag my_tag
```
Результаты по фолдам печатаются в stdout, CSV фолдов — `$P/cache/fold{k}_my_tag_fvp_t1|tb0.999.csv`,
сводка — `$P/cache/report_my_tag.json`.

Сравнение двух вариантов bootstrap по объектам:
```bash
~/venv/bin/python $P/bootstrap_cmp.py "<tag_base>" "<tag_new>"
```
Диагностика: `oracle.py <tag>` (остаток потерь при GT-компонентах), `diag_D.py <tag>` (FP/пропуски),
`diag_obj.py <tag>` (ошибки по объектам), `vol_slope.py <tag>` (калибровка объёма).

## 4. Фоновый запуск (без sleep)

```bash
./runbg.sh <имя> <команда...>      # лог <имя>.log, маркер <имя>.done
./check.sh                          # сводка ГОТОВО/ИДЁТ
```

## 5. Результаты на фолдах (ф1/ф2/ф3/ф4, ср.4; pooled по 266 кадрам)

| вариант | ф1 | ф2 | ф3 | ф4 | pooled | Δ bootstrap |
|---|---:|---:|---:|---:|---:|---|
| 1 рамка, matv2 + form v2 + pool (best 21:10) | 88.98 | 89.98 | 89.60 | 73.35 | 86.44 | — |
| + две рамки | 91.26 | 91.02 | 89.60 | 81.10 | 89.15 | +2.55 [+0.15, +6.49] |
| + память серий gap ∞ sim 0.85 | 91.43 | 92.35 | 88.94 | 84.00 | 90.41 | +1.19 [+0.16, +2.65] |
| + take_t1 | 92.06 | 92.35 | 88.94 | 84.00 | ≈90.7 | +0.39 [0, +1.2] |

Файлы: `publish/SCOREBOARD.md` (все варианты), `agents2/integrator/LOG.md`, `agents2/integrator/REPORT.md`.

## Ограничения

- `real/test` — только инференс; ручные разметки теста в обучении не используются.
- Все пороги (T, плотности, веса zero-shot) считаются по train-объектам фолда/всех 40.
- GPU нужна только для кэшей (п.1); оценка и сборка сабмита работают на CPU.
