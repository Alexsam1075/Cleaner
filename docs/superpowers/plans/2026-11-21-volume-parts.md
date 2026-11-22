# Объём по частям предмета — план реализации

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Заменить текущую оценку объёма (`geometry/features.py`, V=0.565 на GT-масках) на свою геометрию «чистка → части → фигуры → поправка по классу» с V ≥ 0.70 на grouped CV по `object_id`.

**Architecture:** Новый модуль `geometry/volume_parts.py` считает несколько геометрических оценок объёма по маске, глубине и плоскости пола. `evaluation/volume_parts_cv.py` собирает эти оценки на real train и обучает маленькую лог-линейную поправку по классу (форма × материал) с GroupKFold по `object_id`. Затем поправка подключается в `inference/predictor.py`.

**Tech Stack:** Python 3.12 (`~/venv/bin/python` на aion-lab), numpy, opencv, scipy, scikit-learn, pytest (ставится в `~/venv`).

**Spec:** `docs/superpowers/specs/2026-11-21-volume-parts-design.md`

## Global Constraints

- Всё запускается на сервере: `ssh -i ~/.ssh/cleaner_shared_ed25519 -p 4107 cleaner@77.37.192.59`, каталог `~/Cleaner`, данные `~/Cleaner/datasets/raw`.
- Без предобученных сетей. Открытый код с GitHub допускается только как алгоритм, переписанный у нас.
- Никакого подбора по внешнему fold: пороги чистки и поправка — только по train-объектам (GroupKFold по `object_id`).
- Метрика объёма как в `astramis_case/score.py`: `max(0, 1 - |ln(p/g)| / ln 3)`; объём ≤ 0 → 0.
- Плотности для массы — `datasets/raw/densities.csv` (`form,material,density_g_per_ml`).
- Ветка `codex/cleaner-score83-20261121` содержит чужие незакоммиченные правки. Свои файлы только новые (кроме Task 5); коммитить — только после согласования с пользователем, и только свои файлы.
- GPU не нужен; запускать через `nice`.

## Review Focus

1. Маска без валидной глубины (стекло, `valid_frac≈0`) → объём по контуру маски на полу × медианная высота класса, никогда не `None`/0. Тест в Task 3.
2. Крошечный предмет (12–40 мл, высота 5–15 мм): порог шума по высоте не должен обнулить его. Тест в Task 1.
3. Предмет обрезан краем кадра → оценка не падает, флаг `truncated=True` в результате. Тест в Task 3.
4. Знак плоскости пола (нормаль к камере или от неё) → высоты всё равно ≥ 0. Тест в Task 1.
5. Маска захватывает пол вокруг предмета (неточный полигон) → объём не растёт больше чем на 20%. Тест в Task 1.

---

### Task 1: Синтетическая сцена и чистка облака точек

**Files:**
- Create: `tests/synth_scene.py`, `tests/test_volume_clean.py`, `geometry/volume_parts.py`

**Interfaces:**
- Produces:
  - `tests.synth_scene.render(camera: dict, plane: np.ndarray, shapes: list[dict], noise_mm: float = 1.0, seed: int = 0) -> tuple[np.ndarray, list[np.ndarray]]` — depth (H×W, uint16, мм) и булевы маски по фигурам. Фигуры: `{"kind": "box", "center_uv": (u, v), "size": (L, W, H), "yaw": a}`, `{"kind": "cylinder_lying" | "cylinder_standing", "center_uv", "radius", "length", "yaw"}`, `{"kind": "dome", "center_uv", "radius", "height"}`; все размеры в метрах, `u, v` — координаты на полу. Рисуется плотной выборкой поверхности (шаг 1 мм), проекцией камерой и z-буфером.
  - `geometry.volume_parts.ObjectCloud` (dataclass): `uv: (N,2) м`, `h: (N,) м ≥ 0`, `footprint_uv: (K,2) м` — контур маски, спроецированный на пол, `valid_frac: float`, `truncated: bool`.
  - `geometry.volume_parts.refine_plane(depth_mm, mask, plane, camera, ring_px: int = 25) -> np.ndarray` — RANSAC/МНК по кольцу пикселей вокруг маски; если в кольце < 200 валидных точек, возвращает исходную плоскость.
  - `geometry.volume_parts.clean_cloud(depth_mm, mask, plane, camera, erode_px: int = 2, min_h_m: float = 0.003) -> ObjectCloud`.

- [ ] **Step 1: Поставить pytest:** `~/venv/bin/pip install pytest` → `~/venv/bin/python -m pytest --version` печатает версию.
- [ ] **Step 2: Тесты**

```python
def test_box_heights():          # коробка 100×50×40 мм, шум 1 мм
    c = clean_cloud(depth, masks[0], plane, cam)
    assert abs(np.quantile(c.h, 0.98) - 0.040) < 0.004
def test_tiny_object_survives(): # коробка 30×20×8 мм
    assert len(clean_cloud(...).h) > 50
def test_plane_sign():           # plane, умноженная на -1
    assert (clean_cloud(..., -plane, cam).h >= 0).all()
def test_loose_mask():           # маска коробки, расширенная на 10 px
    assert columns_volume_ml(clean_cloud(loose)) < 1.2 * columns_volume_ml(clean_cloud(tight))
def test_refine_plane_recovers_offset():  # исходная плоскость сдвинута на 5 мм
    assert abs(refine_plane(...)[3] - true_plane[3]) < 0.0015
```

`columns_volume_ml(cloud, cell_m=0.004) -> float` — сумма максимальных высот по ячейкам (как в `features.py`, но по очищенному облаку); определяется в этой же задаче.

- [ ] **Step 3: Запустить → FAIL** (`ImportError`).
- [ ] **Step 4: Реализовать** `refine_plane`, `clean_cloud`, `columns_volume_ml`. В `clean_cloud`: эрозия маски `erode_px` (не больше, чем оставляет ≥ 30% пикселей), направление нормали по медиане высот в кольце (пол → ~0), отбрасывание `h < min_h_m`, `truncated` = маска касается края кадра.
- [ ] **Step 5: Запустить** `~/venv/bin/python -m pytest tests/test_volume_clean.py -v` → все PASS.

### Task 2: Части предмета и фигуры

**Files:**
- Modify: `geometry/volume_parts.py`; Create: `tests/test_volume_parts.py`

**Interfaces:**
- Consumes: `ObjectCloud`, `tests.synth_scene.render`.
- Produces:
  - `Part` (dataclass): `uv`, `h`, `s_range: (float, float)` вдоль главной оси.
  - `split_parts(cloud: ObjectCloud, n_slices: int = 16, merge_tol: float = 0.30) -> list[Part]` — PCA отпечатка → срезы вдоль главной оси → профиль (ширина, p95 высоты) на срез → соседние срезы склеиваются, пока относительное изменение профиля < `merge_tol`.
  - `fit_part(part: Part, kinds: tuple[str, ...]) -> PartFit` — `PartFit(kind: str, volume_ml: float, residual: float)`; `kinds` ⊂ `{"box", "cylinder_lying", "cylinder_standing", "ellipsoid", "dome"}`; выбирается фигура с минимальной медианной ошибкой высоты по видимым точкам; невидимая сторона — по симметрии относительно оси части.
  - `parts_volume_ml(cloud, form: str) -> tuple[float, list[PartFit]]` — `form == "bulk"` → `kinds=("dome", "ellipsoid", "box")`, иначе `("box", "cylinder_lying", "cylinder_standing", "ellipsoid")`.

- [ ] **Step 1: Тесты**

```python
def test_lying_cylinder_volume():   # r=35 мм, L=200 мм → 769.7 мл
    v, fits = parts_volume_ml(cloud, "solid"); assert abs(v / 769.7 - 1) < 0.15
def test_standing_cylinder_hidden_back():  # r=40 мм, H=120 мм → 603.2 мл
    assert abs(parts_volume_ml(cloud, "solid")[0] / 603.2 - 1) < 0.15
def test_bottle_two_parts():        # тело r=40 L=180 + горлышко r=13 L=60
    assert len(split_parts(cloud)) == 2
def test_dome_pile():               # купол r=60 мм, h=40 мм → π·h(3r²+h²)/6 ≈ 259.7 мл
    assert abs(parts_volume_ml(cloud, "bulk")[0] / 259.7 - 1) < 0.20
```

- [ ] **Step 2: FAIL → Step 3: реализовать → Step 4:** `pytest tests/test_volume_parts.py -v` → PASS.

### Task 3: Единая функция оценки объёма

**Files:**
- Modify: `geometry/volume_parts.py`; Create: `tests/test_volume_estimate.py`

**Interfaces:**
- Produces: `estimate_volume(depth_mm, mask, plane, camera, form: str, class_height_m: float | None = None) -> dict` с ключами `columns_ml`, `parts_ml`, `hull_ml` (выпуклая оболочка видимых точек + их проекции на пол, `scipy.spatial.ConvexHull`), `footprint_ml` (площадь `footprint_uv` × p95 высоты), `valid_frac`, `truncated`, `n_parts`, `p95_h_m`, `footprint_m2`. Все `*_ml` ≥ 1.0.

- [ ] **Step 1: Тесты**

```python
def test_no_valid_depth_fallback():   # глубина в маске обнулена, class_height_m=0.08
    r = estimate_volume(depth0, mask, plane, cam, "solid", class_height_m=0.08)
    assert r["valid_frac"] == 0 and r["footprint_ml"] > 1 and r["parts_ml"] >= 1
def test_truncated_flag():            # коробка, наполовину за краем кадра
    assert estimate_volume(...)["truncated"] is True
def test_all_keys_positive():
    assert all(r[k] >= 1.0 for k in ("columns_ml", "parts_ml", "hull_ml", "footprint_ml"))
```

При `valid_frac == 0` все `*_ml` = площадь контура маски на полу × `class_height_m` (или 0.05 м, если `None`) × 0.6.

- [ ] **Step 2: FAIL → Step 3: реализовать → Step 4:** `pytest tests -v` → все PASS.

### Task 4: Поправка по классу и grouped CV на GT-масках

**Files:**
- Create: `evaluation/volume_parts_cv.py`; Output: `reports/volume_parts_gt_rows.csv`, `reports/volume_parts_cv.json`, `checkpoints/volume_parts/calibrator.json`

**Interfaces:**
- Consumes: `estimate_volume`.
- Produces:
  - `collect_rows(data_root: Path, split: str = "real") -> pd.DataFrame` — одна строка на аннотацию: `image_id, object_id, form, material, volume_ml` + все ключи `estimate_volume`. Маска из полигона, плоскость из `meta.json`, `refine_plane` применяется внутри. `class_height_m` = медиана `p95_h_m` по train-фолду для класса (считается внутри CV, не заранее).
  - `fit_calibrator(rows: pd.DataFrame) -> dict` — Ridge (alpha подбирается внутренним GroupKFold) на `log(volume_ml)` по признакам: `log` всех `*_ml`, `valid_frac`, `truncated`, `n_parts`, one-hot `form`, one-hot `material`, `form×material`; веса строк = 1 / (кадров у этого `object_id`).
  - `apply_calibrator(cal: dict, est: dict, form: str, material: str) -> float` — мл, клип [1, 20000].
  - CLI: `python -m evaluation.volume_parts_cv --data datasets/raw --folds 5 --seed 0` → печатает и сохраняет V: сырой `columns_ml` из `features.py` (ожидается ≈0.565), каждая сырая оценка, калиброванная OOF; по классам и корзинам `<50, 50–200, 200–1000, ≥1000` мл; mean/std по 3 сидам разбиения.

- [ ] **Step 1: Тест** `tests/test_volume_cv.py::test_no_object_leak` — во всех фолдах пересечение `object_id` train/val пусто; `test_calibrator_roundtrip` — `apply_calibrator(json.loads(json.dumps(cal)), ...)` совпадает с исходным.
- [ ] **Step 2: FAIL → реализовать → PASS.**
- [ ] **Step 3: Прогон** CLI на сервере. Приёмка: калиброванный OOF V ≥ 0.70 (mean по сидам). Если ниже — подбирать `erode_px`, `min_h_m`, `merge_tol` по тому же OOF (а не по внешнему fold), записать таблицу попыток в `reports/volume_parts_cv.json`.
- [ ] **Step 4:** Опционально добавить synth-кадры в train-часть каждого фолда (только объекты train-части) и сравнить V; оставить, только если OOF V растёт.

### Task 5: Подключение в инференс и проверка на предсказанных масках

**Files:**
- Modify: `inference/predictor.py` (`predict_frame`, строка ~97, где сейчас `volume_ml` из `volume_log`)
- Create: `tests/test_predictor_volume.py`

**Interfaces:**
- Consumes: `estimate_volume`, `apply_calibrator`, `checkpoints/volume_parts/calibrator.json`.
- Produces: аргумент `volume_backend: str = "net"` у `predict_frame` и `predict_ids`; при `"parts"` объём = `apply_calibrator(...)` по предсказанной маске, форме и материалу; `mass_g = volume_ml × density(form, material)`; take — как сейчас (порог по объёму из OOF), не меняется.

- [ ] **Step 1: Тест** — `predict_frame(..., volume_backend="parts")` на одном train-кадре возвращает `volume_ml > 0` и `mass_g == pytest.approx(volume_ml * density)`.
- [ ] **Step 2: FAIL → реализовать → PASS.**
- [ ] **Step 3:** Калибратор переобучить только на train-объектах fold 2 (`checkpoints/score83_cv_fold2_noval18/` split), прогнать `validate.py` с чекпоинтом `score83_cv_fold2_noval18` и `volume_backend="parts"`. Приёмка: V выше 0.5242 при тех же D/M; итог сравнить с 72.374; отчёт `reports/volume_parts_fold2_noval18.json`.
- [ ] **Step 4:** Показать пользователю итоговую таблицу и спросить про коммит своих файлов.
