"""
nav.py — маршрут объезда для робота-уборщика SurfexUnit по одному кадру камеры глубины.

Что делает:
1. Строит карту пола сверху (сетка 1 см) по карте глубины и плоскости пола из meta.json:
   где пол виден, где что-то возвышается, где ничего не видно (тень за предметом, край обзора).
2. Делит то, что лежит на полу, на три вида:
   * ЗАБРАТЬ  — объект с take=1: робот проезжает по нему, конвейер забирает;
   * ОБЪЕХАТЬ — объект с take=0 (тяжелее 500 г): препятствие, к нему не ближе запаса;
   * НЕИЗВЕСТНО — выступ над полом, который детектор не узнал (чужой предмет, ведро, щётка,
     ножка стеллажа): тоже препятствие. Безопасное поведение по умолчанию — объехать и позвать оператора.
3. Раздувает препятствия на половину ширины робота + запас и прокладывает путь A*:
   старт у ближнего края обзора → по очереди через все доступные цели «забрать» → дальний край обзора.
   Тени за предметами (пол не виден) проходимы, но дорого: робот туда не лезет без нужды.
   Если пути нет — команда «стоп, позвать оператора».
4. Рисует путь на кадре камеры и на карте сверху, пишет JSON с решениями и точками пути
   в координатах пола (метры: a — вправо, b — вперёд от точки под камерой).

Объекты берутся из одного из источников:
    --labels dataset/labels/real_train.json     разметка (маски-полигоны и take)
    --submission submission.csv                 ответ модели в формате кейса (рамки и take)
    (ничего)                                    только геометрия: всё, что торчит, — препятствие

Проверка по разметке (сколько тяжёлых задели, сколько лёгких убрали):
    python nav.py --data dataset --frames-glob "real/train/*" --submission val_pred.csv \
                  --gt-labels dataset/labels/real_train.json --avoid-mass-g 350 --out nav_eval

Решение по объёму из глубины (робот меряет сам, независимо от модели):
    --geom-volume-ml 300 [--ignore-take]
    Объезжать объект, если его объём по видимой глубине больше 300 мл. С --ignore-take от модели нужны
    только рамки, решение «тяжёлый ли» робот принимает сам.

Режим осторожности:
    --avoid-mass-g 350   объезжать всё, что по оценке модели тяжелее 350 г, даже если take=1.
                         Наезд на тяжёлое опаснее, чем пропущенный лёгкий предмет, а оценка массы
                         ошибается (объём, материал → плотность). Порог ниже 500 г — запас на эти ошибки.
                         take=0 объезжается всегда. Без параметра решение берётся только из take.

Примеры:
    python nav.py --data dataset --frames real/train/r_0001 --labels dataset/labels/real_train.json --out nav_out
    python nav.py --data dataset --frames real/test/r_0008 real/test/r_0011 --submission submission.csv --out nav_out
    python nav.py --data dataset --frames-glob "real/test/*" --submission submission.csv --out nav_out

Нужно: numpy, opencv-python (Pillow — по желанию, для русских подписей).
"""
import argparse
import csv
import glob
import heapq
import json
import math
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

# ---------------------------------------------------------------- параметры
GRID = 0.01                    # м на клетку карты
A_RANGE = (-0.90, 0.90)        # поперёк, м
B_RANGE = (0.15, 2.00)         # вперёд, м
OBST_H = 0.025                 # выше этого над полом — препятствие, если объект не опознан
OBJ_H = 0.008                  # пиксели рамки выше этого считаются объектом (режим submission)
MIN_UNKNOWN_CM2 = 20           # мелочь меньше этой площади не считаем препятствием
UNOBSERVED_COST = 6.0          # во сколько раз дороже ехать туда, где пол не виден
OUTSIDE_FOV_COST = 15.0        # вне поля обзора (по бокам у робота): можно, но только если иначе никак
MIN_HOLE_PX = 2500             # область кадра без глубины больше этого — препятствие «нет глубины»
ATTACH_DIST = 0.08             # выступ ближе этого к найденному объекту считается его частью, м
ATTACH_MAX_TAKE_CM2 = 80       # к лёгкой цели присоединяем только мелкие куски
ATTACH_MAX_TAKE_HOLE_CM2 = 200  # ... а области без глубины — до этой площади
HOLE_MAX_B = 1.3               # дальше этого пропуски глубины — норма для камеры, не препятствие
SELF_MASK_DIR = Path(__file__).with_name("nav_self_masks")

KIND_TAKE, KIND_AVOID, KIND_UNKNOWN = "take", "avoid", "unknown"
RU = {KIND_TAKE: "ЗАБРАТЬ", KIND_AVOID: "ОБЪЕХАТЬ", KIND_UNKNOWN: "НЕИЗВЕСТНО"}
COLORS = {KIND_TAKE: (60, 200, 60), KIND_AVOID: (40, 40, 230), KIND_UNKNOWN: (0, 160, 255)}   # BGR


# ---------------------------------------------------------------- геометрия
class Floor:
    """Координаты пола: начало под камерой, a — вправо, b — вперёд, h — вверх."""

    def __init__(self, plane):
        n = np.array(plane[:3], float)
        k = np.linalg.norm(n)
        n, d = n / k, plane[3] / k
        if d < 0:
            n, d = -n, -d
        self.n, self.d, self.O = n, d, -d * n
        r = np.array([1.0, 0, 0]) - n[0] * n
        self.r = r / np.linalg.norm(r)
        f = np.cross(n, self.r)
        self.f = f if f[2] > 0 else -f

    def to_floor(self, p):
        q = p - self.O
        return np.stack([q @ self.r, q @ self.f, q @ self.n], axis=1)

    def to_cam(self, abh):
        return self.O + abh[:, :1] * self.r + abh[:, 1:2] * self.f + abh[:, 2:3] * self.n

    def ray_floor(self, rays):
        """Пересечение лучей из камеры с полом -> точки пола (a, b)."""
        t = -self.d / (rays @ self.n)
        p = rays * t[:, None]
        ok = t > 0
        return self.to_floor(p)[:, :2], ok


class Cam:
    def __init__(self, cam):
        self.fx, self.fy, self.cx, self.cy = cam["fx"], cam["fy"], cam["cx"], cam["cy"]
        self.W, self.H = cam["width"], cam["height"]

    def rays(self, u, v):
        return np.stack([(u - self.cx) / self.fx, (v - self.cy) / self.fy, np.ones_like(u, dtype=float)], axis=1)

    def project(self, p):
        z = p[:, 2]
        return self.fx * p[:, 0] / z + self.cx, self.fy * p[:, 1] / z + self.cy, z


NA = int(round((A_RANGE[1] - A_RANGE[0]) / GRID))
NB = int(round((B_RANGE[1] - B_RANGE[0]) / GRID))


def to_cell(ab):
    i = np.floor((ab[:, 1] - B_RANGE[0]) / GRID).astype(int)      # строка: вперёд
    j = np.floor((ab[:, 0] - A_RANGE[0]) / GRID).astype(int)      # столбец: вправо
    ok = (i >= 0) & (i < NB) & (j >= 0) & (j < NA)
    return i, j, ok


def cell_center(i, j):
    return A_RANGE[0] + (j + 0.5) * GRID, B_RANGE[0] + (i + 0.5) * GRID


# ---------------------------------------------------------------- карта сверху
def height_map(depth, floor, cam, ignore=None):
    """Макс. высота над полом в клетке (nan — клетка не видна) и поле обзора камеры."""
    valid = depth > 0
    if ignore is not None:
        valid &= ~ignore
    v, u = np.nonzero(valid)
    z = depth[v, u].astype(np.float64) / 1000.0
    p = cam.rays(u.astype(float), v.astype(float)) * z[:, None]
    abh = floor.to_floor(p)
    i, j, ok = to_cell(abh[:, :2])
    hm = np.full((NB, NA), -np.inf)
    np.maximum.at(hm, (i[ok], j[ok]), abh[ok, 2])
    observed = np.isfinite(hm)
    # поле обзора: проекция всех пикселей кадра на пол (там, где луч смотрит вниз)
    uu, vv = np.meshgrid(np.arange(0, cam.W, 4, dtype=float), np.arange(0, cam.H, 4, dtype=float))
    ab, okr = floor.ray_floor(cam.rays(uu.ravel(), vv.ravel()))
    fi, fj, fok = to_cell(ab)
    fov = np.zeros((NB, NA), np.uint8)
    fov[fi[fok & okr], fj[fok & okr]] = 1
    fov = cv2.morphologyEx(fov, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8)) > 0
    # мелкие дыры в карте (редкие точки вдали) — заполняем соседями, если они пол
    hm_filled = hm.copy()
    hm_filled[~observed] = np.nan
    small_gaps = (~observed) & (cv2.morphologyEx(observed.astype(np.uint8), cv2.MORPH_CLOSE,
                                                 np.ones((3, 3), np.uint8)) > 0)
    if small_gaps.any():
        tmp = np.where(observed, hm, 0).astype(np.float32)
        blur = cv2.blur(tmp, (3, 3))
        cnt = cv2.blur(observed.astype(np.float32), (3, 3))
        hm_filled[small_gaps] = (blur / np.maximum(cnt, 1e-6))[small_gaps]
        observed = observed | small_gaps
    return hm_filled, observed, fov


def refine_floor(depth, floor, cam, ignore=None, iters=4):
    """Уточняет плоскость пола по самому кадру: берёт точки почти на полу (|h| < 3 см, ближе 2 м),
    подгоняет плоскость МНК и отбрасывает выбросы. Нужна, когда плоскость из meta.json чуть неточна
    и дальний пол «поднимается» над ней на несколько сантиметров."""
    H, W = depth.shape
    vv, uu = np.mgrid[0:H:4, 0:W:4]
    m = depth[vv, uu] > 0
    if ignore is not None:
        m &= ~ignore[vv, uu]
    v, u = vv[m], uu[m]
    if len(v) < 2000:
        return floor
    z = depth[v, u].astype(np.float64) / 1000.0
    p = cam.rays(u.astype(float), v.astype(float)) * z[:, None]
    abh = floor.to_floor(p)
    keep = (np.abs(abh[:, 2]) < 0.03) & (abh[:, 1] < 2.0)
    for _ in range(iters):
        if keep.sum() < 1500:
            return floor
        q = p[keep]
        c = q.mean(axis=0)
        n = np.linalg.svd(q - c, full_matrices=False)[2][-1]
        if n @ floor.n < 0:
            n = -n
        d = -n @ c
        res = p @ n + d
        keep = (np.abs(res) < 0.012) & (abh[:, 1] < 2.0)
    if np.degrees(np.arccos(np.clip(n @ floor.n, -1, 1))) > 5:      # слишком сильно ушли — не верим
        return floor
    return Floor([n[0], n[1], n[2], d])


def object_footprint(pix_mask, depth, floor, cam):
    """Пятно объекта на полу (клетки) по его пикселям: точки с глубиной + проекция пикселей без глубины."""
    fp = np.zeros((NB, NA), np.uint8)
    v, u = np.nonzero(pix_mask & (depth > 0))
    hmax = 0.0
    if len(v):
        z = depth[v, u].astype(np.float64) / 1000.0
        abh = floor.to_floor(cam.rays(u.astype(float), v.astype(float)) * z[:, None])
        i, j, ok = to_cell(abh[:, :2])
        fp[i[ok], j[ok]] = 1
        hmax = float(np.percentile(abh[:, 2], 98))
    v, u = np.nonzero(pix_mask & (depth == 0))          # стекло, блестящий металл: глубины нет
    if len(v):
        ab, okr = floor.ray_floor(cam.rays(u.astype(float), v.astype(float)))
        i, j, ok = to_cell(ab)
        ok &= okr
        if fp.any():
            # проекция «на пол» уводит пиксель дальше настоящего места; берём только рядом с телом
            near = cv2.dilate(fp, np.ones((15, 15), np.uint8)) > 0
            ok &= near[np.clip(i, 0, NB - 1), np.clip(j, 0, NA - 1)]
        fp[i[ok], j[ok]] = 1
    fp = cv2.morphologyEx(fp, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    cnts, _ = cv2.findContours(fp, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(fp, cnts, -1, 1, thickness=-1)
    return fp > 0, hmax


def measure_geom_volume(pix_mask, depth, floor, cam, cell=0.004):
    """Объём объекта по видимой глубине (как в бейзлайне кейса): пол делится на клетки 4×4 мм,
    в каждой берётся самая высокая точка объекта, высоты складываются. Точки без глубины не входят."""
    v, u = np.nonzero(pix_mask & (depth > 0))
    if len(v) < 30:
        return float("nan")
    z = depth[v, u].astype(np.float64) / 1000.0
    abh = floor.to_floor(cam.rays(u.astype(float), v.astype(float)) * z[:, None])
    keys = np.floor(abh[:, 0] / cell).astype(np.int64) * 100000 + np.floor(abh[:, 1] / cell).astype(np.int64)
    order = np.argsort(keys)
    k, h = keys[order], abh[order, 2]
    starts = np.r_[0, np.nonzero(np.diff(k))[0] + 1]
    tops = np.maximum.reduceat(h, starts)
    return float(np.clip(tops, 0, None).sum() * cell * cell * 1e6)


def polygon_mask(segmentation, h, w):
    m = np.zeros((h, w), np.uint8)
    for seg in segmentation:
        cv2.fillPoly(m, [np.array(seg, np.float32).reshape(-1, 2).round().astype(np.int32)], 1)
    return m > 0


def pixel_heights(depth, floor, cam):
    """Высота над полом для каждого пикселя (nan без глубины)."""
    H, W = depth.shape
    vv, uu = np.mgrid[0:H, 0:W]
    z = depth.astype(np.float64) / 1000.0
    p = cam.rays(uu.ravel().astype(float), vv.ravel().astype(float)) * z.ravel()[:, None]
    h = (p - floor.O) @ floor.n
    h = h.reshape(H, W)
    h[depth == 0] = np.nan
    return h


# ---------------------------------------------------------------- тело робота в кадре
def robot_self_mask(depth, hpix, meta):
    """Пиксели, где в кадре видна сама машина: корпус слева внизу и боковая щётка справа.
    1) статическая карта по серии съёмки (доля обучающих кадров, где пиксель без глубины или выше пола);
    2) всё без глубины/высокое, что касается нижнего края кадра у боков (вращающаяся щётка)."""
    H, W = depth.shape
    m = np.zeros((H, W), bool)
    files = sorted(SELF_MASK_DIR.glob("self_*.png")) if SELF_MASK_DIR.exists() else []
    if files:
        tilt = float(meta.get("tilt_deg", 30))
        best = min(files, key=lambda f: abs(float(f.stem.split("_")[1]) - tilt))
        frac = cv2.imread(str(best), cv2.IMREAD_GRAYSCALE).astype(np.float32) / 255
        if frac.shape != (H, W):
            frac = cv2.resize(frac, (W, H))
        vv = np.arange(H)[:, None]
        st = ((frac >= 0.8) & (vv > 0.4 * H)) | ((frac >= 0.5) & (vv > 0.6 * H))
        st = cv2.morphologyEx(st.astype(np.uint8), cv2.MORPH_OPEN, np.ones((7, 7), np.uint8))
        n, lab, stats, _ = cv2.connectedComponentsWithStats(st, 8)
        for k in range(1, n):
            if stats[k, cv2.CC_STAT_AREA] > 3000:
                m |= lab == k
    bad = ((depth == 0) | (np.nan_to_num(hpix, nan=1) > OBST_H)).astype(np.uint8)
    bad[: int(0.45 * H)] = 0
    n, lab, stats, cent = cv2.connectedComponentsWithStats(bad, 8)
    for k in range(1, n):
        x, y, w, h, area = stats[k]
        touches_bottom = y + h >= H - 2
        at_side = cent[k][0] < 0.3 * W or cent[k][0] > 0.7 * W
        if touches_bottom and at_side and area > 1500:
            m |= lab == k
    return cv2.dilate(m.astype(np.uint8), np.ones((15, 15), np.uint8)) > 0


# ---------------------------------------------------------------- источники объектов
def objects_from_labels(labels_path):
    data = json.loads(Path(labels_path).read_text(encoding="utf-8"))
    by_img = defaultdict(list)
    for a in data["annotations"]:
        by_img[a["image_id"]].append(a)
    out = {}
    for im in data["images"]:
        fid = Path(im["file_name"]).parent.name
        out[fid] = [{"segmentation": a["segmentation"], "bbox": a["bbox"],
                     "take": int(a["attributes"]["take"]),
                     "label": f'{a["attributes"]["form"]} {a["attributes"]["material"]}',
                     "mass": a["attributes"].get("mass_g")} for a in by_img[im["id"]]]
    return out


def objects_from_submission(path):
    out = {}
    with open(path, encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            objs = []
            cell = (row.get("prediction") or "").strip()
            if cell and cell.lower() != "none":
                for part in cell.split(";"):
                    t = part.split()
                    if len(t) != 10:
                        continue
                    x1, y1, x2, y2, sc = map(float, t[:5])
                    objs.append({"box": (x1, y1, x2, y2), "score": sc, "take": int(float(t[9])),
                                 "label": f"{t[5]} {t[6]}", "mass": float(t[8])})
            out[row["image_id"]] = objs
    return out


def object_pixels(obj, depth, hpix):
    H, W = depth.shape
    if "segmentation" in obj:
        return polygon_mask(obj["segmentation"], H, W)
    x1, y1, x2, y2 = (int(round(v)) for v in obj["box"])
    x1, y1, x2, y2 = max(0, x1), max(0, y1), min(W, x2), min(H, y2)
    m = np.zeros((H, W), bool)
    sub_h = hpix[y1:y2, x1:x2]
    high = np.nan_to_num(sub_h, nan=-1) > OBJ_H
    holes = np.isnan(sub_h)
    if high.any():
        near = cv2.dilate(high.astype(np.uint8), np.ones((15, 15), np.uint8)) > 0
        m[y1:y2, x1:x2] = high | (holes & near)
    else:                          # плоский объект или всё в дырах — берём рамку
        m[y1:y2, x1:x2] = True
    return m


# ---------------------------------------------------------------- планирование
def disk(r_cells):
    r = max(1, int(round(r_cells)))
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))


def dijkstra(cost, start, goals_mask):
    """Кратчайший путь по сетке 8-связности до любой клетки goals_mask. cost=inf — непроходимо."""
    NBi, NAi = cost.shape
    dist = np.full(cost.shape, np.inf)
    prev = -np.ones(cost.shape + (2,), int)
    si, sj = start
    if not np.isfinite(cost[si, sj]):
        return None
    dist[si, sj] = 0.0
    pq = [(0.0, si, sj)]
    steps = [(-1, 0, 1), (1, 0, 1), (0, -1, 1), (0, 1, 1),
             (-1, -1, math.sqrt(2)), (-1, 1, math.sqrt(2)), (1, -1, math.sqrt(2)), (1, 1, math.sqrt(2))]
    while pq:
        d, i, j = heapq.heappop(pq)
        if d > dist[i, j]:
            continue
        if goals_mask[i, j]:
            path = [(i, j)]
            while (path[-1][0], path[-1][1]) != (si, sj):
                pi, pj = prev[path[-1][0], path[-1][1]]
                path.append((pi, pj))
            return path[::-1]
        for di, dj, w in steps:
            ni, nj = i + di, j + dj
            if 0 <= ni < NBi and 0 <= nj < NAi:
                c = cost[ni, nj]
                if not np.isfinite(c):
                    continue
                nd = d + w * c
                if nd < dist[ni, nj]:
                    dist[ni, nj] = nd
                    prev[ni, nj] = (i, j)
                    heapq.heappush(pq, (nd, ni, nj))
    return None


def line_free(cost, p, q):
    n = int(max(abs(q[0] - p[0]), abs(q[1] - p[1]))) + 1
    ii = np.linspace(p[0], q[0], n).round().astype(int)
    jj = np.linspace(p[1], q[1], n).round().astype(int)
    c = cost[ii, jj]
    return np.all(np.isfinite(c)) and np.all(c <= 1.0 + 1e-9)


def smooth(path, cost):
    """Срезаем углы: прямой отрезок, если он идёт только по свободному видимому полу."""
    if len(path) < 3:
        return path
    out, k = [path[0]], 0
    while k < len(path) - 1:
        nxt = k + 1
        for m in range(len(path) - 1, k, -1):
            if line_free(cost, path[k], path[m]):
                nxt = m
                break
        out.append(path[nxt])
        k = nxt
    return out


def plan(hm, observed, fov, objects, robot_w, margin):
    """objects: [{kind, footprint}] -> (путь клетками, список посещённых целей, статус, карты)."""
    avoid = np.zeros_like(observed)
    for o in objects:
        if o["kind"] in (KIND_AVOID, KIND_UNKNOWN):
            avoid |= o["footprint"]
    infl = cv2.dilate(avoid.astype(np.uint8), disk((robot_w / 2 + margin) / GRID)) > 0
    cost = np.where(observed, 1.0, UNOBSERVED_COST)
    cost[~fov] = OUTSIDE_FOV_COST
    cost[infl] = np.inf
    # препятствие, которое уходит за край обзора, может продолжаться за ним: невидимый пол рядом закрыт
    edge = cv2.dilate((~fov).astype(np.uint8), disk(0.03 / GRID)) > 0
    fov_rows = np.nonzero(fov.any(axis=1))[0]
    if len(fov_rows):
        edge[: fov_rows.min() + int(0.08 / GRID)] = False      # ближний край обзора — это нос робота, не бок
    at_edge = np.zeros_like(avoid)
    for o in objects:
        if o["kind"] in (KIND_AVOID, KIND_UNKNOWN) and (o["footprint"] & edge).any():
            at_edge |= o["footprint"]
    near_obst = cv2.dilate(at_edge.astype(np.uint8), disk((robot_w / 2 + margin + 0.25) / GRID)) > 0
    cost[(~fov) & near_obst] = np.inf

    # старт: ближайший к роботу видимый ряд по центру
    rows = np.nonzero(fov.any(axis=1))[0]
    if len(rows) == 0:
        return None, [], "нет обзора", cost, infl, []
    start_row = rows.min() + 2
    jc = int((0 - A_RANGE[0]) / GRID)
    start = (start_row, jc)
    if not np.isfinite(cost[start]):
        free = np.nonzero(np.isfinite(cost[start_row]))[0]
        if len(free) == 0:
            return None, [], "стоп: прямо перед роботом препятствие, позвать оператора", cost, infl, []
        start = (start_row, int(free[np.argmin(np.abs(free - jc))]))

    # цели «забрать» — по удалённости вперёд
    targets = [o for o in objects if o["kind"] == KIND_TAKE]
    targets.sort(key=lambda o: np.nonzero(o["footprint"])[0].mean())
    path, visited, cur, breaks = [start], [], start, []
    for o in targets:
        fi, fj = np.nonzero(o["footprint"])
        ci, cj = fi.mean(), fj.mean()
        core = np.zeros(o["footprint"].shape, np.uint8)
        cv2.circle(core, (int(round(cj)), int(round(ci))), max(2, int(0.06 / GRID)), 1, -1)
        core = core > 0
        free = np.isfinite(cost)
        goal = o["footprint"] & core & free
        if not goal.any():                         # центр у препятствия — заезжаем на доступный край
            goal = o["footprint"] & free
        if not goal.any():
            o["status"] = "недоступно: рядом препятствие"
            continue
        leg = dijkstra(cost, cur, goal)
        if leg is None:
            o["status"] = "недоступно: нет проезда"
            continue
        # проезжаем объект насквозь до его дальнего края, чтобы конвейер его забрал
        ii, jj = np.nonzero(goal)
        far = (int(ii.max()), int(round(jj[ii == ii.max()].mean())))
        through = dijkstra(cost, leg[-1], _single(goal.shape, far)) or [leg[-1]]
        path += leg[1:]
        breaks.append(len(path) - 1)
        path += through[1:]
        cur = path[-1]
        breaks.append(len(path) - 1)
        o["status"] = "забрать"
        visited.append(o)

    # дальний край обзора
    far_row = rows.max() - 3
    band = np.zeros_like(observed)
    band[max(far_row - 8, 0):far_row + 1] = True
    band &= np.isfinite(cost)
    leg = dijkstra(cost, cur, band) if band.any() else None
    if leg is None:
        status = "стоп: дальше пути нет, позвать оператора" if not targets or not visited else \
                 "цели собраны, дальше пути нет: позвать оператора"
        return (path if len(path) > 1 else None), visited, status, cost, infl, breaks
    path += leg[1:]
    return path, visited, "ок", cost, infl, breaks


def _single(shape, ij):
    m = np.zeros(shape, bool)
    m[ij] = True
    return m


# ---------------------------------------------------------------- рисование
def _font(size):
    try:
        from PIL import ImageFont
        for p in ("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
                  "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
                  "/Library/Fonts/Arial Unicode.ttf", "C:/Windows/Fonts/arialbd.ttf"):
            if Path(p).exists():
                return ImageFont.truetype(p, size)
    except ImportError:
        pass
    return None


def put_texts(img, items):
    """items: [(text, (x, y), bgr, size)] — русский текст через Pillow, иначе латиница cv2."""
    font_cache = {}
    try:
        from PIL import Image, ImageDraw
        pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        dr = ImageDraw.Draw(pil)
        ok = True
    except ImportError:
        ok = False
    for text, (x, y), bgr, size in items:
        f = font_cache.setdefault(size, _font(size)) if ok else None
        if f is not None:
            tw, th = dr.textbbox((0, 0), text, font=f)[2:]
            dr.rectangle([x - 3, y - 3, x + tw + 3, y + th + 5], fill=(0, 0, 0))
            dr.text((x, y), text, font=f, fill=(bgr[2], bgr[1], bgr[0]))
        else:
            if ok:
                img[:] = cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)
                ok = False
            cv2.putText(img, text.encode("ascii", "ignore").decode() or "?", (x, y + size), 0, size / 30, bgr, 2)
    if ok:
        img[:] = cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)
    return img


def draw_camera(color, floor, cam, path_ab, robot_w, objects, status, swath=None, hpix=None):
    img = color.copy()
    over = img.copy()
    if swath is not None and swath.any():
        # каждый пиксель пола проецируем на карту и красим, если он в полосе робота
        H, W = img.shape[:2]
        vv, uu = np.mgrid[0:H, 0:W]
        ab, okr = floor.ray_floor(cam.rays(uu.ravel().astype(float), vv.ravel().astype(float)))
        i, j, ok = to_cell(ab)
        ok &= okr
        inside = np.zeros(H * W, bool)
        inside[ok] = swath[i[ok], j[ok]] > 0
        inside = inside.reshape(H, W)
        if hpix is not None:
            inside &= np.nan_to_num(hpix, nan=0) < 0.012          # красим пол, а не предметы
        over[inside] = (255, 160, 40)
    if path_ab is not None and len(path_ab) > 1:
        pts = np.array(path_ab)
        cen = floor.to_cam(np.hstack([pts, np.zeros((len(pts), 1))]))
        u, v, z = cam.project(cen)
        cv2.polylines(img, [np.stack([u, v], 1).round().astype(np.int32)], False, (255, 120, 0), 4, cv2.LINE_AA)
    img = cv2.addWeighted(over, 0.35, img, 0.65, 0)
    texts, boxes = [], []
    for o in objects:
        m = o["pixels"]
        cnts, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(img, cnts, -1, COLORS[o["kind"]], 3)
        ys, xs = np.nonzero(m)
        if len(xs) and not (o["kind"] == KIND_UNKNOWN and o["footprint"].sum() < 80 and o["note"] != "нет глубины"):
            label = RU[o["kind"]]
            if o.get("note"):
                label += " · " + o["note"]
            x, y = int(xs.min()), max(40, int(ys.min()) - 28)
            box = (x, y, x + 12 * len(label), y + 26)
            if all(box[2] < b[0] or b[2] < box[0] or box[3] < b[1] or b[3] < box[1] for b in boxes):
                boxes.append(box)
                texts.append((label, (x, y), COLORS[o["kind"]], 20))
    texts.append((f"Маршрут: {status}", (12, 10), (255, 255, 255), 24))
    return put_texts(img, texts)


def draw_bev(hm, observed, fov, objects, infl, path_cells, scale=3):
    NBi, NAi = hm.shape
    img = np.full((NBi, NAi, 3), 25, np.uint8)
    img[fov] = (60, 60, 60)
    img[observed & fov] = (150, 150, 150)
    hcl = np.clip(np.nan_to_num(hm, nan=0) / 0.15, 0, 1)
    shade = (observed & (hcl > 0.1))
    img[shade] = (150 * (1 - hcl[shade, None]) + 255 * hcl[shade, None]).astype(np.uint8)
    img[infl & fov] = (img[infl & fov] * 0.5 + np.array([80, 80, 200]) * 0.5).astype(np.uint8)
    for o in objects:
        img[o["footprint"]] = COLORS[o["kind"]]
    if path_cells:
        for (i0, j0), (i1, j1) in zip(path_cells[:-1], path_cells[1:]):
            cv2.line(img, (j0, i0), (j1, i1), (255, 140, 0), 1)
    img = img[::-1]                                  # вперёд — вверх
    img = cv2.resize(img, (NAi * scale, NBi * scale), interpolation=cv2.INTER_NEAREST)
    # робот внизу по центру
    cx = int((0 - A_RANGE[0]) / GRID * scale)
    cv2.rectangle(img, (cx - 20, img.shape[0] - 8), (cx + 20, img.shape[0] - 1), (255, 255, 255), -1)
    for k in range(1, int(B_RANGE[1] / 0.5) + 1):                         # метки по 0,5 м
        y = img.shape[0] - int((k * 0.5 - B_RANGE[0]) / GRID * scale)
        if 0 < y < img.shape[0]:
            cv2.line(img, (0, y), (12, y), (255, 255, 255), 1)
            cv2.putText(img, f"{k * 0.5:.1f}m", (14, y + 4), 0, 0.4, (255, 255, 255), 1)
    return img


# ---------------------------------------------------------------- один кадр
def process_frame(frame_dir, cam, objs, robot_w, margin, avoid_mass_g=None, gt_objs=None,
                  geom_volume_ml=None, ignore_take=False):
    color = cv2.imread(str(frame_dir / "color.png"))
    depth = cv2.imread(str(frame_dir / "depth.png"), cv2.IMREAD_UNCHANGED)
    meta = json.loads((frame_dir / "meta.json").read_text(encoding="utf-8"))
    floor = Floor(meta["floor_plane"])
    hpix = pixel_heights(depth, floor, cam)
    self_m = robot_self_mask(depth, hpix, meta)
    floor = refine_floor(depth, floor, cam, ignore=self_m)
    hpix = pixel_heights(depth, floor, cam)
    hm, observed, fov = height_map(depth, floor, cam, ignore=self_m)

    objects, known = [], np.zeros((NB, NA), bool)
    for o in objs:
        pix = object_pixels(o, depth, hpix)
        if pix.sum() < 30:
            continue
        fp, hmax = object_footprint(pix, depth, floor, cam)
        if not fp.any():
            continue
        kind = KIND_TAKE if (ignore_take or o["take"] == 1) else KIND_AVOID
        note = o.get("label", "")
        mass = o.get("mass")
        if mass is not None:
            note += f", ~{mass:.0f} г"
        safety = False
        if (kind == KIND_TAKE and avoid_mass_g is not None and mass is not None
                and np.isfinite(mass) and mass > avoid_mass_g):
            kind, safety = KIND_AVOID, True
            note += f" (запас: >{avoid_mass_g:.0f} г)"
        gvol = measure_geom_volume(pix, depth, floor, cam)
        if kind == KIND_TAKE and geom_volume_ml is not None and np.isfinite(gvol) and gvol > geom_volume_ml:
            kind, safety = KIND_AVOID, True
            note += f" (объём по глубине {gvol:.0f} мл > {geom_volume_ml:.0f})"
        objects.append({"kind": kind, "footprint": fp, "pixels": pix, "note": note, "height_m": hmax,
                        "source": o, "safety": safety, "geom_vol": gvol})
        known |= fp

    # неопознанные выступы: всё, что выше OBST_H и не принадлежит найденным объектам
    high = observed & (np.nan_to_num(hm, nan=0) > OBST_H) & ~(cv2.dilate(known.astype(np.uint8), disk(3)) > 0)
    n, lab, stats, _ = cv2.connectedComponentsWithStats(high.astype(np.uint8), 8)
    H, W = depth.shape
    near_obj = [cv2.dilate(o["footprint"].astype(np.uint8), disk(ATTACH_DIST / GRID)) > 0 for o in objects]
    for k in range(1, n):
        fp = lab == k
        # выступ рядом с найденным объектом — кусок того же предмета (край, тень, прозрачная часть)
        owner = next((i for i, nm in enumerate(near_obj) if (fp & nm).any()), None)
        if owner is not None and objects[owner]["kind"] == KIND_TAKE and fp.sum() > ATTACH_MAX_TAKE_CM2:
            owner = None                       # крупный выступ рядом с лёгкой целью — может быть отдельным предметом
        if owner is not None:
            objects[owner]["footprint"] = objects[owner]["footprint"] | fp
            continue
        if stats[k, cv2.CC_STAT_AREA] * GRID * GRID * 1e4 < MIN_UNKNOWN_CM2:
            continue
        # пиксели кадра этого выступа: высокие пиксели, попадающие в его клетки
        hp = (np.nan_to_num(hpix, nan=0) > OBST_H) & ~self_m
        v, u = np.nonzero(hp)
        z = depth[v, u].astype(np.float64) / 1000.0
        ab = floor.to_floor(cam.rays(u.astype(float), v.astype(float)) * z[:, None])[:, :2]
        i, j, ok = to_cell(ab)
        sel = np.zeros(len(v), bool)
        sel[ok] = fp[i[ok], j[ok]]
        pix = np.zeros((H, W), bool)
        pix[v[sel], u[sel]] = True
        pix = cv2.morphologyEx(pix.astype(np.uint8), cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8)) > 0
        objects.append({"kind": KIND_UNKNOWN, "footprint": fp, "pixels": pix,
                        "note": f"выступ {stats[k, cv2.CC_STAT_AREA]} см²", "height_m": float(np.nanmax(hm[fp])),
                        "source": None})

    # большие области без глубины: чёрные, прозрачные, слишком близкие к камере предметы.
    # Пол под ними неизвестен — считаем препятствием, его место — проекция области на пол.
    holes = ((depth == 0) & ~self_m).astype(np.uint8)
    for o in objects:
        holes[cv2.dilate(o["pixels"].astype(np.uint8), np.ones((21, 21), np.uint8)) > 0] = 0
    holes = cv2.morphologyEx(holes, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    n, lab, stats, _ = cv2.connectedComponentsWithStats(holes, 8)
    for k in range(1, n):
        if stats[k, cv2.CC_STAT_AREA] < MIN_HOLE_PX:
            continue
        v, u = np.nonzero(lab == k)
        ab, okr = floor.ray_floor(cam.rays(u.astype(float), v.astype(float)))
        i, j, ok = to_cell(ab)
        ok &= okr
        if ok.sum() < 20 or np.median(ab[ok, 1]) > HOLE_MAX_B:
            continue
        fp = np.zeros((NB, NA), np.uint8)
        fp[i[ok], j[ok]] = 1
        fp = cv2.morphologyEx(fp, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8)) > 0
        if fp.sum() * GRID * GRID * 1e4 < MIN_UNKNOWN_CM2:
            continue
        owner = next((o for o in objects if o["kind"] != KIND_UNKNOWN and o["source"] is not None and
                      (fp & (cv2.dilate(o["footprint"].astype(np.uint8), disk(ATTACH_DIST / GRID)) > 0)).any()), None)
        if owner is not None and owner["kind"] == KIND_TAKE and fp.sum() > ATTACH_MAX_TAKE_HOLE_CM2:
            owner = None                       # область без глубины у цели — обычно её теневая сторона
        if owner is not None:
            owner["footprint"] = owner["footprint"] | fp
            continue
        objects.append({"kind": KIND_UNKNOWN, "footprint": fp, "pixels": lab == k,
                        "note": "нет глубины", "height_m": float("nan"), "source": None})

    path, visited, status, cost, infl, breaks = plan(hm, observed, fov, objects, robot_w, margin)
    path_s = None
    if path:
        # сглаживаем по участкам между целями, чтобы не срезать заезд на объект
        cuts = [0] + [b for b in breaks if 0 < b < len(path) - 1] + [len(path) - 1]
        path_s = [path[0]]
        for a0, a1 in zip(cuts[:-1], cuts[1:]):
            if a1 > a0:
                path_s += smooth(path[a0:a1 + 1], cost)[1:]
    # какую долю каждой цели накрывает полоса робота
    swath = np.zeros((NB, NA), np.uint8)
    if path_s:
        pts = np.array([[j, i] for i, j in path_s], np.int32)
        cv2.polylines(swath, [pts], False, 1, thickness=max(1, int(robot_w / GRID)))
    for o in objects:
        if o["kind"] == KIND_TAKE:
            cov = float((o["footprint"] & (swath > 0)).sum() / max(o["footprint"].sum(), 1))
            o["coverage"] = cov
            if o.get("status") == "забрать" and cov < 0.6:
                o["status"] = f"недобор: накрыто {cov:.0%}"
        elif path_s and (o["footprint"] & (swath > 0)).any():
            o["status"] = "ВНИМАНИЕ: полоса задевает"
    path_ab = [cell_center(i, j) for i, j in path_s] if path_s else None
    hpix_draw = hpix.copy()
    hpix_draw[self_m] = 1.0                        # корпус и щётку робота не красим
    cam_img = draw_camera(color, floor, cam, path_ab, robot_w, objects, status, swath, hpix_draw)
    cnts, _ = cv2.findContours(self_m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(cam_img, cnts, -1, (200, 200, 200), 2)
    bev = draw_bev(hm, observed, fov, objects, infl, path_s)
    h = cam_img.shape[0]
    bev_r = cv2.resize(bev, (int(bev.shape[1] * h / bev.shape[0]), h), interpolation=cv2.INTER_NEAREST)
    panel = np.hstack([cam_img, bev_r])

    report = {
        "frame": frame_dir.name, "status": status, "robot_width_m": robot_w, "margin_m": margin,
        "avoid_mass_g": avoid_mass_g, "geom_volume_ml": geom_volume_ml, "ignore_take": ignore_take,
        "path_floor_m": [[round(a, 3), round(b, 3)] for a, b in path_ab] if path_ab else [],
        "path_length_m": round(float(sum(math.dist(p, q) for p, q in zip(path_ab[:-1], path_ab[1:]))), 3) if path_ab else 0.0,
        "objects": [],
    }
    # проверка по истинной разметке: задели ли тяжёлые, убрали ли лёгкие
    if gt_objs is not None:
        ev = []
        for g in gt_objs:
            pix = object_pixels(g, depth, hpix)
            fp, _ = object_footprint(pix, depth, floor, cam)
            if not fp.any():
                continue
            cov = float((fp & (swath > 0)).sum() / max(fp.sum(), 1))
            heavy = int(g["take"]) == 0
            ev.append({"truth": "тяжёлый" if heavy else "лёгкий", "label": g.get("label", ""),
                       "mass_g": g.get("mass"), "covered": round(cov, 2),
                       "outcome": ("НАЕЗД" if cov > 0.05 else "объехал") if heavy else
                                  ("убрал" if cov >= 0.6 else "не убрал")})
        report["gt_eval"] = ev
    for o in objects:
        ii, jj = np.nonzero(o["footprint"])
        a, b = cell_center(ii.mean(), jj.mean())
        report["objects"].append({
            "decision": RU[o["kind"]], "kind": o["kind"], "note": o["note"],
            "center_floor_m": [round(a, 3), round(b, 3)], "footprint_cm2": int(o["footprint"].sum()),
            "height_cm": (round(o["height_m"] * 100, 1) if np.isfinite(o["height_m"]) else None),
            "covered_by_robot": round(o["coverage"], 2) if "coverage" in o else None,
            "avoided_by_safety_margin": bool(o.get("safety", False)),
            "geom_volume_ml": (round(o["geom_vol"], 1) if np.isfinite(o.get("geom_vol", float("nan"))) else None), "status": o.get("status", "объезд")})
    return panel, report


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default="dataset", help="папка dataset (там camera.json)")
    ap.add_argument("--frames", nargs="*", default=[], help="папки кадров относительно --data")
    ap.add_argument("--frames-glob", default=None, help='шаблон, например "real/test/*"')
    ap.add_argument("--labels", default=None, help="разметка COCO (real_train.json и т.п.)")
    ap.add_argument("--submission", default=None, help="ответ модели в формате кейса")
    ap.add_argument("--robot-width", type=float, default=0.40, help="ширина робота, м")
    ap.add_argument("--margin", type=float, default=0.05, help="запас до препятствия, м")
    ap.add_argument("--avoid-mass-g", type=float, default=None,
                    help="режим осторожности: объезжать всё тяжелее этого (г), даже при take=1")
    ap.add_argument("--geom-volume-ml", type=float, default=None,
                    help="объезжать объект, если его объём по видимой глубине больше этого (мл)")
    ap.add_argument("--ignore-take", action="store_true",
                    help="не верить take из ответа модели: решают только --geom-volume-ml/--avoid-mass-g")
    ap.add_argument("--gt-labels", default=None,
                    help="истинная разметка для проверки: доля задетых тяжёлых и убранных лёгких")
    ap.add_argument("--out", default="nav_out")
    args = ap.parse_args()

    data, out = Path(args.data), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    cam = Cam(json.loads((data / "camera.json").read_text(encoding="utf-8")))
    frames = [data / f for f in args.frames]
    if args.frames_glob:
        frames += sorted(Path(p) for p in glob.glob(str(data / args.frames_glob)) if Path(p).is_dir())
    if args.labels:
        src = objects_from_labels(args.labels)
    elif args.submission:
        src = objects_from_submission(args.submission)
    else:
        src = {}
    gt = objects_from_labels(args.gt_labels) if args.gt_labels else None
    reports = []
    for fdir in frames:
        panel, rep = process_frame(fdir, cam, src.get(fdir.name, []), args.robot_width, args.margin,
                                   args.avoid_mass_g, gt.get(fdir.name, []) if gt is not None else None,
                                   args.geom_volume_ml, args.ignore_take)
        cv2.imwrite(str(out / f"{fdir.name}_nav.jpg"), panel, [cv2.IMWRITE_JPEG_QUALITY, 88])
        reports.append(rep)
        kinds = defaultdict(int)
        for o in rep["objects"]:
            kinds[o["decision"]] += 1
        extra = sum(o["avoided_by_safety_margin"] for o in rep["objects"])
        print(f"{fdir.name}: {rep['status']} | путь {rep['path_length_m']} м | " +
              ", ".join(f"{k}: {v}" for k, v in kinds.items()) + (f" | объехать по запасу: {extra}" if extra else ""))
    if gt is not None:
        ev = [e for r in reports for e in r.get("gt_eval", [])]
        heavy = [e for e in ev if e["truth"] == "тяжёлый"]
        light = [e for e in ev if e["truth"] == "лёгкий"]
        hit = sum(e["outcome"] == "НАЕЗД" for e in heavy)
        got = sum(e["outcome"] == "убрал" for e in light)
        stops = sum(r["status"].startswith("стоп") for r in reports)
        summary = {"frames": len(reports), "heavy_total": len(heavy), "heavy_hit": hit,
                   "heavy_hit_rate": round(hit / len(heavy), 3) if heavy else None,
                   "light_total": len(light), "light_collected": got,
                   "light_collected_rate": round(got / len(light), 3) if light else None,
                   "frames_stopped": stops}
        (out / "nav_eval.json").write_text(json.dumps(summary, ensure_ascii=False, indent=1), encoding="utf-8")
        print("Проверка по разметке:",
              f"тяжёлых задето {hit}/{len(heavy)}" + (f" ({hit / len(heavy):.0%})" if heavy else ""),
              f"| лёгких убрано {got}/{len(light)}" + (f" ({got / len(light):.0%})" if light else ""),
              f"| остановок {stops}/{len(reports)}")
    (out / "nav_report.json").write_text(json.dumps(reports, ensure_ascii=False, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
