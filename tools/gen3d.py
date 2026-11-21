"""
3D-генератор обучающих кадров для кейса Astramis.

Идея: не рисовать мусор из примитивов, а брать НАСТОЯЩИЕ объекты из обучающих кадров
и переносить их в 3D на пустой пол другого кадра.

1. Объект из реального кадра (маска из разметки + карта глубины) превращается в облако
   точек в координатах пола: (вдоль, поперёк, высота над полом). Дыры в глубине (стекло,
   блестящий металл) остаются дырами: мы помним, какие точки были без глубины.
2. Облако поворачивается вокруг вертикали, может отражаться, масштабируется и ставится
   в новое место на пустом полу другого кадра (с его собственной плоскостью пола).
3. Точки проецируются в камеру с z-буфером: ближние закрывают дальние, объекты закрывают
   друг друга честно. Мелкие щели закрываются, края слегка смягчаются, под объектом мягкая тень.
4. Разметка пересчитывается: объём и масса × s³, take по массе ≤ 500 г. object_id
   сохраняется, чтобы при валидации по предметам не было утечки.

Используются только обучающие кадры (real/train), тест не читается.

Запуск:
    python gen3d.py --data dataset --out dataset/gen3d --n 1000 --seed 0
    python gen3d.py ... --exclude-objects b06,s45   # не брать эти предметы (для честной валидации)

Нужно: numpy, opencv-python.
"""
import argparse
import json
import math
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

HEAVY_G = 500.0


# ---------------------------------------------------------------- геометрия
class Floor:
    """Система координат пола для кадра: начало на полу под камерой, r — вправо, f — вперёд, n — вверх."""

    def __init__(self, plane):
        a, b, c, d = plane
        n = np.array([a, b, c], float)
        k = np.linalg.norm(n)
        n, d = n / k, d / k
        if d < 0:                      # камера над полом: высота камеры n·0 + d > 0
            n, d = -n, -d
        self.n, self.d = n, d
        self.O = -d * n
        r = np.array([1.0, 0, 0]) - n[0] * n
        self.r = r / np.linalg.norm(r)
        f = np.cross(n, self.r)
        if f[2] < 0:
            f = -f
        self.f = f

    def to_floor(self, p):
        q = p - self.O
        return np.stack([q @ self.r, q @ self.f, q @ self.n], axis=1)

    def to_cam(self, abh):
        return self.O + abh[:, :1] * self.r + abh[:, 1:2] * self.f + abh[:, 2:3] * self.n


class Cam:
    def __init__(self, cam):
        self.fx, self.fy, self.cx, self.cy = cam["fx"], cam["fy"], cam["cx"], cam["cy"]
        self.W, self.H = cam["width"], cam["height"]

    def unproject(self, u, v, z):
        return np.stack([(u - self.cx) / self.fx * z, (v - self.cy) / self.fy * z, z], axis=1)

    def project(self, p):
        z = p[:, 2]
        return self.fx * p[:, 0] / z + self.cx, self.fy * p[:, 1] / z + self.cy, z


def polygon_mask(segmentation, h, w):
    m = np.zeros((h, w), np.uint8)
    for seg in segmentation:
        cv2.fillPoly(m, [np.array(seg, np.float32).reshape(-1, 2).round().astype(np.int32)], 1)
    return m > 0


# ---------------------------------------------------------------- исходные объекты
def extract_object(frame_dir, ann, cam):
    """Облако точек объекта в координатах пола его кадра."""
    color = cv2.imread(str(frame_dir / "color.png"))
    depth = cv2.imread(str(frame_dir / "depth.png"), cv2.IMREAD_UNCHANGED).astype(np.float32)
    meta = json.loads((frame_dir / "meta.json").read_text(encoding="utf-8"))
    mask = polygon_mask(ann["segmentation"], *depth.shape)
    valid = mask & (depth > 0)
    if valid.sum() < 50:
        return None
    holes = mask & (depth == 0)
    # глубину в дырах восстанавливаем по соседям, чтобы знать, где точка в 3D; сама дыра запоминается
    if holes.any():
        x, y, w, h = cv2.boundingRect(mask.astype(np.uint8))
        pad = 20
        x0, y0 = max(0, x - pad), max(0, y - pad)
        x1, y1 = min(cam.W, x + w + pad), min(cam.H, y + h + pad)
        sub = depth[y0:y1, x0:x1].copy()
        sub_mask = mask[y0:y1, x0:x1]
        # восстанавливаем только по точкам самого объекта, а не по полу вокруг
        obj_only = np.where(sub_mask, sub, 0).astype(np.float32)
        filled = cv2.inpaint(obj_only, (obj_only == 0).astype(np.uint8), 7, cv2.INPAINT_TELEA)
        depth = depth.copy()
        depth[y0:y1, x0:x1] = np.where(sub_mask & (sub == 0), filled, sub)
    v, u = np.nonzero(mask & (depth > 0))
    z = depth[v, u] / 1000.0
    p = cam.unproject(u.astype(float), v.astype(float), z)
    floor = Floor(meta["floor_plane"])
    abh = floor.to_floor(p)
    hole = holes[v, u]
    rgb = color[v, u].astype(np.float32)
    # 1) пол внутри маски (щели в кучке, край полигона) не переносим: там будет пол нового кадра
    keep = hole | (abh[:, 2] > 0.003)
    # 2) «летающие» точки на краях: всё, что далеко от тела объекта по полу
    body = keep & (abh[:, 2] > 0.003)
    if body.sum() < 50:
        return None
    med = np.median(abh[body, :2], axis=0)
    mad = np.median(np.abs(abh[body, :2] - med), axis=0) + 1e-3
    keep &= np.all(np.abs(abh[:, :2] - med) < 4 * mad + 0.02, axis=1)
    # 3) провалы ниже пола и выбросы по высоте
    hmax = np.percentile(abh[body, 2], 99.5) * 1.15 + 0.005
    keep &= (abh[:, 2] > -0.01) & (abh[:, 2] < hmax)
    # 4) точки-дыры держим внутри контура тела объекта на полу (+1 см), иначе они тянутся хвостом
    grid = 0.002
    bab = abh[body & keep, :2]
    lo = bab.min(0) - 0.03
    pts = np.round((bab - lo) / grid).astype(np.int32)
    canvas = np.zeros((pts[:, 1].max() + 40, pts[:, 0].max() + 40), np.uint8)
    cv2.fillPoly(canvas, [cv2.convexHull(pts)], 1)
    canvas = cv2.dilate(canvas, np.ones((11, 11), np.uint8))
    q = np.round((abh[:, :2] - lo) / grid).astype(np.int32)
    inside = (q[:, 0] >= 0) & (q[:, 1] >= 0) & (q[:, 0] < canvas.shape[1]) & (q[:, 1] < canvas.shape[0])
    in_hull = np.zeros(len(q), bool)
    in_hull[inside] = canvas[q[inside, 1], q[inside, 0]] > 0
    keep &= ~hole | in_hull
    abh, hole, rgb = abh[keep], hole[keep], rgb[keep]
    abh[:, 2] = np.clip(abh[:, 2], 0, None)
    center = np.median(abh[:, :2], axis=0)
    return {
        "abh": abh - [center[0], center[1], 0],
        "center": center,
        "rgb": rgb,
        "hole": hole,
        "attrs": ann["attributes"],
        "src": frame_dir.name,
    }


# ---------------------------------------------------------------- рендер
def render_object(obj, floor, cam, pos, theta, mirror, scale, color_gain):
    """Проекция облака в целевой кадр. Возвращает слой: глубина мм, цвет, дыра, маска."""
    abh = obj["abh"].copy()
    if mirror:
        abh[:, 0] = -abh[:, 0]
    c, s = math.cos(theta), math.sin(theta)
    a = c * abh[:, 0] - s * abh[:, 1]
    b = s * abh[:, 0] + c * abh[:, 1]
    abh = np.stack([a * scale + pos[0], b * scale + pos[1], abh[:, 2] * scale], axis=1)
    p = floor.to_cam(abh)
    ok = p[:, 2] > 0.05
    p, rgb, hole = p[ok], obj["rgb"][ok], obj["hole"][ok]
    uu, vv, z = cam.project(p)
    # размер «пятна» точки: при увеличении и приближении точки редеют
    footprint = 2 if scale > 0.9 else 1
    us, vs, zs, idx = [], [], [], []
    base_u, base_v = np.round(uu).astype(int), np.round(vv).astype(int)
    for du in range(footprint):
        for dv in range(footprint):
            us.append(base_u + du); vs.append(base_v + dv); zs.append(z); idx.append(np.arange(len(z)))
    u_all, v_all, z_all, i_all = map(np.concatenate, (us, vs, zs, idx))
    inside = (u_all >= 0) & (u_all < cam.W) & (v_all >= 0) & (v_all < cam.H)
    u_all, v_all, z_all, i_all = u_all[inside], v_all[inside], z_all[inside], i_all[inside]
    if len(z_all) == 0:
        return None
    pix = v_all * cam.W + u_all
    order = np.lexsort((z_all, pix))                  # по пикселю, внутри — ближайшая первой
    pix, z_all, i_all = pix[order], z_all[order], i_all[order]
    first = np.r_[True, pix[1:] != pix[:-1]]
    pix, z_all, i_all = pix[first], z_all[first], i_all[first]

    H, W = cam.H, cam.W
    dep = np.zeros(H * W, np.float32); dep[pix] = z_all * 1000
    col = np.zeros((H * W, 3), np.float32); col[pix] = rgb[i_all] * color_gain
    hol = np.zeros(H * W, np.uint8); hol[pix] = hole[i_all]
    msk = np.zeros(H * W, np.uint8); msk[pix] = 1
    dep, col, hol, msk = dep.reshape(H, W), col.reshape(H, W, 3), hol.reshape(H, W), msk.reshape(H, W)

    # закрываем мелкие щели между точками
    closed = cv2.morphologyEx(msk, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    gaps = (closed > 0) & (msk == 0)
    if gaps.any():
        g8 = gaps.astype(np.uint8)
        col = cv2.inpaint(np.clip(col, 0, 255).astype(np.uint8), g8, 3, cv2.INPAINT_TELEA).astype(np.float32)
        dep = cv2.inpaint(dep, g8, 3, cv2.INPAINT_TELEA)
        hol = (cv2.dilate(hol, np.ones((3, 3), np.uint8)) & closed).astype(np.uint8) | (hol & msk)
    return {"depth": dep, "color": col, "hole": hol > 0, "mask": closed > 0}


def composite(bg_color, bg_depth, layers, rng, photometric=True):
    color = bg_color.astype(np.float32).copy()
    depth = bg_depth.astype(np.float32).copy()
    inst = np.zeros(depth.shape, np.uint8)
    # мягкая контактная тень под каждым объектом
    shade = np.zeros(depth.shape, np.float32)
    for L in layers:
        m = L["mask"].astype(np.float32)
        sh = cv2.GaussianBlur(cv2.dilate(m, np.ones((9, 9), np.uint8)), (0, 0), 9)
        shade = np.maximum(shade, sh)
    color *= (1 - 0.22 * shade)[..., None]
    # объекты по z-буферу: сначала дальние
    zbuf = np.full(depth.shape, np.inf, np.float32)
    for k, L in enumerate(layers, start=1):
        d = np.where(L["mask"], np.where(L["depth"] > 0, L["depth"], 1e9), np.inf)
        win = L["mask"] & (d < zbuf)
        zbuf[win] = d[win]
        inst[win] = k
    for k, L in enumerate(layers, start=1):
        win = inst == k
        # мягкий край: смешиваем цвет объекта с полом на 1–2 пикселя по границе
        alpha = cv2.GaussianBlur(win.astype(np.float32), (3, 3), 0)
        alpha = np.where(win, np.maximum(alpha, 0.6), 0)[..., None]
        color = color * (1 - alpha) + L["color"] * alpha
        depth[win] = np.where(L["hole"][win], 0, L["depth"][win])
    # общий свет: яркость и гамма
    gain = rng.uniform(0.85, 1.15) if photometric else 1.0
    gamma = rng.uniform(0.9, 1.1) if photometric else 1.0
    color = 255 * np.power(np.clip(color * gain, 0, 255) / 255, gamma)
    return np.clip(color, 0, 255).astype(np.uint8), np.clip(depth, 0, 65535).astype(np.uint16), inst


def instance_polygons(m):
    cnts, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return []
    big = max(cv2.contourArea(c) for c in cnts)
    return [c.reshape(-1).astype(float).tolist() for c in cnts
            if cv2.contourArea(c) >= max(20, 0.03 * big) and len(c) >= 3]


# ---------------------------------------------------------------- основной цикл
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="dataset")
    ap.add_argument("--out", default="dataset/gen3d")
    ap.add_argument("--n", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-objects", type=int, default=3)
    ap.add_argument("--scale", default="0.75,1.3", help="диапазон масштаба объекта")
    ap.add_argument("--rot-deg", type=float, default=35, help="макс. поворот вокруг вертикали")
    ap.add_argument("--exclude-objects", default="", help="object_id через запятую, которые не брать")
    ap.add_argument("--prefix", default="g")
    ap.add_argument("--no-photometric", action="store_true",
                    help="не менять цвет и яркость (для обучения распознаванию материала по цвету)")
    args = ap.parse_args()

    data, out = Path(args.data), Path(args.out)
    rng = np.random.RandomState(args.seed)
    cam = Cam(json.loads((data / "camera.json").read_text(encoding="utf-8")))
    smin, smax = map(float, args.scale.split(","))
    exclude = {s.strip() for s in args.exclude_objects.split(",") if s.strip()}

    labels = json.loads((data / "labels" / "real_train.json").read_text(encoding="utf-8"))
    anns = defaultdict(list)
    for a in labels["annotations"]:
        anns[a["image_id"]].append(a)
    empties, sources = [], []
    for im in labels["images"]:
        d = data / Path(im["file_name"]).parent
        if not (d / "color.png").exists():
            continue
        if not anns[im["id"]]:
            empties.append(d)
        for a in anns[im["id"]]:
            if a["attributes"]["object_id"] not in exclude:
                sources.append((d, a))
    print(f"пустых полов: {len(empties)}, исходных объектов: {len(sources)}")

    cache = {}

    def get_obj(i):
        if i not in cache:
            cache[i] = extract_object(sources[i][0], sources[i][1], cam)
        return cache[i]

    out.mkdir(parents=True, exist_ok=True)
    coco = {"images": [], "annotations": [], "categories": [{"id": 1, "name": "debris"}]}
    ann_id = 1
    made = 0
    attempts = 0
    while made < args.n and attempts < args.n * 20:
        attempts += 1
        bg_dir = empties[rng.randint(len(empties))]
        bg_meta = json.loads((bg_dir / "meta.json").read_text(encoding="utf-8"))
        floor = Floor(bg_meta["floor_plane"])
        bg_color = cv2.imread(str(bg_dir / "color.png"))
        bg_depth = cv2.imread(str(bg_dir / "depth.png"), cv2.IMREAD_UNCHANGED)

        k = rng.choice(np.arange(1, args.max_objects + 1), p=None)
        layers, placed = [], []
        for _ in range(k):
            for _try in range(15):
                obj = get_obj(rng.randint(len(sources)))
                if obj is None:
                    continue
                scale = rng.uniform(smin, smax)
                theta = math.radians(rng.uniform(-args.rot_deg, args.rot_deg))
                # новое место: рядом с типичной зоной съёмки, с разбросом
                pos = obj["center"] + [rng.uniform(-0.18, 0.18), rng.uniform(-0.12, 0.15)]
                gain3 = np.ones(3) if args.no_photometric else rng.uniform(0.92, 1.08, size=3)
                L = render_object(obj, floor, cam, pos, theta, rng.rand() < 0.5, scale, gain3)
                if L is None or L["mask"].sum() < 1500:
                    continue
                ys, xs = np.nonzero(L["mask"])
                if xs.min() < 3 or ys.min() < 3 or xs.max() > cam.W - 4 or ys.max() > cam.H - 4:
                    continue                                   # объект обрезан краем кадра
                box = (xs.min(), ys.min(), xs.max(), ys.max())
                if any(not (box[2] + 15 < b[0] or b[2] + 15 < box[0] or box[3] + 15 < b[1] or b[3] + 15 < box[1])
                       for b, _, _ in placed):
                    continue                                   # не ставим объекты друг на друга
                layers.append(L)
                placed.append((box, obj, scale))
                break
        if not placed:
            continue

        color, depth, inst = composite(bg_color, bg_depth, layers, rng, not args.no_photometric)
        made += 1
        fid = f"{args.prefix}_{made:04d}"
        fdir = out / fid
        fdir.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(fdir / "color.png"), color)
        cv2.imwrite(str(fdir / "depth.png"), depth)
        cv2.imwrite(str(fdir / "mask.png"), inst)
        meta = dict(bg_meta)
        meta.update(id=fid, background=bg_dir.name, generator="gen3d",
                    holes_share=round(float((depth == 0).mean()), 4))
        (fdir / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
        image_id = made
        coco["images"].append({"id": image_id, "file_name": f"{out.name}/{fid}/color.png",
                               "width": cam.W, "height": cam.H})
        for j, (box, obj, scale) in enumerate(placed, start=1):
            m = inst == j
            if m.sum() < 500:
                continue
            ys, xs = np.nonzero(m)
            at = obj["attrs"]
            s3 = scale ** 3
            vol = float(at["volume_ml"]) * s3
            mass = float(at["mass_g"]) * s3
            coco["annotations"].append({
                "id": ann_id, "image_id": image_id, "category_id": 1, "iscrowd": 0,
                "bbox": [float(xs.min()), float(ys.min()), float(xs.max() - xs.min() + 1), float(ys.max() - ys.min() + 1)],
                "area": float(m.sum()), "segmentation": instance_polygons(m),
                "attributes": {"form": at["form"], "material": at["material"],
                               "volume_ml": round(vol, 1), "mass_g": round(mass, 1),
                               "take": int(mass <= HEAVY_G), "object_id": at["object_id"],
                               "instance": j, "scale": round(scale, 3), "source_frame": obj["src"]}})
            ann_id += 1
        if made % 50 == 0:
            print(f"  {made}/{args.n}", flush=True)

    (data / "labels").mkdir(exist_ok=True)
    lab_path = data / "labels" / f"{out.name}_train.json"
    lab_path.write_text(json.dumps(coco, ensure_ascii=False), encoding="utf-8")
    print(f"готово: {made} кадров, {ann_id - 1} объектов -> {out}, разметка {lab_path}")


if __name__ == "__main__":
    main()
