"""Объём без нейросетей: только геометрия + якоря.
Якоря масштаба:
  1) плоскость пола + интринсики камеры: любой пиксель на полу -> метры (пересечение луча с полом);
  2) плитка 0.43 м: проверка масштаба (шаг швов на выпрямленном полу должен быть 0.43 м);
  3) точка касания пола: нижняя кромка силуэта лежит на полу -> её 3D-положение известно без глубины,
     а высота верхней кромки = пересечение луча верхнего пикселя с вертикалью над точкой касания (работает и при дырах глубины).
Измерения по каждому объекту (GT-маска, real_train):
  L, W — габариты следа на полу (по глубине и по силуэту), H — высота (p95 по глубине / по силуэту),
  columns — интеграл карты высот (клетки 4 мм), gap — доля следа, где сквозь объект виден пол (пустоты груды),
  fill — columns / (площадь следа × H): насколько объём заполняет свой габарит.
Оценки объёма (все параметры — 1–2 глобальные константы на форму, подбираются на остальных объектах, LOOO):
  E1 columns × k;  E2 L×W×H × k;  E3 площадь следа × H × k;  E4 columns × (1−gap)^a × k;  E5 медиана E1/E3.
Только real_train, real/test не используется."""
import json
import math
import os
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

D = Path(os.environ.get("ASTRAMIS_DATA", "/home/cleaner/Cleaner/datasets/raw"))
CAM = json.load(open(D / "camera.json"))
K = np.array([[CAM["fx"], 0, CAM["cx"]], [0, CAM["fy"], CAM["cy"]], [0, 0, 1.0]])
OUT = Path(__file__).resolve().parent / "cache"


def floor_frame(plane):
    n = np.asarray(plane[:3], float)
    d = float(plane[3])
    s = np.linalg.norm(n)
    n, d = n / s, d / s
    if d < 0:
        n, d = -n, -d
    fwd = np.array([0, 0, 1.0]) - n * n[2]
    fwd /= np.linalg.norm(fwd)
    right = np.cross(fwd, n)
    return np.stack([right, fwd, n]), np.array([0, 0, d])


def ray_floor(u, v, plane):
    R, t = floor_frame(plane)
    rc = np.stack([(u - K[0, 2]) / K[0, 0], (v - K[1, 2]) / K[1, 1], np.ones_like(u, float)], -1)
    rw = rc @ R.T
    s = -t[2] / rw[..., 2]
    return t + rw * s[..., None]


def mask_of(a, h, w):
    m = np.zeros((h, w), np.uint8)
    for s in a["segmentation"]:
        cv2.fillPoly(m, [np.array(s, np.float32).reshape(-1, 2).round().astype(np.int32)], 1)
    return m > 0


def measure(depth, mask, plane, cell=0.004):
    R, t = floor_frame(plane)
    v, u = np.nonzero(mask)
    out = {}
    # --- силуэт (без глубины): точка касания = нижние пиксели каждого столбца маски
    cols = defaultdict(list)
    for x, y in zip(u, v):
        cols[x].append(y)
    bottom = np.array([(x, max(ys)) for x, ys in cols.items()], float)
    top = np.array([(x, min(ys)) for x, ys in cols.items()], float)
    fb = ray_floor(bottom[:, 0], bottom[:, 1], plane)
    out["sil_W"] = float(np.ptp(fb[:, 0]))
    # высота по силуэту: вертикаль над точкой касания, пересечение с лучом верхнего пикселя
    Rc = R.T
    hs = []
    for (xb, yb), (xt, yt), p in zip(bottom, top, fb):
        c = -Rc @ t
        rc = np.array([(xt - K[0, 2]) / K[0, 0], (yt - K[1, 2]) / K[1, 1], 1.0])
        rw = R @ rc
        cam_w = t
        # точка на луче с той же горизонтальной проекцией, что и p (ближайшая по XY)
        dxy = rw[:2]
        s = float(np.dot(p[:2] - cam_w[:2], dxy) / max(np.dot(dxy, dxy), 1e-9))
        hs.append(cam_w[2] + s * rw[2])
    hs = np.clip(np.array(hs), 0, 1)
    out["sil_H"] = float(np.percentile(hs, 90))
    # --- глубина
    ok = mask & (depth > 0)
    out["valid"] = float(ok.sum() / max(mask.sum(), 1))
    if ok.sum() < 30:
        for k in ("L", "W", "H", "colvol", "foot", "gap", "fill"):
            out[k] = np.nan
        return out
    vv, uu = np.nonzero(ok)
    z = depth[vv, uu] / 1000.0
    P = np.stack([(uu - K[0, 2]) / K[0, 0] * z, (vv - K[1, 2]) / K[1, 1] * z, z], 1)
    W_ = P @ R.T + t
    h = W_[:, 2]
    gap = h < 0.004
    out["gap"] = float(gap.mean())
    obj = W_[~gap] if (~gap).sum() > 20 else W_
    # главные оси следа
    xy = obj[:, :2] - obj[:, :2].mean(0)
    ev, evec = np.linalg.eigh(np.cov(xy.T))
    pr = xy @ evec
    ext = np.percentile(pr, 97, 0) - np.percentile(pr, 3, 0)
    out["L"], out["W"] = float(max(ext)), float(min(ext))
    out["H"] = float(np.percentile(obj[:, 2], 95))
    k = np.floor(W_[:, 0] / cell).astype(np.int64) * 100000 + np.floor(W_[:, 1] / cell).astype(np.int64)
    o = np.argsort(k)
    ks, hh = k[o], np.clip(h[o], 0, None)
    st = np.r_[0, np.nonzero(np.diff(ks))[0] + 1]
    tops = np.maximum.reduceat(hh, st)
    out["colvol"] = float(tops.sum() * cell * cell * 1e6)
    out["foot"] = float((tops > 0.004).sum() * cell * cell)
    out["fill"] = float(out["colvol"] / max(out["foot"] * out["H"] * 1e6, 1e-6))
    return out


def tile_check(n=12):
    """Якорь 2: шаг швов плитки на выпрямленном полу (должен быть 0.43 м)."""
    sys_path = Path("/home/cleaner/Cleaner/sim")
    import sys
    sys.path.insert(0, str(sys_path))
    try:
        from make_textures import seam_spacing
    except Exception as e:
        return f"нет sim/make_textures: {e}"
    lab = json.load(open(D / "labels/real_train.json"))
    ids = [Path(i["file_name"]).parent.name for i in lab["images"]][::max(1, len(lab["images"]) // n)][:n]
    os.environ["ASTRAMIS_DATA"] = str(D)
    vals = [s for s in (seam_spacing(f) for f in ids) if s]
    return {"frames": len(vals), "median_m": float(np.median(vals)), "spread": [float(np.percentile(vals, 10)), float(np.percentile(vals, 90))]}


def V(p, g):
    return np.clip(1 - np.abs(np.log(np.clip(p, 1e-3, None) / g)) / math.log(3), 0, 1)


def main():
    lab = json.load(open(D / "labels/real_train.json"))
    ims = {i["id"]: i for i in lab["images"]}
    rows = []
    cache = {}
    for a in lab["annotations"]:
        im = ims[a["image_id"]]
        fd = D / Path(im["file_name"]).parent
        if fd not in cache:
            cache.clear()
            cache[fd] = (cv2.imread(str(fd / "depth.png"), cv2.IMREAD_UNCHANGED), json.load(open(fd / "meta.json"))["floor_plane"])
        depth, plane = cache[fd]
        t = a["attributes"]
        r = measure(depth, mask_of(a, im["height"], im["width"]), plane)
        r.update(oid=t["object_id"], form=t["form"], mat=t["material"], gtv=t["volume_ml"], mass=t["mass_g"], frame=fd.name)
        rows.append(r)
    d = pd.DataFrame(rows)
    d.to_csv(OUT / "algo_volume_rows.csv", index=False)

    # Оценки
    H = d["H"].fillna(d["sil_H"])
    d["E1_columns"] = d["colvol"]
    d["E2_LWH"] = d["L"] * d["W"] * H * 1e6
    d["E3_footH"] = d["foot"] * H * 1e6
    d["E5_geo"] = np.sqrt(d["E1_columns"] * d["E3_footH"])
    objs = sorted(d.oid.unique())
    res = {}
    for est in ("E1_columns", "E2_LWH", "E3_footH", "E5_geo"):
        for per_form in (False, True):
            pred = np.full(len(d), np.nan)
            for o in objs:
                tr = (d.oid != o) & d[est].notna() & (d[est] > 0)
                te = d.oid == o
                for f in (("bulk", "solid") if per_form else (None,)):
                    trf = tr & ((d.form == f) if f else True)
                    tef = te & ((d.form == f) if f else True)
                    if tef.sum() == 0:
                        continue
                    # одна константа: k = медиана log(gt/est) по объектам (каждый объект — один голос)
                    lr = (np.log(d.loc[trf, "gtv"] / d.loc[trf, est])).groupby(d.loc[trf, "oid"]).median()
                    k = math.exp(float(np.median(lr)))
                    pred[tef.values] = d.loc[tef, est].values * k
            pred = np.where(np.isnan(pred), np.nanmedian(d["gtv"]), pred)
            res[f"{est}{' (по форме)' if per_form else ''}"] = pred
    # E4: columns × (1−gap)^a × k (a подбирается на train-объектах, по форме)
    pred = np.full(len(d), np.nan)
    for o in objs:
        tr = (d.oid != o) & d["colvol"].notna()
        te = d.oid == o
        for f in ("bulk", "solid"):
            trf, tef = tr & (d.form == f), te & (d.form == f)
            if tef.sum() == 0:
                continue
            best = None
            for a_ in np.linspace(-3, 3, 25):
                e = d.loc[trf, "colvol"] * (1 - d.loc[trf, "gap"]).clip(0.05) ** a_
                k = math.exp(float(np.median(np.log(d.loc[trf, "gtv"] / e).groupby(d.loc[trf, "oid"]).median())))
                s = V(e * k, d.loc[trf, "gtv"]).mean()
                if best is None or s > best[0]:
                    best = (s, a_, k)
            _, a_, k = best
            pred[tef.values] = d.loc[tef, "colvol"].values * (1 - d.loc[tef, "gap"].values).clip(0.05) ** a_ * k
    res["E4_columns×(1−gap)^a (по форме)"] = np.where(np.isnan(pred), np.nanmedian(d["gtv"]), pred)

    print("== Якорь масштаба: шаг плитки на выпрямленном полу (ожидаем 0.43 м):", tile_check())
    print("\n== Объём без нейросетей, LOOO по 40 предметам (GT-маски, истинная форма) ==")
    bins = [(0, 50), (50, 200), (200, 1000), (1000, 1e9)]
    for k, p in res.items():
        v = V(p, d["gtv"].values)
        b = "  ".join(f"{lo:g}-{hi:g}:{v[(d["gtv"] >= lo) & (d["gtv"] < hi)].mean():.2f}" for lo, hi in bins)
        vb = v[d.form == "bulk"].mean()
        vs = v[d.form == "solid"].mean()
        print(f"  {k:36s} V={v.mean():.3f}  bulk {vb:.2f} solid {vs:.2f} | {b}")
    print("\n  для сравнения: калибратор volume/ (ridge 38 признаков + класс) LOOO ≈ 0.764; агент IRLS ≈ 0.781")

    print("\n== Пустоты груд: виден ли пол сквозь объект (gap) и заполнение габарита (fill), медианы по объекту ==")
    o = d.groupby("oid").agg(form=("form", "first"), mat=("mat", "first"), gtv=("gtv", "first"), colvol=("colvol", "median"),
                             LWH=("E2_LWH", "median"), gap=("gap", "median"), fill=("fill", "median"), H_depth=("H", "median"), H_sil=("sil_H", "median"))
    o["gt/columns"] = o["gtv"] / o["colvol"]
    print(o.sort_values(["form", "gt/columns"]).round(3).to_string())
    print("\n  корреляция log(gt/columns) с gap:", round(float(np.corrcoef(np.log(o["gt/columns"]), o.gap)[0, 1]), 3),
          " с fill:", round(float(np.corrcoef(np.log(o["gt/columns"]), o.fill)[0, 1]), 3))


if __name__ == "__main__":
    main()
