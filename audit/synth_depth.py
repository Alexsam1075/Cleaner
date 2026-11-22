"""Синтетика по глубине: процедурные предметы недостающих типов с ТОЧНЫМ объёмом и массой,
спроецированные в реальную камеру (интринсики + плоскость пола реальных пустых кадров) с шумом D435
(σ по дальности из реальных кадров: 1.5 мм @<0.8 м, 6 мм @1 м, 13 мм @1.5 м; дыры 13%, квантование 1 мм).
Типы: груды с пустотами (коэффициент заполнения 0.35–0.75), полые/надутые ёмкости, полные/пустые бутылки,
тяжёлые плотные предметы, каркасы, большие ≥1 л, тонкие <50 мл.
Выход: audit/cache/synth_rows.csv — те же геометрические признаки, что у реальных (volume_parts), + volume_ml, mass_g, form, material.
Проверка (--eval): LOOO по 40 реальным предметам, калибратор real vs real+synth; take-порог real vs real+synth."""
import argparse
import json
import math
import sys
from multiprocessing import Pool
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

ROOT = Path("/home/cleaner/Cleaner")
sys.path.insert(0, str(ROOT))
D = ROOT / "datasets/raw"
C = ROOT / "audit/cache"
CAM = json.load(open(D / "camera.json"))
FX, FY, CX, CY = CAM["fx"], CAM["fy"], CAM["cx"], CAM["cy"]
W, H = 1280, 720
L3 = math.log(3)
DENS = {"metal": 2.7, "glass": 2.5, "mineral": 2.4, "wood": 0.6, "plastic": 0.95, "paper": 0.7}


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


# ---------- примитивы: пересечение луча (o + t*dir) в мировой системе (пол z=0)
def hit_sphere(o, dr, c, r):
    oc = o - c
    b = (oc * dr).sum(-1)
    cc = (oc * oc).sum(-1) - r * r
    disc = b * b - cc
    t = -b - np.sqrt(np.clip(disc, 0, None))
    return np.where((disc >= 0) & (t > 0), t, np.inf)


def hit_box(o, dr, lo, hi):
    inv = 1.0 / np.where(np.abs(dr) < 1e-9, 1e-9, dr)
    t0 = (lo - o) * inv
    t1 = (hi - o) * inv
    tmin = np.minimum(t0, t1).max(-1)
    tmax = np.maximum(t0, t1).min(-1)
    return np.where((tmax >= tmin) & (tmax > 0), np.where(tmin > 0, tmin, np.inf), np.inf)


def hit_cyl(o, dr, c, r, h):
    """вертикальный цилиндр: центр основания c (z=0), радиус r, высота h; с крышкой."""
    ox, oy = o[..., 0] - c[0], o[..., 1] - c[1]
    dx, dy = dr[..., 0], dr[..., 1]
    a = dx * dx + dy * dy
    b = ox * dx + oy * dy
    cc = ox * ox + oy * oy - r * r
    disc = b * b - a * cc
    t = (-b - np.sqrt(np.clip(disc, 0, None))) / np.where(a < 1e-12, 1e-12, a)
    z = o[..., 2] + t * dr[..., 2]
    side = np.where((disc >= 0) & (t > 0) & (z >= 0) & (z <= h), t, np.inf)
    tc = (h - o[..., 2]) / np.where(np.abs(dr[..., 2]) < 1e-9, 1e-9, dr[..., 2])
    px, py = o[..., 0] + tc * dr[..., 0] - c[0], o[..., 1] + tc * dr[..., 1] - c[1]
    cap = np.where((tc > 0) & (px * px + py * py <= r * r), tc, np.inf)
    return np.minimum(side, cap)


def hit_hcyl(o, dr, c, r, L, yaw):
    """лежащий цилиндр (бутылка) радиуса r, длины L, ось вдоль yaw, центр оси на высоте r."""
    ax = np.array([math.cos(yaw), math.sin(yaw), 0.0])
    cen = np.array([c[0], c[1], r])
    oc = o - cen
    d_par = (dr * ax).sum(-1)
    oc_par = (oc * ax).sum(-1)
    d_perp = dr - d_par[..., None] * ax
    oc_perp = oc - oc_par[..., None] * ax
    a = (d_perp * d_perp).sum(-1)
    b = (oc_perp * d_perp).sum(-1)
    cc = (oc_perp * oc_perp).sum(-1) - r * r
    disc = b * b - a * cc
    t = (-b - np.sqrt(np.clip(disc, 0, None))) / np.where(a < 1e-12, 1e-12, a)
    s = oc_par + t * d_par
    return np.where((disc >= 0) & (t > 0) & (np.abs(s) <= L / 2), t, np.inf)


# ---------- генераторы предметов: список примитивов + label volume_ml, mass_g, form, material
def gen_object(rng, kind):
    prims = []
    if kind == "pile":
        mat = rng.choice(["metal", "mineral", "plastic", "wood", "paper"])
        R = rng.uniform(0.04, 0.16)
        Hc = rng.uniform(0.25, 0.7) * R
        env = math.pi * R * R * Hc / 3 * 1.6          # объём конусной «оболочки» груды, м³
        fill = rng.uniform(0.45, 1.0)                 # насыпной объём относительно оболочки (как в разметке)
        bulk = env * fill
        n = int(rng.integers(25, 70))
        rp = (bulk * 0.55 / n / (4 / 3 * math.pi)) ** (1 / 3)   # радиус кусочка: материал ≈ 55% насыпного
        vol_mat = 0.0
        for _ in range(n):
            rr = rng.uniform(0.6, 1.4) * rp
            a = rng.uniform(0, 2 * math.pi)
            dist = R * math.sqrt(rng.uniform(0, 1)) * 0.92
            zc = max(rr * 0.7, Hc * (1 - dist / R) * rng.uniform(0.4, 1.0))
            prims.append(("sph", np.array([dist * math.cos(a), dist * math.sin(a), zc]), rr))
            vol_mat += 4 / 3 * math.pi * rr ** 3
        return prims, bulk * 1e6, vol_mat * 1e6 * DENS[mat] * rng.uniform(0.7, 1.0), "bulk", mat
    if kind == "hollow":
        mat = rng.choice(["plastic", "paper", "paper"])
        L, Wd, Hh = rng.uniform(0.1, 0.3), rng.uniform(0.06, 0.2), rng.uniform(0.03, 0.14)
        if rng.random() < 0.5:
            prims.append(("box", np.array([-L / 2, -Wd / 2, 0]), np.array([L / 2, Wd / 2, Hh])))
            vol = L * Wd * Hh
        else:
            prims.append(("sph", np.array([0, 0, Hh / 2]), Hh / 2))
            prims.append(("sph", np.array([L * 0.25, 0, Hh * 0.45]), Hh * 0.45))
            prims.append(("sph", np.array([-L * 0.25, 0, Hh * 0.45]), Hh * 0.45))
            vol = 0.6 * L * Wd * Hh
        return prims, vol * 1e6, vol * 1e6 * rng.uniform(0.01, 0.08), "solid", mat
    if kind == "bottle":
        mat = rng.choice(["plastic", "plastic", "glass"])
        r, L = rng.uniform(0.03, 0.06), rng.uniform(0.15, 0.33)
        yaw = rng.uniform(0, math.pi)
        prims.append(("hcyl", np.array([0, 0]), r, L, yaw))
        vol = math.pi * r * r * L
        fill = rng.choice([0.0, 0.0, 0.5, 1.0, 1.0])
        shell = vol * 1e6 * (0.04 if mat == "plastic" else 0.45)
        return prims, vol * 1e6, shell + vol * 1e6 * fill * rng.uniform(0.9, 1.0), "solid", mat
    if kind == "heavy":
        mat = rng.choice(["metal", "mineral", "glass", "wood"])
        if rng.random() < 0.5:
            L, Wd, Hh = rng.uniform(0.06, 0.2), rng.uniform(0.05, 0.15), rng.uniform(0.03, 0.12)
            prims.append(("box", np.array([-L / 2, -Wd / 2, 0]), np.array([L / 2, Wd / 2, Hh])))
            vol = L * Wd * Hh
        else:
            r = rng.uniform(0.03, 0.09)
            prims.append(("sph", np.array([0, 0, r * 0.85]), r))
            vol = 4 / 3 * math.pi * r ** 3 * 0.9
        return prims, vol * 1e6, vol * 1e6 * DENS[mat] * rng.uniform(0.7, 1.0), "solid", mat
    if kind == "big":
        mat = rng.choice(["plastic", "paper", "plastic"])
        L, Wd, Hh = rng.uniform(0.18, 0.35), rng.uniform(0.12, 0.22), rng.uniform(0.08, 0.2)
        if rng.random() < 0.5:
            prims.append(("box", np.array([-L / 2, -Wd / 2, 0]), np.array([L / 2, Wd / 2, Hh])))
            vol = L * Wd * Hh
        else:
            r = min(Wd, Hh) / 2
            prims.append(("cyl", np.array([0, 0]), r, L))
            vol = math.pi * r * r * L
        return prims, vol * 1e6, vol * 1e6 * rng.uniform(0.03, 0.6), "solid", mat
    if kind == "thin":
        mat = rng.choice(["wood", "paper", "plastic", "mineral"])
        L, Wd, Hh = rng.uniform(0.03, 0.15), rng.uniform(0.02, 0.1), rng.uniform(0.004, 0.02)
        prims.append(("box", np.array([-L / 2, -Wd / 2, 0]), np.array([L / 2, Wd / 2, Hh])))
        vol = L * Wd * Hh * rng.uniform(0.5, 1.0)
        form = "bulk" if rng.random() < 0.4 else "solid"
        return prims, vol * 1e6, vol * 1e6 * DENS[mat], form, mat
    if kind == "frame":
        mat = "metal"
        L, Wd, Hh, bar = rng.uniform(0.15, 0.3), rng.uniform(0.1, 0.2), rng.uniform(0.06, 0.15), rng.uniform(0.012, 0.022)
        edges = [(-L / 2, -Wd / 2, 0, L / 2, -Wd / 2 + bar, bar), (-L / 2, Wd / 2 - bar, 0, L / 2, Wd / 2, bar),
                 (-L / 2, -Wd / 2, 0, -L / 2 + bar, Wd / 2, bar), (L / 2 - bar, -Wd / 2, 0, L / 2, Wd / 2, bar),
                 (-L / 2, -Wd / 2, Hh - bar, L / 2, -Wd / 2 + bar, Hh), (-L / 2, Wd / 2 - bar, Hh - bar, L / 2, Wd / 2, Hh),
                 (-L / 2, -Wd / 2, 0, -L / 2 + bar, -Wd / 2 + bar, Hh), (L / 2 - bar, Wd / 2 - bar, 0, L / 2, Wd / 2, Hh)]
        nbar = 0.0
        for e in edges:
            prims.append(("box", np.array(e[:3]), np.array(e[3:])))
            nbar += (e[3] - e[0]) * (e[4] - e[1]) * (e[5] - e[2])
        return prims, L * Wd * Hh * 1e6 * rng.uniform(0.3, 0.6), nbar * 1e6 * 2.7 * 0.45, "bulk", mat
    raise ValueError(kind)


def render(prims, plane, x0, y0, yaw, bg_depth, rng):
    R, t = floor_frame(plane)
    c, s = math.cos(yaw), math.sin(yaw)
    Rz = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1.0]])
    # bbox объекта в мире -> грубая оценка окна пикселей через проекцию углов
    pts = []
    for p in prims:
        if p[0] == "sph":
            pts += [p[1] + d for d in np.array([[1, 1, 1], [-1, -1, 0], [1, -1, 1], [-1, 1, 0]]) * p[2]]
        elif p[0] == "box":
            pts += [np.array([x, y, z]) for x in (p[1][0], p[2][0]) for y in (p[1][1], p[2][1]) for z in (p[1][2], p[2][2])]
        elif p[0] == "cyl":
            pts += [np.array([p[1][0] + sx * p[2], p[1][1] + sy * p[2], z]) for sx in (-1, 1) for sy in (-1, 1) for z in (0, p[3])]
        elif p[0] == "hcyl":
            pts += [np.array([sx * p[3] / 2, sy * p[2], z]) for sx in (-1, 1) for sy in (-1, 1) for z in (0, 2 * p[2])]
    Pw = (np.array(pts) @ Rz.T) + np.array([x0, y0, 0])
    Pc = (Pw - t) @ R
    uv = Pc[:, :2] / Pc[:, 2:3] * np.array([FX, FY]) + np.array([CX, CY])
    u0, v0 = int(max(0, uv[:, 0].min() - 10)), int(max(0, uv[:, 1].min() - 10))
    u1, v1 = int(min(W, uv[:, 0].max() + 10)), int(min(H, uv[:, 1].max() + 10))
    if u1 - u0 < 4 or v1 - v0 < 4 or (Pc[:, 2] <= 0.05).any():
        return None, None
    vv, uu = np.mgrid[v0:v1, u0:u1].astype(float)
    rc = np.stack([(uu - CX) / FX, (vv - CY) / FY, np.ones_like(uu)], -1)
    rw = rc @ R.T
    # в локальную систему объекта (обратный поворот yaw и сдвиг)
    o_loc = (t - np.array([x0, y0, 0])) @ Rz
    d_loc = rw @ Rz
    o = np.broadcast_to(o_loc, d_loc.shape)
    tbest = np.full(uu.shape, np.inf)
    for p in prims:
        if p[0] == "sph":
            th = hit_sphere(o, d_loc, p[1], p[2])
        elif p[0] == "box":
            th = hit_box(o, d_loc, p[1], p[2])
        elif p[0] == "cyl":
            th = hit_cyl(o, d_loc, p[1], p[2], p[3])
        else:
            th = hit_hcyl(o, d_loc, p[1], p[2], p[3], p[4])
        tbest = np.minimum(tbest, th)
    mask = np.isfinite(tbest)
    if mask.sum() < 60:
        return None, None
    z_obj = tbest * rw[..., 2] * 0 + tbest  # глубина = t * (z-компонента луча в камере) = t (rc_z = 1)
    depth = bg_depth.astype(np.float64).copy()
    win = depth[v0:v1, u0:u1]
    zc = z_obj[mask]
    sd = np.interp(zc, [0.4, 0.8, 1.0, 1.5, 2.0], [0.0015, 0.0017, 0.006, 0.0128, 0.02])
    noisy = (zc + rng.normal(0, 1, zc.shape) * sd) * 1000
    hole = rng.random(zc.shape) < 0.08
    noisy[hole] = 0
    win[mask] = np.round(noisy)
    full_mask = np.zeros((H, W), bool)
    full_mask[v0:v1, u0:u1] = mask
    # дыры на краях маски (как у D435): 2-px кольцо с p=0.5
    edge = cv2.dilate(full_mask.astype(np.uint8), np.ones((5, 5), np.uint8)).astype(bool) & ~cv2.erode(full_mask.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool)
    ring_hole = edge & (rng.random((H, W)) < 0.5)
    depth[ring_hole] = 0
    return depth.astype(np.uint16), full_mask


def job(args):
    k, kind, fid, seed = args
    from evaluation.volume_parts_cv import _row_job  # noqa
    from geometry.volume_parts import estimate_volume, refine_plane
    rng = np.random.default_rng(seed)
    fd = D / "real/train" / fid
    bg = cv2.imread(str(fd / "depth.png"), cv2.IMREAD_UNCHANGED)
    plane = np.asarray(json.load(open(fd / "meta.json"))["floor_plane"], float)
    prims, vol, mass, form, mat = gen_object(rng, kind)
    for _ in range(6):
        x0, y0, yaw = rng.uniform(-0.45, 0.45), rng.uniform(0.35, 1.3), rng.uniform(0, 2 * math.pi)
        depth, mask = render(prims, plane, x0, y0, yaw, bg, rng)
        if depth is not None:
            break
    if depth is None:
        return None
    est = estimate_volume(depth, mask, refine_plane(depth, mask, plane, CAM), CAM, form)
    return {"image_id": f"syn_{k:05d}", "object_id": f"syn_{k:05d}", "kind": kind, "form": form, "material": mat,
            "volume_ml": float(vol), "mass_g": float(mass), "split": "synthdepth", **est}


def generate(n, workers):
    lab = json.load(open(D / "labels/real_train.json"))
    has = {a["image_id"] for a in lab["annotations"]}
    empty = [Path(i["file_name"]).parent.name for i in lab["images"] if i["id"] not in has]
    kinds = ["pile"] * 30 + ["hollow"] * 14 + ["bottle"] * 14 + ["heavy"] * 14 + ["big"] * 8 + ["thin"] * 12 + ["frame"] * 8
    rng = np.random.default_rng(0)
    jobs = [(k, kinds[k % len(kinds)], empty[int(rng.integers(len(empty)))], 1000 + k) for k in range(n)]
    with Pool(workers) as pool:
        rows = [r for r in pool.map(job, jobs, chunksize=4) if r]
    df = pd.DataFrame(rows)
    df.to_csv(C / "synth_rows.csv", index=False)
    print("сгенерировано", len(df), df.kind.value_counts().to_dict())
    print(df.groupby("kind").agg(vol=("volume_ml", "median"), mass=("mass_g", "median"), parts=("parts_ml", "median"), heavy=("mass_g", lambda x: (x > 500).mean())).round(1))
    return df


def evaluate():
    from evaluation.volume_parts_cv import apply_calibrator, fit_calibrator
    real = pd.read_csv(ROOT / "reports/volume_parts_gt_rows.csv")
    real = real[real.split == "real"].reset_index(drop=True)
    syn = pd.read_csv(C / "synth_rows.csv")
    cols = [c for c in real.columns if c in syn.columns]
    syn = syn[cols]
    g = real.volume_ml.values
    ids = real.object_id.values

    def V(p):
        return np.clip(1 - np.abs(np.log(np.clip(p, 1e-3, None) / g)) / L3, 0, 1)

    print("\n== Объём: LOOO по 40 реальным предметам, калибратор на real vs real+synth ==")
    for name, n_syn in (("real only", 0), ("real + synth 100", 100), ("real + synth 300", 300), ("real + synth all", len(syn))):
        p = np.zeros(len(g))
        for o in sorted(set(ids)):
            tr = real[ids != o]
            if n_syn:
                tr = pd.concat([tr, syn.sample(n_syn, random_state=0) if n_syn < len(syn) else syn], ignore_index=True)
            cal = fit_calibrator(tr)
            te = real[ids == o]
            p[ids == o] = [apply_calibrator(cal, r, r["form"], r["material"]) for r in te.to_dict("records")]
        v = V(p)
        b = "  ".join(f"{lo:g}-{hi:g}:{v[(g >= lo) & (g < hi)].mean():.2f}" for lo, hi in ((0, 50), (50, 200), (200, 1000), (1000, 1e9)))
        print(f"  {name:20s} V={v.mean():.3f}  bulk {v[real.form == 'bulk'].mean():.2f} solid {v[real.form == 'solid'].mean():.2f} | {b}", flush=True)

    print("\n== take: порог по калиброванному объёму, подобран на real vs real+synth (LOOO) ==")
    for name, use_syn in (("real only", False), ("real + synth", True)):
        pred = np.zeros(len(g), int)
        for o in sorted(set(ids)):
            tr = real[ids != o]
            cal = fit_calibrator(pd.concat([tr, syn], ignore_index=True) if use_syn else tr)
            pool = pd.concat([tr, syn], ignore_index=True) if use_syn else tr
            cv_ = np.array([apply_calibrator(cal, r, r["form"], r["material"]) for r in pool.to_dict("records")])
            obj = pd.DataFrame({"o": pool.object_id.values, "v": cv_, "t": (pool.mass_g.values <= 500).astype(int)}).groupby("o").agg(v=("v", "median"), t=("t", "first"))
            best = (-1, 500)
            for thr in np.unique(obj.v):
                pp = (obj.v <= thr).astype(int)
                s = np.mean([(pp[obj.t == c] == c).mean() for c in (0, 1) if (obj.t == c).any()])
                if s > best[0]:
                    best = (s, thr)
            te = real[ids == o]
            pv = np.array([apply_calibrator(cal, r, r["form"], r["material"]) for r in te.to_dict("records")])
            pred[ids == o] = (pv <= best[1]).astype(int)
        t = (real.mass_g.values <= 500).astype(int)
        S = np.mean([(pred[t == c] == c).mean() for c in (0, 1)])
        heavy_hit = {o: int(round((pred[(ids == o)] == 0).mean() * 100)) for o in sorted(set(ids[t == 0]))}
        print(f"  {name:14s} S={S:.3f}  тяжёлые пойманы (% кадров): {heavy_hit}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=600)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--eval", action="store_true")
    a = ap.parse_args()
    if not a.eval:
        generate(a.n, a.workers)
    evaluate()
