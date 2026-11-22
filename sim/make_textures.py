"""Текстуры пола для полигона.
tile.png  — плитка из реальных кадров (центральные участки плиток, выпрямленные в вид сверху) + швы.
carpet.png, laminate.png — процедурные поверхности с иным сопротивлением.
Размер плитки оценивается по швам на выпрямленных кадрах."""
import json

import cv2
import numpy as np

from floor_texture import rectify
from geom import ROOT

OUT = ROOT / "sim" / "assets"
PX_PER_M = 500
TEX = 1024


def seam_spacing(fid):
    img, valid = rectify(fid, px_per_m=PX_PER_M)
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    bh = cv2.morphologyEx(g, cv2.MORPH_BLACKHAT, np.ones((9, 9), np.uint8))
    bh[cv2.erode(valid.astype(np.uint8), np.ones((15, 15), np.uint8)) == 0] = 0
    _, b = cv2.threshold(bh, 12, 255, cv2.THRESH_BINARY)
    lines = cv2.HoughLines(b, 1, np.pi / 720, 150)
    if lines is None:
        return None
    lines = lines[:, 0]
    th0 = lines[0, 1]
    rhos = sorted(r for r, t in lines if abs(((t - th0 + np.pi / 2) % np.pi) - np.pi / 2) < np.radians(2))
    groups = []
    for r in rhos:
        if groups and abs(r - groups[-1][-1]) < 20:
            groups[-1].append(r)
        else:
            groups.append([r])
    centers = [np.mean(g) for g in groups]
    diffs = np.diff(centers)
    diffs = diffs[diffs > 100]
    return float(np.min(diffs)) / PX_PER_M if len(diffs) else None


def clean_patch(fid, size=180):
    img, valid = rectify(fid, px_per_m=PX_PER_M)
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)
    best, best_s = None, 1e9
    for y in range(img.shape[0] - 420, img.shape[0] - size, 20):
        for x in range(100, img.shape[1] - size, 40):
            if not valid[y:y + size, x:x + size].all():
                continue
            p = g[y:y + size, x:x + size]
            edges = np.abs(cv2.Laplacian(p, cv2.CV_32F)).mean()
            s = p.std() + 3 * edges - 0.05 * p.mean()
            if s < best_s:
                best, best_s = img[y:y + size, x:x + size], s
    return best


def make_seamless(p, out):
    p = cv2.resize(p, (out, out), interpolation=cv2.INTER_CUBIC).astype(np.float32)
    big = np.tile(p, (2, 2, 1))
    sh = np.roll(np.roll(big, out // 2, 0), out // 2, 1)[:out, :out]
    w = np.minimum(np.arange(out), out - 1 - np.arange(out)) / (out / 2)
    w = np.clip(w * 2, 0, 1)
    m = np.outer(w, w)[..., None]
    return (p * m + sh * (1 - m)).clip(0, 255).astype(np.uint8)


def tile_texture(patches, tile_m, grout_m=0.004):
    base = np.mean([make_seamless(p, TEX).astype(np.float32) for p in patches], 0)
    rng = np.random.default_rng(0)
    noise = cv2.GaussianBlur(rng.normal(0, 1, (TEX, TEX)).astype(np.float32), (0, 0), 25)
    noise = noise / noise.std() * 4
    base = base + noise[..., None]
    gpx = max(2, int(round(grout_m / tile_m * TEX)))
    grout = np.array([95, 100, 105], np.float32)
    base[:gpx // 2 + 1] = grout
    base[-(gpx // 2):] = grout
    base[:, :gpx // 2 + 1] = grout
    base[:, -(gpx // 2):] = grout
    return base.clip(0, 255).astype(np.uint8)


def carpet_texture():
    rng = np.random.default_rng(1)
    n = rng.normal(0, 1, (TEX, TEX)).astype(np.float32)
    fine = cv2.GaussianBlur(n, (0, 0), 1.0)
    coarse = cv2.GaussianBlur(rng.normal(0, 1, (TEX, TEX)).astype(np.float32), (0, 0), 12)
    loops = np.sin(np.arange(TEX) / TEX * 2 * np.pi * 96)[None, :] * np.sin(np.arange(TEX) / TEX * 2 * np.pi * 96)[:, None]
    v = 0.55 * fine / fine.std() + 0.3 * coarse / coarse.std() + 0.25 * loops
    col = np.array([60, 55, 120], np.float32)
    img = col + v[..., None] * np.array([14, 12, 22], np.float32)
    return img.clip(0, 255).astype(np.uint8)


def laminate_texture():
    rng = np.random.default_rng(2)
    img = np.zeros((TEX, TEX, 3), np.float32)
    planks = 4
    ph = TEX // planks
    for i in range(planks):
        grain = cv2.GaussianBlur(rng.normal(0, 1, (ph, TEX)).astype(np.float32), (0, 0), sigmaX=40, sigmaY=1.5)
        tone = rng.uniform(-12, 12)
        img[i * ph:(i + 1) * ph] = np.array([70, 110, 150], np.float32) + tone + grain[..., None] / grain.std() * 10
        img[i * ph:i * ph + 2] = 40
    return img.clip(0, 255).astype(np.uint8)


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    ids = ["r_0077", "r_0081", "r_0090", "r_0001", "r_0270"]
    sp = [s for s in (seam_spacing(f) for f in ids) if s]
    tile_m = float(np.median(sp))
    print("Шаг швов (м):", [round(s, 3) for s in sp], "->", round(tile_m, 3))
    patches = [p for p in (clean_patch(f) for f in ids[:4]) if p is not None]
    cv2.imwrite(str(OUT / "tile.png"), tile_texture(patches, tile_m))
    cv2.imwrite(str(OUT / "carpet.png"), carpet_texture())
    cv2.imwrite(str(OUT / "laminate.png"), laminate_texture())
    (OUT / "surfaces.json").write_text(json.dumps({
        "tile_m": tile_m,
        "surfaces": {
            "tile": {"texture": "tile.png", "size_m": tile_m, "friction": 0.6, "rolling_resistance": 0.010},
            "laminate": {"texture": "laminate.png", "size_m": 0.8, "friction": 0.55, "rolling_resistance": 0.012},
            "carpet": {"texture": "carpet.png", "size_m": 0.5, "friction": 0.9, "rolling_resistance": 0.080},
        }}, indent=2), encoding="utf-8")
    print("Готово:", OUT)
