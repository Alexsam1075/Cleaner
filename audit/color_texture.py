"""Идея: материал определять только по цвету и текстуре пикселей объекта, без формы/силуэта/фона.
A) Ручные признаки (только пиксели внутри эрозированной маски):
   цвет — гистограммы HSV и Lab, моменты цвета, доля бликов/пересвета, хроматичность;
   текстура — LBP (uniform, 2 радиуса), энергии Габор-фильтров (4 масштаба × 6 ориентаций), GLCM-подобная статистика
   градиентов, локальный контраст, спектральная энергия.
B) Текстурные патчи: квадраты целиком внутри маски (форма объекта не видна), до 8 штук на объект,
   эмбеддинги DINOv2-B и CLIP-L (+ zero-shot по материалам), усреднение по патчам.
Оценка — тот же честный leave-one-object-out (mat_cv.looo). Только real_train + synth_train."""
import json
import math
import os
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as Fn

from attr_features import polygon_mask, records

D = Path(os.environ.get("ASTRAMIS_DATA", "dataset"))
C = Path(__file__).resolve().parent / "cache"
DEV = "cuda" if torch.cuda.is_available() else "cpu"
GABOR = [cv2.getGaborKernel((31, 31), s, t, l, 0.5, 0, ktype=cv2.CV_32F)
         for s, l in ((2, 5), (3, 8), (5, 12), (8, 20)) for t in np.arange(6) * np.pi / 6]


def lbp(g, r):
    h, w = g.shape
    out = np.zeros((h, w), np.uint8)
    c = g[r:h - r, r:w - r]
    code = np.zeros_like(c, np.int32)
    for k in range(8):
        a = 2 * np.pi * k / 8
        dy, dx = int(round(-r * np.sin(a))), int(round(r * np.cos(a)))
        n = g[r + dy:h - r + dy, r + dx:w - r + dx]
        code |= (n >= c).astype(np.int32) << k
    trans = np.array([bin((v ^ ((v << 1 | v >> 7) & 255))).count("1") for v in range(256)])
    ones = np.array([bin(v).count("1") for v in range(256)])
    lut = np.where(trans <= 2, ones, 9).astype(np.uint8)
    out[r:h - r, r:w - r] = lut[code]
    return out


def handcrafted(rgb, mask):
    inner = cv2.erode(mask.astype(np.uint8), np.ones((7, 7), np.uint8)) > 0
    if inner.sum() < 50:
        inner = mask
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
    g = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    gf = g.astype(np.float32) / 255
    H, S, V = [hsv[..., i][inner].astype(np.float32) for i in range(3)]
    L, A, B = [lab[..., i][inner] for i in range(3)]
    f = []
    f += list(np.histogram(H, 12, (0, 180), weights=S / 255)[0] / max((S / 255).sum(), 1e-6))
    f += list(np.histogram(S, 8, (0, 256))[0] / len(S))
    f += list(np.histogram(V, 8, (0, 256))[0] / len(V))
    f += list(np.histogram(A, 8, (88, 168))[0] / len(A)) + list(np.histogram(B, 8, (88, 168))[0] / len(B))
    for ch in (L, A, B, S, V):
        m, s = ch.mean(), ch.std() + 1e-6
        f += [m / 255, s / 255, (((ch - m) / s) ** 3).mean() / 10]
    f += [(V > 245).mean(), ((V > 220) & (S < 30)).mean(), (V < 40).mean(), np.percentile(V, 99) / 255 - np.median(V) / 255,
          np.hypot(A - 128, B - 128).mean() / 50]
    for r in (1, 3):
        code = lbp(g, r)[inner]
        f += list(np.bincount(code, minlength=10)[:10] / len(code))
    for k in GABOR:
        resp = np.abs(cv2.filter2D(gf, cv2.CV_32F, k))[inner]
        f.append(resp.mean())
    gx, gy = cv2.Sobel(gf, cv2.CV_32F, 1, 0), cv2.Sobel(gf, cv2.CV_32F, 0, 1)
    mag, ang = np.hypot(gx, gy)[inner], np.arctan2(gy, gx)[inner]
    f += list(np.histogram(ang, 8, (-np.pi, np.pi), weights=mag)[0] / max(mag.sum(), 1e-6))
    f += [mag.mean(), np.percentile(mag, 90), (mag > 0.3).mean()]
    loc = cv2.blur(gf, (9, 9))
    lc = np.sqrt(np.abs(cv2.blur(gf * gf, (9, 9)) - loc * loc))[inner]
    f += [lc.mean(), lc.std(), np.percentile(lc, 90)]
    lap = np.abs(cv2.Laplacian(gf, cv2.CV_32F))[inner]
    f += [lap.mean(), np.percentile(lap, 95)]
    return np.nan_to_num(np.array(f, np.float32))


def inner_patches(rgb, mask, n=8, size=224, rng=None):
    """Квадратные патчи целиком внутри маски; если объект мал — самый большой вписанный квадрат."""
    rng = rng or np.random.default_rng(0)
    m8 = mask.astype(np.uint8)
    dist = cv2.distanceTransform(m8, cv2.DIST_L2, 5)
    r_max = dist.max()
    if r_max < 4:
        ys, xs = np.nonzero(mask)
        cy, cx, half = int(ys.mean()), int(xs.mean()), 6
        out = [rgb[max(0, cy - half):cy + half, max(0, cx - half):cx + half]]
    else:
        half = int(max(4, min(r_max * 0.7, 48)))
        ys, xs = np.nonzero(dist >= half * 1.0)
        if len(ys) == 0:
            ys, xs = np.nonzero(dist >= r_max * 0.9)
            half = int(r_max * 0.7)
        pick = rng.choice(len(ys), min(n, len(ys)), replace=False)
        out = [rgb[ys[i] - half:ys[i] + half, xs[i] - half:xs[i] + half] for i in pick]
    out = [p for p in out if p.size > 0]
    return np.stack([cv2.resize(p, (size, size), interpolation=cv2.INTER_CUBIC) for p in out]), half


@torch.no_grad()
def deep_patch_features(all_patches, owners, n_items):
    import open_clip
    import timm
    from mat_features2 import MATS, PROMPTS, TEMPLATES
    res = {}
    dino = timm.create_model("vit_base_patch14_dinov2.lvd142m", pretrained=True, num_classes=0, img_size=224).eval().to(DEV)
    cfg = timm.data.resolve_data_config({}, model=dino)
    mean, std = [torch.tensor(cfg[k], device=DEV).view(1, 3, 1, 1) for k in ("mean", "std")]
    feats = []
    for i in range(0, len(all_patches), 64):
        x = (torch.from_numpy(all_patches[i:i + 64]).to(DEV).permute(0, 3, 1, 2).float() / 255 - mean) / std
        feats.append(dino(x).float().cpu())
    E = torch.cat(feats).numpy()
    res["tex_dinoB"] = np.stack([E[owners == k].mean(0) for k in range(n_items)])
    del dino
    model, _, pre = open_clip.create_model_and_transforms("ViT-L-14", pretrained="dfn2b", device=DEV)
    tok = open_clip.get_tokenizer("ViT-L-14")
    mean = torch.tensor(pre.transforms[-1].mean, device=DEV).view(1, 3, 1, 1)
    std = torch.tensor(pre.transforms[-1].std, device=DEV).view(1, 3, 1, 1)
    T = []
    tex_templates = ["a close-up photo of the texture of {}.", "a macro photo of {} surface.", "the surface of {}.", "a texture of {}."]
    for m in MATS:
        txt = [t.format(p) for p in PROMPTS[m] for t in TEMPLATES + tex_templates]
        e = Fn.normalize(model.encode_text(tok(txt).to(DEV)).float(), dim=-1).mean(0)
        T.append(Fn.normalize(e, dim=0))
    T = torch.stack(T)
    feats = []
    for i in range(0, len(all_patches), 64):
        x = (torch.from_numpy(all_patches[i:i + 64]).to(DEV).permute(0, 3, 1, 2).float() / 255 - mean) / std
        feats.append(Fn.normalize(model.encode_image(x).float(), dim=-1).cpu())
    E = torch.cat(feats)
    res["tex_clipL"] = np.stack([E[torch.from_numpy(owners == k)].mean(0).numpy() for k in range(n_items)])
    zs = (E.to(DEV) @ T.T).cpu().numpy()
    res["tex_clipL_zs"] = np.stack([zs[owners == k].mean(0) for k in range(n_items)])
    return res


def main():
    recs = records()
    hand, patches, owners, halves = [], [], [], []
    cache = {}
    rng = np.random.default_rng(0)
    for k, (src, im, a) in enumerate(recs):
        fd = D / Path(im["file_name"]).parent
        if fd not in cache:
            cache.clear()
            cache[fd] = cv2.cvtColor(cv2.imread(str(fd / "color.png")), cv2.COLOR_BGR2RGB)
        rgb = cache[fd]
        m = polygon_mask(a["segmentation"], im["height"], im["width"])
        hand.append(handcrafted(rgb, m))
        p, half = inner_patches(rgb, m, rng=rng)
        patches.append(p)
        owners += [k] * len(p)
        halves.append(half)
        if k % 200 == 0:
            print("items", k, len(recs), flush=True)
    out = {"hand": np.stack(hand), "patch_half_px": np.array(halves, np.float32)}
    out.update(deep_patch_features(np.concatenate(patches), np.array(owners), len(recs)))
    np.savez_compressed(C / "features_ct.npz", **out)
    print("saved", {k: v.shape for k, v in out.items()}, "медиана полуразмера патча, px:", float(np.median(halves)))
    cols = []
    for i in range(0, len(patches), 75):
        col = list(patches[i][:4]) + [np.zeros((224, 224, 3), np.uint8)] * (4 - min(4, len(patches[i])))
        cols.append(np.concatenate(col, 0))
    cv2.imwrite(str(C / "patch_examples.jpg"), cv2.cvtColor(np.concatenate(cols[:12], 1), cv2.COLOR_RGB2BGR))


if __name__ == "__main__":
    main()
