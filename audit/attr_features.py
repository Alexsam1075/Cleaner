"""Признаки для всех размеченных объектов (real + synth): эмбеддинги нескольких замороженных сетей + геометрия + фото-признаки.
Кэш: audit/cache/features.npz. Запуск на сервере:
    ASTRAMIS_DATA=~/Cleaner/datasets/raw ~/venv/bin/python audit/attr_features.py"""
import json
import os
from pathlib import Path

import cv2
import numpy as np
import timm
import torch

D = Path(os.environ.get("ASTRAMIS_DATA", "dataset"))
OUT = Path(__file__).resolve().parent / "cache"
CAM = json.load(open(D / "camera.json"))
DEV = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
BACKBONES = {
    "dinov2_s": ("vit_small_patch14_dinov2.lvd142m", 224),
    "dinov2_b": ("vit_base_patch14_dinov2.lvd142m", 224),
    "clip_b": ("vit_base_patch16_clip_224.openai", 224),
    "convnext": ("convnext_small.fb_in22k_ft_in1k", 224),
}


def polygon_mask(seg, h, w):
    m = np.zeros((h, w), np.uint8)
    for s in seg:
        cv2.fillPoly(m, [np.array(s, np.float32).reshape(-1, 2).round().astype(np.int32)], 1)
    return m > 0


def crops(rgb, mask, bbox, size=224):
    x, y, w, h = bbox
    s = max(w, h) * 1.15 + 8
    cx, cy = x + w / 2, y + h / 2
    x1, y1, x2, y2 = int(cx - s / 2), int(cy - s / 2), int(cx + s / 2), int(cy + s / 2)
    pad = max(0, -x1, -y1, x2 - rgb.shape[1], y2 - rgb.shape[0])
    rp = cv2.copyMakeBorder(rgb, pad, pad, pad, pad, cv2.BORDER_REFLECT)
    mp = cv2.copyMakeBorder(mask.astype(np.uint8), pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=0)
    c = rp[y1 + pad:y2 + pad, x1 + pad:x2 + pad]
    m = mp[y1 + pad:y2 + pad, x1 + pad:x2 + pad] > 0
    ctx = cv2.resize(c, (size, size), interpolation=cv2.INTER_AREA)
    masked = c.copy()
    masked[~m] = 124
    masked = cv2.resize(masked, (size, size), interpolation=cv2.INTER_AREA)
    return masked, ctx


def geometry(depth, mask, plane, cell=0.004):
    n = np.asarray(plane[:3], float)
    d0 = float(plane[3])
    mm = mask.sum()
    v, u = np.nonzero(mask & (depth > 0))
    out = {"valid": len(v) / max(mm, 1), "area_px": float(mm)}
    er = cv2.erode(mask.astype(np.uint8), np.ones((7, 7), np.uint8)) > 0
    out["holes_inner"] = float(((depth == 0) & er).sum() / max(er.sum(), 1))
    out["holes_edge"] = float(((depth == 0) & mask & ~er).sum() / max((mask & ~er).sum(), 1))
    if len(v) < 30:
        out.update({k: 0.0 for k in ["vol", "vol_thr", "h_med", "h95", "hmax", "foot", "ext1", "ext2", "rough", "zmed", "surf"]})
        return out
    z = depth[v, u] / 1000.0
    P = np.stack([(u - CAM["cx"]) / CAM["fx"] * z, (v - CAM["cy"]) / CAM["fy"] * z, z], 1)
    h = P @ n + d0
    if np.median(h) < 0:
        h = -h
    e1 = np.cross(n, [1.0, 0, 0])
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(n, e1)
    a, b = P @ e1, P @ e2
    k = np.floor(a / cell).astype(np.int64) * 100000 + np.floor(b / cell).astype(np.int64)
    o = np.argsort(k)
    ks, hs = k[o], h[o]
    st = np.r_[0, np.nonzero(np.diff(ks))[0] + 1]
    tops = np.maximum.reduceat(hs, st)
    out["vol"] = float(np.clip(tops, 0, None).sum() * cell * cell * 1e6)
    out["vol_thr"] = float(np.clip(tops - 0.003, 0, None).sum() * cell * cell * 1e6)
    out["foot"] = float(len(tops) * cell * cell * 1e4)
    out["h_med"], out["h95"], out["hmax"] = [float(x) for x in np.percentile(h, [50, 95, 99.5])]
    out["ext1"] = float(np.percentile(a, 97) - np.percentile(a, 3))
    out["ext2"] = float(np.percentile(b, 97) - np.percentile(b, 3))
    if out["ext1"] < out["ext2"]:
        out["ext1"], out["ext2"] = out["ext2"], out["ext1"]
    out["rough"] = float(np.std(h[h > 0.003])) if (h > 0.003).sum() > 5 else 0.0
    out["zmed"] = float(np.median(z))
    px_m = out["zmed"] / CAM["fx"]
    out["surf"] = float(mm * px_m * px_m * 1e4)
    return out


def photo(rgb, mask):
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    H, S, V = [hsv[..., i][mask].astype(np.float32) for i in range(3)]
    g = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)
    lap = np.abs(cv2.Laplacian(g, cv2.CV_32F))[mask]
    hh = np.histogram(H, bins=8, range=(0, 180), weights=S / 255.0)[0]
    hh = hh / max(hh.sum(), 1e-6)
    ring = cv2.dilate(mask.astype(np.uint8), np.ones((25, 25), np.uint8)) > 0
    ring &= ~mask
    bg = g[ring].mean() if ring.any() else g.mean()
    return np.r_[[(V > 240).mean(), ((S < 40) & (V > 200)).mean(), S.mean() / 255, V.mean() / 255, V.std() / 255,
                  np.percentile(V, 99) / 255, lap.mean() / 50, np.percentile(lap, 95) / 100, (g[mask].mean() - bg) / 255], hh].astype(np.float32)


def records():
    out = []
    for name in ("real_train", "synth_train"):
        d = json.load(open(D / "labels" / f"{name}.json"))
        ims = {im["id"]: im for im in d["images"]}
        for a in d["annotations"]:
            out.append((name, ims[a["image_id"]], a))
    return out


@torch.no_grad()
def main():
    OUT.mkdir(parents=True, exist_ok=True)
    recs = records()
    meta, geo, pho, masked, ctx = [], [], [], [], []
    cache = {}
    for i, (src, im, a) in enumerate(recs):
        fd = D / Path(im["file_name"]).parent
        if fd not in cache:
            cache.clear()
            rgb = cv2.cvtColor(cv2.imread(str(fd / "color.png")), cv2.COLOR_BGR2RGB)
            depth = cv2.imread(str(fd / "depth.png"), cv2.IMREAD_UNCHANGED)
            plane = json.load(open(fd / "meta.json"))["floor_plane"]
            cache[fd] = (rgb, depth, plane)
        rgb, depth, plane = cache[fd]
        m = polygon_mask(a["segmentation"], im["height"], im["width"])
        at = a["attributes"]
        meta.append({"src": src, "frame": fd.name, "oid": at["object_id"], "form": at["form"], "mat": at["material"],
                     "vol": at["volume_ml"], "mass": at["mass_g"], "take": int(at["mass_g"] <= 500), "bbox": a["bbox"],
                     "lighting": json.load(open(fd / "meta.json")).get("lighting")})
        geo.append(geometry(depth, m, plane))
        pho.append(photo(rgb, m))
        c1, c2 = crops(rgb, m, a["bbox"])
        masked.append(c1)
        ctx.append(c2)
        if i % 200 == 0:
            print("crops", i, len(recs), flush=True)
    masked, ctx = np.stack(masked), np.stack(ctx)
    emb = {}
    for key, (name, size) in BACKBONES.items():
        model = timm.create_model(name, pretrained=True, num_classes=0, img_size=size) if "dinov2" in name else \
            timm.create_model(name, pretrained=True, num_classes=0)
        model = model.eval().to(DEV)
        cfg = timm.data.resolve_data_config({}, model=model)
        mean = torch.tensor(cfg["mean"], device=DEV).view(1, 3, 1, 1)
        std = torch.tensor(cfg["std"], device=DEV).view(1, 3, 1, 1)
        for tag, arr in (("m", masked), ("c", ctx)):
            feats = []
            for j in range(0, len(arr), 64):
                x = torch.from_numpy(arr[j:j + 64]).to(DEV).permute(0, 3, 1, 2).float() / 255
                x = (x - mean) / std
                f = model(x)
                f2 = model(torch.flip(x, dims=[3]))
                feats.append(((f + f2) / 2).float().cpu().numpy())
            emb[f"{key}_{tag}"] = np.concatenate(feats)
        print("emb", key, emb[f"{key}_m"].shape, flush=True)
        del model
        torch.cuda.empty_cache() if DEV == "cuda" else None
    gk = sorted(geo[0])
    np.savez_compressed(OUT / "features.npz", geo=np.array([[g[k] for k in gk] for g in geo], np.float32), geo_keys=np.array(gk),
                        photo=np.stack(pho), **emb)
    json.dump(meta, open(OUT / "meta.json", "w"))
    print("saved", len(meta))


if __name__ == "__main__":
    main()
