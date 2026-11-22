"""Дополнительные эмбеддинги вырезок для TTA/аугментации: CLIP-L и SigLIP, контекстные вырезки разных масштабов
(1.2, 1.7, 2.6) и повороты c14 (90/180/270), плюс квадратная вырезка без отражения фона. GPU < 2.5 ГБ (батч 16).
Кэш: feats_tta.npz (ключи <tag>_<view>)."""
import json
import os
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as Fn

HERE = Path(__file__).resolve().parent
D = Path(os.environ.get("ASTRAMIS_DATA", "/home/cleaner/Cleaner/datasets/raw"))
DEV = "cuda"
MODELS = {"clipL": ("ViT-L-14", "dfn2b"), "siglip": ("ViT-B-16-SigLIP", "webli")}


def polygon_mask(seg, h, w):
    m = np.zeros((h, w), np.uint8)
    for s in seg:
        cv2.fillPoly(m, [np.array(s, np.float32).reshape(-1, 2).round().astype(np.int32)], 1)
    return m > 0


def crop(rgb, mask, bbox, scale, masked, size=224):
    x, y, w, h = bbox
    s = max(w, h) * scale + 8
    cx, cy = x + w / 2, y + h / 2
    x1, y1, x2, y2 = int(cx - s / 2), int(cy - s / 2), int(cx + s / 2), int(cy + s / 2)
    pad = max(0, -x1, -y1, x2 - rgb.shape[1], y2 - rgb.shape[0])
    rp = cv2.copyMakeBorder(rgb, pad, pad, pad, pad, cv2.BORDER_REFLECT)
    mp = cv2.copyMakeBorder(mask.astype(np.uint8), pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=0)
    c = rp[y1 + pad:y2 + pad, x1 + pad:x2 + pad].copy()
    m = mp[y1 + pad:y2 + pad, x1 + pad:x2 + pad]
    if masked:
        c[m == 0] = 124
    return cv2.resize(c, (size, size), interpolation=cv2.INTER_AREA)


def records():
    out = []
    for name in ("real_train", "synth_train"):
        d = json.load(open(D / "labels" / f"{name}.json"))
        ims = {im["id"]: im for im in d["images"]}
        for a in d["annotations"]:
            out.append((name, ims[a["image_id"]], a))
    return out


@torch.no_grad()
def embed(model, size, mean, std, imgs, bs=16):
    out = []
    for i in range(0, len(imgs), bs):
        x = torch.from_numpy(imgs[i:i + bs]).to(DEV).permute(0, 3, 1, 2).float() / 255
        x = (Fn.interpolate(x, size=(size, size), mode="bilinear", align_corners=False) - mean) / std
        f = model.encode_image(x).float() + model.encode_image(torch.flip(x, dims=[3])).float()
        out.append(Fn.normalize(f, dim=-1).cpu().numpy())
    return np.concatenate(out)


@torch.no_grad()
def main():
    import open_clip
    recs = records()
    meta = json.load(open(HERE / "shared_cache/meta.json"))
    assert len(recs) == len(meta)
    views = {"c12": [], "c17": [], "c26": [], "c14r90": [], "c14r180": [], "c14r270": [], "m12": []}
    cache = {}
    for i, (src, im, a) in enumerate(recs):
        fd = D / Path(im["file_name"]).parent
        assert fd.name == meta[i]["frame"]
        if fd not in cache:
            cache.clear()
            cache[fd] = cv2.cvtColor(cv2.imread(str(fd / "color.png")), cv2.COLOR_BGR2RGB)
        rgb = cache[fd]
        m = polygon_mask(a["segmentation"], im["height"], im["width"])
        views["c12"].append(crop(rgb, m, a["bbox"], 1.2, False))
        views["c17"].append(crop(rgb, m, a["bbox"], 1.7, False))
        views["c26"].append(crop(rgb, m, a["bbox"], 2.6, False))
        views["m12"].append(crop(rgb, m, a["bbox"], 1.2, True))
        c14 = crop(rgb, m, a["bbox"], 1.5, False)
        views["c14r90"].append(np.ascontiguousarray(np.rot90(c14, 1)))
        views["c14r180"].append(np.ascontiguousarray(np.rot90(c14, 2)))
        views["c14r270"].append(np.ascontiguousarray(np.rot90(c14, 3)))
        if i % 200 == 0:
            print("crops", i, len(recs), flush=True)
    views = {k: np.stack(v) for k, v in views.items()}
    out = {}
    for tag, (name, pre) in MODELS.items():
        model, _, prep = open_clip.create_model_and_transforms(name, pretrained=pre, device=DEV)
        model.eval()
        size = model.visual.image_size[0] if isinstance(model.visual.image_size, (tuple, list)) else model.visual.image_size
        mean = torch.tensor(prep.transforms[-1].mean, device=DEV).view(1, 3, 1, 1)
        std = torch.tensor(prep.transforms[-1].std, device=DEV).view(1, 3, 1, 1)
        for v, arr in views.items():
            out[f"{tag}_{v}"] = embed(model, size, mean, std, arr)
            print("emb", tag, v, torch.cuda.max_memory_allocated() / 1e9, flush=True)
        del model
        torch.cuda.empty_cache()
    np.savez_compressed(HERE / "feats_tta.npz", **out)
    print("saved", {k: v.shape for k, v in out.items()})


if __name__ == "__main__":
    main()
