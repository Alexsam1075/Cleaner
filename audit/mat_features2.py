"""Признаки v2 для материала. Идея: у металла и стекла всего 3–5 физических предметов, поэтому одних примеров мало —
нужны модели, которые уже знают, как выглядит «металл» (языковое знание CLIP/SigLIP), и признаки, не привязанные к предмету.
Для каждого объекта:
  * CLIP ViT-L/14 (DFN-2B), SigLIP ViT-B/16, DINOv2-L — эмбеддинги вырезки (маска, контекст, 2 масштаба, flip-TTA);
  * DINOv2-L: патч-токены, усреднённые по маске объекта (текстура именно объекта, без формы силуэта);
  * zero-shot сходства CLIP/SigLIP с ансамблем текстовых описаний каждого материала.
Только real_train + synth_train. Кэш: audit/cache/features2.npz"""
import json
import os
from pathlib import Path

import cv2
import numpy as np
import open_clip
import timm
import torch
import torch.nn.functional as Fn

from attr_features import polygon_mask, records

D = Path(os.environ.get("ASTRAMIS_DATA", "dataset"))
OUT = Path(__file__).resolve().parent / "cache"
DEV = "cuda" if torch.cuda.is_available() else "cpu"
MATS = ["metal", "wood", "plastic", "glass", "paper", "mineral"]
PROMPTS = {
    "metal": ["metal", "shiny metal", "aluminium", "steel", "metal screws and bolts", "an aluminium profile", "a tin can",
              "a metal bar", "scrap metal", "rusty iron"],
    "wood": ["wood", "a piece of wood", "wooden stick", "wood chips", "a wooden block", "sawdust", "a twig", "plywood"],
    "plastic": ["plastic", "a plastic bag", "a plastic bottle", "plastic packaging", "plastic wrapper", "a plastic container",
                "crumpled plastic film", "a plastic cup", "styrofoam"],
    "glass": ["glass", "a glass bottle", "a glass jar", "broken glass", "transparent glass", "a drinking glass"],
    "paper": ["paper", "cardboard", "a crumpled paper", "a paper bag", "a cardboard box", "newspaper", "a paper cup", "a tissue"],
    "mineral": ["stone", "a rock", "gravel", "pebbles", "concrete debris", "a brick", "sand", "crushed stone"],
}
TEMPLATES = ["a photo of {} on the floor.", "a close-up photo of {}.", "{} lying on a tiled floor.", "trash made of {}.", "a photo of an object made of {}."]


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
    return cv2.resize(c, (size, size), interpolation=cv2.INTER_AREA), cv2.resize(m, (size, size), interpolation=cv2.INTER_NEAREST)


def batches(arr, bs=48):
    for i in range(0, len(arr), bs):
        yield i, arr[i:i + bs]


@torch.no_grad()
def clip_feats(name, pretrained, views, out, tag):
    model, _, pre = open_clip.create_model_and_transforms(name, pretrained=pretrained, device=DEV)
    model.eval()
    tok = open_clip.get_tokenizer(name)
    mean = torch.tensor(pre.transforms[-1].mean, device=DEV).view(1, 3, 1, 1)
    std = torch.tensor(pre.transforms[-1].std, device=DEV).view(1, 3, 1, 1)
    size = model.visual.image_size[0] if isinstance(model.visual.image_size, (tuple, list)) else model.visual.image_size
    T = []
    for m in MATS:
        txt = [t.format(p) for p in PROMPTS[m] for t in TEMPLATES]
        e = model.encode_text(tok(txt).to(DEV)).float()
        e = Fn.normalize(e, dim=-1).mean(0)
        T.append(Fn.normalize(e, dim=0))
    T = torch.stack(T)
    for vname, arr in views.items():
        feats = []
        for _, b in batches(arr):
            x = torch.from_numpy(b).to(DEV).permute(0, 3, 1, 2).float() / 255
            x = Fn.interpolate(x, size=(size, size), mode="bilinear", align_corners=False)
            x = (x - mean) / std
            f = model.encode_image(x).float() + model.encode_image(torch.flip(x, dims=[3])).float()
            feats.append(Fn.normalize(f, dim=-1).cpu())
        E = torch.cat(feats)
        out[f"{tag}_{vname}"] = E.numpy()
        out[f"{tag}_{vname}_zs"] = (E.to(DEV) @ T.T).cpu().numpy()
    out[f"{tag}_text"] = T.cpu().numpy()
    del model
    torch.cuda.empty_cache()
    print("done", tag, flush=True)


@torch.no_grad()
def dino_feats(views, masks, out):
    model = timm.create_model("vit_large_patch14_dinov2.lvd142m", pretrained=True, num_classes=0, img_size=224).eval().to(DEV)
    cfg = timm.data.resolve_data_config({}, model=model)
    mean = torch.tensor(cfg["mean"], device=DEV).view(1, 3, 1, 1)
    std = torch.tensor(cfg["std"], device=DEV).view(1, 3, 1, 1)
    for vname, arr in views.items():
        g, pm = [], []
        for i, b in batches(arr, 32):
            x = (torch.from_numpy(b).to(DEV).permute(0, 3, 1, 2).float() / 255 - mean) / std
            tokens = model.forward_features(x)
            cls = tokens[:, 0]
            patches = tokens[:, model.num_prefix_tokens:]
            n = int(patches.shape[1] ** 0.5)
            g.append(cls.float().cpu())
            if vname in masks:
                m = torch.from_numpy(masks[vname][i:i + 32]).to(DEV).float()[:, None]
                m = Fn.interpolate(m, size=(n, n), mode="area").flatten(1)
                m = m / m.sum(1, keepdim=True).clamp(min=1e-3)
                pm.append((m[:, :, None] * patches).sum(1).float().cpu())
        out[f"dinoL_{vname}"] = torch.cat(g).numpy()
        if pm:
            out[f"dinoL_{vname}_mpool"] = torch.cat(pm).numpy()
    del model
    torch.cuda.empty_cache()
    print("done dinoL", flush=True)


def main():
    recs = records()
    views = {"m10": [], "m14": [], "c14": [], "c20": []}
    masks = {"c14": [], "c20": []}
    cache = {}
    for i, (src, im, a) in enumerate(recs):
        fd = D / Path(im["file_name"]).parent
        if fd not in cache:
            cache.clear()
            cache[fd] = cv2.cvtColor(cv2.imread(str(fd / "color.png")), cv2.COLOR_BGR2RGB)
        rgb = cache[fd]
        m = polygon_mask(a["segmentation"], im["height"], im["width"])
        for key, (sc, msk) in {"m10": (1.1, True), "m14": (1.5, True), "c14": (1.5, False), "c20": (2.2, False)}.items():
            c, mm = crop(rgb, m, a["bbox"], sc, msk)
            views[key].append(c)
            if key in masks:
                masks[key].append(mm)
    views = {k: np.stack(v) for k, v in views.items()}
    masks = {k: np.stack(v) for k, v in masks.items()}
    out = {}
    for tag, fn in (("clipL", lambda o: clip_feats("ViT-L-14", "dfn2b", views, o, "clipL")),
                    ("siglip", lambda o: clip_feats("ViT-B-16-SigLIP", "webli", views, o, "siglip")),
                    ("dinoL", lambda o: dino_feats(views, masks, o))):
        part = OUT / f"f2_{tag}.npz"
        if part.exists():
            out.update(dict(np.load(part)))
            continue
        o = {}
        fn(o)
        np.savez_compressed(part, **o)
        out.update(o)
    np.savez_compressed(OUT / "features2.npz", **out)
    print("saved", {k: v.shape for k, v in out.items()})


if __name__ == "__main__":
    main()
