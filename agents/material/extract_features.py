"""Material features v3 (agent material). Reuses crops/records from audit/attr_features.py.
Per annotation (892 = 214 real + 678 synth) we compute, for CLIP ViT-L/14 (DFN2B) and SigLIP ViT-B/16:
  * global embeddings of several views: context crops at scales 1.2/1.5/2.0, masked crop 1.2,
    lighting-normalised (gray-world + CLAHE) crops at 1.5/2.0; hflip TTA inside every view;
  * mask-pooled dense features: MaskCLIP (value-path of last block) and plain patch tokens for CLIP-L,
    mask-restricted attention pooling for SigLIP;
  * text embeddings of several GENERIC prompt sets written up-front (no per-object tuning).
Also depth/specular hand stats. Output: cache/feat3.npz (rows aligned with shared_cache/meta.json).
"""
import json
import os
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as Fn

sys.path.insert(0, "/home/cleaner/Cleaner/audit")
os.environ.setdefault("ASTRAMIS_DATA", "/home/cleaner/Cleaner/datasets/raw")
from attr_features import polygon_mask, records  # noqa: E402

D = Path(os.environ["ASTRAMIS_DATA"])
HERE = Path(__file__).resolve().parent
OUT = HERE / "cache"
DEV = "cuda"
MATS = ["metal", "wood", "plastic", "glass", "paper", "mineral"]

# ---- generic prompt vocabulary (written up-front) --------------------------------------------
WORDS = {
    "name": {"metal": ["metal"], "wood": ["wood"], "plastic": ["plastic"], "glass": ["glass"], "paper": ["paper"], "mineral": ["stone"]},
    "adj": {"metal": ["metallic"], "wood": ["wooden"], "plastic": ["plastic"], "glass": ["glass"], "paper": ["paper"], "mineral": ["stone"]},
    "syn": {"metal": ["metal", "steel", "aluminium", "iron", "metallic"],
            "wood": ["wood", "wooden", "timber", "sawdust", "wood chips"],
            "plastic": ["plastic", "polymer", "plastic film", "plastic bag"],
            "glass": ["glass", "transparent glass", "glassware"],
            "paper": ["paper", "cardboard", "paperboard"],
            "mineral": ["stone", "rock", "mineral", "concrete", "ceramic"]},
}
TEMPL = {
    "t1": ["a photo of an object made of {}."],
    "t5": ["a photo of {} on the floor.", "a close-up photo of {}.", "{} lying on a tiled floor.", "trash made of {}.", "a photo of an object made of {}."],
    "t12": ["a photo of {}.", "a photo of an object made of {}.", "a close-up photo of something made of {}.", "a {} object.",
            "an item made of {} lying on the floor.", "a piece of {} trash.", "a photo of {} material.", "{} debris on the floor.",
            "a low quality photo of an object made of {}.", "a dim photo of an object made of {}.", "a bright photo of an object made of {}.",
            "a blurry photo of an object made of {}."],
    "adj": ["a {} object.", "a photo of a {} object.", "a {} item on the floor.", "a piece of {} trash.", "a close-up photo of a {} object.",
            "a dim photo of a {} object."],
}
PROMPT_SETS = {"name_t1": ("name", "t1"), "name_t5": ("name", "t5"), "name_t12": ("name", "t12"),
               "adj_t": ("adj", "adj"), "syn_t5": ("syn", "t5"), "syn_t12": ("syn", "t12")}

VIEWS = {  # name: (scale, masked, lightnorm)
    "c12": (1.2, False, False), "c15": (1.5, False, False), "c20": (2.0, False, False), "m12": (1.2, True, False),
    "n15": (1.5, False, True), "n20": (2.0, False, True),
}
DENSE_VIEWS = ["c15", "c20", "n15"]
CLIP_NAME = os.environ.get("CLIP_NAME", "ViT-L-14-quickgelu")
OUT_NAME = os.environ.get("OUT_NAME", "feat3q.npz")


def lightnorm(img):
    f = img.astype(np.float32)
    m = f.reshape(-1, 3).mean(0) + 1e-3
    f = np.clip(f * (m.mean() / m), 0, 255).astype(np.uint8)
    lab = cv2.cvtColor(f, cv2.COLOR_RGB2LAB)
    lab[..., 0] = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(lab[..., 0])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)


def crop(rgb, mask, bbox, scale, masked, norm, size=224):
    x, y, w, h = bbox
    s = max(w, h) * scale + 8
    cx, cy = x + w / 2, y + h / 2
    x1, y1, x2, y2 = int(cx - s / 2), int(cy - s / 2), int(cx + s / 2), int(cy + s / 2)
    pad = max(0, -x1, -y1, x2 - rgb.shape[1], y2 - rgb.shape[0])
    rp = cv2.copyMakeBorder(rgb, pad, pad, pad, pad, cv2.BORDER_REFLECT)
    mp = cv2.copyMakeBorder(mask.astype(np.uint8), pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=0)
    c = rp[y1 + pad:y2 + pad, x1 + pad:x2 + pad].copy()
    m = mp[y1 + pad:y2 + pad, x1 + pad:x2 + pad]
    if norm:
        c = lightnorm(c)
    if masked:
        c[m == 0] = 124
    return cv2.resize(c, (size, size), interpolation=cv2.INTER_AREA), cv2.resize(m, (size, size), interpolation=cv2.INTER_NEAREST)


def depth_stats(depth, mask, rgb):
    er = cv2.erode(mask.astype(np.uint8), np.ones((5, 5), np.uint8)) > 0
    if er.sum() < 20:
        er = mask
    hole = float(((depth == 0) & er).sum() / max(er.sum(), 1))
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    V, S = hsv[..., 2][er].astype(np.float32), hsv[..., 1][er].astype(np.float32)
    g = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)
    ring = (cv2.dilate(mask.astype(np.uint8), np.ones((31, 31), np.uint8)) > 0) & ~mask
    bg = g[ring].mean() if ring.any() else g.mean()
    z = depth[er & (depth > 0)].astype(np.float32)
    zs = float(np.std(z) / 1000) if len(z) > 20 else 0.0
    return np.array([hole, (V > 240).mean(), ((V > 215) & (S < 35)).mean(), (V < 45).mean(), V.std() / 255, S.mean() / 255,
                     np.percentile(V, 99) / 255 - np.median(V) / 255, (g[er].mean() - bg) / 255, zs, np.log1p(er.sum()) / 10], np.float32)


def mask_grid(masks, n):
    m = torch.from_numpy(masks).to(DEV).float()[:, None]
    m = Fn.interpolate(m, size=(n, n), mode="area").flatten(1)
    empty = m.sum(1) < 0.5
    m[empty] = 1.0
    return m / m.sum(1, keepdim=True)


@torch.no_grad()
def run_clipL(views, masks, out):
    import open_clip
    model, _, pre = open_clip.create_model_and_transforms(CLIP_NAME, pretrained="dfn2b", device=DEV)
    model.eval().half()
    tok = open_clip.get_tokenizer("ViT-L-14")
    mean = torch.tensor(pre.transforms[-1].mean, device=DEV).view(1, 3, 1, 1)
    std = torch.tensor(pre.transforms[-1].std, device=DEV).view(1, 3, 1, 1)
    for sname, (wk, tk) in PROMPT_SETS.items():
        T = []
        for m in MATS:
            txt = [t.format(p) for p in WORDS[wk][m] for t in TEMPL[tk]]
            e = Fn.normalize(model.encode_text(tok(txt).to(DEV)).float(), dim=-1).mean(0)
            T.append(Fn.normalize(e, dim=0))
        out[f"txt_clipL_{sname}"] = torch.stack(T).cpu().numpy()
    v = model.visual
    last = v.transformer.resblocks[-1]

    def prep(b):
        x = torch.from_numpy(b).to(DEV).permute(0, 3, 1, 2).float() / 255
        return ((x - mean) / std).half()

    def dense_and_global(x):
        t = v._embeds(x)
        for blk in v.transformer.resblocks[:-1]:
            t = blk(t)
        full = last(t)
        pooled = v.ln_post(full[:, 0]) @ v.proj
        ptok = v.ln_post(full[:, 1:]) @ v.proj
        y = last.ln_1(t)
        w_q, w_k, w_v = last.attn.in_proj_weight.chunk(3, dim=0)
        b_q, b_k, b_v = last.attn.in_proj_bias.chunk(3)
        vv = last.attn.out_proj(Fn.linear(y, w_v, b_v))
        mclip = v.ln_post(vv[:, 1:]) @ v.proj
        return pooled.float(), ptok.float(), mclip.float()

    for vname, arr in views.items():
        G, PT, MC = [], [], []
        for i in range(0, len(arr), 32):
            x = prep(arr[i:i + 32])
            xs = torch.cat([x, torch.flip(x, dims=[3])])
            g, pt, mc = dense_and_global(xs)
            n = len(x)
            G.append(Fn.normalize(Fn.normalize(g[:n], dim=-1) + Fn.normalize(g[n:], dim=-1), dim=-1).cpu())
            if vname in DENSE_VIEWS:
                mg = mask_grid(masks[vname][i:i + 32], 16)
                mgf = torch.flip(mg.view(-1, 16, 16), dims=[2]).flatten(1)
                pool = lambda d: (mg[:, :, None] * d[:n]).sum(1) + (mgf[:, :, None] * d[n:]).sum(1)
                PT.append(Fn.normalize(pool(pt), dim=-1).cpu())
                MC.append(Fn.normalize(pool(mc), dim=-1).cpu())
        out[f"clipL_{vname}"] = torch.cat(G).numpy()
        if PT:
            out[f"clipLpt_{vname}"] = torch.cat(PT).numpy()
            out[f"clipLmc_{vname}"] = torch.cat(MC).numpy()
        print("clipL", vname, flush=True)
    del model
    torch.cuda.empty_cache()


@torch.no_grad()
def run_siglip(views, masks, out):
    import open_clip
    model, _, pre = open_clip.create_model_and_transforms("ViT-B-16-SigLIP", pretrained="webli", device=DEV)
    model.eval().half()
    tok = open_clip.get_tokenizer("ViT-B-16-SigLIP")
    mean = torch.tensor(pre.transforms[-1].mean, device=DEV).view(1, 3, 1, 1)
    std = torch.tensor(pre.transforms[-1].std, device=DEV).view(1, 3, 1, 1)
    for sname, (wk, tk) in PROMPT_SETS.items():
        T = []
        for m in MATS:
            txt = [t.format(p) for p in WORDS[wk][m] for t in TEMPL[tk]]
            e = Fn.normalize(model.encode_text(tok(txt).to(DEV)).float(), dim=-1).mean(0)
            T.append(Fn.normalize(e, dim=0))
        out[f"txt_siglip_{sname}"] = torch.stack(T).cpu().numpy()
    trunk = model.visual.trunk
    ap = trunk.attn_pool
    for vname, arr in views.items():
        G, MP = [], []
        for i in range(0, len(arr), 32):
            x = torch.from_numpy(arr[i:i + 32]).to(DEV).permute(0, 3, 1, 2).float() / 255
            x = ((x - mean) / std).half()
            xs = torch.cat([x, torch.flip(x, dims=[3])])
            n = len(x)
            tokens = trunk.forward_features(xs)
            g = trunk.forward_head(tokens).float()
            G.append(Fn.normalize(Fn.normalize(g[:n], dim=-1) + Fn.normalize(g[n:], dim=-1), dim=-1).cpu())
            if vname in DENSE_VIEWS:
                mg = mask_grid(masks[vname][i:i + 32], 14)
                mgf = torch.flip(mg.view(-1, 14, 14), dims=[2]).flatten(1)
                am = torch.cat([mg, mgf]) > 1e-6
                am = am[:, None, None, :]
                pooled = ap(tokens, attn_mask=am).float()
                MP.append(Fn.normalize(Fn.normalize(pooled[:n], dim=-1) + Fn.normalize(pooled[n:], dim=-1), dim=-1).cpu())
        out[f"siglip_{vname}"] = torch.cat(G).numpy()
        if MP:
            out[f"siglipmp_{vname}"] = torch.cat(MP).numpy()
        print("siglip", vname, flush=True)
    del model
    torch.cuda.empty_cache()


def main():
    OUT.mkdir(exist_ok=True)
    recs = records()
    meta_ref = json.load(open(HERE / "shared_cache" / "meta.json"))
    assert len(recs) == len(meta_ref)
    views = {k: [] for k in VIEWS}
    masks = {k: [] for k in DENSE_VIEWS}
    hand = []
    cache = {}
    for i, (src, im, a) in enumerate(recs):
        fd = D / Path(im["file_name"]).parent
        if fd not in cache:
            cache.clear()
            rgb = cv2.cvtColor(cv2.imread(str(fd / "color.png")), cv2.COLOR_BGR2RGB)
            depth = cv2.imread(str(fd / "depth.png"), cv2.IMREAD_UNCHANGED)
            cache[fd] = (rgb, depth)
        rgb, depth = cache[fd]
        assert meta_ref[i]["frame"] == fd.name and meta_ref[i]["oid"] == a["attributes"]["object_id"]
        m = polygon_mask(a["segmentation"], im["height"], im["width"])
        hand.append(depth_stats(depth, m, rgb))
        for k, (sc, msk, nrm) in VIEWS.items():
            c, mm = crop(rgb, m, a["bbox"], sc, msk, nrm)
            views[k].append(c)
            if k in masks:
                masks[k].append(mm)
        if i % 200 == 0:
            print("crops", i, flush=True)
    views = {k: np.stack(v) for k, v in views.items()}
    masks = {k: np.stack(v) for k, v in masks.items()}
    out = {"hand": np.stack(hand)}
    run_clipL(views, masks, out)
    if os.environ.get("SKIP_SIGLIP") != "1":
        run_siglip(views, masks, out)
    np.savez_compressed(OUT / OUT_NAME, **out)
    print("saved", {k: v.shape for k, v in out.items()})


if __name__ == "__main__":
    main()
