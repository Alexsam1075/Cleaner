"""Feature extraction for the form classifier v2.
Samples: (a) GT annotations real_train + synth_train (order of audit/cache/meta.json), (b) cached detections of folds 1-3
(order of agents/form/cache/fold{k}_dets.json; masks re-created by re-running the fold detector).
Per sample we store:
  text sims   : CLIP-L c14 / m14 crop embeddings (already cached) @ new prompt families -> prompts.npz
  dino        : DINOv2-B/14 @448 patch tokens inside mask: mean-pooled token, token-variance statistics, local self-similarity
  depth       : height-map features (depth_feats.depth_features)
  sam         : FastSAM-s 'everything' inside the mask: number of parts, largest part share, entropy of part sizes
Output: cache/gt_v2.npz, cache/fold{k}_v2.npz
"""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import timm
import torch
import torch.nn.functional as Fn

from common import D, W, W0, FOLDS, VIEWS
from attr_features import polygon_mask, records
from mat_features2 import crop
from depth_feats import depth_features, DEPTH_KEYS
from prompts import FAMILIES, TEMPL

DEV = "cuda"
DINO_KEYS = ["tokvar", "tokvar_rel", "selfsim_mean", "selfsim_p10", "n_clusters", "cls_dist", "edge_frac", "eff_rank"]
SAM_KEYS = ["sam_n", "sam_n_sig", "sam_max", "sam_top3", "sam_entropy", "sam_cover", "sam_small_frac"]


class Dino:
    def __init__(self, size=448):
        self.m = timm.create_model("vit_base_patch14_dinov2.lvd142m", pretrained=True, num_classes=0, img_size=size).eval().to(DEV).half()
        cfg = timm.data.resolve_data_config({}, model=self.m)
        self.mean = torch.tensor(cfg["mean"], device=DEV).view(1, 3, 1, 1)
        self.std = torch.tensor(cfg["std"], device=DEV).view(1, 3, 1, 1)
        self.size = size
        self.g = size // 14

    @torch.no_grad()
    def __call__(self, imgs, masks):
        """imgs (B,H,W,3) uint8 RGB crops, masks (B,H,W) bool. Returns pooled (B,768) and stats (B,len(DINO_KEYS))."""
        pooled, stats = [], []
        for i in range(0, len(imgs), 16):
            x = torch.from_numpy(np.ascontiguousarray(imgs[i:i + 16])).to(DEV).permute(0, 3, 1, 2).float() / 255
            x = ((Fn.interpolate(x, size=(self.size, self.size), mode="bilinear", align_corners=False) - self.mean) / self.std).half()
            tok = self.m.forward_features(x).float()
            cls, pt = tok[:, 0], tok[:, self.m.num_prefix_tokens:]
            m = torch.from_numpy(np.ascontiguousarray(masks[i:i + 16])).to(DEV).float()[:, None]
            m = Fn.interpolate(m, size=(self.g, self.g), mode="area").flatten(1)
            for b in range(len(pt)):
                sel = m[b] > 0.5
                if sel.sum() < 4:
                    sel = m[b] > 0.1
                if sel.sum() < 4:
                    sel = torch.ones_like(sel)
                P = pt[b][sel]
                Pn = Fn.normalize(P, dim=-1)
                mu = P.mean(0)
                pooled.append(Fn.normalize(mu, dim=0).cpu().numpy())
                var = ((P - mu) ** 2).sum(1).mean()
                S = Pn @ Pn.T
                n = len(P)
                off = S[~torch.eye(n, dtype=bool, device=DEV)]
                # clusters: k-means-free proxy = number of eigenvalues of token covariance needed for 80% energy
                C = (P - mu).T @ (P - mu) / n
                ev = torch.linalg.eigvalsh(C).flip(0).clamp(min=0)
                cum = ev.cumsum(0) / ev.sum().clamp(min=1e-6)
                n80 = int((cum < 0.8).sum().item()) + 1
                eff_rank = float(torch.exp(-(ev / ev.sum().clamp(min=1e-6) + 1e-12).log().mul(ev / ev.sum().clamp(min=1e-6)).sum()).item())
                # spatial edges: fraction of 4-neighbour pairs (inside mask) with low similarity
                grid = torch.zeros(self.g * self.g, P.shape[1], device=DEV)
                grid[sel] = Pn
                G = grid.view(self.g, self.g, -1)
                selg = sel.view(self.g, self.g)
                simh = (G[:, 1:] * G[:, :-1]).sum(-1)
                simv = (G[1:] * G[:-1]).sum(-1)
                mh = selg[:, 1:] & selg[:, :-1]
                mv = selg[1:] & selg[:-1]
                pairs = torch.cat([simh[mh], simv[mv]])
                edge_frac = float((pairs < 0.5).float().mean().item()) if len(pairs) else 0.0
                cls_dist = float((1 - Fn.cosine_similarity(cls[b][None], P).mean()).item())
                stats.append([float(var), float(var / (P ** 2).sum(1).mean()), float(off.mean()), float(off.quantile(0.1)) if len(off) else 1.0,
                              float(n80), cls_dist, edge_frac, eff_rank])
        return np.stack(pooled), np.array(stats, np.float32)


class Clip:
    def __init__(self):
        import open_clip
        model, _, _ = open_clip.create_model_and_transforms("ViT-L-14", pretrained="dfn2b", device=DEV)
        model.eval()
        tok = open_clip.get_tokenizer("ViT-L-14")
        self.prompts = sorted({p for pb, ps in FAMILIES.values() for p in pb + ps})
        T = []
        with torch.no_grad():
            for p in self.prompts:
                e = Fn.normalize(model.encode_text(tok([t.format(p) for t in TEMPL]).to(DEV)).float(), dim=-1).mean(0)
                T.append(Fn.normalize(e, dim=0))
        self.T = torch.stack(T).cpu().numpy()
        del model
        torch.cuda.empty_cache()


class Sam:
    def __init__(self):
        from ultralytics import FastSAM
        self.m = FastSAM(str(W / "FastSAM-s.pt"))

    def __call__(self, rgb_bgr, mask, bbox):
        x, y, w, h = bbox
        s = max(w, h) * 1.3 + 8
        cx, cy = x + w / 2, y + h / 2
        x1, y1 = int(max(0, cx - s / 2)), int(max(0, cy - s / 2))
        x2, y2 = int(min(rgb_bgr.shape[1], cx + s / 2)), int(min(rgb_bgr.shape[0], cy + s / 2))
        if x2 - x1 < 8 or y2 - y1 < 8:
            return [0.0] * len(SAM_KEYS)
        c = rgb_bgr[y1:y2, x1:x2]
        mc = mask[y1:y2, x1:x2]
        area = max(int(mc.sum()), 1)
        r = self.m.predict(c, imgsz=512, conf=0.25, iou=0.7, retina_masks=True, verbose=False, device=0)[0]
        shares = []
        cover = np.zeros_like(mc)
        if r.masks is not None:
            for k in range(len(r.masks.data)):
                sm = r.masks.data[k].cpu().numpy() > 0.5
                if sm.shape != mc.shape:
                    sm = cv2.resize(sm.astype(np.uint8), (mc.shape[1], mc.shape[0]), interpolation=cv2.INTER_NEAREST) > 0
                inter = int((sm & mc).sum())
                if sm.sum() == 0 or inter < 0.7 * sm.sum() or inter < 0.002 * area or inter > 0.9 * area:
                    continue
                shares.append(inter / area)
                cover |= sm & mc
        shares = np.array(sorted(shares, reverse=True)) if shares else np.zeros(0)
        n = len(shares)
        n_sig = int((shares > 0.02).sum())
        ent = float(-(shares / shares.sum() * np.log(shares / shares.sum() + 1e-9)).sum()) if n else 0.0
        return [float(n), float(n_sig), float(shares[0]) if n else 0.0, float(shares[:3].sum()) if n else 0.0, ent,
                float(cover.sum() / area), float((shares < 0.02).sum() / max(n, 1))]


def process(samples, clip, dino, sam, tag):
    """samples: list of (frame_dir, mask(bool HxW), bbox[x,y,w,h]). Returns dict of arrays."""
    out = {"dino_pool": [], "dino_stats": [], "depth": [], "sam": []}
    cache = {}
    pend_img, pend_msk = [], []

    def flush():
        if pend_img:
            p, s = dino(np.stack(pend_img), np.stack(pend_msk))
            out["dino_pool"].append(p)
            out["dino_stats"].append(s)
            pend_img.clear()
            pend_msk.clear()

    for i, (fd, m, bbox) in enumerate(samples):
        if fd not in cache:
            cache.clear()
            bgr = cv2.imread(str(fd / "color.png"))
            cache[fd] = (bgr, cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), cv2.imread(str(fd / "depth.png"), cv2.IMREAD_UNCHANGED),
                         json.load(open(fd / "meta.json"))["floor_plane"])
        bgr, rgb, depth, plane = cache[fd]
        c, mm = crop(rgb, m, bbox, 1.2, False, size=448)
        pend_img.append(c)
        pend_msk.append(mm > 0)
        f = depth_features(depth, m, plane)
        out["depth"].append([f[k] for k in DEPTH_KEYS])
        out["sam"].append(sam(bgr, m, bbox))
        if len(pend_img) >= 32:
            flush()
        if i % 100 == 0:
            print(f"{tag}: {i}/{len(samples)}", flush=True)
    flush()
    return {"dino_pool": np.concatenate(out["dino_pool"]), "dino_stats": np.concatenate(out["dino_stats"]),
            "depth": np.array(out["depth"], np.float32), "sam": np.array(out["sam"], np.float32)}


def gt_samples():
    recs = records()
    return [(D / Path(im["file_name"]).parent, polygon_mask(a["segmentation"], im["height"], im["width"]), a["bbox"]) for _, im, a in recs]


def det_samples(fold):
    """Re-run the fold detector to recover masks for the cached detections (same settings as detect_cache.py); match by box."""
    from ultralytics import YOLO
    rows = json.load(open(W0 / "cache" / f"fold{fold}_dets.json"))
    det = YOLO(str(FOLDS[fold][1]))
    frames = sorted({r["frame"] for r in rows})
    masks = {}
    for fid in frames:
        fd = D / "real/train" / fid
        r = det.predict(str(fd / "color.png"), imgsz=640, conf=0.01, iou=0.7, max_det=60, retina_masks=True, verbose=False, device=0)[0]
        n = 0 if r.boxes is None else len(r.boxes)
        for j in range(n):
            b = r.boxes.xyxy[j].cpu().numpy().astype(float)
            m = r.masks.data[j].cpu().numpy() > 0.5 if r.masks is not None else None
            masks.setdefault(fid, []).append((b, m))
    samples = []
    miss = 0
    for r in rows:
        best, bm = 1e9, None
        for b, m in masks.get(r["frame"], []):
            d = float(np.abs(b - np.array(r["box"])).max())
            if d < best:
                best, bm = d, m
        bb = r["box"]
        if bm is None or best > 2.0 or bm.sum() < 10:
            miss += 1
            bm = np.zeros((720, 1280), bool)
            bm[int(bb[1]):int(bb[3]) + 1, int(bb[0]):int(bb[2]) + 1] = True
        if bm.shape != (720, 1280):
            bm = cv2.resize(bm.astype(np.uint8), (1280, 720), interpolation=cv2.INTER_NEAREST) > 0
        samples.append((D / "real/train" / r["frame"], bm, [bb[0], bb[1], bb[2] - bb[0], bb[3] - bb[1]]))
    print(f"fold {fold}: {len(rows)} dets, {miss} without matching mask (box used)", flush=True)
    del det
    torch.cuda.empty_cache()
    return samples


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--what", nargs="+", default=["gt", "1", "2", "3"])
    a = ap.parse_args()
    (W / "cache").mkdir(exist_ok=True)
    clip = Clip()
    np.savez(W / "cache/prompts.npz", T=clip.T, prompts=np.array(clip.prompts))
    dino, sam = Dino(), Sam()
    for what in a.what:
        if what == "gt":
            samples = gt_samples()
            out_p = W / "cache/gt_v2.npz"
        else:
            samples = det_samples(int(what))
            out_p = W / f"cache/fold{what}_v2.npz"
        res = process(samples, clip, dino, sam, what)
        np.savez_compressed(out_p, **res, depth_keys=np.array(DEPTH_KEYS), dino_keys=np.array(DINO_KEYS), sam_keys=np.array(SAM_KEYS))
        print("saved", out_p, {k: v.shape for k, v in res.items()}, flush=True)


if __name__ == "__main__":
    main()
