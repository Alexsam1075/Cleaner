"""Visual check: FastSAM 'everything' segments inside GT masks for a few objects -> part count as a bulk cue."""
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

from common import D, W
from attr_features import polygon_mask
from ultralytics import FastSAM

model = FastSAM(str(W / "FastSAM-s.pt"))
lab = json.load(open(D / "labels/real_train.json"))
ims = {im["id"]: im for im in lab["images"]}
want = sys.argv[1:] or ["b06", "s19", "b01", "b07", "s29", "b02", "b12", "s11", "b18", "b13", "s45", "s04"]
seen = {}
tiles = []
for a in lab["annotations"]:
    o = a["attributes"]["object_id"]
    if o not in want or seen.get(o, 0) >= 2:
        continue
    seen[o] = seen.get(o, 0) + 1
    im = ims[a["image_id"]]
    fd = D / Path(im["file_name"]).parent
    rgb = cv2.imread(str(fd / "color.png"))
    m = polygon_mask(a["segmentation"], im["height"], im["width"])
    x, y, w, h = a["bbox"]
    s = max(w, h) * 1.3 + 8
    cx, cy = x + w / 2, y + h / 2
    x1, y1 = int(max(0, cx - s / 2)), int(max(0, cy - s / 2))
    x2, y2 = int(min(rgb.shape[1], cx + s / 2)), int(min(rgb.shape[0], cy + s / 2))
    crop = rgb[y1:y2, x1:x2]
    mc = m[y1:y2, x1:x2]
    r = model.predict(crop, imgsz=512, conf=0.25, iou=0.7, retina_masks=True, verbose=False, device=0)[0]
    vis = crop.copy()
    n_in = 0
    sizes = []
    if r.masks is not None:
        rng = np.random.default_rng(0)
        for k in range(len(r.masks.data)):
            sm = r.masks.data[k].cpu().numpy() > 0.5
            if sm.shape != mc.shape:
                sm = cv2.resize(sm.astype(np.uint8), (mc.shape[1], mc.shape[0]), interpolation=cv2.INTER_NEAREST) > 0
            inter = (sm & mc).sum()
            if inter < 0.7 * sm.sum() or sm.sum() < 0.002 * mc.sum() or sm.sum() > 0.9 * mc.sum():
                continue
            n_in += 1
            sizes.append(inter / mc.sum())
            col = rng.integers(60, 255, 3).tolist()
            vis[sm] = (0.5 * vis[sm] + 0.5 * np.array(col)).astype(np.uint8)
    t = cv2.resize(vis, (256, 256))
    cv2.putText(t, f"{o} n={n_in} max={max(sizes) if sizes else 0:.2f}", (3, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
    tiles.append(t)
    print(o, fd.name, n_in, np.round(sorted(sizes, reverse=True)[:8], 3), flush=True)
while len(tiles) % 6:
    tiles.append(np.zeros((256, 256, 3), np.uint8))
rows = [np.concatenate(tiles[i:i + 6], 1) for i in range(0, len(tiles), 6)]
cv2.imwrite(str(W / "fastsam_vis.jpg"), np.concatenate(rows, 0), [cv2.IMWRITE_JPEG_QUALITY, 80])
