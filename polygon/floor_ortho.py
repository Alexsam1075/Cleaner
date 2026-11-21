"""Top-down floor texture from a real empty frame: ortho-rectify the sharp near floor, flatten light.

  python -m polygon.floor_ortho --frame r_0077 --out polygon/assets/floor_tile.png
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from .render import Camera


def ortho(rgb, cam: Camera, x0, x1, y0, y1, res):
    xs = np.arange(x0, x1, res)
    ys = np.arange(y0, y1, res)
    X, Y = np.meshgrid(xs, ys, indexing="xy")
    P = np.stack([X, Y, np.zeros_like(X)], -1).reshape(-1, 3)
    import torch
    pix, z = cam.project(torch.tensor(P, dtype=torch.float32, device=cam.origin.device))
    pix = pix.cpu().numpy().reshape(X.shape + (2,)).astype(np.float32)
    return cv2.remap(rgb, pix[..., 0], pix[..., 1], cv2.INTER_CUBIC, borderMode=cv2.BORDER_REFLECT)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="datasets/raw", type=Path)
    ap.add_argument("--frame", default="r_0077")
    ap.add_argument("--out", default="polygon/assets/floor_tile.png", type=Path)
    ap.add_argument("--res", type=float, default=0.001)
    args = ap.parse_args()
    cam_json = json.loads((args.data / "camera.json").read_text())
    fdir = args.data / "real/train" / args.frame
    meta = json.loads((fdir / "meta.json").read_text())
    cam = Camera.from_plane(cam_json, meta["floor_plane"])
    rgb = cv2.imread(str(fdir / "color.png"))
    img = ortho(rgb, cam, 0.42, 0.98, -0.30, 0.30, args.res).astype(np.float32)
    illum = cv2.GaussianBlur(img, (0, 0), 90)
    flat = img / np.maximum(illum, 1) * illum.reshape(-1, 3).mean(0)
    flat = np.clip(flat, 0, 255).astype(np.uint8)
    lum = cv2.cvtColor(flat, cv2.COLOR_BGR2GRAY).astype(np.float32)
    med = cv2.GaussianBlur(lum, (0, 0), 25)
    specks = cv2.dilate(((lum < 0.9 * med) | (lum > 1.25 * med)).astype(np.uint8), np.ones((15, 15), np.uint8))
    flat = cv2.inpaint(flat, specks, 9, cv2.INPAINT_TELEA)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(args.out), flat)
    (args.out.with_suffix(".json")).write_text(json.dumps({"size_m": [0.56, 0.60], "res_m": args.res,
                                                           "source": args.frame}))
    print(args.out, flat.shape)


if __name__ == "__main__":
    main()
