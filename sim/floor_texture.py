"""Выпрямляет пол с реальных кадров в вид сверху (метрическая текстура) по плоскости пола и интринсикам."""
import argparse

import cv2
import numpy as np

from geom import ROOT, frame_dir, meta, world_to_pixel


def rectify(fid, x_range=(-1.0, 1.0), y_range=(0.3, 2.3), px_per_m=500):
    img = cv2.imread(str(frame_dir(fid) / "color.png"))
    plane = meta(fid)["floor_plane"]
    xs = np.arange(x_range[0], x_range[1], 1.0 / px_per_m)
    ys = np.arange(y_range[1], y_range[0], -1.0 / px_per_m)
    X, Y = np.meshgrid(xs, ys)
    Pw = np.stack([X, Y, np.zeros_like(X)], -1)
    uv, z = world_to_pixel(Pw, plane)
    mx = uv[..., 0].astype(np.float32)
    my = uv[..., 1].astype(np.float32)
    out = cv2.remap(img, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    valid = (z > 0) & (mx >= 0) & (mx < img.shape[1]) & (my >= 0) & (my < img.shape[0])
    out[~valid] = 0
    return out, valid


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("ids", nargs="+")
    a = ap.parse_args()
    for fid in a.ids:
        out, _ = rectify(fid)
        p = ROOT / "sim" / "out" / f"rect_{fid}.jpg"
        cv2.imwrite(str(p), out)
        print(p)
