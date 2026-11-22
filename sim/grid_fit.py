"""Оценка положения сетки плитки на полу (угол и фаза швов) для каждого кадра — чтобы рендер совпал с фото."""
import json

import cv2
import numpy as np

from floor_texture import rectify
from geom import ROOT

PX = 500
X0, Y1 = -1.0, 2.3


def seam_lines(fid):
    img, valid = rectify(fid, px_per_m=PX)
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    bh = cv2.morphologyEx(g, cv2.MORPH_BLACKHAT, np.ones((9, 9), np.uint8))
    bh[cv2.erode(valid.astype(np.uint8), np.ones((15, 15), np.uint8)) == 0] = 0
    _, b = cv2.threshold(bh, 12, 255, cv2.THRESH_BINARY)
    L = cv2.HoughLines(b, 1, np.pi / 720, 150)
    out = []
    for rho, t in ([] if L is None else L[:, 0]):
        n = np.array([np.cos(t), -np.sin(t)])
        c = rho / PX + X0 * np.cos(t) - Y1 * np.sin(t)
        out.append((n, c))
    return out


def fit(fid, tile_m):
    lines = seam_lines(fid)
    if not lines:
        return None
    phi0 = np.arctan2(lines[0][0][1], lines[0][0][0]) % (np.pi / 2)
    n1 = np.array([np.cos(phi0), np.sin(phi0)])
    n2 = np.array([-np.sin(phi0), np.cos(phi0)])
    ph1, ph2 = [], []
    for n, c in lines:
        for ax, acc in ((n1, ph1), (n2, ph2)):
            dot = n @ ax
            if abs(dot) > np.cos(np.radians(2)):
                acc.append((c * np.sign(dot)) % tile_m)
    circ = lambda a: float((np.angle(np.mean(np.exp(2j * np.pi * np.array(a) / tile_m))) / (2 * np.pi) * tile_m) % tile_m) if a else 0.0
    return {"phi": float(phi0), "c1": circ(ph1), "c2": circ(ph2), "n1": len(ph1), "n2": len(ph2)}


if __name__ == "__main__":
    tile_m = json.loads((ROOT / "sim/assets/surfaces.json").read_text())["tile_m"]
    for fid in ["r_0077", "r_0001", "r_0270", "r_0400"]:
        print(fid, fit(fid, tile_m))
