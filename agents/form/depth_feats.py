"""Depth / height-map features for bulk-vs-solid inside an object mask (RGB-independent).
Piles: low relative median height, rough surface with many local maxima, low fill ratio; solids: smooth, high fill."""
import json

import cv2
import numpy as np

from common import D

CAM = json.load(open(D / "camera.json"))
DEPTH_KEYS = ["hmax", "h_med_rel", "h25_rel", "h75_rel", "low5_frac", "low10_frac", "fill", "fill_thr", "grad_rel", "lap_rel", "n_peaks", "n_peaks_area",
              "cc_half", "cc_half_area", "top_std_rel", "solidity", "valid", "foot", "n_cells", "edge_density"]


def depth_features(depth, mask, plane, cell=0.004):
    n = np.asarray(plane[:3], float)
    d0 = float(plane[3])
    out = {k: 0.0 for k in DEPTH_KEYS}
    mm = int(mask.sum())
    vv, uu = np.nonzero(mask & (depth > 0))
    out["valid"] = len(vv) / max(mm, 1)
    # mask solidity (area / convex hull area) in image
    cnts, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if cnts:
        pts = np.concatenate(cnts)
        hull = cv2.convexHull(pts)
        out["solidity"] = float(mm / max(cv2.contourArea(hull), 1.0))
    if len(vv) < 30:
        return out
    z = depth[vv, uu] / 1000.0
    P = np.stack([(uu - CAM["cx"]) / CAM["fx"] * z, (vv - CAM["cy"]) / CAM["fy"] * z, z], 1)
    h = P @ n + d0
    if np.median(h) < 0:
        h = -h
    hmax = float(np.percentile(h, 99.5))
    out["hmax"] = hmax
    hm = max(hmax, 0.005)
    out["h_med_rel"], out["h25_rel"], out["h75_rel"] = [float(np.percentile(h, q) / hm) for q in (50, 25, 75)]
    out["low5_frac"] = float((h < 0.005).mean())
    out["low10_frac"] = float((h < 0.010).mean())
    # top-view raster
    e1 = np.cross(n, [1.0, 0, 0])
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(n, e1)
    a, b = P @ e1, P @ e2
    ia = np.floor((a - a.min()) / cell).astype(int)
    ib = np.floor((b - b.min()) / cell).astype(int)
    H = np.full((ia.max() + 1, ib.max() + 1), -1.0, np.float32)
    np.maximum.at(H, (ia, ib), h.astype(np.float32))
    occ = H >= 0
    tops = H[occ]
    out["n_cells"] = float(occ.sum())
    out["foot"] = float(occ.sum() * cell * cell * 1e4)
    out["fill"] = float(np.clip(tops, 0, None).mean() / hm)
    out["fill_thr"] = float((np.clip(tops - 0.003, 0, None).mean()) / hm)
    out["top_std_rel"] = float(np.std(tops) / hm)
    # smooth height map (fill holes with min), local maxima count
    Hf = H.copy()
    Hf[~occ] = 0
    Hs = cv2.GaussianBlur(Hf, (0, 0), 1.0)
    dil = cv2.dilate(Hs, np.ones((5, 5), np.uint8))
    peaks = (Hs >= dil - 1e-6) & occ & (Hs > 0.3 * hm) & (Hs > 0.004)
    npk = int(cv2.connectedComponents(peaks.astype(np.uint8))[0] - 1)
    out["n_peaks"] = float(npk)
    out["n_peaks_area"] = float(npk / max(occ.sum() * cell * cell * 1e4, 1.0))
    half = (Hs > 0.5 * hm) & occ
    ncc = int(cv2.connectedComponents(half.astype(np.uint8))[0] - 1)
    out["cc_half"] = float(ncc)
    out["cc_half_area"] = float(ncc / max(occ.sum() * cell * cell * 1e4, 1.0))
    # roughness in image space: gradient of height image inside eroded mask, relative to hmax
    himg = np.zeros(depth.shape, np.float32)
    himg[vv, uu] = h
    valid = np.zeros(depth.shape, np.uint8)
    valid[vv, uu] = 1
    er = cv2.erode(valid, np.ones((5, 5), np.uint8)) > 0
    if er.sum() > 20:
        hs = cv2.GaussianBlur(himg, (0, 0), 1.0)
        gx = cv2.Sobel(hs, cv2.CV_32F, 1, 0, ksize=3) / 8
        gy = cv2.Sobel(hs, cv2.CV_32F, 0, 1, ksize=3) / 8
        g = np.sqrt(gx ** 2 + gy ** 2)[er]
        px_m = float(np.median(z)) / CAM["fx"]  # metres per pixel
        out["grad_rel"] = float(np.median(g) / px_m)  # dimensionless slope
        lap = np.abs(cv2.Laplacian(hs, cv2.CV_32F))[er]
        out["lap_rel"] = float(np.median(lap) / px_m / 10)
        # edge density: fraction of pixels with strong slope
        out["edge_density"] = float((g / px_m > 1.0).mean())
    return out


def depth_vector(f):
    return np.array([f[k] for k in DEPTH_KEYS], np.float32)
