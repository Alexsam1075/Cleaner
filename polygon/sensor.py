"""RealSense D435 depth model for rendered objects (the real background keeps its own depth).

Effects: noise growing with z^2 (calibrated on the frame's floor), flying pixels and holes on
silhouettes, the stereo occlusion band to the left of objects (50 mm baseline), holes on
specular highlights of metal and through glass (IR passes through or is mirrored away),
1 mm quantisation.
"""
from __future__ import annotations

import cv2
import numpy as np


def floor_sigma_mm(depth_u16: np.ndarray, floor: np.ndarray) -> float:
    d = depth_u16.astype(np.float32)
    hp = d - cv2.GaussianBlur(d, (0, 0), 3)
    v = hp[floor & (d > 0)]
    return float(1.4826 * np.median(np.abs(v - np.median(v)))) if len(v) > 500 else 1.2


def simulate_depth(depth_true_m: np.ndarray, bg_depth_u16: np.ndarray, ids: np.ndarray, spec: np.ndarray,
                   props: dict[int, dict], fx: float, sigma_floor_mm: float, rng: np.random.Generator,
                   baseline_m: float = 0.050) -> np.ndarray:
    H, W = ids.shape
    out = bg_depth_u16.astype(np.float32).copy()
    obj = ids >= 0
    if not obj.any():
        return bg_depth_u16.copy()
    z = depth_true_m * 1000.0
    zm = np.median(z[obj])
    # correlated + white noise, sigma ~ z^2 relative to the floor at ~0.55 m
    sig = sigma_floor_mm * (z / 550.0) ** 2
    corr = cv2.GaussianBlur(rng.standard_normal((H, W)).astype(np.float32), (0, 0), 2.0) * 2.2
    noisy = z + sig * (0.6 * corr + 0.5 * rng.standard_normal((H, W)).astype(np.float32))
    out[obj] = noisy[obj]
    holes = np.zeros((H, W), bool)
    for k, pr in props.items():
        m = ids == k
        if not m.any():
            continue
        u8 = m.astype(np.uint8)
        ring = (cv2.dilate(u8, np.ones((3, 3), np.uint8)) - cv2.erode(u8, np.ones((3, 3), np.uint8))) > 0
        r = rng.random((H, W))
        fly = ring & (r < 0.35)
        bgz = cv2.dilate(np.where(m, 0, bg_depth_u16).astype(np.float32), np.ones((5, 5), np.uint8))
        a = rng.random((H, W))
        out[fly] = (noisy * a + bgz * (1 - a))[fly]
        holes |= ring & (r > 0.8)
        # stereo occlusion band left of the object: width = fx * B * (1/z_fg - 1/z_bg)
        zf = float(np.median(z[m])) / 1000
        zb = float(np.median(bg_depth_u16[cv2.dilate(u8, np.ones((25, 25), np.uint8)).astype(bool) & ~m & (bg_depth_u16 > 0)])) / 1000 \
            if (bg_depth_u16 > 0).any() else zf * 1.1
        width = int(round(fx * baseline_m * max(1 / zf - 1 / max(zb, zf + 1e-3), 0)))
        if width > 0:
            shifted = np.zeros_like(m)
            shifted[:, :-width] = m[:, width:]
            band = np.zeros_like(m)
            for s in range(1, width + 1):
                band[:, :-s] |= m[:, s:]
            band &= ~m
            holes |= band & (rng.random((H, W)) < 0.85)
        if pr.get("transmission", 0) > 0.5:        # glass: most pixels lost, some see the floor behind
            blob = cv2.GaussianBlur(rng.standard_normal((H, W)).astype(np.float32), (0, 0), 6)
            lost = m & (blob > np.quantile(blob[m], 0.45))
            behind = m & ~lost & (rng.random((H, W)) < 0.5)
            out[behind] = bg_depth_u16[behind]
            holes |= lost
        if pr.get("metallic", 0) > 0.5 or pr.get("roughness", 1) < 0.2:
            hot = cv2.dilate((spec > 0.35).astype(np.uint8), np.ones((7, 7), np.uint8)).astype(bool) & m
            blob = cv2.GaussianBlur(rng.standard_normal((H, W)).astype(np.float32), (0, 0), 4)
            holes |= hot | (m & (blob > np.quantile(blob[m], 0.8)))
    out[holes] = 0
    return np.clip(np.round(out), 0, 65535).astype(np.uint16)
