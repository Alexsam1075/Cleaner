"""Rich per-object geometric candidates for volume estimation (RGB-D + floor plane).

Builds on geometry/volume_parts.py (clean cloud, heightmap, symmetric completion) and adds
mean-height columns, depth in-painting inside the mask, axis-aligned primitive fits, and
silhouette-based size cues. Everything is a plain dict of floats so it can be cached.
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, "/home/cleaner/Cleaner")
from geometry.volume_parts import (  # noqa: E402
    ObjectCloud, _area_2d, _backproject, _floor_axes, _hull_ml, _unit_plane, _valid_depth,
    clean_cloud, complete_symmetric, estimate_volume, floor_noise_m, heightmap, refine_plane,
    trim_isolated,
)

CELL = 0.004


def _inpaint_depth(depth_mm: np.ndarray, mask: np.ndarray, radius_px: int = 40) -> tuple[np.ndarray, dict]:
    """Fill zero-depth pixels inside the mask from the nearest valid pixels inside the mask."""
    m = mask.astype(np.uint8)
    ys, xs = np.nonzero(m)
    out = depth_mm.copy()
    info = {"hole_frac": 0.0, "hole_inner_frac": 0.0, "hole_edge_frac": 0.0, "inp_frac": 0.0}
    if len(ys) == 0:
        return out, info
    y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
    sub = depth_mm[y0:y1, x0:x1]
    msub = m[y0:y1, x0:x1] > 0
    valid = _valid_depth(sub) & msub
    hole = msub & ~valid
    er = cv2.erode(m[y0:y1, x0:x1], np.ones((7, 7), np.uint8)) > 0
    info["hole_frac"] = float(hole.sum() / max(msub.sum(), 1))
    info["hole_inner_frac"] = float((hole & er).sum() / max(er.sum(), 1))
    info["hole_edge_frac"] = float((hole & ~er).sum() / max((msub & ~er).sum(), 1))
    if hole.sum() == 0 or valid.sum() < 20:
        return out, info
    # nearest valid pixel (inside mask) via distance transform with labels
    dist, lab = cv2.distanceTransformWithLabels((~valid).astype(np.uint8), cv2.DIST_L2, 5,
                                                labelType=cv2.DIST_LABEL_PIXEL)
    vy, vx = np.nonzero(valid)
    lbl_at_valid = lab[vy, vx]
    lut = np.zeros(int(lab.max()) + 1, dtype=np.float64)
    lut[lbl_at_valid] = sub[vy, vx]
    fill = hole & (dist <= radius_px)
    filled = sub.astype(np.float64).copy()
    filled[fill] = lut[lab[fill]]
    # smooth the filled area a little (median of the fill) to avoid speckle
    out[y0:y1, x0:x1] = filled.astype(depth_mm.dtype)
    info["inp_frac"] = float(fill.sum() / max(msub.sum(), 1))
    return out, info


def _cells(cloud: ObjectCloud, cell: float = CELL):
    if len(cloud.h) == 0:
        return None
    ij = np.floor(cloud.uv / cell).astype(np.int64)
    keys, inv = np.unique(ij, axis=0, return_inverse=True)
    inv = inv.ravel()
    cnt = np.bincount(inv)
    mx = np.zeros(len(keys))
    np.maximum.at(mx, inv, cloud.h)
    mean = np.bincount(inv, cloud.h) / cnt
    return keys, mx, mean, cnt


def _axis_extents(uv: np.ndarray, q=(0.01, 0.99)) -> tuple[float, float, np.ndarray]:
    c = uv.mean(axis=0)
    w, v = np.linalg.eigh(np.cov((uv - c).T)) if len(uv) >= 3 else (np.array([0, 1]), np.eye(2))
    ax = v[:, int(np.argmax(w))]
    perp = np.array([-ax[1], ax[0]])
    s, t = (uv - c) @ ax, (uv - c) @ perp
    L = float(np.quantile(s, q[1]) - np.quantile(s, q[0]))
    W = float(np.quantile(t, q[1]) - np.quantile(t, q[0]))
    return max(L, 1e-4), max(W, 1e-4), ax


def _silhouette(mask: np.ndarray, plane, camera, z_med: float) -> dict:
    n, d = _unit_plane(plane)
    ys, xs = np.nonzero(mask)
    out = {}
    out["mask_px"] = float(len(xs))
    out["bbox_w_px"] = float(xs.max() - xs.min() + 1)
    out["bbox_h_px"] = float(ys.max() - ys.min() + 1)
    sx = z_med / float(camera["fx"])
    sy = z_med / float(camera["fy"])
    out["mask_area_at_z_m2"] = out["mask_px"] * sx * sy
    out["bbox_w_m"] = out["bbox_w_px"] * sx
    out["bbox_h_m"] = out["bbox_h_px"] * sy
    # camera tilt vs floor: angle between optical axis and floor normal
    cos_t = abs(n[2])
    out["cos_tilt"] = float(cos_t)
    # vertical extent of the silhouette converted to a height guess assuming
    # the top edge sits above the bottom contact row: h ~ bbox_h_m * cos(tilt) ... crude
    out["sil_height_m"] = out["bbox_h_m"] * float(np.sqrt(max(1 - cos_t ** 2, 1e-6)))
    # PCA extents of the mask in pixels (minAreaRect)
    pts = np.column_stack([xs, ys]).astype(np.float32)
    (_, (rw, rh), _) = cv2.minAreaRect(pts)
    out["rect_long_m"] = max(rw, rh) * sx
    out["rect_short_m"] = min(rw, rh) * sx
    hull = cv2.convexHull(pts)
    out["mask_solidity"] = float(len(xs) / max(cv2.contourArea(hull), 1.0))
    return out


def compute_features(depth_mm: np.ndarray, mask: np.ndarray, plane, camera: dict, form: str) -> dict:
    mask = np.asarray(mask, dtype=bool)
    plane = np.asarray(plane, dtype=np.float64)
    refined = refine_plane(depth_mm, mask, plane, camera)
    out = estimate_volume(depth_mm, mask, refined, camera, form)
    out = {k: (float(v) if not isinstance(v, bool) else float(v)) for k, v in out.items()}
    n, d = _unit_plane(refined)
    valid = mask & _valid_depth(depth_mm)
    z_med = float(np.median(depth_mm[valid])) / 1000.0 if valid.sum() else 0.5
    out["z_med_m"] = z_med
    out.update(_silhouette(mask, refined, camera, z_med))
    noise = floor_noise_m(depth_mm, mask, refined, camera)
    out["floor_noise_m"] = float(noise if noise is not None else 0.001)
    min_h = float(out["min_h_m"])

    # --- in-painted depth
    inp, info = _inpaint_depth(depth_mm, mask)
    out.update(info)

    for tag, dep in (("", depth_mm), ("inp_", inp)):
        cloud = clean_cloud(dep, mask, refined, camera, erode_px=2, min_h_m=min_h)
        c = _cells(cloud)
        if c is None or len(c[0]) < 5:
            for k in ("cells_m2", "colmax_ml", "colmean_ml", "h_mean_m", "h_med_m", "h_p90_m", "h_max_m",
                      "h_std_m", "L_m", "W_m", "box_ml", "ell_ml", "cyl_ml", "hull2d_m2", "sym_cells_m2",
                      "sym_colmax_ml", "sym_hull_ml", "sym_box_ml", "n_pts"):
                out[tag + k] = 0.0
            continue
        keys, mx, mean, cnt = c
        A = len(keys) * CELL * CELL
        out[tag + "n_pts"] = float(len(cloud.h))
        out[tag + "cells_m2"] = A
        out[tag + "colmax_ml"] = float(mx.sum() * CELL * CELL * 1e6)
        out[tag + "colmean_ml"] = float(mean.sum() * CELL * CELL * 1e6)
        out[tag + "h_mean_m"] = float(mean.mean())
        out[tag + "h_med_m"] = float(np.median(mx))
        out[tag + "h_p90_m"] = float(np.quantile(mx, 0.9))
        out[tag + "h_max_m"] = float(np.quantile(mx, 0.995))
        out[tag + "h_std_m"] = float(mx.std())
        uvc = (keys + 0.5) * CELL
        L, W, _ = _axis_extents(uvc)
        H = float(np.quantile(mx, 0.95))
        out[tag + "L_m"], out[tag + "W_m"] = L, W
        out[tag + "box_ml"] = L * W * H * 1e6
        out[tag + "ell_ml"] = math.pi / 6 * L * W * H * 1e6
        out[tag + "cyl_ml"] = math.pi / 4 * W * H * L * 1e6
        out[tag + "hull2d_m2"] = _area_2d(uvc)
        hm = complete_symmetric(trim_isolated(heightmap(cloud, CELL), CELL), CELL)
        out[tag + "sym_cells_m2"] = len(hm.h) * CELL * CELL
        out[tag + "sym_colmax_ml"] = float(hm.h.sum() * CELL * CELL * 1e6)
        out[tag + "sym_hull_ml"] = float(_hull_ml(hm))
        Ls, Ws, _ = _axis_extents(hm.uv)
        out[tag + "sym_box_ml"] = Ls * Ws * H * 1e6
    return out


# ---------------------------------------------------------------- cache builder

ROOT = Path("/home/cleaner/Cleaner/datasets/raw")


def _job(args):
    split, file_name, ann = args
    camera = json.loads((ROOT / "camera.json").read_text())
    fdir = ROOT / Path(file_name).parent
    depth = cv2.imread(str(fdir / "depth.png"), cv2.IMREAD_UNCHANGED)
    meta = json.loads((fdir / "meta.json").read_text())
    mask = np.zeros(depth.shape, np.uint8)
    for poly in ann["segmentation"]:
        cv2.fillPoly(mask, [np.asarray(poly, np.float32).reshape(-1, 2).round().astype(np.int32)], 1)
    at = ann["attributes"]
    try:
        f = compute_features(depth, mask > 0, np.asarray(meta["floor_plane"]), camera, at["form"])
    except Exception as e:  # keep the row, mark failure
        f = {"error": 1.0}
        print("ERR", file_name, e, flush=True)
    row = {"split": split, "frame": fdir.name, "object_id": at["object_id"], "form": at["form"],
           "material": at["material"], "volume_ml": float(at["volume_ml"]), "mass_g": float(at["mass_g"])}
    row.update(f)
    return row


def build(split: str, workers: int = 16):
    import pandas as pd
    from multiprocessing import Pool
    lab = json.loads((ROOT / f"labels/{split}_train.json").read_text())
    imgs = {i["id"]: i["file_name"] for i in lab["images"]}
    jobs = [(split, imgs[a["image_id"]], a) for a in lab["annotations"]]
    with Pool(workers) as pool:
        rows = pool.map(_job, jobs, chunksize=2)
    return pd.DataFrame(rows)


if __name__ == "__main__":
    import pandas as pd
    out = Path("/home/cleaner/Cleaner/agents/volume/cache")
    out.mkdir(parents=True, exist_ok=True)
    for split in sys.argv[1:] or ["real", "synth"]:
        df = build(split)
        df.to_csv(out / f"feat_{split}.csv", index=False)
        print(split, df.shape, flush=True)
