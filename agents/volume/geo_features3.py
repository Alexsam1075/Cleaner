"""Third feature set: shape descriptors + 3D visual-hull-ish volume (silhouette carving on the floor grid).

Idea: the camera sees the object from one side; the mask tells exactly which floor-grid columns
project inside the silhouette.  For every column (u,v) we test heights h in [0, hmax] and keep
the max h whose 3D point projects inside the mask -> a visual-hull upper bound per column,
limited by the top surface from depth where depth exists.  This gives 'vh_ml' (silhouette-only)
and 'vhd_ml' (silhouette ∩ depth-consistent), which fills unseen back side for boxes/containers.
"""
from __future__ import annotations
import json, math, sys
from pathlib import Path
import cv2, numpy as np
sys.path.insert(0, "/home/cleaner/Cleaner")
sys.path.insert(0, "/home/cleaner/Cleaner/agents/volume")
from geometry.volume_parts import _unit_plane, _floor_axes, _valid_depth, clean_cloud, refine_plane, floor_noise_m
from geo_features import _inpaint_depth, ROOT, CELL
from geo_features2 import grad_filter, grid


def project(P, camera):
    z = P[:, 2]
    return np.stack([P[:, 0] / z * camera["fx"] + camera["cx"], P[:, 1] / z * camera["fy"] + camera["cy"]], 1)


def visual_hull(depth_mm, mask, plane, camera, cloud, hmax_m, cell=CELL, dh=0.004, pad_cells=6):
    """Per floor cell, max height whose 3D point projects inside the (slightly dilated) mask.

    Also 'carve' with depth: a candidate point (u,v,h) is removed if the measured depth at its
    pixel is farther than the point (i.e. we see *through* where the solid would be).
    """
    n, d = _unit_plane(plane)
    fwd, side = _floor_axes(n)
    H, W = mask.shape
    m = cv2.dilate(mask.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
    # floor grid bounding box from footprint of cloud + mask-floor hits
    uv = cloud.uv
    if len(uv) < 5:
        return {}
    u0, v0 = uv.min(0) - pad_cells * cell
    u1, v1 = uv.max(0) + pad_cells * cell
    us = np.arange(u0, u1 + cell, cell); vs = np.arange(v0, v1 + cell, cell)
    if len(us) * len(vs) > 400 * 400:
        return {}
    U, V = np.meshgrid(us, vs, indexing="ij")
    base = (U[..., None] * fwd + V[..., None] * side) - d * n  # points on the floor plane (P.n + d = 0)
    hs = np.arange(0.0, hmax_m + dh, dh)
    dep = depth_mm.astype(np.float64) / 1000.0
    valid = _valid_depth(depth_mm)
    top_sil = np.full(U.shape, -1.0)
    top_carve = np.full(U.shape, -1.0)
    for h in hs:
        P = base + h * n
        flat = P.reshape(-1, 3)
        pix = project(flat, camera)
        px = np.round(pix[:, 0]).astype(int); py = np.round(pix[:, 1]).astype(int)
        inb = (px >= 0) & (px < W) & (py >= 0) & (py < H) & (flat[:, 2] > 0.05)
        ins = np.zeros(len(flat), bool)
        ins[inb] = m[py[inb], px[inb]]
        ins = ins.reshape(U.shape)
        top_sil[ins] = h
        # carve: measured depth noticeably larger than the point depth means empty space there
        see_through = np.zeros(len(flat), bool)
        ok = inb.copy()
        ok[inb] &= valid[py[inb], px[inb]]
        see_through[ok] = dep[py[ok], px[ok]] > flat[ok, 2] + 0.012
        keep = ins & ~see_through.reshape(U.shape)
        top_carve[keep] = h
    a = cell * cell * 1e6
    out = {"vh_ml": float(np.clip(top_sil, 0, None).sum() * a), "vhc_ml": float(np.clip(top_carve, 0, None).sum() * a),
           "vh_area_m2": float((top_sil >= 0).sum() * cell * cell), "vhc_area_m2": float((top_carve >= 0).sum() * cell * cell)}
    # silhouette hull with depth top-surface as an additional cap: where depth exists in a cell, use min(depth top, hull)
    cnt, mx, mean = grid(cloud, cell)
    ij = np.floor((cloud.uv - np.array([u0, v0])) / cell).astype(int)
    ij0 = np.floor(cloud.uv / cell).astype(int); ij0 -= ij0.min(0)
    # map cloud cells to grid
    depth_top = np.full(U.shape, np.nan)
    sel = (ij[:, 0] >= 0) & (ij[:, 0] < U.shape[0]) & (ij[:, 1] >= 0) & (ij[:, 1] < U.shape[1])
    np.fmax.at(depth_top, (ij[sel, 0], ij[sel, 1]), cloud.h[sel])
    combo = np.where(np.isnan(depth_top), top_carve, np.minimum(np.nan_to_num(depth_top, nan=0), np.maximum(top_carve, 0)))
    # cells with depth but no hull: keep depth
    combo = np.where(np.isnan(depth_top) | (top_carve >= 0), combo, depth_top)
    out["vhd_ml"] = float(np.clip(np.nan_to_num(combo, nan=0), 0, None).sum() * a)
    return out


def compute_features3(depth_mm, mask, plane, camera, form):
    mask = np.asarray(mask, bool)
    refined = refine_plane(depth_mm, mask, np.asarray(plane, float), camera)
    noise = floor_noise_m(depth_mm, mask, refined, camera)
    min_h = float(np.clip(3.0 * (noise or 0.001), 0.003, 0.006))
    gf, _ = grad_filter(depth_mm, mask)
    gi, _ = _inpaint_depth(gf, mask)
    cloud = clean_cloud(gi, mask, refined, camera, erode_px=1, min_h_m=min_h)
    out = {}
    if len(cloud.h) < 20:
        return out
    hmax = float(np.quantile(cloud.h, 0.995))
    out["h995_m"] = hmax
    out.update(visual_hull(depth_mm, mask, refined, camera, cloud, hmax))
    # a taller hull for glass/metal where depth top may be missing: 1.5x
    r = visual_hull(depth_mm, mask, refined, camera, cloud, hmax * 1.5)
    out["vh15_ml"] = r.get("vh_ml", 0.0); out["vhc15_ml"] = r.get("vhc_ml", 0.0)
    return out


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
        f = compute_features3(depth, mask > 0, meta["floor_plane"], camera, at["form"])
    except Exception:
        import traceback; traceback.print_exc(); f = {"error": 1.0}
    return {"split": split, "frame": fdir.name, "object_id": at["object_id"], **f}


if __name__ == "__main__":
    import pandas as pd
    from multiprocessing import Pool
    out = Path("/home/cleaner/Cleaner/agents/volume/cache")
    for split in sys.argv[1:] or ["real", "synth"]:
        lab = json.loads((ROOT / f"labels/{split}_train.json").read_text())
        imgs = {i["id"]: i["file_name"] for i in lab["images"]}
        jobs = [(split, imgs[a["image_id"]], a) for a in lab["annotations"]]
        with Pool(16) as pool:
            rows = pool.map(_job, jobs, chunksize=2)
        pd.DataFrame(rows).to_csv(out / f"feat3_{split}.csv", index=False); print(split, len(rows), flush=True)
