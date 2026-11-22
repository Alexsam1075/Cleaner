"""Second feature set: flying-pixel filtering, grid erosion, core-cell footprints, envelope volumes."""
from __future__ import annotations
import json, math, sys
from pathlib import Path
import cv2, numpy as np
sys.path.insert(0, "/home/cleaner/Cleaner")
sys.path.insert(0, "/home/cleaner/Cleaner/agents/volume")
from geometry.volume_parts import (_unit_plane, _valid_depth, clean_cloud, refine_plane, floor_noise_m, _area_2d, complete_symmetric, heightmap, trim_isolated, _hull_ml)
from geo_features import _inpaint_depth, ROOT, CELL


def grad_filter(depth_mm, mask, thr_mm=12.0):
    d = depth_mm.astype(np.float32)
    v = _valid_depth(d)
    dmax = cv2.dilate(np.where(v, d, 0), np.ones((3, 3), np.uint8))
    dmin = -cv2.dilate(np.where(v, -d, -1e9), np.ones((3, 3), np.uint8))
    rng = dmax - dmin
    bad = v & (rng > thr_mm)
    out = depth_mm.copy(); out[bad] = 0
    return out, float((bad & mask).sum() / max(mask.sum(), 1))


def grid(cloud, cell=CELL):
    ij = np.floor(cloud.uv / cell).astype(np.int64)
    ij -= ij.min(axis=0)
    H, W = ij.max(axis=0) + 1
    cnt = np.zeros((H, W)); mx = np.zeros((H, W)); sm = np.zeros((H, W))
    np.add.at(cnt, (ij[:, 0], ij[:, 1]), 1); np.maximum.at(mx, (ij[:, 0], ij[:, 1]), cloud.h); np.add.at(sm, (ij[:, 0], ij[:, 1]), cloud.h)
    mean = np.where(cnt > 0, sm / np.maximum(cnt, 1), 0)
    return cnt, mx, mean


def grid_feats(cloud, tag, out, cell=CELL):
    if len(cloud.h) < 5:
        for k in ("colmax", "colmean", "er1_colmax", "er1_colmean", "er2_colmax", "core_colmax", "core_area_m2", "area_m2", "perim_cells", "hmean", "h90", "env_ml", "sym_ml", "sym_hull_ml"):
            out[tag + k] = 0.0
        return
    cnt, mx, mean = grid(cloud, cell)
    occ = (cnt > 0).astype(np.uint8)
    # keep the largest connected component(s) > 5% of cells
    n, lab, stats, _ = cv2.connectedComponentsWithStats(occ, connectivity=8)
    if n > 2:
        keep = np.zeros(n, bool); keep[1:] = stats[1:, cv2.CC_STAT_AREA] >= 0.05 * occ.sum(); keep[0] = False
        occ = keep[lab].astype(np.uint8)
    a = cell * cell * 1e6
    out[tag + "area_m2"] = occ.sum() * cell * cell
    out[tag + "colmax"] = float((mx * occ).sum() * a)
    out[tag + "colmean"] = float((mean * occ).sum() * a)
    er1 = cv2.erode(occ, np.ones((3, 3), np.uint8)); er2 = cv2.erode(occ, np.ones((5, 5), np.uint8))
    out[tag + "er1_colmax"] = float((mx * er1).sum() * a); out[tag + "er1_colmean"] = float((mean * er1).sum() * a)
    out[tag + "er2_colmax"] = float((mx * er2).sum() * a)
    out[tag + "perim_cells"] = float(occ.sum() - er1.sum())
    h = mx[occ > 0]
    h90 = float(np.quantile(h, 0.9)); out[tag + "h90"] = h90; out[tag + "hmean"] = float(mean[occ > 0].mean())
    core = (occ > 0) & (mean >= 0.5 * h90)
    out[tag + "core_area_m2"] = core.sum() * cell * cell
    out[tag + "core_colmax"] = float(mx[core].sum() * a)
    # envelope: fill holes of occupancy, every cell gets max(own, median of neighbourhood) -> closed top surface
    closed = cv2.morphologyEx(occ, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    filled = closed.copy()
    cnts, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(filled, cnts, -1, 1, thickness=cv2.FILLED)
    # top surface in filled cells: dilate max height (5x5) where missing
    top = cv2.dilate(mx.astype(np.float32), np.ones((7, 7), np.uint8))
    env = np.where(occ > 0, mx, top) * filled
    out[tag + "env_ml"] = float(env.sum() * a)
    hm = complete_symmetric(trim_isolated(heightmap(cloud, cell), cell), cell)
    out[tag + "sym_ml"] = float(hm.h.sum() * a)
    out[tag + "sym_hull_ml"] = float(_hull_ml(hm))


def compute_features2(depth_mm, mask, plane, camera, form):
    mask = np.asarray(mask, bool)
    refined = refine_plane(depth_mm, mask, np.asarray(plane, float), camera)
    noise = floor_noise_m(depth_mm, mask, refined, camera)
    min_h = float(np.clip(3.0 * (noise or 0.001), 0.003, 0.006))
    out = {}
    inp, info = _inpaint_depth(depth_mm, mask)
    gf, bad = grad_filter(depth_mm, mask); out["flying_frac"] = bad
    gf_inp, _ = _inpaint_depth(gf, mask)
    for tag, dep, er in (("g_", gf, 1), ("gi_", gf_inp, 1), ("i_", inp, 2), ("i0_", inp, 0)):
        cloud = clean_cloud(dep, mask, refined, camera, erode_px=er, min_h_m=min_h)
        grid_feats(cloud, tag, out)
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
        f = compute_features2(depth, mask > 0, meta["floor_plane"], camera, at["form"])
    except Exception as e:
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
        pd.DataFrame(rows).to_csv(out / f"feat2_{split}.csv", index=False); print(split, len(rows), flush=True)
