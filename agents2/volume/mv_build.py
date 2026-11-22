"""Build per-frame heightmaps (floor-aligned, 4 mm) for all real-train objects and cache them; then
fuse heightmaps of consecutive frames of the same object (series = same object_id, frame gap <= 3, as the
pipeline's link_series would produce) by aligning on the floor (centroid + main axis, with 180° ambiguity
resolved by height-profile correlation) and taking per-cell max/median. Outputs mv_rows.csv with
mv_* candidate volumes per frame (each frame fused with its neighbours in the series)."""
import json
import sys
from multiprocessing import Pool
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

sys.path.insert(0, "/home/cleaner/Cleaner")
sys.path.insert(0, "/home/cleaner/Cleaner/agents/volume")
from geometry.volume_parts import clean_cloud, refine_plane, floor_noise_m, heightmap, trim_isolated, complete_symmetric, _hull_ml  # noqa
from geo_features import _inpaint_depth  # noqa

ROOT = Path("/home/cleaner/Cleaner/datasets/raw")
HERE = Path("/home/cleaner/Cleaner/agents2/volume")
CELL = 0.004


def _job(args):
    file_name, ann = args
    camera = json.loads((ROOT / "camera.json").read_text())
    fdir = ROOT / Path(file_name).parent
    depth = cv2.imread(str(fdir / "depth.png"), cv2.IMREAD_UNCHANGED)
    meta = json.loads((fdir / "meta.json").read_text())
    mask = np.zeros(depth.shape, np.uint8)
    for poly in ann["segmentation"]:
        cv2.fillPoly(mask, [np.asarray(poly, np.float32).reshape(-1, 2).round().astype(np.int32)], 1)
    mask = mask > 0
    plane = refine_plane(depth, mask, np.asarray(meta["floor_plane"], float), camera)
    noise = floor_noise_m(depth, mask, plane, camera)
    min_h = float(np.clip(3.0 * (noise or 0.001), 0.003, 0.006))
    inp, _ = _inpaint_depth(depth, mask)
    cloud = clean_cloud(inp, mask, plane, camera, erode_px=1, min_h_m=min_h)
    hm = trim_isolated(heightmap(cloud, CELL), CELL)
    at = ann["attributes"]
    return {"frame": fdir.name, "oid": at["object_id"], "form": at["form"], "uv": hm.uv.astype(np.float32), "h": hm.h.astype(np.float32)}


def grid_of(uv, h, cell=CELL):
    ij = np.floor(uv / cell).astype(np.int64)
    o = ij.min(0)
    ij = ij - o
    G = np.zeros(tuple(ij.max(0) + 1), np.float32)
    np.maximum.at(G, (ij[:, 0], ij[:, 1]), h)
    return G, o


def canon(uv, h):
    """Centre on weighted centroid, rotate so the main axis is x. Returns (xy, h, R, c)."""
    w = h / h.sum()
    c = (uv * w[:, None]).sum(0)
    d = uv - c
    cov = (d * w[:, None]).T @ d
    ev, V = np.linalg.eigh(cov)
    ax = V[:, 1]
    R = np.array([[ax[0], ax[1]], [-ax[1], ax[0]]])
    return d @ R.T, h, R, c


def rasterize(xy, h, cell, shape, origin):
    ij = np.floor((xy - origin) / cell).astype(np.int64)
    ok = (ij[:, 0] >= 0) & (ij[:, 1] >= 0) & (ij[:, 0] < shape[0]) & (ij[:, 1] < shape[1])
    G = np.full(shape, np.nan, np.float32)
    cnt = np.zeros(shape, np.int32)
    np.add.at(cnt, (ij[ok, 0], ij[ok, 1]), 1)
    Gm = np.zeros(shape, np.float32)
    np.maximum.at(Gm, (ij[ok, 0], ij[ok, 1]), h[ok])
    G[cnt > 0] = Gm[cnt > 0]
    return G


def fuse(frames, mode="max"):
    """frames: list of (uv,h). Canonicalise each; align others to the first by trying 4 flips (x, y)
    and choosing the max correlation of height rasters; fuse per cell."""
    cans = [canon(uv, h) for uv, h in frames]
    ext = max(np.abs(c[0]).max() for c in cans) + 3 * CELL
    n = int(np.ceil(2 * ext / CELL)) + 1
    origin = np.array([-ext, -ext])
    shape = (n, n)
    R0 = rasterize(cans[0][0], cans[0][1], CELL, shape, origin)
    rasters = [R0]
    for xy, h, _, _ in cans[1:]:
        best = None
        for sx in (1, -1):
            for sy in (1, -1):
                G = rasterize(xy * np.array([sx, sy]), h, CELL, shape, origin)
                a = np.nan_to_num(R0)
                b = np.nan_to_num(G)
                s = float((a * b).sum() / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))
                if best is None or s > best[0]:
                    best = (s, G)
        rasters.append(best[1])
    S = np.stack(rasters)
    cnt = np.sum(~np.isnan(S), 0)
    with np.errstate(all="ignore"):
        mx = np.nanmax(S, 0)
        md = np.nanmedian(S, 0)
        mn = np.nanmean(S, 0)
    a = CELL * CELL * 1e6
    occ_any = cnt > 0
    occ_all = cnt == len(rasters)
    out = {"mv_n": len(rasters),
           "mv_max_ml": float(np.nansum(np.where(occ_any, mx, 0)) * a),
           "mv_med_ml": float(np.nansum(np.where(occ_any, md, 0)) * a),
           "mv_mean_ml": float(np.nansum(np.where(occ_any, mn, 0)) * a),
           "mv_union_area_m2": float(occ_any.sum() * CELL * CELL),
           "mv_inter_area_m2": float(occ_all.sum() * CELL * CELL),
           "mv_max_inter_ml": float(np.nansum(np.where(occ_all, mx, 0)) * a)}
    # convex hull volume of the fused top surface
    ii, jj = np.nonzero(occ_any)
    pts_uv = np.c_[ii, jj] * CELL
    from geometry.volume_parts import ObjectCloud
    hm = ObjectCloud(uv=pts_uv, h=mx[ii, jj].astype(np.float64), footprint_uv=np.zeros((0, 2)), valid_frac=1.0, truncated=False)
    out["mv_hull_ml"] = _hull_ml(hm)
    sym = complete_symmetric(hm, CELL)
    out["mv_sym_ml"] = float(sym.h.sum() * a)
    return out


def main():
    cache = HERE / "cache_hm.npz"
    lab = json.loads((ROOT / "labels/real_train.json").read_text())
    imgs = {i["id"]: i["file_name"] for i in lab["images"]}
    jobs = [(imgs[a["image_id"]], a) for a in lab["annotations"]]
    if cache.exists():
        z = np.load(cache, allow_pickle=True)
        hms = list(z["hms"])
    else:
        with Pool(12) as pool:
            hms = pool.map(_job, jobs, chunksize=2)
        np.savez_compressed(cache, hms=np.array(hms, dtype=object))
    print("heightmaps", len(hms), flush=True)
    fnum = np.array([int(h["frame"].split("_")[1]) for h in hms])
    oids = np.array([h["oid"] for h in hms])
    rows = []
    for gap in (3, 1000):
        for i, h in enumerate(hms):
            same = np.where(oids == oids[i])[0]
            # series: chain of frames of same object with consecutive numbering gap <= gap
            ser = [i]
            changed = True
            while changed:
                changed = False
                for j in same:
                    if j in ser:
                        continue
                    if any(abs(fnum[j] - fnum[k]) <= gap for k in ser):
                        ser.append(j)
                        changed = True
            ser = [i] + [j for j in sorted(ser, key=lambda j: fnum[j]) if j != i]
            frames = [(hms[j]["uv"], hms[j]["h"]) for j in ser if len(hms[j]["h"]) > 10]
            if len(frames) == 0:
                continue
            try:
                o = fuse(frames)
            except Exception as e:  # noqa
                print("ERR", h["frame"], e, flush=True)
                continue
            o.update(frame=h["frame"], oid=oids[i], gap=gap, single_ml=float(h["h"].sum() * CELL * CELL * 1e6))
            rows.append(o)
        print("gap", gap, "done", flush=True)
    pd.DataFrame(rows).to_csv(HERE / "mv_rows.csv", index=False)


if __name__ == "__main__":
    main()
