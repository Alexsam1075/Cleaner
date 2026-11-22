"""Оцифровка 40 предметов из реальных RGB-D кадров: карта высот над полом (2.5D) + цвет.
Для каждого object_id берём кадр с самой большой маской и малым числом дыр глубины,
переводим пиксели маски в мировые координаты пола, строим сетку высот 3 мм, заполняем дыры,
сохраняем меш (вершины, грани, цвета) в sim/assets/objects/<id>.npz и атрибуты в objects.json."""
import json
from collections import defaultdict

import cv2
import numpy as np

from geom import DATA, ROOT, cam_to_world, frame_dir, meta, unproject

OUT = ROOT / "sim" / "assets" / "objects"
CELL = 0.003


def polygon_mask(seg, h, w):
    m = np.zeros((h, w), np.uint8)
    for s in seg:
        cv2.fillPoly(m, [np.array(s, np.float32).reshape(-1, 2).round().astype(np.int32)], 1)
    return m > 0


def candidates():
    by_obj = defaultdict(list)
    for name in ("real_train", "synth_train"):
        d = json.loads((DATA / "labels" / f"{name}.json").read_text(encoding="utf-8"))
        ims = {im["id"]: im for im in d["images"]}
        for a in d["annotations"]:
            im = ims[a["image_id"]]
            fid = im["file_name"].split("/")[2]
            by_obj[a["attributes"]["object_id"]].append((name, fid, a, im))
    return by_obj


def build(fid, ann, im):
    fd = frame_dir(fid)
    color = cv2.imread(str(fd / "color.png"))
    depth = cv2.imread(str(fd / "depth.png"), cv2.IMREAD_UNCHANGED)
    plane = meta(fid)["floor_plane"]
    mask = polygon_mask(ann["segmentation"], im["height"], im["width"])
    mask_in = cv2.erode(mask.astype(np.uint8), np.ones((5, 5), np.uint8)) > 0
    ok = mask_in & (depth > 0)
    if ok.sum() < 200:
        return None
    Pw = cam_to_world(unproject(depth), plane)
    pts, cols = Pw[ok], color[ok][:, ::-1].astype(np.float32) / 255.0
    pts = pts[pts[:, 2] > -0.01]
    cols = cols[: len(pts)]
    lo = np.percentile(pts[:, :2], 0.5, axis=0) - 4 * CELL
    hi = np.percentile(pts[:, :2], 99.5, axis=0) + 4 * CELL
    keep = np.all((pts[:, :2] >= lo) & (pts[:, :2] <= hi), 1)
    pts, cols = pts[keep], cols[keep]
    nx, ny = np.ceil((hi - lo) / CELL).astype(int) + 1
    ix = ((pts[:, 0] - lo[0]) / CELL).astype(int)
    iy = ((pts[:, 1] - lo[1]) / CELL).astype(int)
    hmap = np.full((ny, nx), np.nan, np.float32)
    cmap = np.zeros((ny, nx, 3), np.float32)
    cnt = np.zeros((ny, nx), np.float32)
    order = np.argsort(pts[:, 2])
    hmap[iy[order], ix[order]] = np.clip(pts[order, 2], 0, None)
    np.add.at(cmap, (iy, ix), cols)
    np.add.at(cnt, (iy, ix), 1)
    have = ~np.isnan(hmap)
    foot = cv2.morphologyEx(have.astype(np.uint8), cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8)) > 0
    hole = (foot & ~have).astype(np.uint8)
    h8 = np.nan_to_num(hmap)
    scale = max(h8.max(), 1e-3)
    hfill = cv2.inpaint((h8 / scale * 255).astype(np.uint8), hole, 3, cv2.INPAINT_TELEA).astype(np.float32) / 255 * scale
    hmap = np.where(have, h8, np.where(foot, hfill, 0.0))
    hmap = cv2.medianBlur(hmap.astype(np.float32), 3)
    hmap[~foot] = 0
    cmean = cmap / np.maximum(cnt, 1)[..., None]
    known = (cnt > 0).astype(np.uint8)
    c8 = (cmean * 255).astype(np.uint8)
    c8 = cv2.inpaint(c8, ((foot > 0) & (known == 0)).astype(np.uint8), 3, cv2.INPAINT_TELEA)
    cmean = c8.astype(np.float32) / 255
    return hmap, cmean, lo, foot


def heightfield_mesh(hmap, cmean, lo):
    ny, nx = hmap.shape
    gy, gx = np.mgrid[0:ny, 0:nx]
    cx = lo[0] + gx * CELL
    cy = lo[1] + gy * CELL
    verts = np.stack([cx, cy, hmap], -1).reshape(-1, 3)
    centre = np.array([verts[:, 0].mean(), verts[:, 1].mean(), 0.0])
    verts = verts - centre
    idx = np.arange(ny * nx).reshape(ny, nx)
    a, b, c, d = idx[:-1, :-1], idx[:-1, 1:], idx[1:, 1:], idx[1:, :-1]
    any_up = (hmap[:-1, :-1] > 0.002) | (hmap[:-1, 1:] > 0.002) | (hmap[1:, 1:] > 0.002) | (hmap[1:, :-1] > 0.002)
    quads = np.stack([a, b, c, d], -1)[any_up]
    faces = np.concatenate([quads[:, [0, 1, 2]], quads[:, [0, 2, 3]]])
    used = np.unique(faces)
    remap = -np.ones(len(verts), np.int64)
    remap[used] = np.arange(len(used))
    return verts[used].astype(np.float32), remap[faces].astype(np.int32), cmean.reshape(-1, 3)[used].astype(np.float32), centre


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    cats = {}
    for oid, cands in sorted(candidates().items()):
        real = [c for c in cands if c[0] == "real_train"]
        pool = real or cands
        pool = sorted(pool, key=lambda c: -c[2]["area"])
        res = None
        for name, fid, ann, im in pool[:6]:
            res = build(fid, ann, im)
            if res is not None:
                break
        if res is None:
            print(oid, "нет глубины — пропуск")
            continue
        hmap, cmean, lo, foot = res
        v, f, c, centre = heightfield_mesh(hmap, cmean, lo)
        np.savez_compressed(OUT / f"{oid}.npz", verts=v, faces=f, colors=c, centre=centre)
        at = ann["attributes"]
        vol = float(hmap.sum() * CELL * CELL * 1e6)
        cats[oid] = {"form": at["form"], "material": at["material"], "volume_ml": at["volume_ml"], "mass_g": at["mass_g"],
                     "take": at["take"], "source_frame": fid, "source_xy": [round(float(centre[0]), 4), round(float(centre[1]), 4)], "mesh_volume_ml": round(vol, 1),
                     "size_m": [round(float(foot.any(0).sum() * CELL), 3), round(float(foot.any(1).sum() * CELL), 3), round(float(hmap.max()), 3)]}
        print(f"{oid} {at['form']:5s} {at['material']:8s} кадр {fid}  меш {vol:7.0f} мл / метка {at['volume_ml']:6.0f} мл  размер {cats[oid]['size_m']}")
    (ROOT / "sim" / "assets" / "objects.json").write_text(json.dumps(cats, indent=1, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
