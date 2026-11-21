"""Generate dataset-like RGB-D frames: procedural trash rendered into real empty floor frames.

  python -m polygon.generate --data datasets/raw --out polygon_data --n 200 --seed 0

Writes <out>/train/g_XXXXX/{color.png, depth.png, meta.json} and <out>/labels/gen_train.json in the
dataset's COCO format (attributes: form, material, volume_ml, mass_g, take, object_id, kind).
Only real/train frames without annotations are used as backgrounds; the test split is never read.
"""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import cv2
import numpy as np
import torch

from .render import DEV, Background, Camera, render_over_frame
from .sensor import floor_sigma_mm, simulate_depth
from .trash import random_trash


def empty_frames(data: Path) -> list[str]:
    lab = json.loads((data / "labels/real_train.json").read_text())
    used = {a["image_id"] for a in lab["annotations"]}
    return [Path(i["file_name"]).parent.name for i in lab["images"] if i["id"] not in used]


def place(objs, bg: Background, cam: Camera, rng, tries=200):
    """Floor positions where each object's footprint is visible free floor and objects do not overlap."""
    H, W = cam.H, cam.W
    floor = bg.floor.cpu().numpy()
    ys, xs = np.nonzero(floor[int(H * 0.3):int(H * 0.95), int(W * 0.1):int(W * 0.9)])
    ys, xs = ys + int(H * 0.3), xs + int(W * 0.1)
    placed = []
    for ob in objs:
        for _ in range(tries):
            j = int(rng.integers(len(xs)))
            o, d, _ = cam.rays(torch.tensor([[float(xs[j]), float(ys[j])]], device=DEV))
            t = -o[0, 2] / d[0, 2]
            x, y = (o[0] + d[0] * t)[:2].tolist()
            if any(math.hypot(x - px, y - py) < ob.radius + pr + 0.02 for px, py, pr, _ in placed):
                continue
            ang = np.linspace(0, 2 * np.pi, 16, endpoint=False)
            ring = torch.tensor(np.c_[x + ob.radius * np.cos(ang), y + ob.radius * np.sin(ang), np.zeros(16)],
                                device=DEV, dtype=torch.float32)
            pix, z = cam.project(ring)
            pix = pix.cpu().numpy().round().astype(int)
            ok = (pix[:, 0] >= 2) & (pix[:, 0] < W - 2) & (pix[:, 1] >= 2) & (pix[:, 1] < H - 2)
            if not ok.all() or not floor[pix[:, 1], pix[:, 0]].mean() > 0.9:
                continue
            placed.append((x, y, ob.radius, float(rng.uniform(0, 2 * np.pi))))
            break
        else:
            return None
    return placed


def mask_to_ann(mask: np.ndarray):
    cs, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    segs = [c.reshape(-1).astype(float).tolist() for c in cs if len(c) >= 3]
    ys, xs = np.nonzero(mask)
    return segs, [float(xs.min()), float(ys.min()), float(xs.max() - xs.min() + 1), float(ys.max() - ys.min() + 1)], float(mask.sum())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="datasets/raw", type=Path)
    ap.add_argument("--out", default="polygon_data", type=Path)
    ap.add_argument("--n", type=int, default=50)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-objects", type=int, default=3)
    ap.add_argument("--ss", type=int, default=2)
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)
    cam_json = json.loads((args.data / "camera.json").read_text())
    frames = empty_frames(args.data)
    out_train = args.out / "train"
    out_train.mkdir(parents=True, exist_ok=True)
    coco = {"categories": [{"id": 1, "name": "bulk"}, {"id": 2, "name": "solid"}], "images": [], "annotations": []}
    ann_id = 1
    t0 = time.time()
    for i in range(args.n):
        fid = frames[int(rng.integers(len(frames)))]
        fdir = args.data / "real/train" / fid
        meta = json.loads((fdir / "meta.json").read_text())
        rgb = cv2.imread(str(fdir / "color.png"))
        dep = cv2.imread(str(fdir / "depth.png"), cv2.IMREAD_UNCHANGED)
        cam = Camera.from_plane(cam_json, meta["floor_plane"])
        bg = Background.from_frame(rgb, dep, cam)
        objs = [random_trash(rng) for _ in range(int(rng.integers(1, args.max_objects + 1)))]
        pos = place(objs, bg, cam, rng)
        if pos is None:
            continue
        parts, props = [], {}
        for k, (ob, (x, y, _, yaw)) in enumerate(zip(objs, pos)):
            parts += ob.placed(x, y, yaw, k)
            m = ob.parts[0].material
            props[k] = {"transmission": m.transmission, "metallic": m.metallic, "roughness": m.roughness}
        res = render_over_frame(parts, cam, bg, ss=args.ss, seed=args.seed * 100000 + i)
        sigma = floor_sigma_mm(dep, bg.floor.cpu().numpy())
        depth = simulate_depth(res["depth_true"], dep, res["ids"], res["spec"], props, cam.fx, sigma, rng)
        gid = f"g_{args.seed:02d}{i:05d}"
        d = out_train / gid
        d.mkdir(exist_ok=True)
        cv2.imwrite(str(d / "color.png"), res["color"])
        cv2.imwrite(str(d / "depth.png"), depth)
        (d / "meta.json").write_text(json.dumps({**meta, "id": gid, "synthetic": True, "renderer": "polygon",
                                                 "background_frame": fid}, ensure_ascii=False))
        img_id = len(coco["images"]) + 1
        coco["images"].append({"id": img_id, "file_name": f"train/{gid}/color.png", "width": cam.W, "height": cam.H})
        for k, ob in enumerate(objs):
            m = res["ids"] == k
            if m.sum() < 30:
                continue
            segs, bbox, area = mask_to_ann(m)
            coco["annotations"].append({
                "id": ann_id, "image_id": img_id, "bbox": bbox, "area": area, "iscrowd": 0,
                "category_id": 1 if ob.form == "bulk" else 2, "segmentation": segs,
                "attributes": {"form": ob.form, "material": ob.material, "volume_ml": round(ob.volume_ml, 1),
                               "mass_g": round(ob.mass_g, 1), "take": ob.take, "kind": ob.kind,
                               "filled": ob.filled, "object_id": f"{gid}_{k}"}})
            ann_id += 1
        if (i + 1) % 10 == 0:
            print(f"{i + 1}/{args.n} frames, {time.time() - t0:.0f}s", flush=True)
    (args.out / "labels").mkdir(exist_ok=True)
    (args.out / "labels/gen_train.json").write_text(json.dumps(coco))
    print(f"done: {len(coco['images'])} frames, {len(coco['annotations'])} objects, {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
