"""Real-time dataset generator: procedural trash rendered into real empty floor frames.

  python -m polygon.generate_fast --out polygon_data --n 2000 --seed 0

Same output format as polygon.generate (dataset COCO json + color/depth/meta per frame).
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

from .fast_scene import random_trash
from .fastrt import FastCam, Scene, background, render, to_u8
from .generate import empty_frames, mask_to_ann
from .sensor import floor_sigma_mm, simulate_depth


def place(trash, cam: FastCam, floor_np, rng, tries=200):
    H, W = cam.H, cam.W
    ys, xs = np.nonzero(floor_np[int(H * 0.3):int(H * 0.95), int(W * 0.1):int(W * 0.9)])
    ys, xs = ys + int(H * 0.3), xs + int(W * 0.1)
    out = []
    for t in trash:
        for _ in range(tries):
            j = int(rng.integers(len(xs)))
            o, d = cam.ray(xs[j], ys[j])
            p = o + d * (-o[2] / d[2])
            x, y = p[0], p[1]
            if any(math.hypot(x - a, y - b) < t.radius + r + 0.02 for a, b, r, _ in out):
                continue
            ang = np.linspace(0, 2 * np.pi, 16, endpoint=False)
            ok = True
            for a in ang:
                (px, py), z = cam.project((x + t.radius * math.cos(a), y + t.radius * math.sin(a), 0))
                if z <= 0 or not (2 <= px < W - 2 and 2 <= py < H - 2) or not floor_np[int(py), int(px)]:
                    ok = False
                    break
            if ok:
                out.append((x, y, t.radius, float(rng.uniform(0, 2 * np.pi))))
                break
        else:
            return None
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="datasets/raw", type=Path)
    ap.add_argument("--out", default="polygon_data", type=Path)
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-objects", type=int, default=3)
    ap.add_argument("--spp", type=int, default=4)
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)
    cam_json = json.loads((args.data / "camera.json").read_text())
    frames = empty_frames(args.data)
    bgs = {}
    (args.out / "train").mkdir(parents=True, exist_ok=True)
    coco = {"categories": [{"id": 1, "name": "bulk"}, {"id": 2, "name": "solid"}], "images": [], "annotations": []}
    aid, t_render, t0 = 1, 0.0, time.time()
    for i in range(args.n):
        fid = frames[int(rng.integers(len(frames)))]
        fdir = args.data / "real/train" / fid
        meta = json.loads((fdir / "meta.json").read_text())
        cam = FastCam.from_plane(cam_json, meta["floor_plane"])
        if fid not in bgs:
            rgb = cv2.imread(str(fdir / "color.png"))
            dep = cv2.imread(str(fdir / "depth.png"), cv2.IMREAD_UNCHANGED)
            bg = background(rgb, dep, cam)
            bgs[fid] = (bg, dep, floor_sigma_mm(dep, bg["floor_np"]))
        bg, dep, sigma = bgs[fid]
        trash = [random_trash(rng) for _ in range(int(rng.integers(1, args.max_objects + 1)))]
        pos = place(trash, cam, bg["floor_np"], rng)
        if pos is None:
            continue
        sc = Scene()
        props = {}
        for k, (t, (x, y, _, yaw)) in enumerate(zip(trash, pos)):
            t.place(sc, x, y, yaw, k)
            props[k] = {"transmission": t.mat.transmission, "metallic": t.mat.metallic, "roughness": t.mat.roughness}
        torch.cuda.synchronize()
        ts = time.time()
        r = render(sc, cam, bg=bg, spp=args.spp, seed=args.seed * 100000 + i)
        rgb_lin = r["rgb"]
        cov = r["cov"]
        noise = torch.randn_like(rgb_lin) * bg["noise"] * 0.8 * (cov[..., None] > 0)
        torch.cuda.synchronize()
        t_render += time.time() - ts
        img = to_u8(rgb_lin)
        img = np.clip(img.astype(np.float32) + noise.cpu().numpy()[..., ::-1] * 255, 0, 255).astype(np.uint8)
        c = cv2.dilate((cov.cpu().numpy() > 0).astype(np.uint8), np.ones((5, 5), np.uint8)).astype(bool)
        soft = cv2.GaussianBlur(img, (0, 0), 0.75)
        img[c] = soft[c]
        ids = r["id"].cpu().numpy()
        depth = simulate_depth(r["depth"].cpu().numpy(), dep, ids, r["spec"].cpu().numpy(), props, cam.fx, sigma, rng)
        gid = f"f_{args.seed:02d}{i:05d}"
        d = args.out / "train" / gid
        d.mkdir(exist_ok=True)
        cv2.imwrite(str(d / "color.png"), img)
        cv2.imwrite(str(d / "depth.png"), depth)
        (d / "meta.json").write_text(json.dumps({**meta, "id": gid, "synthetic": True, "renderer": "polygon.fastrt",
                                                 "background_frame": fid}))
        img_id = len(coco["images"]) + 1
        coco["images"].append({"id": img_id, "file_name": f"train/{gid}/color.png", "width": cam.W, "height": cam.H})
        for k, t in enumerate(trash):
            m = ids == k
            if m.sum() < 30:
                continue
            segs, bbox, area = mask_to_ann(m)
            coco["annotations"].append({"id": aid, "image_id": img_id, "bbox": bbox, "area": area, "iscrowd": 0,
                                        "category_id": 1 if t.form == "bulk" else 2, "segmentation": segs,
                                        "attributes": {"form": t.form, "material": t.material, "volume_ml": round(t.volume_ml, 1),
                                                       "mass_g": round(t.mass_g, 1), "take": t.take, "kind": t.kind,
                                                       "filled": t.filled, "object_id": f"{gid}_{k}"}})
            aid += 1
        if (i + 1) % 50 == 0:
            print(f"{i + 1}/{args.n}  render {1000 * t_render / (i + 1):.1f} ms/frame  total {time.time() - t0:.0f}s", flush=True)
    (args.out / "labels").mkdir(exist_ok=True)
    (args.out / "labels/gen_train.json").write_text(json.dumps(coco))
    print(f"done: {len(coco['images'])} frames, {len(coco['annotations'])} objects, render {1000 * t_render / max(args.n, 1):.1f} ms/frame, "
          f"total {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
