"""Robot animation on the real-time renderer.

  python -m polygon.animate_fast --out polygon_out/robot.mp4 --frames 300 --W 1280 --H 720
"""
from __future__ import annotations

import argparse
import math
import time
from pathlib import Path

import cv2
import numpy as np
import torch

from .fast_scene import add_robot, random_trash, robot_camera
from .fastrt import FastCam, FloorTex, Scene, render, to_u8

TILE = Path(__file__).resolve().parent / "assets/floor_tile.png"


def path_pose(s):
    x = s
    g = math.exp(-((x - 1.9) / 0.45) ** 2)
    return x, 0.42 * g, math.atan(0.42 * g * (-2 * (x - 1.9) / 0.45 ** 2))


def make_trash(seed=7):
    rng = np.random.default_rng(seed)
    out = []
    for mat, light in (("paper", True), ("plastic", True), ("glass", True), ("metal", True), ("mineral", False), ("wood", True)):
        for _ in range(300):
            t = random_trash(rng, mat)
            if (light and t.take == 1 and t.radius < 0.12) or (not light and t.take == 0):
                break
        out.append(t)
    spots = [(0.75, 0.04), (1.2, -0.06), (1.45, 0.07), (2.6, 0.03), (1.9, 0.02), (3.1, -0.04)]
    return list(zip(out, spots))


def chip(img, txt, x, y, color):
    (w, h), _ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    x, y = int(x - w / 2), int(y)
    cv2.rectangle(img, (x - 5, y - h - 7), (x + w + 5, y + 4), (22, 22, 24), -1)
    cv2.rectangle(img, (x - 5, y - h - 7), (x + w + 5, y + 4), color, 1)
    cv2.putText(img, txt, (x, y - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (235, 235, 235), 1, cv2.LINE_AA)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="polygon_out/robot.mp4", type=Path)
    ap.add_argument("--frames", type=int, default=300)
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--W", type=int, default=1280)
    ap.add_argument("--H", type=int, default=720)
    ap.add_argument("--spp", type=int, default=4)
    args = ap.parse_args()
    import imageio
    floor = FloorTex(TILE)
    trash = make_trash()
    speed = 3.5 / args.frames
    sched = {}
    for f in range(args.frames):
        x, y, yaw = path_pose(f * speed)
        for k, (t, (tx, ty)) in enumerate(trash):
            if t.take and k not in sched and x + 0.40 * math.cos(yaw) > tx and abs(y - ty) < 0.2:
                sched[k] = f
    args.out.parent.mkdir(parents=True, exist_ok=True)
    wr = imageio.get_writer(str(args.out), fps=args.fps, quality=8, macro_block_size=8)
    t_render = 0.0
    t0 = time.time()
    for f in range(args.frames):
        x, y, yaw = path_pose(f * speed)
        sc = Scene()
        add_robot(sc, x, y, yaw, brush_angle=f * 0.3)
        visible = []
        for k, (t, (tx, ty)) in enumerate(trash):
            age = f - sched.get(k, 10 ** 9)
            if age > 8:
                continue
            ox, oy = (tx, ty) if age < 0 else (tx + (x - tx) * age / 8, ty + (y - ty) * age / 8)
            t.place(sc, ox, oy, 0.7 * k, k)
            if age < 0:
                visible.append((t, tx, ty))
        a = math.pi + 0.5 - 2.6 * f / args.frames
        eye = (x + 0.3 + 1.9 * math.cos(a), y + 1.9 * math.sin(a), 1.0)
        cam = FastCam.look_at(eye, (x + 0.4, y, 0.12), W=args.W, H=args.H, fov_deg=52)
        e2, t2 = robot_camera(x, y, yaw)
        rcam = FastCam.look_at(e2, t2, W=424, H=240, fov_deg=70)
        torch.cuda.synchronize()
        ts = time.time()
        img = to_u8(render(sc, cam, floor=floor, spp=args.spp, seed=f)["rgb"])
        ins = to_u8(render(sc, rcam, floor=floor, spp=2, seed=f)["rgb"])
        torch.cuda.synchronize()
        t_render += time.time() - ts
        H, W = args.H, args.W
        img[14:254, W - 438:W - 14] = ins
        cv2.rectangle(img, (W - 439, 13), (W - 13, 255), (235, 235, 235), 1)
        cv2.putText(img, "D435 robot view", (W - 430, 246), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
        for t, tx, ty in visible:
            (px, py), z = cam.project((tx, ty, t.height + 0.05))
            if z > 0 and 0 < px < W and 0 < py < H:
                chip(img, f"{t.material} {t.volume_ml:.0f} ml {t.mass_g:.0f} g {'TAKE' if t.take else 'AVOID'}", px, py,
                     (110, 220, 120) if t.take else (80, 80, 245))
        done = sum(1 for v in sched.values() if v <= f)
        cv2.putText(img, f"collected {done}   {1000 * t_render / (f + 1):.1f} ms/frame render", (16, H - 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2, cv2.LINE_AA)
        wr.append_data(img[..., ::-1])
    wr.close()
    print(f"{args.frames} frames, render {1000 * t_render / args.frames:.1f} ms/frame (main {args.W}x{args.H} spp{args.spp} + inset), "
          f"total {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
