"""Mini animation: the robot drives over tiles, sweeps light trash, detours around a heavy object.

  python -m polygon.animate --out polygon_out/robot.mp4 [--frames 130] [--preview frame.png]

Chase camera view plus an inset of the robot's own D435 view; trash labels show the generated
properties (material, volume, mass, take decision).
"""
from __future__ import annotations

import argparse
import math
import time
from pathlib import Path

import cv2
import numpy as np

from .render import Camera, FloorTexture, render_cg
from .robot import build_robot, robot_camera_pose
from .trash import random_trash

TILE = Path(__file__).resolve().parent / "assets/floor_tile.png"


def path_pose(s: float):
    """Robot pose at travelled distance s: straight line with a smooth detour around x=1.9."""
    x = s
    y = 0.42 * math.exp(-((x - 1.9) / 0.45) ** 2)
    dy = 0.42 * math.exp(-((x - 1.9) / 0.45) ** 2) * (-2 * (x - 1.9) / 0.45 ** 2)
    return x, y, math.atan(dy)


def make_trash(seed=7):
    rng = np.random.default_rng(seed)
    items = []
    for mat, want_light in (("paper", True), ("plastic", True), ("metal", True), ("mineral", False), ("wood", True)):
        for _ in range(200):
            ob = random_trash(rng, mat)
            if want_light and ob.take == 1 and ob.radius < 0.11:
                break
            if not want_light and ob.take == 0:
                break
        items.append(ob)
    spots = [(0.75, 0.04), (1.25, -0.05), (2.7, 0.03), (1.9, 0.02), (3.15, -0.04)]
    return list(zip(items, spots))


def label(img, txt, xy, color):
    x, y = int(xy[0]), int(xy[1])
    (w, h), _ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
    cv2.rectangle(img, (x - 3, y - h - 6), (x + w + 3, y + 3), (25, 25, 25), -1)
    cv2.putText(img, txt, (x, y - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="polygon_out/robot.mp4", type=Path)
    ap.add_argument("--frames", type=int, default=130)
    ap.add_argument("--fps", type=int, default=20)
    ap.add_argument("--W", type=int, default=960)
    ap.add_argument("--H", type=int, default=540)
    ap.add_argument("--ss", type=int, default=1)
    ap.add_argument("--preview", type=Path, default=None, help="render one frame (index --at) to this png")
    ap.add_argument("--at", type=int, default=40)
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--end", type=int, default=None)
    ap.add_argument("--png-dir", type=Path, default=None, help="write frames as PNG instead of a video")
    ap.add_argument("--assemble", type=Path, default=None, help="join PNGs from this dir into --out")
    args = ap.parse_args()
    if args.assemble:
        import imageio
        files = sorted(args.assemble.glob("f_*.png"))
        w = imageio.get_writer(str(args.out), fps=args.fps, quality=8, macro_block_size=1)
        for fp in files:
            w.append_data(cv2.imread(str(fp))[..., ::-1])
        w.close()
        print("assembled", len(files), "->", args.out)
        return
    floor = FloorTexture.load(TILE)
    trash = make_trash()
    speed = 3.4 / args.frames
    # collection schedule for the whole clip (lets frame ranges render in parallel processes)
    schedule = {}
    for f in range(args.frames):
        x, y, yaw = path_pose(f * speed)
        for k, (ob, (tx, ty)) in enumerate(trash):
            if ob.take == 1 and k not in schedule and x + 0.40 * math.cos(yaw) > tx and abs(y - ty) < 0.2:
                schedule[k] = f
    end = args.end if args.end is not None else args.frames
    frames = [args.at] if args.preview else range(args.start, end)
    writer = None
    if args.png_dir:
        args.png_dir.mkdir(parents=True, exist_ok=True)
    elif not args.preview:
        import imageio
        args.out.parent.mkdir(parents=True, exist_ok=True)
        writer = imageio.get_writer(str(args.out), fps=args.fps, quality=8, macro_block_size=8)
    t0 = time.time()
    for f in frames:
        s = f * speed
        x, y, yaw = path_pose(s)
        parts = build_robot(x, y, yaw, brush_angle=f * 0.35)
        shown = []
        for k, (ob, (tx, ty)) in enumerate(trash):
            age = f - schedule.get(k, 10 ** 9)
            if age > 6:
                continue
            ox, oy = (tx, ty) if age < 0 else (tx + (x - tx) * age / 6, ty + (y - ty) * age / 6)
            parts += ob.placed(ox, oy, 0.6 * k, k)
            if age < 0:
                shown.append((ob, tx, ty))
        a = 0.35 + f * 0.004
        eye = (x - 1.25 * math.cos(a), y - 1.25 * math.sin(a) - 0.25, 0.85)
        cam = Camera.look_at(eye, (x + 0.35, y, 0.12), W=args.W, H=args.H, fov_deg=55)
        img = render_cg(parts, cam, floor, ss=args.ss)
        # robot's own camera inset (dataset-like view)
        e2, t2 = robot_camera_pose(x, y, yaw)
        rc = Camera.look_at(e2, t2, W=320, H=180, fov_deg=70)
        inset = render_cg([p for p in parts if p.obj_id >= 0] or parts[:1], rc, floor, ss=1)
        img[12:192, args.W - 332:args.W - 12] = inset
        cv2.rectangle(img, (args.W - 333, 11), (args.W - 11, 193), (240, 240, 240), 1)
        cv2.putText(img, "D435 robot view", (args.W - 326, 186), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)
        import torch
        for ob, tx, ty in shown:
            pix, z = cam.project(torch.tensor([[tx, ty, 0.12]], device=cam.origin.device))
            px, py = pix[0].tolist()
            if 0 < px < args.W and 0 < py < args.H and float(z[0]) > 0:
                txt = f"{ob.material} {ob.volume_ml:.0f}ml {ob.mass_g:.0f}g {'TAKE' if ob.take else 'AVOID'}"
                label(img, txt, (px - 60, py - 10), (120, 255, 120) if ob.take else (90, 90, 255))
        done = sum(1 for v in schedule.values() if v <= f)
        cv2.putText(img, f"collected {done}  t={s / 0.34:.1f}s", (14, args.H - 16),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
        if args.preview:
            args.preview.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(args.preview), img)
        elif args.png_dir:
            cv2.imwrite(str(args.png_dir / f"f_{f:04d}.png"), img)
        else:
            writer.append_data(img[..., ::-1])
        if f % 10 == 0:
            print(f"frame {f} {time.time() - t0:.0f}s", flush=True)
    if writer:
        writer.close()
    print("done", time.time() - t0)


if __name__ == "__main__":
    main()
