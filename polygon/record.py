"""Record the live polygon headless (real time, autopilot on the real model) for docs.

  python -m polygon.record --seconds 40 --out docs/media/raw
Writes chase-view frames and model-view frames as JPEG sequences.
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import cv2
import numpy as np

from .live import Live
from .perception import Perception, overlay


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seconds", type=float, default=40)
    ap.add_argument("--out", type=Path, default=Path("docs/media/raw"))
    ap.add_argument("--W", type=int, default=960)
    ap.add_argument("--H", type=int, default=540)
    ap.add_argument("--seed", type=int, default=4)
    ap.add_argument("--trash", type=int, default=16)
    args = ap.parse_args()
    (args.out / "chase").mkdir(parents=True, exist_ok=True)
    (args.out / "model").mkdir(parents=True, exist_ok=True)
    live = Live(args.W, args.H, 2, 30, Perception())
    live.rng = np.random.default_rng(args.seed)
    live.reset()
    live.trash = []
    for _ in range(args.trash):
        live.spawn(near=False)
    live.auto = True
    live.dist, live.el = 2.6, 0.62
    t0, last, frame, mframe = time.time(), time.time(), 0, None
    while time.time() - t0 < args.seconds:
        now = time.time()
        live.step(min(now - last, 0.1), frame)
        last = now
        live.az += 0.004
        cv2.imwrite(str(args.out / "chase" / f"{frame:05d}.jpg"), live.last_img, [cv2.IMWRITE_JPEG_QUALITY, 92])
        lat = live.perc.latest
        if lat is not None and lat["time"] != mframe:
            mframe = lat["time"]
            cv2.imwrite(str(args.out / "model" / f"{frame:05d}.jpg"), overlay(lat["frame"], lat["dets"], args.W, args.H),
                        [cv2.IMWRITE_JPEG_QUALITY, 92])
        frame += 1
    print(f"{frame} frames in {time.time() - t0:.1f}s, collected {live.collected}, collisions {live.collisions}")


if __name__ == "__main__":
    main()
