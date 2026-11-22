"""Проезд робота по полигону: физика проскальзывания + рендер последовательности + визуальная одометрия.
Модель привода: колёса крутятся с командной скоростью v_cmd; фактическая скорость v = v_cmd * (1 - slip),
slip растёт с сопротивлением качению поверхности (ковёр >> плитка). Камера видит фактическое движение,
одометрия по полу его восстанавливает, а разница с колёсами даёт измеренный slip."""
import argparse
import json
from pathlib import Path

import numpy as np

from geom import ROOT, meta
from odometry import estimate_motion, floor_mask
from run_blender import LIGHTING, camera_pose, read_frame, render

SURF = json.loads((ROOT / "sim/assets/surfaces.json").read_text())["surfaces"]


def surface_at(surfaces, x, y):
    name = "tile"
    for s in surfaces:
        x0, y0, x1, y1 = s["rect"]
        c, sn = np.cos(-s.get("rot", 0.0)), np.sin(-s.get("rot", 0.0))
        if s["type"] == "tile":
            continue
        cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
        lx = c * (x - cx) - sn * (y - cy)
        ly = sn * (x - cx) + c * (y - cy)
        if abs(lx) <= (x1 - x0) / 2 and abs(ly) <= (y1 - y0) / 2:
            name = s["type"]
    return name


def slip_of(surface, rng):
    rr = SURF[surface]["rolling_resistance"]
    return float(np.clip(rr * 5.0 + rng.normal(0, 0.01), 0, 0.9))


def simulate(surfaces, n=40, v_cmd=0.04, start=(0.0, -0.4), yaw0=0.0, seed=0):
    rng = np.random.default_rng(seed)
    x, y, yaw = start[0], start[1], yaw0
    traj = []
    for i in range(n):
        surf = surface_at(surfaces, x, y + 0.17)
        s = slip_of(surf, rng)
        traj.append({"x": x, "y": y, "yaw": yaw, "surface": surf, "slip": s, "v_cmd": v_cmd, "v": v_cmd * (1 - s)})
        x += -np.sin(yaw) * v_cmd * (1 - s)
        y += np.cos(yaw) * v_cmd * (1 - s)
    return traj


def default_scene():
    return {"surfaces": [{"type": "tile", "rect": [-6, -6, 6, 6], "rot": 0.1},
                         {"type": "carpet", "rect": [-1.2, 0.5, 1.2, 1.3], "thickness": 0.012}],
            "objects": [{"id": "s01", "x": 0.35, "y": 1.9, "material": "plastic"},
                        {"id": "b07", "x": -0.4, "y": 2.2, "material": "paper"}],
            "props": [], **LIGHTING["bright"]}


def run(scene, out, n=40, seed=0, scale=0.5, samples=16, plane_id="r_0001"):
    plane = meta(plane_id)["floor_plane"]
    traj = simulate(scene["surfaces"], n=n, seed=seed)
    scene = dict(scene)
    scene["camera"] = {"poses": [camera_pose(plane, t["x"], t["y"], t["yaw"]) for t in traj]}
    frames = render(scene, out, samples=samples, scale=scale)
    res = []
    for i in range(len(frames) - 1):
        c0, z0, _ = read_frame(frames[i])
        c1, _, _ = read_frame(frames[i + 1])
        m = estimate_motion(c0, c1, plane, mask0=floor_mask(z0, plane, K_scale=scale), scale=scale)
        vis = m["forward"] if m else float("nan")
        res.append({**traj[i], "v_visual": vis, "slip_measured": 1 - vis / traj[i]["v_cmd"]})
    Path(out, "drive.json").write_text(json.dumps({"scene": scene, "log": res}), encoding="utf-8")
    return res


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "sim/out/drive"))
    ap.add_argument("--n", type=int, default=40)
    a = ap.parse_args()
    log = run(default_scene(), a.out, n=a.n)
    for r in log:
        print(f"y={r['y']:.2f} {r['surface']:6s} slip true {r['slip']:.3f} measured {r['slip_measured']:.3f}")
