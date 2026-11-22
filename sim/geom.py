import json
import os
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DATA = Path(os.environ.get("ASTRAMIS_DATA", ROOT / "dataset"))
CAM = json.loads((DATA / "camera.json").read_text(encoding="utf-8"))
W, H = CAM["width"], CAM["height"]
K = np.array([[CAM["fx"], 0, CAM["cx"]], [0, CAM["fy"], CAM["cy"]], [0, 0, 1.0]])


def frame_dir(fid):
    for split in ("real/train", "real/test", "synth/train"):
        p = DATA / split / fid
        if p.exists():
            return p
    raise FileNotFoundError(fid)


def meta(fid):
    return json.loads((frame_dir(fid) / "meta.json").read_text(encoding="utf-8"))


def floor_frame(plane):
    """Мировая система: Z вверх (нормаль пола), Y вперёд (проекция оптической оси), начало под камерой.
    Возвращает R (камера->мир: Pw = R @ Pc + t) и t (позиция камеры в мире)."""
    n = np.asarray(plane[:3], float)
    d = float(plane[3])
    s = np.linalg.norm(n)
    n, d = n / s, d / s
    if d < 0:
        n, d = -n, -d
    fwd = np.array([0.0, 0.0, 1.0]) - n * n[2]
    fwd /= np.linalg.norm(fwd)
    right = np.cross(fwd, n)
    R = np.stack([right, fwd, n])
    t = np.array([0.0, 0.0, d])
    return R, t


def unproject(depth_mm):
    v, u = np.mgrid[0:depth_mm.shape[0], 0:depth_mm.shape[1]]
    z = depth_mm.astype(np.float64) / 1000.0
    x = (u - K[0, 2]) / K[0, 0] * z
    y = (v - K[1, 2]) / K[1, 1] * z
    return np.stack([x, y, z], -1)


def cam_to_world(P, plane):
    R, t = floor_frame(plane)
    return P @ R.T + t


def world_to_pixel(Pw, plane):
    R, t = floor_frame(plane)
    Pc = (Pw - t) @ R
    uv = Pc @ K.T
    return uv[..., :2] / uv[..., 2:3], Pc[..., 2]


def floor_ray_hit(u, v, plane):
    """Пиксель -> точка на полу (мир), через пересечение луча с плоскостью."""
    R, t_wc = floor_frame(plane)
    ray_c = np.stack([(u - K[0, 2]) / K[0, 0], (v - K[1, 2]) / K[1, 1], np.ones_like(u, dtype=float)], -1)
    ray_w = ray_c @ R.T
    t = -t_wc[2] / ray_w[..., 2]
    return t_wc + ray_w * t[..., None], t
