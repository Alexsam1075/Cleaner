import os
import json
import subprocess
import tempfile
from pathlib import Path

import cv2
import numpy as np

from geom import CAM, ROOT, floor_frame

BLENDER = os.environ.get("BLENDER", "/Applications/Blender.app/Contents/MacOS/Blender")

LIGHTING = {
    "bright": {"ambient": 0.08, "lights": [{"type": "AREA", "energy": 85, "size": 2.0, "pos": [0.3, 2.2, 2.6]},
                                           {"type": "AREA", "energy": 28, "size": 3.0, "pos": [-1.5, 0.0, 2.6]}]},
    "dim": {"ambient": 0.05, "lights": [{"type": "AREA", "energy": 65, "size": 2.0, "pos": [0.0, 2.5, 2.6]}]},
    "side": {"ambient": 0.05, "lights": [{"type": "AREA", "energy": 65, "size": 1.0, "pos": [-2.0, 1.0, 0.9], "target": [0.2, 1.0, 0]},
                                        {"type": "AREA", "energy": 18, "size": 2.0, "pos": [0.0, 2.0, 2.6]}]},
}


def camera_pose(plane, x=0.0, y=0.0, yaw=0.0):
    """Поза камеры: базовая из плоскости пола кадра, затем сдвиг/поворот робота по полу."""
    R, t = floor_frame(plane)
    c, s = np.cos(yaw), np.sin(yaw)
    Rz = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1.0]])
    return {"R": (Rz @ R).tolist(), "t": (Rz @ t + np.array([x, y, 0.0])).tolist()}


def scaled_cam(scale):
    c = {k: CAM[k] for k in ("width", "height", "fx", "fy", "cx", "cy")}
    for k in c:
        c[k] = c[k] * scale
    c["width"], c["height"] = int(round(c["width"])), int(round(c["height"]))
    return c


def render(spec, out, samples=48, quiet=True, scale=1.0):
    spec = dict(spec)
    spec.setdefault("root", str(ROOT))
    spec["out"] = str(out)
    spec["samples"] = samples
    cam = scaled_cam(scale)
    cam.update(spec["camera"])
    spec["camera"] = cam
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as fh:
        json.dump(spec, fh)
    cmd = [BLENDER, "-b", "--factory-startup", "-t", os.environ.get("BLENDER_THREADS", "0"), "--python", str(ROOT / "sim" / "blender_render.py"), "--", fh.name]
    p = subprocess.run(cmd, capture_output=True, text=True)
    if p.returncode != 0 or "Error" in p.stderr[-2000:]:
        print(p.stdout[-3000:], p.stderr[-3000:])
        raise RuntimeError("blender failed")
    if not quiet:
        print("\n".join(l for l in p.stdout.splitlines() if "RENDERED" in l))
    return sorted(Path(out).glob("f_*"))


def read_frame(fdir):
    fdir = Path(fdir)
    color = cv2.imread(str(fdir / "color.png"))
    if (fdir / "depth.png").is_file() and (fdir / "index.png").is_file():
        depth = cv2.imread(str(fdir / "depth.png"), cv2.IMREAD_UNCHANGED)
        index = cv2.imread(str(fdir / "index.png"), cv2.IMREAD_UNCHANGED)
        if color is None or depth is None or index is None:
            raise ValueError(f"Failed to read RGB-D/index from {fdir}")
        return color, depth.astype(np.float32) / 1000.0, index.astype(np.int32)
    import OpenEXR
    exr = lambda name: OpenEXR.File(str(next(fdir.glob(f"{name}*.exr")))).channels()["RGBA"].pixels[..., 0]
    z = exr("depth")
    idx = np.rint(exr("index")).astype(np.int32)
    return color, z, idx
