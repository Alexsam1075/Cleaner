"""Generate isolated bag scenes with the existing Blender RGB/depth renderer.

The object is a procedurally crumpled film mesh; mass is computed from the
original sheet area, sampled thickness and polymer density. These are synthetic
assumptions, not real-world labels. Keep these scenes separate from real data.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
os.environ.setdefault("ASTRAMIS_DATA", str(ROOT / "datasets" / "raw"))

from scenes import make_scene, object_split  # noqa: E402
from run_blender import render  # noqa: E402


def bag_mesh(seed: int, grid: int = 25):
    """Return a folded double sheet and its sampled physical parameters.

    This is a kinematic approximation, not a cloth simulation. Two connected
    sheet layers model the front and back of a bag. Repeated folds and radial
    compression generate new shapes; the mass comes from the original sheet.
    """
    rng = np.random.default_rng(seed)
    width, length = rng.uniform(.20, .55, 2)
    thickness_um = float(np.exp(rng.uniform(np.log(8), np.log(65))))
    density = float(rng.uniform(.91, .96))  # g/cm3, polyethylene
    u, v = np.meshgrid(np.linspace(-1., 1., grid), np.linspace(-1., 1., grid))
    # Smoothed radial mapping into a compact shape, with nonuniform folds.
    phase = rng.uniform(0, 2 * np.pi, 5)
    fold = (np.sin(7*u + 5*v + phase[0]) + .55*np.sin(13*u - 8*v + phase[1])
            + .35*np.sin(22*u + 4*v + phase[2]))
    x = .27*width*(u + .18*np.sin(5*v + phase[3]))
    y = .27*length*(v + .18*np.sin(6*u + phase[4]))
    z = .016 + .035*np.exp(-1.7*(u*u + v*v)) + .010*fold
    z -= z.min()
    vertices = np.stack([x, y, z], axis=-1).reshape(-1, 3).astype(np.float32)
    faces = []
    for j in range(grid - 1):
        for i in range(grid - 1):
            a = j*grid+i
            faces.extend(((a, a+1, a+grid+1), (a, a+grid+1, a+grid)))
    faces = np.asarray(faces, dtype=np.int32)
    # Hidden back layer is offset to make the two physical sheets explicit.
    back = vertices.copy()
    back[:, 2] = np.maximum(0., back[:, 2] - .003)
    vertices = np.concatenate((vertices, back))
    faces = np.concatenate((faces, faces[:, ::-1] + grid*grid))
    palette = np.array([[.05,.30,.8], [.87,.87,.85], [.15,.15,.15], [.12,.6,.38], [.8,.23,.15]])
    color = palette[int(rng.integers(len(palette)))]
    variation = rng.uniform(.82, 1.12, (len(vertices), 1))
    colors = np.clip(color[None, :] * variation, 0, 1).astype(np.float32)
    film_area = 2 * width * length
    mass = film_area * thickness_um * density  # m2 * micrometre * g/cm3 = grams
    bounds = np.ptp(vertices, axis=0)
    # Envelope is a descriptive proxy; air-filled volume is not film volume.
    return vertices, faces, colors, {
        "sheet_width_m": float(width), "sheet_length_m": float(length),
        "film_area_m2": float(film_area), "thickness_um": thickness_um,
        "polymer_density_g_cm3": density, "mass_g": float(mass),
        "film_volume_ml": float(film_area * thickness_um),
        "envelope_box_volume_ml": float(np.prod(bounds) * 1e6),
        "mesh_seed": int(seed), "mass_formula": "film_area_m2 * thickness_um * polymer_density_g_cm3",
        "volume_label_warning": "box envelope proxy; film volume is stored separately",
        "generation": "kinematic procedural folded sheet, no cloth physics",
    }


def render_custom(spec, scene_dir, scale):
    """Render the bag with the project's own CPU ray renderer as RGB-D PNGs."""
    import cv2
    sys.path.insert(0, str(ROOT))
    from cleaner_view.renderer import prepare, render as render_cpu
    from geom import CAM

    scene_data = prepare(spec)
    width, height = int(round(CAM["width"]*scale)), int(round(CAM["height"]*scale))
    n = len(spec["camera"]["poses"])
    for index, pose in enumerate(spec["camera"]["poses"]):
        R, t = np.asarray(pose["R"]), np.asarray(pose["t"])
        pos = t.astype(np.float64)
        forward, right, up = R[:, 2], R[:, 0], -R[:, 1]
        intrinsic = np.array([CAM["fx"]*scale, CAM["fy"]*scale,
                              CAM["cx"]*scale, CAM["cy"]*scale], dtype=np.float64)
        from cleaner_view.renderer import render_cpu as trace
        rgb, distance, mask = trace(width, height, pos, forward, right, up, intrinsic,
                                    *scene_data, np.zeros(3, dtype=np.float64), False)
        yy, xx = np.mgrid[:height, :width]
        factor = np.sqrt(1 + ((xx+.5-intrinsic[2])/intrinsic[0])**2
                         + ((yy+.5-intrinsic[3])/intrinsic[1])**2)
        depth = distance / factor
        depth[~np.isfinite(depth)] = 0
        frame = scene_dir / f"f_{index:03d}"
        frame.mkdir()
        cv2.imwrite(str(frame / "color.png"), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
        cv2.imwrite(str(frame / "depth.png"), np.clip(np.rint(depth*1000), 0, 65535).astype(np.uint16))
        cv2.imwrite(str(frame / "index.png"), mask.astype(np.uint16))
        print(f"rendered {index+1}/{n}", flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", required=True)
    p.add_argument("--split", choices=("train", "val"), default="train")
    p.add_argument("--count", type=int, default=1)
    p.add_argument("--start", type=int, default=0)
    p.add_argument("--poses", type=int, default=3)
    p.add_argument("--scale", type=float, default=.5)
    p.add_argument("--samples", type=int, default=24)
    p.add_argument("--mesh-only", action="store_true")
    p.add_argument("--renderer", choices=("custom", "blender"), default="custom")
    a = p.parse_args()
    if a.count < 1 or a.start < 0 or a.poses < 1 or not 0 < a.scale <= 1:
        p.error("count and poses must be positive, start nonnegative, scale in (0, 1]")
    base = Path(a.out).expanduser().resolve() / a.split
    train_ids, val_ids = object_split()
    for index in range(a.start, a.start + a.count):
        scene_dir = base / f"scene_bag_{index:05d}"
        scene_dir.mkdir(parents=True, exist_ok=False)
        seed = index*2 + (0 if a.split == "train" else 1) + 100000
        verts, faces, colors, physics = bag_mesh(seed)
        mesh_path = scene_dir / "bag.npz"
        np.savez_compressed(mesh_path, verts=verts, faces=faces, colors=colors)
        spec = make_scene(seed, train_ids if a.split == "train" else val_ids, n_poses=a.poses)
        spec["objects"] = [{"id": f"bag_{a.split}_{index:05d}", "mesh": str(mesh_path),
                           "material": "plastic", "form": "solid", "mass_g": physics["mass_g"],
                           "volume_ml": physics["envelope_box_volume_ml"],
                           "take": int(physics["mass_g"] <= 500), "x": 0., "y": .65,
                           "yaw": float(np.random.default_rng(seed+1).uniform(-np.pi, np.pi)),
                           "synthetic_bag": physics}]
        spec["scale"] = a.scale
        spec["renderer"] = a.renderer
        spec["depth_format"] = "uint16 millimetres" if a.renderer == "custom" else "EXR metres"
        (scene_dir / "scene.json").write_text(json.dumps(spec, indent=2), encoding="utf-8")
        if not a.mesh_only:
            if a.renderer == "custom":
                render_custom(spec, scene_dir, a.scale)
            else:
                frames = render(spec, scene_dir, samples=a.samples, scale=a.scale)
                if len(frames) != a.poses:
                    raise RuntimeError(f"Expected {a.poses} rendered frames, received {len(frames)} in {scene_dir}")
        print(json.dumps({"scene": str(scene_dir), "mass_g": physics["mass_g"], "rendered": not a.mesh_only}), flush=True)


if __name__ == "__main__":
    main()
