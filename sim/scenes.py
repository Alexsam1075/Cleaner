"""Генератор сцен полигона и рендер обучающего набора.
Сцена: пол из плитки + (опционально) ковёр/ламинат, 1–4 оцифрованных предмета датасета,
0–2 «неизвестных» предмета (примитивы со случайной формой/цветом/материалом), 0–2 препятствия.
Для каждой сцены рендерится несколько поз робота. Метки: маски предметов (index pass), поверхности, атрибуты."""
import argparse
import json
import random
from pathlib import Path

import numpy as np

from geom import ROOT, meta
from run_blender import LIGHTING, camera_pose, render

OBJ = json.loads((ROOT / "sim/assets/objects.json").read_text(encoding="utf-8"))
DENS = {"metal": 2.0, "wood": 0.7, "plastic": 0.4, "glass": 1.2, "paper": 0.3, "mineral": 1.8}
UNKNOWN_COLORS = {"metal": [(0.6, 0.6, 0.62), (0.7, 0.55, 0.3)], "wood": [(0.45, 0.28, 0.12), (0.6, 0.42, 0.22)],
                  "plastic": [(0.1, 0.3, 0.8), (0.9, 0.8, 0.1), (0.1, 0.6, 0.2), (0.9, 0.9, 0.9)],
                  "glass": [(0.7, 0.85, 0.75)], "paper": [(0.85, 0.82, 0.75), (0.6, 0.45, 0.3)],
                  "mineral": [(0.5, 0.48, 0.45), (0.35, 0.3, 0.28)]}
PLANES = [meta(f)["floor_plane"] for f in ("r_0001", "r_0246", "r_0375")]


def unknown_object(rng, k):
    mat = rng.choice(list(DENS))
    prim = rng.choice(["box", "cylinder", "sphere"])
    dx, dy = rng.uniform(0.04, 0.25, 2)
    dz = rng.uniform(0.02, 0.15)
    if prim == "sphere":
        dz = min(dx, dy)
    vol_ml = dx * dy * dz * {"box": 1.0, "cylinder": np.pi / 4, "sphere": np.pi / 6}[prim] * 1e6
    fill = rng.uniform(0.15, 1.0)
    mass = vol_ml * DENS[mat] * fill
    col = UNKNOWN_COLORS[mat][rng.integers(len(UNKNOWN_COLORS[mat]))]
    col = tuple(float(np.clip(c + rng.normal(0, 0.05), 0, 1)) for c in col)
    return {"id": f"u{k:02d}_{prim}", "primitive": prim, "dims": [float(dx), float(dy), float(dz)], "color": col,
            "material": mat, "form": "solid", "volume_ml": round(vol_ml, 1), "mass_g": round(mass, 1),
            "take": int(mass <= 500), "unknown": True}


def make_scene(seed, object_ids, allow_carpet=True, n_poses=12):
    rng = np.random.default_rng(seed)
    surfaces = [{"type": "tile", "rect": [-6, -6, 6, 6], "rot": float(rng.uniform(0, np.pi / 2)),
                 "offset": [float(rng.uniform(0, 0.43)), float(rng.uniform(0, 0.43))]}]
    extra = rng.choice(["none", "carpet", "laminate"], p=[0.3, 0.45 if allow_carpet else 0, 0.25 if allow_carpet else 0.7] if allow_carpet else [0.6, 0.0, 0.4])
    if extra != "none":
        cx, cy = rng.uniform(-0.8, 0.8), rng.uniform(0.4, 1.6)
        w, d = rng.uniform(0.6, 1.8, 2)
        surfaces.append({"type": str(extra), "rect": [cx - w / 2, cy - d / 2, cx + w / 2, cy + d / 2],
                         "rot": float(rng.uniform(0, np.pi)), "thickness": float(rng.uniform(0.006, 0.02))})
    objs = []
    n_known = int(rng.integers(1, 5))
    for oid in rng.choice(object_ids, size=min(n_known, len(object_ids)), replace=False):
        o = OBJ[oid]
        objs.append({"id": str(oid), "material": o["material"], "form": o["form"], "volume_ml": o["volume_ml"],
                     "mass_g": o["mass_g"], "take": o["take"], "unknown": False})
    for k in range(int(rng.integers(0, 3))):
        objs.append(unknown_object(rng, k))
    placed = []
    for o in objs:
        for _ in range(50):
            x, y = rng.uniform(-0.5, 0.5), rng.uniform(0.3, 1.2)
            if all(np.hypot(x - a, y - b) > 0.3 for a, b in placed):
                break
        placed.append((x, y))
        o.update({"x": float(x), "y": float(y), "yaw": float(rng.uniform(0, 2 * np.pi))})
    props = []
    for _ in range(int(rng.integers(0, 3))):
        props.append({"x": float(rng.uniform(-1.5, 1.5)), "y": float(rng.uniform(1.0, 3.0)), "w": float(rng.uniform(0.05, 0.4)),
                      "d": float(rng.uniform(0.05, 0.4)), "h": float(rng.uniform(0.1, 0.6)), "yaw": float(rng.uniform(0, 3)),
                      "color": [float(c) for c in rng.uniform(0.02, 0.5, 3)]})
    light_name = str(rng.choice(list(LIGHTING)))
    light = json.loads(json.dumps(LIGHTING[light_name]))
    gain = float(rng.uniform(0.7, 1.3))
    for L in light["lights"]:
        L["energy"] *= gain
        L["pos"] = [p + float(rng.normal(0, 0.3)) for p in L["pos"]]
    poses = []
    for _ in range(n_poses):
        plane = PLANES[int(rng.integers(len(PLANES)))]
        poses.append(camera_pose(plane, float(rng.normal(0, 0.12)), float(rng.uniform(-0.2, 0.1)), float(rng.normal(0, 0.12))))
    return {"camera": {"poses": poses}, "surfaces": surfaces, "objects": objs, "props": props,
            "lighting": light_name, "exposure": float(rng.normal(0, 0.15)), **light}


def object_split():
    ids = sorted(OBJ)
    val = ids[::5]
    return [i for i in ids if i not in val], val


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--poses", type=int, default=12)
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--split", default="train", choices=["train", "val"])
    ap.add_argument("--no-carpet", action="store_true")
    ap.add_argument("--scale", type=float, default=0.5)
    ap.add_argument("--samples", type=int, default=24)
    ap.add_argument("--out", default=str(ROOT / "sim/data"))
    a = ap.parse_args()
    tr, va = object_split()
    ids = tr if a.split == "train" else va
    for s in range(a.start, a.start + a.n):
        out = Path(a.out) / a.split / f"scene_{s:04d}"
        if (out / "scene.json").exists():
            continue
        spec = make_scene(s * 7919 + (0 if a.split == "train" else 1), ids, allow_carpet=not a.no_carpet, n_poses=a.poses)
        render(spec, out, samples=a.samples, scale=a.scale)
        spec["scale"] = a.scale
        (out / "scene.json").write_text(json.dumps(spec), encoding="utf-8")
        print("scene", s, a.split, flush=True)
