"""Build the real-time 3D map page: dataset RGB-D frames as textured 3D meshes, the robot at the
camera pose, predictions of the best pipeline (detector + material + parts volume) and the
ground truth per object.

  python -m polygon.build_map --out polygon_out/map.html --frames 30
"""
from __future__ import annotations

import argparse
import base64
import json
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RUNS = Path.home() / "Cleaner/checkpoints"
FOLDS = {
    "fold1": ("score83_rgb320_fold1", "reports/volmod_score83_rgb320_fold1.csv"),
    "fold2": ("score83_cv_fold2_noval18", "reports/volmod_score83_cv_fold2_noval18.csv"),
    "fold3": ("score83_cv_fold3_noval18", "reports/volmod_score83_cv_fold3_noval18.csv"),
}
GRID_W, GRID_H = 160, 90


def plane_axes(plane):
    p = np.asarray(plane, float)
    n, d = p[:3] / np.linalg.norm(p[:3]), p[3] / np.linalg.norm(p[:3])
    if d < 0:
        n, d = -n, -d
    fwd = np.array([0, 0, 1.0]) - n * n[2]
    fwd /= np.linalg.norm(fwd)
    return np.stack([fwd, np.cross(n, fwd), n]), d


def to_world(xs, ys, z_m, cam, R, d):
    pc = np.stack([(xs - cam["cx"]) / cam["fx"] * z_m, (ys - cam["cy"]) / cam["fy"] * z_m, z_m], -1)
    w = pc @ R.T
    w[..., 2] += d
    return w


def b64(arr: np.ndarray) -> str:
    return base64.b64encode(arr.tobytes()).decode()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=ROOT / "datasets/raw", type=Path)
    ap.add_argument("--out", default=ROOT / "polygon_out/map.html", type=Path)
    ap.add_argument("--frames", type=int, default=30)
    args = ap.parse_args()
    cam = json.loads((args.data / "camera.json").read_text())
    lab = json.loads((args.data / "labels/real_train.json").read_text())
    names = {i["id"]: Path(i["file_name"]).parent.name for i in lab["images"]}
    gt = {}
    for a in lab["annotations"]:
        gt.setdefault(names[a["image_id"]], []).append(a)
    rows = []
    for fold, (run, csv) in FOLDS.items():
        p = ROOT / csv
        if p.exists():
            df = pd.read_csv(p)
            rows += [(fold, run, r.image_id, r.prediction) for r in df.itertuples() if str(r.prediction) != "none"]
    rows.sort(key=lambda r: r[2])
    step = max(1, len(rows) // args.frames)
    chosen = rows[::step][: args.frames]
    metrics = {}
    for fold, (run, _) in FOLDS.items():
        rep = ROOT / f"reports/sweep_{run}_knots.json"
        if rep.exists():
            metrics[fold] = json.loads(rep.read_text())["parts_volume"]
    frames = []
    for fold, run, fid, cell in chosen:
        fdir = args.data / "real/train" / fid
        meta = json.loads((fdir / "meta.json").read_text())
        R, d = plane_axes(meta["floor_plane"])
        rgb = cv2.imread(str(fdir / "color.png"))
        dep = cv2.imread(str(fdir / "depth.png"), cv2.IMREAD_UNCHANGED).astype(np.float32)
        small = cv2.resize(dep, (GRID_W, GRID_H), interpolation=cv2.INTER_NEAREST)
        ok, jpg = cv2.imencode(".jpg", cv2.resize(rgb, (640, 360)), [cv2.IMWRITE_JPEG_QUALITY, 82])
        objs = []
        for tok in (t.split() for t in cell.split(";")):
            x1, y1, x2, y2 = (float(v) for v in tok[:4])
            ys, xs = np.mgrid[int(y1):int(y2), int(x1):int(x2)]
            z = dep[int(y1):int(y2), int(x1):int(x2)] / 1000
            w = to_world(xs.astype(float), ys.astype(float), z, cam, R, d)
            sel = (z > 0) & (w[..., 2] > 0.008)
            if sel.sum() > 20:
                pts = w[sel]
                lo, hi = np.percentile(pts, 3, axis=0), np.percentile(pts, 97, axis=0)
                lo[2] = 0.0
            else:
                c = to_world(np.array([(x1 + x2) / 2]), np.array([y2]), np.array([0.6]), cam, R, d)[0]
                lo, hi = c - [0.03, 0.03, 0], c + [0.03, 0.03, 0.04]
            best, giou = None, 0.0
            for g in gt.get(fid, []):
                gx, gy, gw, gh = g["bbox"]
                ix = max(0, min(x2, gx + gw) - max(x1, gx))
                iy = max(0, min(y2, gy + gh) - max(y1, gy))
                u = (x2 - x1) * (y2 - y1) + gw * gh - ix * iy
                if u > 0 and ix * iy / u > giou:
                    best, giou = g, ix * iy / u
            objs.append({"box2d": [x1, y1, x2, y2], "lo": lo.round(4).tolist(), "hi": hi.round(4).tolist(),
                         "score": float(tok[4]), "form": tok[5], "material": tok[6], "volume": float(tok[7]),
                         "mass": float(tok[8]), "take": int(tok[9]),
                         "gt": None if best is None or giou < 0.5 else {
                             "object_id": best["attributes"]["object_id"], "material": best["attributes"]["material"],
                             "volume": best["attributes"]["volume_ml"], "mass": best["attributes"]["mass_g"],
                             "take": best["attributes"]["take"]}})
        frames.append({"id": fid, "fold": fold, "run": run, "plane_d": d, "R": R.round(6).tolist(),
                       "depth": b64(small.astype(np.uint16)), "img": base64.b64encode(jpg.tobytes()).decode(),
                       "surface": meta.get("surface"), "lighting": meta.get("lighting"), "objects": objs})
    ok, tile = cv2.imencode(".jpg", cv2.imread(str(ROOT / "polygon/assets/floor_tile.png")), [cv2.IMWRITE_JPEG_QUALITY, 85])
    data = {"camera": cam, "grid": [GRID_W, GRID_H], "frames": frames, "metrics": metrics,
            "floor_tile": base64.b64encode(tile.tobytes()).decode(),
            "pipeline": {"detector": "C_joint YOLO11-seg (fold weights)", "material": "attributes_rgb_only (fold weights)",
                         "volume": "volume/ parts geometry + class calibration (knots 50/300)",
                         "take": "attribute head; mass = volume x class density"}}
    html = (Path(__file__).resolve().parent / "map_template.html").read_text(encoding="utf-8")
    html = html.replace("/*__DATA__*/null", json.dumps(data))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(html, encoding="utf-8")
    print(args.out, f"{len(frames)} frames, {args.out.stat().st_size / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
