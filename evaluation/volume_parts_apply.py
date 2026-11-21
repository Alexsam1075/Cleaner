"""Swap only the volume (and mass) of an existing submission for the parts-based estimate, then score.

The detector is re-run with the submission's settings to recover each predicted mask; boxes are
matched by IoU. D, M and the take decision of the original rows stay as they are, so the change
in the score isolates V. The calibrator is fitted only on the split's training object IDs.

Usage (server, from the repo root):
  python -m evaluation.volume_parts_apply --run ~/Cleaner/checkpoints/score83_cv_fold2_noval18 \
      --candidate <csv> --out reports/volume_parts_fold2_noval18.json
"""
from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from evaluation.volume_parts_cv import fit_calibrator
from inference.volume_backend import density_for, load_densities, parts_volume_for_detection


def _iou(a, b) -> float:
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def _load_metric(path: Path):
    spec = importlib.util.spec_from_file_location("astramis_score", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def score(metric, labels: Path, csv_path: Path) -> dict:
    solution = metric.labels_to_solution(labels)
    sub = pd.read_csv(csv_path)
    total = float(metric.score(solution, sub, "image_id"))
    gt = {str(i): metric._parse(c, False, str(i)) for i, c in zip(solution["image_id"], solution["objects"])}
    rows = dict(zip(sub["image_id"].astype(str), sub["prediction"]))
    pred = {i: metric._parse(rows.get(i, "none"), True, i) for i in gt}
    parts = metric.components(gt, pred)
    assert abs(parts["Score"] - total) < 1e-5
    return parts


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, type=Path)
    ap.add_argument("--candidate", required=True, type=Path)
    ap.add_argument("--data", default="datasets/raw", type=Path)
    ap.add_argument("--rows", default="reports/volume_parts_gt_rows.csv", type=Path)
    ap.add_argument("--score-py", default=Path.home() / "Cleaner/references/score.py", type=Path)
    ap.add_argument("--conf", type=float, default=0.08)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--out", default="reports/volume_parts_fold2_noval18.json", type=Path)
    ap.add_argument("--labels", default=None, type=Path, help="val_labels.json (default: under --run)")
    ap.add_argument("--knots", default="", help="comma-separated ml knots, empty = none")
    ap.add_argument("--material-noise", type=float, default=None)
    args = ap.parse_args()
    from ultralytics import YOLO

    split = json.loads((args.run / "logs/object_id_split.json").read_text())
    train_ids = set(split["train_object_ids"])
    val_ids = set(split["validation_object_ids"])
    assert not train_ids & val_ids
    rows = pd.read_csv(args.rows)
    kw = {"knots_ml": tuple(float(k) for k in args.knots.split(",") if k)}
    if args.material_noise is not None:
        kw["material_noise"] = args.material_noise
    cal = fit_calibrator(rows[rows["object_id"].isin(train_ids)].reset_index(drop=True), **kw)
    dens = load_densities(args.data / "densities.csv")
    camera = json.loads((args.data / "camera.json").read_text())
    detector = YOLO(str(args.run / "checkpoints/experiments/C_joint/detector.pt"))
    labels = args.labels or args.run / "logs/validation/C_joint/rgb_only/val_labels.json"
    val_frames = {Path(i["file_name"]).parent.name for i in json.loads(labels.read_text())["images"]}

    cand = pd.read_csv(args.candidate)
    out_rows, matched, unmatched = [], 0, 0
    variants = {"parts_volume": [], "parts_volume_mass_take": []}
    for image_id, cell in zip(cand["image_id"].astype(str), cand["prediction"].astype(str)):
        objs = [] if cell.strip().lower() in {"", "none"} else [p.split() for p in cell.split(";")]
        fdir = args.data / "real/train" / image_id
        if image_id not in val_frames or not objs:
            for v in variants.values():
                v.append({"image_id": image_id, "prediction": cell})
            continue
        depth = cv2.imread(str(fdir / "depth.png"), cv2.IMREAD_UNCHANGED)
        plane = np.asarray(json.loads((fdir / "meta.json").read_text())["floor_plane"], dtype=np.float64)
        res = detector.predict(source=str(fdir / "color.png"), imgsz=args.imgsz, conf=args.conf, max_det=30,
                               retina_masks=True, verbose=False)[0]
        boxes = res.boxes.xyxy.cpu().numpy() if res.boxes is not None else np.zeros((0, 4))
        masks = res.masks.data.cpu().numpy() > 0.5 if res.masks is not None else None
        cells = {k: [] for k in variants}
        for tok in objs:
            box = [float(t) for t in tok[:4]]
            form, material, take = tok[5], tok[6], int(tok[9])
            ious = [_iou(box, b) for b in boxes]
            k = int(np.argmax(ious)) if ious else -1
            if k >= 0 and ious[k] >= 0.9 and masks is not None:
                mask = masks[k]
                if mask.shape != depth.shape:
                    mask = cv2.resize(mask.astype(np.uint8), depth.shape[::-1], interpolation=cv2.INTER_NEAREST) > 0
                matched += 1
            else:
                mask = np.zeros(depth.shape, bool)
                x1, y1, x2, y2 = [int(round(v)) for v in box]
                mask[max(0, y1):y2, max(0, x1):x2] = True
                unmatched += 1
            vol, mass = parts_volume_for_detection(depth, mask, plane, camera, form, material, cal, dens)
            head = " ".join(tok[:7])
            cells["parts_volume"].append(f"{head} {vol:.2f} {mass:.2f} {take}")
            cells["parts_volume_mass_take"].append(f"{head} {vol:.2f} {mass:.2f} {int(mass <= 500.0)}")
        for k, v in variants.items():
            v.append({"image_id": image_id, "prediction": ";".join(cells[k])})

    metric = _load_metric(args.score_py)
    report = {"run": str(args.run), "candidate": str(args.candidate), "train_objects": len(train_ids),
              "validation_objects": sorted(val_ids), "matched_masks": matched, "box_fallback": unmatched,
              "calibrator_alpha": cal["alpha"], "baseline": score(metric, labels, args.candidate)}
    for k, v in variants.items():
        p = args.out.with_name(f"{args.out.stem}_{k}.csv")
        pd.DataFrame(v, columns=["image_id", "prediction"]).to_csv(p, index=False)
        report[k] = score(metric, labels, p)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, ensure_ascii=False))
    for k in ["baseline", *variants]:
        r = report[k]
        print(f"{k:24s} Score={r['Score']:.3f} D={r['D']:.4f} M={r['M']:.4f} V={r['V']:.4f} S={r['S']:.4f}")
    print(f"masks matched={matched} box_fallback={unmatched}")


if __name__ == "__main__":
    main()
