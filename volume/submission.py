"""Recompute volume and mass in any submission CSV; everything else in each row is kept.

Each row's own form and material are used, so a better material classifier upstream feeds
straight into the volume. Masks are recovered by re-running the detector and matching boxes.

  python -m volume.submission --submission in.csv --out out.csv --detector detector.pt \
      [--data datasets/raw] [--calibrator checkpoints/volume_parts/calibrator.json] \
      [--labels val_labels.json [--oracle-material]]

``--labels`` also scores both files with the official score.py; ``--oracle-material`` replaces
each row's material by the matched ground-truth one (upper bound of what the volume can do).
"""
from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
from typing import Callable

import cv2
import numpy as np
import pandas as pd

try:
    from .estimator import VolumeEstimator
except ImportError:
    from volume.estimator import VolumeEstimator


def _iou(a, b) -> float:
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def rewrite_submission(df: pd.DataFrame, estimator: VolumeEstimator,
                       load_frame: Callable[[str], tuple[np.ndarray, np.ndarray]],
                       mask_for_box: Callable[[str, list[float]], np.ndarray],
                       material_for: Callable[[str, list[float], str], str] | None = None,
                       take_threshold_g: float | None = None) -> pd.DataFrame:
    """Return a copy of ``df`` with tokens 8-9 (volume, mass) recomputed for every object.

    ``take_threshold_g`` also replaces take by ``mass <= threshold`` (mass = volume x class density).
    """
    out = []
    for image_id, cell in zip(df["image_id"].astype(str), df["prediction"].astype(str)):
        if cell.strip().lower() in {"", "none"}:
            out.append({"image_id": image_id, "prediction": cell})
            continue
        depth, plane = load_frame(image_id)
        parts = []
        for tok in (p.split() for p in cell.split(";")):
            box = [float(t) for t in tok[:4]]
            form, material = tok[5], tok[6]
            if material_for is not None:
                material = material_for(image_id, box, material)
            r = estimator.estimate(depth, mask_for_box(image_id, box), plane, form, material)
            take = tok[9] if take_threshold_g is None else str(int(r.mass_g <= take_threshold_g))
            parts.append(" ".join(tok[:7] + [f"{r.volume_ml:.2f}", f"{r.mass_g:.2f}", take]))
        out.append({"image_id": image_id, "prediction": ";".join(parts)})
    return pd.DataFrame(out, columns=["image_id", "prediction"])


def _frame_dir(data: Path, image_id: str) -> Path:
    for split in ("test", "train"):
        d = data / "real" / split / image_id
        if d.is_dir():
            return d
    raise FileNotFoundError(image_id)


class DetectorMasks:
    """Predicted masks by re-running the detector once per frame; box mask if nothing matches."""

    def __init__(self, data: Path, weights: Path, conf: float = 0.08, imgsz: int = 640, min_iou: float = 0.9):
        from ultralytics import YOLO
        self.data, self.model, self.conf, self.imgsz, self.min_iou = data, YOLO(str(weights)), conf, imgsz, min_iou
        self.cache: dict[str, tuple[np.ndarray, np.ndarray | None]] = {}
        self.matched = self.fallback = 0

    def __call__(self, image_id: str, box: list[float]) -> np.ndarray:
        fdir = _frame_dir(self.data, image_id)
        if image_id not in self.cache:
            res = self.model.predict(source=str(fdir / "color.png"), imgsz=self.imgsz, conf=self.conf,
                                     max_det=30, retina_masks=True, verbose=False)[0]
            boxes = res.boxes.xyxy.cpu().numpy() if res.boxes is not None else np.zeros((0, 4))
            masks = res.masks.data.cpu().numpy() > 0.5 if res.masks is not None else None
            self.cache = {image_id: (boxes, masks)}
        boxes, masks = self.cache[image_id]
        h, w = cv2.imread(str(fdir / "depth.png"), cv2.IMREAD_UNCHANGED).shape
        ious = [_iou(box, b) for b in boxes]
        if ious and masks is not None and max(ious) >= self.min_iou:
            self.matched += 1
            m = masks[int(np.argmax(ious))]
            return m if m.shape == (h, w) else cv2.resize(m.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST) > 0
        self.fallback += 1
        m = np.zeros((h, w), bool)
        x1, y1, x2, y2 = (int(round(v)) for v in box)
        m[max(0, y1):y2, max(0, x1):x2] = True
        return m


def _gt_material_lookup(labels: Path) -> Callable[[str, list[float], str], str]:
    lab = json.loads(labels.read_text(encoding="utf-8"))
    names = {i["id"]: Path(i["file_name"]).parent.name for i in lab["images"]}
    gt: dict[str, list] = {}
    for a in lab["annotations"]:
        x, y, w, h = a["bbox"]
        gt.setdefault(names[a["image_id"]], []).append(([x, y, x + w, y + h], a["attributes"]["material"]))

    def lookup(image_id: str, box: list[float], predicted: str) -> str:
        cands = gt.get(image_id, [])
        if not cands:
            return predicted
        best = max(cands, key=lambda c: _iou(box, c[0]))
        return best[1] if _iou(box, best[0]) >= 0.5 else predicted
    return lookup


def _score(score_py: Path, labels: Path, csv: Path) -> dict:
    spec = importlib.util.spec_from_file_location("astramis_score", score_py)
    metric = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(metric)
    solution = metric.labels_to_solution(labels)
    sub = pd.read_csv(csv)
    gt = {str(i): metric._parse(c, False, str(i)) for i, c in zip(solution["image_id"], solution["objects"])}
    rows = dict(zip(sub["image_id"].astype(str), sub["prediction"]))
    pred = {i: metric._parse(rows.get(i, "none"), True, i) for i in gt}
    return metric.components(gt, pred)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--submission", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--detector", required=True, type=Path)
    ap.add_argument("--data", default=Path("datasets/raw"), type=Path)
    ap.add_argument("--calibrator", default=Path("checkpoints/volume_parts/calibrator.json"), type=Path)
    ap.add_argument("--conf", type=float, default=0.08)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--labels", type=Path, default=None)
    ap.add_argument("--oracle-material", action="store_true")
    ap.add_argument("--take-from-mass", type=float, default=None, metavar="G",
                    help="replace take by mass <= G grams (mass = volume x class density)")
    ap.add_argument("--score-py", type=Path, default=Path(__file__).resolve().parents[1] / "references/score.py")
    args = ap.parse_args()
    if args.oracle_material and not args.labels:
        ap.error("--oracle-material needs --labels")
    est = VolumeEstimator.load(args.calibrator, args.data / "densities.csv", args.data / "camera.json")
    masks = DetectorMasks(args.data, args.detector, args.conf, args.imgsz)

    def load_frame(image_id: str):
        fdir = _frame_dir(args.data, image_id)
        depth = cv2.imread(str(fdir / "depth.png"), cv2.IMREAD_UNCHANGED)
        plane = np.asarray(json.loads((fdir / "meta.json").read_text())["floor_plane"], dtype=np.float64)
        return depth, plane

    material_for = _gt_material_lookup(args.labels) if args.oracle_material else None
    out = rewrite_submission(pd.read_csv(args.submission), est, load_frame, masks, material_for,
                             args.take_from_mass)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out, index=False)
    print(f"wrote {args.out}  masks matched={masks.matched} box_fallback={masks.fallback}")
    if args.labels:
        for name, path in (("input", args.submission), ("output", args.out)):
            r = _score(args.score_py, args.labels, path)
            print(f"{name:6s} Score={r['Score']:.3f} D={r['D']:.4f} M={r['M']:.4f} V={r['V']:.4f} S={r['S']:.4f}")


if __name__ == "__main__":
    main()
