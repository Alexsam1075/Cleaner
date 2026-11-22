"""Run each fold's detector on ALL real train frames (val + train frames of the fold) at low conf, per-class NMS,
and cache per detection: box, conf, cls, mask stats, depth geometry, CLIP-L/SigLIP embeddings of 4 crop views,
IoU/form/oid of best GT match. Output: W/cache/fold{k}_dets.npz + .json"""
import argparse
import json
import sys

import cv2
import numpy as np
import torch

from common import C, D, FOLDS, FORMS, VIEWS, W, box_iou, gt_by_frame, load_split
from encoders import Encoders
from attr_features import geometry  # noqa: E402  (audit)
from mat_features2 import crop  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--folds", type=int, nargs="+", default=[1, 2, 3])
    ap.add_argument("--conf", type=float, default=0.01)
    a = ap.parse_args()
    from ultralytics import YOLO
    (W / "cache").mkdir(exist_ok=True)
    enc = Encoders()
    gt = gt_by_frame()
    frames = sorted(gt)
    for fold in a.folds:
        det = YOLO(str(FOLDS[fold][1]))
        rows, embs = [], {(t, v): [] for t in enc.m for v in VIEWS}
        pending = {v: [] for v in VIEWS}

        def flush():
            for v in VIEWS:
                if pending[v]:
                    arr = np.stack(pending[v])
                    for t in enc.m:
                        embs[(t, v)].append(enc.embed(t, arr))
                    pending[v] = []

        for fi, fid in enumerate(frames):
            fd = D / "real/train" / fid
            rgb = cv2.cvtColor(cv2.imread(str(fd / "color.png")), cv2.COLOR_BGR2RGB)
            depth = cv2.imread(str(fd / "depth.png"), cv2.IMREAD_UNCHANGED)
            plane = json.load(open(fd / "meta.json"))["floor_plane"]
            r = det.predict(str(fd / "color.png"), imgsz=640, conf=a.conf, iou=0.7, max_det=60, retina_masks=True, verbose=False, device=0)[0]
            n = 0 if r.boxes is None else len(r.boxes)
            for j in range(n):
                b = r.boxes.xyxy[j].cpu().numpy().astype(float)
                m = None
                if r.masks is not None:
                    m = r.masks.data[j].cpu().numpy() > 0.5
                    if m.shape != depth.shape:
                        m = cv2.resize(m.astype(np.uint8), (depth.shape[1], depth.shape[0]), interpolation=cv2.INTER_NEAREST) > 0
                mask_ok = m is not None and m.sum() >= 10
                if not mask_ok:
                    m = np.zeros(depth.shape, bool)
                    m[int(b[1]):int(b[3]) + 1, int(b[0]):int(b[2]) + 1] = True
                g = geometry(depth, m, plane)
                bbox = [b[0], b[1], b[2] - b[0], b[3] - b[1]]
                best, bj = 0.0, -1
                for k, o in enumerate(gt[fid]):
                    v = box_iou(b, o["box"])
                    if v > best:
                        best, bj = v, k
                rows.append({"fold": fold, "frame": fid, "box": b.tolist(), "conf": float(r.boxes.conf[j]), "cls": FORMS[int(r.boxes.cls[j])],
                             "mask_ok": bool(mask_ok), "mask_area": int(m.sum()), "box_area": float(bbox[2] * bbox[3]),
                             "geo": {k: float(v) for k, v in g.items()},
                             "gt_iou": best, "gt_form": gt[fid][bj]["form"] if bj >= 0 else None, "gt_oid": gt[fid][bj]["oid"] if bj >= 0 else None,
                             "gt_idx": bj})
                for v, (sc, msk) in VIEWS.items():
                    pending[v].append(crop(rgb, m, bbox, sc, msk)[0])
            if len(pending["c14"]) >= 64:
                flush()
            if fi % 40 == 0:
                print(f"fold {fold} frame {fi}/{len(frames)} dets so far {len(rows)}", flush=True)
        flush()
        out = {f"{t}_{v}": (np.concatenate(embs[(t, v)]) if embs[(t, v)] else np.zeros((0, 768), np.float32)) for (t, v) in embs}
        for (t, s), T in enc.text.items():
            out[f"text_{t}_{s}"] = T
        np.savez_compressed(W / "cache" / f"fold{fold}_dets.npz", **out)
        json.dump(rows, open(W / "cache" / f"fold{fold}_dets.json", "w"))
        print(f"fold {fold}: {len(rows)} detections cached", flush=True)
        del det
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
