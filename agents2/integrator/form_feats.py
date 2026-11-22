"""Признаки v2 формы (DINOv2 / глубина / FastSAM из agents2/form/extract.py) для МОИХ детекций:
audit/cache/fold{1,2,3}_dets.pkl, fold4_dets_holdout4.pkl и cache/test_dets.pkl. Маски восстанавливаются повторным прогоном
детектора фолда (сопоставление по рамке, как в extract.det_samples). Выход: cache/form_v2_fold{k}.npz, cache/form_v2_test.npz
(порядок = порядок детекций в pkl). Только инференс; меток теста нет."""
import argparse
import json
import pickle
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

ROOT = Path("/home/cleaner/Cleaner")
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "agents2/form"))
sys.path.insert(0, str(ROOT / "audit"))
from extract import Dino, Sam, process, DEPTH_KEYS, DINO_KEYS, SAM_KEYS  # noqa: E402
from full_pipeline import FOLDS, C  # noqa: E402

D = ROOT / "datasets/raw"
DET4 = Path("/home/cleaner/CleanerVolume/holdout/det/weights/last.pt")
DET_TEST = ROOT / "audit/final/det_all/weights/last.pt"


def samples_for(dets, det_path, img_dir, conf, iou, max_det):
    from ultralytics import YOLO
    det = YOLO(str(det_path))
    frames = sorted({d["frame"] for d in dets})
    masks = {}
    for fid in frames:
        r = det.predict(str(img_dir / fid / "color.png"), imgsz=640, conf=conf, iou=iou, max_det=max_det, retina_masks=True, verbose=False, device=0)[0]
        n = 0 if r.boxes is None else len(r.boxes)
        for j in range(n):
            b = r.boxes.xyxy[j].cpu().numpy().astype(float)
            m = r.masks.data[j].cpu().numpy() > 0.5 if r.masks is not None else None
            masks.setdefault(fid, []).append((b, m))
    out, miss = [], 0
    for d in dets:
        best, bm = 1e9, None
        for b, m in masks.get(d["frame"], []):
            dist = float(np.abs(b - np.array(d["box"])).max())
            if dist < best:
                best, bm = dist, m
        bb = d["box"]
        if bm is None or best > 2.0 or bm.sum() < 10:
            miss += 1
            bm = np.zeros((720, 1280), bool)
            bm[int(bb[1]):int(bb[3]) + 1, int(bb[0]):int(bb[2]) + 1] = True
        if bm.shape != (720, 1280):
            bm = cv2.resize(bm.astype(np.uint8), (1280, 720), interpolation=cv2.INTER_NEAREST) > 0
        out.append((img_dir / d["frame"], bm, [bb[0], bb[1], bb[2] - bb[0], bb[3] - bb[1]]))
    print(f"{det_path.parent.parent.name}: {len(dets)} детекций, без маски {miss}", flush=True)
    del det
    torch.cuda.empty_cache()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--what", nargs="+", default=["1", "2", "3", "4", "test"])
    a = ap.parse_args()
    dino, sam = Dino(), Sam()
    for what in a.what:
        if what == "test":
            dets, _, _ = pickle.load(open(HERE / "cache/test_dets.pkl", "rb"))
            samples = samples_for(dets, DET_TEST, D / "real/test", 0.01, 0.7, 60)
        elif what == "4":
            dets, _ = pickle.load(open(C / "fold4_dets_holdout4.pkl", "rb"))
            samples = samples_for(dets, DET4, D / "real/train", 0.08, 0.7, 30)
        else:
            dets, _ = pickle.load(open(C / f"fold{what}_dets.pkl", "rb"))
            samples = samples_for(dets, FOLDS[int(what)][1], D / "real/train", 0.08, 0.7, 30)
        res = process(samples, None, dino, sam, what)
        out_p = HERE / "cache" / f"form_v2_{'test' if what == 'test' else 'fold' + what}.npz"
        np.savez_compressed(out_p, **res, depth_keys=np.array(DEPTH_KEYS), dino_keys=np.array(DINO_KEYS), sam_keys=np.array(SAM_KEYS))
        print("saved", out_p, {k: v.shape for k, v in res.items()}, flush=True)


if __name__ == "__main__":
    main()
