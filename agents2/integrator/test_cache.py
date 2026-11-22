"""Кэш детекций и эмбеддингов на тесте (как в predict_test.py, только инференс real/test; никаких меток).
Сохраняет cache/test_dets.pkl: (dets, embs) в том же формате, что audit/cache/fold*_dets.pkl
(+ new_geo_by_form для обеих форм, conf -> score), чтобы сабмиты собирались без повторного прогона детектора."""
import json
import pickle
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

ROOT = Path("/home/cleaner/Cleaner")
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "agents/form"))
sys.path.insert(0, str(ROOT / "audit"))
from common import VIEWS  # noqa: E402
from encoders import Encoders  # noqa: E402
import full_pipeline as fp  # noqa: E402
from attr_features import geometry  # noqa: E402
from mat_features2 import crop  # noqa: E402

D = ROOT / "datasets/raw"
FORMS = ["bulk", "solid"]


def main():
    from ultralytics import YOLO
    sample = pd.read_csv(ROOT / "references/sample_submission.csv")
    frames = sample.image_id.astype(str).tolist()
    test_dir = D / "real/test"
    assert all((test_dir / f).is_dir() for f in frames)
    all_objs = sorted({m["oid"] for m in json.load(open(fp.C / "meta.json")) if m["src"] == "real_train"})
    det = YOLO(str(ROOT / "audit/final/det_all/weights/last.pt"))
    enc = Encoders()
    cam = json.load(open(D / "camera.json"))
    est, _ = fp.volume_estimator_for(all_objs, cam)
    dets, views = [], {v: [] for v in VIEWS}
    for k, fid in enumerate(frames):
        fd = test_dir / fid
        rgb = cv2.cvtColor(cv2.imread(str(fd / "color.png")), cv2.COLOR_BGR2RGB)
        depth = cv2.imread(str(fd / "depth.png"), cv2.IMREAD_UNCHANGED)
        plane = json.load(open(fd / "meta.json"))["floor_plane"]
        r = det.predict(str(fd / "color.png"), imgsz=640, conf=0.01, iou=0.7, max_det=60, retina_masks=True, verbose=False, device=0)[0]
        n = 0 if r.boxes is None else len(r.boxes)
        for j in range(n):
            b = r.boxes.xyxy[j].cpu().numpy().astype(float)
            m = None
            if r.masks is not None:
                m = r.masks.data[j].cpu().numpy() > 0.5
                if m.shape != depth.shape:
                    m = cv2.resize(m.astype(np.uint8), (depth.shape[1], depth.shape[0]), interpolation=cv2.INTER_NEAREST) > 0
            if m is None or m.sum() < 10:
                m = np.zeros(depth.shape, bool)
                m[int(b[1]):int(b[3]) + 1, int(b[0]):int(b[2]) + 1] = True
            g = geometry(depth, m, plane)
            det_form = FORMS[int(r.boxes.cls[j])]
            new_geo = {f: est.geometry(depth, m, np.asarray(plane, float), f) for f in FORMS}
            bbox = [b[0], b[1], b[2] - b[0], b[3] - b[1]]
            dets.append({"frame": fid, "box": b.tolist(), "score": float(r.boxes.conf[j]), "conf": float(r.boxes.conf[j]), "form": det_form,
                         "det_form": det_form, "geo": g, "geo_vol": g["vol"], "new_geo": new_geo[det_form], "new_geo_by_form": new_geo})
            for v, (sc, msk) in VIEWS.items():
                views[v].append(crop(rgb, m, bbox, sc, msk)[0])
        if k % 30 == 0:
            print(f"кадр {k}/{len(frames)}, детекций {len(dets)}", flush=True)
    embs = {}
    for t in enc.m:
        for v in VIEWS:
            embs[(t, v)] = enc.embed(t, np.stack(views[v])) if views[v] else np.zeros((0, 768), np.float32)
    (HERE / "cache").mkdir(exist_ok=True)
    pickle.dump((dets, embs, frames), open(HERE / "cache" / "test_dets.pkl", "wb"))
    print(f"сохранено {len(dets)} детекций на {len(frames)} кадрах")


if __name__ == "__main__":
    main()
