"""Smoke test of the integrator API: FormClassifier.predict on single detections vs predict_bank on the cache (fold 2, 5 dets)."""
import json
import cv2
import numpy as np
from common import D, FOLDS, load_split
from bank import det_bank
from predict_form import FormClassifier, rank_and_filter, pool_series
from ultralytics import YOLO

sp = load_split(2)
fc = FormClassifier(topk=3, tune_thr=False).fit(sp["train_object_ids"])
rows, bank = det_bank(2)
pb = fc.predict_bank(bank)
det = YOLO(str(FOLDS[2][1]))
checked = 0
for i, r in enumerate(rows):
    if r["conf"] < 0.5 or checked >= 5:
        continue
    fd = D / "real/train" / r["frame"]
    bgr = cv2.imread(str(fd / "color.png"))
    depth = cv2.imread(str(fd / "depth.png"), cv2.IMREAD_UNCHANGED)
    plane = json.load(open(fd / "meta.json"))["floor_plane"]
    res = det.predict(str(fd / "color.png"), imgsz=640, conf=0.01, iou=0.7, max_det=60, retina_masks=True, verbose=False, device=0)[0]
    j = int(np.argmin([np.abs(b - np.array(r["box"])).max() for b in res.boxes.xyxy.cpu().numpy()]))
    m = res.masks.data[j].cpu().numpy() > 0.5
    if m.shape != depth.shape:
        m = cv2.resize(m.astype(np.uint8), (depth.shape[1], depth.shape[0]), interpolation=cv2.INTER_NEAREST) > 0
    embs = {("clipL", v): bank.emb[f"clipL_{v}"][i] for v in ("c14", "m10", "m14")}
    b = r["box"]
    p = fc.predict(embs, r["geo"], depth, m, plane, rgb_bgr=bgr, bbox=[b[0], b[1], b[2] - b[0], b[3] - b[1]])
    print(f"{r['frame']} gt {r['gt_form']} p_bulk api {p:.3f} vs bank {pb[i]:.3f}")
    checked += 1
dets = [{"frame": r["frame"], "box": r["box"], "conf": r["conf"], "p_bulk": float(p)} for r, p in zip(rows, pb)]
pooled = pool_series(dets, bank.emb["clipL_c14"])
kept = rank_and_filter([d for d in pooled if d["frame"] == "r_0041"])
print("r_0041:", [(d["form"], round(d["score"], 3), [int(v) for v in d["box"]]) for d in kept])
