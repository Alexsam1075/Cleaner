"""YOLO-mask check (no test data): for folds 1-3 (fold detectors + their validation real frames of real_train),
match detections to GT objects (box IoU>=0.5), compute geometry (+gi_env) on the YOLO mask, and compare
VolumeModel (trained on the fold's train objects) vs the previous ridge calibrator, true form/material,
per frame and with series median. GPU use: YOLO n-seg only."""
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

ROOT = Path("/home/cleaner/Cleaner")
HERE = ROOT / "agents2/volume"
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "references"))
import score as S  # noqa: E402
from evaluation.volume_parts_cv import apply_calibrator, fit_calibrator  # noqa: E402
from predict_volume import VolumeModel  # noqa: E402
from volume_cv import breakdown, fmt, series_median  # noqa: E402

D = ROOT / "datasets/raw"
FOLDS = {1: (ROOT / "configs/object_id_split.json", ROOT / "checkpoints/experiments/D_domain_randomized/detector.pt"),
         2: (ROOT / "checkpoints/score83_cv_fold2_noval18/configs/object_id_split.json", ROOT / "checkpoints/score83_cv_fold2_noval18/checkpoints/experiments/C_joint/detector.pt"),
         3: (ROOT / "checkpoints/score83_cv_fold3_noval18/configs/object_id_split.json", ROOT / "checkpoints/score83_cv_fold3_noval18/checkpoints/experiments/C_joint/detector.pt")}


def poly_mask(segs, h, w):
    m = np.zeros((h, w), np.uint8)
    for s in segs:
        cv2.fillPoly(m, [np.asarray(s, np.float32).reshape(-1, 2).round().astype(np.int32)], 1)
    return m > 0


def main():
    from ultralytics import YOLO
    cam = json.load(open(D / "camera.json"))
    lab = json.load(open(D / "labels/real_train.json"))
    imgs = {i["id"]: Path(i["file_name"]).parent.name for i in lab["images"]}
    by_frame = {}
    for a in lab["annotations"]:
        by_frame.setdefault(imgs[a["image_id"]], []).append(a)
    cache = HERE / "yolo_check_rows.csv"
    if cache.exists():
        d = pd.read_csv(cache)
    else:
        rows = []
        vm0 = VolumeModel()
        for fold, (split_p, det_p) in FOLDS.items():
            split = json.load(open(split_p))
            det = YOLO(str(det_p))
            for fid in split["validation_real_frames"]:
                anns = by_frame.get(fid, [])
                if not anns:
                    continue
                fd = D / "real/train" / fid
                depth = cv2.imread(str(fd / "depth.png"), cv2.IMREAD_UNCHANGED)
                plane = np.asarray(json.load(open(fd / "meta.json"))["floor_plane"], float)
                r = det.predict(str(fd / "color.png"), imgsz=640, conf=0.05, retina_masks=True, verbose=False, device=0)[0]
                boxes = r.boxes.xyxy.cpu().numpy() if r.boxes is not None else np.zeros((0, 4))
                for a in anns:
                    x, y, w, h = a["bbox"]
                    gb = (x, y, x + w, y + h)
                    at = a["attributes"]
                    ious = [S._iou(tuple(b), gb) for b in boxes]
                    if not ious or max(ious) < 0.5 or r.masks is None:
                        continue
                    ym = r.masks.data[int(np.argmax(ious))].cpu().numpy() > 0.5
                    if ym.shape != depth.shape:
                        ym = cv2.resize(ym.astype(np.uint8), (depth.shape[1], depth.shape[0]), interpolation=cv2.INTER_NEAREST) > 0
                    geo = vm0.geometry(depth, ym, plane, at["form"], cam)
                    rows.append({"fold": fold, "image_id": fid, "object_id": at["object_id"], "form": at["form"], "material": at["material"],
                                 "volume_ml": float(at["volume_ml"]), **{k: (float(v) if not isinstance(v, bool) else float(v)) for k, v in geo.items()}})
                print("fold", fold, fid, len(rows), flush=True)
            del det
        d = pd.DataFrame(rows)
        d.to_csv(cache, index=False)
    print("matched detections:", len(d), "objects:", d.object_id.nunique())
    gt_rows = VolumeModel.training_rows()
    p_new, p_old, p_b4 = np.zeros(len(d)), np.zeros(len(d)), np.zeros(len(d))
    for fold, (split_p, _) in FOLDS.items():
        split = json.load(open(split_p))
        train_objs = set(split["train_object_ids"])
        sel = (d.fold == fold).values
        assert not (set(d.object_id[sel]) & train_objs), "val object in train"
        vm = VolumeModel().fit(train_objs)
        cal = fit_calibrator(gt_rows[gt_rows.object_id.isin(train_objs)])
        for i in np.where(sel)[0]:
            r = d.iloc[i].to_dict()
            p_new[i] = vm.predict(r, r["form"], r["material"])
            r2 = dict(r)
            r2.pop("gi_env_ml")
            p_b4[i] = vm.predict(r2, r["form"], r["material"])
            p_old[i] = apply_calibrator(cal, r, r["form"], r["material"])
    sid = __import__("volume_cv").series_ids(d)
    for name, p in (("YOLO masks | previous ridge | frame", p_old), ("YOLO masks | VolumeModel base4 | frame", p_b4), ("YOLO masks | VolumeModel ref5 | frame", p_new),
                    ("YOLO masks | previous ridge | series median", series_median(p_old, sid)), ("YOLO masks | VolumeModel base4 | series median", series_median(p_b4, sid)),
                    ("YOLO masks | VolumeModel ref5 | series median", series_median(p_new, sid))):
        print(fmt(name, breakdown(d, p)), flush=True)
    for fold in (1, 2, 3):
        s = (d.fold == fold).values
        print(f"  fold {fold}: old {breakdown(d[s], p_old[s])['V']:.3f}  base4 {breakdown(d[s], p_b4[s])['V']:.3f}  ref5 {breakdown(d[s], p_new[s])['V']:.3f}  (n={s.sum()})")
    d.assign(pred_old=p_old, pred_base4=p_b4, pred_ref5=p_new).to_csv(HERE / "yolo_check_preds.csv", index=False)


if __name__ == "__main__":
    main()
