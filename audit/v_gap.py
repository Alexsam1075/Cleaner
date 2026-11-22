"""Где теряется V между «GT-маска + истинный материал» (0.77) и конвейером (0.71) на фолдах 1–3.
Для каждого валидационного GT-объекта, сопоставленного с детекцией (IoU ≥ 0.5):
  1) GT-маска,  истинный материал  -> калибратор фолда
  2) GT-маска,  предсказанный мат.
  3) YOLO-маска, истинный материал
  4) YOLO-маска, предсказанный мат.  (= конвейер)
плюс диагностика масок: IoU, доля маски вне GT, высота/объём сырых оценок."""
import json
import math
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

ROOT = Path("/home/cleaner/Cleaner")
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "references"))
import score as S  # noqa: E402
from evaluation.volume_parts_cv import apply_calibrator, fit_calibrator  # noqa: E402
from volume.estimator import VolumeEstimator  # noqa: E402
from inference.volume_backend import load_densities  # noqa: E402

D = ROOT / "datasets/raw"
C = ROOT / "audit/cache"
L3 = math.log(3)
FOLDS = {1: (ROOT / "configs/object_id_split.json", ROOT / "checkpoints/experiments/D_domain_randomized/detector.pt"),
         2: (ROOT / "checkpoints/score83_cv_fold2_noval18/configs/object_id_split.json", ROOT / "checkpoints/score83_cv_fold2_noval18/checkpoints/experiments/C_joint/detector.pt"),
         3: (ROOT / "checkpoints/score83_cv_fold3_noval18/configs/object_id_split.json", ROOT / "checkpoints/score83_cv_fold3_noval18/checkpoints/experiments/C_joint/detector.pt")}


def V(p, g):
    return max(0.0, 1 - abs(math.log(max(p, 1e-3) / g)) / L3)


def poly_mask(seg, h, w):
    m = np.zeros((h, w), np.uint8)
    for s in seg:
        cv2.fillPoly(m, [np.array(s, np.float32).reshape(-1, 2).round().astype(np.int32)], 1)
    return m > 0


def main():
    from ultralytics import YOLO
    lab = json.load(open(D / "labels/real_train.json"))
    ims = {i["id"]: i for i in lab["images"]}
    by_frame = {}
    for a in lab["annotations"]:
        by_frame.setdefault(Path(ims[a["image_id"]]["file_name"]).parent.name, []).append(a)
    cam = json.load(open(D / "camera.json"))
    rows_gt = pd.read_csv(ROOT / "reports/volume_parts_gt_rows.csv")
    rows_gt = rows_gt[rows_gt.split == "real"]
    out = []
    for fold, (sp, det_p) in FOLDS.items():
        split = json.load(open(sp))
        tr = rows_gt[rows_gt.object_id.isin(split["train_object_ids"])]
        est = VolumeEstimator(fit_calibrator(tr), load_densities(D / "densities.csv"), cam)
        sub = pd.read_csv(C / f"final_fold{fold}_E_D+border.csv")
        pm = {i: (S._parse(c, True, i) if str(c) != "none" else []) for i, c in zip(sub.image_id.astype(str), sub.prediction)}
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
                gmask = poly_mask(a["segmentation"], depth.shape[0], depth.shape[1])
                cand = [p for p in pm.get(fid, []) if S._iou(p["box"], gb) >= 0.5]
                if not cand:
                    continue
                p = max(cand, key=lambda q: S._iou(q["box"], gb))
                ious = [S._iou(tuple(b), gb) for b in boxes]
                if not ious or max(ious) < 0.5 or r.masks is None:
                    continue
                ym = r.masks.data[int(np.argmax(ious))].cpu().numpy() > 0.5
                if ym.shape != gmask.shape:
                    ym = cv2.resize(ym.astype(np.uint8), (gmask.shape[1], gmask.shape[0]), interpolation=cv2.INTER_NEAREST) > 0
                form = at["form"]
                geo_g = est.geometry(depth, gmask, plane, form)
                geo_y = est.geometry(depth, ym, plane, form)
                v = {}
                for mname, geo in (("gt", geo_g), ("yolo", geo_y)):
                    for matname, mat in (("true", at["material"]), ("pred", p["material"])):
                        v[f"{mname}_{matname}"] = est.estimate(None, None, None, form, mat, geometry=geo).volume_ml
                inter = (ym & gmask).sum()
                out.append({"fold": fold, "oid": at["object_id"], "form": form, "mat": at["material"], "pmat": p["material"], "gtv": at["volume_ml"],
                            "mask_iou": inter / max((ym | gmask).sum(), 1), "yolo_out": (ym & ~gmask).sum() / max(ym.sum(), 1),
                            "yolo_miss": (gmask & ~ym).sum() / max(gmask.sum(), 1),
                            "parts_gt": geo_g["parts_ml"], "parts_yolo": geo_y["parts_ml"], **v})
        print("fold", fold, "rows", sum(1 for o in out if o["fold"] == fold), flush=True)
    d = pd.DataFrame(out)
    d.to_csv(C / "v_gap_rows.csv", index=False)
    for k in ("gt_true", "gt_pred", "yolo_true", "yolo_pred"):
        d["V_" + k] = [V(p, g) for p, g in zip(d[k], d.gtv)]
    print("\n== V по 4 условиям (среднее по сопоставленным объектам, по фолдам) ==")
    print(d.groupby("fold")[["V_gt_true", "V_gt_pred", "V_yolo_true", "V_yolo_pred"]].mean().round(3).to_string())
    print("ВСЕ:", d[["V_gt_true", "V_gt_pred", "V_yolo_true", "V_yolo_pred"]].mean().round(3).to_dict())
    print("\nмаски YOLO: медиана IoU", round(d.mask_iou.median(), 3), "доля вне GT", round(d.yolo_out.median(), 3), "недобор", round(d.yolo_miss.median(), 3))
    print("отношение сырого parts_ml (YOLO/GT): медиана", round(float(np.median(d.parts_yolo / d.parts_gt)), 3),
          "p10/p90", np.round(np.percentile(d.parts_yolo / d.parts_gt, [10, 90]), 2).tolist())
    print("\n== Объекты, где маска YOLO стоит больше всего V ==")
    o = d.groupby("oid").agg(fold=("fold", "first"), form=("form", "first"), mat=("mat", "first"), n=("gtv", "size"), iou=("mask_iou", "median"),
                             out=("yolo_out", "median"), parts_ratio=("parts_yolo", "median"), V_gt=("V_gt_true", "mean"), V_yolo=("V_yolo_true", "mean"))
    o["parts_ratio"] = o["parts_ratio"] / d.groupby("oid").parts_gt.median()
    o["loss"] = (o.V_gt - o.V_yolo) * o.n
    print(o.sort_values("loss", ascending=False).head(10).round(3).to_string())


if __name__ == "__main__":
    main()
