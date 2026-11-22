"""Детектор крупнее: yolo11s-seg, 960 px, 40 эпох, real+synth, без валидации (last.pt).
--fold 2|3 : обучение на train-объектах фолда (для честной проверки через score.py);
--fold 0   : на всех 40 объектах (для сабмита). real/test не используется."""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path("/home/cleaner/Cleaner")
sys.path.insert(0, str(ROOT))
from datasets.prepare_yolo import prepare_yolo_dataset  # noqa: E402
from datasets.common import load_coco, frame_id  # noqa: E402

D = ROOT / "datasets/raw"
OUT = ROOT / "audit/final"
SPLITS = {2: ROOT / "checkpoints/score83_cv_fold2_noval18/configs/object_id_split.json",
          3: ROOT / "checkpoints/score83_cv_fold3_noval18/configs/object_id_split.json"}


def main():
    from ultralytics import YOLO
    ap = argparse.ArgumentParser()
    ap.add_argument("--fold", type=int, required=True)
    ap.add_argument("--model", default="yolo11s-seg.pt")
    ap.add_argument("--imgsz", type=int, default=960)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch", type=int, default=8)
    a = ap.parse_args()
    if a.fold == 0:
        real, synth = load_coco(D, "real_train"), load_coco(D, "synth_train")
        split = {"train_real_frames": [frame_id(i) for i in real["images"]], "train_synth_frames": [frame_id(i) for i in synth["images"]],
                 "validation_real_frames": []}
    else:
        split = json.load(open(SPLITS[a.fold]))
    name = f"big_f{a.fold}"
    yaml_path = prepare_yolo_dataset(D, split, OUT / f"yolo_{name}", train_real=True, train_synth=True, validation=False)
    model = YOLO(a.model)
    model.train(data=str(yaml_path), epochs=a.epochs, imgsz=a.imgsz, batch=a.batch, workers=4, device=0, project=str(OUT), name=name,
                exist_ok=True, seed=0, deterministic=True, optimizer="AdamW", lr0=0.001, lrf=0.01, weight_decay=0.0005,
                amp=True, val=False, save=True, plots=False, close_mosaic=8, mosaic=0.3, fliplr=0.5, hsv_v=0.4, scale=0.3)
    print("DONE", OUT / name / "weights/last.pt", flush=True)


if __name__ == "__main__":
    main()
