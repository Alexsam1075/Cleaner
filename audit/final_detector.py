"""Финальный детектор для теста: тот же рецепт, что у честных фолдов (YOLO11n-seg, 640, 18 эпох, real+synth, без валидации, last.pt),
но обучен на ВСЕХ 40 объектах train (все real_train кадры + все synth_train). real/test не используется."""
import json
import sys
from pathlib import Path

ROOT = Path("/home/cleaner/Cleaner")
sys.path.insert(0, str(ROOT.parent))
sys.path.insert(0, str(ROOT))
from datasets.prepare_yolo import prepare_yolo_dataset  # noqa: E402
from datasets.common import load_coco, frame_id  # noqa: E402

D = ROOT / "datasets/raw"
OUT = ROOT / "audit/final"


def main():
    from ultralytics import YOLO
    OUT.mkdir(parents=True, exist_ok=True)
    real = load_coco(D, "real_train")
    synth = load_coco(D, "synth_train")
    split = {"train_real_frames": [frame_id(i) for i in real["images"]],
             "train_synth_frames": [frame_id(i) for i in synth["images"]],
             "validation_real_frames": []}
    yaml_path = prepare_yolo_dataset(D, split, OUT / "yolo_all", train_real=True, train_synth=True, validation=False)
    model = YOLO(str(ROOT / "yolo11n-seg.pt"))
    model.train(data=str(yaml_path), epochs=18, imgsz=640, batch=8, workers=4, device=0, project=str(OUT), name="det_all",
                exist_ok=True, seed=0, deterministic=True, optimizer="AdamW", lr0=0.001, lrf=0.01, weight_decay=0.0005,
                amp=True, val=False, save=True, plots=False, close_mosaic=4, mosaic=0.2, fliplr=0.5)
    w = OUT / "det_all/weights/last.pt"
    print("DETECTOR", w, w.exists())
    json.dump({"weights": str(w), "train_real_frames": len(split["train_real_frames"]), "train_synth_frames": len(split["train_synth_frames"]),
               "recipe": "yolo11n-seg 640 e18 joint val=false last.pt"}, open(OUT / "detector_run.json", "w"), indent=1)


if __name__ == "__main__":
    main()
