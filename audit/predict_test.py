"""Сабмит на тест: тот же конвейер, что дал 83.6 на фолдах 1–3, но все модели обучены на ВСЕХ 40 объектах train.
Тестовые кадры (real/test) используются только для инференса: их метки не существуют и не читаются.

  рамки/маски : детектор YOLO11n-seg, обученный на всех train (audit/final_detector.py), conf ≥ 0.08
  форма       : FormModel агента (CLIP-L c14 + zero-shot, настройки по внутреннему CV); score = conf × p(форма);
                NMS 0.6 между классами, отсев рамок у края кадра (вариант E)
  материал    : линейная голова CLIP-L c14 ⊕ zero-shot CLIP-L+SigLIP (generic_syn), вес — внутренний 5-fold CV
  объём       : volume/ (части + калибратор, обученный на всех 40 объектах по GT-маскам), маска YOLO, предсказанный материал
  take        : калиброванный объём ≤ T, T подобран на 40 объектах по сбалансированной точности
  масса       : объём × плотность класса (не влияет на метрику)
Опция --two-class: вторая форма для неуверенных рамок (p_max ≤ 0.65) — по умолчанию выключена."""
import argparse
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

ROOT = Path("/home/cleaner/Cleaner")
sys.path.insert(0, str(ROOT / "agents/form"))
sys.path.insert(0, str(ROOT / "audit"))
from common import VIEWS  # noqa: E402
from encoders import Encoders  # noqa: E402
from form_cv import load_bank  # noqa: E402
from form_model import GEO_KEYS, FeatureBank, FormModel  # noqa: E402
from form_pipeline import touches_border  # noqa: E402
import full_pipeline as fp  # noqa: E402
from attr_features import geometry  # noqa: E402
from mat_features2 import crop  # noqa: E402

D = ROOT / "datasets/raw"
OUT = ROOT / "audit/final"
FORMS = ["bulk", "solid"]


def iou(a, b):
    w = min(a[2], b[2]) - max(a[0], b[0])
    h = min(a[3], b[3]) - max(a[1], b[1])
    if w <= 0 or h <= 0:
        return 0.0
    i = w * h
    return i / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - i)


def main():
    from ultralytics import YOLO
    ap = argparse.ArgumentParser()
    ap.add_argument("--detector", default=str(OUT / "det_all/weights/last.pt"))
    ap.add_argument("--conf", type=float, default=0.08)
    ap.add_argument("--two-class", action="store_true", help="вторая форма для неуверенных рамок (p_max <= 0.65)")
    ap.add_argument("--two-class-all", action="store_true", help="вторая форма для ВСЕХ рамок (эксплуатация AP)")
    ap.add_argument("--agg", type=float, default=0.90, help="агрегация объёма/take по сериям кадров (порог сходства; 0 = выкл)")
    ap.add_argument("--agg-gap", type=int, default=3)
    ap.add_argument("--prompts", default="rich", choices=list(fp.PROMPT_SETS))
    ap.add_argument("--take", default="hybrid", choices=["volume", "hybrid"], help="hybrid: metal/glass/mineral по объёму ≤ T, остальные по массе(плотность класса) ≤ 500")
    ap.add_argument("--out", default=str(ROOT / "audit/submission.csv"))
    ap.add_argument("--limit", type=int, default=0, help="только первые N кадров (smoke-тест)")
    ap.add_argument("--split", default="test", choices=["test", "train"], help="train: прогон по real/train для score.py check (диагностика)")
    a = ap.parse_args()

    if a.split == "test":
        sample = pd.read_csv(ROOT / "references/sample_submission.csv")
        frames = sample.image_id.astype(str).tolist()
    else:
        frames = sorted(p.name for p in (D / "real/train").iterdir() if p.is_dir())
    if a.limit:
        frames = frames[:a.limit]
    test_dir = D / "real" / a.split
    assert all((test_dir / f).is_dir() for f in frames)
    all_objs = sorted({m["oid"] for m in json.load(open(fp.C / "meta.json")) if m["src"] == "real_train"})
    assert len(all_objs) == 40

    # ---- детекции + вырезки на тестовых кадрах
    det = YOLO(a.detector)
    enc = Encoders()
    cam = json.load(open(D / "camera.json"))
    est, take_thr = fp.volume_estimator_for(all_objs, cam)
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
            dets.append({"frame": fid, "box": b.tolist(), "conf": float(r.boxes.conf[j]), "det_form": det_form, "geo": g, "new_geo": new_geo})
            for v, (sc, msk) in VIEWS.items():
                views[v].append(crop(rgb, m, bbox, sc, msk)[0])
        if k % 30 == 0:
            print(f"кадр {k}/{len(frames)}, детекций {len(dets)}", flush=True)
    emb = {}
    for t in enc.m:
        for v in VIEWS:
            emb[f"{t}_{v}"] = enc.embed(t, np.stack(views[v])) if views[v] else np.zeros((0, 768), np.float32)
    emb["geo"] = np.array([[d["geo"][k] for k in GEO_KEYS] for d in dets], np.float32).reshape(len(dets), len(GEO_KEYS))

    # ---- форма: FormModel на всех 40 объектах
    bank, y, oid, src, meta = load_bank()
    fm = FormModel(bank, y, oid, src, verbose=True).fit(all_objs)
    p_bulk = fm.predict_bank(FeatureBank(emb, bank.text))[:, 0] if dets else np.zeros(0)

    # ---- материал: голова + zero-shot на всех 40 объектах
    F, meta2 = fp.train_cache()
    enc2 = fp.Encoders()
    predict_mat, w_zs = fp.material_model(F, meta2, all_objs, enc2.text, a.prompts)
    embs = {(t, v): emb[f"{t}_{v}"] for t in fp.MODELS for v in fp.VIEWS}
    P_mat = predict_mat(emb["clipL_c14"], fp.zero_shot(embs, enc2.text, a.prompts)) if dets else np.zeros((0, 6))

    # ---- объём по кадру, затем агрегация по сериям (медиана объёма в группе «тот же предмет»)
    forms = ["bulk" if pb >= 0.5 else "solid" for pb in p_bulk]
    mats = [fp.MATS[int(pm.argmax())] for pm in P_mat]
    vols = np.array([est.estimate(None, None, None, f, m, geometry=d["new_geo"][f]).volume_ml for d, f, m in zip(dets, forms, mats)]) if dets else np.zeros(0)
    dens = np.array([est.estimate(None, None, None, f, m, geometry=d["new_geo"][f]).density_g_per_ml for d, f, m in zip(dets, forms, mats)]) if dets else np.zeros(0)
    if a.agg > 0 and dets:
        gid = fp.link_series(dets, {(t, v): emb[f"{t}_{v}"] for t in fp.MODELS for v in VIEWS}, a.agg, a.agg_gap)
        for g in set(gid):
            ii = np.where(gid == g)[0]
            vols[ii] = float(np.exp(np.median(np.log(np.maximum(vols[ii], 1.0)))))
        print(f"серии: {len(set(gid))} групп из {len(dets)} детекций, с ≥2 кадрами {sum(1 for g in set(gid) if (gid == g).sum() > 1)}")
    # ---- плотности классов по train-объектам (для массы и гибридного take)
    from collections import defaultdict as _dd
    by = _dd(list)
    seen = set()
    for m in meta2:
        if m["src"] == "real_train" and m["oid"] not in seen:
            seen.add(m["oid"])
            by[(m["form"], m["mat"])].append(m["mass"] / m["vol"])
            by[("*", m["mat"])].append(m["mass"] / m["vol"])
    cls_dens = {k: float(np.exp(np.mean(np.log(v)))) for k, v in by.items()}
    HEAVY = {"metal", "glass", "mineral"}
    # ---- сборка по кадрам (вариант E)
    per = defaultdict(list)
    for d, pb, form, matl, vol, dn in zip(dets, p_bulk, forms, mats, vols, dens):
        if d["conf"] < a.conf or touches_border(d["box"]):
            continue
        pmax = max(pb, 1 - pb)
        mass_cls = float(vol) * cls_dens.get((form, matl), cls_dens.get(("*", matl), 0.5))
        if a.take == "hybrid":
            take = int(vol <= take_thr) if matl in HEAVY else int(mass_cls <= 500)
        else:
            take = int(vol <= take_thr)
        per[d["frame"]].append({"box": d["box"], "score": d["conf"] * pmax, "form": form, "material": matl,
                                "volume": float(vol), "mass": mass_cls, "take": take, "conf": d["conf"], "pmax": pmax})
    rows = []
    stats = defaultdict(int)
    for fid in frames:
        keep = []
        for c in sorted(per.get(fid, []), key=lambda x: -x["score"]):
            if all(iou(c["box"], k["box"]) < 0.6 for k in keep):
                keep.append(c)
        out = list(keep)
        if a.two_class or a.two_class_all:
            for c in keep:
                if a.two_class_all or c["pmax"] <= 0.65:
                    out.append({**c, "form": "solid" if c["form"] == "bulk" else "bulk", "score": c["conf"] * (1 - c["pmax"])})
        out = sorted(out, key=lambda x: -x["score"])[:30]
        for o in out:
            stats[o["form"]] += 1
            stats[o["material"]] += 1
            stats["take" + str(o["take"])] += 1
        rows.append({"image_id": fid, "prediction": fp.serialize(out)})
    sub = pd.DataFrame(rows, columns=["image_id", "prediction"])
    sub.to_csv(a.out, index=False)
    n_none = int((sub.prediction == "none").sum())
    print(f"\nГотово: {a.out}  кадров {len(sub)}, пустых (none) {n_none}, объектов {sum(stats[f] for f in FORMS)}")
    print("форма/материал/take:", dict(stats), f"| вес zero-shot материала {w_zs}, порог take {take_thr:.0f} мл, форма cfg {fm.cfg}")
    sys.path.insert(0, str(ROOT / "references"))
    import score as S
    for i, c in zip(sub.image_id, sub.prediction):
        S._parse(c, True, i)
    print("формат проверен через score._parse: OK")


if __name__ == "__main__":
    main()
