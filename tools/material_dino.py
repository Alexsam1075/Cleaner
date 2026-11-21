"""Материал по признакам DINOv2 (замороженный бэкбон + логистическая регрессия).

Почему так: реальных физических объектов всего 40, и дообучать сеть целиком на них — переобучение.
Замороженный DINOv2 даёт общие признаки текстуры/блеска, а сверху учится только линейная голова.

1) Проверка по object_id (честная: отложенные предметы целиком не видны при обучении):
    python tools/material_dino.py cv --data dataset --out mat_dino
   печатает M (среднее по материалам, как в метрике) для разных C и с/без synth.
2) Переписать материал в готовом сабмите:
    python tools/material_dino.py apply --data dataset --submission submission.csv --out submission_mat.csv
   (обучается на всём real+synth с лучшими настройками из cv, если есть mat_dino/best.json)

Признаки кэшируются в --cache (по умолчанию mat_dino/emb_*.npz), повторный запуск быстрый.
Бэкбон: --backbone dinov2_vitb14 (по умолчанию) | dinov2_vits14 | dinov2_vitl14 | hist (без torch, для отладки).
Нужны: torch, torchvision, scikit-learn; веса качаются через torch.hub (facebookresearch/dinov2).
"""
import argparse
import csv
import json
import os

import numpy as np
from PIL import Image

MATS = ["metal", "wood", "plastic", "glass", "paper", "mineral"]


# ---------------- данные ----------------

def load_objects(data, labels_path, src):
    d = json.load(open(labels_path))
    files = {im["id"]: im["file_name"] for im in d["images"]}
    out = []
    for a in d["annotations"]:
        x, y, w, h = a["bbox"]
        at = a["attributes"]
        out.append(dict(path=os.path.join(data, files[a["image_id"]]), box=[x, y, x + w, y + h],
                        material=at["material"], object_id=at["object_id"], src=src,
                        frame=os.path.basename(os.path.dirname(files[a["image_id"]]))))
    return out


def crops(path, box, size=224):
    """Два вида одного объекта: плотная рамка (+8%) и рамка с контекстом (+35%), оба квадратные."""
    img = Image.open(path).convert("RGB")
    W, H = img.size
    x1, y1, x2, y2 = box
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    side = max(x2 - x1, y2 - y1, 16)
    out = []
    for pad in (1.08, 1.35):
        s = side * pad / 2
        c = img.crop((int(cx - s), int(cy - s), int(cx + s), int(cy + s)))  # за краем — чёрное
        out.append(c.resize((size, size), Image.BICUBIC))
    return out


# ---------------- признаки ----------------

class Backbone:
    def __init__(self, name, device=None):
        self.name = name
        if name == "hist":
            return
        import torch
        self.torch = torch
        self.dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = torch.hub.load("facebookresearch/dinov2", name).to(self.dev).eval()
        self.mean = torch.tensor([0.485, 0.456, 0.406], device=self.dev).view(1, 3, 1, 1)
        self.std = torch.tensor([0.229, 0.224, 0.225], device=self.dev).view(1, 3, 1, 1)

    def __call__(self, pil_list):
        if self.name == "hist":   # отладочный вариант без torch: HSV-гистограмма
            feats = []
            for p in pil_list:
                hsv = np.asarray(p.convert("HSV")).reshape(-1, 3)
                h = np.histogramdd(hsv, bins=(8, 4, 4), range=((0, 256),) * 3)[0].ravel()
                feats.append(h / h.sum())
            return np.array(feats, dtype=np.float32)
        torch = self.torch
        x = torch.stack([torch.from_numpy(np.asarray(p, dtype=np.float32) / 255.0).permute(2, 0, 1) for p in pil_list])
        x = (x.to(self.dev) - self.mean) / self.std
        out = []
        with torch.no_grad():
            for flip in (False, True):            # TTA: зеркало
                xx = torch.flip(x, dims=[3]) if flip else x
                f = self.model.forward_features(xx)
                cls = f["x_norm_clstoken"]
                pat = f["x_norm_patchtokens"].mean(1)
                out.append(torch.cat([cls, pat], 1))
        return ((out[0] + out[1]) / 2).float().cpu().numpy()


def embed(objs, bb, cache, bs=32):
    if cache and os.path.exists(cache):
        z = np.load(cache, allow_pickle=True)
        if list(z["keys"]) == [o["path"] + str(o["box"]) for o in objs]:
            return z["emb"]
    embs = []
    for i in range(0, len(objs), bs):
        chunk = objs[i:i + bs]
        tight, ctx = [], []
        for o in chunk:
            a, b = crops(o["path"], o["box"])
            tight.append(a)
            ctx.append(b)
        e = np.concatenate([bb(tight), bb(ctx)], 1)
        embs.append(e)
        print(f"  признаки {min(i + bs, len(objs))}/{len(objs)}", flush=True)
    emb = np.concatenate(embs, 0)
    if cache:
        os.makedirs(os.path.dirname(cache) or ".", exist_ok=True)
        np.savez(cache, emb=emb, keys=np.array([o["path"] + str(o["box"]) for o in objs]))
    return emb


# ---------------- модель ----------------

def norm(e):
    return e / (np.linalg.norm(e, axis=1, keepdims=True) + 1e-8)


def fit(Xtr, ytr, C, w=None):
    from sklearn.linear_model import LogisticRegression
    m = LogisticRegression(C=C, max_iter=3000, class_weight="balanced")
    m.fit(norm(Xtr), ytr, sample_weight=w)
    return m


def proba(m, X):
    p = np.zeros((len(X), len(MATS)))
    p[:, m.classes_] = m.predict_proba(norm(X))
    return p


def macro_acc(y, pred):
    return float(np.mean([np.mean(pred[y == c] == c) for c in range(len(MATS)) if np.any(y == c)]))


def run_cv(a):
    real = load_objects(a.data, os.path.join(a.data, "labels", "real_train.json"), "real")
    syn = load_objects(a.data, os.path.join(a.data, "labels", "synth_train.json"), "synth") if not a.no_synth else []
    objs = real + syn
    bb = Backbone(a.backbone)
    E = embed(objs, bb, os.path.join(a.out, f"emb_train_{a.backbone}.npz"))
    y = np.array([MATS.index(o["material"]) for o in objs])
    gid = np.array([o["object_id"] for o in objs])
    is_real = np.array([o["src"] == "real" for o in objs])
    ids = np.array(sorted(set(gid[is_real])))
    results = {}
    for use_syn in ([False, True] if syn else [False]):
        for C in a.C:
            accs = []
            for seed in range(a.seeds):
                perm = np.random.RandomState(seed).permutation(ids)
                pred = np.full(len(objs), -1)
                for fo in np.array_split(perm, a.folds):
                    va = np.isin(gid, fo) & is_real
                    tr = ~np.isin(gid, fo) & (is_real | use_syn)
                    w = np.where(is_real[tr], 1.0, a.synth_weight)
                    m = fit(E[tr], y[tr], C, w)
                    pred[va] = proba(m, E[va]).argmax(1)
                accs.append(macro_acc(y[is_real], pred[is_real]))
            key = f"C={C} synth={'yes' if use_syn else 'no'}"
            results[key] = (float(np.mean(accs)), float(np.std(accs)), C, use_syn)
            print(f"{key:24s} M = {np.mean(accs):.3f} ± {np.std(accs):.3f}", flush=True)
    best = max(results.values(), key=lambda r: r[0])
    json.dump(dict(backbone=a.backbone, C=best[2], synth=best[3], M=best[0], all={k: v[:2] for k, v in results.items()}),
              open(os.path.join(a.out, "best.json"), "w"), ensure_ascii=False, indent=1)
    print("лучшее:", best, "-> записано", os.path.join(a.out, "best.json"))


# ---------------- применение к сабмиту ----------------

def run_apply(a):
    cfg = dict(backbone=a.backbone, C=a.C[0], synth=not a.no_synth)
    bj = os.path.join(a.out, "best.json")
    if os.path.exists(bj):
        cfg.update(json.load(open(bj)))
        print("настройки из", bj, cfg.get("M"))
    real = load_objects(a.data, os.path.join(a.data, "labels", "real_train.json"), "real")
    syn = load_objects(a.data, os.path.join(a.data, "labels", "synth_train.json"), "synth") if cfg["synth"] else []
    objs = real + syn
    bb = Backbone(cfg["backbone"])
    E = embed(objs, bb, os.path.join(a.out, f"emb_train_{cfg['backbone']}.npz"))
    y = np.array([MATS.index(o["material"]) for o in objs])
    w = np.array([1.0 if o["src"] == "real" else a.synth_weight for o in objs])
    m = fit(E, y, cfg["C"], w)

    rows = list(csv.DictReader(open(a.submission, newline="")))
    tobj, where = [], []
    for ri, r in enumerate(rows):
        cell = r["prediction"].strip()
        if not cell or cell.lower() == "none":
            continue
        for oi, part in enumerate(cell.split(";")):
            t = part.split()
            path = os.path.join(a.data, "real", a.split, r["image_id"], "color.png")
            tobj.append(dict(path=path, box=[float(v) for v in t[:4]]))
            where.append((ri, oi))
    Et = embed(tobj, bb, os.path.join(a.out, f"emb_{os.path.basename(a.submission)}_{cfg['backbone']}.npz"))
    P = proba(m, Et)
    changed = 0
    parts = {ri: rows[ri]["prediction"].split(";") for ri, _ in where}
    for (ri, oi), p in zip(where, P):
        t = parts[ri][oi].split()
        new = MATS[int(p.argmax())]
        changed += t[6] != new
        t[6] = new
        parts[ri][oi] = " ".join(t)
    for ri, ps in parts.items():
        rows[ri]["prediction"] = ";".join(ps)
    with open(a.out_csv, "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=["image_id", "prediction"])
        wr.writeheader()
        for r in rows:
            wr.writerow({"image_id": r["image_id"], "prediction": r["prediction"]})
    np.savez(os.path.join(a.out, "test_material_proba.npz"), proba=P, where=np.array(where))
    print(f"объектов: {len(where)}, материал изменён у {changed}; записано {a.out_csv}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["cv", "apply"])
    ap.add_argument("--data", default="dataset")
    ap.add_argument("--backbone", default="dinov2_vitb14")
    ap.add_argument("--out", default="mat_dino", help="папка для кэша и best.json")
    ap.add_argument("--C", type=float, nargs="+", default=[0.3, 1.0, 3.0, 10.0])
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--no-synth", action="store_true")
    ap.add_argument("--synth-weight", type=float, default=0.5)
    ap.add_argument("--submission")
    ap.add_argument("--split", default="test")
    ap.add_argument("--out-csv", default="submission_mat.csv")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    run_cv(a) if a.mode == "cv" else run_apply(a)


if __name__ == "__main__":
    main()
