"""Проверка классификатора материала на переобучение и утечки.
  1. Утечка через промпты: описания металла и других материалов я писал, глядя на train-предметы.
     Сравниваем с общими описаниями (только слово материала) — честный нижний вариант.
  2. Перестановочный тест: перемешиваем материалы между объектами; честный конвейер должен упасть до случайного уровня.
  3. Разрыв train/OOF для линейной головы.
  4. Чувствительность к температуре zero-shot и весу смешивания.
  5. Доверительный интервал bootstrap по объектам.
  6. Устойчивость к освещению (bright/dim/side).
  7. Разрыв масок: на кадрах валидации фолдов 1–3 берём маски детектора YOLO, обученного без этих предметов,
     и сравниваем с эталонными масками на тех же кадрах. Классификатор обучен только на train-объектах фолда.
Только real_train + synth_train, real/test не используется."""
import json
import math
import os
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as Fn

from mat_cv import MATS, fit_predict, weights
from mat_features2 import PROMPTS, TEMPLATES, crop

C = Path(__file__).resolve().parent / "cache"
D = Path(os.environ.get("ASTRAMIS_DATA", "dataset"))
ROOT = Path(__file__).resolve().parent.parent
DEV = "cuda" if torch.cuda.is_available() else "cpu"
VIEWS = {"m10": (1.1, True), "m14": (1.5, True), "c14": (1.5, False), "c20": (2.2, False)}
FOLDS = {
    1: (ROOT / "configs/object_id_split.json", ROOT / "checkpoints/experiments/D_domain_randomized/detector.pt"),
    2: (ROOT / "checkpoints/score83_cv_fold2_noval18/configs/object_id_split.json", ROOT / "checkpoints/score83_cv_fold2_noval18/checkpoints/experiments/C_joint/detector.pt"),
    3: (ROOT / "checkpoints/score83_cv_fold3_noval18/configs/object_id_split.json", ROOT / "checkpoints/score83_cv_fold3_noval18/checkpoints/experiments/C_joint/detector.pt"),
}
PROMPT_SETS = {
    "tuned": (PROMPTS, TEMPLATES),
    "generic_1": ({m: [m] for m in MATS}, ["a photo of an object made of {}."]),
    "generic_5": ({m: [m] for m in MATS}, TEMPLATES),
    "generic_syn": ({"metal": ["metal", "metallic object"], "wood": ["wood", "wooden object"], "plastic": ["plastic", "plastic object"],
                     "glass": ["glass", "glass object"], "paper": ["paper", "paper or cardboard object"], "mineral": ["stone", "mineral or stone object"]},
                    TEMPLATES),
}
MODELS = {"clipL": ("ViT-L-14", "dfn2b"), "siglip": ("ViT-B-16-SigLIP", "webli")}


def softmax(z):
    z = z - z.max(1, keepdims=True)
    p = np.exp(z)
    return p / p.sum(1, keepdims=True)


def l2n(X):
    return X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-8)


def M_frame(P, mat, idx):
    pred = P[idx].argmax(1)
    return float(np.mean([(pred[mat[idx] == c] == c).mean() for c in range(6) if (mat[idx] == c).any()]))


def M_obj(P, mat, oid, idx):
    hits = defaultdict(list)
    for o in sorted(set(oid[idx])):
        ii = idx[oid[idx] == o]
        hits[mat[ii[0]]].append(int(np.log(P[ii] + 1e-9).mean(0).argmax() == mat[ii[0]]))
    return float(np.mean([np.mean(v) for v in hits.values()])), {MATS[c]: f"{sum(v)}/{len(v)}" for c, v in sorted(hits.items())}


def std_fit_predict(X, y_all, w, tr, Xte):
    mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-6
    Z = np.concatenate([(X - mu) / sd, (Xte - mu) / sd])
    te = np.arange(len(X), len(X) + len(Xte))
    yy = np.concatenate([y_all, np.zeros(len(Xte), int)])
    return fit_predict(Z, yy, w, tr, te)


@torch.no_grad()
def gpu_stage():
    import open_clip
    from ultralytics import YOLO

    from attr_features import polygon_mask
    out = {}
    models = {}
    for tag, (name, pre_name) in MODELS.items():
        model, _, pre = open_clip.create_model_and_transforms(name, pretrained=pre_name, device=DEV)
        model.eval()
        tok = open_clip.get_tokenizer(name)
        size = model.visual.image_size[0] if isinstance(model.visual.image_size, (tuple, list)) else model.visual.image_size
        models[tag] = (model, pre, size)
        for sname, (prompts, templates) in PROMPT_SETS.items():
            T = []
            for m in MATS:
                txt = [t.format(p) for p in prompts[m] for t in templates]
                e = Fn.normalize(model.encode_text(tok(txt).to(DEV)).float(), dim=-1).mean(0)
                T.append(Fn.normalize(e, dim=0))
            out[f"text_{tag}_{sname}"] = torch.stack(T).cpu().numpy()

    def embed(tag, imgs):
        model, pre, size = models[tag]
        mean = torch.tensor(pre.transforms[-1].mean, device=DEV).view(1, 3, 1, 1)
        std = torch.tensor(pre.transforms[-1].std, device=DEV).view(1, 3, 1, 1)
        res = []
        for i in range(0, len(imgs), 48):
            x = torch.from_numpy(imgs[i:i + 48]).to(DEV).permute(0, 3, 1, 2).float() / 255
            x = Fn.interpolate(x, size=(size, size), mode="bilinear", align_corners=False)
            x = (x - mean) / std
            f = model.encode_image(x).float() + model.encode_image(torch.flip(x, dims=[3])).float()
            res.append(Fn.normalize(f, dim=-1).cpu().numpy())
        return np.concatenate(res)

    lab = json.load(open(D / "labels/real_train.json"))
    ims = {im["id"]: im for im in lab["images"]}
    by_frame = defaultdict(list)
    for a in lab["annotations"]:
        by_frame[Path(ims[a["image_id"]]["file_name"]).parent.name].append(a)
    rows, views = [], {k: [] for k in VIEWS}
    for fold, (split_path, det_path) in FOLDS.items():
        split = json.load(open(split_path))
        det = YOLO(str(det_path))
        for fid in split["validation_real_frames"]:
            anns = by_frame.get(fid, [])
            if not anns:
                continue
            fd = D / "real/train" / fid
            rgb = cv2.cvtColor(cv2.imread(str(fd / "color.png")), cv2.COLOR_BGR2RGB)
            r = det.predict(str(fd / "color.png"), imgsz=640, conf=0.05, retina_masks=True, verbose=False, device=0)[0]
            boxes = r.boxes.xyxy.cpu().numpy() if r.boxes is not None else np.zeros((0, 4))
            for a in anns:
                x, y, w, h = a["bbox"]
                g = np.array([x, y, x + w, y + h])
                best, bi = 0.0, -1
                for j, b in enumerate(boxes):
                    iw = max(0, min(g[2], b[2]) - max(g[0], b[0]))
                    ih = max(0, min(g[3], b[3]) - max(g[1], b[1]))
                    iou = iw * ih / ((g[2] - g[0]) * (g[3] - g[1]) + (b[2] - b[0]) * (b[3] - b[1]) - iw * ih + 1e-9)
                    if iou > best:
                        best, bi = iou, j
                gt_mask = polygon_mask(a["segmentation"], rgb.shape[0], rgb.shape[1])
                if bi >= 0 and best >= 0.5 and r.masks is not None:
                    pm = r.masks.data[bi].cpu().numpy() > 0.5
                    if pm.shape != gt_mask.shape:
                        pm = cv2.resize(pm.astype(np.uint8), (gt_mask.shape[1], gt_mask.shape[0]), interpolation=cv2.INTER_NEAREST) > 0
                    b = boxes[bi]
                    pbox = [float(b[0]), float(b[1]), float(b[2] - b[0]), float(b[3] - b[1])]
                    inter = (pm & gt_mask).sum()
                    mask_iou = inter / max((pm | gt_mask).sum(), 1)
                else:
                    pm, pbox, mask_iou = None, None, 0.0
                rows.append({"fold": fold, "frame": fid, "bbox": a["bbox"], "oid": a["attributes"]["object_id"],
                             "mat": a["attributes"]["material"], "box_iou": float(best), "mask_iou": float(mask_iou), "matched": pm is not None})
                for k, (sc, msk) in VIEWS.items():
                    if pm is None:
                        views[k].append(np.zeros((224, 224, 3), np.uint8))
                    else:
                        views[k].append(crop(rgb, pm, pbox, sc, msk)[0])
        print("fold", fold, "rows", len(rows), flush=True)
    for tag in MODELS:
        for k in VIEWS:
            out[f"pred_{tag}_{k}"] = embed(tag, np.stack(views[k]))
    np.savez_compressed(C / "overfit_gpu.npz", **out)
    json.dump(rows, open(C / "overfit_pred_rows.json", "w"))


def zs_from(F, T, prefix_keys, img_prefix="", temp=0.01):
    logits = []
    for tag in MODELS:
        for k in VIEWS:
            logits.append(F[f"{img_prefix}{tag}_{k}"] @ T[tag].T)
    return softmax(np.mean(logits, 0) / temp)


def inner_weight(X, oid, src, mat, train_objs, Z, grid=(0.0, 0.25, 0.4, 0.5, 0.6, 0.75, 1.0), seed=0):
    rs = np.random.default_rng(seed)
    real = src == "real_train"
    Pin = np.zeros((len(mat), 6))
    for g in np.array_split(rs.permutation(train_objs), 5):
        tr = np.where(np.isin(oid, train_objs) & ~np.isin(oid, g) & real)[0]
        te = np.where(np.isin(oid, g) & real)[0]
        mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-6
        Pin[te] = fit_predict((X - mu) / sd, mat, weights(tr, oid, src, mat, 0.0), tr, te)
    ii = np.where(np.isin(oid, train_objs) & real)[0]
    return max(grid, key=lambda w: M_frame(softmax((1 - w) * np.log(Pin + 1e-9) + w * np.log(Z + 1e-9)), mat, ii))


def main():
    if not (C / "overfit_gpu.npz").exists():
        gpu_stage()
    F = dict(np.load(C / "features.npz"))
    F.update(dict(np.load(C / "features2.npz")))
    G = dict(np.load(C / "overfit_gpu.npz"))
    meta = json.load(open(C / "meta.json"))
    oid = np.array([m["oid"] for m in meta])
    src = np.array([m["src"] for m in meta])
    mat = np.array([MATS.index(m["mat"]) for m in meta])
    light = np.array([m.get("lighting") for m in meta])
    real = src == "real_train"
    idx = np.where(real)[0]
    objs = sorted(set(oid[real]))
    X = (l2n(F["clipL_c14"]) * math.sqrt(768 / 64)).astype(np.float32)
    report = {}

    def show(name, P, key=None):
        mo, per = M_obj(P, mat, oid, idx)
        mf = M_frame(P, mat, idx)
        print(f"  {name:52s} M={mf:.3f}  M_obj={mo:.3f} {per}", flush=True)
        report[key or name] = {"M": mf, "M_obj": mo, "per_object": per}

    print("== Линейная голова LOOO (CLIP-L, объект с контекстом) ==")
    P_lin = np.zeros((len(mat), 6))
    for o in objs:
        tr = np.where((oid != o) & real)[0]
        te = np.where((oid == o) & real)[0]
        mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-6
        P_lin[te] = fit_predict((X - mu) / sd, mat, weights(tr, oid, src, mat, 0.0), tr, te)
    show("linear OOF", P_lin)
    mu, sd = X[idx].mean(0), X[idx].std(0) + 1e-6
    P_tr = np.zeros((len(mat), 6))
    P_tr[idx] = fit_predict((X - mu) / sd, mat, weights(idx, oid, src, mat, 0.0), idx, idx)
    show("linear TRAIN (обучение и оценка на тех же объектах)", P_tr)

    print("\n== 1. Утечка через промпты ==")
    Zs = {}
    for sname in PROMPT_SETS:
        T = {tag: G[f"text_{tag}_{sname}"] for tag in MODELS}
        Z = zs_from(F, T, None)
        Zs[sname] = Z
        show(f"zero-shot, промпты={sname}", Z)
    for sname in PROMPT_SETS:
        Z = Zs[sname]
        Pfin = np.zeros((len(mat), 6))
        ws = []
        for o in objs:
            w = inner_weight(X, oid, src, mat, [q for q in objs if q != o], Z, seed=len(ws))
            ws.append(w)
            te = np.where((oid == o) & real)[0]
            Pfin[te] = softmax((1 - w) * np.log(P_lin[te] + 1e-9) + w * np.log(Z[te] + 1e-9))
        show(f"linear ⊕ zero-shot({sname}), w≈{np.median(ws):.2f}", Pfin, key=f"fused_{sname}")
        np.save(C / f"fused_{sname}.npy", Pfin)

    print("\n== 2. Перестановочный тест (материалы перемешаны между объектами) ==")
    obj_mat = {o: mat[np.where(oid == o)[0][0]] for o in objs}
    perm_scores = []
    for s in range(5):
        rs = np.random.default_rng(100 + s)
        shuffled = dict(zip(objs, rs.permutation([obj_mat[o] for o in objs])))
        mp = np.array([shuffled.get(o, m) for o, m in zip(oid, mat)])
        Pp = np.zeros((len(mat), 6))
        for o in objs:
            tr = np.where((oid != o) & real)[0]
            te = np.where((oid == o) & real)[0]
            m_, s_ = X[tr].mean(0), X[tr].std(0) + 1e-6
            Pp[te] = fit_predict((X - m_) / s_, mp, weights(tr, oid, src, mp, 0.0), tr, te)
        perm_scores.append(M_frame(Pp, mp, idx))
        print(f"  перестановка {s}: M={perm_scores[-1]:.3f}", flush=True)
    report["permutation_M"] = perm_scores
    print(f"  среднее {np.mean(perm_scores):.3f} (случайный уровень ≈ {1/6:.3f})")

    print("\n== 4. Чувствительность ==")
    T = {tag: G[f"text_{tag}_generic_5"] for tag in MODELS}
    for temp in (0.005, 0.01, 0.02, 0.05):
        Z = zs_from(F, T, None, temp=temp)
        for w in (0.3, 0.5, 0.7):
            P = softmax((1 - w) * np.log(P_lin + 1e-9) + w * np.log(Z + 1e-9))
            print(f"  generic_5 T={temp:<6} w={w}: M={M_frame(P, mat, idx):.3f}  M_obj={M_obj(P, mat, oid, idx)[0]:.3f}")

    print("\n== 5. Bootstrap по объектам (95% интервал) ==")
    for key in ("fused_tuned", "fused_generic_5"):
        P = np.load(C / f"{key}.npy")
        bs = []
        rs = np.random.default_rng(0)
        for _ in range(1000):
            pick = rs.choice(objs, len(objs), replace=True)
            ii = np.concatenate([idx[oid[idx] == o] for o in pick])
            bs.append(M_frame(P, mat, ii))
        lo, hi = np.percentile(bs, [2.5, 97.5])
        print(f"  {key:18s} M={M_frame(P, mat, idx):.3f}  95% CI [{lo:.3f}, {hi:.3f}]")
        report[f"{key}_ci"] = [float(lo), float(hi)]

    print("\n== 6. По освещению ==")
    P = np.load(C / "fused_generic_5.npy")
    for lg in sorted(set(light[idx])):
        ii = idx[light[idx] == lg]
        print(f"  {lg:7s} кадров {len(ii):3d}  acc={(P[ii].argmax(1) == mat[ii]).mean():.3f}  M={M_frame(P, mat, ii):.3f}")

    print("\n== 7. Разрыв масок: YOLO-маски vs эталонные, фолды 1–3 ==")
    rows = json.load(open(C / "overfit_pred_rows.json"))
    key_of = {(m["frame"], tuple(np.round(m["bbox"], 1))): i for i, m in enumerate(meta) if m["src"] == "real_train"}
    Fp = {f"{tag}_{k}": G[f"pred_{tag}_{k}"] for tag in MODELS for k in VIEWS}
    Xp = (l2n(Fp["clipL_c14"]) * math.sqrt(768 / 64)).astype(np.float32)
    for sname in ("tuned", "generic_5"):
        T = {tag: G[f"text_{tag}_{sname}"] for tag in MODELS}
        Zp = zs_from(Fp, T, None)
        Zg = Zs[sname]
        res = defaultdict(lambda: {"gt": [], "pred": [], "mat": [], "oid": [], "miou": []})
        for fold in FOLDS:
            split = json.load(open(FOLDS[fold][0]))
            train_objs = [o for o in objs if o in set(split["train_object_ids"])]
            tr = np.where(np.isin(oid, train_objs) & real)[0]
            w = inner_weight(X, oid, src, mat, train_objs, Zg, seed=fold)
            rr = [i for i, r in enumerate(rows) if r["fold"] == fold]
            gi = np.array([key_of[(rows[i]["frame"], tuple(np.round(rows[i]["bbox"], 1)))] for i in rr])
            Pl_gt = std_fit_predict(X, mat, weights(tr, oid, src, mat, 0.0), tr, X[gi])
            Pl_pr = std_fit_predict(X, mat, weights(tr, oid, src, mat, 0.0), tr, Xp[rr])
            Pg = softmax((1 - w) * np.log(Pl_gt + 1e-9) + w * np.log(Zg[gi] + 1e-9))
            Pp = softmax((1 - w) * np.log(Pl_pr + 1e-9) + w * np.log(Zp[rr] + 1e-9))
            matched = np.array([rows[i]["matched"] for i in rr])
            res[fold]["gt"] = Pg
            res[fold]["pred"] = np.where(matched[:, None], Pp, np.full_like(Pp, 1 / 6))
            res[fold]["mat"] = mat[gi]
            res[fold]["oid"] = oid[gi]
            res[fold]["miou"] = [rows[i]["mask_iou"] for i in rr if rows[i]["matched"]]
            res[fold]["matched"] = matched
        print(f"  промпты={sname}")
        for fold in FOLDS:
            r = res[fold]
            ii = np.arange(len(r["mat"]))
            mg = M_frame(r["gt"], r["mat"], ii)
            mp = M_frame(r["pred"], r["mat"], ii)
            per = {MATS[c]: f"{(r['pred'][r['mat'] == c].argmax(1) == c).sum()}/{(r['mat'] == c).sum()}" for c in range(6) if (r["mat"] == c).any()}
            print(f"    fold {fold}: объектов-кадров {len(ii):3d}, сопоставлено {r['matched'].mean():.0%}, mask IoU {np.median(r['miou']):.2f} | "
                  f"M эталон-маска={mg:.3f}  M YOLO-маска={mp:.3f}  {per}")
            report[f"maskgap_{sname}_fold{fold}"] = {"M_gt": mg, "M_pred": mp}
    json.dump(report, open(C / "overfit_report.json", "w"), indent=1, default=str)


if __name__ == "__main__":
    main()
