"""Полный конвейер валидации через официальный score.py на фолдах 1–3 по object_id.
Для каждого фолда всё, что обучается или подбирается, использует ТОЛЬКО train-объекты фолда:
  детектор   — сохранённый детектор фолда (YOLO11-seg), рамки, маски, форма, уверенность;
  материал   — новый классификатор: линейная голова на CLIP-L (объект целиком с контекстом), обученная на train-объектах,
               ⊕ zero-shot CLIP-L+SigLIP с общими описаниями; вес смешивания — внутренний 5-fold CV по train-объектам;
  объём      — геометрия по глубине внутри предсказанной маски (клетки 4 мм над плоскостью пола)
               × поправка по классу (форма, материал), выученная на train-объектах (лог-отношение, усреднение по объекту, shrink);
  take       — A: масса = объём × плотность класса (по train-объектам) ≤ 500 г;
               B: геометрический объём ≤ T, T подобран на train-объектах по сбалансированной точности.
Папка real/test не используется."""
import argparse
import csv
import json
import math
import os
import sys
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as Fn

from attr_features import geometry
from mat_cv import MATS, fit_predict, weights
from mat_features2 import TEMPLATES, crop
from take_model import TakeModel

C = Path(__file__).resolve().parent / "cache"
ROOT = Path(__file__).resolve().parent.parent
D = Path(os.environ.get("ASTRAMIS_DATA", ROOT / "datasets/raw"))
DEV = "cuda" if torch.cuda.is_available() else "cpu"
FORMS = ["bulk", "solid"]
VIEWS = {"m10": (1.1, True), "m14": (1.5, True), "c14": (1.5, False), "c20": (2.2, False)}
MODELS = {"clipL": ("ViT-L-14", "dfn2b"), "siglip": ("ViT-B-16-SigLIP", "webli")}
PROMPT_SETS = {
    "generic_5": {m: [m] for m in MATS},
    "rich": {"metal": ["metal", "metallic object", "a metal can", "an aluminium can", "a dark painted metal tray", "a steel plate", "a metal bar", "metal screws and bolts", "an aluminium profile"],
             "wood": ["wood", "wooden object", "a wooden plank", "a wooden stick", "wood chips", "a piece of timber", "a wooden block"],
             "plastic": ["plastic", "plastic object", "a plastic bottle", "a plastic bag", "a plastic container", "plastic packaging", "plastic film"],
             "glass": ["glass", "glass object", "a glass bottle", "a brown glass bottle", "a dark amber glass bottle with a label", "a clear glass bottle", "a glass jar"],
             "paper": ["paper", "paper or cardboard object", "a cardboard box", "a flat cardboard sheet", "a paper bag", "a paper cup", "crumpled paper"],
             "mineral": ["stone", "mineral or stone object", "a rock", "a stone slab", "a brick", "gravel", "a concrete block"]},
    "generic_syn": {"metal": ["metal", "metallic object"], "wood": ["wood", "wooden object"], "plastic": ["plastic", "plastic object"],
                    "glass": ["glass", "glass object"], "paper": ["paper", "paper or cardboard object"], "mineral": ["stone", "mineral or stone object"]},
}
FOLDS = {
    1: (ROOT / "configs/object_id_split.json", ROOT / "checkpoints/experiments/D_domain_randomized/detector.pt",
        "детектор выбран как best.pt по этому же фолду — D оптимистичен"),
    2: (ROOT / "checkpoints/score83_cv_fold2_noval18/configs/object_id_split.json",
        ROOT / "checkpoints/score83_cv_fold2_noval18/checkpoints/experiments/C_joint/detector.pt", "last.pt, без валидации"),
    3: (ROOT / "checkpoints/score83_cv_fold3_noval18/configs/object_id_split.json",
        ROOT / "checkpoints/score83_cv_fold3_noval18/checkpoints/experiments/C_joint/detector.pt", "last.pt, без валидации"),
}
sys.path.insert(0, str(ROOT / "references"))
sys.path.insert(0, str(ROOT))
import score as S  # noqa: E402
from evaluation.volume_parts_cv import fit_calibrator  # noqa: E402
from volume.estimator import VolumeEstimator  # noqa: E402
from inference.volume_backend import load_densities  # noqa: E402

GT_ROWS = ROOT / "reports/volume_parts_gt_rows.csv"


def volume_estimator_for(train_objs, cam):
    """Калибратор нового объёма (по частям) обучается только на train-объектах фолда, по GT-маскам."""
    rows = pd.read_csv(GT_ROWS)
    rows = rows[(rows["split"] == "real") & rows["object_id"].isin(list(train_objs))]
    assert set(rows["object_id"]) == set(train_objs), "калибратор объёма: train-объекты не совпали"
    cal = fit_calibrator(rows)
    if os.environ.get("VOL_IRLS", "0") == "1":
        from vol_cal2 import VolCal2
        vc = VolCal2(use_zs=False, irls=True).fit(rows, None)
        cal = dict(cal)
        cal["_irls"] = vc
    est = VolumeEstimator(cal, load_densities(D / "densities.csv"), cam)
    if "_irls" in cal:
        _vc = cal["_irls"]

        def _estimate(depth_mm, mask, plane, form, material=None, material_probs=None, geometry=None, _est=est):
            geo = geometry if geometry is not None else _est.geometry(depth_mm, mask, plane, form)
            r = pd.DataFrame([{**geo, "form": form, "material": material}])
            vol = float(_vc.predict(r, None, [material])[0])
            from inference.volume_backend import density_for
            dens = density_for(_est.densities, form, material)
            from volume.estimator import VolumeResult
            return VolumeResult(vol, vol * dens, dens, geo)
        est.estimate = _estimate
    # порог take по калиброванному объёму на тех же train-объектах (медиана по объекту)
    per_obj = rows.groupby("object_id")
    xs, ys = [], []
    for o, g in per_obj:
        r0 = g.iloc[0]
        v = np.median([est.estimate(None, None, None, r["form"], r["material"], geometry=r.to_dict()).volume_ml for _, r in g.iterrows()])
        xs.append(v)
        ys.append(int(r0["mass_g"] <= 500))
    xs, ys = np.array(xs), np.array(ys)
    best_t, best_s = 500.0, -1
    for t in np.unique(xs):
        for tt in (t - 1e-6, t + 1e-6):
            p = (xs <= tt).astype(int)
            sc = np.mean([(p[ys == v] == v).mean() for v in (0, 1) if (ys == v).any()])
            if sc > best_s:
                best_t, best_s = tt, sc
    return est, float(best_t)


def softmax(z):
    z = z - z.max(1, keepdims=True)
    p = np.exp(z)
    return p / p.sum(1, keepdims=True)


def l2n(X):
    return X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-8)


def M_frame(P, mat, idx):
    pred = P[idx].argmax(1)
    return float(np.mean([(pred[mat[idx] == c] == c).mean() for c in range(6) if (mat[idx] == c).any()]))


class Encoders:
    @torch.no_grad()
    def __init__(self):
        import open_clip
        self.m = {}
        self.text = {}
        for tag, (name, pre_name) in MODELS.items():
            model, _, pre = open_clip.create_model_and_transforms(name, pretrained=pre_name, device=DEV)
            model.eval()
            size = model.visual.image_size[0] if isinstance(model.visual.image_size, (tuple, list)) else model.visual.image_size
            mean = torch.tensor(pre.transforms[-1].mean, device=DEV).view(1, 3, 1, 1)
            std = torch.tensor(pre.transforms[-1].std, device=DEV).view(1, 3, 1, 1)
            self.m[tag] = (model, size, mean, std)
            tok = open_clip.get_tokenizer(name)
            for sname, prompts in PROMPT_SETS.items():
                T = []
                for mt in MATS:
                    txt = [t.format(p) for p in prompts[mt] for t in TEMPLATES]
                    e = Fn.normalize(model.encode_text(tok(txt).to(DEV)).float(), dim=-1).mean(0)
                    T.append(Fn.normalize(e, dim=0))
                self.text[(tag, sname)] = torch.stack(T).cpu().numpy()

    @torch.no_grad()
    def embed(self, tag, imgs):
        model, size, mean, std = self.m[tag]
        out = []
        for i in range(0, len(imgs), 48):
            x = torch.from_numpy(imgs[i:i + 48]).to(DEV).permute(0, 3, 1, 2).float() / 255
            x = (Fn.interpolate(x, size=(size, size), mode="bilinear", align_corners=False) - mean) / std
            f = model.encode_image(x).float() + model.encode_image(torch.flip(x, dims=[3])).float()
            out.append(Fn.normalize(f, dim=-1).cpu().numpy())
        return np.concatenate(out) if out else np.zeros((0, 768), np.float32)


def zero_shot(embs, text, sname, temp=0.01):
    logits = [embs[(tag, v)] @ text[(tag, sname)].T for tag in MODELS for v in VIEWS]
    return softmax(np.mean(logits, 0) / temp)


def train_cache():
    F = dict(np.load(C / "features.npz"))
    F.update(dict(np.load(C / "features2.npz")))
    meta = json.load(open(C / "meta.json"))
    return F, meta


def material_model(F, meta, train_objs, text, sname):
    oid = np.array([m["oid"] for m in meta])
    src = np.array([m["src"] for m in meta])
    mat = np.array([MATS.index(m["mat"]) for m in meta])
    real = src == "real_train"
    X = (l2n(F["clipL_c14"]) * math.sqrt(768 / 64)).astype(np.float32)
    Zg = softmax(np.mean([F[f"{tag}_{v}"] @ text[(tag, sname)].T for tag in MODELS for v in VIEWS], 0) / 0.01)
    tr = np.where(np.isin(oid, list(train_objs)) & real)[0]
    assert len(set(oid[tr])) == len(train_objs), "материал: не все train-объекты попали в обучение"
    assert not set(oid[tr]) - set(train_objs)
    rs = np.random.default_rng(0)
    Pin = np.zeros((len(mat), 6))
    for g in np.array_split(rs.permutation(sorted(train_objs)), 5):
        a = np.where(np.isin(oid, train_objs) & ~np.isin(oid, g) & real)[0]
        b = np.where(np.isin(oid, g) & real)[0]
        mu, sd = X[a].mean(0), X[a].std(0) + 1e-6
        Pin[b] = fit_predict((X - mu) / sd, mat, weights(a, oid, src, mat, 0.0), a, b)
    grid = [0.0, 0.25, 0.4, 0.5, 0.6, 0.75, 1.0]
    w = max(grid, key=lambda w: M_frame(softmax((1 - w) * np.log(Pin + 1e-9) + w * np.log(Zg + 1e-9)), mat, tr))
    mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-6

    def predict(x_clip_c14, zs):
        Xn = (l2n(x_clip_c14) * math.sqrt(768 / 64)).astype(np.float32)
        Z = np.concatenate([(X - mu) / sd, (Xn - mu) / sd])
        yy = np.concatenate([mat, np.zeros(len(Xn), int)])
        P = fit_predict(Z, yy, weights(tr, oid, src, mat, 0.0), tr, np.arange(len(X), len(X) + len(Xn)))
        return softmax((1 - w) * np.log(P + 1e-9) + w * np.log(zs + 1e-9))

    return predict, w


def volume_density_model(F, meta, train_objs):
    gk = list(F["geo_keys"])
    gv = F["geo"][:, gk.index("vol")]
    per_obj = {}
    for i, m in enumerate(meta):
        if m["src"] != "real_train" or m["oid"] not in train_objs:
            continue
        d = per_obj.setdefault(m["oid"], {"cls": (m["form"], m["mat"]), "vol": m["vol"], "mass": m["mass"], "ratios": [], "geo": []})
        if gv[i] > 1:
            d["ratios"].append(math.log(m["vol"] / gv[i]))
            d["geo"].append(gv[i])
    obj_ratio = {o: float(np.mean(d["ratios"])) for o, d in per_obj.items() if d["ratios"]}
    glob = float(np.mean(list(obj_ratio.values())))
    by_cls, by_mat = defaultdict(list), defaultdict(list)
    dens_cls, dens_mat = defaultdict(list), defaultdict(list)
    for o, d in per_obj.items():
        if o in obj_ratio:
            by_cls[d["cls"]].append(obj_ratio[o])
            by_mat[d["cls"][1]].append(obj_ratio[o])
        dens_cls[d["cls"]].append(d["mass"] / d["vol"])
        dens_mat[d["cls"][1]].append(d["mass"] / d["vol"])
    shrink = 2.0

    def scale(form, mat):
        vals = by_cls.get((form, mat)) or by_mat.get(mat) or []
        if not vals:
            return math.exp(glob)
        wgt = len(vals) / (len(vals) + shrink)
        return math.exp(wgt * float(np.mean(vals)) + (1 - wgt) * glob)

    all_d = [x for v in dens_cls.values() for x in v]

    def density(form, mat):
        v = dens_cls.get((form, mat)) or dens_mat.get(mat) or all_d
        return float(np.median(v))

    geo_take = [(float(np.median(d["geo"])), int(d["mass"] <= 500)) for d in per_obj.values() if d["geo"]]
    xs = np.array([g for g, _ in geo_take])
    ys = np.array([t for _, t in geo_take])
    best_t, best_s = 500.0, -1
    for t in np.unique(xs):
        for tt in (t - 1e-6, t + 1e-6):
            p = (xs <= tt).astype(int)
            s = np.mean([(p[ys == v] == v).mean() for v in (0, 1) if (ys == v).any()])
            if s > best_s:
                best_t, best_s = tt, s
    return scale, density, best_t


def build_labels(frames):
    lab = json.load(open(D / "labels/real_train.json"))
    keep = [im for im in lab["images"] if Path(im["file_name"]).parent.name in frames]
    ids = {im["id"] for im in keep}
    data = {"categories": lab["categories"], "images": keep, "annotations": [a for a in lab["annotations"] if a["image_id"] in ids]}
    p = C / f"val_labels_{abs(hash(tuple(sorted(frames)))) % 10**8}.json"
    p.write_text(json.dumps(data))
    return p


def link_series(dets, embs, thr, gap):
    """Группы «один и тот же предмет» по соседним кадрам: номер кадра близко (≤gap) и сходство вырезок ≥ thr.
    Возвращает group_id для каждой детекции."""
    E = np.concatenate([embs[("clipL", "c14")], embs[("siglip", "c14")]], 1)
    E = E / (np.linalg.norm(E, axis=1, keepdims=True) + 1e-8)
    fnum = np.array([int(d["frame"].split("_")[1]) for d in dets])
    order = np.argsort(fnum, kind="stable")
    gid = -np.ones(len(dets), int)
    groups = []  # (mean_emb, n, last_frame)
    for i in order:
        best, bj = -1.0, -1
        for j, (m, n, lf) in enumerate(groups):
            if 0 < fnum[i] - lf <= gap or fnum[i] == lf:
                sim = float(E[i] @ m / (np.linalg.norm(m) + 1e-8))
                if sim > best:
                    best, bj = sim, j
        if bj >= 0 and best >= thr:
            m, n, lf = groups[bj]
            groups[bj] = ((m * n + E[i]) / (n + 1), n + 1, max(lf, fnum[i]))
            gid[i] = bj
        else:
            groups.append((E[i].copy(), 1, fnum[i]))
            gid[i] = len(groups) - 1
    return gid


def touches_border_px(b, margin=2, w=1280, h=720):
    return b[0] <= margin or b[1] <= margin or b[2] >= w - margin or b[3] >= h - margin


def _iou(a, b):
    w = min(a[2], b[2]) - max(a[0], b[0])
    h = min(a[3], b[3]) - max(a[1], b[1])
    if w <= 0 or h <= 0:
        return 0.0
    i = w * h
    return i / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - i)


def serialize(objs):
    if not objs:
        return "none"
    return ";".join(f"{o['box'][0]:.1f} {o['box'][1]:.1f} {o['box'][2]:.1f} {o['box'][3]:.1f} {o['score']:.5f} {o['form']} {o['material']} "
                    f"{o['volume']:.2f} {o['mass']:.2f} {int(o['take'])}" for o in objs[:30])


def detect_fold(det, frames, est_tmp, enc, conf=0.08, tta=False):
    dets = []
    views = {k: [] for k in VIEWS}
    for fid in frames:
        fd = D / "real/train" / fid
        rgb = cv2.cvtColor(cv2.imread(str(fd / "color.png")), cv2.COLOR_BGR2RGB)
        depth = cv2.imread(str(fd / "depth.png"), cv2.IMREAD_UNCHANGED)
        plane = json.load(open(fd / "meta.json"))["floor_plane"]
        r = det.predict(str(fd / "color.png"), imgsz=640, conf=conf, max_det=30, retina_masks=True, verbose=False, device=0, augment=tta)[0]
        if r.boxes is None:
            continue
        for j in range(len(r.boxes)):
            b = r.boxes.xyxy[j].cpu().numpy().astype(float)
            if r.masks is not None:
                m = r.masks.data[j].cpu().numpy() > 0.5
                if m.shape != depth.shape:
                    m = cv2.resize(m.astype(np.uint8), (depth.shape[1], depth.shape[0]), interpolation=cv2.INTER_NEAREST) > 0
            else:
                m = np.zeros(depth.shape, bool)
            if m.sum() < 10:
                m = np.zeros(depth.shape, bool)
                m[int(b[1]):int(b[3]), int(b[0]):int(b[2])] = True
            g = geometry(depth, m, plane)
            form = FORMS[int(r.boxes.cls[j])]
            new_geo = est_tmp.geometry(depth, m, np.asarray(plane, float), form)
            other = "solid" if form == "bulk" else "bulk"
            new_geo_other = est_tmp.geometry(depth, m, np.asarray(plane, float), other)
            bbox = [b[0], b[1], b[2] - b[0], b[3] - b[1]]
            dets.append({"frame": fid, "box": b.tolist(), "score": float(r.boxes.conf[j]), "form": form, "geo_vol": g["vol"], "new_geo": new_geo,
                         "new_geo_by_form": {form: new_geo, other: new_geo_other}, "geo": g})
            for k, (sc, msk) in VIEWS.items():
                views[k].append(crop(rgb, m, bbox, sc, msk)[0])
    embs = {(tag, k): enc.embed(tag, np.stack(views[k])) if views[k] else np.zeros((0, 768)) for tag in MODELS for k in VIEWS}
    return dets, embs


def run_fold(fold, enc, F, meta, args):
    from ultralytics import YOLO
    split_path, det_path, note = FOLDS[fold]
    if args.detector:
        det_path, note = Path(args.detector), f"детектор {Path(args.detector).parent.parent.name}"
    split = json.load(open(split_path))
    train_objs = sorted(split["train_object_ids"])
    val_objs = set(split["validation_object_ids"])
    assert not (set(train_objs) & val_objs)
    frames = list(split["validation_real_frames"])
    det = YOLO(str(det_path))
    cam = json.load(open(D / "camera.json"))
    est_tmp, new_thr_tmp = volume_estimator_for(train_objs, cam)
    cache_p = C / f"fold{fold}_dets{args.tag}.pkl"
    import pickle
    if cache_p.exists() and not getattr(args, "no_cache", False):
        dets, embs = pickle.load(open(cache_p, "rb"))
        print(f"  fold {fold}: детекции из кэша ({len(dets)})", flush=True)
    else:
        dets, embs = detect_fold(det, frames, est_tmp, enc, args.conf, args.tta)
        pickle.dump((dets, embs), open(cache_p, "wb"))
    if False:
        dets = []
    scale, density, geo_thr = volume_density_model(F, meta, train_objs)
    est_new, new_thr = est_tmp, new_thr_tmp
    labels_path = build_labels(set(frames))
    sol = S.labels_to_solution(labels_path)
    old_rows = []
    results = {}
    for sname in PROMPT_SETS:
        predict, w = material_model(F, meta, train_objs, enc.text, sname)
        P = predict(embs[("clipL", "c14")], zero_shot(embs, enc.text, sname)) if dets else np.zeros((0, 6))
        VARIANTS = [("new_parts", "B_new_thr", "det"), ("new_parts", "B_new_thr", "cls")]
        sys.path.insert(0, str(ROOT / "agents/form"))
        from form_cv import load_bank
        from form_model import FeatureBank, FormModel, GEO_KEYS
        fbank, fy, foid, fsrc, _ = load_bank()
        fm = FormModel(fbank, fy, foid, fsrc, grid={"spec": ["clipL_c14+zs"], "sw": [0.5], "wd": [1e-2]}).fit(train_objs)
        femb = {f"{t}_{v}": embs[(t, v)] for t in MODELS for v in VIEWS}
        femb["geo"] = np.array([[d["geo"][k] for k in GEO_KEYS] for d in dets], np.float32).reshape(len(dets), len(GEO_KEYS))
        p_bulk = fm.predict_bank(FeatureBank(femb, fbank.text))[:, 0] if dets else np.zeros(0)
        gid = link_series(dets, embs, args.agg, args.agg_gap) if (args.agg > 0 and dets) else None
        if gid is not None:
            Pg = P.copy()
            for g in set(gid):
                ii = np.where(gid == g)[0]
                if args.agg_mat == "pool":
                    lp = np.log(P[ii] + 1e-9).mean(0)
                    Pg[ii] = np.exp(lp) / np.exp(lp).sum()
                elif args.agg_mat == "conf":
                    Pg[ii] = P[ii[int(np.argmax(P[ii].max(1)))]]
            P_use = Pg
            n_multi = sum(1 for g in set(gid) if (gid == g).sum() > 1)
            print(f"  fold {fold} | серии: {len(set(gid))} групп из {len(dets)} детекций, групп с ≥2 кадрами {n_multi}", flush=True)
        else:
            P_use = P
        for vol_mode, policy, form_src in VARIANTS:
            per_frame = defaultdict(list)
            vols_tmp = {}
            if gid is not None:
                for di, (d, p) in enumerate(zip(dets, P_use)):
                    m_ = MATS[int(p.argmax())]
                    if form_src == "cls":
                        f_ = "bulk" if p_bulk[di] >= 0.5 else "solid"
                        geo_ = d.get("new_geo_by_form", {}).get(f_, d["new_geo"])
                    else:
                        f_, geo_ = d["form"], d["new_geo"]
                    vols_tmp[di] = est_new.estimate(None, None, None, f_, m_, geometry=geo_).volume_ml
                vol_group = {}
                for g in set(gid):
                    ii = [i for i in np.where(gid == g)[0]]
                    vol_group[g] = float(np.exp(np.median([math.log(max(vols_tmp[i], 1.0)) for i in ii])))
            for di, (d, p) in enumerate(zip(dets, P_use)):
                matl = MATS[int(p.argmax())]
                if form_src == "cls":
                    pb = float(p_bulk[di])
                    fcls = "bulk" if pb >= 0.5 else "solid"
                    d = dict(d, form=fcls, score=d["score"] * max(pb, 1 - pb), conf=d["score"], new_geo=d.get("new_geo_by_form", {}).get(fcls, d["new_geo"]))
                    if touches_border_px(d["box"]):
                        continue
                raw = d["geo_vol"] if d["geo_vol"] > 1 else float(np.exp(np.mean([math.log(x["vol"]) for x in meta if x["src"] == "real_train" and x["oid"] in train_objs])))
                if vol_mode == "old_geo":
                    vol = raw
                    mass = vol * density(d["form"], matl)
                elif vol_mode == "old_geo_class":
                    vol = raw * scale(d["form"], matl)
                    mass = vol * density(d["form"], matl)
                else:
                    probs = {MATS[i]: float(p[i]) for i in range(6)} if vol_mode == "new_parts_probs" else None
                    rr = est_new.estimate(None, None, None, d["form"], matl, material_probs=probs, geometry=d["new_geo"])
                    vol, mass = rr.volume_ml, rr.mass_g
                    if gid is not None:
                        vol = vol_group[gid[di]]
                        mass = vol * rr.density_g_per_ml
                if policy == "A_mass":
                    take = int(mass <= 500)
                elif policy == "B_geo_thr":
                    take = int(raw <= geo_thr)
                else:
                    take = int(vol <= new_thr)
                per_frame[d["frame"]].append({"box": d["box"], "score": d["score"], "form": d["form"], "material": matl,
                                              "volume": vol, "mass": mass, "take": take})
            if form_src == "cls":
                for f in list(per_frame):
                    keep = []
                    for c in sorted(per_frame[f], key=lambda o: -o["score"]):
                        if all(_iou(c["box"], k["box"]) < 0.6 for k in keep):
                            keep.append(c)
                    per_frame[f] = keep
            sub = pd.DataFrame({"image_id": frames, "prediction": [serialize(sorted(per_frame.get(f, []), key=lambda o: -o["score"])) for f in frames]})
            out_csv = C / f"fold{fold}_{sname}_{vol_mode}_{policy}_{form_src}{args.tag}{('_agg_' + args.agg_mat) if gid is not None else ''}.csv"
            sub.to_csv(out_csv, index=False)
            total = S.score(sol, sub, "image_id")
            gt = {str(i): S._parse(c, False, str(i)) for i, c in zip(sol.image_id, sol.objects)}
            pr = {str(i): S._parse(c, True, str(i)) for i, c in zip(sub.image_id, sub.prediction)}
            comp = S.components(gt, pr)
            assert abs(comp["Score"] - total) < 1e-6
            results[(sname, vol_mode, policy, form_src)] = comp
            print(f"  fold {fold} | {sname:11s} w_zs={w:.2f} | объём={vol_mode:10s} | take={policy:9s} | форма={form_src} | "
                  f"Score {comp['Score']:6.2f}  D {comp['D']:.3f}  M {comp['M']:.3f}  V {comp['V']:.3f}  S {comp['S']:.3f}", flush=True)
    old = {}
    for name, path in (("old_noval18", ROOT / f"checkpoints/score83_cv_fold{fold}_noval18/logs/validation/C_joint/rgb_only/multimodal_submission.csv"),):
        if path.exists():
            s_old = pd.read_csv(path)
            gt = {str(i): S._parse(c, False, str(i)) for i, c in zip(sol.image_id, sol.objects)}
            sd = dict(zip(s_old.image_id.astype(str), s_old.prediction))
            pr = {i: S._parse(sd.get(i, "none"), True, i) for i in gt}
            old[name] = S.components(gt, pr)
    return results, {"note": note, "geo_thr_ml": geo_thr, "new_thr_ml": new_thr, "n_frames": len(frames), "n_dets": len(dets), "val_objects": sorted(val_objs), "old": old}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--folds", type=int, nargs="+", default=[1, 2, 3])
    ap.add_argument("--conf", type=float, default=0.08)
    ap.add_argument("--no-cache", action="store_true")
    ap.add_argument("--detector", default=None, help="путь к весам детектора вместо детектора фолда")
    ap.add_argument("--tag", default="", help="суффикс кэша/выходов")
    ap.add_argument("--tta", action="store_true", help="TTA детектора (flip)")
    ap.add_argument("--agg", type=float, default=0.0, help="агрегация по сериям кадров: порог сходства эмбеддингов (0 = выкл)")
    ap.add_argument("--agg-gap", type=int, default=3, help="макс. разрыв номеров кадров внутри серии")
    ap.add_argument("--agg-mat", default="pool", choices=["pool", "frame", "conf"], help="материал по серии: pool=среднее log-p, frame=как есть, conf=кадр с макс. уверенностью")
    a = ap.parse_args()
    enc = Encoders()
    F, meta = train_cache()
    allres, info = {}, {}
    for fold in a.folds:
        print(f"== fold {fold}: {FOLDS[fold][2]}", flush=True)
        r, i = run_fold(fold, enc, F, meta, a)
        allres[fold], info[fold] = r, i
        if i["old"]:
            for k, v in i["old"].items():
                print(f"  fold {fold} | для сравнения {k}: Score {v['Score']:6.2f}  D {v['D']:.3f}  M {v['M']:.3f}  V {v['V']:.3f}  S {v['S']:.3f}")
    print("\n== Среднее по фолдам ==")
    keys = list(next(iter(allres.values())).keys())
    summary = {}
    for k in keys:
        comps = [allres[f][k] for f in allres]
        mean = {c: float(np.mean([x[c] for x in comps])) for c in ("Score", "D", "M", "V", "S")}
        sd = float(np.std([x["Score"] for x in comps]))
        summary["|".join(k)] = {"mean": mean, "std_score": sd, "per_fold": {f: allres[f][k] for f in allres}}
        print(f"  {k[0]:11s} объём={k[1]:10s} take={k[2]:9s} форма={k[3]} | Score {mean['Score']:6.2f} ± {sd:4.2f} | D {mean['D']:.3f} M {mean['M']:.3f} V {mean['V']:.3f} S {mean['S']:.3f}")
    json.dump({"summary": summary, "info": info}, open(C / f"full_pipeline_report{a.tag}{('_agg_' + a.agg_mat) if a.agg > 0 else ''}.json", "w"), indent=1, default=str)


if __name__ == "__main__":
    main()
