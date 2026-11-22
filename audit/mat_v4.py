"""Материал v4 — только train (40 предметов, LOOO, вложенный CV).
 (1) физические признаки: шероховатость по глубине, темнота, насыщенность, лапласиан (текстура), дыры глубины,
     блики — как доп. вход к голове материала (стек: [голова CLIP ⊕ zero-shot] + признаки → логрегрессия, внутренний CV);
 (2) нормализация освещения вырезки перед CLIP: CLAHE по L-каналу и gray-world; эмбеддинги пересчитываются;
 (3) бутылки на train: кадры s11/s18/s29 (пластик) и s04/s36/s38 (стекло) — отдельная строка в отчёте;
 (4) аэрозоль: s05 (банка, металл) — отдельная строка.
Выход: таблица M/M_obj по вариантам, по классам, по освещению; лучший вариант сохраняется в cache/mat_v4_best.json."""
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as Fn

ROOT = Path("/home/cleaner/Cleaner")
sys.path.insert(0, str(ROOT / "audit"))
import full_pipeline as fp  # noqa: E402
from full_pipeline import D, MATS, MODELS, VIEWS  # noqa: E402
from mat_cv import fit_predict, weights  # noqa: E402
from mat_features2 import crop  # noqa: E402
from attr_features import polygon_mask, records  # noqa: E402

C = ROOT / "audit/cache"
DEV = "cuda"


def l2n(X):
    return X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-8)


def clahe_rgb(rgb):
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)
    lab[..., 0] = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(lab[..., 0])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)


def grayworld(rgb):
    f = rgb.astype(np.float32)
    m = f.reshape(-1, 3).mean(0) + 1e-6
    f = f * (m.mean() / m)
    return np.clip(f, 0, 255).astype(np.uint8)


def phys_feats(rgb, depth, mask, geo):
    inner = cv2.erode(mask.astype(np.uint8), np.ones((7, 7), np.uint8)) > 0
    if inner.sum() < 50:
        inner = mask
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    V, S_ = hsv[..., 2][inner].astype(np.float32), hsv[..., 1][inner].astype(np.float32)
    g = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)
    lap = np.abs(cv2.Laplacian(g, cv2.CV_32F))[inner]
    return np.array([
        geo["rough"] * 100, geo["holes_inner"], geo["holes_edge"], geo["valid"],
        V.mean() / 255, np.mean(V < 60), np.percentile(V, 99) / 255, np.mean(V > 240),
        S_.mean() / 255, np.mean(S_ < 50),
        lap.mean() / 20, np.percentile(lap, 90) / 50,
        math.log1p(geo["h95"] * 100), math.log1p(geo["ext1"] * 100), math.log1p(geo["ext2"] * 100),
    ], np.float32)


@torch.no_grad()
def embed_variants(enc):
    """Пересчёт CLIP-L c14 и zero-shot (rich) для вырезок с нормализацией освещения; + физические признаки."""
    out_p = C / "mat_v4_feats.npz"
    if out_p.exists():
        return dict(np.load(out_p))
    recs = records()
    F = dict(np.load(C / "features.npz"))
    gk = list(F["geo_keys"])
    crops = {"clahe": [], "gray": []}
    phys = []
    cache = {}
    for i, (src, im, a) in enumerate(recs):
        fd = D / Path(im["file_name"]).parent
        if fd not in cache:
            cache.clear()
            rgb = cv2.cvtColor(cv2.imread(str(fd / "color.png")), cv2.COLOR_BGR2RGB)
            depth = cv2.imread(str(fd / "depth.png"), cv2.IMREAD_UNCHANGED)
            cache[fd] = (rgb, depth)
        rgb, depth = cache[fd]
        m = polygon_mask(a["segmentation"], im["height"], im["width"])
        geo = dict(zip(gk, F["geo"][i]))
        phys.append(phys_feats(rgb, depth, m, geo))
        c14, _ = crop(rgb, m, a["bbox"], 1.5, False)
        crops["clahe"].append(clahe_rgb(c14))
        crops["gray"].append(grayworld(c14))
        if i % 200 == 0:
            print("crops", i, len(recs), flush=True)
    out = {"phys": np.stack(phys)}
    for k, arr in crops.items():
        arr = np.stack(arr)
        for tag in MODELS:
            out[f"{tag}_c14_{k}"] = enc.embed(tag, arr)
        print("emb", k, flush=True)
    np.savez_compressed(out_p, **out)
    return out


def M_frame(P, mat, idx):
    pred = P[idx].argmax(1)
    return float(np.mean([(pred[mat[idx] == c] == c).mean() for c in range(6) if (mat[idx] == c).any()]))


def M_obj(P, mat, oid, idx):
    hits = defaultdict(list)
    for o in sorted(set(oid[idx])):
        ii = idx[oid[idx] == o]
        hits[mat[ii[0]]].append(int(np.log(P[ii] + 1e-9).mean(0).argmax() == mat[ii[0]]))
    return float(np.mean([np.mean(v) for v in hits.values()])), {MATS[c]: f"{sum(v)}/{len(v)}" for c, v in sorted(hits.items())}


def looo_head(X, mat, oid, src, real, objs):
    P = np.zeros((len(mat), 6))
    for o in objs:
        tr = np.where((oid != o) & real)[0]
        te = np.where((oid == o) & real)[0]
        mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-6
        P[te] = fit_predict((X - mu) / sd, mat, weights(tr, oid, src, mat, 0.0), tr, te)
    return P


def fuse_nested(P_lin, Z, X, mat, oid, src, real, objs):
    from overfit_check import inner_weight
    Pf = np.zeros_like(P_lin)
    ws = []
    for o in objs:
        w = inner_weight(X, oid, src, mat, [q for q in objs if q != o], Z, seed=len(ws))
        ws.append(w)
        te = np.where((oid == o) & real)[0]
        Pf[te] = fp.softmax((1 - w) * np.log(P_lin[te] + 1e-9) + w * np.log(Z[te] + 1e-9))
    return Pf, float(np.median(ws))


def stack_nested(P_base, PH, mat, oid, real, objs):
    """Стек: [log P_base, физические признаки] → логрегрессия; C подбирается внутренним 5-fold по объектам."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    Xs = np.c_[np.log(P_base + 1e-9), PH]
    P = np.zeros_like(P_base)
    for o in objs:
        tr = np.where((oid != o) & real)[0]
        te = np.where((oid == o) & real)[0]
        tr_objs = sorted(set(oid[tr]))
        rs = np.random.default_rng(1)
        groups = np.array_split(rs.permutation(tr_objs), 5)
        best = None
        for Cv in (0.03, 0.1, 0.3, 1.0):
            Pin = np.zeros((len(tr), 6))
            for g in groups:
                a = np.array([i for i in tr if oid[i] not in set(g)])
                b = np.array([i for i in tr if oid[i] in set(g)])
                sc = StandardScaler().fit(Xs[a])
                clf = LogisticRegression(C=Cv, max_iter=2000, class_weight="balanced").fit(sc.transform(Xs[a]), mat[a])
                pb = np.zeros((len(b), 6))
                pb[:, clf.classes_] = clf.predict_proba(sc.transform(Xs[b]))
                Pin[np.searchsorted(tr, b)] = pb
            s = M_frame(Pin, mat[tr], np.arange(len(tr)))
            if best is None or s > best[0]:
                best = (s, Cv)
        sc = StandardScaler().fit(Xs[tr])
        clf = LogisticRegression(C=best[1], max_iter=2000, class_weight="balanced").fit(sc.transform(Xs[tr]), mat[tr])
        pt = np.zeros((len(te), 6))
        pt[:, clf.classes_] = clf.predict_proba(sc.transform(Xs[te]))
        P[te] = pt
    return P


def main():
    enc = fp.Encoders()
    F = dict(np.load(C / "features.npz"))
    F.update(dict(np.load(C / "features2.npz")))
    V4 = embed_variants(enc)
    meta = json.load(open(C / "meta.json"))
    oid = np.array([m["oid"] for m in meta])
    src = np.array([m["src"] for m in meta])
    mat = np.array([MATS.index(m["mat"]) for m in meta])
    light = np.array([m.get("lighting") or "" for m in meta])
    real = src == "real_train"
    idx = np.where(real)[0]
    objs = sorted(set(oid[real]))
    T = {t: enc.text[(t, "rich")] for t in MODELS}
    results = {}

    def report(name, P):
        mf = M_frame(P, mat, idx)
        mo, per = M_obj(P, mat, oid, idx)
        by_light = {lg: round(M_frame(P, mat, idx[light[idx] == lg]), 3) for lg in ("bright", "dim", "side")}
        bottles = {o: MATS[int(np.log(P[idx[oid[idx] == o]] + 1e-9).mean(0).argmax())] for o in ("s11", "s18", "s29", "s04", "s36", "s38", "s05", "b13", "b06")}
        print(f"  {name:44s} M={mf:.3f} M_obj={mo:.3f} {per} | свет {by_light} | бутылки/металл {bottles}", flush=True)
        results[name] = {"M": mf, "M_obj": mo, "per": per, "light": by_light}
        return mf

    print("== Материал v4, LOOO по 40 предметам (train), описания rich ==")
    Xb = (l2n(F["clipL_c14"]) * math.sqrt(768 / 64)).astype(np.float32)
    Zb = fp.softmax(np.mean([F[f"{t}_{v}"] @ T[t].T for t in MODELS for v in VIEWS], 0) / 0.01)
    P_lin = looo_head(Xb, mat, oid, src, real, objs)
    P_base, w = fuse_nested(P_lin, Zb, Xb, mat, oid, src, real, objs)
    report(f"база: голова ⊕ zero-shot (w≈{w:.2f})", P_base)

    PH = V4["phys"]
    P_phys = looo_head(np.c_[Xb, PH * 2.0], mat, oid, src, real, objs)
    P_phys_f, w2 = fuse_nested(P_phys, Zb, np.c_[Xb, PH * 2.0], mat, oid, src, real, objs)
    report(f"(1) голова на [CLIP + физ.признаки] ⊕ zs (w≈{w2:.2f})", P_phys_f)
    P_stack = stack_nested(P_base, PH, mat, oid, real, objs)
    report("(1) стек: [база, физ.признаки] → логрег", P_stack)

    for norm in ("clahe", "gray"):
        Xn = (l2n(V4[f"clipL_c14_{norm}"]) * math.sqrt(768 / 64)).astype(np.float32)
        Zn = fp.softmax(np.mean([V4[f"{t}_c14_{norm}"] @ T[t].T for t in MODELS], 0) / 0.01)
        P_ln = looo_head(Xn, mat, oid, src, real, objs)
        P_nf, wn = fuse_nested(P_ln, Zn, Xn, mat, oid, src, real, objs)
        report(f"(2) нормализация {norm}: голова ⊕ zs (w≈{wn:.2f})", P_nf)
        lp = 0.5 * (np.log(P_base + 1e-9) + np.log(P_nf + 1e-9))
        P_ens = np.exp(lp - lp.max(1, keepdims=True))
        P_ens /= P_ens.sum(1, keepdims=True)
        report(f"(2) ансамбль база + {norm}", P_ens)
        if norm == "clahe":
            P_clahe = P_nf
    lp = 0.5 * (np.log(P_stack + 1e-9) + np.log(P_clahe + 1e-9))
    P_all = np.exp(lp - lp.max(1, keepdims=True))
    P_all /= P_all.sum(1, keepdims=True)
    report("(1)+(2): стек с физ.признаками + CLAHE", P_all)
    json.dump(results, open(C / "mat_v4_results.json", "w"), indent=1, ensure_ascii=False)


if __name__ == "__main__":
    main()
