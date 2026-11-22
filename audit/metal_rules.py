"""Металл: эвристики + память силуэтов. LOOO по 40 train-предметам, вложенный подбор весов; тест — только инференс.
 (a) Эвристика «тёмный металл»: доля тёмных пикселей, низкая насыщенность, гладкость по глубине (rough), дыры глубины.
     Собирается в один скор m = σ(w·z) где z — стандартизованные признаки, w подбирается логрегрессией
     «metal vs остальные» на train (LOOO). Применяется как мягкая подсказка: log P(metal) += λ·logit(m).
 (b) Память силуэтов: дескриптор маски без цвета и бликов — 7 моментов Ху (log), вытянутость, заполненность (solidity),
     высота/ширина следа, отношение высоты к длине. kNN по train-предметам (один голос на предмет) → P_sil(материал).
     Смешивание: log P += μ·log P_sil. λ, μ — внутренний 5-fold CV по объектам.
 (c) Объединение (a)+(b).
Отчёт: M, M_obj, металл по предметам, по освещению; затем применение к тесту: какие кадры меняют материал."""
import json
import math
import pickle
import sys
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

ROOT = Path("/home/cleaner/Cleaner")
sys.path.insert(0, str(ROOT / "audit"))
import full_pipeline as fp  # noqa: E402
from full_pipeline import C, D, MATS, MODELS, VIEWS  # noqa: E402
from mat_cv import fit_predict, weights  # noqa: E402
from attr_features import polygon_mask, records  # noqa: E402
from mat_v4 import M_frame, M_obj, l2n  # noqa: E402
from overfit_check import inner_weight  # noqa: E402

METAL = MATS.index("metal")


def silhouette_desc(mask, geo):
    m8 = mask.astype(np.uint8)
    cnts, _ = cv2.findContours(m8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return np.zeros(13, np.float32)
    c = max(cnts, key=cv2.contourArea)
    hu = cv2.HuMoments(cv2.moments(c)).flatten()
    hu = -np.sign(hu) * np.log10(np.abs(hu) + 1e-12)
    area = cv2.contourArea(c)
    hull = cv2.convexHull(c)
    solidity = area / max(cv2.contourArea(hull), 1)
    (_, _), (w, h), _ = cv2.minAreaRect(c)
    elong = max(w, h) / max(min(w, h), 1)
    ext1, ext2, h95 = max(geo["ext1"], 1e-3), max(geo["ext2"], 1e-3), max(geo["h95"], 1e-3)
    return np.array([*hu, solidity, math.log(elong), math.log(ext1 * 100), math.log(ext2 * 100), math.log(h95 / ext1), math.log(ext2 / ext1)], np.float32)[:13]


def phys(rgb, mask, geo):
    inner = cv2.erode(mask.astype(np.uint8), np.ones((7, 7), np.uint8)) > 0
    if inner.sum() < 50:
        inner = mask
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    V, S_ = hsv[..., 2][inner].astype(np.float32), hsv[..., 1][inner].astype(np.float32)
    g = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)
    lap = np.abs(cv2.Laplacian(g, cv2.CV_32F))[inner]
    return np.array([np.mean(V < 60), V.mean() / 255, np.percentile(V, 99) / 255, np.mean(V > 235), S_.mean() / 255, np.mean(S_ < 50),
                     math.log(max(geo["rough"], 1e-4)), geo["holes_inner"], geo["holes_edge"], lap.mean() / 20], np.float32)


def build_train():
    p = C / "metal_rules_feats.npz"
    if p.exists():
        d = dict(np.load(p))
        return d["sil"], d["phys"]
    F = dict(np.load(C / "features.npz"))
    gk = list(F["geo_keys"])
    recs = records()
    sil, ph = [], []
    cache = {}
    for i, (src, im, a) in enumerate(recs):
        fd = D / Path(im["file_name"]).parent
        if fd not in cache:
            cache.clear()
            cache[fd] = cv2.cvtColor(cv2.imread(str(fd / "color.png")), cv2.COLOR_BGR2RGB)
        rgb = cache[fd]
        m = polygon_mask(a["segmentation"], im["height"], im["width"])
        geo = dict(zip(gk, F["geo"][i]))
        sil.append(silhouette_desc(m, geo))
        ph.append(phys(rgb, m, geo))
        if i % 200 == 0:
            print("feats", i, len(recs), flush=True)
    sil, ph = np.stack(sil), np.stack(ph)
    np.savez_compressed(p, sil=sil, phys=ph)
    return sil, ph


def logreg_metal(Xtr, ytr, w, C_=0.3):
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    sc = StandardScaler().fit(Xtr)
    clf = LogisticRegression(C=C_, max_iter=2000, class_weight="balanced").fit(sc.transform(Xtr), ytr, sample_weight=w)
    return lambda X: clf.predict_proba(sc.transform(X))[:, 1]


def sil_knn(S_tr, mat_tr, oid_tr, S_te, k=7):
    """kNN по силуэтам: голос каждого train-предмета = макс. сходство среди его кадров (один голос на предмет)."""
    mu, sd = S_tr.mean(0), S_tr.std(0) + 1e-6
    A, B = (S_tr - mu) / sd, (S_te - mu) / sd
    objs = sorted(set(oid_tr))
    out = np.zeros((len(S_te), 6))
    for j in range(len(B)):
        d2 = ((A - B[j]) ** 2).sum(1)
        per_obj = {o: (d2[oid_tr == o].min(), mat_tr[oid_tr == o][0]) for o in objs}
        top = sorted(per_obj.values(), key=lambda t: t[0])[:k]
        wsum = np.zeros(6)
        for d_, m_ in top:
            wsum[m_] += math.exp(-d_ / (2 * np.median([t[0] for t in top]) + 1e-6))
        out[j] = (wsum + 1e-3) / (wsum + 1e-3).sum()
    return out


def main():
    enc = fp.Encoders()
    F = dict(np.load(C / "features.npz"))
    F.update(dict(np.load(C / "features2.npz")))
    SIL, PH = build_train()
    meta = json.load(open(C / "meta.json"))
    oid = np.array([m["oid"] for m in meta])
    src = np.array([m["src"] for m in meta])
    mat = np.array([MATS.index(m["mat"]) for m in meta])
    light = np.array([m.get("lighting") or "" for m in meta])
    real = src == "real_train"
    idx = np.where(real)[0]
    objs = sorted(set(oid[real]))
    T = {t: enc.text[(t, "rich")] for t in MODELS}
    Z = fp.softmax(np.mean([F[f"{t}_{v}"] @ T[t].T for t in MODELS for v in VIEWS], 0) / 0.01)
    X = (l2n(F["clipL_c14"]) * math.sqrt(768 / 64)).astype(np.float32)

    def report(name, P):
        mf = M_frame(P, mat, idx)
        mo, per = M_obj(P, mat, oid, idx)
        by_light = {lg: round(M_frame(P, mat, idx[light[idx] == lg]), 3) for lg in ("bright", "dim", "side")}
        metal_obj = {o: MATS[int(np.log(P[idx[oid[idx] == o]] + 1e-9).mean(0).argmax())] for o in ("b02", "b06", "b13", "s05", "s45")}
        print(f"  {name:36s} M={mf:.3f} M_obj={mo:.3f} {per} | свет {by_light} | металл {metal_obj}", flush=True)

    # база (OOF, вложенный вес zs)
    P_lin = np.zeros((len(mat), 6))
    for o in objs:
        tr = np.where((oid != o) & real)[0]
        te = np.where((oid == o) & real)[0]
        mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-6
        P_lin[te] = fit_predict((X - mu) / sd, mat, weights(tr, oid, src, mat, 0.0), tr, te)
    P_base = np.zeros_like(P_lin)
    ws = []
    for o in objs:
        w = inner_weight(X, oid, src, mat, [q for q in objs if q != o], Z, seed=len(ws))
        ws.append(w)
        te = np.where((oid == o) & real)[0]
        P_base[te] = fp.softmax((1 - w) * np.log(P_lin[te] + 1e-9) + w * np.log(Z[te] + 1e-9))
    print("== Металл: эвристики и память силуэтов (LOOO по 40 предметам, rich) ==")
    report("база", P_base)

    # (a) эвристика металла: OOF-скор m, затем подсказка с λ по внутреннему CV
    m_oof = np.zeros(len(mat))
    for o in objs:
        tr = np.where((oid != o) & real)[0]
        te = np.where((oid == o) & real)[0]
        f = logreg_metal(PH[tr], (mat[tr] == METAL).astype(int), weights(tr, oid, src, mat, 0.0))
        m_oof[te] = f(PH[te])
    print(f"  эвристика сама по себе: AUC metal vs rest = {_auc(m_oof[idx], mat[idx] == METAL):.3f}; по предметам: " +
          ", ".join(f"{o}:{m_oof[idx[oid[idx] == o]].mean():.2f}" for o in ("b02", "b06", "b13", "s05", "s45", "s11", "s34", "s19")))

    def apply_rule(P, m, lam):
        L = np.log(P + 1e-9).copy()
        L[:, METAL] += lam * (np.log(m + 1e-6) - np.log(1 - m + 1e-6))
        return fp.softmax(L)

    # (b) память силуэтов OOF
    P_sil = np.zeros((len(mat), 6))
    for o in objs:
        tr = np.where((oid != o) & real)[0]
        te = np.where((oid == o) & real)[0]
        P_sil[te] = sil_knn(SIL[tr], mat[tr], oid[tr], SIL[te])
    report("силуэты сами по себе (kNN)", P_sil)

    def apply_sil(P, Ps, mu_):
        return fp.softmax(np.log(P + 1e-9) + mu_ * np.log(Ps + 1e-9))

    # подбор λ, μ вложенно: внутри каждого внешнего фолда — по OOF train-объектов (используем P_base/m_oof/P_sil как прокси inner-OOF)
    lam_grid = [0.0, 0.25, 0.5, 1.0, 1.5]
    mu_grid = [0.0, 0.25, 0.5, 1.0]
    P_a, P_b, P_c = np.zeros_like(P_base), np.zeros_like(P_base), np.zeros_like(P_base)
    picks = defaultdict(list)
    for o in objs:
        tr = np.where((oid != o) & real)[0]
        te = np.where((oid == o) & real)[0]
        best_l = max(lam_grid, key=lambda l: M_frame(apply_rule(P_base, m_oof, l), mat, tr))
        best_m = max(mu_grid, key=lambda u: M_frame(apply_sil(P_base, P_sil, u), mat, tr))
        best_lm = max(((l, u) for l in lam_grid for u in mu_grid), key=lambda t: M_frame(apply_sil(apply_rule(P_base, m_oof, t[0]), P_sil, t[1]), mat, tr))
        picks["λ"].append(best_l)
        picks["μ"].append(best_m)
        picks["λμ"].append(best_lm)
        P_a[te] = apply_rule(P_base, m_oof, best_l)[te]
        P_b[te] = apply_sil(P_base, P_sil, best_m)[te]
        P_c[te] = apply_sil(apply_rule(P_base, m_oof, best_lm[0]), P_sil, best_lm[1])[te]
    report(f"(a) база + эвристика металла (λ≈{np.median(picks['λ'])})", P_a)
    report(f"(b) база + память силуэтов (μ≈{np.median(picks['μ'])})", P_b)
    report("(c) база + эвристика + силуэты", P_c)
    json.dump({"lam": float(np.median(picks["λ"])), "mu": float(np.median(picks["μ"])), "lam_mu": [float(x) for x in np.median(np.array(picks["λμ"]), 0)]},
              open(C / "metal_rules_params.json", "w"))

    # применение к тесту (инференс): какие кадры меняют материал
    tp = C / "test_dets.pkl"
    if tp.exists():
        dets, embs = pickle.load(open(tp, "rb"))
        print("\n== Тест: что меняют правила (инференс, без меток) ==")
        gk = list(F["geo_keys"])
        import cv2 as _cv
        from ultralytics import YOLO
        det = YOLO(str(ROOT / "audit/final/det_all/weights/last.pt"))
        pred_mat, _ = fp.material_model(F, meta, objs, enc.text, "rich")
        zs_t = fp.zero_shot({(t, v): embs[(t, v)] for t in MODELS for v in VIEWS}, enc.text, "rich")
        Pt = pred_mat(embs[("clipL", "c14")], zs_t)
        f_metal = logreg_metal(PH[idx], (mat[idx] == METAL).astype(int), weights(idx, oid, src, mat, 0.0))
        lam, mu_ = np.median(picks["λ"]), np.median(picks["μ"])
        changes = []
        for k, d in enumerate(dets):
            if d["score"] < 0.3:
                continue
            fd = D / "real/test" / d["frame"]
            rgb = _cv.cvtColor(_cv.imread(str(fd / "color.png")), _cv.COLOR_BGR2RGB)
            r = det.predict(str(fd / "color.png"), imgsz=640, conf=0.08, retina_masks=True, verbose=False, device=0)[0]
            if r.masks is None:
                continue
            j = int(np.argmax([fp._iou(d["box"], b) for b in r.boxes.xyxy.cpu().numpy().tolist()]))
            m = r.masks.data[j].cpu().numpy() > 0.5
            if m.shape != rgb.shape[:2]:
                m = _cv.resize(m.astype(np.uint8), (rgb.shape[1], rgb.shape[0]), interpolation=_cv.INTER_NEAREST) > 0
            g = d["geo"]
            mscore = f_metal(phys(rgb, m, g)[None])[0]
            Ps = sil_knn(SIL[idx], mat[idx], oid[idx], silhouette_desc(m, g)[None])
            P2 = apply_sil(apply_rule(Pt[k:k + 1], np.array([mscore]), lam), Ps, mu_)
            a, b = MATS[int(Pt[k].argmax())], MATS[int(P2[0].argmax())]
            if a != b:
                changes.append((d["frame"], a, b, round(float(mscore), 2)))
        print(f"  изменений материала на тесте: {len(changes)}")
        for c in changes:
            print("   ", c)


def _auc(s, y):
    from sklearn.metrics import roc_auc_score
    return roc_auc_score(y, s)


if __name__ == "__main__":
    main()
