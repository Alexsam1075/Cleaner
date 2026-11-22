"""Честная оценка атрибутов на уровне объектов: leave-one-object-out по 40 физическим предметам (real-кадры),
синтетика с тем же object_id исключается из обучения фолда. Подбор гиперпараметров — только внутри обучающей части.
Считает M (macro по материалам, как score.py), V, S на реальных кадрах при истинных масках (верхняя граница для атрибутов).

Варианты:
  материал: logreg на эмбеддингах разных сетей (+фото-признаки, +контекст), с/без синтетики;
  объём:   геометрия сырая / калибровка ridge (геометрия) / + предсказанный класс (OOF);
  take:    объём*плотность(табл.) <= T / логрег на признаках / регрессия log-массы <= 500; T подбирается на inner-CV."""
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.preprocessing import StandardScaler

C = Path(__file__).resolve().parent / "cache"
F = np.load(C / "features.npz")
META = json.load(open(C / "meta.json"))
MATS = ["metal", "wood", "plastic", "glass", "paper", "mineral"]
FORMS = ["bulk", "solid"]
DENS = {}
import csv
import os

for r in csv.DictReader(open(Path(os.environ.get("ASTRAMIS_DATA", "dataset")) / "densities.csv")):
    DENS[(r["form"], r["material"])] = float(r["density_g_per_ml"])

src = np.array([m["src"] for m in META])
oid = np.array([m["oid"] for m in META])
mat = np.array([MATS.index(m["mat"]) for m in META])
form = np.array([FORMS.index(m["form"]) for m in META])
vol = np.array([m["vol"] for m in META], float)
mass = np.array([m["mass"] for m in META], float)
take = np.array([m["take"] for m in META])
gk = list(F["geo_keys"])
G = F["geo"]
geo = lambda k: G[:, gk.index(k)]
REAL = src == "real_train"
OBJS = sorted(set(oid[REAL]))


def macro_M(pred, true):
    return float(np.mean([(pred[true == c] == c).mean() for c in np.unique(true)]))


def V_of(p, g):
    p = np.clip(p, 1e-3, None)
    return float(np.clip(1 - np.abs(np.log(p / g)) / math.log(3), 0, 1).mean())


def S_of(p, t):
    return float(np.mean([(p[t == v] == v).mean() for v in (0, 1) if (t == v).any()]))


def geo_feats():
    v = np.log1p(geo("vol"))
    return np.c_[v, np.log1p(geo("vol_thr")), np.log1p(geo("foot")), geo("h95") * 20, geo("hmax") * 20, np.log1p(geo("ext1") * 100),
                 np.log1p(geo("ext2") * 100), geo("rough") * 100, geo("valid"), geo("holes_inner"), np.log1p(geo("surf"))]


def emb(name):
    return F[name]


def mat_features(kind):
    parts = {"photo": F["photo"], "geo": geo_feats()}
    out = []
    for k in kind.split("+"):
        out.append(parts[k] if k in parts else emb(k))
    return np.concatenate(out, 1)


def sample_weights(idx):
    """Каждый физический объект суммарно весит одинаково; синтетика весит 0.5 от реального."""
    w = np.ones(len(idx))
    cnt = defaultdict(int)
    for i in idx:
        cnt[(oid[i], src[i])] += 1
    for j, i in enumerate(idx):
        w[j] = (1.0 if src[i] == "real_train" else 0.5) / cnt[(oid[i], src[i])]
    by_mat = defaultdict(float)
    for j, i in enumerate(idx):
        by_mat[mat[i]] += w[j]
    for j, i in enumerate(idx):
        w[j] /= by_mat[mat[i]]
    return w / w.mean()


def fit_mat(X, idx, Cval=0.5):
    sc = StandardScaler().fit(X[idx])
    clf = LogisticRegression(C=Cval, max_iter=3000)
    clf.fit(sc.transform(X[idx]), mat[idx], sample_weight=sample_weights(idx))
    return sc, clf


def loo(kind, use_synth=True, Cval=0.5):
    X = mat_features(kind)
    proba = np.zeros((len(META), 6))
    for o in OBJS:
        tr = np.where((oid != o) & (REAL | (use_synth & ~REAL)))[0]
        te = np.where((oid == o) & REAL)[0]
        sc, clf = fit_mat(X, tr, Cval)
        p = np.zeros((len(te), 6))
        p[:, clf.classes_] = clf.predict_proba(sc.transform(X[te]))
        proba[te] = p
    return proba


def report_mat(proba, name):
    idx = np.where(REAL)[0]
    pred = proba[idx].argmax(1)
    M = macro_M(pred, mat[idx])
    per = {MATS[c]: f"{(pred[mat[idx] == c] == c).sum()}/{(mat[idx] == c).sum()}" for c in range(6)}
    by_obj = defaultdict(list)
    for i in idx:
        by_obj[oid[i]].append(proba[i])
    obj_ok = defaultdict(list)
    for o, ps in by_obj.items():
        c = MATS.index(META[np.where(oid == o)[0][0]]["mat"])
        obj_ok[c].append(int(np.mean(ps, 0).argmax() == c))
    Mobj = float(np.mean([np.mean(v) for v in obj_ok.values()]))
    print(f"  {name:42s} M(кадры)={M:.3f}  M(объекты, голосование)={Mobj:.3f}  {per}")
    return M


def main():
    print("== Материал: leave-one-object-out (40 фолдов), истинные маски ==")
    res = {}
    for kind in ["photo+geo", "convnext_m", "clip_b_m", "dinov2_s_m", "dinov2_b_m", "dinov2_b_c", "clip_b_c",
                 "dinov2_b_m+clip_b_m", "dinov2_b_m+photo", "clip_b_m+photo", "dinov2_b_m+clip_b_m+photo"]:
        for syn in (False, True):
            p = loo(kind, syn)
            res[(kind, syn)] = (report_mat(p, f"{kind} {'+synth' if syn else 'real'}"), p)
    best = max(res, key=lambda k: res[k][0])
    print("ЛУЧШИЙ материал:", best, round(res[best][0], 3))
    P = res[best][1]
    np.save(C / "mat_oof.npy", P)

    print("\n== Форма (bulk/solid) по тем же признакам ==")
    X = mat_features(best[0])
    fp = np.zeros(len(META))
    for o in OBJS:
        tr = np.where((oid != o) & (REAL | (best[1] & ~REAL)))[0]
        te = np.where((oid == o) & REAL)[0]
        sc = StandardScaler().fit(X[tr])
        clf = LogisticRegression(C=0.5, max_iter=3000, class_weight="balanced").fit(sc.transform(X[tr]), form[tr])
        fp[te] = clf.predict_proba(sc.transform(X[te]))[:, 1]
    idx = np.where(REAL)[0]
    fpred = (fp[idx] > 0.5).astype(int)
    print(f"  form acc={(fpred == form[idx]).mean():.3f}  bulk {((fpred == 0) & (form[idx] == 0)).sum()}/{(form[idx] == 0).sum()}  solid {((fpred == 1) & (form[idx] == 1)).sum()}/{(form[idx] == 1).sum()}")

    print("\n== Объём (LOO по объектам; класс — OOF-предсказание, не истина) ==")
    Gf = geo_feats()
    pm = P.argmax(1)
    pf = (fp > 0.5).astype(int)
    onehot = lambda m, f: np.c_[np.eye(6)[m], np.eye(2)[f], np.eye(6)[m] * f[:, None]]
    out = {}
    raw = np.where(geo("vol") > 1, geo("vol"), np.median(vol))
    out["геометрия как есть"] = raw[idx]
    for name, Xv in [("ridge(геометрия)", Gf), ("ridge(геометрия+OOF-класс)", np.c_[Gf, onehot(pm, pf)]),
                     ("ridge(геометрия+OOF-класс+эмб)", np.c_[Gf, onehot(pm, pf), mat_features("dinov2_b_m")[:, :64]])]:
        pv = np.zeros(len(META))
        for o in OBJS:
            tr = np.where((oid != o) & REAL)[0]
            te = np.where((oid == o) & REAL)[0]
            sc = StandardScaler().fit(Xv[tr])
            r = Ridge(alpha=3.0).fit(sc.transform(Xv[tr]), np.log(vol[tr]))
            pv[te] = np.exp(r.predict(sc.transform(Xv[te])))
        out[name] = pv[idx]
    bins = [(0, 50), (50, 200), (200, 1000), (1000, 1e9)]
    for k, p in out.items():
        b = " ".join(f"{lo:g}-{hi:g}:{V_of(p[(vol[idx] >= lo) & (vol[idx] < hi)], vol[idx][(vol[idx] >= lo) & (vol[idx] < hi)]):.2f}" for lo, hi in bins)
        print(f"  {k:36s} V={V_of(p, vol[idx]):.3f}   {b}")
    best_v = max(out, key=lambda k: V_of(out[k], vol[idx]))
    PV = out[best_v]

    print("\n== Take (порог/модель подбираются во внутреннем CV по объектам) ==")
    dens = np.array([DENS[(FORMS[f], MATS[m])] for f, m in zip(pf[idx], pm[idx])])
    est_mass = PV * dens
    objs_idx = oid[idx]

    def inner_thr(values, mask_tr):
        best_t, best_s = 500, -1
        for t in np.quantile(values[mask_tr], np.linspace(0.02, 0.98, 97)):
            s = S_of((values[mask_tr] <= t).astype(int), take[idx][mask_tr])
            if s > best_s:
                best_t, best_s = t, s
        return best_t

    for name, val in [("объём×плотность ≤ 500 (фикс.)", None), ("объём×плотность ≤ T(inner)", est_mass), ("объём ≤ T(inner)", PV)]:
        pred = np.zeros(len(idx), int)
        for o in OBJS:
            te = objs_idx == o
            if val is None:
                pred[te] = (est_mass[te] <= 500).astype(int)
            else:
                t = inner_thr(val, ~te)
                pred[te] = (val[te] <= t).astype(int)
        print(f"  {name:34s} S={S_of(pred, take[idx]):.3f}  тяжёлые {((pred == 0) & (take[idx] == 0)).sum()}/{(take[idx] == 0).sum()}  лёгкие {((pred == 1) & (take[idx] == 1)).sum()}/{(take[idx] == 1).sum()}")
    Xt = np.c_[np.log(PV), np.log(est_mass), Gf, P[idx], pf[idx]]
    pred = np.zeros(len(idx), int)
    for o in OBJS:
        te = objs_idx == o
        sc = StandardScaler().fit(Xt[~te])
        clf = LogisticRegression(C=0.3, class_weight="balanced", max_iter=3000).fit(sc.transform(Xt[~te]), take[idx][~te])
        pred[te] = clf.predict(sc.transform(Xt[te]))
    print(f"  {'логрег(объём, масса, геом, P(мат))':34s} S={S_of(pred, take[idx]):.3f}  тяжёлые {((pred == 0) & (take[idx] == 0)).sum()}/{(take[idx] == 0).sum()}  лёгкие {((pred == 1) & (take[idx] == 1)).sum()}/{(take[idx] == 1).sum()}")


if __name__ == "__main__":
    main()
