"""Материал v3: объект целиком (силуэт + форма + контекст) + отдельные признаки цвета и текстуры.
Честный leave-one-object-out (40 предметов), только real_train + synth_train.

Блоки:
  shape  — эмбеддинги всей вырезки объекта: CLIP-L контекст (c14), CLIP-L маска (m10), DINOv2-L маска;
  color  — ручные признаки цвета (HSV/Lab гистограммы, моменты, блики, хроматичность);
  tex    — ручные признаки текстуры (LBP, Габор, градиенты, локальный контраст) + эмбеддинги текстурных патчей;
  zs     — zero-shot CLIP/SigLIP по описаниям материалов (вся вырезка и патчи).
Сравнение: каждый блок отдельно, shape+color, shape+tex, shape+color+tex, и ансамбль с zero-shot.
Вес zero-shot подбирается во ВЛОЖЕННОМ LOOO (внутри 39 обучающих объектов), внешний объект не участвует."""
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np

from mat_cv import MATS, fit_predict, standardize, weights

C = Path(__file__).resolve().parent / "cache"
HAND_COLOR = slice(0, 12 + 8 + 8 + 16 + 15 + 5)


def load():
    F = dict(np.load(C / "features.npz"))
    F.update(dict(np.load(C / "features2.npz")))
    F.update(dict(np.load(C / "features_ct.npz")))
    meta = json.load(open(C / "meta.json"))
    return F, np.array([m["oid"] for m in meta]), np.array([m["src"] for m in meta]), np.array([MATS.index(m["mat"]) for m in meta])


def l2n(X):
    return X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-8)


def softmax(z):
    z = z - z.max(1, keepdims=True)
    p = np.exp(z)
    return p / p.sum(1, keepdims=True)


def M_frame(P, mat, idx):
    pred = P[idx].argmax(1)
    return float(np.mean([(pred[mat[idx] == c] == c).mean() for c in range(6) if (mat[idx] == c).any()]))


def M_obj(P, mat, oid, idx):
    hits = defaultdict(list)
    for o in sorted(set(oid[idx])):
        ii = idx[oid[idx] == o]
        hits[mat[ii[0]]].append(int(np.log(P[ii] + 1e-9).mean(0).argmax() == mat[ii[0]]))
    return float(np.mean([np.mean(v) for v in hits.values()])), {MATS[c]: f"{sum(v)}/{len(v)}" for c, v in sorted(hits.items())}


def block(F, name):
    hand = F["hand"]
    if name == "color":
        return hand[:, HAND_COLOR]
    if name == "tex_hand":
        return hand[:, HAND_COLOR.stop:]
    X = F[name]
    return l2n(X) * math.sqrt(X.shape[1] / 64)


def build(F, names):
    return np.concatenate([block(F, n) for n in names], 1).astype(np.float32)


def looo(X, oid, src, mat, objs, real, synth_w=0.0, wd=1e-2, restrict=None):
    P = np.zeros((len(mat), 6))
    pool = restrict if restrict is not None else np.ones(len(mat), bool)
    for o in objs:
        tr = np.where(pool & (oid != o) & (real | (synth_w > 0)))[0]
        te = np.where(pool & (oid == o) & real)[0]
        P[te] = fit_predict(standardize(X, tr), mat, weights(tr, oid, src, mat, synth_w), tr, te, wd=wd)
    return P


def main():
    F, oid, src, mat = load()
    real = src == "real_train"
    idx = np.where(real)[0]
    objs = sorted(set(oid[real]))
    ZS = ["clipL_m10_zs", "clipL_m14_zs", "clipL_c14_zs", "clipL_c20_zs", "siglip_m10_zs", "siglip_m14_zs", "siglip_c14_zs", "siglip_c20_zs"]
    zs_full = softmax(sum(F[k] for k in ZS) / len(ZS) / 0.01)
    zs_tex = softmax(F["tex_clipL_zs"] / 0.01)

    def show(name, P):
        mo, per = M_obj(P, mat, oid, idx)
        print(f"  {name:44s} M={M_frame(P, mat, idx):.3f}  M_obj={mo:.3f} {per}", flush=True)

    print("== Zero-shot ==")
    show("zs: вся вырезка (CLIP-L+SigLIP)", zs_full)
    show("zs: только текстурные патчи (CLIP-L)", zs_tex)
    show("zs: вырезка × патчи", softmax(np.log(zs_full + 1e-9) + 0.5 * np.log(zs_tex + 1e-9)))

    print("\n== Линейная голова: блоки по отдельности и вместе (LOOO) ==")
    configs = {
        "color": ["color"], "tex_hand": ["tex_hand"], "color+tex_hand": ["color", "tex_hand"],
        "tex_patches(dinoB)": ["tex_dinoB"], "tex_patches(clipL)": ["tex_clipL"],
        "shape: clipL_c14": ["clipL_c14"],
        "shape+color": ["clipL_c14", "color"],
        "shape+tex_hand": ["clipL_c14", "tex_hand"],
        "shape+color+tex_hand": ["clipL_c14", "color", "tex_hand"],
        "shape+tex_patches": ["clipL_c14", "tex_clipL"],
        "shape+color+tex_hand+tex_patches": ["clipL_c14", "color", "tex_hand", "tex_clipL"],
        "shape2(c14+m10)+color+tex": ["clipL_c14", "clipL_m10", "color", "tex_hand", "tex_clipL"],
    }
    lin = {}
    for name, names in configs.items():
        P = looo(build(F, names), oid, src, mat, objs, real)
        lin[name] = P
        show(name, P)

    print("\n== Ансамбль с zero-shot, вес подбирается во вложенном LOOO ==")
    zs_comb = softmax(np.log(zs_full + 1e-9) + 0.5 * np.log(zs_tex + 1e-9))
    grid = [0.0, 0.25, 0.4, 0.5, 0.6, 0.75, 1.0]
    for name in ["shape: clipL_c14", "shape+color", "shape+color+tex_hand", "shape+color+tex_hand+tex_patches", "shape2(c14+m10)+color+tex"]:
        X = build(F, configs[name])
        for zname, Z in (("zs_full", zs_full), ("zs_full×tex", zs_comb)):
            Pfin = np.zeros((len(mat), 6))
            chosen = []
            for o in objs:
                inner_objs = [q for q in objs if q != o]
                inner_mask = oid != o
                Pin = np.zeros((len(mat), 6))
                rs = np.random.default_rng(len(chosen))
                groups = np.array_split(rs.permutation(inner_objs), 5)
                for g in groups:
                    trm = inner_mask & ~np.isin(oid, g) & real
                    tem = np.isin(oid, g) & real
                    tr, te_in = np.where(trm)[0], np.where(tem)[0]
                    Pin[te_in] = fit_predict(standardize(X, tr), mat, weights(tr, oid, src, mat, 0.0), tr, te_in)
                ii = np.where(inner_mask & real)[0]
                best_w = max(grid, key=lambda w: M_frame(softmax((1 - w) * np.log(Pin + 1e-9) + w * np.log(Z + 1e-9)), mat, ii))
                chosen.append(best_w)
                te = np.where((oid == o) & real)[0]
                Pfin[te] = softmax((1 - best_w) * np.log(lin[name][te] + 1e-9) + best_w * np.log(Z[te] + 1e-9))
            show(f"{name} ⊕ {zname} (w≈{np.median(chosen):.2f})", Pfin)
            np.save(C / f"v3_{name.replace(' ', '').replace(':', '')}_{zname}.npy", Pfin)


if __name__ == "__main__":
    main()
