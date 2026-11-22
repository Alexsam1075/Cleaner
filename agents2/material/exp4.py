"""Эксперимент 4: TTA по масштабам/поворотам (усреднение эмбеддингов и/или вероятностей), поворотная аугментация головы,
усреднение по сетке w вместо argmax, DINOv2-головы в ансамбле, прототипный классификатор (Tip-adapter-подобный)."""
import argparse
import json
import math

import numpy as np

from common import HERE, MATS, MODELS, VIEWS, Standardizer, class_weights, fit_logreg, inner_folds, l2n, predict_logreg, report, softmax, M_frame
from exp1 import SET_KEYS, W_GRID, fuse, zs_logits
from exp2 import Runner2

GEN = [s for s in SET_KEYS if not s.startswith("rich") and not s.startswith("generic_syn")]
ROTS = ("c14", "c14r90", "c14r180", "c14r270")
SCALES = ("c12", "c14", "c17", "c20")


class Runner4(Runner2):
    def __init__(self):
        super().__init__()
        T = dict(np.load(HERE / "feats_tta.npz"))
        self.F.update(T)
        # усреднённые эмбеддинги (TTA на уровне признака)
        for tag in MODELS:
            self.F[f"{tag}_rotavg"] = l2n(np.mean([self.F[f"{tag}_{v}"] for v in ROTS], 0))
            self.F[f"{tag}_scaleavg"] = l2n(np.mean([self.F[f"{tag}_{v}"] for v in SCALES], 0))
            self.F[f"{tag}_rotscaleavg"] = l2n(np.mean([self.F[f"{tag}_{v}"] for v in ROTS + ("c12", "c17", "c20")], 0))

    def zs(self, skeys, views, mode="mean", temp=0.01):
        L = np.mean([zs_logits(self.F, s, MODELS, list(views), mode) for s in skeys], 0)
        return softmax(L / temp)


def m_general(R, train_objs, te, head_cfg=(("c14",), ("c14",)), zs_views=("m10", "m14", "c14", "c20"), skeys=("rich|floor",), mode="mean",
              tag="clipL", w_mode="argmax", wd=1e-2, extra_heads=()):
    """Голова (train_views→test_views, с опц. доп. головами на других признаках, среднее log-p) ⊕ zs по zs_views.
    w_mode: argmax по inner-OOF | avg (среднее предсказаний по w∈{0.4..0.75}) | fixed0.6."""
    tr = np.where(np.isin(R.oid, train_objs) & R.real)[0]
    Z = R.zs(skeys, zs_views, mode)
    heads = [(tag, head_cfg)] + list(extra_heads)
    Pin = softmax(np.mean([np.log(R.head_multiview(train_objs, None, t, c[0], c[1], wd, inner=True) + 1e-9) for t, c in heads], 0))
    Pte = softmax(np.mean([np.log(R.head_multiview(train_objs, te, t, c[0], c[1], wd) + 1e-9) for t, c in heads], 0))
    if w_mode == "argmax":
        w = max(W_GRID, key=lambda w: M_frame(fuse(Pin, Z, w), R.mat, tr))
        return fuse(Pte, Z[te], w), {"w": w}
    if w_mode == "avg":
        ws = [0.4, 0.5, 0.6, 0.75]
        return np.mean([fuse(Pte, Z[te], w) for w in ws], 0), {"w": "avg"}
    w = float(w_mode.replace("fixed", ""))
    return fuse(Pte, Z[te], w), {"w": w}


def m_proto(R, train_objs, te, skeys=("rich|floor",), views=("c14",), beta_grid=(0.0, 0.5, 1.0, 2.0, 4.0), temp=0.01, tag="clipL"):
    """Прототипы: text_emb(класс) + β·mean(image_emb класса на train, взвешенно по предметам) → cos → softmax/temp.
    Для каждой модели/вида; усреднение логитов. β вложенно (5-fold по объектам)."""
    tr = np.where(np.isin(R.oid, train_objs) & R.real)[0]
    TB = __import__("exp1").TB

    def logits(trn, idx, beta):
        out = []
        for tg in MODELS:
            T = np.mean([TB[f"{tg}|{s}"] for s in skeys], 0)
            for v in views:
                E = R.F[f"{tg}_{v}"]
                w = class_weights(trn, R.oid, R.src, R.mat)
                protos = []
                for c in range(6):
                    sel = R.mat[trn] == c
                    mu = (E[trn][sel] * w[sel, None]).sum(0) / w[sel].sum()
                    protos.append(l2n(T[c] + beta * mu))
                out.append(E[idx] @ np.stack(protos).T)
        return np.mean(out, 0)

    best, best_s = None, -1
    for beta in beta_grid:
        P = np.zeros((len(R.mat), 6))
        for g in inner_folds(train_objs, 5, 0):
            a = np.where(np.isin(R.oid, train_objs) & ~np.isin(R.oid, g) & R.real)[0]
            b = np.where(np.isin(R.oid, g) & R.real)[0]
            P[b] = softmax(logits(a, b, beta) / temp)
        s = M_frame(P, R.mat, tr)
        if s > best_s:
            best, best_s = beta, s
    return softmax(logits(tr, te, best) / temp), {"beta": best}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", default="all")
    a = ap.parse_args()
    R = Runner4()
    res = {}
    c14 = (("c14",), ("c14",))
    if a.part in ("all", "tta"):
        print("== 1. TTA: голова CLIP c14 ⊕ zs; варианты видов ==")
        _, res["base"] = R.run(m_general, "база (c14 голова, zs 4 вида)")
        _, res["base_avgw"] = R.run(m_general, "база, w усреднён по сетке", w_mode="avg")
        _, res["base_w06"] = R.run(m_general, "база, w=0.6 фикс.", w_mode="fixed0.6")
        _, res["zs_scales"] = R.run(m_general, "zs по масштабам c12..c20", zs_views=SCALES)
        _, res["zs_rots"] = R.run(m_general, "zs по поворотам c14", zs_views=ROTS)
        _, res["zs_rots_scales"] = R.run(m_general, "zs по поворотам + масштабам", zs_views=ROTS + ("c12", "c17", "c20"))
        _, res["zs_all"] = R.run(m_general, "zs по всем видам (11)", zs_views=tuple(VIEWS) + ROTS[1:] + ("c12", "c17", "c26"))
        print("\n== 2. Голова: поворотная аугментация / TTA ==")
        _, res["head_rot_tta"] = R.run(m_general, "голова c14 → TTA повороты", head_cfg=(("c14",), ROTS))
        _, res["head_rot_aug"] = R.run(m_general, "голова rot-aug → c14", head_cfg=(ROTS, ("c14",)))
        _, res["head_rot_aug_tta"] = R.run(m_general, "голова rot-aug → TTA повороты", head_cfg=(ROTS, ROTS))
        _, res["head_rotavg"] = R.run(m_general, "голова на rotavg-эмбеддинге", head_cfg=(("rotavg",), ("rotavg",)))
        _, res["head_scaleavg"] = R.run(m_general, "голова на scaleavg-эмбеддинге", head_cfg=(("scaleavg",), ("scaleavg",)))
        _, res["head_rotscaleavg"] = R.run(m_general, "голова на rotscaleavg-эмбеддинге", head_cfg=(("rotscaleavg",), ("rotscaleavg",)))
        _, res["head_scale_aug_tta"] = R.run(m_general, "голова scale-aug c12..c20 → TTA масштабы", head_cfg=(SCALES, SCALES))
        _, res["head_rot_aug_tta_zsrot"] = R.run(m_general, "голова rot-aug→TTA + zs повороты+масштабы", head_cfg=(ROTS, ROTS), zs_views=ROTS + ("c12", "c17", "c20"))
        _, res["head_rotavg_zsrot"] = R.run(m_general, "голова rotavg + zs повороты+масштабы", head_cfg=(("rotavg",), ("rotavg",)), zs_views=ROTS + ("c12", "c17", "c20"))
    if a.part in ("all", "ens"):
        print("\n== 3. Ансамбли голов (среднее log-p) ⊕ zs ==")
        _, res["ens_clip_siglip_rotavg"] = R.run(m_general, "avg[clip rotavg, siglip rotavg] ⊕ zs", head_cfg=(("rotavg",), ("rotavg",)), extra_heads=[("siglip", (("rotavg",), ("rotavg",)))])
        _, res["ens_clip_dino"] = R.run(m_general, "avg[clip c14, dinoL c14] ⊕ zs", extra_heads=[("dinoL", c14)])
        _, res["ens_clip_dino_mpool"] = R.run(m_general, "avg[clip c14, dinoL c14_mpool] ⊕ zs", extra_heads=[("dinoL", (("c14_mpool",), ("c14_mpool",)))])
        _, res["ens_clip_siglip_dino"] = R.run(m_general, "avg[clip c14, siglip c14, dinoL c14] ⊕ zs", extra_heads=[("siglip", c14), ("dinoL", c14)])
        print("\n== 4. Прототипы (текст + β·среднее изображений) ==")
        _, res["proto_c14"] = R.run(m_proto, "прототипы rich, c14")
        _, res["proto_4v"] = R.run(m_proto, "прототипы rich, 4 вида", views=tuple(VIEWS))
        _, res["proto_gen_c14"] = R.run(m_proto, "прототипы общие, c14", skeys=tuple(GEN))
    json.dump(res, open(HERE / f"exp4_{a.part}.json", "w"), indent=1, default=str)


if __name__ == "__main__":
    main()
