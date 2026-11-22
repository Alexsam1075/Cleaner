"""Эксперимент 6: голова в «пространстве описаний» — признаки = косинусы вырезки ко всем формулировкам общих словарей
(CLIP-L + SigLIP, виды c14+c20), логрегрессия с L2. Сама по себе, ⊕ zs lse, и в среднем с CLIP-головой.
Плюс: голова на конкатенации [CLIP c14 ⊕ prompt-sims]."""
import argparse
import json
import math

import numpy as np

from common import HERE, MATS, MODELS, VIEWS, Standardizer, class_weights, fit_logreg, inner_folds, l2n, predict_logreg, report, softmax, M_frame
from exp1 import SET_KEYS, TB, W_GRID, fuse
from exp5 import Runner5

DICTS_ALL = ["rich|floor", "names|floor", "waste|floor", "surface|floor", "shape_free|floor", "packaging|floor"]
DICTS_GEN = ["names|floor", "waste|floor", "surface|floor", "shape_free|floor", "packaging|floor"]


class Runner6(Runner5):
    def prompt_sims(self, dicts, views=("c14", "c20"), tags=MODELS):
        k = ("PS", tuple(dicts), tuple(views), tuple(tags))
        if k not in self.ZL:
            cols = []
            for tag in tags:
                Pm = np.concatenate([TB[f"{tag}|{d}|per|{mt}"] for d in dicts for mt in MATS])
                cols.append(np.mean([self.F[f"{tag}_{v}"] @ Pm.T for v in views], 0))
            self.ZL[k] = np.concatenate(cols, 1).astype(np.float32)
        return self.ZL[k]

    def ps_head(self, train_objs, te, dicts, views, wd, inner=False, scale=10.0, concat_clip=False):
        X = self.prompt_sims(dicts, views) * scale
        if concat_clip:
            X = np.concatenate([self.X, X], 1)
        if inner:
            P = np.zeros((len(self.mat), 6))
            for g in inner_folds(train_objs, 5, 0):
                a = [q for q in train_objs if q not in set(g)]
                b = np.where(np.isin(self.oid, g) & self.real)[0]
                P[b] = self.ps_head(a, b, dicts, views, wd, scale=scale, concat_clip=concat_clip)
            return P
        tr = np.where(np.isin(self.oid, train_objs) & self.real)[0]
        st = Standardizer(X[tr])
        W, b = fit_logreg(st(X[tr]), self.mat[tr], class_weights(tr, self.oid, self.src, self.mat), wd=wd)
        return predict_logreg(st(X[te]), W, b)


def m_ps(R, train_objs, te, dicts=DICTS_ALL, views=("c14", "c20"), wd_grid=(1e-2, 1e-1, 1.0), with_zs=True, with_clip=False, concat_clip=False,
         skeys=("rich|floor",), zs_views=("c14", "c20")):
    tr = np.where(np.isin(R.oid, train_objs) & R.real)[0]
    Z = softmax(R.L(skeys, zs_views, 0.01) / 0.01)
    best, bs = None, -1
    for wd in wd_grid:
        Pin = R.ps_head(train_objs, None, dicts, views, wd, inner=True, concat_clip=concat_clip)
        if with_clip:
            Pin = softmax((np.log(Pin + 1e-9) + np.log(R.inner_oof(train_objs) + 1e-9)) / 2)
        for w in (W_GRID if with_zs else [0.0]):
            s = M_frame(fuse(Pin, Z, w), R.mat, tr)
            if s > bs:
                best, bs = (wd, w), s
    wd, w = best
    Pte = R.ps_head(train_objs, te, dicts, views, wd, concat_clip=concat_clip)
    if with_clip:
        Pte = softmax((np.log(Pte + 1e-9) + np.log(R.head_full(train_objs, te) + 1e-9)) / 2)
    return fuse(Pte, Z[te], w), {"wd": wd, "w": w}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", default="all")
    a = ap.parse_args()
    R = Runner6()
    res = {}
    print("== Голова в пространстве описаний ==")
    print("  размерность:", R.prompt_sims(DICTS_ALL).shape, R.prompt_sims(DICTS_GEN).shape)
    _, res["ps_all_alone"] = R.run(m_ps, "ps-голова (все словари) без zs", with_zs=False)
    _, res["ps_gen_alone"] = R.run(m_ps, "ps-голова (общие словари) без zs", dicts=DICTS_GEN, with_zs=False)
    _, res["ps_all_zs"] = R.run(m_ps, "ps-голова (все) ⊕ zs lse")
    _, res["ps_gen_zs"] = R.run(m_ps, "ps-голова (общие) ⊕ zs lse", dicts=DICTS_GEN)
    _, res["ps_all_clip_zs"] = R.run(m_ps, "avg[ps-голова (все), clip-голова] ⊕ zs lse", with_clip=True)
    _, res["ps_gen_clip_zs"] = R.run(m_ps, "avg[ps-голова (общие), clip-голова] ⊕ zs lse", dicts=DICTS_GEN, with_clip=True)
    _, res["ps_concat_zs"] = R.run(m_ps, "голова [clip c14 ⊕ ps все] ⊕ zs lse", concat_clip=True)
    _, res["ps_rich_only_zs"] = R.run(m_ps, "ps-голова (только rich) ⊕ zs lse", dicts=["rich|floor"])
    _, res["ps_all_4v_zs"] = R.run(m_ps, "ps-голова (все, 4 вида) ⊕ zs lse", views=tuple(VIEWS))
    json.dump(res, open(HERE / f"exp6_{a.part}.json", "w"), indent=1, default=str)


if __name__ == "__main__":
    main()
