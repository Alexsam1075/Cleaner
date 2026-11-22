"""Эксперимент 5: детальный анализ лучшей конфигурации (голова c14 ⊕ zs rich lse c14+c20), вариации lse-температуры,
комбинация lse-наборов (rich + общие), ансамбль нескольких шаблонов, стабильность выбора."""
import argparse
import json

import numpy as np

from common import HERE, MATS, MODELS, VIEWS, report, softmax, M_frame
from exp1 import Runner, SET_KEYS, TB, W_GRID, fuse

GEN = [s for s in SET_KEYS if not s.startswith("rich") and not s.startswith("generic_syn")]


def lse_logits(F, skey, tags, views, tau=0.01):
    out = []
    for tag in tags:
        for v in views:
            E = F[f"{tag}_{v}"]
            cols = []
            for mt in MATS:
                Pm = TB[f"{tag}|{skey}|per|{mt}"]
                s = E @ Pm.T / tau
                m = s.max(1, keepdims=True)
                cols.append((m[:, 0] + np.log(np.exp(s - m).mean(1))) * tau)
            out.append(np.stack(cols, 1))
    return np.mean(out, 0)


class Runner5(Runner):
    def L(self, skeys, views, tau=0.01, tags=MODELS, mode="lse"):
        k = ("L", tuple(skeys), tuple(views), tau, tuple(tags), mode)
        if k not in self.ZL:
            if mode == "lse":
                self.ZL[k] = np.mean([lse_logits(self.F, s, tags, views, tau) for s in skeys], 0)
            elif mode == "max":
                self.ZL[k] = np.mean([lse_logits(self.F, s, tags, views, 1e-4) for s in skeys], 0)
            else:
                self.ZL[k] = np.mean([self.logits(s, "mean", list(views), list(tags)) for s in skeys], 0)
        return self.ZL[k]


def m_lse(R, train_objs, te, skeys=("rich|floor",), views=("c14", "c20"), tau=0.01, temp=0.01, tags=MODELS, mode="lse", w_grid=W_GRID, sel_views=None):
    tr = np.where(np.isin(R.oid, train_objs) & R.real)[0]
    Pin = R.inner_oof(train_objs)
    if sel_views:
        best, bs = None, -1
        for vs in sel_views:
            Z = softmax(R.L(skeys, vs, tau, tags, mode) / temp)
            for w in w_grid:
                s = M_frame(fuse(Pin, Z, w), R.mat, tr)
                if s > bs:
                    best, bs = (vs, w), s
        views, w = best
    Z = softmax(R.L(skeys, views, tau, tags, mode) / temp)
    if not sel_views:
        w = max(w_grid, key=lambda w: M_frame(fuse(Pin, Z, w), R.mat, tr))
    Pte = R.head_full(train_objs, te)
    return fuse(Pte, Z[te], w), {"w": w, "views": views}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", default="all")
    a = ap.parse_args()
    R = Runner5()
    res = {}
    print("== 1. lse: температура внутри класса, виды c14+c20 ==")
    for tau in (0.005, 0.01, 0.02, 0.05):
        _, res[f"lse_tau{tau}"] = R.run(m_lse, f"rich lse tau={tau}, c14+c20", tau=tau)
    _, res["max"] = R.run(m_lse, "rich max по описаниям, c14+c20", mode="max")
    _, res["mean"] = R.run(m_lse, "rich mean, c14+c20", mode="mean")
    print("\n== 2. lse по другим шаблонам / наборам ==")
    for s in ("rich|plain", "rich|robot", "rich|light"):
        _, res[f"lse_{s}"] = R.run(m_lse, f"{s} lse, c14+c20", skeys=(s,))
    _, res["lse_rich_all_t"] = R.run(m_lse, "rich все 4 шаблона lse, c14+c20", skeys=("rich|floor", "rich|plain", "rich|robot", "rich|light"))
    _, res["lse_rich_fp"] = R.run(m_lse, "rich floor+plain lse, c14+c20", skeys=("rich|floor", "rich|plain"))
    _, res["lse_rich_waste"] = R.run(m_lse, "rich + waste lse, c14+c20", skeys=("rich|floor", "waste|floor"))
    _, res["lse_rich_names"] = R.run(m_lse, "rich + names lse, c14+c20", skeys=("rich|floor", "names|floor"))
    _, res["lse_rich_surface"] = R.run(m_lse, "rich + surface lse, c14+c20", skeys=("rich|floor", "surface|floor"))
    _, res["lse_waste"] = R.run(m_lse, "waste lse, c14+c20", skeys=("waste|floor",))
    _, res["lse_gen5"] = R.run(m_lse, "names+waste+surface+shape_free (floor) lse", skeys=("names|floor", "waste|floor", "surface|floor", "shape_free|floor"))
    print("\n== 3. lse: модели по отдельности, виды ==")
    _, res["lse_clip"] = R.run(m_lse, "rich lse, только CLIP-L, c14+c20", tags=("clipL",))
    _, res["lse_siglip"] = R.run(m_lse, "rich lse, только SigLIP, c14+c20", tags=("siglip",))
    _, res["lse_c20"] = R.run(m_lse, "rich lse, c20", views=("c20",))
    _, res["lse_c14c20_fine"] = R.run(m_lse, "rich lse c14+c20, мелкая сетка w", w_grid=[0.3, 0.4, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.9])
    _, res["lse_selviews"] = R.run(m_lse, "rich lse, виды вложенно {c14, c20, c14+c20, 4}", sel_views=[("c14",), ("c20",), ("c14", "c20"), tuple(VIEWS)])
    json.dump(res, open(HERE / f"exp5_{a.part}.json", "w"), indent=1, default=str)


if __name__ == "__main__":
    main()
