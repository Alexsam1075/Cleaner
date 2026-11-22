"""Эксперимент 7: словарь broad (lse), ансамбли lse-наборов, голова + второй член (DINOv2 / SigLIP) с текущим лучшим zs,
усреднение по сетке w, аугментации освещения (mat_aug_feats) с новым zs."""
import argparse
import json
import math

import numpy as np

from common import HERE, CACHE, MATS, MODELS, VIEWS, Standardizer, class_weights, fit_logreg, inner_folds, l2n, predict_logreg, report, softmax, M_frame
from exp1 import W_GRID, fuse
from exp5 import Runner5

ZS = dict(skeys=("rich|floor",), views=("c14", "c20"))


class Runner7(Runner5):
    def __init__(self):
        super().__init__()
        A = dict(np.load(CACHE / "mat_aug_feats.npz"))
        for k, v in A.items():
            self.F[k.replace("clipL_c14_", "clipL_aug_")] = v
        self._vc = {}

    def Xv(self, key):
        if key not in self._vc:
            self._vc[key] = (l2n(self.F[key]) * math.sqrt(768 / 64)).astype(np.float32)
        return self._vc[key]

    def head_keys(self, train_objs, te, train_keys, test_keys, wd=1e-2, inner=False, aug_w=1.0):
        """Голова по ключам признаков F (обучение на объединении train_keys; первый ключ — оригинал, остальные — аугментации с весом aug_w)."""
        if inner:
            P = np.zeros((len(self.mat), 6))
            for g in inner_folds(train_objs, 5, 0):
                a = [q for q in train_objs if q not in set(g)]
                b = np.where(np.isin(self.oid, g) & self.real)[0]
                P[b] = self.head_keys(a, b, train_keys, test_keys, wd, aug_w=aug_w)
            return P
        tr = np.where(np.isin(self.oid, train_objs) & self.real)[0]
        Xtr = np.concatenate([self.Xv(k)[tr] for k in train_keys])
        ytr = np.concatenate([self.mat[tr]] * len(train_keys))
        w0 = class_weights(tr, self.oid, self.src, self.mat)
        w = np.concatenate([w0] + [w0 * aug_w] * (len(train_keys) - 1))
        st = Standardizer(Xtr)
        W, b = fit_logreg(st(Xtr), ytr, w, wd=wd)
        return softmax(np.mean([np.log(predict_logreg(st(self.Xv(k)[te]), W, b) + 1e-9) for k in test_keys], 0))


def m7(R, train_objs, te, heads=((("clipL_c14",), ("clipL_c14",)),), skeys=("rich|floor",), zs_views=("c14", "c20"), tau=0.01, w_mode="argmax", aug_w=1.0, wd=1e-2):
    tr = np.where(np.isin(R.oid, train_objs) & R.real)[0]
    Z = softmax(R.L(skeys, zs_views, tau) / 0.01)
    Pin = softmax(np.mean([np.log(R.head_keys(train_objs, None, h[0], h[1], wd, inner=True, aug_w=aug_w) + 1e-9) for h in heads], 0))
    Pte = softmax(np.mean([np.log(R.head_keys(train_objs, te, h[0], h[1], wd, aug_w=aug_w) + 1e-9) for h in heads], 0))
    if w_mode == "argmax":
        w = max(W_GRID, key=lambda w: M_frame(fuse(Pin, Z, w), R.mat, tr))
        return fuse(Pte, Z[te], w), {"w": w}
    scores = np.array([M_frame(fuse(Pin, Z, w), R.mat, tr) for w in W_GRID])
    keep = [w for w, s in zip(W_GRID, scores) if s >= scores.max() - 0.01]
    return np.mean([fuse(Pte, Z[te], w) for w in keep], 0), {"w": tuple(keep)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", default="all")
    a = ap.parse_args()
    R = Runner7()
    res = {}
    c14 = (("clipL_c14",), ("clipL_c14",))
    print("== 1. Наборы описаний (lse, c14+c20) ==")
    _, res["rich"] = R.run(m7, "rich lse (текущий лучший)")
    _, res["broad"] = R.run(m7, "broad lse", skeys=("broad|floor",))
    _, res["broad_plain"] = R.run(m7, "broad|plain lse", skeys=("broad|plain",))
    _, res["rich_broad"] = R.run(m7, "rich + broad lse", skeys=("rich|floor", "broad|floor"))
    _, res["rich_fl"] = R.run(m7, "rich floor+light lse", skeys=("rich|floor", "rich|light"))
    _, res["rich_avgw"] = R.run(m7, "rich lse, w усреднён по ~лучшим", w_mode="avg")
    print("\n== 2. Вторые головы ==")
    _, res["clip_dino"] = R.run(m7, "avg[clip c14, dinoL c14] ⊕ zs", heads=(c14, (("dinoL_c14",), ("dinoL_c14",))))
    _, res["clip_dino_m10"] = R.run(m7, "avg[clip c14, dinoL m10] ⊕ zs", heads=(c14, (("dinoL_m10",), ("dinoL_m10",))))
    _, res["clip_dino_mpool"] = R.run(m7, "avg[clip c14, dinoL c14_mpool] ⊕ zs", heads=(c14, (("dinoL_c14_mpool",), ("dinoL_c14_mpool",))))
    _, res["clip_siglip"] = R.run(m7, "avg[clip c14, siglip c14] ⊕ zs", heads=(c14, (("siglip_c14",), ("siglip_c14",))))
    _, res["clip_clipm10"] = R.run(m7, "avg[clip c14, clip m10] ⊕ zs", heads=(c14, (("clipL_m10",), ("clipL_m10",))))
    print("\n== 3. Аугментации освещения головы с новым zs ==")
    augs = ["clipL_aug_dark", "clipL_aug_bright", "clipL_aug_side_l", "clipL_aug_side_r", "clipL_aug_gray"]
    _, res["aug_light"] = R.run(m7, "голова + aug(light,gray) w=0.5 ⊕ zs", heads=((("clipL_c14",) + tuple(augs), ("clipL_c14",)),), aug_w=0.5)
    _, res["aug_light_tta"] = R.run(m7, "голова + aug → TTA(gray) ⊕ zs", heads=((("clipL_c14",) + tuple(augs), ("clipL_c14", "clipL_aug_gray")),), aug_w=0.5)
    _, res["wd3e-2"] = R.run(m7, "rich lse, wd=3e-2", wd=3e-2)
    _, res["wd3e-3"] = R.run(m7, "rich lse, wd=3e-3", wd=3e-3)
    json.dump(res, open(HERE / f"exp7_{a.part}.json", "w"), indent=1, default=str)


if __name__ == "__main__":
    main()
