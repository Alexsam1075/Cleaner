"""Материал: устойчивость к освещению через аугментацию при обучении головы.
Для каждой train-вырезки (c14) делаются версии: тёмная (гамма 1.8), светлая (гамма 0.55), боковой градиент (слева светло,
справа темно и наоборот), обесцвеченная (серая), тёплый/холодный сдвиг. Их CLIP-L эмбеддинги — дополнительные обучающие
примеры с тем же материалом (вес 0.5). На тесте: усреднение вероятностей головы по оригиналу + серой версии (TTA).
LOOO по 40 предметам: все копии отложенного предмета исключены. Сравнение: база / аугментация / аугментация + TTA,
по освещению bright/dim/side и по классам. Только train."""
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import torch

ROOT = Path("/home/cleaner/Cleaner")
sys.path.insert(0, str(ROOT / "audit"))
import full_pipeline as fp  # noqa: E402
from full_pipeline import D, MATS, MODELS, VIEWS  # noqa: E402
from mat_cv import fit_predict  # noqa: E402
from mat_features2 import crop  # noqa: E402
from attr_features import polygon_mask, records  # noqa: E402
from mat_v4 import M_frame, M_obj, l2n  # noqa: E402

C = ROOT / "audit/cache"
AUGS = ["dark", "bright", "side_l", "side_r", "gray", "warm", "cool"]


def augment(img, kind):
    f = img.astype(np.float32) / 255
    if kind == "dark":
        f = f ** 1.8
    elif kind == "bright":
        f = f ** 0.55
    elif kind in ("side_l", "side_r"):
        g = np.linspace(1.35, 0.55, img.shape[1], dtype=np.float32)
        if kind == "side_r":
            g = g[::-1]
        f = f * g[None, :, None]
    elif kind == "gray":
        gr = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255
        f = np.repeat(gr[..., None], 3, 2)
    elif kind == "warm":
        f = f * np.array([1.12, 1.0, 0.85], np.float32)
    elif kind == "cool":
        f = f * np.array([0.85, 1.0, 1.12], np.float32)
    return np.clip(f * 255, 0, 255).astype(np.uint8)


@torch.no_grad()
def build(enc):
    p = C / "mat_aug_feats.npz"
    if p.exists():
        return dict(np.load(p))
    recs = records()
    crops = {k: [] for k in AUGS}
    cache = {}
    for i, (src, im, a) in enumerate(recs):
        fd = D / Path(im["file_name"]).parent
        if fd not in cache:
            cache.clear()
            cache[fd] = cv2.cvtColor(cv2.imread(str(fd / "color.png")), cv2.COLOR_BGR2RGB)
        rgb = cache[fd]
        m = polygon_mask(a["segmentation"], im["height"], im["width"])
        c14, _ = crop(rgb, m, a["bbox"], 1.5, False)
        for k in AUGS:
            crops[k].append(augment(c14, k))
        if i % 200 == 0:
            print("crops", i, len(recs), flush=True)
    out = {}
    for k in AUGS:
        out[f"clipL_c14_{k}"] = enc.embed("clipL", np.stack(crops[k]))
        print("emb", k, flush=True)
    np.savez_compressed(p, **out)
    return out


def weights_rows(idx, oid, src, mat, aug_w):
    cnt = defaultdict(int)
    for i in idx:
        cnt[(oid[i], src[i])] += 1
    w = np.array([(aug_w if src[i].startswith("aug") else 1.0) / cnt[(oid[i], src[i])] for i in idx])
    per = defaultdict(float)
    for j, i in enumerate(idx):
        per[mat[i]] += w[j]
    w = np.array([w[j] / per[mat[i]] for j, i in enumerate(idx)])
    return w / w.mean()


def main():
    enc = fp.Encoders()
    F = dict(np.load(C / "features.npz"))
    F.update(dict(np.load(C / "features2.npz")))
    A = build(enc)
    meta = json.load(open(C / "meta.json"))
    n = len(meta)
    oid0 = np.array([m["oid"] for m in meta])
    src0 = np.array([m["src"] for m in meta])
    mat0 = np.array([MATS.index(m["mat"]) for m in meta])
    light = np.array([m.get("lighting") or "" for m in meta])
    real = src0 == "real_train"
    idx = np.where(real)[0]
    objs = sorted(set(oid0[real]))
    T = {t: enc.text[(t, "rich")] for t in MODELS}
    Z = fp.softmax(np.mean([F[f"{t}_{v}"] @ T[t].T for t in MODELS for v in VIEWS], 0) / 0.01)
    X0 = (l2n(F["clipL_c14"]) * math.sqrt(768 / 64)).astype(np.float32)
    from overfit_check import inner_weight

    def report(name, P):
        mf = M_frame(P, mat0, idx)
        mo, per = M_obj(P, mat0, oid0, idx)
        by_light = {lg: round(M_frame(P, mat0, idx[light[idx] == lg]), 3) for lg in ("bright", "dim", "side")}
        key = {o: MATS[int(np.log(P[idx[oid0[idx] == o]] + 1e-9).mean(0).argmax())] for o in ("s11", "s18", "s29", "s04", "s36", "s38", "s05", "b13", "b06", "s40", "b16")}
        print(f"  {name:40s} M={mf:.3f} M_obj={mo:.3f} {per} | свет {by_light} | {key}", flush=True)

    def looo(aug_sets, aug_w=0.5, tta=()):
        # собрать расширенную таблицу: оригиналы (все meta) + аугментации только для real train
        Xs, oids, srcs, mats = [X0], [oid0], [src0], [mat0]
        for k in aug_sets:
            Xa = (l2n(A[f"clipL_c14_{k}"][idx]) * math.sqrt(768 / 64)).astype(np.float32)
            Xs.append(Xa)
            oids.append(oid0[idx])
            srcs.append(np.array([f"aug_{k}"] * len(idx)))
            mats.append(mat0[idx])
        X = np.concatenate(Xs)
        oid = np.concatenate(oids)
        src = np.concatenate(srcs)
        mat = np.concatenate(mats)
        trainable = (src == "real_train") | np.char.startswith(src.astype(str), "aug")
        P = np.zeros((n, 6))
        for o in objs:
            tr = np.where(trainable & (oid != o))[0]
            te = np.where((oid0 == o) & real)[0]
            mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-6
            Xn = (X - mu) / sd
            w = weights_rows(tr, oid, src, mat, aug_w)
            Ps = [fit_predict(Xn, mat, w, tr, te)]
            for k in tta:
                Xt = (l2n(A[f"clipL_c14_{k}"][te]) * math.sqrt(768 / 64)).astype(np.float32)
                Zt = np.concatenate([Xn, (Xt - mu) / sd])
                yy = np.concatenate([mat, np.zeros(len(te), int)])
                Ps.append(fit_predict(Zt, yy, w, tr, np.arange(len(X), len(X) + len(te))))
            lp = np.mean([np.log(p_ + 1e-9) for p_ in Ps], 0)
            P[te] = np.exp(lp - lp.max(1, keepdims=True))
            P[te] /= P[te].sum(1, keepdims=True)
        return P

    def fuse(P_lin):
        Pf = np.zeros_like(P_lin)
        ws = []
        for o in objs:
            w = inner_weight(X0, oid0, src0, mat0, [q for q in objs if q != o], Z, seed=len(ws))
            ws.append(w)
            te = np.where((oid0 == o) & real)[0]
            Pf[te] = fp.softmax((1 - w) * np.log(P_lin[te] + 1e-9) + w * np.log(Z[te] + 1e-9))
        return Pf

    print("== Аугментация освещения при обучении головы, LOOO по 40 предметам, rich ==")
    report("база: голова", looo([]))
    report("база: голова ⊕ zs", fuse(looo([])))
    for name, sets in (("light (dark,bright,side)", ["dark", "bright", "side_l", "side_r"]), ("gray", ["gray"]),
                       ("light+gray", ["dark", "bright", "side_l", "side_r", "gray"]), ("all 7", AUGS)):
        for aw in (0.3, 0.7):
            P = looo(sets, aw)
            report(f"aug {name} w={aw}: голова", P)
            report(f"aug {name} w={aw}: голова ⊕ zs", fuse(P))
    P = looo(AUGS, 0.5, tta=("gray",))
    report("aug all w=.5 + TTA(gray): голова ⊕ zs", fuse(P))
    P = looo(AUGS, 0.5, tta=("gray", "dark", "bright"))
    report("aug all w=.5 + TTA(gray,dark,bright): ⊕ zs", fuse(P))


if __name__ == "__main__":
    main()
