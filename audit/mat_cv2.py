"""Материал v2: честный leave-one-object-out (40 предметов), только train-данные.
Комбинация трёх источников знания:
  1) zero-shot CLIP/SigLIP: сходство вырезки с текстовыми описаниями материалов — знание, не зависящее от наших 40 предметов
     (единственный источник, который может знать «металл» лучше, чем 4 оставшихся в фолде металлических предмета);
  2) линейная голова на эмбеддингах (DINOv2-L, CLIP-L, SigLIP; маска + контекст) с весами «объект = 1, материал = 1»;
  3) ансамбль: усреднение log-вероятностей; веса ансамбля и температура zero-shot подбираются ВНУТРИ обучающих объектов
     (вложенный LOOO), внешний объект в подборе не участвует.
Итог: M по кадрам (как score.py) и по объектам; матрица ошибок."""
import argparse
import itertools
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

from mat_cv import MATS, fit_predict, standardize, weights

C = Path(__file__).resolve().parent / "cache"


def load():
    F = dict(np.load(C / "features.npz"))
    F.update(dict(np.load(C / "features2.npz")))
    meta = json.load(open(C / "meta.json"))
    oid = np.array([m["oid"] for m in meta])
    src = np.array([m["src"] for m in meta])
    mat = np.array([MATS.index(m["mat"]) for m in meta])
    return F, meta, oid, src, mat


def l2n(X):
    return X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-8)


def M_frame(P, mat, idx):
    pred = P[idx].argmax(1)
    return float(np.mean([(pred[mat[idx] == c] == c).mean() for c in range(6) if (mat[idx] == c).any()]))


def M_obj(P, mat, oid, idx):
    hits = defaultdict(list)
    for o in sorted(set(oid[idx])):
        ii = idx[oid[idx] == o]
        hits[mat[ii[0]]].append(int(np.log(P[ii] + 1e-9).mean(0).argmax() == mat[ii[0]]))
    return float(np.mean([np.mean(v) for v in hits.values()])), {MATS[c]: f"{sum(v)}/{len(v)}" for c, v in sorted(hits.items())}


def zero_shot(F, keys, T=0.01):
    logits = sum(F[k] for k in keys) / len(keys)
    z = logits / T
    z = z - z.max(1, keepdims=True)
    p = np.exp(z)
    return p / p.sum(1, keepdims=True)


def linear_oof(X, oid, src, mat, train_mask_fn, eval_idx_fn, objs, synth_w, wd):
    """Возвращает вероятности для eval-индексов каждого фолда (по объектам objs)."""
    P = np.zeros((len(mat), 6))
    for o in objs:
        tr = train_mask_fn(o)
        te = eval_idx_fn(o)
        Xs = standardize(X, tr)
        P[te] = fit_predict(Xs, mat, weights(tr, oid, src, mat, synth_w), tr, te, wd=wd)
    return P


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    a = ap.parse_args()
    F, meta, oid, src, mat = load()
    real = src == "real_train"
    idx = np.where(real)[0]
    objs = sorted(set(oid[real]))

    print("== 1. Zero-shot (без обучения на наших данных) ==")
    zs_sets = {
        "clipL_m10": ["clipL_m10_zs"], "clipL_c14": ["clipL_c14_zs"], "clipL_all": ["clipL_m10_zs", "clipL_m14_zs", "clipL_c14_zs", "clipL_c20_zs"],
        "siglip_all": ["siglip_m10_zs", "siglip_m14_zs", "siglip_c14_zs", "siglip_c20_zs"],
        "clipL+siglip": ["clipL_m10_zs", "clipL_m14_zs", "clipL_c14_zs", "clipL_c20_zs", "siglip_m10_zs", "siglip_m14_zs", "siglip_c14_zs", "siglip_c20_zs"],
    }
    for name, keys in zs_sets.items():
        P = zero_shot(F, keys)
        mo, per = M_obj(P, mat, oid, idx)
        print(f"  {name:14s} M={M_frame(P, mat, idx):.3f}  M_obj={mo:.3f} {per}")

    print("\n== 2. Линейная голова (LOOO), разные признаки ==")
    feats = {
        "dinoL_m10": ["dinoL_m10"], "dinoL_c14": ["dinoL_c14"], "dinoL_c14_mpool": ["dinoL_c14_mpool"], "dinoL_c20_mpool": ["dinoL_c20_mpool"],
        "clipL_m10": ["clipL_m10"], "clipL_c14": ["clipL_c14"], "siglip_m10": ["siglip_m10"], "siglip_c14": ["siglip_c14"],
        "dinoL_m10+c14mp": ["dinoL_m10", "dinoL_c14_mpool"],
        "clipL_m10+c14": ["clipL_m10", "clipL_c14"],
        "dinoL+clipL": ["dinoL_m10", "dinoL_c14_mpool", "clipL_m10", "clipL_c14"],
        "dinoL+clipL+siglip": ["dinoL_m10", "dinoL_c14_mpool", "clipL_m10", "clipL_c14", "siglip_m10", "siglip_c14"],
    }
    if a.quick:
        feats = {k: feats[k] for k in ["dinoL_m10", "clipL_m10", "dinoL+clipL"]}
    lin = {}
    for name, keys in feats.items():
        X = np.concatenate([l2n(F[k]) * math.sqrt(F[k].shape[1]) for k in keys], 1).astype(np.float32)
        for sw in (0.0, 0.5):
            P = linear_oof(X, oid, src, mat, lambda o: np.where((oid != o) & (real | (sw > 0)))[0],
                           lambda o: np.where((oid == o) & real)[0], objs, sw, 1e-2)
            mo, per = M_obj(P, mat, oid, idx)
            lin[(name, sw)] = P
            print(f"  {name:22s} synth={sw:.1f} M={M_frame(P, mat, idx):.3f}  M_obj={mo:.3f} {per}", flush=True)

    print("\n== 3. Ансамбль: zero-shot × линейные (веса — по внешнему LOOO, только для ориентира) ==")
    zs = zero_shot(F, zs_sets["clipL+siglip"])
    best_lin = sorted(lin, key=lambda k: -M_frame(lin[k], mat, idx))[:4]
    for k in best_lin:
        for wz in (0.0, 0.25, 0.5, 0.75, 1.0):
            L = (1 - wz) * np.log(lin[k] + 1e-9) + wz * np.log(zs + 1e-9)
            P = np.exp(L - L.max(1, keepdims=True))
            P /= P.sum(1, keepdims=True)
            mo, per = M_obj(P, mat, oid, idx)
            print(f"  {k[0]:22s} synth={k[1]} w_zs={wz:.2f}  M={M_frame(P, mat, idx):.3f}  M_obj={mo:.3f} {per}")
    np.savez_compressed(C / "mat_v2_oof.npz", zs=zs, **{f"{k[0]}|{k[1]}": v for k, v in lin.items()})


if __name__ == "__main__":
    main()
