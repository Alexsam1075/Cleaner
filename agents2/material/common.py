"""Общий код агента material: загрузка кэша признаков, метрики, GPU-логрегрессия, вложенный LOOO."""
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
CACHE = HERE / "shared_cache"
MATS = ["metal", "wood", "plastic", "glass", "paper", "mineral"]
MODELS = ["clipL", "siglip"]
VIEWS = ["m10", "m14", "c14", "c20"]
LIGHTS = ["bright", "dim", "side"]
import os
DEV = "cuda" if (torch.cuda.is_available() and os.environ.get("MAT_GPU", "0") == "1") else "cpu"
torch.set_num_threads(int(os.environ.get("MAT_THREADS", "8")))


def softmax(z):
    z = z - z.max(-1, keepdims=True)
    p = np.exp(z)
    return p / p.sum(-1, keepdims=True)


def l2n(X):
    return X / (np.linalg.norm(X, axis=-1, keepdims=True) + 1e-8)


def load_bank():
    F = dict(np.load(CACHE / "features.npz"))
    F.update(dict(np.load(CACHE / "features2.npz")))
    meta = json.load(open(CACHE / "meta.json"))
    oid = np.array([m["oid"] for m in meta])
    src = np.array([m["src"] for m in meta])
    mat = np.array([MATS.index(m["mat"]) for m in meta])
    light = np.array([m.get("lighting") or "" for m in meta])
    return F, meta, oid, src, mat, light


def M_frame(P, mat, idx):
    pred = P[idx].argmax(1)
    return float(np.mean([(pred[mat[idx] == c] == c).mean() for c in range(6) if (mat[idx] == c).any()]))


def per_class(P, mat, idx):
    pred = P[idx].argmax(1)
    return {MATS[c]: f"{int((pred[mat[idx] == c] == c).sum())}/{int((mat[idx] == c).sum())}" for c in range(6) if (mat[idx] == c).any()}


def M_obj(P, mat, oid, idx):
    hits = defaultdict(list)
    wrong = {}
    for o in sorted(set(oid[idx])):
        ii = idx[oid[idx] == o]
        p = int(np.log(P[ii] + 1e-9).mean(0).argmax())
        hits[mat[ii[0]]].append(int(p == mat[ii[0]]))
        if p != mat[ii[0]]:
            wrong[o] = MATS[p]
    return float(np.mean([np.mean(v) for v in hits.values()])), {MATS[c]: f"{sum(v)}/{len(v)}" for c, v in sorted(hits.items())}, wrong


def by_light(P, mat, idx, light):
    return {lg: round(M_frame(P, mat, idx[light[idx] == lg]), 3) for lg in LIGHTS}


def report(name, P, mat, oid, idx, light, show_wrong=True):
    mf = M_frame(P, mat, idx)
    mo, per_o, wrong = M_obj(P, mat, oid, idx)
    bl = by_light(P, mat, idx, light)
    s = f"  {name:44s} M={mf:.3f} M_obj={mo:.3f} {per_class(P, mat, idx)} | свет {bl}"
    if show_wrong:
        s += f" | ошибки по предметам {wrong}"
    print(s, flush=True)
    return {"M": mf, "M_obj": mo, "per_class": per_class(P, mat, idx), "per_obj": per_o, "light": bl, "wrong": wrong}


def class_weights(idx, oid, src, mat, synth_w=0.0):
    """Каждый физический предмет весит одинаково, каждый материал весит одинаково."""
    cnt = defaultdict(int)
    for i in idx:
        cnt[(oid[i], src[i])] += 1
    w = np.array([(1.0 if src[i] == "real_train" else synth_w) / cnt[(oid[i], src[i])] for i in idx])
    per = defaultdict(float)
    for j, i in enumerate(idx):
        per[mat[i]] += w[j]
    w = np.array([w[j] / per[mat[i]] for j, i in enumerate(idx)])
    return w / w.mean()


def fit_logreg(X, y, w, wd=1e-2, epochs=300, W0=None, anchor=0.0, n_cls=6):
    """Многоклассовая логрегрессия (LBFGS, GPU). Если задан W0 (якорь): штраф anchor*||W-W0||² + wd*||W||²."""
    Xt = torch.tensor(X, device=DEV, dtype=torch.float32)
    yt = torch.tensor(y, device=DEV)
    wt = torch.tensor(w, device=DEV, dtype=torch.float32)
    W = torch.zeros(X.shape[1], n_cls, device=DEV)
    if W0 is not None:
        W = torch.tensor(W0, device=DEV, dtype=torch.float32).clone()
        W0t = W.clone()
    W.requires_grad_(True)
    b = torch.zeros(n_cls, device=DEV, requires_grad=True)
    opt = torch.optim.LBFGS([W, b], lr=1, max_iter=epochs, history_size=20, line_search_fn="strong_wolfe")

    def closure():
        opt.zero_grad()
        loss = (torch.nn.functional.cross_entropy(Xt @ W + b, yt, reduction="none") * wt).sum() / wt.sum() + wd * (W ** 2).sum()
        if W0 is not None and anchor > 0:
            loss = loss + anchor * ((W - W0t) ** 2).sum()
        loss.backward()
        return loss

    opt.step(closure)
    return W.detach().cpu().numpy(), b.detach().cpu().numpy()


def predict_logreg(X, W, b):
    return softmax(X @ W + b)


class Standardizer:
    def __init__(self, X):
        self.mu, self.sd = X.mean(0), X.std(0) + 1e-6

    def __call__(self, X):
        return (X - self.mu) / self.sd


def inner_folds(train_objs, k=5, seed=0):
    rs = np.random.default_rng(seed)
    return np.array_split(rs.permutation(sorted(train_objs)), k)


def fold_sets(oid, src, test_obj, train_objs=None):
    """Индексы обучения (только real-кадры train-объектов) и теста (real-кадры отложенного предмета)."""
    real = src == "real_train"
    if train_objs is None:
        tr = np.where((oid != test_obj) & real)[0]
    else:
        tr = np.where(np.isin(oid, list(train_objs)) & real)[0]
    te = np.where((oid == test_obj) & real)[0]
    return tr, te
