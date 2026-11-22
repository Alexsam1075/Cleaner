"""Shared evaluation helpers for the material classifier (agent material).
Leave-one-object-out over 40 physical objects; synthetic frames of the held-out object are excluded;
scoring only on real frames. M = macro over materials of per-frame accuracy."""
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
MATS = ["metal", "wood", "plastic", "glass", "paper", "mineral"]
DEV = "cuda" if torch.cuda.is_available() else "cpu"


def load_meta():
    meta = json.load(open(HERE / "shared_cache" / "meta.json"))
    oid = np.array([m["oid"] for m in meta])
    src = np.array([m["src"] for m in meta])
    mat = np.array([MATS.index(m["mat"]) for m in meta])
    light = np.array([str(m.get("lighting")) for m in meta])
    return meta, oid, src, mat, light


def load_feats():
    F = dict(np.load(HERE / "cache" / "feat3.npz"))
    for name in ("features2.npz", "features_ct.npz", "features.npz"):
        for k, v in np.load(HERE / "shared_cache" / name).items():
            F.setdefault("old_" + k, v)
    return F


def l2n(X):
    return X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-8)


def softmax(z):
    z = z - z.max(1, keepdims=True)
    p = np.exp(z)
    return p / p.sum(1, keepdims=True)


def M_frame(P, mat, idx):
    pred = P[idx].argmax(1)
    return float(np.mean([(pred[mat[idx] == c] == c).mean() for c in range(6) if (mat[idx] == c).any()]))


def per_mat(P, mat, idx):
    pred = P[idx].argmax(1)
    return {MATS[c]: f"{(pred[mat[idx] == c] == c).sum()}/{(mat[idx] == c).sum()}" for c in range(6) if (mat[idx] == c).any()}


def M_obj(P, mat, oid, idx):
    hits = defaultdict(list)
    for o in sorted(set(oid[idx])):
        ii = idx[oid[idx] == o]
        hits[mat[ii[0]]].append(int(np.log(P[ii] + 1e-9).mean(0).argmax() == mat[ii[0]]))
    return float(np.mean([np.mean(v) for v in hits.values()])), {MATS[c]: f"{sum(v)}/{len(v)}" for c, v in sorted(hits.items())}


def weights(idx, oid, src, mat, synth_w=0.0):
    cnt = defaultdict(int)
    for i in idx:
        cnt[(oid[i], src[i])] += 1
    w = np.array([(1.0 if src[i] == "real_train" else synth_w) / cnt[(oid[i], src[i])] for i in idx])
    per = defaultdict(float)
    for j, i in enumerate(idx):
        per[mat[i]] += w[j]
    w = np.array([w[j] / per[mat[i]] for j, i in enumerate(idx)])
    return w / w.mean()


def fit_logreg(Xtr, ytr, w, wd=1e-2, iters=300, W0=None, anchor=None, anchor_wd=0.0, n_cls=6):
    """Multinomial logreg (LBFGS, GPU). Optional text anchor: penalty anchor_wd*||W - anchor||^2."""
    Xt = torch.tensor(Xtr, device=DEV, dtype=torch.float32)
    yt = torch.tensor(ytr, device=DEV)
    wt = torch.tensor(w, device=DEV, dtype=torch.float32)
    W = torch.zeros(Xtr.shape[1], n_cls, device=DEV) if W0 is None else torch.tensor(W0, device=DEV, dtype=torch.float32).clone()
    W.requires_grad_(True)
    b = torch.zeros(n_cls, device=DEV, requires_grad=True)
    A = None if anchor is None else torch.tensor(anchor, device=DEV, dtype=torch.float32)
    opt = torch.optim.LBFGS([W, b], lr=1, max_iter=iters, history_size=20, line_search_fn="strong_wolfe")

    def closure():
        opt.zero_grad()
        loss = (torch.nn.functional.cross_entropy(Xt @ W + b, yt, reduction="none") * wt).sum() / wt.sum() + wd * (W ** 2).sum()
        if A is not None:
            loss = loss + anchor_wd * ((W - A) ** 2).sum()
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


def emb_block(X):
    return (l2n(X) * math.sqrt(X.shape[1] / 64)).astype(np.float32)


def looo_linear(X, y, oid, src, real, objs, wd=1e-2, synth_w=0.0, restrict_objs=None, **kw):
    """Standard LOOO for a linear head; returns OOF probs for real frames of each object."""
    P = np.zeros((len(y), 6))
    pool = np.ones(len(y), bool) if restrict_objs is None else np.isin(oid, restrict_objs)
    for o in objs:
        tr = np.where(pool & (oid != o) & (real | (synth_w > 0)))[0]
        te = np.where(pool & (oid == o) & real)[0]
        if len(te) == 0:
            continue
        st = Standardizer(X[tr])
        W, b = fit_logreg(st(X[tr]), y[tr], weights(tr, oid, src, y, synth_w), wd=wd, **kw)
        P[te] = predict_logreg(st(X[te]), W, b)
    return P


def inner_folds(train_objs, k=5, seed=0):
    rs = np.random.default_rng(seed)
    return np.array_split(rs.permutation(train_objs), k)


def bootstrap_ci(P, mat, oid, idx, objs, n=1000, seed=0):
    rs = np.random.default_rng(seed)
    bs = []
    for _ in range(n):
        pick = rs.choice(objs, len(objs), replace=True)
        ii = np.concatenate([idx[oid[idx] == o] for o in pick])
        bs.append(M_frame(P, mat, ii))
    return [float(x) for x in np.percentile(bs, [2.5, 97.5])]


def confusion(P, mat, idx):
    C = np.zeros((6, 6), int)
    for t, p in zip(mat[idx], P[idx].argmax(1)):
        C[t, p] += 1
    return C
