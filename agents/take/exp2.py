"""Experiment 2: cost-sensitive classifiers / mass regressors on [nested calibrated volume, class,
zero-shot heaviness sims, embedding PCA], fast thanks to the volume cache. Nested LOOO:
outer object o held out; for each outer fold the feature matrix of all other objects uses volumes
calibrated without {o, q}; inner LOOO over the 39 training objects picks the threshold."""
from __future__ import annotations

import json
import pickle
import sys
import time

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

sys.path.insert(0, "/home/cleaner/Cleaner/agents/take")
from take_data import HEAVY, MATS, FORMS, CACHE, SHARED, build_table, load_feats
from take_lib import LOG500, balanced_acc, choose_threshold, fmt, object_weights, ridge_fit, ridge_predict, summarize

df = build_table()
F = load_feats()
real = df[df.real].reset_index(drop=True)
VC = pickle.load(open(CACHE / "vol_cache.pkl", "rb"))
V = VC["V"]
real["mat_noisy"] = VC["mat_noisy"]
OBJS = VC["objs"]
OID = real.oid.values
HEAVY_Y = real.heavy.values
LOGMASS = np.log(real.mass.values)
ZS_COLS = [c for c in real.columns if c.startswith("zs_")]
FUSED = np.load(SHARED / "fused_generic_syn.npy")
ROWS = {o: np.where(OID == o)[0] for o in OBJS}


def vol_matrix(outer: str, noisy: bool) -> np.ndarray:
    v = np.zeros(len(real))
    for q in OBJS:
        if q == outer and not noisy:
            v[ROWS[q]] = V[(outer, q)]["true"]
        else:
            v[ROWS[q]] = V[(outer, q)]["noisy" if noisy else "true"]
    return v


class EmbPCA:
    def __init__(self, keys, n):
        self.keys, self.n = keys, n

    def fit(self, idx):
        X = np.concatenate([F[k][idx] for k in self.keys], 1)
        self.mu = X.mean(0)
        U, S, Vt = np.linalg.svd(X - self.mu, full_matrices=False)
        self.V = Vt[: self.n].T
        self.scale = S[: self.n] / np.sqrt(len(idx))
        return self

    def transform(self, idx):
        X = np.concatenate([F[k][idx] for k in self.keys], 1)
        return ((X - self.mu) @ self.V) / self.scale


def onehot(d: pd.DataFrame, mat_col):
    return np.concatenate([np.array([(d.form == f).values for f in FORMS], float).T,
                           np.array([(d[mat_col] == m).values for m in MATS], float).T], 1)


def features(vol: np.ndarray, cfg: dict, mat_col: str, pca=None) -> np.ndarray:
    d = real
    cols = [np.log(vol)[:, None]]
    if cfg.get("geo"):
        cols.append(np.log(np.maximum(d[["parts_ml", "g_vol", "footprint_ml"]].values.astype(float), 1.0)))
        cols.append(d[["p95_h_m", "g_h95", "g_ext1", "g_ext2", "g_valid", "g_holes_inner"]].values.astype(float))
    if cfg.get("cls", True):
        cols.append(onehot(d, mat_col))
    if cfg.get("matprobs"):
        cols.append(FUSED[d.idx.values])
    if cfg.get("zs"):
        z = d[cfg["zs"] if isinstance(cfg["zs"], list) else ZS_COLS].values * 20.0
        cols.append(z)
    if cfg.get("vol_x_mat"):
        cols.append(np.log(vol)[:, None] * onehot(d, mat_col)[:, 2:])
    if pca is not None:
        cols.append(pca.transform(d.idx.values))
    return np.concatenate(cols, 1)


def fit_predict(X, tr_mask, te_mask, cfg):
    Xtr, Xte = X[tr_mask], X[te_mask]
    mu, sd = Xtr.mean(0), Xtr.std(0)
    sd[sd < 1e-9] = 1.0
    Xtr, Xte = (Xtr - mu) / sd, (Xte - mu) / sd
    w = object_weights(OID[tr_mask])
    yh = HEAVY_Y[tr_mask]
    w[yh == 1] *= 0.5 / w[yh == 1].sum()
    w[yh == 0] *= 0.5 / w[yh == 0].sum()
    if cfg.get("model", "logreg") == "logreg":
        m = LogisticRegression(C=cfg.get("C", 1.0), max_iter=5000)
        m.fit(Xtr, yh, sample_weight=w * len(w))
        return m.decision_function(Xte)
    m = ridge_fit(Xtr, LOGMASS[tr_mask], w, cfg.get("alpha", 10.0))
    return ridge_predict(m, Xte)


def run(cfg, noisy_train=False, noisy_test=False, inner=True, verbose=False, labels=None):
    global HEAVY_Y
    y = HEAVY_Y if labels is None else labels
    saved = HEAVY_Y
    HEAVY_Y = y
    score = np.zeros(len(real))
    pred = np.zeros(len(real), bool)
    ths = []
    default = LOG500 if cfg.get("model") == "ridge" else 0.0
    for o in OBJS:
        te = OID == o
        tr = ~te
        pca = EmbPCA(cfg["pca"], cfg.get("npca", 8)).fit(np.where(tr)[0]) if cfg.get("pca") else None
        Xtr_full = features(vol_matrix(o, noisy_train), cfg, "mat_noisy" if noisy_train else "mat", pca)
        if noisy_test != noisy_train:
            Xte_full = features(vol_matrix(o, noisy_test), cfg, "mat_noisy" if noisy_test else "mat", pca)
        else:
            Xte_full = Xtr_full
        X = Xtr_full.copy()
        X[te] = Xte_full[te]
        s = fit_predict(X, tr, te, cfg)
        score[te] = s
        if inner:
            oof = np.zeros(len(real))
            for q in OBJS:
                if q == o:
                    continue
                ite = OID == q
                oof[ite] = fit_predict(Xtr_full, tr & ~ite, ite, cfg)
            t = choose_threshold(oof[tr], y[tr], object_weights(OID[tr]), default=default)
        else:
            t = default
        ths.append(t)
        pred[te] = s > t
        if verbose:
            print(o, y[te][0], f"t={t:.2f}", np.round(s, 2))
    HEAVY_Y = saved
    return score, pred, ths


def permutation(cfg, n=5, seed=0):
    rng = np.random.default_rng(seed)
    mass = real.groupby("oid").mass.first()
    out = []
    for _ in range(n):
        perm = dict(zip(OBJS, rng.permutation([mass[o] for o in OBJS])))
        y = np.array([int(perm[o] > 500) for o in OID])
        _, p, _ = run(cfg, labels=y)
        out.append(balanced_acc(y, p))
    return out


CFGS = {
    "L0 logreg vol+cls": {"zs": False},
    "L1 logreg vol+cls+zs": {"zs": True},
    "L2 logreg vol+cls+zs C=0.3": {"zs": True, "C": 0.3},
    "L3 logreg vol+cls+zs C=3": {"zs": True, "C": 3.0},
    "L4 logreg vol+geo+cls+zs": {"zs": True, "geo": True},
    "L5 logreg vol+cls+zs+clipPCA8": {"zs": True, "pca": ["clipL_c14"], "npca": 8},
    "L6 logreg vol+cls+zs+dinoPCA8": {"zs": True, "pca": ["dinoL_m10"], "npca": 8},
    "L7 logreg vol+matprobs+zs": {"zs": True, "cls": False, "matprobs": True},
    "L8 logreg vol+cls+zs+volxmat": {"zs": True, "vol_x_mat": True},
    "L9 logreg vol+cls+clipPCA8": {"zs": False, "pca": ["clipL_c14"], "npca": 8},
    "R0 ridge vol+cls": {"model": "ridge", "zs": False, "alpha": 3.0},
    "R1 ridge vol+cls+zs a=10": {"model": "ridge", "zs": True, "alpha": 10.0},
    "R2 ridge vol+cls+zs a=30": {"model": "ridge", "zs": True, "alpha": 30.0},
    "R3 ridge vol+geo+cls+zs a=30": {"model": "ridge", "zs": True, "geo": True, "alpha": 30.0},
    "R4 ridge vol+cls+zs+clipPCA8 a=30": {"model": "ridge", "zs": True, "pca": ["clipL_c14"], "npca": 8, "alpha": 30.0},
}


def main():
    which = sys.argv[1:] or list(CFGS)
    results = {}
    for name in which:
        cfg = CFGS[name]
        t = time.time()
        score, pred, ths = run(cfg)
        r = summarize(real, pred)
        r["thresholds"] = [float(x) for x in ths]
        print(fmt(name, r), f"| {time.time()-t:.0f}s", flush=True)
        _, pred_n, _ = run(cfg, noisy_train=False, noisy_test=True)
        rn = summarize(real, pred_n)
        print(fmt("   noisy-material test", rn), flush=True)
        r["noisy_test"] = rn
        results[name] = r
        np.save(CACHE / f"score_{name.split()[0]}.npy", score)
    json.dump(results, open(CACHE / f"exp2_results_{'_'.join(w.split()[0] for w in which)}.json", "w"), indent=1)


if __name__ == "__main__":
    main()
