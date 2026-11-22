"""Experiment 3: physics decomposition. mass = calibrated volume x density; density predicted from
appearance (material one-hot, zero-shot concepts, embedding PCA / kNN) by LOOO. Diagnostics per object."""
from __future__ import annotations

import pickle
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, "/home/cleaner/Cleaner/agents/take")
from take_data import HEAVY, MATS, FORMS, CACHE, SHARED, build_table, load_feats
from take_lib import LOG500, choose_threshold, fmt, object_weights, ridge_fit, ridge_predict, summarize

df = build_table()
F = load_feats()
real = df[df.real].reset_index(drop=True)
VC = pickle.load(open(CACHE / "vol_cache.pkl", "rb"))
V = VC["V"]
OBJS = VC["objs"]
OID = real.oid.values
ROWS = {o: np.where(OID == o)[0] for o in OBJS}
LOGDENS = np.log(real.mass.values / real.vol.values)
ZS_COLS = [c for c in real.columns if c.startswith("zs_")]
pd.set_option("display.width", 250)


def onehot(d, mat_col="mat"):
    return np.concatenate([np.array([(d.form == f).values for f in FORMS], float).T,
                           np.array([(d[mat_col] == m).values for m in MATS], float).T], 1)


def l2n(x):
    return x / np.linalg.norm(x, axis=1, keepdims=True)


def feat(cfg, idx_fit):
    cols = []
    if cfg.get("cls", True):
        cols.append(onehot(real))
    if cfg.get("zs"):
        cols.append(real[ZS_COLS].values * 20)
    if cfg.get("emb"):
        X = np.concatenate([l2n(F[k][real.idx.values]) for k in cfg["emb"]], 1)
        mu = X[idx_fit].mean(0)
        U, S, Vt = np.linalg.svd(X[idx_fit] - mu, full_matrices=False)
        n = cfg.get("npca", 16)
        cols.append(((X - mu) @ Vt[:n].T) / (S[:n] / np.sqrt(len(idx_fit))))
    return np.concatenate(cols, 1)


def loo_density(cfg):
    """LOOO predicted log density per frame."""
    pred = np.zeros(len(real))
    for o in OBJS:
        te = OID == o
        tr = ~te
        X = feat(cfg, np.where(tr)[0])
        w = object_weights(OID[tr])
        if cfg.get("model") == "knn":
            # object-mean embeddings, cosine kNN in training objects
            E = l2n(np.concatenate([l2n(F[k][real.idx.values]) for k in cfg["emb"]], 1))
            objs_tr = [q for q in OBJS if q != o]
            M = l2n(np.stack([E[ROWS[q]].mean(0) for q in objs_tr]))
            ld = np.array([LOGDENS[ROWS[q]][0] for q in objs_tr])
            S = E[te] @ M.T
            k = cfg.get("k", 3)
            top = np.argsort(-S, 1)[:, :k]
            wts = np.exp(S[np.arange(len(S))[:, None], top] / cfg.get("tau", 0.05))
            pred[te] = (wts * ld[top]).sum(1) / wts.sum(1)
        else:
            m = ridge_fit(X[tr], LOGDENS[tr], w, cfg.get("alpha", 10.0))
            pred[te] = ridge_predict(m, X[te])
    return pred


def evaluate(name, logd):
    vol_outer = np.zeros(len(real))
    for o in OBJS:
        vol_outer[ROWS[o]] = V[(o, o)]["true"]
    logmass = np.log(vol_outer) + logd
    logmass_gt = np.log(real.vol.values) + logd
    r_fixed = summarize(real, logmass > LOG500)
    r_gt = summarize(real, logmass_gt > LOG500)
    print(fmt(f"{name} [fixed500]", r_fixed))
    print(fmt(f"{name} [GT vol]", r_gt))
    tab = real.assign(ld=logd, lm=logmass).groupby("oid").agg(mat=("mat", "first"), mass=("mass", "first"), vol=("vol", "first"),
                                                             dens_true=("mass", lambda s: 0), ld=("ld", "mean"), lm=("lm", "median"))
    tab["dens_true"] = (tab.mass / tab.vol).round(2)
    tab["dens_pred"] = np.exp(tab.ld).round(2)
    tab["mass_pred"] = np.exp(tab.lm).round(0)
    return r_fixed, tab


CFGS = {
    "D0 class mean": {"cls": True, "alpha": 1.0},
    "D1 cls+zs a10": {"cls": True, "zs": True, "alpha": 10.0},
    "D2 cls+zs a30": {"cls": True, "zs": True, "alpha": 30.0},
    "D3 cls+zs a100": {"cls": True, "zs": True, "alpha": 100.0},
    "D4 cls+clipPCA16 a30": {"cls": True, "emb": ["clipL_c14"], "npca": 16, "alpha": 30.0},
    "D5 cls+clipm10PCA16 a30": {"cls": True, "emb": ["clipL_m10"], "npca": 16, "alpha": 30.0},
    "D6 cls+dinoPCA16 a30": {"cls": True, "emb": ["dinoL_m10"], "npca": 16, "alpha": 30.0},
    "D7 cls+zs+clipPCA16 a30": {"cls": True, "zs": True, "emb": ["clipL_c14"], "npca": 16, "alpha": 30.0},
    "K1 knn3 clipL_c14": {"model": "knn", "emb": ["clipL_c14"], "k": 3},
    "K2 knn3 clipL_m10": {"model": "knn", "emb": ["clipL_m10"], "k": 3},
    "K3 knn3 dinoL_m10": {"model": "knn", "emb": ["dinoL_m10"], "k": 3},
    "K4 knn5 clipL_c14+m10": {"model": "knn", "emb": ["clipL_c14", "clipL_m10"], "k": 5},
    "K5 knn1 clipL_c14": {"model": "knn", "emb": ["clipL_c14"], "k": 1},
}


def main():
    which = sys.argv[1:] or list(CFGS)
    tabs = {}
    for name in which:
        logd = loo_density(CFGS[name])
        r, tab = evaluate(name, logd)
        tabs[name] = tab
        np.save(CACHE / f"logd_{name.split()[0]}.npy", logd)
    base = tabs[which[0]][["mat", "mass", "vol", "dens_true"]].copy()
    for name in which:
        base[name.split()[0]] = tabs[name]["dens_pred"]
    print(base.sort_values(["mat", "dens_true"]))


if __name__ == "__main__":
    main()
