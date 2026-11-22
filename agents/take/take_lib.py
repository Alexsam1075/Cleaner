"""Nested leave-one-object-out evaluation harness for the TAKE decision."""
from __future__ import annotations

import numpy as np
import pandas as pd

from take_data import HEAVY, MATS, FORMS, build_table, load_feats

LOG500 = float(np.log(500.0))


# ----------------------------------------------------------------------------- metrics

def balanced_acc(y_heavy: np.ndarray, pred_heavy: np.ndarray, w: np.ndarray | None = None) -> float:
    y, p = np.asarray(y_heavy).astype(bool), np.asarray(pred_heavy).astype(bool)
    w = np.ones(len(y)) if w is None else np.asarray(w, float)
    accs = []
    for cls in (True, False):
        m = y == cls
        if m.sum() == 0:
            continue
        accs.append(float(np.sum(w[m] * (p[m] == cls)) / np.sum(w[m])))
    return float(np.mean(accs))


def summarize(df: pd.DataFrame, pred_heavy: np.ndarray) -> dict:
    """df: real rows; pred_heavy aligned. S = mean(frame acc on heavy, frame acc on light)."""
    y = df.heavy.values.astype(bool)
    p = np.asarray(pred_heavy).astype(bool)
    heavy_acc = float((p[y] == True).mean())
    light_acc = float((p[~y] == False).mean())
    per_obj = df.assign(hit=(p == y)).groupby("oid").hit.mean()
    heavy_obj = {o: round(float(per_obj[o]), 2) for o in HEAVY}
    light_miss = {o: round(float(1 - per_obj[o]), 2) for o in per_obj.index if o not in HEAVY and per_obj[o] < 1}
    obj_bal = 0.5 * (np.mean([per_obj[o] for o in HEAVY]) + np.mean([per_obj[o] for o in per_obj.index if o not in HEAVY]))
    return {"S": 0.5 * (heavy_acc + light_acc), "heavy_acc": heavy_acc, "light_acc": light_acc,
            "S_obj": float(obj_bal), "heavy_caught": sum(v >= 0.5 for v in heavy_obj.values()),
            "heavy_obj": heavy_obj, "light_miss": light_miss}


def fmt(name: str, r: dict) -> str:
    hv = " ".join(f"{o}:{v:.2f}" for o, v in r["heavy_obj"].items())
    lm = " ".join(f"{o}:{v:.2f}" for o, v in r["light_miss"].items())
    return (f"{name:34s} S={r['S']:.3f} heavy={r['heavy_acc']:.3f} light={r['light_acc']:.3f} "
            f"caught={r['heavy_caught']}/7 | {hv} | light-miss: {lm}")


# ----------------------------------------------------------------------------- weights

def object_weights(oids: np.ndarray, src: np.ndarray | None = None, synth_w: float = 0.5) -> np.ndarray:
    """Each physical object gets total weight 1 (real frames share 1-synth_w*, synth share synth_w)."""
    oids = np.asarray(oids)
    w = np.zeros(len(oids))
    for o in np.unique(oids):
        m = oids == o
        if src is None:
            w[m] = 1.0 / m.sum()
            continue
        r, s = m & (src == "real_train"), m & (src != "real_train")
        if r.sum() and s.sum():
            w[r] = (1 - synth_w) / r.sum()
            w[s] = synth_w / s.sum()
        else:
            w[m] = 1.0 / m.sum()
    return w


# ----------------------------------------------------------------------------- models

def ridge_fit(X, y, w, alpha):
    mu = np.average(X, axis=0, weights=w)
    sd = np.sqrt(np.average((X - mu) ** 2, axis=0, weights=w))
    sd[sd < 1e-9] = 1.0
    Z = (X - mu) / sd
    ym = float(np.average(y, weights=w))
    A = Z.T @ (Z * w[:, None]) + alpha * np.eye(Z.shape[1])
    coef = np.linalg.solve(A, Z.T @ (w * (y - ym)))
    return coef, mu, sd, ym


def ridge_predict(model, X):
    coef, mu, sd, ym = model
    return ym + ((X - mu) / sd) @ coef


def choose_threshold(score: np.ndarray, y_heavy: np.ndarray, w: np.ndarray, default: float | None = None) -> float:
    """Threshold on `score` (heavy if score > t) maximising weighted balanced accuracy; ties ->
    threshold closest to `default` (or widest margin)."""
    s = np.asarray(score, float)
    order = np.unique(s)
    cands = np.concatenate([[order[0] - 1e-6], (order[:-1] + order[1:]) / 2, [order[-1] + 1e-6]])
    if default is not None:
        cands = np.concatenate([cands, [default]])
    best, best_t = -1, cands[0]
    for t in cands:
        b = balanced_acc(y_heavy, s > t, w)
        if b > best + 1e-12 or (abs(b - best) <= 1e-12 and default is not None and abs(t - default) < abs(best_t - default)):
            best, best_t = b, t
    return float(best_t)


def run_loo(df: pd.DataFrame, make_score, use_synth: bool, inner_threshold: bool = True,
            default_t: float | None = None, inner_folds: int | None = None, seed: int = 0,
            verbose: bool = False) -> tuple[np.ndarray, np.ndarray, list]:
    """Outer LOOO over the 40 objects. make_score(train_df, test_df) -> (score_test, aux).
    Threshold chosen on inner-LOOO (or inner K-fold over objects) OOF scores of the training objects.
    Returns (score, pred_heavy, thresholds) aligned with df[df.real]."""
    real = df[df.real].reset_index(drop=True)
    objs = sorted(real.oid.unique())
    score = np.zeros(len(real))
    pred = np.zeros(len(real), bool)
    thresholds = []
    rng = np.random.default_rng(seed)
    for o in objs:
        tr = df[(df.oid != o) & (df.real | use_synth)]
        te = real[real.oid == o]
        s_te, _ = make_score(tr, te)
        score[te.index] = s_te
        if inner_threshold:
            tr_real = tr[tr.real]
            tr_objs = sorted(tr_real.oid.unique())
            if inner_folds is None or inner_folds >= len(tr_objs):
                groups = [[q] for q in tr_objs]
            else:
                perm = list(rng.permutation(tr_objs))
                groups = [perm[k::inner_folds] for k in range(inner_folds)]
            oof = np.zeros(len(tr_real))
            pos = {i: k for k, i in enumerate(tr_real.index)}
            for g in groups:
                itr = tr[~tr.oid.isin(g)]
                ite = tr_real[tr_real.oid.isin(g)]
                s, _ = make_score(itr, ite)
                oof[[pos[i] for i in ite.index]] = s
            w = object_weights(tr_real.oid.values)
            t = choose_threshold(oof, tr_real.heavy.values, w, default=default_t)
        else:
            t = default_t
        thresholds.append(t)
        pred[te.index] = s_te > t
        if verbose:
            print(o, f"t={t:.3f}", np.round(s_te, 2), te.heavy.iloc[0])
    return score, pred, thresholds


def permutation_check(df: pd.DataFrame, make_score, use_synth: bool, n: int = 5, seed: int = 0, **kw) -> list:
    """Shuffle mass across objects (keeping frames of an object together); S should drop to ~0.5."""
    rng = np.random.default_rng(seed)
    out = []
    objs = sorted(df.oid.unique())
    mass = df.groupby("oid").mass.first()
    for k in range(n):
        perm = dict(zip(objs, rng.permutation([mass[o] for o in objs])))
        d2 = df.copy()
        d2["mass"] = d2.oid.map(perm)
        d2["heavy"] = (d2.mass > 500).astype(int)
        d2["take"] = 1 - d2["heavy"]
        _, p, _ = run_loo(d2, make_score, use_synth, **kw)
        r = d2[d2.real].reset_index(drop=True)
        y = r.heavy.values.astype(bool)
        out.append(0.5 * ((p[y]).mean() + (~p[~y]).mean()))
    return out
