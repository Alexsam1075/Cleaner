"""Experiment 4: physics model  log m = log V_cal + log rho,  rho predicted from material (+form) and a
curated handful of zero-shot VLM cues (full bottle / empty can / metal block / bag / foam / hollow).
Target either GT density (mass/vol) or 'effective density' mass/V_cal. Nested LOOO; alpha and the
decision offset are chosen on inner LOOO of the 39 training objects."""
from __future__ import annotations

import json
import pickle
import sys
import time

import numpy as np
import pandas as pd

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
ROWS = {o: np.where(OID == o)[0] for o in OBJS}
HEAVY_Y = real.heavy.values
LOGMASS = np.log(real.mass.values)
LOGVOL_GT = np.log(real.vol.values)
FUSED = np.load(SHARED / "fused_generic_syn.npy")[real.idx.values]

CUES = {
    "basic": ["heavy-light", "metal_block", "empty_can", "full_bottle", "bag", "foam_light", "hollow-solid"],
    "basic+": ["heavy-light", "metal_block", "empty_can", "full_bottle", "liquid", "bag", "foam_light", "hollow-solid",
               "paper_light", "rock", "solid_plastic", "crushed", "tool_part", "powder", "screws"],
    "min": ["metal_block", "empty_can", "full_bottle", "bag"],
    "none": [],
}


def cue_matrix(names, tags=("clipL", "siglip")):
    cols = []
    for tag in tags:
        for n in names:
            if "-" in n:
                a, b = n.split("-")
                cols.append(real[f"zs_{tag}_{a}"].values - real[f"zs_{tag}_{b}"].values)
            else:
                cols.append(real[f"zs_{tag}_{n}"].values)
    return np.array(cols).T * 20.0 if cols else np.zeros((len(real), 0))


def vol_vec(outer, noisy):
    v = np.zeros(len(real))
    for q in OBJS:
        v[ROWS[q]] = V[(outer, q)]["noisy" if noisy else "true"]
    return v


def class_feats(mat_col, cfg):
    d = real
    cols = [np.array([(d[mat_col] == m).values for m in MATS], float).T]
    if cfg.get("form", True):
        cols.append((d.form == "solid").values[:, None].astype(float))
    if cfg.get("form_x_mat"):
        cols.append(cols[0] * cols[1])
    if cfg.get("matprobs"):
        cols = [FUSED] + cols[1:]
    return np.concatenate(cols, 1)


def design(cfg, mat_col, logv):
    cols = [class_feats(mat_col, cfg), cue_matrix(CUES[cfg.get("cues", "basic")], cfg.get("tags", ("clipL", "siglip")))]
    if cfg.get("use_logv"):
        cols.append(logv[:, None])
    return np.concatenate(cols, 1)


def fit_pred(X, y, tr, te, alpha):
    w = object_weights(OID[tr])
    m = ridge_fit(X[tr], y[tr], w, alpha)
    return ridge_predict(m, X[te])


def scores_for(cfg, outer, tr, te, alpha, noisy_train, noisy_test):
    """log-mass score for te rows, model fitted on tr rows (all volumes nested w.r.t. outer)."""
    logv_tr = np.log(vol_vec(outer, noisy_train))
    logv_te = np.log(vol_vec(outer, noisy_test))
    y = LOGMASS - (logv_tr if cfg.get("target", "eff") == "eff" else LOGVOL_GT)
    Xtr = design(cfg, "mat_noisy" if noisy_train else "mat", logv_tr)
    Xte = design(cfg, "mat_noisy" if noisy_test else "mat", logv_te)
    X = Xtr.copy()
    X[te] = Xte[te]
    logd = fit_pred(X, y, tr, te, alpha)
    return logv_te[te] + logd


ALPHAS = (1.0, 3.0, 10.0, 30.0, 100.0)


def run(cfg, noisy_train=False, noisy_test=False, labels=None, verbose=False):
    global HEAVY_Y
    y = HEAVY_Y if labels is None else labels
    score = np.zeros(len(real))
    pred = np.zeros(len(real), bool)
    chosen = []
    alphas = ALPHAS if cfg.get("alpha") is None else (cfg["alpha"],)
    for o in OBJS:
        te = OID == o
        tr = ~te
        # inner LOOO over training objects: OOF log-mass per alpha
        best = None
        for a in alphas:
            oof = np.zeros(len(real))
            for q in OBJS:
                if q == o:
                    continue
                ite = OID == q
                oof[ite] = scores_for(cfg, o, tr & ~ite, ite, a, noisy_train, noisy_train)
            w = object_weights(OID[tr])
            if cfg.get("tune_t", True):
                t = choose_threshold(oof[tr], y[tr], w, default=LOG500)
                # shrink the tuned offset toward log 500 (robustness with 6 inner heavy objects)
                t = LOG500 + cfg.get("t_shrink", 0.5) * (t - LOG500)
            else:
                t = LOG500
            b = balanced_acc(y[tr], oof[tr] > t, w)
            if best is None or b > best[0] + 1e-9:
                best = (b, a, t)
        _, a, t = best
        s = scores_for(cfg, o, tr, te, a, noisy_train, noisy_test)
        score[te] = s
        pred[te] = s > t
        chosen.append((a, float(t)))
        if verbose:
            print(o, y[te][0], f"a={a} t={np.exp(t):.0f}g", np.round(np.exp(s)).astype(int))
    return score, pred, chosen


def permutation(cfg, n=5, seed=0):
    rng = np.random.default_rng(seed)
    mass = real.groupby("oid").mass.first()
    out = []
    global LOGMASS
    saved = LOGMASS.copy()
    for _ in range(n):
        perm = dict(zip(OBJS, rng.permutation([mass[o] for o in OBJS])))
        LOGMASS = np.log(np.array([perm[o] for o in OID]))
        y = (LOGMASS > LOG500).astype(int)
        _, p, _ = run(cfg, labels=y)
        out.append(balanced_acc(y, p))
    LOGMASS = saved
    return out


CFGS = {
    "P0 mat only, GT-dens": {"cues": "none", "form": False, "target": "gt"},
    "P1 mat only, eff-dens": {"cues": "none", "form": False, "target": "eff"},
    "P2 mat+form, eff": {"cues": "none", "target": "eff"},
    "P3 mat+min cues, eff": {"cues": "min", "form": False, "target": "eff"},
    "P4 mat+basic cues, eff": {"cues": "basic", "form": False, "target": "eff"},
    "P5 mat+form+basic cues, eff": {"cues": "basic", "target": "eff"},
    "P6 mat+basic+ cues, eff": {"cues": "basic+", "form": False, "target": "eff"},
    "P7 mat+basic cues, GT-dens": {"cues": "basic", "form": False, "target": "gt"},
    "P8 mat+basic cues clipL only": {"cues": "basic", "form": False, "target": "eff", "tags": ("clipL",)},
    "P9 mat+basic cues+logV": {"cues": "basic", "form": False, "target": "eff", "use_logv": True},
    "P10 matprobs+basic cues": {"cues": "basic", "form": False, "target": "eff", "matprobs": True},
    "P11 mat+basic, no t tuning": {"cues": "basic", "form": False, "target": "eff", "tune_t": False},
    "P12 mat+basic, full t tuning": {"cues": "basic", "form": False, "target": "eff", "t_shrink": 1.0},
}


def main():
    which = sys.argv[1:] or list(CFGS)
    results = {}
    for name in which:
        cfg = CFGS[name]
        t0 = time.time()
        score, pred, ch = run(cfg)
        r = summarize(real, pred)
        r["chosen"] = ch
        print(fmt(name, r), f"| alphas={sorted(set(a for a, _ in ch))} t~{np.exp(np.median([t for _, t in ch])):.0f}g | {time.time()-t0:.0f}s", flush=True)
        _, pn, _ = run(cfg, noisy_test=True)
        rn = summarize(real, pn)
        print(fmt("   noisy material (test)", rn), flush=True)
        r["noisy_test"] = rn
        results[name] = r
        np.save(CACHE / f"score_{name.split()[0]}.npy", score)
    json.dump(results, open(CACHE / f"exp4_results_{'_'.join(w.split()[0] for w in which)}.json", "w"), indent=1)


if __name__ == "__main__":
    main()
