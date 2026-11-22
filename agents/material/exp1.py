"""Exploration 1: diagnostics of the new features (feat3.npz).
 (a) zero-shot M per model/view/prompt set (no selection involved -> plain LOOO-equivalent numbers);
 (b) linear head LOOO per feature block / view-averaged / dense-pooled;
 (c) nested fusion for a few promising combos.
"""
import sys
import numpy as np
from cvlib import *

F = load_feats()
meta, oid, src, mat, light = load_meta()
real = src == "real_train"
idx = np.where(real)[0]
objs = sorted(set(oid[real]))
PS = ["name_t1", "name_t5", "name_t12", "adj_t", "syn_t5", "syn_t12"]
VIEWS = ["c12", "c15", "c20", "m12", "n15", "n20"]


def show(name, P):
    mo, per = M_obj(P, mat, oid, idx)
    print(f"  {name:60s} M={M_frame(P, mat, idx):.3f} Mobj={mo:.3f} {per_mat(P, mat, idx)}", flush=True)


def zs_logits(model, view, ps):
    return F[f"{model}_{view}"] @ F[f"txt_{model}_{ps}"].T


print("== (a) zero-shot: rows=view, cols=prompt set, each cell M (clipL / siglip / both) ==")
for v in VIEWS + ["clipLmc_c15", "clipLpt_c15", "clipLmc_c20", "clipLmc_n15", "siglipmp_c15", "siglipmp_c20"]:
    row = []
    for ps in PS:
        cells = []
        if v in VIEWS:
            a = zs_logits("clipL", v, ps)
            b = zs_logits("siglip", v, ps)
            cells = [M_frame(a, mat, idx), M_frame(b, mat, idx), M_frame(softmax(a / 0.01) * 0 + (softmax(a / 0.01) + softmax(b / 0.01)) / 2, mat, idx)]
        else:
            m = "clipL" if v.startswith("clipL") else "siglip"
            a = F[v] @ F[f"txt_{m}_{ps}"].T
            cells = [M_frame(a, mat, idx)]
        row.append("/".join(f"{c:.2f}" for c in cells))
    print(f"  {v:14s} " + "  ".join(f"{ps}:{r}" for ps, r in zip(PS, row)))

print("\n== (a2) zero-shot multi-view ensembles (mean logits over views) ==")
for ps in PS:
    for name, vs in {"ctx3": ["c12", "c15", "c20"], "ctx3+norm2": ["c12", "c15", "c20", "n15", "n20"], "all6": VIEWS, "c15+n15": ["c15", "n15"]}.items():
        a = np.mean([zs_logits("clipL", v, ps) for v in vs], 0)
        b = np.mean([zs_logits("siglip", v, ps) for v in vs], 0)
        show(f"zs {ps} {name} clipL|siglip|both", np.stack([a, b]).mean(0) * 0 + (softmax(a / 0.01) + softmax(b / 0.01)) / 2)
        print(f"      clipL only M={M_frame(a, mat, idx):.3f}   siglip only M={M_frame(b, mat, idx):.3f}")

print("\n== (b) linear head LOOO (real only), wd=1e-2 ==")
blocks = {
    "clipL_c15": ["clipL_c15"], "clipL_c12": ["clipL_c12"], "clipL_c20": ["clipL_c20"], "clipL_n15": ["clipL_n15"], "clipL_m12": ["clipL_m12"],
    "old clipL_c14": ["old_clipL_c14"],
    "clipLmc_c15": ["clipLmc_c15"], "clipLpt_c15": ["clipLpt_c15"], "clipLmc_c20": ["clipLmc_c20"],
    "siglip_c15": ["siglip_c15"], "siglipmp_c15": ["siglipmp_c15"],
    "clipL_c15+clipLmc_c15": ["clipL_c15", "clipLmc_c15"],
    "clipL_c15+siglipmp_c15": ["clipL_c15", "siglipmp_c15"],
    "clipL_c15+hand": ["clipL_c15", "hand"],
}
lin = {}
for name, keys in blocks.items():
    X = np.concatenate([emb_block(F[k]) if k != "hand" else F[k] for k in keys], 1)
    for wd in (1e-2,):
        P = looo_linear(X, mat, oid, src, real, objs, wd=wd)
        lin[name] = P
        show(f"lin {name} wd={wd}", P)

print("\n== (b2) linear head with multi-view training augmentation (views as extra rows), test = mean prob over views ==")


def looo_multiview(views, keys_fn, wd=1e-2):
    Xs = [keys_fn(v) for v in views]
    P = np.zeros((len(mat), 6))
    for o in objs:
        tr = np.where((oid != o) & real)[0]
        te = np.where((oid == o) & real)[0]
        Xtr = np.concatenate([X[tr] for X in Xs])
        ytr = np.concatenate([mat[tr]] * len(Xs))
        w = np.concatenate([weights(tr, oid, src, mat)] * len(Xs))
        st = Standardizer(Xtr)
        W, b = fit_logreg(st(Xtr), ytr, w, wd=wd)
        P[te] = np.mean([predict_logreg(st(X[te]), W, b) for X in Xs], 0)
    return P


for name, vs in {"ctx3": ["c12", "c15", "c20"], "ctx3+norm2": ["c12", "c15", "c20", "n15", "n20"], "c15+n15": ["c15", "n15"]}.items():
    P = looo_multiview(vs, lambda v: emb_block(F[f"clipL_{v}"]))
    lin[f"mv_{name}"] = P
    show(f"lin clipL multiview {name}", P)
P = looo_multiview(["c15", "c20", "n15"], lambda v: emb_block(F[f"clipLmc_{v}"]))
lin["mv_mc3"] = P
show("lin clipLmc multiview c15,c20,n15", P)

np.savez_compressed(HERE / "cache" / "exp1_lin.npz", **lin)

print("\n== (c) nested fusion lin ⊕ zero-shot (prompt set, temperature, w chosen by inner 5-fold grouped CV) ==")


def nested_fuse(Xfn_views, Zcands, grid_w=(0.0, 0.2, 0.35, 0.5, 0.65, 0.8, 1.0), temps=(0.01, 0.02), wd=1e-2, seed=0):
    Xs = Xfn_views
    Pout = np.zeros((len(mat), 6))
    chosen = []
    for o in objs:
        T = [q for q in objs if q != o]
        Pin = np.zeros((len(mat), 6))
        for g in inner_folds(T, 5, seed):
            tr = np.where(np.isin(oid, T) & ~np.isin(oid, g) & real)[0]
            te = np.where(np.isin(oid, g) & real)[0]
            Xtr = np.concatenate([X[tr] for X in Xs])
            st = Standardizer(Xtr)
            W, b = fit_logreg(st(Xtr), np.concatenate([mat[tr]] * len(Xs)), np.concatenate([weights(tr, oid, src, mat)] * len(Xs)), wd=wd)
            Pin[te] = np.mean([predict_logreg(st(X[te]), W, b) for X in Xs], 0)
        ii = np.where(np.isin(oid, T) & real)[0]
        best, bk = -1, None
        for zname, Zl in Zcands.items():
            for t in temps:
                Z = softmax(Zl / t)
                for w in grid_w:
                    s = M_frame(softmax((1 - w) * np.log(Pin + 1e-9) + w * np.log(Z + 1e-9)), mat, ii)
                    if s > best:
                        best, bk = s, (zname, t, w)
        chosen.append(bk)
        tr = np.where(np.isin(oid, T) & real)[0]
        te = np.where((oid == o) & real)[0]
        Xtr = np.concatenate([X[tr] for X in Xs])
        st = Standardizer(Xtr)
        W, b = fit_logreg(st(Xtr), np.concatenate([mat[tr]] * len(Xs)), np.concatenate([weights(tr, oid, src, mat)] * len(Xs)), wd=wd)
        Pl = np.mean([predict_logreg(st(X[te]), W, b) for X in Xs], 0)
        Z = softmax(Zcands[bk[0]][te] / bk[1])
        Pout[te] = softmax((1 - bk[2]) * np.log(Pl + 1e-9) + bk[2] * np.log(Z + 1e-9))
    from collections import Counter
    return Pout, Counter(chosen)


def zl(models, views, ps):
    return np.mean([zs_logits(m, v, ps) for m in models for v in views], 0)


Zc = {}
for ps in PS:
    Zc[f"{ps}|both|ctx3n2"] = zl(["clipL", "siglip"], ["c12", "c15", "c20", "n15", "n20"], ps)
    Zc[f"{ps}|both|c15"] = zl(["clipL", "siglip"], ["c15"], ps)
    Zc[f"{ps}|clipL|ctx3n2"] = zl(["clipL"], ["c12", "c15", "c20", "n15", "n20"], ps)
for lname, vs in {"c15": ["c15"], "ctx3+norm2": ["c12", "c15", "c20", "n15", "n20"]}.items():
    P, ch = nested_fuse([emb_block(F[f"clipL_{v}"]) for v in vs], Zc)
    show(f"NESTED lin clipL[{lname}] ⊕ zs(select over {len(Zc)} cands)", P)
    print("      chosen:", ch.most_common(5))
    np.save(HERE / "cache" / f"exp1_nested_{lname}.npy", P)
