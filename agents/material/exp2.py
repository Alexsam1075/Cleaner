"""Exploration 2: text-anchored linear head in the VLM embedding space.
logits = X @ W.T + b, X = L2-normalised image embedding, W initialised at text prototypes / tau,
penalty lam * ||W - W0||^2 -> lam large = zero-shot, lam small = free linear head.
All configs (prompt set x lam) are optimised jointly in one LBFGS run (losses are independent).
Multi-view: train rows = all views, test = mean logits over views. Selection of (prompt set, lam, tau) nested.
"""
import itertools
import sys
from collections import Counter

import numpy as np
import torch

from cvlib import *

F = load_feats()
meta, oid, src, mat, light = load_meta()
real = src == "real_train"
idx = np.where(real)[0]
objs = sorted(set(oid[real]))
PS = ["name_t1", "name_t5", "name_t12", "adj_t", "syn_t5", "syn_t12"]
LAMS = [1e-3, 1e-2, 3e-2, 1e-1, 3e-1, 1.0, 3.0]
TAUS = [0.01, 0.02]
CFGS = list(itertools.product(PS, LAMS, TAUS))


def show(name, P):
    mo, per = M_obj(P, mat, oid, idx)
    print(f"  {name:70s} M={M_frame(P, mat, idx):.3f} Mobj={mo:.3f} {per_mat(P, mat, idx)}", flush=True)


def fit_anchored_multi(Xtr_views, ytr, w, Tdict, cfgs, iters=100):
    """Xtr_views: list of (n,d) arrays (one per view) for the same rows. Tdict: ps -> (6,d) text prototypes.
    Returns W (C,d,6), b (C,6) for every config in cfgs = [(ps, lam, tau)]; all fitted jointly (independent losses)."""
    X = torch.tensor(np.concatenate(Xtr_views), device=DEV, dtype=torch.float32)
    y = torch.tensor(np.concatenate([ytr] * len(Xtr_views)), device=DEV)
    ww = torch.tensor(np.concatenate([w] * len(Xtr_views)), device=DEV, dtype=torch.float32)
    ww = ww / ww.sum()
    C = len(cfgs)
    lam = torch.tensor([c[1] for c in cfgs], device=DEV).view(C, 1, 1)
    W0 = torch.stack([torch.tensor(Tdict[c[0]], device=DEV, dtype=torch.float32).T / c[2] for c in cfgs])  # C,d,6
    dW = torch.zeros_like(W0, requires_grad=True)
    b = torch.zeros(C, 6, device=DEV, requires_grad=True)
    opt = torch.optim.LBFGS([dW, b], lr=1, max_iter=iters, history_size=20, line_search_fn="strong_wolfe")

    def closure():
        opt.zero_grad()
        logits = torch.einsum("nd,cdk->cnk", X, W0 + dW) + b[:, None, :]
        ce = torch.nn.functional.cross_entropy(logits.reshape(-1, 6), y.repeat(C), reduction="none").view(C, -1)
        loss = (ce * ww).sum() + (lam * dW ** 2).sum()
        loss.backward()
        return loss

    opt.step(closure)
    return (W0 + dW).detach(), b.detach()


def predict_multi(Xte_views, W, b):
    Z = 0
    for X in Xte_views:
        Z = Z + torch.einsum("nd,cdk->cnk", torch.tensor(X, device=DEV, dtype=torch.float32), W) + b[:, None, :]
    return (Z / len(Xte_views)).cpu().numpy()  # C,n,6


def run(models, views, tag, synth_w=0.0):
    """Returns nested result + oracle table. Logits of the models are summed."""
    Xv = {m: [l2n(F[f"{m}_{v}"]).astype(np.float32) for v in views] for m in models}
    nc = len(CFGS)
    oof_logits = np.zeros((nc, len(mat), 6))  # outer OOF for every config (for oracle)
    Pout = np.zeros((len(mat), 6))
    chosen = []
    for o in objs:
        Tobjs = [q for q in objs if q != o]
        inner = np.zeros((nc, len(mat), 6))
        for g in inner_folds(Tobjs, 5, 0):
            tr = np.where(np.isin(oid, Tobjs) & ~np.isin(oid, g) & (real | (synth_w > 0)))[0]
            te = np.where(np.isin(oid, g) & real)[0]
            w = weights(tr, oid, src, mat, synth_w)
            for m in models:
                W, b = fit_anchored_multi([X[tr] for X in Xv[m]], mat[tr], w, {ps: F[f"txt_{m}_{ps}"] for ps in PS}, CFGS)
                inner[:, te] += predict_multi([X[te] for X in Xv[m]], W, b)
        ii = np.where(np.isin(oid, Tobjs) & real)[0]
        sc = [M_frame(inner[k], mat, ii) for k in range(nc)]
        best = int(np.argmax(sc))
        chosen.append(CFGS[best])
        tr = np.where(np.isin(oid, Tobjs) & (real | (synth_w > 0)))[0]
        te = np.where((oid == o) & real)[0]
        w = weights(tr, oid, src, mat, synth_w)
        for m in models:
            W, b = fit_anchored_multi([X[tr] for X in Xv[m]], mat[tr], w, {ps: F[f"txt_{m}_{ps}"] for ps in PS}, CFGS)
            oof_logits[:, te] += predict_multi([X[te] for X in Xv[m]], W, b)
        Pout[te] = softmax(oof_logits[best, te])
        print(".", end="", flush=True)
    print()
    show(f"NESTED anchored {tag}", Pout)
    print("      chosen:", Counter(chosen).most_common(4))
    oracle = sorted([(M_frame(oof_logits[k], mat, idx), CFGS[k]) for k in range(nc)], reverse=True)
    print("      ORACLE (outer-selected, optimistic) top5:", [(f"{s:.3f}", c) for s, c in oracle[:5]])
    print("      per-lam best (oracle):", {l: f"{max(s for s, c in oracle if c[1] == l):.3f}" for l in LAMS})
    np.savez_compressed(HERE / "cache" / f"exp2_{tag}.npz", nested=Pout, oof=oof_logits, cfgs=np.array([str(c) for c in CFGS]))
    return Pout


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    if which in ("all", "a"):
        run(["clipL"], ["c15"], "clipL_c15")
        run(["clipL"], ["c12", "c15", "c20"], "clipL_ctx3")
        run(["siglip"], ["c12", "c15", "c20"], "siglip_ctx3")
        run(["clipL", "siglip"], ["c12", "c15", "c20"], "both_ctx3")
    if which in ("all", "b"):
        run(["clipL", "siglip"], ["c15"], "both_c15")
        run(["clipL", "siglip"], ["c12", "c15", "c20", "n15", "n20"], "both_ctx3n2")
        run(["clipL", "siglip"], ["c12", "c15", "c20"], "both_ctx3_syn", synth_w=0.3)
