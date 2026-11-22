"""Variants of the zs-ensemble prior: (a) which views, (b) object-level standardization, (c) include depth features as extra
standardized "votes", (d) blend scale. Non-nested quick scan (lambda fixed), to pick a small robust design for the nested model."""
import numpy as np

from bank import gt_bank
from form_cv import report
from form_model2 import head_predict, bal_acc
from prompts import FAMILIES

bank, y, oid, src, meta = gt_bank()
real = src == "real_train"
objs = sorted(set(oid[real]))
VIEWS = ("c14", "m10", "m14")
cols, names = [], []
for fam in FAMILIES:
    pb, ps = FAMILIES[fam]
    tb = bank.T[[bank.pi[p] for p in pb]].mean(0)
    ts = bank.T[[bank.pi[p] for p in ps]].mean(0)
    tb, ts = tb / np.linalg.norm(tb), ts / np.linalg.norm(ts)
    for v in VIEWS:
        E = bank.emb[f"clipL_{v}"]
        cols.append(E @ ts - E @ tb)
        names.append((fam, v))
Dm = np.stack(cols, 1)
dk = list(np.load("/home/cleaner/Cleaner/agents2/form/cache/gt_v2.npz")["depth_keys"])
DEP = bank.emb["depth"]
# depth votes oriented so that larger = more solid: fill, fill_thr, h_med_rel, h75_rel, solidity, -low10_frac, -n_peaks_area
dep_cols = np.stack([DEP[:, dk.index("fill_thr")], DEP[:, dk.index("fill")], DEP[:, dk.index("h_med_rel")], DEP[:, dk.index("h75_rel")],
                     DEP[:, dk.index("solidity")], -DEP[:, dk.index("low10_frac")], -DEP[:, dk.index("edge_density")]], 1)


def best_thr(s, yy):
    cand = np.quantile(s, np.linspace(0.05, 0.95, 91))
    return max(cand, key=lambda t: bal_acc(s, yy, t))


def zs_ens(M, tr_all, S, robust=False):
    if robust:
        med = np.median(M[tr_all], 0)
        mad = np.median(np.abs(M[tr_all] - med), 0) * 1.4826 + 1e-6
        Z = (M - med) / mad
        Z = np.clip(Z, -3, 3)
    else:
        Z = (M - M[tr_all].mean(0)) / (M[tr_all].std(0) + 1e-6)
    return Z[:, S].mean(1)


X = bank.features("clipL_c14+zs")
head_cache = {}
for o in objs:
    tr_all = np.where(oid != o)[0]
    te = np.where((oid == o) & real)[0]
    ph = head_predict(X, y, oid, src, tr_all, te, 0.5, 1e-2)[:, 1]
    head_cache[o] = np.log(ph + 1e-6) - np.log(1 - ph + 1e-6)


def run(M, S, lam, scale, label, robust=False, only_ens=False):
    P = np.zeros(len(y))
    for o in objs:
        tr_all = np.where(oid != o)[0]
        tr = np.where((oid != o) & real)[0]
        te = np.where((oid == o) & real)[0]
        s = zs_ens(M, tr_all, S, robust)
        t = best_thr(s[tr], y[tr])
        z = (0 if only_ens else head_cache[o]) + lam * scale * (s[te] - t)
        P[te] = 1 / (1 + np.exp(-z))
    return report(P, y, oid, real, label)


ALL = list(range(Dm.shape[1]))
for vset in (("c14",), ("m10",), ("m14",), ("c14", "m10", "m14")):
    S = [i for i, (f, v) in enumerate(names) if v in vset]
    run(Dm, S, 1, 2.0, f"ens-only views={vset}", only_ens=True)
run(Dm, ALL, 1, 2.0, "ens-only robust z", robust=True, only_ens=True)
MD = np.concatenate([Dm, dep_cols], 1)
run(MD, list(range(MD.shape[1])), 1, 2.0, "ens-only zs+depth votes", only_ens=True)
run(MD, list(range(MD.shape[1])), 1, 2.0, "ens-only zs+depth votes robust", robust=True, only_ens=True)
DEPW = np.concatenate([Dm, np.repeat(dep_cols, 4, axis=1)], 1)
run(DEPW, list(range(DEPW.shape[1])), 1, 2.0, "ens-only zs+depth votes x4", only_ens=True)
for lam in (2.0, 4.0, 6.0):
    run(Dm, ALL, lam, 2.0, f"head + {lam}*ens(all)")
    run(Dm, ALL, lam, 2.0, f"head + {lam}*ens(all) robust", robust=True)
    run(MD, list(range(MD.shape[1])), lam, 2.0, f"head + {lam}*ens(zs+depth)")
    run(DEPW, list(range(DEPW.shape[1])), lam, 2.0, f"head + {lam}*ens(zs+depth x4)")
