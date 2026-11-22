"""Three-term blend: head + lam1 * zero-shot prior + lam2 * depth prior (equal-weight z-votes of height-map features whose sign is
fixed a priori: fuller/higher-relative/convex => solid). Fixed-lambda LOOO grid to see whether depth separates soft solids
(s01, s11) from paper piles (b07)."""
import numpy as np

from bank import gt_bank
from form_cv import report
from form_model2 import head_predict, Prior, logit, sigmoid

bank, y, oid, src, meta = gt_bank()
real = src == "real_train"
objs = sorted(set(oid[real]))
dk = list(np.load("/home/cleaner/Cleaner/agents2/form/cache/gt_v2.npz")["depth_keys"])
DEP = bank.emb["depth"]
SIGNED = {"fill_thr": 1, "fill": 1, "h_med_rel": 1, "h75_rel": 1, "solidity": 1, "low10_frac": -1, "edge_density": -1, "n_peaks_area": -1}
MD = np.stack([s * DEP[:, dk.index(k)] for k, s in SIGNED.items()], 1)
MZ = bank.zs_matrix()
X = bank.features("clipL_c14+zs")
head = {}
for o in objs:
    tr_all = np.where(oid != o)[0]
    te = np.where((oid == o) & real)[0]
    head[o] = logit(head_predict(X, y, oid, src, tr_all, te, 0.5, 1e-2)[:, 1])
pri = {}
for o in objs:
    tr_all = np.where(oid != o)[0]
    tr = np.where((oid != o) & real)[0]
    te = np.where((oid == o) & real)[0]
    pri[o] = (Prior().fit(MZ, tr_all, y[tr], tr).z(MZ)[te], Prior().fit(MD, tr_all, y[tr], tr).z(MD)[te])
print("depth prior per object (z, mean over frames):", {o: round(float(pri[o][1].mean()), 2) for o in ["s01", "s11", "s12", "b07", "b06", "s19", "s48", "b19"]})
for lam1 in (0.0, 2.0, 4.0, 8.0):
    for lam2 in (0.0, 1.0, 2.0, 4.0):
        P = np.zeros(len(y))
        for o in objs:
            te = np.where((oid == o) & real)[0]
            P[te] = sigmoid(head[o] + lam1 * pri[o][0] + lam2 * pri[o][1])
        report(P, y, oid, real, f"head + {lam1}*zs + {lam2}*depth")
