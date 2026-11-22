import numpy as np
from sklearn.metrics import roc_auc_score
from form_cv import load_bank, report
from form_model import head_predict, ZS_SPECS, l2n
from common import W
bank, y, oid, src, meta = load_bank()
real = src == "real_train"; idx = np.where(real)[0]; objs = sorted(set(oid[real]))
H = np.load(W / "cache/gt_hires.npz")
for k in ("q_c14", "q_m14", "c12"): bank.emb[f"clipL_{k}"] = H[k]
# tile-level zero-shot: per tile sims, then mean/max over tiles
T = bank.text
for k in ("q_c14_tiles", "q_m14_tiles"):
    E = H[k]
    for s in ("mix", "pile", "parts", "multi"):
        d = (E @ T[("clipL", s)].T); d = d[..., 1] - d[..., 0]  # (N,4)
        for agg, v in (("mean", d.mean(1)), ("min", d.min(1)), ("max", d.max(1))):
            print(f"{k} {s} {agg} AUC {roc_auc_score(y[idx], v[idx]):.3f}")
    bank.emb[f"clipL_{k}"] = E
def looo(X, sw, wd):
    P = np.zeros((len(y), 2))
    for o in objs:
        tr = np.where((oid != o) & (real | (sw > 0)))[0]; te = np.where((oid == o) & real)[0]
        P[te] = head_predict(X, y, oid, src, tr, te, sw, wd)
    return P
ZS_SPECS["zsq"] = (("clipL",), ("pile", "many", "count", "mix", "multi", "crumpled", "parts", "trash"), ("c14", "m10", "m14", "q_c14", "q_m14"))
ZS_SPECS["zsq2"] = (("clipL",), ("pile", "many", "count", "mix", "multi", "crumpled", "parts", "trash"), ("q_c14", "q_m14", "c12"))
for spec in ("clipL_q_c14", "clipL_q_c14+zs", "clipL_c12+zs", "clipL_c14+clipL_q_c14+zs", "clipL_c14+zsq", "clipL_q_c14+zsq", "zsq", "zsq2", "clipL_c14+zsq2", "clipL_q_m14+zs"):
    report(looo(bank.features(spec), 0.5, 1e-2 if "clipL" in spec else 1e-1), y, oid, real, spec)
