import numpy as np
from form_cv import load_bank, report
from form_model import head_predict, ZS_SPECS
bank, y, oid, src, meta = load_bank()
real = src == "real_train"; objs = sorted(set(oid[real]))
def looo(X, sw, wd):
    P = np.zeros((len(y), 2))
    for o in objs:
        tr = np.where((oid != o) & (real | (sw > 0)))[0]; te = np.where((oid == o) & real)[0]
        P[te] = head_predict(X, y, oid, src, tr, te, sw, wd)
    return P
ZS_SPECS["zsc"] = (("clipL",), ("pile", "many", "count", "mix", "multi", "crumpled", "parts", "trash"), ("c14",))
ZS_SPECS["zscm"] = (("clipL",), ("pile", "many", "count", "mix", "multi", "crumpled", "parts", "trash"), ("c14", "m10"))
ZS_SPECS["zs4"] = (("clipL",), ("pile", "mix", "parts", "crumpled"), ("c14", "m10", "m14"))
Ps = {}
for spec, sw, wd in [("clipL_c14+zs", 0.5, 3e-3), ("clipL_c14+zs", 0.5, 3e-2), ("clipL_c14+zsc", 0.5, 1e-2), ("clipL_c14+zscm", 0.5, 1e-2), ("clipL_c14+zs4", 0.5, 1e-2),
                     ("clipL_c14+dinoL_m10+zs", 0.5, 1e-2), ("clipL_c14+clipL_m14+zs", 0.5, 1e-2), ("clipL_m10+zs", 0.5, 1e-2), ("clipL_m14+zs", 0.5, 1e-2),
                     ("clipL_c14+siglip_c14+zs", 0.5, 1e-2), ("clipL_c14+geo+zs", 0.5, 1e-2), ("clipL_c14+zs", 0.5, 1e-2), ("zs", 0.5, 1e-1), ("clipL_c14+clipL_m10+zs", 0.5, 1e-2)]:
    X = bank.features(spec)
    if spec.endswith("+zs") and False: pass
    P = looo(X, sw, wd); Ps[(spec, sw, wd)] = P
    report(P, y, oid, real, f"{spec} sw={sw} wd={wd}")
ens = [("clipL_c14+zs", 0.5, 1e-2), ("clipL_c14+geo+zs", 0.5, 1e-2), ("clipL_c14+clipL_m10+zs", 0.5, 1e-2), ("zs", 0.5, 1e-1)]
report(np.mean([Ps[k] for k in ens], 0), y, oid, real, "ensemble4")
report(np.mean([Ps[k] for k in ens[:3]], 0), y, oid, real, "ensemble3")
