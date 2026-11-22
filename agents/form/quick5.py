import numpy as np
from sklearn.metrics import roc_auc_score
from form_cv import load_bank, report
from form_model import head_predict, l2n
bank, y, oid, src, meta = load_bank()
real = src == "real_train"; objs = sorted(set(oid[real])); idx = np.where(real)[0]
def looo(X, sw, wd, exclude=()):
    P = np.zeros((len(y), 2))
    for o in objs:
        tr = np.where((oid != o) & ~np.isin(oid, exclude) & (real | (sw > 0)))[0]; te = np.where((oid == o) & real)[0]
        P[te] = head_predict(X, y, oid, src, tr, te, sw, wd)
    return P
Z = bank.zs_features("zs") / 3
Zs = (Z - Z[idx].mean(0)) / Z[idx].std(0)
m = Zs.mean(1)
print("mean z-scored zs AUC", roc_auc_score(y[idx], m[idx]))
per = {o: float(m[idx][oid[idx] == o].mean()) for o in objs}; print("order", " ".join(sorted(per, key=per.get)))
# average embedding over views
bank.emb["clipL_avg"] = l2n(sum(l2n(bank.emb[f"clipL_{v}"]) for v in ("c14", "m14", "c20", "m10")))
bank.emb["clipL_avg2"] = l2n(l2n(bank.emb["clipL_c14"]) + l2n(bank.emb["clipL_m14"]))
report(looo(bank.features("clipL_avg+zs"), 0.5, 1e-2), y, oid, real, "clipL_avg(4 views)+zs")
report(looo(bank.features("clipL_avg2+zs"), 0.5, 1e-2), y, oid, real, "clipL_avg(c14,m14)+zs")
report(looo(bank.features("clipL_avg"), 0.5, 1e-2), y, oid, real, "clipL_avg(4 views)")
report(looo(bank.features("clipL_c14+zs"), 0.5, 1e-2, exclude=("b07",)), y, oid, real, "clipL_c14+zs, b07 excluded from train")
report(looo(bank.features("clipL_c14+zs"), 0.5, 1e-2, exclude=("b06",)), y, oid, real, "clipL_c14+zs, b06 excluded from train")
report(looo(bank.features("zs"), 0.5, 1e-1, exclude=("b07",)), y, oid, real, "zs, b07 excluded from train")
# prompt-set-wise structure: use only 'multi','pile','crumpled','parts' on c14 -> 4 features
from form_model import ZS_SPECS
ZS_SPECS["zs_small"] = (("clipL",), ("multi", "pile", "crumpled", "parts"), ("c14",))
for wd in (1e-2, 1e-1, 1.0):
    report(looo(bank.features("zs_small"), 0.5, wd), y, oid, real, f"zs_small wd={wd}")
    report(looo(bank.features("clipL_c14+zs_small"), 0.5, wd), y, oid, real, f"clipL_c14+zs_small wd={wd}")
