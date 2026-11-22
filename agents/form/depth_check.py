import numpy as np
from sklearn.metrics import roc_auc_score
from form_cv import load_bank, report
from form_model import head_predict
from common import W
bank, y, oid, src, meta = load_bank()
real = src == "real_train"; idx = np.where(real)[0]; objs = sorted(set(oid[real]))
Z = np.load(W / "cache/gt_depth.npz"); X = Z["X"]; keys = list(Z["keys"])
for j, k in enumerate(keys):
    v = X[:, j]
    auc = roc_auc_score(y[idx], v[idx]); aucs = roc_auc_score(y[~real], v[~real])
    per = {o: float(np.median(v[idx][oid[idx] == o])) for o in objs}; order = sorted(per, key=per.get)
    print(f"{k:14s} AUC real {auc:.3f} synth {aucs:.3f}  {' '.join(f'{o}:{per[o]:.3g}' for o in order)}"[:330])
Xd = np.sign(X) * np.log1p(np.abs(X))
bank.emb["dep"] = Xd
def looo(Xf, sw, wd):
    P = np.zeros((len(y), 2))
    for o in objs:
        tr = np.where((oid != o) & (real | (sw > 0)))[0]; te = np.where((oid == o) & real)[0]
        P[te] = head_predict(Xf, y, oid, src, tr, te, sw, wd)
    return P
for sw in (0.0, 0.5):
    for wd in (1e-2, 1e-1):
        report(looo(Xd.astype(np.float32), sw, wd), y, oid, real, f"depth only sw={sw} wd={wd}")
for wd in (1e-2, 1e-1):
    report(looo(np.concatenate([bank.features("clipL_c14+zs"), Xd], 1).astype(np.float32), 0.5, wd), y, oid, real, f"clipL_c14+zs+depth wd={wd}")
    report(looo(np.concatenate([bank.features("clipL_c14+zs"), Xd * 2], 1).astype(np.float32), 0.5, wd), y, oid, real, f"clipL_c14+zs+depth*2 wd={wd}")
    report(looo(np.concatenate([bank.features("zs"), Xd], 1).astype(np.float32), 0.5, wd), y, oid, real, f"zs+depth wd={wd}")
