import numpy as np
from form_cv import load_bank, report
from form_model import head_predict
bank, y, oid, src, meta = load_bank()
real = src == "real_train"; objs = sorted(set(oid[real]))
def looo(X, sw, wd):
    P = np.zeros((len(y), 2))
    for o in objs:
        tr = np.where((oid != o) & (real | (sw > 0)))[0]; te = np.where((oid == o) & real)[0]
        P[te] = head_predict(X, y, oid, src, tr, te, sw, wd)
    return P
for spec, wd in [("zs3", 1e-1), ("zsn", 1e-1), ("zs3c", 1e-1), ("clipL_c14+zs3", 1e-2), ("clipL_c14+zs3", 1e-1), ("clipL_c14+zsn", 1e-2), ("clipL_c14+zs3c", 1e-2), ("clipL_c14+zs+zsn", 1e-2)]:
    report(looo(bank.features(spec), 0.5, wd), y, oid, real, f"{spec} wd={wd}")
