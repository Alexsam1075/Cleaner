"""Fixed-config LOOO: zero-shot sims as learned features (instead of fixed-temperature blend)."""
import numpy as np
from form_cv import load_bank, report
from form_model import head_predict
bank, y, oid, src, meta = load_bank()
real = src == "real_train"; objs = sorted(set(oid[real]))
def zsfeat(tags=("clipL",), sets=("pile","many","count","mix"), views=("c14","m10","m14")):
    cols = []
    for t in tags:
        for s in sets:
            for v in views:
                S = bank.emb[f"{t}_{v}"] @ bank.text[(t, s)].T
                cols.append(S[:, 1] - S[:, 0])
    return np.stack(cols, 1).astype(np.float32)
for name, X in [("zs clipL all", zsfeat()), ("zs clipL+siglip all", zsfeat(("clipL","siglip"))),
                ("zs clipL c14", zsfeat(views=("c14",))),
                ("clipL_c14 + zs clipL all*3", np.concatenate([bank.features("clipL_c14"), zsfeat()*3], 1)),
                ("clipL_c14 + zs clipL all*1", np.concatenate([bank.features("clipL_c14"), zsfeat()], 1)),
                ("clipL_c14 + zs both*3", np.concatenate([bank.features("clipL_c14"), zsfeat(("clipL","siglip"))*3], 1)),
                ("clipL_c14+m10 + zs clipL*3", np.concatenate([bank.features("clipL_c14+clipL_m10"), zsfeat()*3], 1)),
                ("clipL_c14 + geo + zs*3", np.concatenate([bank.features("clipL_c14+geo"), zsfeat()*3], 1))]:
    for sw in (0.5,):
        for wd in (1e-2, 1e-1, 1.0):
            P = np.zeros((len(y), 2))
            for o in objs:
                tr = np.where((oid != o) & (real | (sw > 0)))[0]; te = np.where((oid == o) & real)[0]
                P[te] = head_predict(X, y, oid, src, tr, te, sw, wd)
            report(P, y, oid, real, f"{name} sw={sw} wd={wd}")
