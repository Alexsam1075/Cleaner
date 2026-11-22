"""Equal-weight zero-shot ensemble (mean of z-scored sim differences) as a 1-D feature; z-scoring uses train stats only."""
import numpy as np
from form_cv import load_bank, report
from form_model import head_predict, ZS_SPECS
bank, y, oid, src, meta = load_bank()
real = src == "real_train"; idx = np.where(real)[0]; objs = sorted(set(oid[real]))
def looo_fn(make, sw, wd):
    P = np.zeros((len(y), 2))
    for o in objs:
        tr = np.where((oid != o) & (real | (sw > 0)))[0]; te = np.where((oid == o) & real)[0]
        X = make(tr)
        P[te] = head_predict(X, y, oid, src, tr, te, sw, wd)
    return P
Zraw = bank.zs_features("zs") / 3.0
Zc = bank.zs_features("zs2") / 3.0
def zsm(Z, scale):
    def make(tr):
        trr = tr[real[tr]]
        Zs = (Z - Z[trr].mean(0)) / (Z[trr].std(0) + 1e-6)
        return (Zs.mean(1, keepdims=True) * scale).astype(np.float32)
    return make
def combo(Z, scale, base_spec):
    B = bank.features(base_spec)
    def make(tr):
        return np.concatenate([B, zsm(Z, scale)(tr)], 1)
    return make
for sw in (0.5,):
    report(looo_fn(zsm(Zraw, 1), sw, 1e-2), y, oid, real, "zs_mean clipL (1-D)")
    report(looo_fn(zsm(Zc, 1), sw, 1e-2), y, oid, real, "zs_mean clipL+siglip (1-D)")
    for sc in (1, 3, 10):
        for wd in (1e-2, 1e-1):
            report(looo_fn(combo(Zraw, sc, "clipL_c14"), sw, wd), y, oid, real, f"clipL_c14 + zs_mean*{sc} wd={wd}")
    report(looo_fn(combo(Zraw, 3, "clipL_c14+zs"), sw, 1e-2), y, oid, real, "clipL_c14 + zs + zs_mean*3")
    report(looo_fn(combo(Zraw, 10, "zs"), sw, 1e-1), y, oid, real, "zs + zs_mean*10")
# per-object output for zs_mean
P = looo_fn(zsm(Zraw, 1), 0.5, 1e-2)
print({o: round(float(P[idx][oid[idx] == o, 0].mean()), 2) for o in objs})
