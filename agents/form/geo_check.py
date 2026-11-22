import numpy as np
from sklearn.metrics import roc_auc_score
from form_cv import load_bank
from form_model import GEO_KEYS
bank, y, oid, src, meta = load_bank()
real = src == "real_train"; idx = np.where(real)[0]; objs = sorted(set(oid[real]))
G = bank.emb["geo"]
extra = {"h_med/hmax": G[:, GEO_KEYS.index("h_med")] / (G[:, GEO_KEYS.index("hmax")] + 1e-3),
         "rough/hmax": G[:, GEO_KEYS.index("rough")] / (G[:, GEO_KEYS.index("hmax")] + 1e-3),
         "vol/(foot*hmax)": G[:, GEO_KEYS.index("vol")] / (G[:, GEO_KEYS.index("foot")] * G[:, GEO_KEYS.index("hmax")] * 1e2 + 1e-3),
         "ext2/ext1": G[:, GEO_KEYS.index("ext2")] / (G[:, GEO_KEYS.index("ext1")] + 1e-3)}
for k in GEO_KEYS + list(extra):
    v = G[:, GEO_KEYS.index(k)] if k in GEO_KEYS else extra[k]
    auc = roc_auc_score(y[idx], v[idx])
    per = {o: float(np.median(v[idx][oid[idx] == o])) for o in objs}
    order = sorted(per, key=per.get)
    print(f"{k:16s} AUC {auc:.3f}  {' '.join(f'{o}:{per[o]:.3g}' for o in order)}"[:400])
