"""Why does s11 (white folded bag) go bulk under the prior? Per-family z-scores for s11, s12, b06, b07, s19 on GT masks (fold-1 training stats)."""
import numpy as np
from bank import gt_bank
from form_model2 import Prior
from prompts import FAMILIES
from common import load_split

bank, y, oid, src, meta = gt_bank()
real = src == "real_train"
M = bank.zs_matrix()
names = [(f, v) for f in FAMILIES for v in ("c14", "m10", "m14")]
sp = load_split(1)
tr_all = np.where(np.isin(oid, sp["train_object_ids"]))[0]
tr_real = np.where(np.isin(oid, sp["train_object_ids"]) & real)[0]
pr = Prior().fit(M, tr_all, y[tr_real], tr_real)
Z = (M - pr.mu) / pr.sd
print("prior t =", pr.t)
objs = ["s11", "s12", "s01", "s48", "s19", "b06", "b07", "b19", "b01", "s16", "b10"]
print(f"{'family/view':18s}" + "".join(f"{o:>7s}" for o in objs))
for j, (f, v) in enumerate(names):
    print(f"{f + '/' + v:18s}" + "".join(f"{Z[real & (oid == o), j].mean():7.2f}" for o in objs))
print(f"{'MEAN':18s}" + "".join(f"{Z[real & (oid == o)].mean():7.2f}" for o in objs))
# per family (3 views averaged)
print()
fam_means = {}
for i, f in enumerate(FAMILIES):
    fam_means[f] = {o: Z[real & (oid == o)][:, 3 * i:3 * i + 3].mean() for o in objs}
# objects-level AUC of each family on all 40 training objects (to see which families are weak)
from sklearn.metrics import roc_auc_score
allobjs = sorted(set(oid[real]))
yo = np.array([o.startswith("s") for o in allobjs])
for i, f in enumerate(FAMILIES):
    po = np.array([Z[real & (oid == o)][:, 3 * i:3 * i + 3].mean() for o in allobjs])
    print(f"{f:12s} objAUC {roc_auc_score(yo, po):.3f}  s11 {fam_means[f]['s11']:+.2f} s12 {fam_means[f]['s12']:+.2f} b06 {fam_means[f]['b06']:+.2f} b07 {fam_means[f]['b07']:+.2f}")
