"""Per-family / per-view zero-shot diagnostics: object-level AUC and where the hard objects rank."""
import numpy as np
from sklearn.metrics import roc_auc_score
from bank import gt_bank
from prompts import FAMILIES

bank, y, oid, src, meta = gt_bank()
real = src == "real_train"
idx = np.where(real)[0]
objs = sorted(set(oid[real]))
yo = np.array([o.startswith("s") for o in objs])
hard = ["b06", "b07", "b01", "b18", "b19", "s19", "s48", "s11", "s12"]
print(f"{'family':12s} {'view':4s} {'AUCfr':6s} {'AUCobj':6s} | " + " ".join(f"{h:>5s}" for h in hard) + "   (rank among 40, 0=most bulk-like)")
for fam in FAMILIES:
    pb, ps = FAMILIES[fam]
    tb = bank.T[[bank.pi[p] for p in pb]].mean(0)
    ts = bank.T[[bank.pi[p] for p in ps]].mean(0)
    tb, ts = tb / np.linalg.norm(tb), ts / np.linalg.norm(ts)
    for v in ("c14", "m10", "m14"):
        E = bank.emb[f"clipL_{v}"]
        d = E @ ts - E @ tb
        per = {o: float(d[idx][oid[idx] == o].mean()) for o in objs}
        order = sorted(objs, key=per.get)
        po = np.array([per[o] for o in objs])
        print(f"{fam:12s} {v:4s} {roc_auc_score(y[idx], d[idx]):.3f}  {roc_auc_score(yo, po):.3f}  | " + " ".join(f"{order.index(h):5d}" for h in hard))
# single prompts: best bulk-detectors for b07/b06 among solids
print("\nsingle prompts (c14): AUC(bulk) object-level and rank of b06,b07")
E = bank.emb["clipL_c14"]
S = E @ bank.T.T
rows = []
for j, p in enumerate(bank.prompts):
    per = np.array([S[idx][oid[idx] == o, j].mean() for o in objs])
    rows.append((roc_auc_score(~yo, per), p, sorted(objs, key=lambda o: -per[objs.index(o)]).index("b06"), sorted(objs, key=lambda o: -per[objs.index(o)]).index("b07")))
for r in sorted(rows, reverse=True)[:25]:
    print(f"  {r[0]:.3f} b06@{r[2]:2d} b07@{r[3]:2d}  {r[1]}")
# depth / sam / dstat per-object medians for hard objects
for blk, keys in (("depth", list(np.load("/home/cleaner/Cleaner/agents2/form/cache/gt_v2.npz")["depth_keys"])),
                  ("sam", list(np.load("/home/cleaner/Cleaner/agents2/form/cache/gt_v2.npz")["sam_keys"])),
                  ("dino_stats", list(np.load("/home/cleaner/Cleaner/agents2/form/cache/gt_v2.npz")["dino_keys"]))):
    X = bank.emb[blk]
    print(f"\n{blk}: {'key':14s} AUCobj | " + " ".join(f"{h:>6s}" for h in hard) + " | bulk_med solid_med")
    for j, k in enumerate(keys):
        per = {o: float(np.median(X[idx][oid[idx] == o, j])) for o in objs}
        po = np.array([per[o] for o in objs])
        print(f"  {k:14s} {roc_auc_score(yo, po):.3f}  | " + " ".join(f"{per[h]:6.3g}" for h in hard) + f" | {np.median(po[~yo]):7.3g} {np.median(po[yo]):7.3g}")
