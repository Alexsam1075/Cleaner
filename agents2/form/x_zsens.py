"""Zero-shot ensembles as a robust prior: standardized (sim_solid - sim_bulk) averaged over families/views, no learned weights
(only threshold and family subset chosen on training objects). Also blend with the CLIP head."""
import numpy as np
from sklearn.metrics import roc_auc_score

from bank import gt_bank
from form_cv import report
from form_model2 import head_predict
from prompts import FAMILIES

bank, y, oid, src, meta = gt_bank()
real = src == "real_train"
objs = sorted(set(oid[real]))
VIEWS = ("c14", "m10", "m14")
cols, names = [], []
for fam in FAMILIES:
    pb, ps = FAMILIES[fam]
    tb = bank.T[[bank.pi[p] for p in pb]].mean(0)
    ts = bank.T[[bank.pi[p] for p in ps]].mean(0)
    tb, ts = tb / np.linalg.norm(tb), ts / np.linalg.norm(ts)
    for v in VIEWS:
        E = bank.emb[f"clipL_{v}"]
        cols.append(E @ ts - E @ tb)
        names.append((fam, v))
Dm = np.stack(cols, 1)


def zscore(Dm, tr):
    return (Dm - Dm[tr].mean(0)) / (Dm[tr].std(0) + 1e-6)


def obj_auc(s, idx):
    o_ = oid[idx]
    objs_ = sorted(set(o_))
    po = np.array([s[idx][o_ == o].mean() for o in objs_])
    yo = np.array([y[idx][o_ == o][0] for o in objs_])
    return roc_auc_score(yo, po)


def best_thr(s, yy):
    cand = np.quantile(s, np.linspace(0.05, 0.95, 91))
    return max(cand, key=lambda t: np.mean([((s > t) == 1)[yy == 1].mean(), ((s <= t))[yy == 0].mean()]))


def to_prob(s, t, scale=1.0):
    return 1 / (1 + np.exp(-(s - t) * scale))


def looo_ens(select, label, k=6):
    P = np.zeros(len(y))
    for o in objs:
        tr = np.where((oid != o) & real)[0]
        te = np.where((oid == o) & real)[0]
        Z = zscore(Dm, np.where(oid != o)[0])
        if select == "all":
            S = list(range(Dm.shape[1]))
        elif select == "M":
            S = [i for i, (f, v) in enumerate(names) if f in ("matpile", "paperpile", "metalpile", "plasticpile", "woodpile", "stonepile")]
        elif select == "M+gen":
            S = [i for i, (f, v) in enumerate(names) if f in ("matpile", "paperpile", "metalpile", "plasticpile", "woodpile", "stonepile", "pile", "mix", "several", "parts", "count")]
        elif select == "greedy":
            # forward selection by object-level AUC on training objects
            S = []
            cur = None
            for _ in range(k):
                best = None
                for i in range(Dm.shape[1]):
                    if i in S:
                        continue
                    s = Z[:, S + [i]].mean(1)
                    a = obj_auc(s, tr)
                    if best is None or a > best[0]:
                        best = (a, i)
                if cur is not None and best[0] <= cur:
                    break
                cur = best[0]
                S.append(best[1])
        elif select == "topk":
            aucs = [obj_auc(Z[:, i], tr) for i in range(Dm.shape[1])]
            S = list(np.argsort(aucs)[::-1][:k])
        s = Z[:, S].mean(1)
        t = best_thr(s[tr], y[tr])
        P[te] = to_prob(s[te], t, 2.0)
    return report(P, y, oid, real, label)


for sel, k in (("all", 0), ("M", 0), ("M+gen", 0), ("topk", 6), ("topk", 12), ("greedy", 6), ("greedy", 10)):
    looo_ens(sel, f"zs-ens {sel} k={k}", k)

# blend: head (clipL_c14+zs) logit + lambda * ensemble z-score (M+gen)
X = bank.features("clipL_c14+zs")
S = [i for i, (f, v) in enumerate(names) if f in ("matpile", "paperpile", "metalpile", "plasticpile", "woodpile", "stonepile", "pile", "mix", "several", "parts", "count")]
for lam in (0.0, 0.5, 1.0, 2.0, 4.0):
    P = np.zeros(len(y))
    for o in objs:
        tr_all = np.where(oid != o)[0]
        tr = np.where((oid != o) & real)[0]
        te = np.where((oid == o) & real)[0]
        Z = zscore(Dm, tr_all)
        s = Z[:, S].mean(1)
        t = best_thr(s[tr], y[tr])
        ph = head_predict(X, y, oid, src, tr_all, te, 0.5, 1e-2)[:, 1]
        z = np.log(ph + 1e-6) - np.log(1 - ph + 1e-6) + lam * (s[te] - t) * 2.0
        P[te] = 1 / (1 + np.exp(-z))
    report(P, y, oid, real, f"head clipL_c14+zs + {lam} * zs-ens(M+gen)")
