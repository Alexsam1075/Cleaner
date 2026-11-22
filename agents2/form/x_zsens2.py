"""Per-object view of zs-ensemble (all families) vs head, and blends with lambda chosen by inner CV (nested)."""
import numpy as np

from bank import gt_bank
from form_cv import report
from form_model2 import head_predict, bal_acc
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
GEN = [i for i, (f, v) in enumerate(names) if f not in ("robot", "loose", "shape", "plural", "stack")]


def ens_score(tr_all, S):
    Z = (Dm - Dm[tr_all].mean(0)) / (Dm[tr_all].std(0) + 1e-6)
    return Z[:, S].mean(1)


def best_thr(s, yy):
    cand = np.quantile(s, np.linspace(0.05, 0.95, 91))
    return max(cand, key=lambda t: bal_acc(s, yy, t))


X = bank.features("clipL_c14+zs")
# per-object table
ens_o, head_o = {}, {}
for o in objs:
    tr_all = np.where(oid != o)[0]
    tr = np.where((oid != o) & real)[0]
    te = np.where((oid == o) & real)[0]
    s = ens_score(tr_all, list(range(Dm.shape[1])))
    t = best_thr(s[tr], y[tr])
    ens_o[o] = (s[te] - t).mean()
    ph = head_predict(X, y, oid, src, tr_all, te, 0.5, 1e-2)[:, 1]
    head_o[o] = (np.log(ph + 1e-6) - np.log(1 - ph + 1e-6)).mean()
print("object  true   ens(z-thr)  head(logit)")
for o in objs:
    print(f"{o}  {'solid' if o.startswith('s') else 'bulk ':5s}  {ens_o[o]:+6.2f}      {head_o[o]:+6.2f}")

# nested blend: lambda and scale chosen by inner CV (grouped 5-fold over training objects)
LAMS = (0.0, 1.0, 2.0, 4.0, 8.0)
for S_name, S in (("all", list(range(Dm.shape[1]))), ("gen", GEN)):
    P = np.zeros(len(y))
    for o in objs:
        tr_objs = [x for x in objs if x != o]
        rs = np.random.default_rng(0)
        groups = np.array_split(rs.permutation(tr_objs), 5)
        inner = {lam: np.full(len(y), np.nan) for lam in LAMS}
        for g in groups:
            rest = [x for x in tr_objs if x not in set(g)]
            a_all = np.where(np.isin(oid, rest))[0]
            a = np.where(np.isin(oid, rest) & real)[0]
            b = np.where(np.isin(oid, g) & real)[0]
            s = ens_score(a_all, S)
            t = best_thr(s[a], y[a])
            ph = head_predict(X, y, oid, src, a_all, b, 0.5, 1e-2)[:, 1]
            zh = np.log(ph + 1e-6) - np.log(1 - ph + 1e-6)
            for lam in LAMS:
                inner[lam][b] = 1 / (1 + np.exp(-(zh + lam * 2.0 * (s[b] - t))))
        tr = np.where(np.isin(oid, tr_objs) & real)[0]
        best = max(LAMS, key=lambda lam: (bal_acc(inner[lam][tr], y[tr]), np.mean(np.log(np.where(y[tr] == 1, inner[lam][tr], 1 - inner[lam][tr]) + 1e-9))))
        tr_all = np.where(oid != o)[0]
        te = np.where((oid == o) & real)[0]
        s = ens_score(tr_all, S)
        t = best_thr(s[tr], y[tr])
        ph = head_predict(X, y, oid, src, tr_all, te, 0.5, 1e-2)[:, 1]
        zh = np.log(ph + 1e-6) - np.log(1 - ph + 1e-6)
        P[te] = 1 / (1 + np.exp(-(zh + best * 2.0 * (s[te] - t))))
        print(f"  held-out {o}: lam {best} p_solid {np.round(P[te], 2)}", flush=True)
    report(P, y, oid, real, f"NESTED head + lam*ens({S_name})")
