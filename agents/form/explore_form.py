"""Quick LOOO exploration for FORM on cached GT-mask features (no nesting yet)."""
import json, math, sys
from collections import defaultdict
from pathlib import Path
import numpy as np, torch
sys.path.insert(0, "/home/cleaner/Cleaner/audit")
from mat_cv import fit_predict
C = Path("/home/cleaner/Cleaner/audit/cache")
W = Path("/home/cleaner/Cleaner/agents/form")
F = dict(np.load(C / "features.npz")); F.update(dict(np.load(C / "features2.npz")))
meta = json.load(open(C / "meta.json"))
oid = np.array([m["oid"] for m in meta]); src = np.array([m["src"] for m in meta])
y = np.array([0 if m["form"] == "bulk" else 1 for m in meta]); real = src == "real_train"
objs = sorted(set(oid[real]))
def l2(X): return X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-8)
def feats(spec):
    parts = []
    for k in spec.split("+"):
        if k == "geo":
            G = F["geo"]; parts.append(np.sign(G) * np.log1p(np.abs(G)))
        elif k == "photo": parts.append(F["photo"])
        else: parts.append(l2(F[k]) * math.sqrt(F[k].shape[1]) / 8)
    return np.concatenate(parts, 1).astype(np.float32)
def weights(idx, synth_w):
    cnt = defaultdict(int)
    for i in idx: cnt[(oid[i], src[i])] += 1
    w = np.array([(1.0 if src[i] == "real_train" else synth_w) / cnt[(oid[i], src[i])] for i in idx])
    per = defaultdict(float)
    for j, i in enumerate(idx): per[y[i]] += w[j]
    w = np.array([w[j] / per[y[i]] for j, i in enumerate(idx)])
    return w / w.mean()
def looo(X, synth_w, wd):
    P = np.zeros(len(y))
    for o in objs:
        tr = np.where((oid != o) & (real | (synth_w > 0)))[0]
        te = np.where((oid == o) & real)[0]
        mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-6
        P[te] = fit_predict((X - mu) / sd, y, weights(tr, synth_w), tr, te, wd=wd, n_cls=2)[:, 1]
    return P
def report(P, tag):
    idx = np.where(real)[0]; pred = (P[idx] > 0.5).astype(int)
    acc_b = (pred[y[idx] == 0] == 0).mean(); acc_s = (pred[y[idx] == 1] == 1).mean()
    wrong = sorted(set(oid[idx][pred != y[idx]]))
    print(f"{tag:60s} bulk {acc_b:.3f} solid {acc_s:.3f} bal {(acc_b+acc_s)/2:.3f} frame_acc {(pred==y[idx]).mean():.3f} wrong_objs {wrong}", flush=True)
    return (acc_b + acc_s) / 2
if __name__ == "__main__":
    specs = sys.argv[1:] or ["clipL_c14", "clipL_m10", "clipL_m14", "clipL_c20", "siglip_c14", "siglip_m10", "dinoL_c14", "dinoL_m10", "dinoL_c14_mpool", "geo", "photo", "geo+photo", "clipL_c14+geo", "clipL_c14+siglip_c14", "clipL_c14+dinoL_m10"]
    for spec in specs:
        X = feats(spec)
        for sw in (0.0, 0.5):
            for wd in (1e-2, 1e-1):
                P = looo(X, sw, wd)
                report(P, f"{spec} sw={sw} wd={wd}")
