"""LOOO-проверка расширенного набора описаний (rich) против generic_syn: материал на 40 предметах, GT-маски."""
import json, math, sys
from pathlib import Path
import numpy as np, torch, torch.nn.functional as Fn
sys.path.insert(0, "/home/cleaner/Cleaner/audit")
import full_pipeline as fp
from mat_cv import fit_predict, weights
from overfit_check import M_frame, M_obj, l2n, softmax, inner_weight, MATS
C = Path("/home/cleaner/Cleaner/audit/cache")
F = dict(np.load(C / "features.npz")); F.update(dict(np.load(C / "features2.npz")))
meta = json.load(open(C / "meta.json"))
oid = np.array([m["oid"] for m in meta]); src = np.array([m["src"] for m in meta]); mat = np.array([MATS.index(m["mat"]) for m in meta])
real = src == "real_train"; idx = np.where(real)[0]; objs = sorted(set(oid[real]))
enc = fp.Encoders()
X = (l2n(F["clipL_c14"]) * math.sqrt(768 / 64)).astype(np.float32)
P_lin = np.zeros((len(mat), 6))
for o in objs:
    tr = np.where((oid != o) & real)[0]; te = np.where((oid == o) & real)[0]
    mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-6
    P_lin[te] = fit_predict((X - mu) / sd, mat, weights(tr, oid, src, mat, 0.0), tr, te)
for sname in ("generic_syn", "rich"):
    T = {tag: enc.text[(tag, sname)] for tag in fp.MODELS}
    Z = softmax(np.mean([F[f"{tag}_{v}"] @ T[tag].T for tag in fp.MODELS for v in fp.VIEWS], 0) / 0.01)
    mo, per = M_obj(Z, mat, oid, idx)
    print(f"zero-shot {sname:12s} M={M_frame(Z, mat, idx):.3f} M_obj={mo:.3f} {per}")
    Pf = np.zeros((len(mat), 6)); ws = []
    for o in objs:
        w = inner_weight(X, oid, src, mat, [q for q in objs if q != o], Z, seed=len(ws)); ws.append(w)
        te = np.where((oid == o) & real)[0]
        Pf[te] = softmax((1 - w) * np.log(P_lin[te] + 1e-9) + w * np.log(Z[te] + 1e-9))
    mo, per = M_obj(Pf, mat, oid, idx)
    print(f"fused     {sname:12s} M={M_frame(Pf, mat, idx):.3f} M_obj={mo:.3f} w≈{np.median(ws):.2f} {per}", flush=True)
