"""Exp13: robustness to wrong input material. Variants: material only as offset (no slope), noise-augmented
training (soft material labels), probability-weighted prediction (as the pipeline does with material_probs).
Reported clean / 8% noisy, frame-level and series-median."""
import sys
import warnings

warnings.filterwarnings("ignore")
sys.path.insert(0, "/home/cleaner/Cleaner/agents2/volume")
import math

import numpy as np
import pandas as pd

from common import *
from vmodels import *

df = load_rows()
y = np.log(df.volume_ml.values)
ids = df.object_id.values
fnum = df.image_id.str[2:].astype(int).values
ref = ["columns_ml", "parts_ml", "hull_ml", "footprint_ml", "gi_env_ml"]
rng = np.random.default_rng(5)
noisy = df.copy()
flip = rng.random(len(df)) < 0.08
noisy.loc[flip, "material"] = [rng.choice([m for m in MATS if m != x]) for x in df.material[flip]]


def series_ids(gap=3):
    sid = np.zeros(len(df), int)
    k = 0
    for o in sorted(set(ids)):
        ii = np.where(ids == o)[0]
        ii = ii[np.argsort(fnum[ii])]
        sid[ii[0]] = k
        for a, b in zip(ii[:-1], ii[1:]):
            if fnum[b] - fnum[a] > gap:
                k += 1
            sid[b] = k
        k += 1
    return sid


SID = series_ids(3)


def smed(pred):
    out = pred.copy()
    for s in set(SID):
        out[SID == s] = np.median(pred[SID == s])
    return out


def design2(d, cols, inter, knots, inter_mat=True):
    X = np.stack([np.log(np.maximum(d[c].to_numpy(float), 1.0)) for c in cols], 1)
    form = (d.form == "bulk").to_numpy().astype(float)[:, None]
    mat = np.stack([(d.material == m).to_numpy() for m in MATS], 1).astype(float)
    main = X[:, cols.index(inter)][:, None]
    parts = [X, form, mat, form * mat, main * form]
    if inter_mat:
        parts.append(main * mat)
    parts += [np.maximum(0, main - math.log(t)) for t in knots]
    return np.hstack(parts)


def show(name, p, pn):
    report(f"{name} | frame clean", p, df)
    report(f"{name} | frame noisy8%", pn, df)
    report(f"{name} | series clean", smed(p), df)
    report(f"{name} | series noisy8%", smed(pn), df)


for im in (True, False):
    X = design2(df, ref, "gi_env_ml", (50, 300), im)
    Xn = design2(noisy, ref, "gi_env_ml", (50, 300), im)
    show(f"IRLS inter_mat={im}", nested_looo(X, y, ids, fit_irls, IRLS_GRID), nested_looo(X, y, ids, fit_irls, IRLS_GRID, Xeval=Xn))

# noise-augmented training: each row duplicated with the 5 wrong materials at weight q/5
X = design2(df, ref, "gi_env_ml", (50, 300))
Xn = design2(noisy, ref, "gi_env_ml", (50, 300))
w0 = w_of(ids)


def augmented(q):
    rows, ws, ys, idl = [X], [w0 * (1 - q)], [y], [ids]
    for m in MATS:
        d2 = df.copy()
        sel = (d2.material != m).to_numpy()
        d2["material"] = m
        rows.append(design2(d2, ref, "gi_env_ml", (50, 300))[sel])
        ws.append(w0[sel] * q / 5)
        ys.append(y[sel])
        idl.append(ids[sel])
    return np.vstack(rows), np.concatenate(ws), np.concatenate(ys), np.concatenate(idl)


for q in (0.03, 0.08):
    Xa, wa, ya, ia = augmented(q)
    p, pn = np.zeros(len(y)), np.zeros(len(y))
    for o in sorted(set(ids)):
        tr = ia != o
        m = irls(Xa[tr], ya[tr], wa[tr], 10.0, floor=0.2)
        p[ids == o] = m(X[ids == o])
        pn[ids == o] = m(Xn[ids == o])
    show(f"IRLS a10 noise-aug q={q}", p, pn)
p, pn = np.zeros(len(y)), np.zeros(len(y))
for o in sorted(set(ids)):
    tr = ids != o
    m = irls(X[tr], y[tr], w0[tr], 10.0, floor=0.2)
    p[ids == o] = m(X[ids == o])
    pn[ids == o] = m(Xn[ids == o])
show("IRLS a10 fixed (no aug)", p, pn)

# probability-weighted prediction: p(true)=0.8, rest shared (simulating a calibrated classifier)
Xm = {m: design2(df.assign(material=m), ref, "gi_env_ml", (50, 300)) for m in MATS}
p, pw = np.zeros(len(y)), np.zeros(len(y))
for o in sorted(set(ids)):
    tr = ids != o
    m = irls(X[tr], y[tr], w0[tr], 10.0, floor=0.2)
    te = ids == o
    p[te] = m(X[te])
    acc = np.zeros(te.sum())
    for mt in MATS:
        pr = np.where(noisy.material.values[te] == mt, 0.8, 0.04)
        acc += pr * m(Xm[mt][te])
    pw[te] = acc
show("IRLS a10 prob-weighted (0.8 on noisy label)", p, pw)
