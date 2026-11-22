"""Классификация материала: честный leave-one-object-out (LOOO) по 40 физическим предметам.
Данные: только real_train + synth_train (папка real/test не используется).
Синтетика предмета, отложенного в фолде, из обучения исключается.
Метрика: M = macro-среднее точности по материалам на реальных кадрах (как в score.py), плюс точность по объектам.

Классификатор — L2-нормированные эмбеддинги + логистическая регрессия на torch (GPU), веса: каждый физический
объект весит одинаково, каждый материал весит одинаково. Перебор вариантов признаков параллелен."""
import argparse
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

C = Path(__file__).resolve().parent / "cache"
MATS = ["metal", "wood", "plastic", "glass", "paper", "mineral"]
DEV = "cuda" if torch.cuda.is_available() else "cpu"


def load():
    F = dict(np.load(C / "features.npz"))
    meta = json.load(open(C / "meta.json"))
    return F, meta


def standardize(X, tr):
    mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-6
    return (X - mu) / sd


def l2(X):
    return X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-8)


def fit_predict(X, y, w, tr, te, wd=1e-2, epochs=300, n_cls=6):
    """Многоклассовая логрегрессия, полный батч, LBFGS на GPU."""
    Xt = torch.tensor(X[tr], device=DEV, dtype=torch.float32)
    yt = torch.tensor(y[tr], device=DEV)
    wt = torch.tensor(w, device=DEV, dtype=torch.float32)
    W = torch.zeros(X.shape[1], n_cls, device=DEV, requires_grad=True)
    b = torch.zeros(n_cls, device=DEV, requires_grad=True)
    opt = torch.optim.LBFGS([W, b], lr=1, max_iter=epochs, history_size=20, line_search_fn="strong_wolfe")

    def closure():
        opt.zero_grad()
        loss = (torch.nn.functional.cross_entropy(Xt @ W + b, yt, reduction="none") * wt).sum() / wt.sum() + wd * (W ** 2).sum()
        loss.backward()
        return loss

    opt.step(closure)
    with torch.no_grad():
        return torch.softmax(torch.tensor(X[te], device=DEV, dtype=torch.float32) @ W + b, 1).cpu().numpy()


def weights(idx, oid, src, mat, synth_w):
    cnt = defaultdict(int)
    for i in idx:
        cnt[(oid[i], src[i])] += 1
    w = np.array([(1.0 if src[i] == "real_train" else synth_w) / cnt[(oid[i], src[i])] for i in idx])
    per_mat = defaultdict(float)
    for j, i in enumerate(idx):
        per_mat[mat[i]] += w[j]
    w = np.array([w[j] / per_mat[mat[i]] for j, i in enumerate(idx)])
    return w / w.mean()


def looo(X, oid, src, mat, real, objs, synth_w=0.5, wd=1e-2):
    P = np.zeros((len(mat), 6))
    for o in objs:
        tr = np.where((oid != o) & (real | (synth_w > 0)))[0]
        te = np.where((oid == o) & real)[0]
        Xs = standardize(X, tr)
        P[te] = fit_predict(Xs, mat, weights(tr, oid, src, mat, synth_w), tr, te, wd=wd)
    return P


def scores(P, oid, mat, real):
    idx = np.where(real)[0]
    pred = P[idx].argmax(1)
    M = float(np.mean([(pred[mat[idx] == c] == c).mean() for c in range(6) if (mat[idx] == c).any()]))
    by = defaultdict(list)
    for i in idx:
        by[oid[i]].append(i)
    hits = defaultdict(list)
    for o, ii in by.items():
        hits[mat[ii[0]]].append(int(P[ii].mean(0).argmax() == mat[ii[0]]))
    Mobj = float(np.mean([np.mean(v) for v in hits.values()]))
    per = {MATS[c]: f"{(pred[mat[idx] == c] == c).sum()}/{(mat[idx] == c).sum()}" for c in range(6)}
    objhit = {MATS[c]: f"{sum(hits[c])}/{len(hits[c])}" for c in range(6)}
    return M, Mobj, per, objhit


def features(F, spec):
    parts = []
    for k in spec.split("+"):
        if k == "photo":
            parts.append(F["photo"])
        elif k == "geo":
            parts.append(np.log1p(np.abs(F["geo"])) * np.sign(F["geo"]))
        else:
            parts.append(l2(F[k]) * math.sqrt(F[k].shape[1]) / 8)
    return np.concatenate(parts, 1).astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--specs", nargs="+", default=None)
    a = ap.parse_args()
    F, meta = load()
    oid = np.array([m["oid"] for m in meta])
    src = np.array([m["src"] for m in meta])
    mat = np.array([MATS.index(m["mat"]) for m in meta])
    real = src == "real_train"
    objs = sorted(set(oid[real]))
    specs = a.specs or ["photo+geo", "convnext_m", "clip_b_m", "dinov2_s_m", "dinov2_b_m", "dinov2_b_c", "clip_b_c",
                        "dinov2_b_m+clip_b_m", "dinov2_b_m+clip_b_m+photo", "dinov2_b_m+clip_b_m+photo+geo",
                        "dinov2_b_m+dinov2_b_c+clip_b_m+clip_b_c+photo"]
    results = {}
    for spec in specs:
        X = features(F, spec)
        for sw in (0.0, 0.5):
            for wd in (1e-3, 1e-2):
                P = looo(X, oid, src, mat, real, objs, synth_w=sw, wd=wd)
                M, Mo, per, oh = scores(P, oid, mat, real)
                key = f"{spec} | synth_w={sw} wd={wd}"
                results[key] = {"M": M, "M_obj": Mo, "per_frame": per, "per_object": oh}
                np.save(C / f"oof_{abs(hash(key)) % 10**8}.npy", P)
                results[key]["oof"] = f"oof_{abs(hash(key)) % 10**8}.npy"
                print(f"{key:70s} M={M:.3f} M_obj={Mo:.3f} {per} obj {oh}", flush=True)
    json.dump(results, open(C / "mat_cv_results.json", "w"), indent=1)


if __name__ == "__main__":
    main()
