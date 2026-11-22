"""Form (bulk vs solid) model: linear head on CLIP-L embeddings (+ optional extras) blended with CLIP/SigLIP zero-shot.
All hyperparameters (feature spec, synth weight, weight decay, zero-shot set, blend weight) are chosen by grouped inner CV
over the given training objects only."""
import math
from collections import defaultdict

import numpy as np
import torch

from common import PROMPT_SETS, l2n
from mat_cv import fit_predict  # GPU logreg (audit)

GEO_KEYS = ["area_px", "ext1", "ext2", "foot", "h95", "h_med", "hmax", "holes_edge", "holes_inner", "rough", "surf", "valid", "vol", "vol_thr", "zmed"]
GRID = {
    "spec": ["clipL_c14", "clipL_c14+zs", "zs", "clipL_c14+clipL_m10+zs", "clipL_c14+geo+zs", "clipL_c14+zs2"],
    "sw": [0.5, 1.0],
    "wd": [1e-2, 1e-1],
}
ZS_SETS = ["none"]
W_GRID = []
TEMP = 0.01
ALL_SETS = ("pile", "many", "count", "mix", "multi", "crumpled", "parts", "trash", "matpile", "several")
ZS_SPECS = {"zs": (("clipL",), ALL_SETS[:8], ("c14", "m10", "m14")),
            "zs2": (("clipL", "siglip"), ALL_SETS[:8], ("c14", "m10", "m14")),
            "zs3": (("clipL",), ALL_SETS, ("c14", "m10", "m14")),
            "zs3c": (("clipL",), ALL_SETS, ("c14",)),
            "zsn": (("clipL",), ("matpile", "several"), ("c14", "m14"))}


def geo_vec(g):
    v = np.array([g[k] for k in GEO_KEYS], np.float32)
    return np.sign(v) * np.log1p(np.abs(v))


class FeatureBank:
    """Holds embeddings per sample: dict name -> (N, d). `geo` is (N, 15) raw geometry; `text` dict (tag, set) -> (2, d)."""

    def __init__(self, emb, text):
        self.emb, self.text = emb, text

    def zs_features(self, name, scale=3.0):
        """Zero-shot sim differences (solid - bulk) for each (model, prompt set, view) -> (N, k)."""
        tags, sets, views = ZS_SPECS[name]
        cols = []
        for t in tags:
            for s in sets:
                for v in views:
                    S = self.emb[f"{t}_{v}"] @ self.text[(t, s)].T
                    cols.append(S[:, 1] - S[:, 0])
        return (np.stack(cols, 1) * scale).astype(np.float32)

    def features(self, spec):
        parts = []
        for k in spec.split("+"):
            if k == "geo":
                parts.append(np.sign(self.emb["geo"]) * np.log1p(np.abs(self.emb["geo"])))
            elif k in ZS_SPECS:
                parts.append(self.zs_features(k))
            else:
                parts.append(l2n(self.emb[k]) * math.sqrt(self.emb[k].shape[1]) / 8)
        return np.concatenate(parts, 1).astype(np.float32)

    def zs_logits(self, zs):
        """Returns (N, 2) averaged zero-shot cosine sims (bulk, solid) or None."""
        if zs == "none":
            return None
        tags, sets, views = zs.split(":")
        L = [self.emb[f"{t}_{v}"] @ self.text[(t, s)].T for t in tags.split("+") for s in sets.split("+") for v in views.split("+")]
        return np.mean(L, 0)


def zs_prob(L, temp=TEMP):
    z = L / temp
    z = z - z.max(1, keepdims=True)
    p = np.exp(z)
    return p / p.sum(1, keepdims=True)


def blend(p_head, Lz, w):
    if Lz is None or w == 0:
        return p_head
    pz = zs_prob(Lz)
    z = (1 - w) * np.log(p_head + 1e-9) + w * np.log(pz + 1e-9)
    z = z - z.max(1, keepdims=True)
    p = np.exp(z)
    return p / p.sum(1, keepdims=True)


def sample_weights(idx, oid, src, y, synth_w):
    cnt = defaultdict(int)
    for i in idx:
        cnt[(oid[i], src[i])] += 1
    w = np.array([(1.0 if src[i] == "real_train" else synth_w) / cnt[(oid[i], src[i])] for i in idx])
    per = defaultdict(float)
    for j, i in enumerate(idx):
        per[y[i]] += w[j]
    w = np.array([w[j] / per[y[i]] for j, i in enumerate(idx)])
    return w / w.mean()


def bal_acc(p, y):
    pred = (p[:, 1] > 0.5).astype(int)
    return float(np.mean([(pred[y == c] == c).mean() for c in (0, 1) if (y == c).any()]))


def fit_head(X, y, oid, src, tr, sw, wd):
    mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-6
    Xs = (X - mu) / sd
    w = sample_weights(tr, oid, src, y, sw)
    return mu, sd, w


def head_predict(X, y, oid, src, tr, te, sw, wd):
    mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-6
    Xs = (X - mu) / sd
    return fit_predict(Xs, y, sample_weights(tr, oid, src, y, sw), tr, te, wd=wd, n_cls=2)


class FormModel:
    """Train on a set of object ids (real + synth annotations of those objects), select hyperparameters by inner grouped CV."""

    def __init__(self, bank, y, oid, src, grid=GRID, zs_sets=ZS_SETS, w_grid=W_GRID, inner_folds=5, seed=0, verbose=False):
        self.bank, self.y, self.oid, self.src = bank, y, oid, src
        self.grid, self.zs_sets, self.w_grid, self.inner_folds, self.seed, self.verbose = grid, zs_sets, w_grid, inner_folds, seed, verbose

    def _idx(self, objs, sw):
        real = self.src == "real_train"
        return np.where(np.isin(self.oid, list(objs)) & (real | (sw > 0)))[0]

    def fit(self, train_objs):
        train_objs = sorted(train_objs)
        real = self.src == "real_train"
        rs = np.random.default_rng(self.seed)
        groups = np.array_split(rs.permutation(train_objs), self.inner_folds)
        tr_real = np.where(np.isin(self.oid, train_objs) & real)[0]
        results = {}
        for spec in self.grid["spec"]:
            X = self.bank.features(spec)
            for sw in self.grid["sw"]:
                for wd in self.grid["wd"]:
                    Pin = np.zeros((len(self.y), 2))
                    for g in groups:
                        rest = [o for o in train_objs if o not in set(g)]
                        a = self._idx(rest, sw)
                        b = np.where(np.isin(self.oid, g) & real)[0]
                        Pin[b] = head_predict(X, self.y, self.oid, self.src, a, b, sw, wd)
                    for zs in self.zs_sets:
                        Lz = self.bank.zs_logits(zs)
                        for w in ([0.0] if zs == "none" else self.w_grid):
                            P = blend(Pin[tr_real], None if Lz is None else Lz[tr_real], w)
                            # criterion: balanced accuracy, tie-break by mean log-likelihood
                            ll = float(np.mean(np.log(P[np.arange(len(tr_real)), self.y[tr_real]] + 1e-9)))
                            results[(spec, sw, wd, zs, w)] = (bal_acc(P, self.y[tr_real]), ll)
        best = max(results, key=lambda k: results[k])
        self.cfg = dict(zip(("spec", "sw", "wd", "zs", "w"), best))
        self.inner_score = results[best]
        self.inner_results = results
        if self.verbose:
            print("  selected", self.cfg, "inner bal_acc %.3f ll %.3f" % results[best])
        self.tr = self._idx(train_objs, self.cfg["sw"])
        self.X = self.bank.features(self.cfg["spec"])
        return self

    def predict_bank(self, bank_te):
        """Predict P(bulk, solid) for every sample of another FeatureBank (same keys)."""
        cfg = self.cfg
        Xte = bank_te.features(cfg["spec"])
        Xall = np.concatenate([self.X, Xte])
        yall = np.concatenate([self.y, np.zeros(len(Xte), int)])
        oid_all = np.concatenate([self.oid, np.array(["_te"] * len(Xte))])
        src_all = np.concatenate([self.src, np.array(["_te"] * len(Xte))])
        te = np.arange(len(self.X), len(Xall))
        P = head_predict(Xall, yall, oid_all, src_all, self.tr, te, cfg["sw"], cfg["wd"])
        return blend(P, bank_te.zs_logits(cfg["zs"]), cfg["w"])

    def predict_idx(self, te):
        """Predict for indices of the training bank (held-out objects)."""
        P = head_predict(self.X, self.y, self.oid, self.src, self.tr, te, self.cfg["sw"], self.cfg["wd"])
        Lz = self.bank.zs_logits(self.cfg["zs"])
        return blend(P, None if Lz is None else Lz[te], self.cfg["w"])
