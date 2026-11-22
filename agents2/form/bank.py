"""Feature bank v2 for form (bulk/solid): CLIP-L crop embeddings (cached by the previous agent), zero-shot prompt-family
similarity differences, DINOv2 patch statistics, depth height-map features, FastSAM part statistics."""
import json
import math

import numpy as np

from common import C, W, W0
from prompts import FAMILIES, FAMILY_GROUPS
from form_model import GEO_KEYS

ZS_VIEWS = ("c14", "m10", "m14")


def l2n(X):
    return X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-8)


def slog(v):
    return np.sign(v) * np.log1p(np.abs(v))


class Bank:
    """emb: dict name -> (N, d). Required: clipL_{c14,m10,m14}; optional: dino_pool, dino_stats, depth, sam, geo."""

    def __init__(self, emb, T, prompts):
        self.emb = emb
        self.T, self.prompts = T, list(prompts)
        self.pi = {p: i for i, p in enumerate(self.prompts)}
        self._cache = {}

    def n(self):
        return len(self.emb["clipL_c14"])

    def zs(self, group, views=ZS_VIEWS, scale=3.0):
        """(N, k): per family and view, mean sim(solid prompts) - mean sim(bulk prompts)."""
        key = ("zs", group, views)
        if key not in self._cache:
            cols = []
            for fam in FAMILY_GROUPS[group]:
                pb, ps = FAMILIES[fam]
                tb = self.T[[self.pi[p] for p in pb]].mean(0)
                ts = self.T[[self.pi[p] for p in ps]].mean(0)
                tb, ts = tb / np.linalg.norm(tb), ts / np.linalg.norm(ts)
                for v in views:
                    E = self.emb[f"clipL_{v}"]
                    cols.append(E @ ts - E @ tb)
            self._cache[key] = (np.stack(cols, 1) * scale).astype(np.float32)
        return self._cache[key]

    def zs_matrix(self, group="zsAll", views=ZS_VIEWS):
        """Raw (N, families*views) matrix of sim(solid) - sim(bulk) (unscaled) for the ensemble prior."""
        return self.zs(group, views, scale=1.0)

    def block(self, k):
        if k.startswith("zs"):
            return self.zs(k)
        if k == "depth":
            X = self.emb["depth"].copy()
            return slog(X).astype(np.float32)
        if k == "dstat":
            return slog(self.emb["dino_stats"]).astype(np.float32)
        if k == "sam":
            return slog(self.emb["sam"]).astype(np.float32)
        if k == "geo":
            return slog(self.emb["geo"]).astype(np.float32)
        if k == "dino":
            E = self.emb["dino_pool"]
            return (l2n(E) * math.sqrt(E.shape[1]) / 8).astype(np.float32)
        E = self.emb[k]
        return (l2n(E) * math.sqrt(E.shape[1]) / 8).astype(np.float32)

    def features(self, spec):
        return np.concatenate([self.block(k) for k in spec.split("+")], 1).astype(np.float32)


def load_prompts():
    Z = np.load(W / "cache/prompts.npz")
    return Z["T"], list(Z["prompts"])


def gt_bank():
    """All GT annotations (real_train + synth_train), order of audit/cache/meta.json."""
    meta = json.load(open(C / "meta.json"))
    F2 = np.load(C / "features2.npz")
    F = np.load(C / "features.npz")
    V2 = np.load(W / "cache/gt_v2.npz")
    emb = {k: F2[k] for k in ("clipL_c14", "clipL_m10", "clipL_m14", "clipL_c20", "siglip_c14", "dinoL_c14_mpool", "dinoL_m10")}
    emb["geo"] = np.stack([F["geo"][:, list(F["geo_keys"]).index(k)] for k in GEO_KEYS], 1)
    for k in ("dino_pool", "dino_stats", "depth", "sam"):
        emb[k] = V2[k]
    T, prompts = load_prompts()
    bank = Bank(emb, T, prompts)
    oid = np.array([m["oid"] for m in meta])
    src = np.array([m["src"] for m in meta])
    y = np.array([0 if m["form"] == "bulk" else 1 for m in meta])
    return bank, y, oid, src, meta


def det_bank(fold):
    """Cached detections of fold k (agents/form/cache) + v2 features."""
    rows = json.load(open(W0 / "cache" / f"fold{fold}_dets.json"))
    E = dict(np.load(W0 / "cache" / f"fold{fold}_dets.npz"))
    emb = {k: v for k, v in E.items() if not k.startswith("text_")}
    emb["geo"] = np.array([[r["geo"][k] for k in GEO_KEYS] for r in rows], np.float32).reshape(len(rows), len(GEO_KEYS))
    V2 = np.load(W / f"cache/fold{fold}_v2.npz")
    for k in ("dino_pool", "dino_stats", "depth", "sam"):
        emb[k] = V2[k]
    T, prompts = load_prompts()
    return rows, Bank(emb, T, prompts)
