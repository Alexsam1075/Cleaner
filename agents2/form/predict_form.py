"""FormClassifier: public interface for the integrator.

    fc = FormClassifier().fit(train_object_ids)            # GT-mask features of real+synth annotations of those objects
    p_bulk = fc.predict(embs_by_tag_view, geo_dict, depth, mask, plane, rgb_bgr=..., bbox=...)   # one detection
    p_bulk = fc.predict_bank(bank)                          # many detections (Bank from bank.py)
    dets = pool_series(dets, embs)                          # optional: average form logits over a series of the same object
    dets = rank_and_filter(detections)                      # one box per object, score = conf * p(form)^gamma

Detection dict for rank_and_filter: {"box": [x1,y1,x2,y2], "conf": float, "p_bulk": float, ...}; returns the kept
detections with added fields "form" and "score", sorted by score. Hyperparameters chosen in form_pipeline.py (see REPORT.md)."""
import json
from pathlib import Path

import numpy as np

from common import W
from bank import Bank, gt_bank, load_prompts
from form_model2 import GRID, FormModel2, logit, sigmoid
from evalkit import suppress, touches_border

DEFAULTS = {"gamma": 1.0, "nms": 0.6, "contain": 0.8, "border": True, "conf": 0.08, "topk": 3, "pool_sim": 0.9, "pool_gap": 10}


class FormClassifier:
    def __init__(self, topk=DEFAULTS["topk"], grid=GRID, tune_thr=True):
        self.topk, self.grid, self.tune_thr = topk, grid, tune_thr
        self._gt = None
        self._ext = None

    def _ensure(self):
        if self._gt is None:
            self._gt = gt_bank()
            self.T, self.prompts = load_prompts()

    def fit(self, train_object_ids):
        self._ensure()
        bank, y, oid, src, meta = self._gt
        self.model = FormModel2(bank, y, oid, src, grid=self.grid, topk=self.topk,
                                thr_grid=np.round(np.arange(0.3, 0.71, 0.05), 2) if self.tune_thr else (0.5,)).fit(sorted(train_object_ids))
        self.thr = self.model.thr
        return self

    def _ext_models(self):
        if self._ext is None:
            from extract import Clip, Dino, Sam  # noqa: F401 (Clip not needed: prompts cached)
            self._ext = (Dino(), Sam())
        return self._ext

    def _calibrate(self, p_solid):
        """Monotone map so that the tuned threshold lands on 0.5; returns p_bulk."""
        t = self.thr
        p = np.where(p_solid > t, 0.5 + 0.5 * (p_solid - t) / (1 - t), 0.5 * p_solid / t)
        return 1.0 - p

    def predict_bank(self, bank):
        return self._calibrate(self.model.predict_bank(bank))

    def predict(self, embs, geo, depth, mask, plane, rgb_bgr=None, bbox=None):
        """embs: {(tag, view): (d,)} with at least ("clipL", "c14"/"m10"/"m14"); geo: dict of audit geometry keys;
        depth (HxW uint16 mm), mask (HxW bool), plane (4,), rgb_bgr (HxWx3) + bbox [x,y,w,h] for DINO/FastSAM features.
        Returns p_bulk (float)."""
        from depth_feats import depth_features, DEPTH_KEYS
        from form_model import GEO_KEYS
        from mat_features2 import crop
        import cv2
        dino, sam = self._ext_models()
        if bbox is None:
            ys, xs = np.nonzero(mask)
            bbox = [xs.min(), ys.min(), xs.max() - xs.min() + 1, ys.max() - ys.min() + 1]
        rgb = cv2.cvtColor(rgb_bgr, cv2.COLOR_BGR2RGB)
        c, mm = crop(rgb, mask, bbox, 1.2, False, size=448)
        pool, stats = dino(c[None], (mm > 0)[None])
        f = depth_features(depth, mask, plane)
        emb = {f"clipL_{v}": np.asarray(embs[("clipL", v)], np.float32)[None] for v in ("c14", "m10", "m14")}
        emb["geo"] = np.array([[geo[k] for k in GEO_KEYS]], np.float32)
        emb["depth"] = np.array([[f[k] for k in DEPTH_KEYS]], np.float32)
        emb["dino_pool"], emb["dino_stats"] = pool, stats
        emb["sam"] = np.array([sam(rgb_bgr, mask, bbox)], np.float32)
        return float(self.predict_bank(Bank(emb, self.T, self.prompts))[0])


def rank_and_filter(detections, gamma=DEFAULTS["gamma"], nms=DEFAULTS["nms"], contain=DEFAULTS["contain"], border=DEFAULTS["border"],
                    conf=DEFAULTS["conf"], max_det=30):
    """One box per object: drop low conf / border boxes, score = conf * max(p,1-p)^gamma, class-agnostic NMS + containment suppression."""
    cand = []
    for d in detections:
        if d["conf"] < conf or (border and touches_border(d["box"])):
            continue
        p = float(d["p_bulk"])
        pm = max(p, 1 - p)
        cand.append(dict(d, form="bulk" if p >= 0.5 else "solid", score=d["conf"] * pm ** gamma))
    return suppress(cand, nms=nms, contain=contain, key="score")[:max_det]


def frame_no(frame):
    return int(str(frame).split("_")[-1])


def pool_series(detections, embs, sim=DEFAULTS["pool_sim"], gap=DEFAULTS["pool_gap"], min_conf=0.08):
    """Object memory for form: detections (dicts with frame, conf, p_bulk) + L2-normalised CLIP-L c14 embeddings (N, d).
    Greedy single-linkage: a detection joins the cluster of its most similar earlier detection if cos >= sim and the
    frame gap <= gap. p_bulk of every member is replaced by the confidence-weighted mean logit of its cluster.
    Returns a new list (same order) with pooled p_bulk and field "cluster"."""
    E = np.asarray(embs, np.float32)
    E = E / (np.linalg.norm(E, axis=1, keepdims=True) + 1e-8)
    order = sorted(range(len(detections)), key=lambda i: frame_no(detections[i]["frame"]))
    cid = [-1] * len(detections)
    nxt = 0
    for i in order:
        if detections[i]["conf"] < min_conf:
            cid[i] = nxt
            nxt += 1
            continue
        fi = frame_no(detections[i]["frame"])
        best = None
        for j in order:
            if j == i or cid[j] < 0 or detections[j]["conf"] < min_conf:
                continue
            if abs(fi - frame_no(detections[j]["frame"])) > gap:
                continue
            s = float(E[i] @ E[j])
            if s >= sim and (best is None or s > best[0]):
                best = (s, cid[j])
        if best is None:
            cid[i] = nxt
            nxt += 1
        else:
            cid[i] = best[1]
    groups = {}
    for k, c in enumerate(cid):
        groups.setdefault(c, []).append(k)
    out = []
    for k, d in enumerate(detections):
        g = groups[cid[k]]
        w = np.array([detections[j]["conf"] for j in g])
        z = np.array([logit(1 - float(detections[j]["p_bulk"])) for j in g])
        out.append(dict(d, p_bulk=float(1 - sigmoid((w * z).sum() / w.sum())), cluster=int(cid[k])))
    return out
