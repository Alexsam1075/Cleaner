"""Shared evaluation helpers for the form agent v2: cached fold detections, official score.py, box filters."""
import json
from collections import Counter, defaultdict

import numpy as np
import pandas as pd

from common import D, W, W0, box_iou, load_split  # noqa: F401 (sets sys.path)
import score as S  # noqa: E402
from form_model import GEO_KEYS, FeatureBank

OUT = W / "out"
OUT.mkdir(exist_ok=True)


def det_rows(fold):
    return json.load(open(W0 / "cache" / f"fold{fold}_dets.json"))


def det_bank(fold, text):
    rows = det_rows(fold)
    E = dict(np.load(W0 / "cache" / f"fold{fold}_dets.npz"))
    emb = {k: v for k, v in E.items() if not k.startswith("text_")}
    emb["geo"] = np.array([[r["geo"][k] for k in GEO_KEYS] for r in rows], np.float32).reshape(len(rows), len(GEO_KEYS))
    return rows, FeatureBank(emb, text)


_LAB = None


def labels():
    global _LAB
    if _LAB is None:
        _LAB = json.load(open(D / "labels/real_train.json"))
    return _LAB


def solution_for(frames):
    lab = labels()
    keep = [im for im in lab["images"] if im["file_name"].split("/")[-2] in frames]
    ids = {im["id"] for im in keep}
    data = {"categories": lab["categories"], "images": keep, "annotations": [a for a in lab["annotations"] if a["image_id"] in ids]}
    p = OUT / f"labels_{abs(hash(tuple(sorted(frames)))) % 10**8}.json"
    p.write_text(json.dumps(data))
    return S.labels_to_solution(p)


def serialize(objs):
    if not objs:
        return "none"
    return ";".join(f"{o['box'][0]:.1f} {o['box'][1]:.1f} {o['box'][2]:.1f} {o['box'][3]:.1f} {o['score']:.5f} {o['form']} plastic 100 100 1" for o in objs[:30])


def containment(inner, outer):
    w = min(inner[2], outer[2]) - max(inner[0], outer[0])
    h = min(inner[3], outer[3]) - max(inner[1], outer[1])
    if w <= 0 or h <= 0:
        return 0.0
    return w * h / ((inner[2] - inner[0]) * (inner[3] - inner[1]))


def touches_border(b, margin=2, w=1280, h=720):
    return b[0] <= margin or b[1] <= margin or b[2] >= w - margin or b[3] >= h - margin


def suppress(dets, nms=0.6, contain=None, key="score"):
    """Greedy class-agnostic suppression: drop a box if IoU >= nms with a kept higher-score box,
    or if >= `contain` of its area lies inside a kept higher-score box."""
    keep = []
    for d in sorted(dets, key=lambda d: -d[key]):
        ok = True
        for k in keep:
            if box_iou(d["box"], k["box"]) >= nms or (contain is not None and containment(d["box"], k["box"]) >= contain):
                ok = False
                break
        if ok:
            keep.append(d)
    return keep


def evaluate(per_frame, frames, sol):
    """per_frame: {frame: [ {box, score, form} ]}. Returns dict with AP per class, D, confusion of matched boxes and FP count."""
    sub = pd.DataFrame({"image_id": frames, "prediction": [serialize(per_frame.get(f, [])) for f in frames]})
    gt = {str(i): S._parse(c, False, str(i)) for i, c in zip(sol.image_id, sol.objects)}
    pr = {str(i): S._parse(c, True, str(i)) for i, c in zip(sub.image_id, sub.prediction)}
    ap = {f: S._average_precision(gt, pr, f) for f in ("bulk", "solid")}
    conf = Counter()
    n_fp = 0
    for img, objs in gt.items():
        preds = pr.get(img, [])
        used = [False] * len(objs)
        for p in sorted(preds, key=lambda x: -x["score"]):
            best, bj = 0.0, -1
            for j, g in enumerate(objs):
                if used[j]:
                    continue
                v = S._iou(p["box"], g["box"])
                if v > best:
                    best, bj = v, j
            if best >= 0.5:
                used[bj] = True
                conf[(objs[bj]["form"], p["form"])] += 1
            else:
                n_fp += 1
    return {"AP_bulk": ap["bulk"], "AP_solid": ap["solid"], "D": float(np.mean([ap["bulk"], ap["solid"]])),
            "confusion": {f"{a}->{b}": v for (a, b), v in sorted(conf.items())}, "n_fp": n_fp,
            "n_pred": sum(len(v) for v in pr.values())}, sub


def fmt(res):
    return (f"AP_bulk {res['AP_bulk']:.3f} AP_solid {res['AP_solid']:.3f} D {res['D']:.3f} FP {res['n_fp']:3d} npred {res['n_pred']:3d} "
            f"conf {res['confusion']}")
