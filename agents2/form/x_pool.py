"""Series pooling test: average form logits over detections of the same physical object (clusters by CLIP-L c14 cosine
similarity within a window of nearby frames), then re-run rank_and_filter. End-to-end D on folds 1-3 (val frames).
Purity of clusters is checked against GT object ids (reported, not used)."""
import numpy as np
from collections import defaultdict
from common import load_split
from bank import det_bank
from predict_form import FormClassifier, rank_and_filter
from evalkit import evaluate, fmt, solution_for
from form_model2 import logit, sigmoid


def cluster(dets, E, sim_thr, gap):
    """Greedy single-linkage over detections sorted by frame: link if cos >= sim_thr and |frame gap| <= gap. Returns cluster ids."""
    order = sorted(range(len(dets)), key=lambda i: dets[i]["frame"])
    cid = [-1] * len(dets)
    nxt = 0
    for i in order:
        fi = int(dets[i]["frame"].split("_")[1])
        best = None
        for j in order:
            if cid[j] < 0 or j == i:
                continue
            fj = int(dets[j]["frame"].split("_")[1])
            if abs(fi - fj) <= gap and float(E[i] @ E[j]) >= sim_thr:
                s = float(E[i] @ E[j])
                if best is None or s > best[0]:
                    best = (s, cid[j])
        if best is None:
            cid[i] = nxt
            nxt += 1
        else:
            cid[i] = best[1]
    return cid


for fold in (1, 2, 3):
    sp = load_split(fold)
    val = set(sp["validation_real_frames"])
    frames = sorted(val)
    fc = FormClassifier(topk=3, tune_thr=False).fit(sp["train_object_ids"])
    rows, bank = det_bank(fold)
    pb = fc.predict_bank(bank)
    idx = [i for i, r in enumerate(rows) if r["frame"] in val and r["conf"] >= 0.08]
    dets = [dict(rows[i], p_bulk=float(pb[i])) for i in idx]
    E = bank.emb["clipL_c14"][idx]
    E = E / np.linalg.norm(E, axis=1, keepdims=True)
    sol = solution_for(val)
    base = evaluate({f: rank_and_filter([d for d in dets if d["frame"] == f]) for f in frames}, frames, sol)[0]
    print(f"fold {fold} base      {fmt(base)}")
    for sim_thr in (0.85, 0.9, 0.93):
        for gap in (3, 10):
            cid = cluster(dets, E, sim_thr, gap)
            groups = defaultdict(list)
            for k, c in enumerate(cid):
                groups[c].append(k)
            # purity: fraction of dets whose cluster majority gt_oid equals their own (only TP dets)
            pur, n = 0, 0
            for g in groups.values():
                oids = [dets[k]["gt_oid"] for k in g if dets[k]["gt_iou"] >= 0.5]
                if oids:
                    maj = max(set(oids), key=oids.count)
                    pur += sum(o == maj for o in oids)
                    n += len(oids)
            pooled = []
            for k, d in enumerate(dets):
                g = groups[cid[k]]
                w = np.array([dets[j]["conf"] for j in g])
                z = np.array([logit(1 - dets[j]["p_bulk"]) for j in g])  # logit of p_solid
                zp = float((w * z).sum() / w.sum())
                pooled.append(dict(d, p_bulk=1 - sigmoid(zp)))
            res = evaluate({f: rank_and_filter([d for d in pooled if d["frame"] == f]) for f in frames}, frames, sol)[0]
            print(f"fold {fold} sim {sim_thr} gap {gap:2d} clusters {len(groups):3d} purity {pur / max(n, 1):.3f} {fmt(res)}")
