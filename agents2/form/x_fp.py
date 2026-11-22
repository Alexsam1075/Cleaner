"""Remaining FP boxes after rank_and_filter on val frames (folds 2,3) and their effect on AP: what are they, where do they rank."""
import json
import numpy as np
from collections import defaultdict
from common import load_split
from bank import det_bank
from predict_form import FormClassifier, rank_and_filter
from evalkit import evaluate, solution_for

for fold in (2, 3):
    sp = load_split(fold)
    val = set(sp["validation_real_frames"])
    fc = FormClassifier(topk=3, tune_thr=False).fit(sp["train_object_ids"])
    rows, bank = det_bank(fold)
    pb = fc.predict_bank(bank)
    byf = defaultdict(list)
    for r, p in zip(rows, pb):
        if r["frame"] in val:
            byf[r["frame"]].append(dict(r, p_bulk=float(p)))
    kept = {f: rank_and_filter(L) for f, L in byf.items()}
    allk = sorted([d for L in kept.values() for d in L], key=lambda d: -d["score"])
    print(f"== fold {fold}: kept {len(allk)} boxes on {len(kept)} frames")
    for rank, d in enumerate(allk):
        tag = "TP " if d["gt_iou"] >= 0.5 else "FP "
        ok = (d["form"] == d["gt_form"]) if d["gt_iou"] >= 0.5 else None
        if d["gt_iou"] < 0.5 or not ok:
            print(f"  rank {rank:3d}/{len(allk)} {tag} {d['frame']} form {d['form']} (gt {d['gt_form']} {d['gt_oid']}) score {d['score']:.3f} conf {d['conf']:.2f} p_bulk {d['p_bulk']:.2f} iou {d['gt_iou']:.2f} box {[int(v) for v in d['box']]} mask_area {d['mask_area']}")
