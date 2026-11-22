"""End-to-end D (mean AP50 over bulk/solid, official score.py) on folds 1-3.
Fold k: cached detections of fold k's detector on all real frames (agents/form/cache, conf>=0.01) + v2 features
-> FormClassifier trained on the fold's train_object_ids (nested inner CV for all choices)
-> rank_and_filter variants (gamma / containment / nms) -> CSV per fold (val frames) -> score.py.
Ranking hyperparameters are reported for all variants; the recommended default is fixed a priori (gamma=1, contain=0.8),
train-frame D is printed as a sanity check (it is ~1.0 for every variant, so it cannot be used for selection).
Usage: ~/venv/bin/python form_pipeline.py [--folds 1 2 3] [--topk 3] [--quick]"""
import argparse
import json
from collections import defaultdict

import numpy as np

from common import W, load_split
from bank import det_bank
from form_model2 import GRID
from predict_form import FormClassifier, rank_and_filter, pool_series
from evalkit import OUT, evaluate, fmt, solution_for

VARIANTS = {
    "A_det_form_conf": dict(src="det"),
    "B_cls_g1": dict(gamma=1.0, contain=None),
    "C_cls_g1_cont0.8": dict(gamma=1.0, contain=0.8),
    "D_cls_g2_cont0.8": dict(gamma=2.0, contain=0.8),
    "E_cls_g0.5_cont0.8": dict(gamma=0.5, contain=0.8),
    "F_cls_g1_cont0.7": dict(gamma=1.0, contain=0.7),
    "G_cls_g1_cont0.8_nms0.5": dict(gamma=1.0, contain=0.8, nms=0.5),
    "H_cls_g1_cont0.8_conf0.2": dict(gamma=1.0, contain=0.8, conf=0.2),
    "P_pool_g1_cont0.8": dict(gamma=1.0, contain=0.8, pool=True),
    "Q_pool_g2_cont0.8": dict(gamma=2.0, contain=0.8, pool=True),
    "R_pool_sim0.85_gap3": dict(gamma=1.0, contain=0.8, pool=True, pool_sim=0.85, pool_gap=3),
    "S_pool_sim0.93_gap10": dict(gamma=1.0, contain=0.8, pool=True, pool_sim=0.93, pool_gap=10),
}


def run_variant(rows, p_bulk, frames, cfg, embs=None):
    dets, E = [], []
    for i, (r, p) in enumerate(zip(rows, p_bulk)):
        if r["frame"] not in frames:
            continue
        if cfg.get("src") == "det":
            p = 0.9 if r["cls"] == "bulk" else 0.1
        dets.append({"frame": r["frame"], "box": r["box"], "conf": r["conf"], "p_bulk": p})
        E.append(embs[i])
    if cfg.get("pool"):
        dets = pool_series(dets, np.stack(E), **{k[5:]: v for k, v in cfg.items() if k.startswith("pool_")})
    byf = defaultdict(list)
    for d in dets:
        byf[d["frame"]].append(d)
    kw = {k: v for k, v in cfg.items() if k != "src" and not k.startswith("pool")}
    return {f: rank_and_filter(L, **kw) for f, L in byf.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--folds", type=int, nargs="+", default=[1, 2, 3])
    ap.add_argument("--topk", type=int, default=3)
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--no-thr", action="store_true", help="fixed 0.5 threshold instead of inner-CV tuned")
    ap.add_argument("--tag", default="")
    a = ap.parse_args()
    report = {}
    grid = GRID if not a.quick else {"spec": ["clipL_c14+zs"], "sw": [0.5], "wd": [1e-2], "lam": [0.0, 1.0, 2.0, 4.0, 8.0]}
    for fold in a.folds:
        sp = load_split(fold)
        val = sorted(sp["validation_real_frames"])
        trf = sorted(sp["train_real_frames"])
        fc = FormClassifier(topk=a.topk, grid=grid, tune_thr=not a.no_thr).fit(sp["train_object_ids"])
        print(f"fold {fold}: cfgs {fc.model.cfgs} thr {fc.thr} inner {fc.model.inner_score}", flush=True)
        rows, bank = det_bank(fold)
        pb = fc.predict_bank(bank)
        for name, fr in (("train", set(trf)), ("val", set(val))):
            idx = [i for i, r in enumerate(rows) if r["frame"] in fr and r["gt_iou"] >= 0.5 and r["conf"] >= 0.08]
            yy = np.array([rows[i]["gt_form"] == "bulk" for i in idx])
            pp = pb[idx] >= 0.5
            wrong = defaultdict(list)
            for i in idx:
                if (pb[i] >= 0.5) != (rows[i]["gt_form"] == "bulk"):
                    wrong[rows[i]["gt_oid"]].append(round(float(pb[i]), 2))
            print(f"fold {fold} {name}-frame TP dets n={len(idx)} form acc {np.mean(pp == yy):.3f} (bulk {np.mean(pp[yy]):.3f} solid {np.mean(~pp[~yy]):.3f}) wrong {dict(wrong)}", flush=True)
        rep = {"cfgs": fc.model.cfgs, "thr": fc.thr, "inner": fc.model.inner_score, "val": {}, "train": {}}
        solv, solt = solution_for(set(val)), solution_for(set(trf))
        for vname, cfg in VARIANTS.items():
            rv, sub = evaluate(run_variant(rows, pb, set(val), cfg, bank.emb["clipL_c14"]), val, solv)
            rt, _ = evaluate(run_variant(rows, pb, set(trf), cfg, bank.emb["clipL_c14"]), trf, solt)
            rep["val"][vname], rep["train"][vname] = rv, rt
            sub.to_csv(OUT / f"fold{fold}_{vname}{a.tag}.csv", index=False)
            print(f"fold {fold} val {vname:26s} {fmt(rv)} | train D {rt['D']:.3f} FP {rt['n_fp']}", flush=True)
        report[fold] = rep
    print("\n== D per fold (val) ==")
    for vname in VARIANTS:
        ds = [report[f]["val"][vname]["D"] for f in report]
        print(f"  {vname:26s} {' '.join(f'{d:.3f}' for d in ds)}  mean {np.mean(ds):.3f}")
    json.dump(report, open(OUT / f"form_pipeline_report{a.tag}.json", "w"), indent=1, default=str)


if __name__ == "__main__":
    main()
