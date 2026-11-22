"""End-to-end D evaluation on folds 1-3 via the official score.py.
For fold k: cached detections of fold k's detector (cache/fold{k}_dets.*, produced by detect_cache.py, conf>=0.01 on all real frames)
-> form classifier trained on fold k's train_object_ids (GT-mask features of real+synth annotations of those objects; hyperparameters by
inner grouped CV) -> ranking / filtering variants whose thresholds are tuned on the fold's TRAIN real frames -> CSV -> score.py.
Usage: ~/venv/bin/python form_pipeline.py [--folds 1 2 3]"""
import argparse
import json
from collections import Counter, defaultdict

import numpy as np
import pandas as pd

from common import C, D, FOLDS, W, box_iou, gt_by_frame, gt_cache, load_split  # noqa: F401  (sets sys.path)
import score as S  # references/score.py  # noqa: E402
from form_cv import load_bank
from form_model import FeatureBank, FormModel, GEO_KEYS

OUT = W / "out"


def det_bank(fold, text):
    rows = json.load(open(W / "cache" / f"fold{fold}_dets.json"))
    E = dict(np.load(W / "cache" / f"fold{fold}_dets.npz"))
    emb = {k: v for k, v in E.items() if not k.startswith("text_")}
    emb["geo"] = np.array([[r["geo"][k] for k in GEO_KEYS] for r in rows], np.float32).reshape(len(rows), len(GEO_KEYS))
    return rows, FeatureBank(emb, text)


def build_labels(frames):
    lab = json.load(open(D / "labels/real_train.json"))
    keep = [im for im in lab["images"] if im["file_name"].split("/")[-2] in frames]
    ids = {im["id"] for im in keep}
    data = {"categories": lab["categories"], "images": keep, "annotations": [a for a in lab["annotations"] if a["image_id"] in ids]}
    p = OUT / f"labels_{abs(hash(tuple(sorted(frames)))) % 10**8}.json"
    p.write_text(json.dumps(data))
    return p


def serialize(objs):
    if not objs:
        return "none"
    return ";".join(f"{o['box'][0]:.1f} {o['box'][1]:.1f} {o['box'][2]:.1f} {o['box'][3]:.1f} {o['score']:.5f} {o['form']} plastic 100 100 1" for o in objs[:30])


def nms_cross_class(dets, thr):
    """Greedy NMS over boxes regardless of class (dets sorted by score desc)."""
    keep = []
    for d in sorted(dets, key=lambda d: -d["score"]):
        if all(box_iou(d["box"], k["box"]) < thr for k in keep):
            keep.append(d)
    return keep


def apply_variant(dets, p_bulk, cfg):
    """dets: list of cached detection rows; p_bulk: our classifier prob per det.
    cfg: dict(conf, form_src, two_class, nms, border, score_mode, min_pmax). Returns per-frame list of output objects."""
    per_frame = defaultdict(list)
    cand = []
    for d, pb in zip(dets, p_bulk):
        if d["conf"] < cfg["conf"]:
            continue
        if cfg.get("border") and touches_border(d["box"]):
            continue
        if cfg.get("hmin") and d["geo"]["hmax"] < cfg["hmin"] and d["geo"]["valid"] > 0.3:
            continue
        if cfg["form_src"] == "det":
            pb_use = 1.0 if d["cls"] == "bulk" else 0.0
        elif cfg["form_src"] == "cls":
            pb_use = pb
        else:  # blend of detector class (soft) and classifier
            pd_ = 0.9 if d["cls"] == "bulk" else 0.1
            a = cfg.get("alpha", 0.5)
            z = a * np.log([pd_, 1 - pd_]) + (1 - a) * np.log([pb + 1e-9, 1 - pb + 1e-9])
            z = np.exp(z - z.max())
            pb_use = float(z[0] / z.sum())
        pmax = max(pb_use, 1 - pb_use)
        if cfg.get("min_pmax") and pmax < cfg["min_pmax"]:
            continue
        cand.append({"frame": d["frame"], "box": d["box"], "conf": d["conf"], "pb": pb_use, "score": d["conf"] * pmax if cfg["score_mode"] == "prod" else d["conf"]})
    byf = defaultdict(list)
    for c in cand:
        byf[c["frame"]].append(c)
    for f, L in byf.items():
        if cfg.get("nms"):
            L = nms_cross_class(L, cfg["nms"])
        out = []
        for c in L:
            form = "bulk" if c["pb"] >= 0.5 else "solid"
            pmax = max(c["pb"], 1 - c["pb"])
            if cfg["score_mode"] == "prod":
                out.append({"box": c["box"], "score": c["conf"] * pmax, "form": form})
                if cfg.get("two_class"):
                    out.append({"box": c["box"], "score": c["conf"] * (1 - pmax), "form": "solid" if form == "bulk" else "bulk"})
            else:
                out.append({"box": c["box"], "score": c["conf"], "form": form})
                if cfg.get("two_class"):
                    out.append({"box": c["box"], "score": c["conf"] * (1 - pmax) / max(pmax, 1e-6), "form": "solid" if form == "bulk" else "bulk"})
        out.sort(key=lambda o: -o["score"])
        per_frame[f] = out[:30]
    return per_frame


def touches_border(b, margin=2, w=1280, h=720):
    return b[0] <= margin or b[1] <= margin or b[2] >= w - margin or b[3] >= h - margin


def evaluate(per_frame, frames, sol):
    sub = pd.DataFrame({"image_id": frames, "prediction": [serialize(per_frame.get(f, [])) for f in frames]})
    gt = {str(i): S._parse(c, False, str(i)) for i, c in zip(sol.image_id, sol.objects)}
    pr = {str(i): S._parse(c, True, str(i)) for i, c in zip(sub.image_id, sub.prediction)}
    ap = {f: S._average_precision(gt, pr, f) for f in ("bulk", "solid")}
    comp = S.components(gt, pr)
    # confusion / FP on matched (class-agnostic) boxes
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
                conf[(g_form := objs[bj]["form"], p["form"])] += 1
            else:
                n_fp += 1
    return {"AP_bulk": ap["bulk"], "AP_solid": ap["solid"], "D": comp["D"], "confusion": {f"{a}->{b}": v for (a, b), v in sorted(conf.items())},
            "n_fp": n_fp, "n_pred": sum(len(v) for v in pr.values())}, sub


VARIANTS = {
    "A_baseline_det(conf0.08)": dict(conf=0.08, form_src="det", score_mode="conf"),
    "B_cls_form_only": dict(conf=0.08, form_src="cls", score_mode="conf"),
    "C_cls_form+prod_score": dict(conf=0.08, form_src="cls", score_mode="prod"),
    "D_C+nms0.6": dict(conf=0.08, form_src="cls", score_mode="prod", nms=0.6),
    "E_D+border": dict(conf=0.08, form_src="cls", score_mode="prod", nms=0.6, border=True),
    "F_E_blend_det": dict(conf=0.08, form_src="blend", alpha=0.3, score_mode="prod", nms=0.6, border=True),
    "G_E_conf0.01": dict(conf=0.01, form_src="cls", score_mode="prod", nms=0.6, border=True),
    "H_E_conf0.2": dict(conf=0.2, form_src="cls", score_mode="prod", nms=0.6, border=True),
    "E_twoclass": dict(conf=0.08, form_src="cls", score_mode="prod", nms=0.6, border=True, two_class=True),
    "G_twoclass": dict(conf=0.01, form_src="cls", score_mode="prod", nms=0.6, border=True, two_class=True),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--folds", type=int, nargs="+", default=[1, 2, 3])
    ap.add_argument("--quick", action="store_true", help="fixed classifier config (no inner CV)")
    a = ap.parse_args()
    OUT.mkdir(exist_ok=True)
    bank, y, oid, src, meta = load_bank()
    report = {}
    for fold in a.folds:
        sp = load_split(fold)
        train_objs = sorted(sp["train_object_ids"])
        val_frames = sorted(sp["validation_real_frames"])
        tr_frames = sorted(sp["train_real_frames"])
        grid = None if not a.quick else {"spec": ["clipL_c14+zs"], "sw": [0.5], "wd": [1e-2]}
        model = FormModel(bank, y, oid, src, **({"grid": grid} if grid else {}), verbose=True).fit(train_objs)
        rows, dbank = det_bank(fold, bank.text)
        P = model.predict_bank(dbank)
        p_bulk = P[:, 0]
        # sanity: classifier accuracy on TP detections, train vs val frames
        for name, fr in (("train", set(tr_frames)), ("val", set(val_frames))):
            idx = [i for i, r in enumerate(rows) if r["frame"] in fr and r["gt_iou"] >= 0.5 and r["conf"] >= 0.08]
            yy = np.array([rows[i]["gt_form"] == "bulk" for i in idx]); pp = p_bulk[idx] >= 0.5
            dd = np.array([rows[i]["cls"] == "bulk" for i in idx])
            print(f"fold {fold} {name}-frame TP dets: n={len(idx)} cls acc {np.mean(pp == yy):.3f} (bulk {np.mean(pp[yy]):.3f} solid {np.mean(~pp[~yy]):.3f}) | det acc {np.mean(dd == yy):.3f}")
        rep = {"cfg": model.cfg, "inner_bal_acc": model.inner_score[0]}
        for split_name, frames in (("val", val_frames), ("train", tr_frames)):
            sol = S.labels_to_solution(build_labels(set(frames)))
            dets = [r for r in rows if r["frame"] in set(frames)]
            pb = p_bulk[[i for i, r in enumerate(rows) if r["frame"] in set(frames)]]
            rep[split_name] = {}
            for vname, cfg in VARIANTS.items():
                res, sub = evaluate(apply_variant(dets, pb, cfg), frames, sol)
                rep[split_name][vname] = res
                if split_name == "val":
                    sub.to_csv(OUT / f"fold{fold}_{vname}.csv", index=False)
                print(f"fold {fold} {split_name:5s} {vname:28s} AP_bulk {res['AP_bulk']:.3f} AP_solid {res['AP_solid']:.3f} D {res['D']:.3f} "
                      f"FP {res['n_fp']:3d} npred {res['n_pred']:3d} conf {res['confusion']}", flush=True)
        report[fold] = rep
    print("\n== mean over folds (val) ==")
    for vname in VARIANTS:
        ds = [report[f]["val"][vname]["D"] for f in report]
        print(f"  {vname:28s} D per fold {' '.join(f'{d:.3f}' for d in ds)}  mean {np.mean(ds):.3f}")
    json.dump(report, open(OUT / "form_pipeline_report.json", "w"), indent=1, default=str)


if __name__ == "__main__":
    main()
