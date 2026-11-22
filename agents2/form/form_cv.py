"""Leave-one-object-out evaluation of form (bulk/solid) on GT masks, 40 real objects.
1) fixed-config LOOO for many feature-block combinations (reference, non-nested);
2) nested LOOO: for each held-out object, feature spec / synth weight / weight decay / threshold are chosen by inner
   5-fold grouped CV on the remaining 39 objects (FormModel2), optionally top-k ensemble.
Usage: ~/venv/bin/python form_cv.py [--quick] [--topk 1 3] [--stage fixed nested]"""
import argparse
import json
from collections import Counter

import numpy as np

from common import W
from bank import gt_bank
from form_model2 import GRID, FormModel2, head_predict

FIXED = ["clipL_c14+zs", "zs", "zsM", "zsC", "zs+zsM", "zs+zsM+zsC", "depth", "dstat", "sam", "depth+sam", "depth+dstat+sam", "dino", "dino+zs",
         "clipL_c14+zs+depth", "clipL_c14+zs+dstat", "clipL_c14+zs+sam", "clipL_c14+zs+depth+sam", "clipL_c14+zs+zsM", "clipL_c14+zs+zsM+zsC",
         "clipL_c14+zs+zsM+depth+sam", "clipL_c14+zs+zsM+zsC+depth+dstat+sam", "zs+zsM+zsC+depth+dstat+sam", "clipL_c14+dino+zs",
         "clipL_c14+dino+zs+depth+sam", "clipL_c14+zs+geo"]


def report(p_solid, y, oid, real, tag, thr=0.5):
    idx = np.where(real)[0]
    pred = (p_solid[idx] > thr).astype(int)
    acc_b = (pred[y[idx] == 0] == 0).mean()
    acc_s = (pred[y[idx] == 1] == 1).mean()
    wrong = Counter(str(o) for o in oid[idx][pred != y[idx]])
    objs = sorted(set(oid[idx]))
    obj_hit = {o: int((p_solid[idx][oid[idx] == o].mean() > thr) == y[idx][oid[idx] == o][0]) for o in objs}
    print(f"{tag:50s} bulk {acc_b:.3f} solid {acc_s:.3f} bal {(acc_b + acc_s) / 2:.3f} frame {(pred == y[idx]).mean():.3f} "
          f"obj {np.mean(list(obj_hit.values())):.3f} wrong {dict(wrong)}", flush=True)
    return {"acc_bulk": float(acc_b), "acc_solid": float(acc_s), "bal": float((acc_b + acc_s) / 2), "frame_acc": float((pred == y[idx]).mean()),
            "obj_acc": float(np.mean(list(obj_hit.values()))), "wrong_frames_by_object": dict(wrong)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--topk", type=int, nargs="+", default=[1, 3])
    ap.add_argument("--stage", nargs="+", default=["fixed", "nested"])
    a = ap.parse_args()
    bank, y, oid, src, meta = gt_bank()
    real = src == "real_train"
    objs = sorted(set(oid[real]))
    out = {}
    if "fixed" in a.stage:
        print("== fixed-config LOOO (reference, not nested) ==")
        for spec in FIXED:
            X = bank.features(spec)
            for sw, wd in ((0.5, 1e-2), (0.5, 1e-1)):
                P = np.zeros(len(y))
                for o in objs:
                    tr = np.where((oid != o) & (real | (sw > 0)))[0]
                    te = np.where((oid == o) & real)[0]
                    P[te] = head_predict(X, y, oid, src, tr, te, sw, wd)[:, 1]
                out[f"fixed|{spec}|sw={sw}|wd={wd}"] = report(P, y, oid, real, f"{spec} sw={sw} wd={wd}")
    if "nested" in a.stage:
        grid = GRID if not a.quick else {"spec": ["clipL_c14+zs"], "sw": [0.5], "wd": [1e-2], "lam": [0.0, 1.0, 2.0, 4.0, 8.0]}
        for topk in a.topk:
            print(f"== nested LOOO topk={topk} (inner 5-fold grouped CV on the other 39 objects; threshold from inner OOF) ==")
            P = np.zeros(len(y))
            P05 = np.zeros(len(y))
            chosen, thrs = [], []
            for o in objs:
                m = FormModel2(bank, y, oid, src, grid=grid, topk=topk, thr_grid=np.round(np.arange(0.3, 0.71, 0.05), 2)).fit([x for x in objs if x != o])
                te = np.where((oid == o) & real)[0]
                p = m.predict_idx(te)
                P05[te] = p
                # map through threshold: shift so that thr -> 0.5 (monotone), keeps downstream code at 0.5
                P[te] = np.where(p > m.thr, 0.5 + 0.5 * (p - m.thr) / (1 - m.thr), 0.5 * p / m.thr)
                chosen += m.cfgs
                thrs.append(m.thr)
                print(f"  held-out {o}: cfgs {m.cfgs} thr {m.thr} inner {m.inner_score[0]:.3f} -> p_solid {np.round(p, 2)} (true {'solid' if y[te][0] else 'bulk'})", flush=True)
            out[f"nested_topk{topk}"] = report(P, y, oid, real, f"NESTED LOOO topk={topk} (tuned thr)")
            out[f"nested_topk{topk}_thr0.5"] = report(P05, y, oid, real, f"NESTED LOOO topk={topk} (thr 0.5)")
            out[f"nested_topk{topk}"]["chosen_configs"] = Counter(json.dumps(c) for c in chosen).most_common()
            out[f"nested_topk{topk}"]["thresholds"] = thrs
            np.save(W / f"form_looo_oof_topk{topk}.npy", P)
    json.dump(out, open(W / "form_cv_results.json", "w"), indent=1)


if __name__ == "__main__":
    main()
