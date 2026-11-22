"""Leave-one-object-out evaluation of the form classifier on GT masks (real_train, 40 physical objects).
Nested: for each held-out object, all hyperparameters are selected by grouped inner CV on the remaining 39 objects.
Also reports fixed-config LOOO for reference variants. Usage: ~/venv/bin/python form_cv.py [--quick]"""
import argparse
import json
from collections import Counter

import numpy as np

from common import C, W, gt_cache
from form_model import GRID, ZS_SETS, W_GRID, FeatureBank, FormModel, bal_acc, blend, head_predict, zs_prob


def load_bank():
    F, meta = gt_cache()
    text = {tuple(k.split("|")): v for k, v in np.load(W / "text_cache.npz").items()}
    emb = {k: F[k] for k in F if k.startswith(("clipL_", "siglip_", "dinoL_")) and not k.endswith(("_zs", "_text"))}
    emb["geo"] = np.stack([F["geo"][:, list(F["geo_keys"]).index(k)] for k in
                           ["area_px", "ext1", "ext2", "foot", "h95", "h_med", "hmax", "holes_edge", "holes_inner", "rough", "surf", "valid", "vol", "vol_thr", "zmed"]], 1)
    bank = FeatureBank(emb, text)
    oid = np.array([m["oid"] for m in meta])
    src = np.array([m["src"] for m in meta])
    y = np.array([0 if m["form"] == "bulk" else 1 for m in meta])
    return bank, y, oid, src, meta


def report(P, y, oid, real, tag):
    idx = np.where(real)[0]
    pred = (P[idx, 1] > 0.5).astype(int)
    acc_b = (pred[y[idx] == 0] == 0).mean()
    acc_s = (pred[y[idx] == 1] == 1).mean()
    wrong = Counter(oid[idx][pred != y[idx]])
    # per-object (majority over frames)
    objs = sorted(set(oid[idx]))
    obj_hit = {o: int((P[idx][oid[idx] == o, 1].mean() > 0.5) == y[idx][oid[idx] == o][0]) for o in objs}
    line = (f"{tag:55s} bulk {acc_b:.3f} solid {acc_s:.3f} bal {(acc_b + acc_s) / 2:.3f} frame {(pred == y[idx]).mean():.3f} "
            f"obj {np.mean(list(obj_hit.values())):.3f} wrong {dict(wrong)}")
    print(line, flush=True)
    return {"acc_bulk": float(acc_b), "acc_solid": float(acc_s), "bal": float((acc_b + acc_s) / 2), "frame_acc": float((pred == y[idx]).mean()),
            "obj_acc": float(np.mean(list(obj_hit.values()))), "wrong_frames_by_object": {k: int(v) for k, v in wrong.items()}}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    a = ap.parse_args()
    bank, y, oid, src, meta = load_bank()
    real = src == "real_train"
    objs = sorted(set(oid[real]))
    out = {}
    # 1) reference: fixed configs, plain LOOO (no selection -> no nesting needed)
    print("== fixed-config LOOO (reference) ==")
    for spec, sw, wd, zs, w in [("clipL_c14", 0.0, 1e-2, "none", 0), ("clipL_c14", 0.5, 1e-2, "none", 0),
                                ("clipL_c14", 0.5, 1e-2, "clipL:mix:c14", 0.5),
                                ("zs", 0.5, 1e-1, "none", 0), ("zs2", 0.5, 1e-1, "none", 0),
                                ("clipL_c14+zs", 0.5, 1e-2, "none", 0), ("clipL_c14+zs", 1.0, 1e-2, "none", 0),
                                ("clipL_c14+zs2", 0.5, 1e-2, "none", 0),
                                ("clipL_c14+clipL_m10+zs", 0.5, 1e-2, "none", 0), ("clipL_c14+geo+zs", 0.5, 1e-2, "none", 0)]:
        X = bank.features(spec)
        Lz = bank.zs_logits(zs)
        P = np.zeros((len(y), 2))
        for o in objs:
            tr = np.where((oid != o) & (real | (sw > 0)))[0]
            te = np.where((oid == o) & real)[0]
            P[te] = blend(head_predict(X, y, oid, src, tr, te, sw, wd), None if Lz is None else Lz[te], w)
        out[f"fixed|{spec}|sw={sw}|wd={wd}|{zs}|w={w}"] = report(P, y, oid, real, f"{spec} sw={sw} wd={wd} {zs} w={w}")
    for zs in ["clipL:mix:c14", "clipL:pile:c14", "clipL:parts:m10", "clipL:multi:c14"]:
        P = zs_prob(bank.zs_logits(zs))
        out[f"zeroshot|{zs}"] = report(P, y, oid, real, f"zero-shot only {zs}")
    # 2) nested LOOO
    print("== nested LOOO (hyperparameters selected by inner 5-fold grouped CV on the 39 other objects) ==")
    grid = GRID if not a.quick else {"spec": ["clipL_c14"], "sw": [0.5], "wd": [1e-2]}
    P = np.zeros((len(y), 2))
    chosen = []
    for o in objs:
        m = FormModel(bank, y, oid, src, grid=grid).fit([x for x in objs if x != o])
        te = np.where((oid == o) & real)[0]
        P[te] = m.predict_idx(te)
        chosen.append(m.cfg)
        print(f"  held-out {o}: cfg {m.cfg} inner {m.inner_score[0]:.3f} -> p_solid mean {P[te, 1].mean():.2f} (true {'solid' if y[te][0] else 'bulk'})", flush=True)
    out["nested"] = report(P, y, oid, real, "NESTED LOOO")
    out["nested"]["chosen_configs"] = Counter(json.dumps(c, sort_keys=True) for c in chosen).most_common()
    np.save(W / "form_looo_oof.npy", P)
    json.dump(out, open(W / "form_cv_results.json", "w"), indent=1)


if __name__ == "__main__":
    main()
