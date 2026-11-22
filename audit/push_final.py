"""Последние рычаги поверх «память + вторая форма» (88.0): фолды 1–3, score.py.
 (a) материал: сдвиг log-p к редким классам, lambda·log(1/доля класса в train) — M макро по классам;
 (b) take: пороги по группам материала (metal/glass/mineral vs paper/plastic/wood), подобраны на train-объектах."""
import glob
import json
import math
import pickle
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path("/home/cleaner/Cleaner")
sys.path.insert(0, str(ROOT / "audit"))
import full_pipeline as fp  # noqa: E402
from full_pipeline import C, D, FOLDS, MATS, MODELS, VIEWS, S, _iou, serialize, touches_border_px  # noqa: E402
from object_memory import cluster, pool  # noqa: E402

HEAVY = {"metal", "glass", "mineral"}


def thr_by_group(rows, est, group):
    obj = defaultdict(list)
    meta_o = {}
    for r in rows.to_dict("records"):
        v = est.estimate(None, None, None, r["form"], r["material"], geometry=r).volume_ml
        obj[r["object_id"]].append(v)
        meta_o[r["object_id"]] = (r["material"], int(r["mass_g"] <= 500))
    xs = np.array([np.median(v) for o, v in obj.items() if (meta_o[o][0] in HEAVY) == group])
    ys = np.array([meta_o[o][1] for o in obj if (meta_o[o][0] in HEAVY) == group])
    best = (-1, 500.0)
    for t in np.unique(xs):
        for tt in (t - 1e-6, t + 1e-6):
            p = (xs <= tt).astype(int)
            s = np.mean([(p[ys == c] == c).mean() for c in (0, 1) if (ys == c).any()])
            if s > best[0]:
                best = (s, float(tt))
    return best[1]


def main():
    enc = fp.Encoders()
    F, meta = fp.train_cache()
    mat_count = Counter(m["mat"] for m in meta if m["src"] == "real_train")
    tot = sum(mat_count.values())
    prior = np.array([mat_count[m] / tot for m in MATS])
    res = defaultdict(list)
    for fold in (1, 2, 3):
        split = json.load(open(FOLDS[fold][0]))
        train_objs = sorted(split["train_object_ids"])
        frames = list(split["validation_real_frames"])
        cam = json.load(open(D / "camera.json"))
        est, take_thr = fp.volume_estimator_for(train_objs, cam)
        rows = pd.read_csv(ROOT / "reports/volume_parts_gt_rows.csv")
        rows = rows[(rows.split == "real") & rows.object_id.isin(train_objs)]
        thrH, thrL = thr_by_group(rows, est, True), thr_by_group(rows, est, False)
        dets, embs = pickle.load(open(C / f"fold{fold}_dets.pkl", "rb"))
        predict_mat, _ = fp.material_model(F, meta, train_objs, enc.text, "generic_syn")
        embs_t = {(t, v): embs[(t, v)] for t in MODELS for v in VIEWS}
        P0 = predict_mat(embs[("clipL", "c14")], fp.zero_shot(embs_t, enc.text, "generic_syn"))
        sys.path.insert(0, str(ROOT / "agents/form"))
        from form_cv import load_bank
        from form_model import FeatureBank, FormModel, GEO_KEYS
        fbank, fy, foid, fsrc, _ = load_bank()
        fm = FormModel(fbank, fy, foid, fsrc, grid={"spec": ["clipL_c14+zs"], "sw": [0.5], "wd": [1e-2]}).fit(train_objs)
        femb = {f"{t}_{v}": embs[(t, v)] for t in MODELS for v in VIEWS}
        femb["geo"] = np.array([[d["geo"][k] for k in GEO_KEYS] for d in dets], np.float32).reshape(len(dets), len(GEO_KEYS))
        p_bulk = fm.predict_bank(FeatureBank(femb, fbank.text))[:, 0]
        cid = cluster(dets, embs, near=0.90, far=1.01, gap=3)
        sol = next(S.labels_to_solution(p) for p in glob.glob(str(C / "val_labels_*.json")) if set(S.labels_to_solution(p).image_id.astype(str)) == set(frames))
        gt = {str(i): S._parse(c, False, str(i)) for i, c in zip(sol.image_id, sol.objects)}
        print(f"== fold {fold}: пороги take heavy-группа {thrH:.0f} мл, light-группа {thrL:.0f} мл, общий {take_thr:.0f}", flush=True)
        for lam in (0.0, 0.25, 0.5, 0.75, 1.0):
            lp = np.log(P0 + 1e-9) - lam * np.log(prior)[None, :]
            P = np.exp(lp - lp.max(1, keepdims=True))
            P /= P.sum(1, keepdims=True)
            forms = ["bulk" if pb >= 0.5 else "solid" for pb in p_bulk]
            mats = [MATS[int(p.argmax())] for p in P]
            vols = np.array([est.estimate(None, None, None, f, m, geometry=d.get("new_geo_by_form", {}).get(f, d["new_geo"])).volume_ml for d, f, m in zip(dets, forms, mats)])
            _, _, vols = pool(dets, cid, P, p_bulk, vols, use_mat=False, use_form=False)
            for take_mode in ("общий", "по группе", "по группе, heavy = общий"):
                per = defaultdict(list)
                for d, pb, f, m, v in zip(dets, p_bulk, forms, mats, vols):
                    if d["score"] < 0.08 or touches_border_px(d["box"]):
                        continue
                    if take_mode == "общий":
                        t = take_thr
                    elif take_mode == "по группе":
                        t = thrH if m in HEAVY else thrL
                    else:
                        t = take_thr if m in HEAVY else max(thrL, take_thr)
                    dn = est.estimate(None, None, None, f, m, geometry=d.get("new_geo_by_form", {}).get(f, d["new_geo"])).density_g_per_ml
                    pm = max(pb, 1 - pb)
                    per[d["frame"]].append({"box": d["box"], "score": d["score"] * pm, "form": f, "material": m, "volume": float(v), "mass": float(v * dn),
                                            "take": int(v <= t), "conf": d["score"], "pmax": pm})
                for fr in per:
                    keep = []
                    for c in sorted(per[fr], key=lambda o: -o["score"]):
                        if all(_iou(c["box"], k["box"]) < 0.6 for k in keep):
                            keep.append(c)
                    extra = [dict(c, form="solid" if c["form"] == "bulk" else "bulk", score=c["conf"] * (1 - c["pmax"])) for c in keep]
                    per[fr] = sorted(keep + extra, key=lambda o: -o["score"])[:30]
                sub = pd.DataFrame({"image_id": frames, "prediction": [serialize(per.get(f_, [])) for f_ in frames]})
                pr = {str(i): S._parse(c, True, str(i)) for i, c in zip(sub.image_id, sub.prediction)}
                comp = S.components(gt, pr)
                res[(lam, take_mode)].append(comp)
                print(f"  lambda={lam:.2f} take={take_mode:26s} Score {comp['Score']:6.2f} | D {comp['D']:.3f} M {comp['M']:.3f} V {comp['V']:.3f} S {comp['S']:.3f}", flush=True)
                if fold == 3:
                    sub.to_csv(C / f"pf_fold{fold}_{lam}_{take_mode.replace(' ', '_').replace(',', '')}.csv", index=False)
    print("\n== Среднее по фолдам ==")
    for k, cs in res.items():
        m = {x: np.mean([c[x] for c in cs]) for x in ("Score", "D", "M", "V", "S")}
        print(f"  lambda={k[0]:.2f} take={k[1]:26s} Score {m['Score']:6.2f} ± {np.std([c['Score'] for c in cs]):.2f} | D {m['D']:.3f} M {m['M']:.3f} V {m['V']:.3f} S {m['S']:.3f}")


if __name__ == "__main__":
    main()
