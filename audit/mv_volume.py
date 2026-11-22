"""Многоракурсный объём: в кластере «тот же предмет» 2–5 кадров с разных сторон. Какая агрегация объёмов по кадрам
ближе к истине: медиана (сейчас), макс, p75, лог-среднее, взвешенное по видимости (valid_frac), max по сырой геометрии
с общей калибровкой. Фолды 1–3, те же детекции и кластеры, score.py."""
import glob
import json
import pickle
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path("/home/cleaner/Cleaner")
sys.path.insert(0, str(ROOT / "audit"))
import full_pipeline as fp  # noqa: E402
from full_pipeline import C, D, FOLDS, MATS, MODELS, VIEWS, S, _iou, serialize, touches_border_px  # noqa: E402
from object_memory import cluster  # noqa: E402

AGGS = {
    "median": lambda v, w, g: float(np.exp(np.median(np.log(v)))),
    "logmean": lambda v, w, g: float(np.exp(np.mean(np.log(v)))),
    "p75": lambda v, w, g: float(np.exp(np.percentile(np.log(v), 75))),
    "max": lambda v, w, g: float(v.max()),
    "w_valid": lambda v, w, g: float(np.exp(np.average(np.log(v), weights=w + 1e-3))),
    "geo_max_cal": lambda v, w, g: float(g),  # калиброванный объём от кадра с макс. сырым parts_ml
    "mid_med_max": lambda v, w, g: float(np.sqrt(np.exp(np.median(np.log(v))) * v.max())),
}


def main():
    enc = fp.Encoders()
    F, meta = fp.train_cache()
    res = defaultdict(list)
    for fold in (1, 2, 3):
        split = json.load(open(FOLDS[fold][0]))
        train_objs = sorted(split["train_object_ids"])
        frames = list(split["validation_real_frames"])
        cam = json.load(open(D / "camera.json"))
        est, take_thr = fp.volume_estimator_for(train_objs, cam)
        dets, embs = pickle.load(open(C / f"fold{fold}_dets.pkl", "rb"))
        predict_mat, _ = fp.material_model(F, meta, train_objs, enc.text, "generic_syn")
        embs_t = {(t, v): embs[(t, v)] for t in MODELS for v in VIEWS}
        P = predict_mat(embs[("clipL", "c14")], fp.zero_shot(embs_t, enc.text, "generic_syn"))
        sys.path.insert(0, str(ROOT / "agents/form"))
        from form_cv import load_bank
        from form_model import FeatureBank, FormModel, GEO_KEYS
        fbank, fy, foid, fsrc, _ = load_bank()
        fm = FormModel(fbank, fy, foid, fsrc, grid={"spec": ["clipL_c14+zs"], "sw": [0.5], "wd": [1e-2]}).fit(train_objs)
        femb = {f"{t}_{v}": embs[(t, v)] for t in MODELS for v in VIEWS}
        femb["geo"] = np.array([[d["geo"][k] for k in GEO_KEYS] for d in dets], np.float32).reshape(len(dets), len(GEO_KEYS))
        p_bulk = fm.predict_bank(FeatureBank(femb, fbank.text))[:, 0]
        cid = cluster(dets, embs, near=0.90, far=1.01, gap=3)
        forms = ["bulk" if pb >= 0.5 else "solid" for pb in p_bulk]
        mats = [MATS[int(p.argmax())] for p in P]
        geos = [d.get("new_geo_by_form", {}).get(f, d["new_geo"]) for d, f in zip(dets, forms)]
        rr = [est.estimate(None, None, None, f, m, geometry=g) for f, m, g in zip(forms, mats, geos)]
        vol0 = np.array([r.volume_ml for r in rr])
        dens = np.array([r.density_g_per_ml for r in rr])
        valid = np.array([g["valid_frac"] for g in geos])
        raw = np.array([g["parts_ml"] for g in geos])
        sol = next(S.labels_to_solution(p) for p in glob.glob(str(C / "val_labels_*.json")) if set(S.labels_to_solution(p).image_id.astype(str)) == set(frames))
        gt = {str(i): S._parse(c, False, str(i)) for i, c in zip(sol.image_id, sol.objects)}
        for name, fn in AGGS.items():
            vols = vol0.copy()
            for c in set(cid):
                ii = np.where(cid == c)[0]
                if len(ii) < 2:
                    continue
                gmax = vol0[ii[int(np.argmax(raw[ii]))]]
                vols[ii] = fn(vol0[ii], valid[ii], gmax)
            per = defaultdict(list)
            for d, pb, f, m, v, dn in zip(dets, p_bulk, forms, mats, vols, dens):
                if d["score"] < 0.08 or touches_border_px(d["box"]):
                    continue
                pm = max(pb, 1 - pb)
                per[d["frame"]].append({"box": d["box"], "score": d["score"] * pm, "form": f, "material": m, "volume": float(v), "mass": float(v * dn),
                                        "take": int(v <= take_thr), "conf": d["score"], "pmax": pm})
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
            res[name].append(comp)
            print(f"  fold {fold} | {name:12s} Score {comp['Score']:6.2f} | V {comp['V']:.3f} S {comp['S']:.3f}", flush=True)
            sub.to_csv(C / f"mv_fold{fold}_{name}.csv", index=False)
    print("\n== Среднее по фолдам ==")
    for name, cs in res.items():
        m = {x: np.mean([c[x] for c in cs]) for x in ("Score", "D", "M", "V", "S")}
        print(f"  {name:12s} Score {m['Score']:6.2f} ± {np.std([c['Score'] for c in cs]):.2f} | D {m['D']:.3f} M {m['M']:.3f} V {m['V']:.3f} S {m['S']:.3f}")


if __name__ == "__main__":
    main()
