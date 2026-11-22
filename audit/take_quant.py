"""Порог take по распределению объёмов самого тестового набора (без меток): доля «объехать» среди кластеров-предметов
должна быть близка к доле тяжёлых в обучении (7/40 = 17.5%). Варианты: train-порог (сейчас), квантиль теста, смесь.
Поверх «память + 2 формы», объём по кластеру — медиана. Фолды 1–3, score.py."""
import glob
import json
import math
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

HEAVY_FRAC = 7 / 40


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
        vols = np.array([r.volume_ml for r in rr])
        dens = np.array([r.density_g_per_ml for r in rr])
        for c in set(cid):
            ii = np.where(cid == c)[0]
            if len(ii) > 1:
                vols[ii] = float(np.exp(np.median(np.log(np.maximum(vols[ii], 1.0)))))
        ok = np.array([d["score"] >= 0.08 and not touches_border_px(d["box"]) for d in dets])
        # объём на кластер-предмет (один голос на предмет), только принятые рамки с conf>=0.3 (чтобы FP не портили квантиль)
        strong = ok & (np.array([d["score"] for d in dets]) >= 0.3)
        cl_vol = np.array([vols[(cid == c) & strong].mean() for c in set(cid[strong])])
        q_thr = float(np.quantile(cl_vol, 1 - HEAVY_FRAC))
        sol = next(S.labels_to_solution(p) for p in glob.glob(str(C / "val_labels_*.json")) if set(S.labels_to_solution(p).image_id.astype(str)) == set(frames))
        gt = {str(i): S._parse(c, False, str(i)) for i, c in zip(sol.image_id, sol.objects)}
        n_heavy_gt = sum(1 for i in gt for g in gt[i] if g["take"] == 0)
        n_gt = sum(len(v) for v in gt.values())
        variants = {"train-порог": take_thr, "квантиль теста": q_thr, "смесь (геом. среднее)": math.sqrt(take_thr * q_thr),
                    "max(train, квантиль)": max(take_thr, q_thr)}
        print(f"== fold {fold}: train-порог {take_thr:.0f} мл, квантиль теста {q_thr:.0f} мл; тяжёлых в GT {n_heavy_gt}/{n_gt} кадров-объектов", flush=True)
        for name, thr in variants.items():
            per = defaultdict(list)
            for d, pb, f, m, v, dn, o in zip(dets, p_bulk, forms, mats, vols, dens, ok):
                if not o:
                    continue
                pm = max(pb, 1 - pb)
                per[d["frame"]].append({"box": d["box"], "score": d["score"] * pm, "form": f, "material": m, "volume": float(v), "mass": float(v * dn),
                                        "take": int(v <= thr), "conf": d["score"], "pmax": pm})
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
            share0 = np.mean([p["take"] == 0 for i in pr for p in pr[i]])
            print(f"  {name:24s} порог {thr:6.0f} | доля take=0 {share0:.0%} | Score {comp['Score']:6.2f} | S {comp['S']:.3f}", flush=True)
            sub.to_csv(C / f"tq_fold{fold}_{name.split()[0]}.csv", index=False)
    print("\n== Среднее по фолдам ==")
    for name, cs in res.items():
        m = {x: np.mean([c[x] for c in cs]) for x in ("Score", "D", "M", "V", "S")}
        print(f"  {name:24s} Score {m['Score']:6.2f} ± {np.std([c['Score'] for c in cs]):.2f} | D {m['D']:.3f} M {m['M']:.3f} V {m['V']:.3f} S {m['S']:.3f}")


if __name__ == "__main__":
    main()
