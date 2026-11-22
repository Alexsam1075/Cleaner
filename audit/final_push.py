"""Последние варианты поверх финала (память + 2 формы): правила take с учётом материала/плотности и ансамбль описаний.
Фолды 1–3, score.py. Все пороги/плотности — только по train-объектам фолда."""
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

HEAVY = {"metal", "glass", "mineral"}


def class_density(meta, train_objs):
    by = defaultdict(list)
    seen = set()
    for m in meta:
        if m["src"] == "real_train" and m["oid"] in train_objs and m["oid"] not in seen:
            seen.add(m["oid"])
            by[(m["form"], m["mat"])].append(m["mass"] / m["vol"])
            by[("*", m["mat"])].append(m["mass"] / m["vol"])
    return {k: float(np.exp(np.mean(np.log(v)))) for k, v in by.items()}


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
        dens = class_density(meta, set(train_objs))
        dets, embs = pickle.load(open(C / f"fold{fold}_dets.pkl", "rb"))
        embs_t = {(t, v): embs[(t, v)] for t in MODELS for v in VIEWS}
        Pm = {}
        for sname in ("generic_syn", "rich"):
            pm, _ = fp.material_model(F, meta, train_objs, enc.text, sname)
            Pm[sname] = pm(embs[("clipL", "c14")], fp.zero_shot(embs_t, enc.text, sname))
        lp = 0.5 * (np.log(Pm["generic_syn"] + 1e-9) + np.log(Pm["rich"] + 1e-9))
        Pm["ens"] = np.exp(lp - lp.max(1, keepdims=True))
        Pm["ens"] /= Pm["ens"].sum(1, keepdims=True)
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
        print(f"== fold {fold}: порог объёма {take_thr:.0f} мл", flush=True)
        for sname in ("generic_syn", "rich", "ens"):
            P = Pm[sname]
            forms = ["bulk" if pb >= 0.5 else "solid" for pb in p_bulk]
            mats = [MATS[int(p.argmax())] for p in P]
            vols = np.array([est.estimate(None, None, None, f, m, geometry=d.get("new_geo_by_form", {}).get(f, d["new_geo"])).volume_ml for d, f, m in zip(dets, forms, mats)])
            for c in set(cid):
                ii = np.where(cid == c)[0]
                if len(ii) > 1:
                    vols[ii] = float(np.exp(np.median(np.log(np.maximum(vols[ii], 1.0)))))
            mass_cls = np.array([v * dens.get((f, m), dens.get(("*", m), 0.5)) for v, f, m in zip(vols, forms, mats)])
            rules = {
                "объём ≤ T": lambda v, m, mc: int(v <= take_thr),
                "масса(класс) ≤ 500": lambda v, m, mc: int(mc <= 500),
                "гибрид: тяжёлые классы по объёму, лёгкие по массе": lambda v, m, mc: int(v <= take_thr) if m in HEAVY else int(mc <= 500),
                "гибрид: объём≤T ИЛИ (лёгкий класс и масса≤500)": lambda v, m, mc: int(v <= take_thr or (m not in HEAVY and mc <= 500)),
                "объём ≤ 1.5T для лёгких классов": lambda v, m, mc: int(v <= (take_thr if m in HEAVY else 1.5 * take_thr)),
            }
            for rname, rule in rules.items():
                per = defaultdict(list)
                for d, pb, f, m, v, mc in zip(dets, p_bulk, forms, mats, vols, mass_cls):
                    if d["score"] < 0.08 or touches_border_px(d["box"]):
                        continue
                    pm_ = max(pb, 1 - pb)
                    per[d["frame"]].append({"box": d["box"], "score": d["score"] * pm_, "form": f, "material": m, "volume": float(v), "mass": float(mc),
                                            "take": rule(v, m, mc), "conf": d["score"], "pmax": pm_})
                for fr in per:
                    keep = []
                    for c in sorted(per[fr], key=lambda o: -o["score"]):
                        if all(_iou(c["box"], k["box"]) < 0.6 for k in keep):
                            keep.append(c)
                    per[fr] = sorted(keep + [dict(c, form="solid" if c["form"] == "bulk" else "bulk", score=c["conf"] * (1 - c["pmax"])) for c in keep], key=lambda o: -o["score"])[:30]
                sub = pd.DataFrame({"image_id": frames, "prediction": [serialize(per.get(f_, [])) for f_ in frames]})
                pr = {str(i): S._parse(c, True, str(i)) for i, c in zip(sub.image_id, sub.prediction)}
                comp = S.components(gt, pr)
                res[(sname, rname)].append(comp)
                print(f"  {sname:11s} | {rname:48s} Score {comp['Score']:6.2f} | M {comp['M']:.3f} V {comp['V']:.3f} S {comp['S']:.3f}", flush=True)
    print("\n== Среднее по фолдам ==")
    for k, cs in sorted(res.items(), key=lambda kv: -np.mean([c["Score"] for c in kv[1]])):
        m = {x: np.mean([c[x] for c in cs]) for x in ("Score", "D", "M", "V", "S")}
        print(f"  {k[0]:11s} | {k[1]:48s} Score {m['Score']:6.2f} ± {np.std([c['Score'] for c in cs]):.2f} | D {m['D']:.3f} M {m['M']:.3f} V {m['V']:.3f} S {m['S']:.3f}")


if __name__ == "__main__":
    main()
