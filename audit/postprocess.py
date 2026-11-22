"""Пересчёт сабмитов валидации с постобработкой через оригинальный score.py.
Запуск из корня проекта Cleaner (где logs/validation/ и references/score.py):
    python <путь>/audit/postprocess.py [--densities datasets/raw/densities.csv]"""
import argparse
import copy
import csv
import sys
from pathlib import Path

import pandas as pd

ap = argparse.ArgumentParser()
ap.add_argument("--root", default=".")
ap.add_argument("--densities", default="datasets/raw/densities.csv")
ap.add_argument("--thr", type=float, default=500.0)
a = ap.parse_args()
root = Path(a.root)
sys.path.insert(0, str(root / "references"))
import score as S  # noqa: E402

dens = {f"{r['form']}_{r['material']}": float(r["density_g_per_ml"]) for r in csv.DictReader(open(root / a.densities))}


def load(d, f, gt):
    s = pd.read_csv(d / f)
    m = dict(zip(s.image_id.astype(str), s.prediction))
    return {i: S._parse(m[i], True, i) for i in gt}


def postprocess(mm, geo, use_geo=True, rule_take=True, dup=True, thr=500.0):
    pr = copy.deepcopy(mm)
    for i in pr:
        extra = []
        for p in pr[i]:
            if use_geo:
                g = max(geo[i], key=lambda q: S._iou(p["box"], q["box"]), default=None)
                if g is not None and S._iou(p["box"], g["box"]) > 0.7:
                    p["volume"] = g["volume"]
            if rule_take:
                p["take"] = int(p["volume"] * dens[p["form"] + "_" + p["material"]] <= thr)
            if dup:
                q = dict(p)
                q["form"] = "solid" if p["form"] == "bulk" else "bulk"
                q["score"] = p["score"] * 0.05
                extra.append(q)
        pr[i] += extra
    return pr


for d in sorted((root / "logs/validation").glob("*/*")):
    if not (d / "multimodal_submission.csv").exists():
        continue
    sol = S.labels_to_solution(d / "val_labels.json")
    gt = {i: S._parse(c, False, i) for i, c in zip(sol.image_id.astype(str), sol.objects)}
    mm = load(d, "multimodal_submission.csv", gt)
    geo = load(d, "baseline_logic_same_detector_submission.csv", gt)
    print(f"== {d.parent.name}/{d.name}")
    for name, kw in [("как есть", None), ("+геом. объём", dict(rule_take=False, dup=False)),
                     ("+take по массе", dict(dup=False)), ("+дубль формы (всё)", dict())]:
        pr = mm if kw is None else postprocess(mm, geo, thr=a.thr, **kw)
        c = S.components(gt, pr)
        print(f"   {name:20s} Score {c['Score']:6.2f} | D {c['D']:.3f} M {c['M']:.3f} V {c['V']:.3f} S {c['S']:.3f}")
