"""Разбор D по формам: AP bulk/solid, матрица ошибок формы по сопоставленным рамкам (IoU≥0.5), ошибки по объектам.
Запуск: ~/venv/bin/python form_cm.py <tag1> [<tag2> ...] [--folds 1 2 3 4]"""
import argparse
import copy
import json
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd
import sys

sys.path.insert(0, "/home/cleaner/Cleaner/references")
import score as S  # noqa: E402

HERE = Path(__file__).resolve().parent
D = Path("/home/cleaner/Cleaner/datasets/raw")
lab = json.load(open(D / "labels/real_train.json"))
img = {im["id"]: Path(im["file_name"]).parent.name for im in lab["images"]}
gt = {}
for a in lab["annotations"]:
    x, y, w, h = a["bbox"]
    at = a["attributes"]
    gt.setdefault(img[a["image_id"]], []).append({"box": (x, y, x + w, y + h), "form": at["form"], "material": at["material"],
                                                   "volume": at["volume_ml"], "take": int(at["mass_g"] <= 500), "oid": at["object_id"]})


def run(tag, folds):
    tot = Counter()
    wrong_all = defaultdict(list)
    for f in folds:
        p = HERE / "cache" / f"fold{f}_{tag}.csv"
        if not p.exists():
            continue
        sub = pd.read_csv(p)
        G, P = {}, {}
        cm = Counter()
        wrong = defaultdict(int)
        nobj = defaultdict(int)
        for i, c in zip(sub.image_id, sub.prediction):
            G[i] = gt.get(i, [])
            P[i] = S._parse(c, True, i)
            m = S._match(G[i], copy.deepcopy(P[i]))
            for g, pr in zip(G[i], m):
                nobj[g["oid"]] += 1
                if pr is not None:
                    cm[(g["form"], pr["form"])] += 1
                    if pr["form"] != g["form"]:
                        wrong[g["oid"]] += 1
                else:
                    cm[(g["form"], "miss")] += 1
        ap = {k: S._average_precision(G, P, k) for k in ("bulk", "solid")}
        comp = S.components(G, P)
        tot.update(cm)
        ws = ", ".join(f"{o}:{wrong[o]}/{nobj[o]}" for o in sorted(wrong))
        print(f"  ф{f} D {comp['D']:.3f} AP_bulk {ap['bulk']:.3f} AP_solid {ap['solid']:.3f} | bulk→bulk {cm[('bulk','bulk')]} bulk→solid {cm[('bulk','solid')]} "
              f"solid→solid {cm[('solid','solid')]} solid→bulk {cm[('solid','bulk')]} miss {cm[('bulk','miss')]+cm[('solid','miss')]} | ошибки: {ws}")
        for o in wrong:
            wrong_all[o].append((f, wrong[o], nobj[o]))
    print(f"  всего: bulk→bulk {tot[('bulk','bulk')]} bulk→solid {tot[('bulk','solid')]} solid→solid {tot[('solid','solid')]} solid→bulk {tot[('solid','bulk')]}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("tags", nargs="+")
    ap.add_argument("--folds", type=int, nargs="+", default=[1, 2, 3, 4])
    a = ap.parse_args()
    for t in a.tags:
        print(t)
        run(t, a.folds)
