"""Остаток потерь по компонентам: Score по 4 фолдам (pooled и по фолдам), если сопоставленным рамкам подставить
GT-форму / GT-материал / GT-объём / GT-take (по одному и все вместе). Запуск: ~/venv/bin/python oracle.py <tag>"""
import copy
import json
import sys
from pathlib import Path

import pandas as pd

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
tag = sys.argv[1]
MODES = ["as is", "GT form", "GT material", "GT volume", "GT take", "GT form+mat+vol+take"]


def fix(G, P, mode):
    out = {}
    for i in P:
        pr = copy.deepcopy(P[i])
        m = S._match(G[i], pr)
        for g, p in zip(G[i], m):
            if p is None:
                continue
            if "form" in mode:
                p["form"] = g["form"]
            if "mat" in mode:
                p["material"] = g["material"]
            if "vol" in mode:
                p["volume"] = g["volume"]
            if "take" in mode:
                p["take"] = g["take"]
        out[i] = pr
    return out


allG, allP = {}, {}
rows = []
for f in (1, 2, 3, 4):
    sub = pd.read_csv(HERE / "cache" / f"fold{f}_{tag}.csv")
    G, P = {}, {}
    for i, c in zip(sub.image_id, sub.prediction):
        G[i], P[i] = gt.get(i, []), S._parse(c, True, i)
    allG.update(G)
    allP.update(P)
    rows.append((f"ф{f}", {m: S.components(G, fix(G, P, m)) for m in MODES}))
rows.append(("pooled", {m: S.components(allG, fix(allG, allP, m)) for m in MODES}))
print(f"{tag}: Score (D/M/V/S) при подстановке GT сопоставленным рамкам")
print("| фолд | " + " | ".join(MODES) + " |")
print("|---|" + "---:|" * len(MODES))
for name, r in rows:
    print(f"| {name} | " + " | ".join(f"{r[m]['Score']:.2f}" for m in MODES) + " |")
r = rows[-1][1]
print("pooled компоненты: " + "; ".join(f"{m}: D {r[m]['D']:.3f} M {r[m]['M']:.3f} V {r[m]['V']:.3f} S {r[m]['S']:.3f}" for m in MODES))
