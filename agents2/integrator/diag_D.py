"""Диагностика D: потери из-за формы vs из-за рамок (FP/пропуски). Аргумент: tag CSV (cache/fold{f}_<tag>.csv)."""
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
for f in (1, 2, 3, 4):
    sub = pd.read_csv(HERE / "cache" / f"fold{f}_{tag}.csv")
    G, P, P2, P3 = {}, {}, {}, {}
    fp_info, miss_info = [], []
    for i, c in zip(sub.image_id, sub.prediction):
        G[i] = gt.get(i, [])
        P[i] = S._parse(c, True, i)
        pr = copy.deepcopy(P[i])
        m = S._match(G[i], pr)
        for g, p in zip(G[i], m):
            if p is not None:
                p["form"] = g["form"]
            else:
                miss_info.append((i, g["oid"]))
        P2[i] = pr
        matched = [p for p in m if p is not None]
        P3[i] = [p for p in pr if any(p is q for q in matched)]
        for p in pr:
            if not any(p is q for q in matched):
                fp_info.append((i, p["form"], round(p["score"], 2)))
    c0, c1, c2 = S.components(G, P), S.components(G, P2), S.components(G, P3)
    print(f"fold {f}: D {c0['D']:.3f} | с GT-формой {c1['D']:.3f} | + без FP {c2['D']:.3f} | "
          f"FP {len(fp_info)} из {sum(len(P[i]) for i in P)} рамок, пропусков GT {len(miss_info)} из {sum(len(G[i]) for i in G)}")
    print("   FP:", fp_info[:12])
    print("   пропуски:", miss_info[:12])
