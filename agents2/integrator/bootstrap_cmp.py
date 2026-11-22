"""Bootstrap по объектам (4 фолда, 266 кадров): разница Score двух вариантов (cache/fold{f}_<tag>.csv).
Запуск: ~/venv/bin/python bootstrap_cmp.py <tag_base> <tag1> [<tag2> ...]"""
import json
import sys
from pathlib import Path

import numpy as np
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


def pooled(tag, folds=(1, 2, 3, 4)):
    G, P = {}, {}
    for f in folds:
        sub = pd.read_csv(HERE / "cache" / f"fold{f}_{tag}.csv")
        for i, c in zip(sub.image_id, sub.prediction):
            G[i] = gt.get(i, [])
            P[i] = S._parse(c, True, i)
    return G, P


def main():
    tags = sys.argv[1:]
    res = {t: pooled(t) for t in tags}
    for k, (G, P) in res.items():
        c = S.components(G, P)
        print(f"{k:36s} pooled: Score {c['Score']:.2f} D {c['D']:.3f} M {c['M']:.3f} V {c['V']:.3f} S {c['S']:.3f}")
    oids = sorted({o["oid"] for v in gt.values() for o in v})
    frames_of = {o: set() for o in oids}
    for i, v in gt.items():
        for o in v:
            frames_of[o["oid"]].add(i)
    G0, P0 = res[tags[0]]
    rs = np.random.default_rng(0)
    for t in tags[1:]:
        G1, P1 = res[t]
        diffs = []
        for _ in range(300):
            pick = rs.choice(oids, len(oids))
            Gs, Ps0, Ps1 = {}, {}, {}
            for n, o in enumerate(pick):
                for i in frames_of[o]:
                    if i not in G0:
                        continue
                    key = f"{i}#{n}"
                    Gs[key], Ps0[key], Ps1[key] = G0[i], P0[i], P1[i]
            diffs.append(S.components(Gs, Ps1)["Score"] - S.components(Gs, Ps0)["Score"])
        d = np.array(diffs)
        print(f"Δ({t} − {tags[0]}) = {d.mean():+.2f}, 95% [{np.percentile(d, 2.5):+.2f}, {np.percentile(d, 97.5):+.2f}], P(Δ>0)={(d > 0).mean():.2f}")


if __name__ == "__main__":
    main()
