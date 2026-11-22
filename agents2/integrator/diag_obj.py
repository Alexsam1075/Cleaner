"""Диагностика по объектам (сопоставленные рамки, 4 фолда): форма/материал/объём/take против GT. Запуск: python diag_obj.py <tag>"""
import json, math, sys
from collections import defaultdict
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, "/home/cleaner/Cleaner/references")
import score as S
HERE = Path(__file__).resolve().parent
D = Path("/home/cleaner/Cleaner/datasets/raw")
lab = json.load(open(D / "labels/real_train.json"))
img = {im["id"]: Path(im["file_name"]).parent.name for im in lab["images"]}
gt = {}
for a in lab["annotations"]:
    x, y, w, h = a["bbox"]; at = a["attributes"]
    gt.setdefault(img[a["image_id"]], []).append({"box": (x, y, x + w, y + h), "form": at["form"], "material": at["material"],
        "volume": at["volume_ml"], "mass": at["mass_g"], "take": int(at["mass_g"] <= 500), "oid": at["object_id"]})
tag = sys.argv[1]
rows = []
for f in (1, 2, 3, 4):
    sub = pd.read_csv(HERE / "cache" / f"fold{f}_{tag}.csv")
    for i, c in zip(sub.image_id, sub.prediction):
        G = gt.get(i, []); P = S._parse(c, True, i)
        for g, p in zip(G, S._match(G, P)):
            r = dict(fold=f, oid=g["oid"], frame=i, gform=g["form"], gmat=g["material"], gvol=g["volume"], gmass=g["mass"], gtake=g["take"], matched=p is not None)
            if p is not None:
                r.update(pform=p["form"], pmat=p["material"], pvol=p["volume"], ptake=p["take"], pmass=None)
                r["vacc"] = max(0, 1 - abs(math.log(p["volume"] / g["volume"])) / math.log(3))
                r["lr"] = math.log(p["volume"] / g["volume"])
            rows.append(r)
df = pd.DataFrame(rows)
print("объектов-кадров", len(df), "несопоставлено", (~df.matched).sum())
m = df[df.matched]
agg = m.groupby(["fold", "oid"]).agg(n=("frame", "size"), gform=("gform", "first"), gmat=("gmat", "first"), gvol=("gvol", "first"), gmass=("gmass", "first"), gtake=("gtake", "first"),
    form_err=("pform", lambda s: int((s != m.loc[s.index, "gform"]).sum())), mat_err=("pmat", lambda s: int((s != m.loc[s.index, "gmat"]).sum())),
    take_err=("ptake", lambda s: int((s != m.loc[s.index, "gtake"]).sum())), vacc=("vacc", "mean"), lr=("lr", "median"), pvol=("pvol", "median"), ptake=("ptake", "mean"))
pd.set_option("display.width", 250); pd.set_option("display.max_rows", 300)
print(agg.round(2).to_string())
print("\nS по фолдам (heavy/light):")
for f in (1, 2, 3, 4):
    x = df[df.fold == f]
    for t in (0, 1):
        y = x[x.gtake == t]
        print(f"  ф{f} take={t}: n={len(y)} acc={(y.matched & (y.ptake == t)).mean():.3f}")
print("\nтяжёлые (>500 г) объекты, предсказание take=1 (ошибка):")
print(agg[(agg.gtake == 0) & (agg.take_err > 0)].round(2).to_string())
print("\nлёгкие объекты с ошибкой take:")
print(agg[(agg.gtake == 1) & (agg.take_err > 0)].round(2).to_string())
