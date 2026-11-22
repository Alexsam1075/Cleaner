import json, math, sys
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
    gt.setdefault(img[a["image_id"]], []).append({"box": (x, y, x + w, y + h), "form": at["form"], "material": at["material"], "volume": at["volume_ml"], "mass": at["mass_g"], "oid": at["object_id"]})
tag = sys.argv[1]
rows = []
for f in (1, 2, 3, 4):
    sub = pd.read_csv(HERE / "cache" / f"fold{f}_{tag}.csv")
    for i, c in zip(sub.image_id, sub.prediction):
        G = gt.get(i, []); P = S._parse(c, True, i)
        for g, p in zip(G, S._match(G, P)):
            if p is None: continue
            rows.append(dict(fold=f, oid=g["oid"], gform=g["form"], pform=p["form"], gmat=g["material"], pmat=p["material"], lp=math.log(p["volume"]), lg=math.log(g["volume"])))
df = pd.DataFrame(rows)
def vacc(lp, lg): return np.maximum(0, 1 - np.abs(lp - lg) / math.log(3))
print("as is V(pooled, frame-level) =", vacc(df.lp, df.lg).mean().round(4))
# per-object (median per object) fit to avoid frame-count weighting
ob = df.groupby(["fold", "oid"]).agg(lp=("lp", "median"), lg=("lg", "first"), pform=("pform", lambda s: s.mode()[0]), gform=("gform", "first")).reset_index()
b, a = np.polyfit(ob.lp, ob.lg, 1)
print(f"pooled fit (object-level): lg = {a:.3f} + {b:.3f}*lp ; resid sd {np.std(ob.lg-(a+b*ob.lp)):.3f} vs raw sd {np.std(ob.lg-ob.lp):.3f}")
for form in ("bulk", "solid"):
    o = ob[ob.pform == form]
    b2, a2 = np.polyfit(o.lp, o.lg, 1)
    print(f"  {form}: n={len(o)} lg = {a2:.3f} + {b2:.3f}*lp")
# LOFO
out = df.copy(); out["lp2"] = out.lp
for f in (1, 2, 3, 4):
    o = ob[ob.fold != f]
    b, a = np.polyfit(o.lp, o.lg, 1)
    m = df.fold == f
    out.loc[m, "lp2"] = a + b * df.lp[m]
    print(f"  LOFO fold {f}: a={a:.3f} b={b:.3f}  V {vacc(df.lp[m], df.lg[m]).mean():.3f} -> {vacc(out.lp2[m], out.lg[m]).mean():.3f}")
print("LOFO V pooled:", vacc(out.lp2, out.lg).mean().round(4))
# per-form LOFO
out["lp3"] = out.lp
for f in (1, 2, 3, 4):
    for form in ("bulk", "solid"):
        o = ob[(ob.fold != f) & (ob.pform == form)]
        b, a = np.polyfit(o.lp, o.lg, 1)
        m = (df.fold == f) & (df.pform == form)
        out.loc[m, "lp3"] = a + b * df.lp[m]
for f in (1,2,3,4):
    m = df.fold == f
    print(f"  LOFO per-form fold {f}: V {vacc(df.lp[m], df.lg[m]).mean():.3f} -> {vacc(out.lp3[m], out.lg[m]).mean():.3f}")
print("LOFO per-form V pooled:", vacc(out.lp3, out.lg).mean().round(4))
# shrink-only (b fixed slope through mean): lg = mean + b*(lp-mean)
print(ob.assign(r=(ob.lg-ob.lp).round(2)).sort_values("lp").to_string())
