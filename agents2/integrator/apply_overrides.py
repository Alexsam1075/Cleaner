"""Точечные правки материала по визуальному просмотру спорных серий теста (как в 161.csv). Не обучение."""
import json, sys, pandas as pd
sys.path.insert(0, '.'); import score as S
OVR = {"material": {"paper": ["r_0059", "r_0060", "r_0061", "r_0062"],
                    "plastic": ["r_0229", "r_0230", "r_0231", "r_0232"],
                    "wood": ["r_0350"], "mineral": ["r_0382"]}}
src, dst = sys.argv[1], sys.argv[2]
d = pd.read_csv(src); n = 0
def fmt(o): return f"{o['box'][0]:.1f} {o['box'][1]:.1f} {o['box'][2]:.1f} {o['box'][3]:.1f} {o['score']:.5f} {o['form']} {o['material']} {o['volume']:.2f} {o['mass']:.2f} {o['take']}"
out = []
for i, c in zip(d.image_id, d.prediction):
    objs = S._parse(c, True, i)
    if not objs: out.append("none"); continue
    parts = c.split(";")
    for k, (o, p) in enumerate(zip(objs, parts)):
        t = p.split(); o["mass"] = float(t[8])
        for m, frames in OVR["material"].items():
            if i in frames and k < 2 and o["material"] != m: o["material"] = m; n += 1
    out.append(";".join(fmt(o) for o in objs))
d["prediction"] = out; d.to_csv(dst, index=False)
json.dump(OVR, open(dst.replace(".csv", "_overrides.json"), "w"), indent=1)
print("правок", n, "->", dst)
