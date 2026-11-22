"""Exp8: 1-3 scalar zero-shot 'fill' contrasts from cached CLIP-L / SigLIP crops (bag vs rigid, hollow vs solid,
light vs heavy) as extra calibrator features, with and without a bulk interaction. Nested LOOO."""
import json
import sys
import warnings

warnings.filterwarnings("ignore")
sys.path.insert(0, "/home/cleaner/Cleaner/agents2/volume")
import numpy as np

from common import *
from vmodels import *

df = load_rows()
y = np.log(df.volume_ml.values)
ids = df.object_id.values
ref = ["columns_ml", "parts_ml", "hull_ml", "footprint_ml", "gi_env_ml"]
X_ref = design(df, ref, inter="gi_env_ml", knots=(50, 300))

C = ROOT + "/audit/cache/"
meta = json.load(open(C + "meta.json"))
key = {(m["frame"], m["oid"]): i for i, m in enumerate(meta) if m["src"] == "real_train"}
ridx = np.array([key[(f, o)] for f, o in zip(df.image_id, df.object_id)])
F2 = np.load(C + "features2.npz")
T = np.load(ROOT + "/agents/take/cache/zs_text.npz", allow_pickle=True)
names = list(T["concepts"])


def sims(tag, view):
    E = F2[f"{tag}_{view}"][ridx]
    E = E / np.linalg.norm(E, axis=1, keepdims=True)
    Tt = T[f"{tag}_concepts"]
    Tt = Tt / np.linalg.norm(Tt, axis=1, keepdims=True)
    S = E @ Tt.T
    return S - S.mean(1, keepdims=True)


def contrast(S, pos, neg):
    return S[:, [names.index(p) for p in pos]].mean(1) - S[:, [names.index(n) for n in neg]].mean(1)


feats = {}
for tag in ("clipL", "siglip"):
    S = np.mean([sims(tag, v) for v in ("m10", "c14", "c20")], 0) * 20
    feats[f"{tag}_bag"] = contrast(S, ["bag", "paper_light", "foam_light"], ["rock", "metal_block", "solid_plastic"])
    feats[f"{tag}_hollow"] = contrast(S, ["hollow", "empty_bottle", "empty_can"], ["solid", "full_bottle", "heavy"])
    feats[f"{tag}_light"] = contrast(S, ["light"], ["heavy"])
    feats[f"{tag}_crushed"] = contrast(S, ["crushed"], ["intact"])
for k in feats:
    df[k] = feats[k]
df["bag"] = 0.5 * (df.clipL_bag + df.siglip_bag)
df["hollow"] = 0.5 * (df.clipL_hollow + df.siglip_hollow)
df["light"] = 0.5 * (df.clipL_light + df.siglip_light)
bulk = (df.form == "bulk").to_numpy(float)

# correlation of per-object residual of the reference with each contrast
p_ref = np.load(HERE + "/cache_exp1_ref_irls.npy")
r = p_ref - y
import pandas as pd
g = pd.DataFrame({"r": r, "o": ids, "form": df.form, **{k: df[k] for k in ["bag", "hollow", "light", "clipL_crushed"]}}).groupby("o").agg("median" if False else {"r": "median", "form": "first", "bag": "median", "hollow": "median", "light": "median", "clipL_crushed": "median"})
for fm in ("bulk", "solid"):
    s = g[g.form == fm]
    print(fm, s[["bag", "hollow", "light", "clipL_crushed"]].corrwith(s.r, method="spearman").round(2).to_dict())

report("ref IRLS", nested_looo(X_ref, y, ids, fit_irls, IRLS_GRID), df)
for ex in (["bag"], ["hollow"], ["light"], ["bag", "hollow"], ["clipL_bag"], ["siglip_bag"], ["clipL_crushed"]):
    E = df[ex].to_numpy(float)
    X = design(df, ref, inter="gi_env_ml", knots=(50, 300), extra=E)
    report("IRLS + " + ",".join(ex), nested_looo(X, y, ids, fit_irls, IRLS_GRID), df)
    X = design(df, ref, inter="gi_env_ml", knots=(50, 300), extra=np.c_[E * bulk[:, None], E * (1 - bulk)[:, None]])
    report("IRLS + " + ",".join(ex) + " x form", nested_looo(X, y, ids, fit_irls, IRLS_GRID), df)
