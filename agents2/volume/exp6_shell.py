"""Exp6: physically-motivated boundary-layer ('shell') correction as a replacement/complement for the log-knots;
nested knot selection (honesty check); base4-only variants (pipeline geometry dict has no gi_env_ml)."""
import math
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
base = ["columns_ml", "parts_ml", "hull_ml", "footprint_ml"]
ref = base + ["gi_env_ml"]
CELL = 0.004
# shell: one cell-wide ring around the footprint times the object height; log ratio to the main estimate
h = np.maximum(df.p95_h_m.values, 0.004)
area = np.maximum(df.footprint_m2.values, 1e-5)
perim = 2 * np.sqrt(math.pi * area)             # perimeter of an equivalent disc
df["shell_ratio"] = np.log1p(perim * CELL * 1e6 * h / np.maximum(df.gi_env_ml.values, 1))   # ring volume / env volume
df["shell_ratio_b"] = np.log1p(perim * CELL * 1e6 * h / np.maximum(df.parts_ml.values, 1))
df["hcell_ratio"] = np.log1p(CELL / h)                                                        # one cell of height noise / height
df["inv_h"] = 1.0 / (h * 1000)                                                                 # 1/mm
df["inv_cbrt"] = 1.0 / np.cbrt(np.maximum(df.gi_env_ml.values, 1))                             # 1/characteristic length

X_ref = design(df, ref, inter="gi_env_ml", knots=(50, 300))
p_ref = nested_looo(X_ref, y, ids, fit_irls, IRLS_GRID)
report("ref IRLS (knots 50/300)", p_ref, df)
for ex in (["shell_ratio"], ["hcell_ratio"], ["inv_h"], ["inv_cbrt"], ["shell_ratio", "hcell_ratio"], ["inv_h", "inv_cbrt"]):
    X = design(df, ref, inter="gi_env_ml", knots=(), extra=df[ex].to_numpy(float))
    report("no knots + " + ",".join(ex), nested_looo(X, y, ids, fit_irls, IRLS_GRID), df)
    X = design(df, ref, inter="gi_env_ml", knots=(50, 300), extra=df[ex].to_numpy(float))
    report("knots + " + ",".join(ex), nested_looo(X, y, ids, fit_irls, IRLS_GRID), df)


# nested selection of knots (honest) : grid of knot sets x alpha, floor fixed 0.2
def fit_knotsel(Xall, yt, w, g):
    kset, a = g
    cols = KSLICE[kset]
    m = irls(Xall[:, cols], yt, w, a, floor=0.2)
    return lambda Xn: m(np.asarray(Xn)[:, cols])


KNOT_SETS = [(50, 300), (50, 300, 1000), (30, 200), (100, 500), (30, 100, 300), (50,), (300,)]
allk = sorted({t for ks in KNOT_SETS for t in ks})
X0 = design(df, ref, inter="gi_env_ml", knots=())
main = np.log(np.maximum(df.gi_env_ml.values, 1))
Xk = np.hstack([X0] + [np.maximum(0, main - math.log(t))[:, None] for t in allk])
n0 = X0.shape[1]
KSLICE = {ks: list(range(n0)) + [n0 + allk.index(t) for t in ks] for ks in KNOT_SETS}
grid = [(ks, a) for ks in KNOT_SETS for a in (3, 10, 30, 100)]
p, picks = nested_looo(Xk, y, ids, fit_knotsel, grid, return_picks=True)
report("IRLS nested knot selection", p, df)
from collections import Counter
print("  picked knot sets:", Counter(k for k, _ in picks))

# base4-only (pipeline geometry dict) variants
Xb = design(df, base, inter="parts_ml", knots=(50, 300))
p_b = nested_looo(Xb, y, ids, fit_irls, IRLS_GRID)
report("base4 IRLS main=parts", p_b, df)
Xb = design(df, base, inter="columns_ml", knots=(50, 300))
report("base4 IRLS main=columns", nested_looo(Xb, y, ids, fit_irls, IRLS_GRID), df)
Xb = design(df, base, inter="footprint_ml", knots=(50, 300))
report("base4 IRLS main=footprint", nested_looo(Xb, y, ids, fit_irls, IRLS_GRID), df)
df["geo_mean_ml"] = np.exp(np.mean([np.log(np.maximum(df[c].values, 1)) for c in base], 0))
Xb = design(df, base + ["geo_mean_ml"], inter="geo_mean_ml", knots=(50, 300))
report("base4 IRLS main=geomean", nested_looo(Xb, y, ids, fit_irls, IRLS_GRID), df)
Xb = design(df, base + ["p95_h_m", "footprint_m2"], inter="parts_ml", knots=(50, 300))
report("base4+h+fp IRLS", nested_looo(Xb, y, ids, fit_irls, IRLS_GRID), df)
np.save(HERE + "/cache_exp6_base4_irls.npy", p_b)
