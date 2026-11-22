"""Exp3: multi-view fused heightmaps as extra candidates (series = consecutive frames of the same object)."""
import sys
import warnings

warnings.filterwarnings("ignore")
sys.path.insert(0, "/home/cleaner/Cleaner/agents2/volume")
import numpy as np
import pandas as pd

from common import *
from vmodels import *

df = load_rows()
mv = pd.read_csv(HERE + "/mv_rows.csv")
y = np.log(df.volume_ml.values)
ids = df.object_id.values
ref = ["columns_ml", "parts_ml", "hull_ml", "footprint_ml", "gi_env_ml"]
X_ref = design(df, ref, inter="gi_env_ml", knots=(50, 300))
p_i = nested_looo(X_ref, y, ids, fit_irls, IRLS_GRID)
report("ref IRLS", p_i, df)

for gap in (3, 1000):
    m = mv[mv.gap == gap].set_index("frame").loc[df.image_id].reset_index()
    print(f"gap={gap}: frames with >=2 views: {(m.mv_n >= 2).sum()}  mean views {m.mv_n.mean():.2f}")
    for c in ["single_ml", "mv_max_ml", "mv_med_ml", "mv_mean_ml", "mv_hull_ml", "mv_sym_ml", "mv_max_inter_ml"]:
        l = np.log(np.maximum(m[c].values, 1) / df.volume_ml.values)
        med = pd.Series(l).groupby(ids).transform("median").values
        print(f"  {c:16s} within sd={np.std(l - med):.3f} between sd={np.std(med):.3f}")
    d = df.copy()
    for c in ["mv_max_ml", "mv_med_ml", "mv_mean_ml", "mv_hull_ml", "mv_sym_ml", "mv_max_inter_ml", "mv_union_area_m2"]:
        d[c] = m[c].values
    d["mv_n"] = np.log(m.mv_n.values)
    for ex in (["mv_max_ml"], ["mv_med_ml"], ["mv_hull_ml"], ["mv_max_ml", "mv_hull_ml"], ["mv_med_ml", "mv_sym_ml"]):
        X = design(d, ref + ex, inter="gi_env_ml", knots=(50, 300))
        report(f"gap{gap} IRLS ref+{','.join(ex)}", nested_looo(X, y, ids, fit_irls, IRLS_GRID), df)
    # replace single-view candidates with multi-view ones
    X = design(d, ["mv_max_ml", "mv_med_ml", "mv_hull_ml", "footprint_ml", "gi_env_ml"], inter="gi_env_ml", knots=(50, 300))
    report(f"gap{gap} IRLS mv-replace", nested_looo(X, y, ids, fit_irls, IRLS_GRID), df)
    X = design(d, ["mv_max_ml", "mv_med_ml", "mv_hull_ml", "footprint_ml"], inter="mv_max_ml", knots=(50, 300))
    report(f"gap{gap} IRLS mv-only", nested_looo(X, y, ids, fit_irls, IRLS_GRID), df)
