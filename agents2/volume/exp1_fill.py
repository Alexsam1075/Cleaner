"""Exp1: reproduce reference; fill-ratio targets per form; void features; quantile/IRLS; class prior shrink."""
import sys
import warnings

warnings.filterwarnings("ignore")
sys.path.insert(0, "/home/cleaner/Cleaner/agents2/volume")
import numpy as np
import pandas as pd

from common import *
from vmodels import *

df = load_rows()
y = np.log(df.volume_ml.values)
ids = df.object_id.values
form = df.form.values
base = ["columns_ml", "parts_ml", "hull_ml", "footprint_ml"]
ref = base + ["gi_env_ml"]

X_ref = design(df, ref, inter="gi_env_ml", knots=(50, 300))
p_ref_r = nested_looo(X_ref, y, ids, fit_ridge, RIDGE_GRID)
report("ref ridge", p_ref_r, df)
p_ref_i = nested_looo(X_ref, y, ids, fit_irls, IRLS_GRID)
report("ref IRLS", p_ref_i, df)

# ---- per-form models on fill ratio target with void features
void_feats = ["gap", "fill", "valid_frac", "hole_inner_frac", "hole_edge_frac", "flying_frac", "mask_solidity", "n_parts"]
shape_feats = ["p95_h_m", "inp_h_med_m", "inp_h_std_m", "cos_tilt"]


def per_form(fitter, grid, cols, inter, knots, extra_cols, offset_col=None, name=""):
    pred = np.zeros(len(y))
    for fm in ("bulk", "solid"):
        s = form == fm
        d = df[s]
        X = design(d, cols, inter=inter, knots=knots, extra=d[extra_cols].to_numpy(float) if extra_cols else None)
        yy = y[s]
        off = np.log(np.maximum(d[offset_col].to_numpy(float), 1.0)) if offset_col else 0.0
        pred[s] = off + nested_looo(X, yy - off, ids[s], fitter, grid)
    return report(name, pred, df)


per_form(fit_ridge, RIDGE_GRID, ref, "gi_env_ml", (50, 300), [], name="per-form ridge (ref feats)")
per_form(fit_irls, IRLS_GRID, ref, "gi_env_ml", (50, 300), [], name="per-form IRLS (ref feats)")
per_form(fit_ridge, RIDGE_GRID, ref, "gi_env_ml", (50, 300), void_feats, name="per-form ridge + void")
per_form(fit_irls, IRLS_GRID, ref, "gi_env_ml", (50, 300), void_feats, name="per-form IRLS + void")
per_form(fit_ridge, RIDGE_GRID, ref, "gi_env_ml", (50, 300), void_feats, offset_col="gi_env_ml", name="per-form ridge fill-ratio + void")
per_form(fit_irls, IRLS_GRID, ref, "gi_env_ml", (50, 300), void_feats, offset_col="gi_env_ml", name="per-form IRLS fill-ratio + void")
per_form(fit_irls, IRLS_GRID, ref, "gi_env_ml", (50, 300), void_feats + shape_feats, offset_col="gi_env_ml", name="per-form IRLS fill-ratio + void+shape")

# joint model with void features (ref design + extras)
for ex in (void_feats, ["gap", "fill"], ["fill"], ["gap"], ["cos_tilt"], ["hole_edge_frac", "fill"]):
    X = design(df, ref, inter="gi_env_ml", knots=(50, 300), extra=df[ex].to_numpy(float))
    report("joint IRLS + " + ",".join(ex), nested_looo(X, y, ids, fit_irls, IRLS_GRID), df)

# add visual-hull / sym-hull candidates for solids (correlate 0.73 with fill ratio for solids)
for ex in (["vhc_ml"], ["vh15_ml"], ["inp_sym_box_ml"], ["vhc_ml", "inp_sym_box_ml"]):
    X = design(df, ref + ex, inter="gi_env_ml", knots=(50, 300))
    report("joint IRLS + " + ",".join(ex), nested_looo(X, y, ids, fit_irls, IRLS_GRID), df)
    per_form(fit_irls, IRLS_GRID, ref + ex, "gi_env_ml", (50, 300), [], name="per-form IRLS + " + ",".join(ex))

np.save(HERE + "/cache_exp1_ref_irls.npy", p_ref_i)
