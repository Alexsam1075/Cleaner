"""Shared data loading for the TAKE agent: per-annotation table (real + synth), cached embeddings,
zero-shot concept similarities, and nested-LOOO calibrated volumes."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path("/home/cleaner/Cleaner")
HERE = ROOT / "agents" / "take"
CACHE = HERE / "cache"
SHARED = HERE / "shared_cache"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

MATS = ["metal", "wood", "plastic", "glass", "paper", "mineral"]
FORMS = ["bulk", "solid"]
HEAVY = ["b06", "s08", "s19", "s32", "s36", "s45", "s47"]


def load_meta() -> pd.DataFrame:
    df = pd.DataFrame(json.load(open(SHARED / "meta.json")))
    df["real"] = df.src == "real_train"
    df["heavy"] = (df.mass > 500).astype(int)
    df["idx"] = np.arange(len(df))
    return df


def load_feats() -> dict:
    out = {}
    for f in ("features.npz", "features2.npz", "features_ct.npz"):
        d = np.load(SHARED / f, allow_pickle=True)
        for k in d.files:
            out[k] = d[k]
    return out


def geo_table(F: dict) -> pd.DataFrame:
    return pd.DataFrame(F["geo"], columns=[f"g_{k}" for k in F["geo_keys"]])


def zs_concepts(F: dict, tags=("clipL", "siglip"), views=("m10", "c14", "c20")) -> pd.DataFrame:
    """Cosine similarity of every cached image embedding with the heaviness concept bank
    (per-image centred over concepts), averaged over views."""
    T = np.load(CACHE / "zs_text.npz")
    names = list(T["concepts"])
    cols = {}
    for tag in tags:
        acc = None
        for v in views:
            S = F[f"{tag}_{v}"] @ T[f"{tag}_concepts"].T
            S = S - S.mean(1, keepdims=True)
            acc = S if acc is None else acc + S
        acc /= len(views)
        for i, n in enumerate(names):
            cols[f"zs_{tag}_{n}"] = acc[:, i]
    return pd.DataFrame(cols)


def parts_rows() -> pd.DataFrame:
    rows = pd.read_csv(ROOT / "reports" / "volume_parts_gt_rows.csv")
    return rows


def loo_calibrated_volume(rows: pd.DataFrame, material_col: str = "material") -> pd.Series:
    """Nested leave-one-object-out calibrated volume (ml) for every real annotation, using the
    project's class-aware ridge calibrator fitted on the other 39 objects (alpha by inner CV)."""
    from evaluation.volume_parts_cv import apply_calibrator, fit_calibrator
    out = pd.Series(np.nan, index=rows.index)
    for o in sorted(rows.object_id.unique()):
        tr = rows[rows.object_id != o]
        cal = fit_calibrator(tr)
        te = rows[rows.object_id == o]
        for i, r in te.iterrows():
            out[i] = apply_calibrator(cal, r.to_dict(), r["form"], r[material_col])
    return out


def build_table(force: bool = False) -> pd.DataFrame:
    """Full per-annotation table (892 rows, synth included) with geometry, zs sims, parts geometry
    (real only) and the nested-LOOO calibrated volume (real only)."""
    p = CACHE / "table.parquet"
    if p.exists() and not force:
        return pd.read_parquet(p)
    df = load_meta()
    F = load_feats()
    df = pd.concat([df, geo_table(F), zs_concepts(F)], axis=1)
    rows = parts_rows()
    rows["cal_vol_loo"] = loo_calibrated_volume(rows)
    keep = ["image_id", "object_id", "columns_ml", "parts_ml", "hull_ml", "footprint_ml", "valid_frac", "truncated",
            "n_parts", "p95_h_m", "footprint_m2", "fp_area_0", "fp_area_20", "fp_area_40", "fp_area_80", "fp_area_160",
            "min_h_m", "cal_vol_loo"]
    rows = rows[keep].rename(columns={"image_id": "frame", "object_id": "oid"})
    df = df.merge(rows, on=["frame", "oid"], how="left")
    assert len(df) == 892
    df.to_parquet(p)
    return df


if __name__ == "__main__":
    df = build_table(force=True)
    r = df[df.real]
    print(r.groupby("oid")[["vol", "mass", "parts_ml", "cal_vol_loo", "g_vol"]].median().round(0))
