"""VolumeModel: calibrated object volume (ml) from the geometry dict of geometry/volume_parts.estimate_volume.

Model = IRLS (metric-shaped weights) ridge in log space on
    log(columns, parts, hull, footprint [, gi_env]) + form + material + form*material
    + main*material + main*form + hinge(main - ln 50) + hinge(main - ln 300)
with alpha/floor picked by inner grouped 5-fold CV on the training objects.
``gi_env_ml`` (gradient-filtered, hole-inpainted, closed-top envelope volume) is used when available
(``VolumeModel.geometry(depth, mask, plane, form)`` computes it); if the dict lacks it the model falls back to
the base4 variant (same training protocol).  Series aggregation: ``aggregate_series`` (median of log volumes).

Usage:
    vm = VolumeModel().fit(train_object_ids)                # rows from reports/volume_parts_gt_rows.csv (+ cached gi_env)
    v = vm.predict(geometry_dict, form, material)           # ml
    v = vm.predict(geometry_dict, form, material_probs={...})
"""
from __future__ import annotations

import json
import math
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path("/home/cleaner/Cleaner")
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "agents/volume"))

MATS = ["glass", "metal", "mineral", "paper", "plastic", "wood"]
BASE = ["columns_ml", "parts_ml", "hull_ml", "footprint_ml"]
REF = BASE + ["gi_env_ml"]
KNOTS = (50.0, 300.0)
L3 = math.log(3.0)
GRID = [(a, fl) for a in (1, 3, 10, 30, 100) for fl in (0.1, 0.2, 0.3)]
ROWS_CSV = ROOT / "reports/volume_parts_gt_rows.csv"
FEAT2_CSV = ROOT / "agents/volume/cache/feat2_real.csv"


# ---------------------------------------------------------------- linear algebra
def _ridge(X, y, w, a):
    mu = np.average(X, 0, weights=w)
    sd = np.sqrt(np.average((X - mu) ** 2, 0, weights=w))
    sd[sd < 1e-9] = 1.0
    Z = (X - mu) / sd
    ym = float(np.average(y, weights=w))
    coef = np.linalg.solve(Z.T @ (Z * w[:, None]) + a * np.eye(Z.shape[1]), Z.T @ (w * (y - ym)))
    return {"mu": mu, "sd": sd, "ym": ym, "coef": coef}


def _apply(m, X):
    return m["ym"] + ((np.asarray(X, float) - m["mu"]) / m["sd"]) @ m["coef"]


def _irls(X, y, w, a, floor, iters=8):
    m = _ridge(X, y, w, a)
    for _ in range(iters):
        r = np.abs(_apply(m, X) - y)
        m = _ridge(X, y, w * np.where(r < L3, 1.0 / np.maximum(r, floor), 0.1 / L3), a)
    return m


def _vscore(p, y):
    return np.maximum(0.0, 1.0 - np.abs(p - y) / L3)


def _features(est: dict, form: str, material: str, cols: list[str]) -> np.ndarray:
    x = [math.log(max(float(est[c]), 1.0)) for c in cols]
    main = x[-1]
    f = 1.0 if form == "bulk" else 0.0
    mat = [1.0 if material == m else 0.0 for m in MATS]
    out = x + [f] + mat + [f * mm for mm in mat] + [main * mm for mm in mat] + [main * f]
    out += [max(0.0, main - math.log(t)) for t in KNOTS]
    return np.asarray(out, float)


def _design(rows: pd.DataFrame, cols) -> np.ndarray:
    return np.vstack([_features(r, r["form"], r["material"], cols) for r in rows.to_dict("records")])


# ---------------------------------------------------------------- model
class VolumeModel:
    def __init__(self, grid=GRID, inner_folds: int = 5, seed: int = 0):
        self.grid, self.inner_folds, self.seed = list(grid), inner_folds, seed
        self.models: dict[str, dict] = {}

    # -- data
    @staticmethod
    def training_rows() -> pd.DataFrame:
        rows = pd.read_csv(ROWS_CSV)
        rows = rows[rows["split"] == "real"].reset_index(drop=True)
        if FEAT2_CSV.exists():
            f2 = pd.read_csv(FEAT2_CSV)
            if len(f2) == len(rows) and (f2["frame"].values == rows["image_id"].values).all():
                rows["gi_env_ml"] = f2["gi_env_ml"].values
        return rows

    def fit(self, train_object_ids=None, rows: pd.DataFrame | None = None):
        rows = self.training_rows() if rows is None else rows.reset_index(drop=True)
        if train_object_ids is not None:
            rows = rows[rows["object_id"].isin(set(train_object_ids))].reset_index(drop=True)
        y = np.log(rows["volume_ml"].to_numpy(float))
        ids = rows["object_id"].to_numpy()
        cnt = Counter(ids)
        w = np.array([1.0 / cnt[o] for o in ids])
        variants = {"base4": BASE}
        if "gi_env_ml" in rows.columns and rows["gi_env_ml"].notna().all():
            variants["ref5"] = REF
        for name, cols in variants.items():
            X = _design(rows, cols)
            g = self._select(X, y, ids, w)
            self.models[name] = {"cols": cols, "g": g, **_irls(X, y, w, *g)}
        self.n_train_objects = len(cnt)
        return self

    def _select(self, X, y, ids, w):
        uo = np.array(sorted(set(ids)))
        np.random.default_rng(self.seed).shuffle(uo)
        fo = {o: k % self.inner_folds for k, o in enumerate(uo)}
        f = np.array([fo[o] for o in ids])
        best = None
        for g in self.grid:
            p = np.zeros(len(y))
            for k in range(self.inner_folds):
                m = _irls(X[f != k], y[f != k], w[f != k], *g)
                p[f == k] = _apply(m, X[f == k])
            s = float(np.sum(w * _vscore(p, y)) / w.sum())
            if best is None or s > best[0]:
                best = (s, g)
        return best[1]

    # -- inference
    def geometry(self, depth_mm, mask, plane, form, camera: dict | None = None) -> dict:
        """estimate_volume dict + gi_env_ml (the richer candidate used by the ref5 variant)."""
        from geometry.volume_parts import estimate_volume, refine_plane
        camera = camera or json.loads((ROOT / "datasets/raw/camera.json").read_text())
        mask = np.asarray(mask, bool)
        refined = refine_plane(depth_mm, mask, np.asarray(plane, float), camera)
        geo = estimate_volume(depth_mm, mask, refined, camera, form)
        try:
            from geo_features2 import compute_features2
            geo["gi_env_ml"] = float(max(compute_features2(depth_mm, mask, plane, camera, form)["gi_env_ml"], 1.0))
        except Exception:  # noqa: BLE001 - fall back to base4 if the extra feature fails
            pass
        return geo

    def _predict_log(self, est: dict, form: str, material: str) -> float:
        name = "ref5" if ("ref5" in self.models and est.get("gi_env_ml") is not None and float(est.get("gi_env_ml") or 0) > 0) else "base4"
        m = self.models[name]
        return float(_apply(m, _features(est, form, material, m["cols"])[None])[0])

    def predict(self, geometry: dict, form: str, material: str | None = None,
                material_probs: dict[str, float] | None = None) -> float:
        if form not in ("bulk", "solid"):
            raise ValueError(form)
        if material_probs:
            tot = sum(max(p, 0.0) for p in material_probs.values())
            lv = sum(max(p, 0.0) / tot * self._predict_log(geometry, form, m) for m, p in material_probs.items() if p > 0)
        else:
            if material not in MATS:
                raise ValueError(material)
            lv = self._predict_log(geometry, form, material)
        return float(np.clip(math.exp(lv), 1.0, 20000.0))

    @staticmethod
    def aggregate_series(volumes_ml) -> float:
        """Series-level volume: median in log space of per-frame predictions (+0.025 V in LOOO)."""
        v = np.asarray(volumes_ml, float)
        return float(np.exp(np.median(np.log(np.maximum(v, 1.0)))))

    # -- persistence
    def to_json(self) -> dict:
        return {"models": {k: {"cols": v["cols"], "g": list(v["g"]), "mu": v["mu"].tolist(), "sd": v["sd"].tolist(),
                               "ym": v["ym"], "coef": v["coef"].tolist()} for k, v in self.models.items()}}

    @classmethod
    def from_json(cls, d: dict):
        self = cls()
        for k, v in d["models"].items():
            self.models[k] = {"cols": v["cols"], "g": tuple(v["g"]), "mu": np.asarray(v["mu"]), "sd": np.asarray(v["sd"]),
                              "ym": float(v["ym"]), "coef": np.asarray(v["coef"])}
        return self


if __name__ == "__main__":
    vm = VolumeModel().fit()
    out = Path(__file__).resolve().parent / "volume_model_all40.json"
    out.write_text(json.dumps(vm.to_json(), indent=1))
    print("fitted on", vm.n_train_objects, "objects;", {k: v["g"] for k, v in vm.models.items()}, "->", out)
