"""Grouped-by-object CV for the parts-based volume estimate and its class-aware log calibration.

Usage (server): ``python -m evaluation.volume_parts_cv --data datasets/raw --folds 5``
"""
from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

FORMS = ["bulk", "solid"]
MATERIALS = ["glass", "metal", "mineral", "paper", "plastic", "wood"]
EST_KEYS = ["columns_ml", "parts_ml", "hull_ml", "footprint_ml"]
ALPHAS = [0.3, 1.0, 3.0, 10.0, 30.0]
BINS = [(0, 50), (50, 200), (200, 1000), (1000, 1e9)]


def volume_score(pred, true) -> float:
    """Official V: mean of max(0, 1 - |ln(p/g)| / ln 3); non-positive prediction scores 0."""
    out = []
    for p, g in zip(np.asarray(pred, dtype=float), np.asarray(true, dtype=float)):
        out.append(0.0 if p <= 0 or g <= 0 else max(0.0, 1.0 - abs(math.log(p / g)) / math.log(3.0)))
    return float(np.mean(out)) if out else 0.0


def group_folds(object_ids, n_folds: int = 5, seed: int = 0) -> list[tuple[np.ndarray, np.ndarray]]:
    """Folds over unique objects (shuffled by seed); every frame of an object lands in one fold."""
    ids = np.asarray(object_ids)
    uniq = np.array(sorted(set(ids.tolist())))
    np.random.default_rng(seed).shuffle(uniq)
    fold_of = {o: k % n_folds for k, o in enumerate(uniq)}
    f = np.array([fold_of[o] for o in ids])
    return [(np.nonzero(f != k)[0], np.nonzero(f == k)[0]) for k in range(n_folds)]


# ---------------------------------------------------------------- calibration

def _class_heights(rows: pd.DataFrame) -> dict:
    ok = rows[rows["n_parts"] > 0]
    by_cls = ok.groupby(["form", "material"])["p95_h_m"].median()
    by_form = ok.groupby("form")["p95_h_m"].median()
    out = {f"{f}|{m}": float(v) for (f, m), v in by_cls.items()}
    out.update({f"{f}|*": float(v) for f, v in by_form.items()})
    out["*|*"] = float(ok["p95_h_m"].median()) if len(ok) else 0.05
    return out


def _class_height(heights: dict, form: str, material: str) -> float:
    for key in (f"{form}|{material}", f"{form}|*", "*|*"):
        if key in heights and heights[key] > 0:
            return float(heights[key])
    return 0.05


def _features(est: dict, form: str, material: str, heights: dict) -> np.ndarray:
    vals = {k: float(est[k]) for k in EST_KEYS}
    if int(est["n_parts"]) == 0:              # no usable depth: footprint x typical class height
        v = max(float(est["footprint_m2"]) * _class_height(heights, form, material) * 0.6 * 1e6, 1.0)
        vals = {k: v for k in EST_KEYS}
    x = [math.log(max(vals[k], 1.0)) for k in EST_KEYS]
    x += [float(est["valid_frac"]), float(bool(est["truncated"])), float(min(int(est["n_parts"]), 4)),
          float(int(est["n_parts"]) == 0)]
    x += [float(form == f) for f in FORMS]
    x += [float(material == m) for m in MATERIALS]
    x += [float(form == f and material == m) for f in FORMS for m in MATERIALS]
    # class-specific slope on the main estimate lets each class rescale its own bias
    x += [x[1] * float(material == m) for m in MATERIALS]
    return np.asarray(x, dtype=np.float64)


def _weights(object_ids) -> np.ndarray:
    ids = list(object_ids)
    count = defaultdict(int)
    for o in ids:
        count[o] += 1
    return np.asarray([1.0 / count[o] for o in ids])


def _ridge(X: np.ndarray, y: np.ndarray, w: np.ndarray, alpha: float) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    mu = np.average(X, axis=0, weights=w)
    sd = np.sqrt(np.average((X - mu) ** 2, axis=0, weights=w))
    sd[sd < 1e-9] = 1.0
    Z = (X - mu) / sd
    ym = float(np.average(y, weights=w))
    A = Z.T @ (Z * w[:, None]) + alpha * np.eye(Z.shape[1])
    coef = np.linalg.solve(A, Z.T @ (w * (y - ym)))
    return coef, mu, sd, ym


def _design(rows: pd.DataFrame, heights: dict) -> np.ndarray:
    return np.vstack([_features(r, r["form"], r["material"], heights) for r in rows.to_dict("records")])


def fit_calibrator(rows: pd.DataFrame, alpha: float | None = None, inner_folds: int = 4) -> dict:
    """Weighted ridge on log(volume); alpha chosen by inner grouped CV on these rows only."""
    heights = _class_heights(rows)
    X = _design(rows, heights)
    y = np.log(rows["volume_ml"].to_numpy(dtype=float))
    w = _weights(rows["object_id"])
    if alpha is None:
        ids = rows["object_id"].to_numpy()
        n_inner = min(inner_folds, len(set(ids)))
        best = None
        for a in ALPHAS:
            pred = np.zeros(len(y))
            for tr, va in group_folds(ids, n_inner, seed=123):
                coef, mu, sd, ym = _ridge(X[tr], y[tr], w[tr], a)
                pred[va] = ym + ((X[va] - mu) / sd) @ coef
            s = volume_score(np.exp(pred), np.exp(y))
            if best is None or s > best[0]:
                best = (s, a)
        alpha = best[1]
    coef, mu, sd, ym = _ridge(X, y, w, alpha)
    return {"alpha": float(alpha), "coef": coef.tolist(), "mu": mu.tolist(), "sd": sd.tolist(),
            "intercept": ym, "class_heights": heights, "n_features": int(X.shape[1])}


def apply_calibrator(cal: dict, est: dict, form: str, material: str) -> float:
    x = _features(est, form, material, cal["class_heights"])
    z = (x - np.asarray(cal["mu"])) / np.asarray(cal["sd"])
    return float(np.clip(math.exp(cal["intercept"] + float(z @ np.asarray(cal["coef"]))), 1.0, 20000.0))


# ---------------------------------------------------------------- data

def _row_job(args) -> dict:
    import cv2
    from geometry.features import compute_geometry_features
    from geometry.volume_parts import estimate_volume, refine_plane
    root, file_name, ann, camera = args
    fdir = Path(root) / Path(file_name).parent
    depth = cv2.imread(str(fdir / "depth.png"), cv2.IMREAD_UNCHANGED)
    meta = json.loads((fdir / "meta.json").read_text())
    mask = np.zeros(depth.shape, np.uint8)
    for poly in ann["segmentation"]:
        cv2.fillPoly(mask, [np.asarray(poly, np.float32).reshape(-1, 2).round().astype(np.int32)], 1)
    mask = mask.astype(bool)
    plane = np.asarray(meta["floor_plane"], dtype=np.float64)
    at = ann["attributes"]
    est = estimate_volume(depth, mask, refine_plane(depth, mask, plane, camera), camera, at["form"])
    _, old = compute_geometry_features(depth.astype(np.float32), mask, plane, camera)
    return {"image_id": Path(file_name).parent.name, "object_id": at["object_id"], "form": at["form"],
            "material": at["material"], "volume_ml": float(at["volume_ml"]), "mass_g": float(at["mass_g"]),
            "old_geom_ml": float(old["geometric_volume_ml"] or 0.0), **est}


def collect_rows(data_root: Path, split: str = "real", workers: int = 8) -> pd.DataFrame:
    data_root = Path(data_root)
    camera = json.loads((data_root / "camera.json").read_text())
    lab = json.loads((data_root / f"labels/{split}_train.json").read_text())
    imgs = {i["id"]: i["file_name"] for i in lab["images"]}
    jobs = [(str(data_root), imgs[a["image_id"]], a, camera) for a in lab["annotations"]]
    with Pool(workers) as pool:
        rows = pool.map(_row_job, jobs, chunksize=4)
    df = pd.DataFrame(rows)
    df["split"] = split
    return df


# ---------------------------------------------------------------- report

def _breakdown(df: pd.DataFrame, pred_col: str) -> dict:
    out = {"V": volume_score(df[pred_col], df["volume_ml"])}
    out["by_class"] = {f"{f}|{m}": volume_score(g[pred_col], g["volume_ml"])
                       for (f, m), g in df.groupby(["form", "material"])}
    out["by_bin"] = {f"{lo:g}-{hi:g}": volume_score(g[pred_col], g["volume_ml"])
                     for lo, hi in BINS
                     for g in [df[(df["volume_ml"] >= lo) & (df["volume_ml"] < hi)]] if len(g)}
    return out


def oof_predictions(real: pd.DataFrame, synth: pd.DataFrame | None, n_folds: int, seed: int) -> np.ndarray:
    pred = np.zeros(len(real))
    for tr, va in group_folds(real["object_id"].to_numpy(), n_folds, seed):
        train = real.iloc[tr]
        if synth is not None:
            allowed = set(train["object_id"])
            train = pd.concat([train, synth[synth["object_id"].isin(allowed)]], ignore_index=True)
        cal = fit_calibrator(train)
        val = real.iloc[va]
        pred[va] = [apply_calibrator(cal, r, r["form"], r["material"]) for r in val.to_dict("records")]
    return pred


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="datasets/raw")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--with-synth", action="store_true")
    ap.add_argument("--rows", default="reports/volume_parts_gt_rows.csv")
    ap.add_argument("--out", default="reports/volume_parts_cv.json")
    ap.add_argument("--calibrator", default="checkpoints/volume_parts/calibrator.json")
    args = ap.parse_args()

    rows_path = Path(args.rows)
    if rows_path.exists():
        real = pd.read_csv(rows_path)
    else:
        real = collect_rows(Path(args.data), "real")
        rows_path.parent.mkdir(parents=True, exist_ok=True)
        real.to_csv(rows_path, index=False)
    synth = None
    if args.with_synth:
        sp = rows_path.with_name(rows_path.stem + "_synth.csv")
        synth = pd.read_csv(sp) if sp.exists() else collect_rows(Path(args.data), "synth")
        synth.to_csv(sp, index=False)

    report = {"n_rows": len(real), "n_objects": int(real["object_id"].nunique()), "raw": {}, "calibrated": []}
    for col in ["old_geom_ml", *EST_KEYS]:
        report["raw"][col] = _breakdown(real, col)
    for seed in range(args.seeds):
        real[f"oof_{seed}"] = oof_predictions(real, synth, args.folds, seed)
        report["calibrated"].append(_breakdown(real, f"oof_{seed}"))
    vs = [c["V"] for c in report["calibrated"]]
    report["calibrated_V_mean"], report["calibrated_V_std"] = float(np.mean(vs)), float(np.std(vs))

    print(f"rows={len(real)} objects={report['n_objects']} with_synth={args.with_synth}")
    for col, r in report["raw"].items():
        print(f"raw {col:13s} V={r['V']:.4f}")
    print(f"calibrated OOF V={report['calibrated_V_mean']:.4f} ± {report['calibrated_V_std']:.4f}")
    print("by bin (seed 0):", {k: round(v, 3) for k, v in report["calibrated"][0]["by_bin"].items()})
    print("by class (seed 0):", {k: round(v, 3) for k, v in report["calibrated"][0]["by_class"].items()})

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(report, indent=2, ensure_ascii=False))
    final_train = real if synth is None else pd.concat([real, synth], ignore_index=True)
    cal = fit_calibrator(final_train)
    Path(args.calibrator).parent.mkdir(parents=True, exist_ok=True)
    Path(args.calibrator).write_text(json.dumps(cal, indent=2))
    real.to_csv(rows_path.with_name(rows_path.stem + "_oof.csv"), index=False)


if __name__ == "__main__":
    main()
