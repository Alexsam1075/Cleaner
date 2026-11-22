"""Работа над V.
Часть 1 — разложение потерь V на фолдах 1–3: что съедает V между идеалом и конвейером
  (размер train-набора фолда, маски YOLO, ошибки материала).
Часть 2 — улучшения калибратора, честный LOOO по 40 предметам (GT-маски, истинный класс) и на фолдах:
  A  текущий fit_calibrator (ridge, 38 признаков, alpha по внутреннему CV)
  B  иерархическое сжатие: остаток log(gt/parts) = глобальная регрессия по геометрии
     + поправка класса, сжатая к форме и к глобальной по числу ОБЪЕКТОВ в классе (эмпирический Байес)
  C  B + IRLS (веса под метрику V)
  D  среднее A и C в лог-пространстве
Папка real/test не используется."""
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path("/home/cleaner/Cleaner")
sys.path.insert(0, str(ROOT))
from evaluation.volume_parts_cv import apply_calibrator, fit_calibrator  # noqa: E402

C = ROOT / "audit/cache"
L3 = math.log(3)
FOLDS = {1: ROOT / "configs/object_id_split.json",
         2: ROOT / "checkpoints/score83_cv_fold2_noval18/configs/object_id_split.json",
         3: ROOT / "checkpoints/score83_cv_fold3_noval18/configs/object_id_split.json"}
EST = ["columns_ml", "parts_ml", "hull_ml", "footprint_ml"]


def V(p, g):
    return np.clip(1 - np.abs(np.log(np.clip(p, 1e-3, None) / g)) / L3, 0, 1)


def geo_X(r):
    lv = np.log(np.maximum(r[EST].values.astype(float), 1.0))
    main = lv[:, 1:2]
    return np.c_[lv - main, main, np.maximum(0, main - math.log(50)), np.maximum(0, main - math.log(300)),
                 r["valid_frac"].values, r["truncated"].astype(float).values, np.minimum(r["n_parts"].values, 4),
                 r["p95_h_m"].values * 20]


def wridge(X, y, w, alpha):
    mu = np.average(X, 0, w)
    sd = np.sqrt(np.average((X - mu) ** 2, 0, w)) + 1e-6
    Z = (X - mu) / sd
    ym = np.average(y, weights=w)
    coef = np.linalg.solve(Z.T @ (Z * w[:, None]) + alpha * np.eye(Z.shape[1]), Z.T @ (w * (y - ym)))
    return lambda Xn: ym + ((Xn - mu) / sd) @ coef


class Hier:
    """log(gt) = log(parts) + f(geometry) + δ_form + δ_class; δ сжаты по числу объектов (k = prior strength)."""

    def __init__(self, alpha=10.0, k=2.0, irls=False):
        self.alpha, self.k, self.irls = alpha, k, irls

    def fit(self, r):
        y = np.log(r["volume_ml"].values) - np.log(np.maximum(r["parts_ml"].values, 1.0))
        X = geo_X(r)
        cnt = r.groupby("object_id")["object_id"].transform("size").values
        w = 1.0 / cnt
        f = wridge(X, y, w, self.alpha)
        for _ in range(6 if self.irls else 0):
            res = np.abs(f(X) - y)
            f = wridge(X, y, w / np.maximum(res, 0.15), self.alpha)
        self.f = f
        res = y - f(X)
        obj = pd.DataFrame({"oid": r["object_id"].values, "form": r["form"].values, "mat": r["material"].values, "res": res}) \
            .groupby("oid").agg(form=("form", "first"), mat=("mat", "first"), res=("res", "mean"))
        self.d_form = {}
        for fm, g in obj.groupby("form"):
            self.d_form[fm] = g.res.sum() / (len(g) + self.k)
        self.d_cls = {}
        for (fm, m), g in obj.groupby(["form", "mat"]):
            self.d_cls[(fm, m)] = (g.res - self.d_form[fm]).sum() / (len(g) + self.k)
        return self

    def predict(self, r, materials=None):
        mats = r["material"].values if materials is None else materials
        base = np.log(np.maximum(r["parts_ml"].values, 1.0)) + self.f(geo_X(r))
        off = np.array([self.d_form.get(fm, 0.0) + self.d_cls.get((fm, m), 0.0) for fm, m in zip(r["form"].values, mats)])
        return np.exp(base + off)


def tune_hier(r, irls):
    """alpha, k по внутреннему 5-fold по объектам."""
    objs = np.array(sorted(r.object_id.unique()))
    rs = np.random.default_rng(3)
    rs.shuffle(objs)
    groups = np.array_split(objs, 5)
    best = None
    for alpha in (1, 3, 10, 30, 100):
        for k in (0.5, 1, 2, 4, 8):
            p = np.zeros(len(r))
            for g in groups:
                te = r.object_id.isin(g).values
                p[te] = Hier(alpha, k, irls).fit(r[~te]).predict(r[te])
            s = V(p, r.volume_ml.values).mean()
            if best is None or s > best[0]:
                best = (s, alpha, k)
    return Hier(best[1], best[2], irls).fit(r), best


def looo(rows, method):
    pred = np.zeros(len(rows))
    for o in sorted(rows.object_id.unique()):
        tr, te = rows[rows.object_id != o], rows[rows.object_id == o]
        if method == "A":
            cal = fit_calibrator(tr)
            pred[te.index] = [apply_calibrator(cal, x, x["form"], x["material"]) for x in te.to_dict("records")]
        else:
            m, _ = tune_hier(tr, irls=(method == "C"))
            pred[te.index] = m.predict(te)
    return pred


def bins_str(v, g):
    return "  ".join(f"{lo:g}-{hi:g}:{v[(g >= lo) & (g < hi)].mean():.2f}" for lo, hi in ((0, 50), (50, 200), (200, 1000), (1000, 1e9)))


def main():
    rows = pd.read_csv(ROOT / "reports/volume_parts_gt_rows.csv")
    rows = rows[rows.split == "real"].reset_index(drop=True)
    g = rows.volume_ml.values

    print("== Часть 2: калибраторы, LOOO по 40 предметам (GT-маски, истинный класс) ==")
    P = {}
    for m in ("A", "B", "C"):
        P[m] = looo(rows, m)
        v = V(P[m], g)
        print(f"  {m}: V={v.mean():.3f}  bulk {v[rows.form == 'bulk'].mean():.2f} solid {v[rows.form == 'solid'].mean():.2f} | {bins_str(v, g)}", flush=True)
    P["D"] = np.sqrt(P["A"] * P["C"])
    v = V(P["D"], g)
    print(f"  D (A⊕C): V={v.mean():.3f}  bulk {v[rows.form == 'bulk'].mean():.2f} solid {v[rows.form == 'solid'].mean():.2f} | {bins_str(v, g)}")

    print("\n== Часть 1: разложение потерь V на фолдах (val-объекты фолда) ==")
    for fold, sp in FOLDS.items():
        split = json.load(open(sp))
        tro, vao = set(split["train_object_ids"]), set(split["validation_object_ids"])
        tr = rows[rows.object_id.isin(tro)]
        va = rows[rows.object_id.isin(vao)].reset_index(drop=True)
        cal = fit_calibrator(tr)
        pA = np.array([apply_calibrator(cal, x, x["form"], x["material"]) for x in va.to_dict("records")])
        mC, best = tune_hier(tr, irls=True)
        pC = mC.predict(va)
        idx40 = rows.object_id.isin(vao).values
        print(f"  fold {fold}: GT-маски+истинный класс, калибратор на 39 объектах (LOOO) V={V(P['A'][idx40], rows.volume_ml.values[idx40]).mean():.3f}"
              f" | на train фолда (30 объектов): A {V(pA, va.volume_ml.values).mean():.3f}  C {V(pC, va.volume_ml.values).mean():.3f}"
              f"  (C: alpha={best[1]}, k={best[2]})")


if __name__ == "__main__":
    main()
