"""Гибрид объёма: алгоритмические измерения (якорь пол+камера, силуэт, карта высот, части) + калибровка.
Цель — понять, почему гибрид не выше ~0.76, и попробовать сильнее.
LOOO по 40 предметам, GT-маски, истинные форма/материал; все гиперпараметры — внутренний grouped CV.
Только real_train."""
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path("/home/cleaner/Cleaner")
sys.path.insert(0, str(ROOT))
from evaluation.volume_parts_cv import EST_KEYS, KNOTS_ML, _class_heights, _design, _weights  # noqa: E402

C = Path(__file__).resolve().parent / "cache"
L3 = math.log(3)


def V(p, g):
    return np.clip(1 - np.abs(np.log(np.clip(p, 1e-3, None) / g)) / L3, 0, 1)


def load():
    a = pd.read_csv(ROOT / "reports/volume_parts_gt_rows.csv")
    a = a[a.split == "real"].reset_index(drop=True)
    b = pd.read_csv(C / "algo_volume_rows.csv")
    assert len(a) == len(b) and (a.image_id.values == b.frame.values).all() and (a.object_id.values == b.oid.values).all()
    extra = pd.DataFrame({
        "x_logcol": np.log1p(b.colvol.fillna(0)), "x_logfoot": np.log1p(b.foot.fillna(0) * 1e4),
        "x_H": b.H.fillna(b.sil_H) * 20, "x_silH": b.sil_H * 20, "x_silW": np.log1p(b.sil_W * 100),
        "x_gap": b.gap.fillna(0), "x_fill": b.fill.fillna(0.7).clip(0, 2), "x_valid": b.valid,
        "x_logL": np.log1p(b.L.fillna(0) * 100), "x_logW": np.log1p(b.W.fillna(0) * 100),
        "x_thin": (b.H.fillna(b.sil_H) < 0.03).astype(float),
    })
    return a, extra


def ridge(X, y, w, alpha):
    mu, sd = np.average(X, 0, w), np.sqrt(np.average((X - np.average(X, 0, w)) ** 2, 0, w)) + 1e-6
    Z = (X - mu) / sd
    ym = np.average(y, weights=w)
    A = Z.T @ (Z * w[:, None]) + alpha * np.eye(Z.shape[1])
    coef = np.linalg.solve(A, Z.T @ (w * (y - ym)))
    return lambda Xn: ym + ((Xn - mu) / sd) @ coef


def irls_fit(X, y, w, alpha, iters=8, floor=0.15):
    """Подгонка под метрику V: веса ~ 1/max(|r|, floor) (L1-подобно, как у агента)."""
    ww = w.copy()
    f = ridge(X, y, ww, alpha)
    for _ in range(iters):
        r = np.abs(f(X) - y)
        ww = w / np.maximum(r, floor)
        f = ridge(X, y, ww, alpha)
    return f


def group_folds(ids, k, seed):
    u = np.array(sorted(set(ids)))
    rs = np.random.default_rng(seed)
    rs.shuffle(u)
    for g in np.array_split(u, k):
        yield ~np.isin(ids, g), np.isin(ids, g)


def nested_looo(X, y, ids, w0, fitter, alphas=(0.3, 1, 3, 10, 30, 100)):
    pred = np.zeros(len(y))
    chosen = []
    for o in sorted(set(ids)):
        tr, te = ids != o, ids == o
        best = None
        for a in alphas:
            p = np.zeros(tr.sum())
            for itr, iva in group_folds(ids[tr], 5, 7):
                f = fitter(X[tr][itr], y[tr][itr], w0[tr][itr], a)
                p[iva] = f(X[tr][iva])
            s = V(np.exp(p), np.exp(y[tr])).mean()
            if best is None or s > best[0]:
                best = (s, a)
        chosen.append(best[1])
        pred[te] = fitter(X[tr], y[tr], w0[tr], best[1])(X[te])
    return np.exp(pred), chosen


def report(name, p, a, extra_txt=""):
    g = a.volume_ml.values
    v = V(p, g)
    bins = [(0, 50), (50, 200), (200, 1000), (1000, 1e9)]
    b = "  ".join(f"{lo:g}-{hi:g}:{v[(g >= lo) & (g < hi)].mean():.2f}" for lo, hi in bins)
    print(f"  {name:52s} V={v.mean():.3f}  bulk {v[a.form == 'bulk'].mean():.2f} solid {v[a.form == 'solid'].mean():.2f} | {b} {extra_txt}", flush=True)
    return v.mean()


def main():
    a, extra = load()
    ids = a.object_id.values
    y = np.log(a.volume_ml.values)
    heights = _class_heights(a)
    Xb = _design(a, heights, list(EST_KEYS), list(KNOTS_ML))
    Xe = np.c_[Xb, extra.values]
    w0 = _weights(a.object_id).astype(float)

    print("== Потолки (сколько вообще можно выжать) ==")
    base_p, _ = nested_looo(Xb, y, ids, w0, ridge)
    lr = np.log(base_p) - y
    bias = pd.Series(lr).groupby(ids).transform("mean").values
    print(f"  текущий гибрид (ridge, признаки volume/):              V={V(base_p, np.exp(y)).mean():.3f}")
    print(f"  если убрать постоянную ошибку каждого предмета:       V={V(np.exp(y + (lr - bias)), np.exp(y)).mean():.3f}")
    print(f"  если убрать только разброс между кадрами предмета:    V={V(np.exp(y + bias), np.exp(y)).mean():.3f}")
    ob = pd.DataFrame({"oid": ids, "form": a.form, "mat": a.material, "gt": a.volume_ml, "ratio": np.exp(lr)}).groupby("oid").agg(
        form=("form", "first"), mat=("mat", "first"), gt=("gt", "first"), ratio=("ratio", "median"))
    print("  самые большие постоянные ошибки (pred/gt):")
    print("   ", ", ".join(f"{o}:{r:.2f}" for o, r in ob.ratio.sort_values().items() if r < 0.7 or r > 1.43))

    print("\n== Гибриды (nested LOOO) ==")
    report("A ridge, признаки volume/", base_p, a)
    pB, ch = nested_looo(Xe, y, ids, w0, ridge)
    report("B ridge, + якорь/силуэт/пустоты/габарит", pB, a, f"alpha≈{np.median(ch)}")
    pC, ch = nested_looo(Xb, y, ids, w0, irls_fit)
    report("C IRLS под метрику V, признаки volume/", pC, a, f"alpha≈{np.median(ch)}")
    pE, ch = nested_looo(Xe, y, ids, w0, irls_fit)
    report("D IRLS под метрику V, + якорь/силуэт", pE, a, f"alpha≈{np.median(ch)}")
    # E: тонкие предметы — объём = площадь следа × толщина класса (толщина — медиана по train-объектам класса)
    pT = pE.copy()
    thin = extra.x_thin.values > 0.5
    b = pd.read_csv(C / "algo_volume_rows.csv")
    for o in sorted(set(ids)):
        te = (ids == o) & thin
        if not te.any():
            continue
        tr = (ids != o) & thin
        th = (a.volume_ml[tr] / (b.foot[tr].clip(lower=1e-5) * 1e6)).groupby(ids[tr]).median()
        same = [q for q in th.index if a.material[ids == q].iloc[0] == a.material[te].iloc[0]]
        t = float(np.median(th[same] if len(same) >= 2 else th))
        pT[te] = np.sqrt(pE[te] * b.foot[te].clip(lower=1e-5).values * 1e6 * t)
    report("E D + тонкие: площадь силуэта × толщина класса", pT, a, f"(тонких кадров {thin.sum()})")
    # F: ансамбль B и D
    report("F среднее (B, D) в лог-пространстве", np.sqrt(pB * pE), a)
    np.save(C / "hybrid_best_oof.npy", pE)


if __name__ == "__main__":
    main()
