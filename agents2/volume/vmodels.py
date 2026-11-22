"""Fitters and nested-LOOO harness. Everything is in log space; weights 1/frames-per-object."""
import math

import numpy as np

from common import L3, MATS, inner_folds, vscore, w_of


def ridge(X, y, w, a):
    mu = np.average(X, 0, weights=w)
    sd = np.sqrt(np.average((X - mu) ** 2, 0, weights=w))
    sd[sd < 1e-9] = 1.0
    Z = (X - mu) / sd
    ym = np.average(y, weights=w)
    coef = np.linalg.solve(Z.T @ (Z * w[:, None]) + a * np.eye(Z.shape[1]), Z.T @ (w * (y - ym)))
    return lambda Xn: ym + ((np.asarray(Xn) - mu) / sd) @ coef


def irls(X, y, w, a, floor=0.15, iters=8):
    """Truncated-L1 surrogate of the metric: weight 1/|r| inside ln3, tiny weight outside."""
    m = ridge(X, y, w, a)
    for _ in range(iters):
        r = np.abs(m(X) - y)
        m = ridge(X, y, w * np.where(r < L3, 1.0 / np.maximum(r, floor), 0.1 / L3), a)
    return m


def design(df, cols, inter=None, knots=(), cls=True, extra=None):
    X = np.stack([np.log(np.maximum(df[c].to_numpy(float), 1.0)) if c.endswith("_ml") else df[c].to_numpy(float)
                  for c in cols], 1)
    parts = [X]
    if cls:
        form = (df.form == "bulk").to_numpy().astype(float)[:, None]
        mat = np.stack([(df.material == m).to_numpy() for m in MATS], 1).astype(float)
        parts += [form, mat, form * mat]
        if inter is not None:
            main = X[:, cols.index(inter)][:, None]
            parts += [main * mat, main * form]
    for t in knots:
        main = X[:, cols.index(inter)]
        parts.append(np.maximum(0, main - math.log(t))[:, None])
    if extra is not None:
        parts.append(np.asarray(extra, float).reshape(len(df), -1))
    return np.hstack(parts)


def nested_looo(X, y, ids, fitter, grid, Xeval=None, inner=5, seed=0, return_picks=False):
    """Outer LOOO over objects; inner grouped CV picks the grid item maximizing weighted V."""
    X = np.asarray(X)
    Xeval = X if Xeval is None else np.asarray(Xeval)
    pred = np.zeros(len(y))
    w = w_of(ids)
    picks = []
    for o in sorted(set(ids)):
        tr = ids != o
        Xt, yt, it, wt = X[tr], y[tr], ids[tr], w[tr]
        if len(grid) > 1:
            f = inner_folds(it, inner, seed)
            best = None
            for g in grid:
                p = np.zeros(len(yt))
                for k in range(inner):
                    p[f == k] = fitter(Xt[f != k], yt[f != k], wt[f != k], g)(Xt[f == k])
                s = np.sum(wt * vscore(p, yt)) / wt.sum()
                if best is None or s > best[0]:
                    best = (s, g)
            g = best[1]
        else:
            g = grid[0]
        picks.append(g)
        pred[~tr] = fitter(Xt, yt, wt, g)(Xeval[~tr])
    return (pred, picks) if return_picks else pred


def fit_ridge(X, y, w, g):
    return ridge(X, y, w, g)


def fit_irls(X, y, w, g):
    a, fl = g if isinstance(g, tuple) else (g, 0.15)
    return irls(X, y, w, a, floor=fl)


RIDGE_GRID = [0.3, 1, 3, 10, 30, 100]
IRLS_GRID = [(a, fl) for a in (1, 3, 10, 30, 100) for fl in (0.1, 0.2, 0.3)]
