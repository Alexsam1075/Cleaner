"""Exp2: direct smooth-metric optimisation, group-wise ridge penalties, physically motivated shape features,
class prior shrink by geometry confidence. All nested LOOO."""
import sys
import warnings

warnings.filterwarnings("ignore")
sys.path.insert(0, "/home/cleaner/Cleaner/agents2/volume")
import numpy as np
from scipy.optimize import minimize

from common import *
from vmodels import *

df = load_rows()
y = np.log(df.volume_ml.values)
ids = df.object_id.values
base = ["columns_ml", "parts_ml", "hull_ml", "footprint_ml"]
ref = base + ["gi_env_ml"]
X_ref = design(df, ref, inter="gi_env_ml", knots=(50, 300))


# ---- direct optimisation of a smooth surrogate of V (log-cosh-like truncated loss), warm start from IRLS
def fit_direct(X, yt, w, g, tau=0.05):
    a, fl = g
    m0 = irls(X, yt, w, a, floor=fl)
    mu = np.average(X, 0, weights=w)
    sd = np.sqrt(np.average((X - mu) ** 2, 0, weights=w))
    sd[sd < 1e-9] = 1.0
    Z = np.c_[np.ones(len(X)), (X - mu) / sd]
    c0 = np.linalg.lstsq(Z, m0(X), rcond=None)[0]
    wn = w / w.sum()
    lam = a * 1e-3

    def f_and_g(theta):
        r = Z @ theta - yt
        ab = np.sqrt(r * r + tau * tau)
        inside = ab < L3
        pen = np.where(inside, ab, L3 + 0.05 * (ab - L3))
        dpen = np.where(inside, 1.0, 0.05) * r / ab
        reg = theta.copy(); reg[0] = 0
        return np.sum(wn * pen) / L3 + lam * np.sum(reg ** 2), Z.T @ (wn * dpen) / L3 + 2 * lam * reg

    th = minimize(f_and_g, c0, jac=True, method="L-BFGS-B", options={"maxiter": 200}).x
    return lambda Xn: th[0] + ((np.asarray(Xn) - mu) / sd) @ th[1:]


p_i = nested_looo(X_ref, y, ids, fit_irls, IRLS_GRID)
report("ref IRLS", p_i, df)


# ---- group-wise penalties: geometry block vs class block vs interaction block
def ridge_groups(X, yt, w, lam):
    mu = np.average(X, 0, weights=w)
    sd = np.sqrt(np.average((X - mu) ** 2, 0, weights=w))
    sd[sd < 1e-9] = 1.0
    Z = (X - mu) / sd
    ym = np.average(yt, weights=w)
    coef = np.linalg.solve(Z.T @ (Z * w[:, None]) + np.diag(lam), Z.T @ (w * (yt - ym)))
    return lambda Xn: ym + ((np.asarray(Xn) - mu) / sd) @ coef


n_geo = len(ref)
n_cls = 1 + 6 + 6
n_int = 6 + 1
n_kn = 2
assert X_ref.shape[1] == n_geo + n_cls + n_int + n_kn


def fit_grp(X, yt, w, g):
    ag, ac, ai, fl = g
    lam = np.concatenate([np.full(n_geo, ag), np.full(n_cls, ac), np.full(n_int, ai), np.full(n_kn, ag)])
    m = ridge_groups(X, yt, w, lam)
    for _ in range(8):
        r = np.abs(m(X) - yt)
        m = ridge_groups(X, yt, w * np.where(r < L3, 1 / np.maximum(r, fl), 0.1 / L3), lam)
    return m


grid = [(ag, ac, ai, 0.2) for ag in (3, 10, 30) for ac in (3, 10, 30, 100) for ai in (10, 30, 100, 300)]
p_g = nested_looo(X_ref, y, ids, fit_grp, grid)
report("IRLS group penalties", p_g, df)

# ---- physically motivated shape descriptors on top of ref
df["log_h"] = np.log(np.maximum(df.p95_h_m, 0.005))
df["log_fp"] = np.log(np.maximum(df.footprint_m2, 1e-4))
df["aspect"] = np.log(np.maximum(df.inp_L_m, 1e-3) / np.maximum(df.inp_W_m, 1e-3))
df["h_over_w"] = np.log(np.maximum(df.p95_h_m, 0.005) / np.maximum(df.inp_W_m, 1e-3))
df["hidden"] = np.log(np.maximum(df.inp_sym_cells_m2, 1e-5) / np.maximum(df.inp_cells_m2, 1e-5))
df["env_over_col"] = np.log(np.maximum(df.gi_env_ml, 1) / np.maximum(df.i0_colmax_ml, 1))
df["hull_over_env"] = np.log(np.maximum(df.hull_ml, 1) / np.maximum(df.gi_env_ml, 1))
df["bimod"] = (df.inp_h_std_m / np.maximum(df.inp_h_mean_m, 1e-3))
for ex in (["log_h"], ["h_over_w"], ["hidden"], ["env_over_col"], ["hull_over_env"], ["bimod"], ["aspect"],
           ["log_h", "cos_tilt"], ["h_over_w", "cos_tilt"], ["cos_tilt", "hidden"]):
    X = design(df, ref, inter="gi_env_ml", knots=(50, 300), extra=df[ex].to_numpy(float))
    report("IRLS + " + ",".join(ex), nested_looo(X, y, ids, fit_irls, IRLS_GRID), df)

# ---- shrink toward class prior when geometry is unreliable (valid_frac low / truncated / tiny)
conf = df.valid_frac.values * (1 - 0.5 * df.truncated.values.astype(float))
for k in (0.5, 1.0, 2.0):
    pred = np.zeros(len(y))
    for o in sorted(set(ids)):
        tr = ids != o
        m = irls(X_ref[tr], y[tr], w_of(ids[tr]), 10.0, floor=0.2)
        # class prior from training rows of the same form|material (fallback form)
        d = df[tr]
        pri = []
        for r in df[~tr].to_dict("records"):
            s = d[(d.form == r["form"]) & (d.material == r["material"])]
            if len(s) < 3:
                s = d[d.form == r["form"]]
            pri.append(np.median(np.log(s.volume_ml)))
        pm = m(X_ref[~tr])
        lam = np.clip(k * (1 - conf[~tr]), 0, 0.9)
        pred[~tr] = (1 - lam) * pm + lam * np.array(pri)
    report(f"IRLS a10 + prior shrink k={k}", pred, df)
report("IRLS a10 fixed (no nested)", nested_looo(X_ref, y, ids, fit_irls, [(10, 0.2)]), df)
