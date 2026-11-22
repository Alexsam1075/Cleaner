"""Лучший калибратор V: признаки агента (feat_real.csv: inp_* с заполнением дыр, симметрия, box/ellipse/hull, силуэт,
доли дыр) + feat2 (gi_env и пр.) + класс; модели: ridge, IRLS (под метрику V), Huber-подобная; ансамбль.
Честный LOOO по 40 предметам, все гиперпараметры во внутреннем 5-fold по объектам. GT-маски, истинный класс,
плюс проверка с 8% неверного материала. Папка real/test не используется."""
import math
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, "/home/cleaner/Cleaner/agents/volume")
from exp1 import MATS, design, ids, real, ridge, w_of, y  # noqa: E402

L3 = math.log(3)
C = "/home/cleaner/Cleaner/agents/volume/cache/"


def Vsc(p, t):
    return np.maximum(0, 1 - np.abs(p - t) / L3)


def fit_irls(X, yt, w, a, floor=0.15, iters=8):
    m = ridge(X, yt, w, a)
    for _ in range(iters):
        r = np.abs(m(X) - yt)
        wi = np.where(r < L3, 1 / np.maximum(r, floor), 0.1 / L3)
        m = ridge(X, yt, w * wi, a)
    return m


def fit_plain(X, yt, w, a, **k):
    return ridge(X, yt, w, a)


def nested(X, fitter, alphas=(1, 3, 10, 30, 100), floors=(None,), mats_eval=None):
    pred = np.zeros(len(y))
    w0 = w_of(ids)
    picks = []
    for o in sorted(set(ids)):
        tr = ids != o
        Xt, yt, it, wt = X[tr], y[tr], ids[tr], w0[tr]
        ut = np.array(sorted(set(it)))
        np.random.default_rng(0).shuffle(ut)
        fo = {u: k % 5 for k, u in enumerate(ut)}
        f = np.array([fo[u] for u in it])
        best = None
        for a in alphas:
            for fl in floors:
                kw = {} if fl is None else {"floor": fl}
                p = np.zeros(len(yt))
                for k in range(5):
                    p[f == k] = fitter(Xt[f != k], yt[f != k], wt[f != k], a, **kw)(Xt[f == k])
                s = np.sum(wt * Vsc(p, yt)) / wt.sum()
                if best is None or s > best[0]:
                    best = (s, a, kw)
        picks.append((best[1], best[2].get("floor")))
        m = fitter(Xt, yt, wt, best[1], **best[2])
        pred[~tr] = m(X[~tr] if mats_eval is None else mats_eval[~tr])
    return pred, picks


def rep(name, p):
    v = Vsc(p, y)
    g = real.volume_ml.values
    b = "  ".join(f"{lo:g}-{hi:g}:{v[(g >= lo) & (g < hi)].mean():.2f}" for lo, hi in ((0, 50), (50, 200), (200, 1000), (1000, 1e9)))
    print(f"  {name:58s} V={v.mean():.4f}  bulk {v[real.form == 'bulk'].mean():.2f} solid {v[real.form == 'solid'].mean():.2f} | {b}", flush=True)
    return v.mean()


def main():
    f2 = pd.read_csv(C + "feat2_real.csv")
    f2 = f2.rename(columns={c: c + "_ml" for c in f2.columns if c.endswith(("colmax", "colmean"))})
    df = pd.concat([real, f2.drop(columns=["split", "frame", "object_id"])], axis=1)
    base = ["columns_ml", "parts_ml", "hull_ml", "footprint_ml"]
    rich = base + ["inp_colmean_ml", "inp_colmax_ml", "inp_sym_colmax_ml", "inp_box_ml", "inp_ell_ml", "inp_sym_hull_ml"]
    ref = base + ["gi_env_ml"]
    sets = {
        "base4 (как volume/)": design(df, base, inter="parts_ml", knots=(50, 300)),
        "base4 + gi_env (лучший у агента)": design(df, ref, inter="gi_env_ml", knots=(50, 300)),
        "rich10 (заполнение дыр, симметрия, примитивы)": design(df, rich, inter="inp_colmean_ml", knots=(50, 300)),
    }
    P = {}
    print("== LOOO по 40 предметам, GT-маски, истинный класс ==")
    for sname, X in sets.items():
        P[(sname, "ridge")], _ = nested(X, fit_plain)
        rep(f"{sname} | ridge", P[(sname, "ridge")])
        P[(sname, "irls")], picks = nested(X, fit_irls, floors=(0.1, 0.2, 0.3))
        rep(f"{sname} | IRLS (alpha, floor во вложенном CV)", P[(sname, "irls")])
    keys = list(P)
    best3 = sorted(keys, key=lambda k: -Vsc(P[k], y).mean())[:3]
    ens = np.mean([P[k] for k in best3], 0)
    rep("ансамбль 3 лучших (среднее в лог-пространстве)*", ens)
    print("   * выбор 3 лучших сделан по внешнему LOOO — оптимистично; честный вариант ниже")
    ens_all = np.mean([P[k] for k in keys], 0)
    rep("ансамбль всех 6 (без выбора, честно)", ens_all)
    ens_irls = np.mean([P[(s, "irls")] for s in sets], 0)
    rep("ансамбль 3 IRLS (без выбора, честно)", ens_irls)
    np.save(C + "v_best_oof.npy", ens_irls)

    print("\n== Устойчивость к ошибкам материала (8% кадров с неверным материалом на входе) ==")
    rng = np.random.default_rng(5)
    noisy = df.copy()
    flip = rng.random(len(df)) < 0.08
    noisy.loc[flip, "material"] = [rng.choice([m for m in MATS if m != x]) for x in df.material[flip]]
    for sname, cols, inter in (("base4 + gi_env", ref, "gi_env_ml"), ("rich10", rich, "inp_colmean_ml")):
        Xc = design(df, cols, inter=inter, knots=(50, 300))
        Xn = design(noisy, cols, inter=inter, knots=(50, 300))
        p, _ = nested(Xc, fit_irls, floors=(0.2,), mats_eval=Xn)
        rep(f"{sname} | IRLS, материал с шумом 8%", p)


if __name__ == "__main__":
    main()
