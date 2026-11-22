"""Варианты правила take для pipeline_v2 (--external take_xxx=<этот файл>:fn). Все пороги/плотности — только по train-объектам фолда.
 t1: плотность по материалу без формы (устойчиво к ошибке формы)
 t2: материалы без единого тяжёлого (>500 г) train-объекта -> take=1; остальные как hybrid
 t3: t1 + t2
 t4: t2 + для тяжёлых классов объём ≤ T, для остальных масса по плотности материала (= t3), плюс T только если в train есть тяжёлые этого материала"""
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
import eval_folds as E
from eval_folds import HEAVY


def _heavy_mats(ctx):
    seen, hm = set(), set()
    for m in ctx["meta"]:
        if m["src"] == "real_train" and m["oid"] in set(ctx["train_objs"]) and m["oid"] not in seen:
            seen.add(m["oid"])
            if m["mass"] > 500:
                hm.add(m["mat"])
    return hm


def _mass(ctx, vols, forms, mats, by_form):
    dens = ctx["dens"]
    if by_form:
        return np.array([v * dens.get((f, m), dens.get(("*", m), 0.5)) for v, f, m in zip(vols, forms, mats)])
    return np.array([v * dens.get(("*", m), 0.5) for v, m in zip(vols, mats)])


def take_t1(ctx, vols, forms, mats):
    T = ctx["take_thr"]
    mass = _mass(ctx, vols, forms, mats, False)
    take = np.array([int(v <= T) if m in HEAVY else int(mc <= 500) for v, m, mc in zip(vols, mats, mass)])
    return take, mass


def take_t2(ctx, vols, forms, mats):
    T = ctx["take_thr"]
    hm = _heavy_mats(ctx)
    ctx.setdefault("notes", []).append(f"heavy mats in train: {sorted(hm)}")
    mass = _mass(ctx, vols, forms, mats, True)
    take = np.array([1 if m not in hm else (int(v <= T) if m in HEAVY else int(mc <= 500)) for v, m, mc in zip(vols, mats, mass)])
    return take, mass


def take_t3(ctx, vols, forms, mats):
    T = ctx["take_thr"]
    hm = _heavy_mats(ctx)
    ctx.setdefault("notes", []).append(f"heavy mats in train: {sorted(hm)}")
    mass = _mass(ctx, vols, forms, mats, False)
    take = np.array([1 if m not in hm else (int(v <= T) if m in HEAVY else int(mc <= 500)) for v, m, mc in zip(vols, mats, mass)])
    return take, mass


# ---------------------------------------------------------------- лог-лог масса: log m = a_mat + b_mat·log V (полые предметы)
def _train_logvol(ctx):
    """Калиброванный объём (медиана по кадрам) и масса каждого train-объекта фолда — те же данные, что для порога T."""
    if "_tr_lv" in ctx:
        return ctx["_tr_lv"]
    import pandas as pd
    rows = pd.read_csv(E.fp.GT_ROWS)
    rows = rows[(rows["split"] == "real") & rows["object_id"].isin(ctx["train_objs"])]
    est = ctx["est"]
    out = []
    for o, g in rows.groupby("object_id"):
        r0 = g.iloc[0]
        v = np.median([est.estimate(None, None, None, r["form"], r["material"], geometry=r.to_dict()).volume_ml for _, r in g.iterrows()])
        out.append((r0["material"], r0["form"], float(np.log(max(v, 1.0))), float(np.log(r0["mass_g"])), float(np.log(r0["volume_ml"]))))
    ctx["_tr_lv"] = out
    return out


def _loglog_fit(ctx, shrink=2.0, use_gt_vol=False):
    tr = _train_logvol(ctx)
    X = np.array([t[4] if use_gt_vol else t[2] for t in tr])
    Y = np.array([t[3] for t in tr])
    b0, a0 = np.polyfit(X, Y, 1)
    fit = {"*": (a0, b0)}
    for m in set(t[0] for t in tr):
        ii = np.array([t[0] == m for t in tr])
        x, y = X[ii], Y[ii]
        if len(x) >= 3 and np.ptp(x) > 0.3:
            # ridge к общему наклону: минимизируем Σ(y−a−bx)² + shrink·(b−b0)²
            xm, ym = x.mean(), y.mean()
            sxx = ((x - xm) ** 2).sum()
            sxy = ((x - xm) * (y - ym)).sum()
            b = (sxy + shrink * b0) / (sxx + shrink)
            a = ym - b * xm
        else:
            b = b0
            a = (y - b * x).mean()
        fit[m] = (a, b)
    return fit


def _loglog_mass(fit, vols, mats):
    return np.array([float(np.exp(fit.get(m, fit["*"])[0] + fit.get(m, fit["*"])[1] * np.log(max(v, 1.0)))) for v, m in zip(vols, mats)])


def take_t4(ctx, vols, forms, mats):
    """Все материалы: take = масса(лог-лог по материалу) ≤ 500."""
    fit = _loglog_fit(ctx)
    ctx.setdefault("notes", []).append("loglog " + " ".join(f"{m}:{a:.1f}+{b:.2f}x" for m, (a, b) in sorted(fit.items())))
    mass = _loglog_mass(fit, vols, mats)
    return (mass <= 500).astype(int), mass


def take_t5(ctx, vols, forms, mats):
    """Гибрид: metal/glass/mineral — объём ≤ T; paper/plastic/wood — масса лог-лог ≤ 500."""
    fit = _loglog_fit(ctx)
    T = ctx["take_thr"]
    mass = _loglog_mass(fit, vols, mats)
    take = np.array([int(v <= T) if m in HEAVY else int(mc <= 500) for v, m, mc in zip(vols, mats, mass)])
    return take, mass


def take_t6(ctx, vols, forms, mats):
    """Как t4, но наклон общий для всех материалов (только сдвиг по материалу)."""
    fit = _loglog_fit(ctx, shrink=1e6)
    mass = _loglog_mass(fit, vols, mats)
    return (mass <= 500).astype(int), mass


def take_t7(ctx, vols, forms, mats):
    """t1 + бумага всегда take=1 (в train максимум бумаги 160 г; масса бумаги не растёт с объёмом)."""
    take, mass = take_t1(ctx, vols, forms, mats)
    take = np.array([1 if m == "paper" else t for t, m in zip(take, mats)])
    return take, mass
