"""pipeline_v2: копия финального конвейера с подключаемыми компонентами + проверки п.4:
  (a) порог формы по вложенному CV (inner OOF FormModel на train-объектах) вместо 0.5;
  (b) агрегация формы/материала по сериям с проверкой согласия (object_memory.pool, gate);
  (c) порог массы для лёгких классов (paper/plastic/wood): 500 г vs подбор на train по сбалансированной точности;
  (d) порог объёма для тяжёлых классов только по тяжёлым train-объектам (thr_by_group).
Все варианты считаются в одном процессе на общих контекстах фолдов (детекции из кэша).
Запуск: ~/venv/bin/python pipeline_v2.py [--variants a b c d ...] [--external material=модуль:функция ...]"""
import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import eval_folds as E  # noqa: E402
from eval_folds import HEAVY, MATS, fp  # noqa: E402
from form_model import FeatureBank, bal_acc, blend, head_predict  # noqa: E402
from object_memory import pool  # noqa: E402


# ---------------------------------------------------------------- (a) порог формы по вложенному CV
def form_thr_cv(ctx, grid=np.round(np.arange(0.30, 0.71, 0.05), 2)):
    fm = ctx["form_model"]
    cfg = fm.cfg
    real = fm.src == "real_train"
    rs = np.random.default_rng(fm.seed)
    train_objs = sorted(ctx["train_objs"])
    groups = np.array_split(rs.permutation(train_objs), fm.inner_folds)
    tr_real = np.where(np.isin(fm.oid, train_objs) & real)[0]
    X = fm.X
    Pin = np.zeros((len(fm.y), 2))
    for g in groups:
        rest = [o for o in train_objs if o not in set(g)]
        a = fm._idx(rest, cfg["sw"])
        b = np.where(np.isin(fm.oid, g) & real)[0]
        Pin[b] = head_predict(X, fm.y, fm.oid, fm.src, a, b, cfg["sw"], cfg["wd"])
    Lz = fm.bank.zs_logits(cfg["zs"])
    P = blend(Pin[tr_real], None if Lz is None else Lz[tr_real], cfg["w"])
    y = fm.y[tr_real]
    oid = fm.oid[tr_real]
    # сбалансированная точность по объектам (каждый объект — 1 голос, как в метрике D по рамкам не считается, но честнее)
    best = None
    for t in grid:
        pred_solid = (P[:, 0] < t).astype(int)   # p_bulk >= t -> bulk(0), иначе solid(1)
        acc = {}
        for o in set(oid):
            ii = oid == o
            acc[o] = float((pred_solid[ii] == y[ii]).mean())
        per_cls = [np.mean([acc[o] for o in acc if y[oid == o][0] == c]) for c in (0, 1)]
        s = float(np.mean(per_cls))
        key = (s, -abs(t - 0.5))
        if best is None or key > best[0]:
            best = (key, float(t))
    return best[1], best[0][0]


# ---------------------------------------------------------------- (c) порог массы лёгких классов по train
def mass_thr_cv(ctx):
    import pandas as pd
    rows = pd.read_csv(fp.GT_ROWS)
    rows = rows[(rows["split"] == "real") & rows["object_id"].isin(ctx["train_objs"])]
    est, dens = ctx["est"], ctx["dens"]
    xs, ys = [], []
    for o, g in rows.groupby("object_id"):
        r0 = g.iloc[0]
        if r0["material"] in HEAVY:
            continue
        v = np.median([est.estimate(None, None, None, r["form"], r["material"], geometry=r.to_dict()).volume_ml for _, r in g.iterrows()])
        xs.append(v * dens.get((r0["form"], r0["material"]), dens.get(("*", r0["material"]), 0.5)))
        ys.append(int(r0["mass_g"] <= 500))
    xs, ys = np.array(xs), np.array(ys)
    u = np.sort(np.unique(xs))
    cands = list((u[:-1] + u[1:]) / 2) + [500.0]
    best = None
    for t in cands:
        p = (xs <= t).astype(int)
        s = np.mean([(p[ys == v] == v).mean() for v in (0, 1) if (ys == v).any()])
        key = (s, -abs(np.log(t / 500.0)))
        if best is None or key > best[0]:
            best = (key, float(t))
    return best[1], best[0][0], int((ys == 0).sum()), len(ys)


# ---------------------------------------------------------------- (d) порог объёма по тяжёлым классам
def heavy_thr_cv(ctx):
    import pandas as pd
    rows = pd.read_csv(fp.GT_ROWS)
    rows = rows[(rows["split"] == "real") & rows["object_id"].isin(ctx["train_objs"])]
    est = ctx["est"]
    xs, ys = [], []
    for o, g in rows.groupby("object_id"):
        r0 = g.iloc[0]
        if r0["material"] not in HEAVY:
            continue
        xs.append(np.median([est.estimate(None, None, None, r["form"], r["material"], geometry=r.to_dict()).volume_ml for _, r in g.iterrows()]))
        ys.append(int(r0["mass_g"] <= 500))
    xs, ys = np.array(xs), np.array(ys)
    u = np.sort(np.unique(xs))
    cands = list((u[:-1] + u[1:]) / 2)
    best = None
    for t in cands:
        p = (xs <= t).astype(int)
        s = np.mean([(p[ys == v] == v).mean() for v in (0, 1) if (ys == v).any()])
        key = (s, -abs(np.log(t / ctx["take_thr"])))
        if best is None or key > best[0]:
            best = (key, float(t))
    return best[1], best[0][0], int((ys == 0).sum()), len(ys)


# ---------------------------------------------------------------- варианты
_MEMO = {}


def memo(fn, key):
    """Базовые материал/форма одинаковы для всех вариантов фолда — считаем один раз на фолд."""
    def wrapped(ctx):
        k = (key, ctx["fold"], ctx["prompts"])
        if k not in _MEMO:
            _MEMO[k] = (fn(ctx), {kk: ctx[kk] for kk in ("w_zs", "form_model", "form_bank") if kk in ctx})
        out, extra = _MEMO[k]
        ctx.update(extra)
        return out.copy() if hasattr(out, "copy") else out
    return wrapped


def make_variants(names, external, combos=()):
    EXT = {}
    base = E.base_components({"take": "hybrid", "mass_thr": 500.0})
    base["material"] = memo(E.comp_material, "material")
    base["form"] = memo(E.comp_form, "form")
    E.comp_form_cached = base["form"]
    V = {}

    def with_(**kw):
        c = dict(base)
        c.update(kw)
        return c

    V["base"] = (with_(), {})
    if "a" in names:
        def form_a(ctx):
            pb = base["form"](ctx)
            t, s = form_thr_cv(ctx)
            ctx["form_thr"] = t
            ctx.setdefault("notes", []).append(f"form_thr={t:.2f} (inner bal.acc {s:.3f})")
            return pb
        V["a_form_thr_cv"] = (with_(form=form_a), {})
    if "b" in names:
        for gate in (0.75, 0.9):
            def series_b(ctx, P, pb, gate=gate):
                if ctx["gid"] is None:
                    return P, pb
                P2, pb2, _ = pool(ctx["dets"], ctx["gid"], P, pb, np.ones(len(P)), gate=gate, use_mat=True, use_form=False, use_vol=False)
                ctx.setdefault("notes", []).append(f"mat pooled: {int((P2.argmax(1) != P.argmax(1)).sum())} изменений")
                return P2, pb2
            V[f"b_mat_gate{gate}"] = (with_(series=series_b), {})

        def series_bf(ctx, P, pb):
            if ctx["gid"] is None:
                return P, pb
            _, pb2, _ = pool(ctx["dets"], ctx["gid"], P, pb, np.ones(len(P)), form_gate=0.7, use_mat=False, use_form=True, use_vol=False)
            ctx.setdefault("notes", []).append(f"form pooled: {int(((pb2 >= 0.5) != (pb >= 0.5)).sum())} изменений")
            return P, pb2
        V["b_form_gate0.7"] = (with_(series=series_bf), {})

        def series_bmf(ctx, P, pb):
            if ctx["gid"] is None:
                return P, pb
            P2, pb2, _ = pool(ctx["dets"], ctx["gid"], P, pb, np.ones(len(P)), gate=0.75, form_gate=0.7, use_mat=True, use_form=True, use_vol=False)
            return P2, pb2
        V["b_mat0.75+form0.7"] = (with_(series=series_bmf), {})
    if "c" in names:
        def take_c(ctx, vols, forms, mats):
            if "mass_thr_cv" not in ctx:
                t, s, nh, n = mass_thr_cv(ctx)
                ctx["mass_thr_cv"] = t
                ctx.setdefault("notes", []).append(f"mass_thr={t:.0f} г (bal.acc {s:.3f}, тяжёлых {nh}/{n} лёгких классов)")
            return E.comp_take(ctx, vols, forms, mats, "hybrid", ctx["mass_thr_cv"])
        V["c_mass_thr_cv"] = (with_(take=take_c), {})

        def take_c2(ctx, vols, forms, mats):
            if "mass_thr_cv" not in ctx:
                t, s, nh, n = mass_thr_cv(ctx)
                ctx["mass_thr_cv"] = t
            return E.comp_take(ctx, vols, forms, mats, "mass", ctx["mass_thr_cv"])
        V["c_all_mass_thr_cv"] = (with_(take=take_c2), {})
    if "d" in names:
        def take_d(ctx, vols, forms, mats):
            if "heavy_thr" not in ctx:
                t, s, nh, n = heavy_thr_cv(ctx)
                ctx["heavy_thr"] = t
                ctx.setdefault("notes", []).append(f"heavy_thr={t:.0f} мл (bal.acc {s:.3f}, тяжёлых {nh}/{n})")
            dens, T = ctx["dens"], ctx["heavy_thr"]
            mass = np.array([v * dens.get((f, m), dens.get(("*", m), 0.5)) for v, f, m in zip(vols, forms, mats)])
            take = np.array([int(v <= T) if m in HEAVY else int(mc <= 500) for v, m, mc in zip(vols, mats, mass)])
            return take, mass
        V["d_heavy_thr_group"] = (with_(take=take_d), {})
    if "e" in names:
        # (e) порог объёма T: не argmax сбалансированной точности (хрупкий при 5–6 тяжёлых объектах),
        # а сбалансированная логистическая регрессия по log-объёму на train-объектах (точка p=0.5)
        def take_e(ctx, vols, forms, mats):
            if "T_logit" not in ctx:
                import pandas as pd
                rows = pd.read_csv(fp.GT_ROWS)
                rows = rows[(rows["split"] == "real") & rows["object_id"].isin(ctx["train_objs"])]
                est = ctx["est"]
                xs, ys = [], []
                for o, g in rows.groupby("object_id"):
                    r0 = g.iloc[0]
                    xs.append(np.log(np.median([est.estimate(None, None, None, r["form"], r["material"], geometry=r.to_dict()).volume_ml for _, r in g.iterrows()])))
                    ys.append(int(r0["mass_g"] > 500))
                xs, ys = np.array(xs), np.array(ys)
                w = np.where(ys == 1, 0.5 / max(ys.sum(), 1), 0.5 / max((ys == 0).sum(), 1))
                a_, b_ = 0.0, 0.0
                for _ in range(500):   # градиентный спуск, L2 слабая
                    z = a_ * (xs - xs.mean()) + b_
                    p = 1 / (1 + np.exp(-z))
                    ga = np.sum(w * (p - ys) * (xs - xs.mean())) + 1e-3 * a_
                    gb = np.sum(w * (p - ys))
                    a_ -= 2.0 * ga
                    b_ -= 2.0 * gb
                T = float(np.exp(xs.mean() - b_ / max(a_, 1e-6)))
                ctx["T_logit"] = T
                ctx.setdefault("notes", []).append(f"T_logit={T:.0f} мл (было {ctx['take_thr']:.0f})")
            dens, T = ctx["dens"], ctx["T_logit"]
            mass = np.array([v * dens.get((f, m), dens.get(("*", m), 0.5)) for v, f, m in zip(vols, forms, mats)])
            take = np.array([int(v <= T) if m in HEAVY else int(mc <= 500) for v, m, mc in zip(vols, mats, mass)])
            return take, mass
        V["e_T_logit"] = (with_(take=take_e), {})
    if "f" in names:
        # (f) порог объёма T = середина (геом.) плато наилучшей сбалансированной точности на train-объектах,
        # а не его нижний край (full_pipeline берёт первый максимум → T упирается в самый большой лёгкий объект)
        def take_f(ctx, vols, forms, mats):
            if "T_mid" not in ctx:
                import pandas as pd
                rows = pd.read_csv(fp.GT_ROWS)
                rows = rows[(rows["split"] == "real") & rows["object_id"].isin(ctx["train_objs"])]
                est = ctx["est"]
                xs, ys = [], []
                for o, g in rows.groupby("object_id"):
                    r0 = g.iloc[0]
                    xs.append(np.median([est.estimate(None, None, None, r["form"], r["material"], geometry=r.to_dict()).volume_ml for _, r in g.iterrows()]))
                    ys.append(int(r0["mass_g"] <= 500))
                xs, ys = np.array(xs), np.array(ys)
                u = np.sort(np.unique(xs))
                cands = np.sqrt(u[:-1] * u[1:])
                scores = []
                for t in cands:
                    p = (xs <= t).astype(int)
                    scores.append(np.mean([(p[ys == v] == v).mean() for v in (0, 1) if (ys == v).any()]))
                scores = np.array(scores)
                best = np.where(scores >= scores.max() - 1e-9)[0]
                # плато = все лучшие кандидаты; берём геом. середину между крайними соседями плато
                lo = u[best.min()]
                hi = u[best.max() + 1]
                T = float(np.sqrt(lo * hi))
                ctx["T_mid"] = T
                ctx.setdefault("notes", []).append(f"T_mid={T:.0f} мл (плато {lo:.0f}–{hi:.0f}; было {ctx['take_thr']:.0f})")
            dens, T = ctx["dens"], ctx["T_mid"]
            mass = np.array([v * dens.get((f, m), dens.get(("*", m), 0.5)) for v, f, m in zip(vols, forms, mats)])
            take = np.array([int(v <= T) if m in HEAVY else int(mc <= 500) for v, m, mc in zip(vols, mats, mass)])
            return take, mass
        V["f_T_plateau_mid"] = (with_(take=take_f), {})
    if "g" in names:
        # (g) ранжирование рамок для AP: score = conf × p(форма) (base) vs conf, conf×sqrt(p), conf×(0.5+0.5p); conf_min/NMS
        import math as _m
        for sname, fn in {"conf": lambda c, p: c, "conf*sqrt(p)": lambda c, p: c * _m.sqrt(p), "conf*(.5+.5p)": lambda c, p: c * (0.5 + 0.5 * p),
                          "p_only": lambda c, p: p * (0.5 + 0.5 * c)}.items():
            def form_g(ctx, fn=fn):
                ctx["score_fn"] = fn
                return base["form"](ctx)
            V[f"g_score={sname}"] = (with_(form=form_g), {})
        for cm, nm in ((0.05, 0.6), (0.12, 0.6), (0.08, 0.5), (0.08, 0.7)):
            def form_g2(ctx, cm=cm, nm=nm):
                ctx["conf_min"], ctx["nms"] = cm, nm
                return base["form"](ctx)
            V[f"g_conf{cm}_nms{nm}"] = (with_(form=form_g2), {})
    if "h" in names:
        V["h_take_volume"] = (with_(take=lambda ctx, v, f, m: E.comp_take(ctx, v, f, m, "volume")), {})

        def take_h2(ctx, vols, forms, mats):
            # объём ≤ T для тяжёлых классов, объём ≤ T_light для лёгких классов (оба порога — середина плато bal.acc на train)
            if "T_light" not in ctx:
                import pandas as pd
                rows = pd.read_csv(fp.GT_ROWS)
                rows = rows[(rows["split"] == "real") & rows["object_id"].isin(ctx["train_objs"])]
                est = ctx["est"]
                xs, ys, hv = [], [], []
                for o, g in rows.groupby("object_id"):
                    r0 = g.iloc[0]
                    xs.append(np.median([est.estimate(None, None, None, r["form"], r["material"], geometry=r.to_dict()).volume_ml for _, r in g.iterrows()]))
                    ys.append(int(r0["mass_g"] <= 500))
                    hv.append(r0["material"] in HEAVY)
                xs, ys, hv = np.array(xs), np.array(ys), np.array(hv)

                def mid(x, y):
                    u = np.sort(np.unique(x))
                    c = np.sqrt(u[:-1] * u[1:])
                    sc = np.array([np.mean([((x <= t).astype(int)[y == k] == k).mean() for k in (0, 1) if (y == k).any()]) for t in c])
                    b = np.where(sc >= sc.max() - 1e-9)[0]
                    return float(np.sqrt(u[b.min()] * u[b.max() + 1]))
                ctx["T_all"] = mid(xs, ys)
                ctx["T_light"] = mid(xs[~hv], ys[~hv]) if (ys[~hv] == 0).any() else 1e9
                ctx.setdefault("notes", []).append(f"T_all={ctx['T_all']:.0f}, T_light={ctx['T_light']:.0f}")
            dens = ctx["dens"]
            mass = np.array([v * dens.get((f, m), dens.get(("*", m), 0.5)) for v, f, m in zip(vols, forms, mats)])
            take = np.array([int(v <= ctx["T_all"]) if m in HEAVY else int(v <= ctx["T_light"]) for v, m in zip(vols, mats)])
            return take, mass
        V["h_vol_T_by_group"] = (with_(take=take_h2), {})
    for spec in external:
        # формат: имя=module_path:function[,material|form|volume|take]
        name, rest = spec.split("=", 1)
        modpath, func = rest.split(":")
        role = name.split("_")[0]
        import importlib.util
        sp = importlib.util.spec_from_file_location(Path(modpath).stem, modpath)
        mod = importlib.util.module_from_spec(sp)
        sys.path.insert(0, str(Path(modpath).parent))
        sp.loader.exec_module(mod)
        ext_fn = getattr(mod, func)
        EXT[name] = (role, ext_fn)
        if role == "series":
            continue
        V[name] = (with_(**{role: ext_fn}), {})
        if role == "form":
            mats = [v[0]["material"] for k, v in V.items() if k.startswith("material_") and k.endswith("_v2")]
            if mats:
                V[name + "+matv2"] = (with_(form=ext_fn, material=mats[0]), {})
                if "b" in names:
                    V[name + "+matv2+b"] = (with_(form=ext_fn, material=mats[0], series=V["b_mat_gate0.75"][0]["series"]), {})
        if role == "material":
            def mat_ens(ctx, ext_fn=ext_fn):
                P0 = base["material"](ctx)
                w0 = ctx.get("w_zs")
                P1 = ext_fn(ctx)
                ctx["w_zs"] = (w0, ctx.get("w_zs"))
                lp = 0.5 * (np.log(P0 + 1e-9) + np.log(P1 + 1e-9))
                P = np.exp(lp - lp.max(1, keepdims=True))
                P /= P.sum(1, keepdims=True)
                ctx.setdefault("notes", []).append(f"ens: argmax отличается от base в {int((P.argmax(1) != P0.argmax(1)).sum())} дет.")
                return P
            V[name + "+base_ens"] = (with_(material=mat_ens), {})
            if "a" in names:
                V[name + "+a"] = (with_(material=ext_fn, form=V["a_form_thr_cv"][0]["form"]), {})
            if "b" in names:
                V[name + "+b_mat0.75"] = (with_(material=ext_fn, series=V["b_mat_gate0.75"][0]["series"]), {})
            if "a" in names and "b" in names:
                V[name + "+a+b_mat0.75"] = (with_(material=ext_fn, form=V["a_form_thr_cv"][0]["form"], series=V["b_mat_gate0.75"][0]["series"]), {})
    # --combo имя=токен+токен+...: токены — имена внешних компонентов (role_xxx из --external) или встроенные
    # b (материал по серии gate .75), a (порог формы CV); несколько series-компонентов выполняются по цепочке
    for spec in combos:
        name, rest = spec.split("=", 1)
        kw, chain = {}, []
        for tok in rest.split("+"):
            if tok == "b":
                chain.append(V["b_mat_gate0.75"][0]["series"])
            elif tok == "a":
                kw["form"] = V["a_form_thr_cv"][0]["form"]
            else:
                role, fn = EXT[tok]
                if role == "series":
                    chain.append(fn)
                else:
                    kw[role] = fn
        if chain:
            def series_chain(ctx, P, pb, chain=tuple(chain)):
                for f in chain:
                    P, pb = f(ctx, P, pb)
                return P, pb
            kw["series"] = series_chain
        V[name] = (with_(**kw), {})
    if "k" in names:
        # (k) take для тяжёлых классов по массе класса (объём × плотность класса ≤ 500) — т.е. «всё по массе класса», но с плотностями
        # по (форма, материал); и «масса ≤ 500 ИЛИ объём ≤ T» для тяжёлых
        def take_k1(ctx, vols, forms, mats):
            dens, T = ctx["dens"], ctx["take_thr"]
            mass = np.array([v * dens.get((f, m), dens.get(("*", m), 0.5)) for v, f, m in zip(vols, forms, mats)])
            take = np.array([int(v <= T and mc <= 500) if m in HEAVY else int(mc <= 500) for v, m, mc in zip(vols, mats, mass)])
            return take, mass

        def take_k2(ctx, vols, forms, mats):
            dens, T = ctx["dens"], ctx["take_thr"]
            mass = np.array([v * dens.get((f, m), dens.get(("*", m), 0.5)) for v, f, m in zip(vols, forms, mats)])
            take = np.array([int(v <= T or mc <= 500) if m in HEAVY else int(mc <= 500) for v, m, mc in zip(vols, mats, mass)])
            return take, mass
        V["k_heavy_vol_and_mass"] = (with_(take=take_k1), {})
        V["k_heavy_vol_or_mass"] = (with_(take=take_k2), {})
        if external:
            ext = [v for k, v in V.items() if k.endswith("_v2")][0][0]["material"]
            V["ext+k_and"] = (with_(material=ext, take=take_k1), {})
            V["ext+k_or"] = (with_(material=ext, take=take_k2), {})
    if "w" in names:
        # (w) форма = w·p_bulk(CLIP) + (1−w)·p_bulk(детектор); p_bulk(детектор) по перекрывающимся рамкам (IoU≥0.6):
        # conf_bulk / (conf_bulk + conf_solid); без пары — 1/0 по классу рамки
        from eval_folds import _iou as _iou_

        def det_pbulk(dets):
            out = np.zeros(len(dets))
            for i, d in enumerate(dets):
                cb = cs = 0.0
                for j, o in enumerate(dets):
                    if o["frame"] == d["frame"] and _iou_(d["box"], o["box"]) >= 0.6:
                        if o["form"] == "bulk":
                            cb = max(cb, o["score"])
                        else:
                            cs = max(cs, o["score"])
                out[i] = cb / (cb + cs) if cb + cs > 0 else (1.0 if d["form"] == "bulk" else 0.0)
            return out
        for w in (0.3, 0.5, 0.7):
            def form_w(ctx, w=w):
                pb = base["form"](ctx)
                pd_ = det_pbulk(ctx["dets"])
                mix = w * pb + (1 - w) * pd_
                ctx.setdefault("notes", []).append(f"форма w={w}: смен {int(((mix >= 0.5) != (pb >= 0.5)).sum())}")
                return mix
            V[f"w_form_mix{w}"] = (with_(form=form_w), {})
            if external:
                ext = [v for k, v in V.items() if k.endswith("_v2")][0][0]["material"]
                V[f"ext+w_form_mix{w}"] = (with_(material=ext, form=form_w), {})
    if "only_ext" in names:
        V = {k: v for k, v in V.items() if k == "base" or "_v2" in k}
    if "hx" in names and external:
        # take-варианты поверх внешнего материала
        ext = [v for k, v in V.items() if k.endswith("_v2")][0][0]["material"]
        if "h_vol_T_by_group" in V:
            V["ext+h_vol_T_by_group"] = (with_(material=ext, take=V["h_vol_T_by_group"][0]["take"]), {})
        if "h_take_volume" in V:
            V["ext+h_take_volume"] = (with_(material=ext, take=V["h_take_volume"][0]["take"]), {})
    return V


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--folds", type=int, nargs="+", default=[1, 2, 3])
    ap.add_argument("--variants", nargs="+", default=["a", "b", "c", "d"])
    ap.add_argument("--external", nargs="*", default=[])
    ap.add_argument("--combo", nargs="*", default=[])
    ap.add_argument("--only", nargs="*", default=None, help="считать только эти варианты (+base)")
    ap.add_argument("--agg", type=float, default=0.90)
    ap.add_argument("--agg-gap", type=int, default=3)
    ap.add_argument("--tag", default="v2")
    ap.add_argument("--two-box", type=float, default=0.0, help="вторая рамка с противоположной формой, если её p >= порога (0 = выкл); score = conf*p")
    ap.add_argument("--two-box-mult", type=float, nargs="*", default=[0.999], help="множители score второй рамки (несколько = несколько вариантов)")
    a = ap.parse_args()
    text = E.load_text()
    F, meta = fp.train_cache()
    V = make_variants(set(a.variants), a.external, a.combo)
    if a.only is not None:
        V = {k: v for k, v in V.items() if k == "base" or k in set(a.only)}
    opts = {"take": "hybrid", "agg": a.agg, "agg_gap": a.agg_gap, "form_thr": 0.5, "mass_thr": 500.0, "prompts": "rich"}
    res = defaultdict(dict)
    for fold in a.folds:
        ctx0 = E.fold_context(fold, text, F, meta, "rich")
        print(f"== fold {fold}: T={ctx0['take_thr']:.0f} мл, детекций {len(ctx0['dets'])}", flush=True)
        VV = list(V.items())
        if a.two_box > 0:
            VV = [(f"{n}|tb{m}", (c, m)) for n, (c, _) in VV for m in a.two_box_mult]
        for name, (comp, tbm) in VV:
            ctx = dict(ctx0)
            ctx.pop("form_thr", None)
            if a.two_box > 0:
                ctx["two_box"], ctx["two_box_mult"] = a.two_box, tbm
            sub, c = E.run_fold(ctx, comp, opts)
            sub.to_csv(HERE / "cache" / f"fold{fold}_{a.tag}_{name}.csv", index=False)
            res[name][fold] = c
            print(f"  {name:22s} Score {c['Score']:6.2f} D {c['D']:.3f} M {c['M']:.3f} V {c['V']:.3f} S {c['S']:.3f} | {'; '.join(ctx.get('notes', []))}", flush=True)
    print("\n== Среднее ==")
    out = {}
    for name, r in res.items():
        comps = [r[f] for f in sorted(r)]
        mean = {k: float(np.mean([x[k] for x in comps])) for k in ("Score", "D", "M", "V", "S")}
        sd = float(np.std([x["Score"] for x in comps]))
        f23 = float(np.mean([r[f]["Score"] for f in r if f in (2, 3)]))
        f4 = r[4]["Score"] if 4 in r else float("nan")
        m123 = float(np.mean([r[f]["Score"] for f in r if f in (1, 2, 3)]))
        out[name] = {"mean": mean, "std": sd, "f23": f23, "f4": f4, "per_fold": r}
        print(f"  {name:22s} ф1-3 {m123:6.2f} ± {sd:4.2f} | ф2-3 {f23:6.2f} | ф4 {f4:6.2f} | D {mean['D']:.3f} M {mean['M']:.3f} V {mean['V']:.3f} S {mean['S']:.3f}")
    json.dump(out, open(HERE / "cache" / f"report_{a.tag}.json", "w"), indent=1)


if __name__ == "__main__":
    main()
