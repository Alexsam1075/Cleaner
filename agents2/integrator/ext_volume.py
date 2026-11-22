"""Адаптер agents2/volume/predict_volume.VolumeModel к pipeline_v2 (--external volume_v2=<этот файл>:comp_volume).
Модель обучается на GT-строках train-объектов фолда (reports/volume_parts_gt_rows.csv + gi_env из agents/volume/cache),
предсказывает по геометрии детекций (new_geo: columns/parts/hull/footprint → вариант base4, т.к. gi_env_ml для детекций
не считается без масок). Порог take T пересчитывается той же процедурой, что в full_pipeline.volume_estimator_for,
но на объёмах VolumeModel (иначе T и объёмы несогласованы). aggregate_series агента = медиана log-объёма по серии =
то, что уже делает run_fold по gid (link_series 0.90)."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path("/home/cleaner/Cleaner")
sys.path.insert(0, str(ROOT / "agents2/volume"))
from predict_volume import VolumeModel  # noqa: E402

sys.path.insert(0, str(ROOT / "audit"))
from full_pipeline import GT_ROWS  # noqa: E402

_FITTED = {}


def _model(ctx):
    key = tuple(ctx["train_objs"])
    if key not in _FITTED:
        vm = VolumeModel().fit(ctx["train_objs"])
        rows = pd.read_csv(GT_ROWS)
        rows = rows[(rows["split"] == "real") & rows["object_id"].isin(ctx["train_objs"])]
        xs, ys = [], []
        for o, g in rows.groupby("object_id"):
            r0 = g.iloc[0]
            xs.append(np.median([vm.predict(r.to_dict(), r["form"], r["material"]) for _, r in g.iterrows()]))
            ys.append(int(r0["mass_g"] <= 500))
        xs, ys = np.array(xs), np.array(ys)
        best_t, best_s = 500.0, -1
        for t in np.unique(xs):
            for tt in (t - 1e-6, t + 1e-6):
                p = (xs <= tt).astype(int)
                sc = np.mean([(p[ys == v] == v).mean() for v in (0, 1) if (ys == v).any()])
                if sc > best_s:
                    best_t, best_s = tt, sc
        _FITTED[key] = (vm, float(best_t))
    return _FITTED[key]


def _geo(d, form):
    return d.get("new_geo_by_form", {}).get(form, d["new_geo"])


def comp_volume(ctx, forms, mats):
    vm, T = _model(ctx)
    T0 = ctx["take_thr"]
    ctx["take_thr"] = T
    ctx.setdefault("notes", []).append(f"volume_v2 {list(vm.models)} g={ {k: v['g'] for k, v in vm.models.items()} } T {T0:.0f}→{T:.0f}")
    return np.array([vm.predict(_geo(d, f), f, m) for d, f, m in zip(ctx["dets"], forms, mats)])


def comp_volume_soft(ctx, forms, mats):
    """То же, но материал — взвешивание по вероятностям P (ctx['P'])."""
    from full_pipeline import MATS
    vm, T = _model(ctx)
    ctx["take_thr"] = T
    P = ctx["P"]
    ctx.setdefault("notes", []).append("volume_v2 soft-material")
    return np.array([vm.predict(_geo(d, f), f, material_probs={m: float(p) for m, p in zip(MATS, pr)}) for d, f, pr in zip(ctx["dets"], forms, P)])


def comp_volume_keepT(ctx, forms, mats):
    """VolumeModel без пересчёта T (T базового калибратора)."""
    vm, _ = _model(ctx)
    ctx.setdefault("notes", []).append("volume_v2 keepT")
    return np.array([vm.predict(_geo(d, f), f, m) for d, f, m in zip(ctx["dets"], forms, mats)])
