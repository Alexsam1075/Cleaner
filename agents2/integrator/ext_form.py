"""Адаптер agents2/form/predict_form.FormClassifier (финал 20:22: FormModel2 = голова + zero-shot ensemble-prior,
3× inner CV, top-3) к pipeline_v2 (--external form_v2=<этот файл>:comp_form). Признаки v2 (DINO/глубина/FastSAM) для моих
детекций — cache/form_v2_fold{k}.npz / form_v2_test.npz (form_feats.py: маски повторным прогоном детектора фолда).
series_pool — pool_series агента (кластеры CLIP-L c14 cos≥0.9, окно ±10 кадров, взвешенное по conf среднее логитов)."""
import sys
from pathlib import Path

import numpy as np

ROOT = Path("/home/cleaner/Cleaner")
HERE = Path(__file__).resolve().parent
_MODS = ("common", "form_model", "bank", "prompts", "evalkit", "depth_feats", "form_model2", "predict_form", "extract")
_saved = {k: sys.modules.pop(k) for k in _MODS if k in sys.modules}
sys.path.insert(0, str(ROOT / "agents2/form"))
from bank import Bank, load_prompts  # noqa: E402
from form_model import GEO_KEYS  # noqa: E402
from predict_form import DEFAULTS, FormClassifier, pool_series  # noqa: E402
sys.path.remove(str(ROOT / "agents2/form"))
for k in _MODS:
    sys.modules.pop(k, None)
sys.modules.update(_saved)

_FITTED = {}


def _v2(ctx):
    name = "test" if ctx["fold"] == "test" else f"fold{ctx['fold']}"
    Z = np.load(HERE / "cache" / f"form_v2_{name}.npz")
    assert len(Z["depth"]) == len(ctx["dets"]), (name, len(Z["depth"]), len(ctx["dets"]))
    return Z


def comp_form(ctx, topk=3, tune_thr=False):
    key = (ctx["fold"], tuple(ctx["train_objs"]), topk, tune_thr)
    if key not in _FITTED:
        _FITTED[key] = FormClassifier(topk=topk, tune_thr=tune_thr).fit(ctx["train_objs"])
    fc = _FITTED[key]
    dets, embs = ctx["dets"], ctx["embs"]
    Z = _v2(ctx)
    emb = {f"clipL_{v}": np.asarray(embs[("clipL", v)], np.float32) for v in ("c14", "m10", "m14", "c20")}
    emb["siglip_c14"] = np.asarray(embs[("siglip", "c14")], np.float32)
    emb["geo"] = np.array([[d["geo"][k] for k in GEO_KEYS] for d in dets], np.float32).reshape(len(dets), len(GEO_KEYS))
    for k in ("dino_pool", "dino_stats", "depth", "sam"):
        emb[k] = Z[k]
    T, prompts = load_prompts()
    ctx["form_cfg"] = (fc.model.cfgs, fc.thr)
    ctx.setdefault("notes", []).append(f"form_v2 cfgs={[(c[0], c[3]) for c in fc.model.cfgs]} thr={fc.thr} inner={tuple(round(x, 3) for x in fc.model.inner_score)}")
    return fc.predict_bank(Bank(emb, T, prompts))


def comp_form_thr(ctx):
    return comp_form(ctx, topk=3, tune_thr=True)


def series_pool(ctx, P, pb, sim=DEFAULTS["pool_sim"], gap=DEFAULTS["pool_gap"]):
    """Память серий для формы (pool_series агента) поверх любого p_bulk; материал не трогает."""
    if len(pb) == 0:
        return P, pb
    dd = [{"frame": d["frame"], "conf": d["score"], "p_bulk": float(p)} for d, p in zip(ctx["dets"], pb)]
    out = pool_series(dd, ctx["embs"][("clipL", "c14")], sim=sim, gap=gap)
    pb2 = np.array([o["p_bulk"] for o in out])
    ctx.setdefault("notes", []).append(f"form pool_series: смен {int(((pb2 >= 0.5) != (pb >= 0.5)).sum())}, кластеров {len({o['cluster'] for o in out})}")
    return P, pb2
