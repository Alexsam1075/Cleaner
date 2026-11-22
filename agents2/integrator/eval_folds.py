"""Честная оценка финального конвейера (predict_test.py: 1 рамка, rich, гибридный take, память по сериям)
на фолдах 1–3 через официальный score.py. Детекции берутся из кэша audit/cache/fold{k}_dets.pkl (детектор фолда,
обученный без val-объектов), все остальные модели/пороги обучаются только на train-объектах фолда.

Компоненты подключаемые (для pipeline_v2): словарь `comp` с функциями
  material(ctx) -> P (n,6) по MATS;  form(ctx) -> p_bulk (n,);  volume(ctx, forms, mats) -> vols (n,)
  take(ctx, vols, forms, mats) -> take (n,)
ctx: dict(fold, train_objs, frames, dets, embs, est, take_thr, dens, text, F, meta, P, p_bulk, gid)

Запуск: ~/venv/bin/python eval_folds.py [--take hybrid|volume|mass] [--prompts rich] [--agg 0.90] [--form-thr 0.5|cv]
        [--agg-mat none|gate] [--mass-thr 500|cv] [--tag имя]"""
import argparse
import json
import math
import pickle
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path("/home/cleaner/Cleaner")
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "audit"))
sys.path.insert(0, str(ROOT / "agents/form"))
sys.path.insert(0, str(ROOT / "references"))
import full_pipeline as fp  # noqa: E402
from full_pipeline import C, D, FOLDS, MATS, MODELS, VIEWS, _iou, serialize, touches_border_px  # noqa: E402
import score as S  # noqa: E402
from form_cv import load_bank  # noqa: E402
from form_model import FeatureBank, FormModel, GEO_KEYS  # noqa: E402

HEAVY = {"metal", "glass", "mineral"}
FORMS = ["bulk", "solid"]
TEXT_CACHE = HERE / "text_mat.npz"


def load_text():
    t = np.load(TEXT_CACHE)
    return {tuple(k.split("|")): t[k] for k in t.files}


def class_density(meta, train_objs):
    by = defaultdict(list)
    seen = set()
    for m in meta:
        if m["src"] == "real_train" and m["oid"] in train_objs and m["oid"] not in seen:
            seen.add(m["oid"])
            by[(m["form"], m["mat"])].append(m["mass"] / m["vol"])
            by[("*", m["mat"])].append(m["mass"] / m["vol"])
    return {k: float(np.exp(np.mean(np.log(v)))) for k, v in by.items()}


def gt_solution(frames):
    lab = json.load(open(D / "labels/real_train.json"))
    keep = [im for im in lab["images"] if Path(im["file_name"]).parent.name in set(frames)]
    ids = {im["id"] for im in keep}
    data = {"categories": lab["categories"], "images": keep, "annotations": [a for a in lab["annotations"] if a["image_id"] in ids]}
    p = HERE / "cache" / f"val_labels_fold_{len(frames)}_{abs(hash(tuple(sorted(frames)))) % 10**8}.json"
    p.parent.mkdir(exist_ok=True)
    p.write_text(json.dumps(data))
    sol = S.labels_to_solution(p)
    return {str(i): S._parse(c, False, str(i)) for i, c in zip(sol.image_id, sol.objects)}


def geo_for(d, form):
    return d.get("new_geo_by_form", {}).get(form, d["new_geo"])


# ---------------------------------------------------------------- базовые компоненты (= predict_test.py)

def comp_material(ctx):
    pm, w = fp.material_model(ctx["F"], ctx["meta"], ctx["train_objs"], ctx["text"], ctx["prompts"])
    embs = ctx["embs"]
    ctx["w_zs"] = w
    return pm(embs[("clipL", "c14")], fp.zero_shot(embs, ctx["text"], ctx["prompts"]))


def comp_form(ctx):
    fbank, fy, foid, fsrc, _ = load_bank()
    grid = {"spec": ["clipL_c14+zs"], "sw": [0.5], "wd": [1e-2]}
    fm = FormModel(fbank, fy, foid, fsrc, grid=grid).fit(ctx["train_objs"])
    dets, embs = ctx["dets"], ctx["embs"]
    femb = {f"{t}_{v}": embs[(t, v)] for t in MODELS for v in VIEWS}
    femb["geo"] = np.array([[d["geo"][k] for k in GEO_KEYS] for d in dets], np.float32).reshape(len(dets), len(GEO_KEYS))
    ctx["form_model"] = fm
    ctx["form_bank"] = (fbank, fy, foid, fsrc)
    return fm.predict_bank(FeatureBank(femb, fbank.text))[:, 0]


def comp_volume(ctx, forms, mats):
    est = ctx["est"]
    return np.array([est.estimate(None, None, None, f, m, geometry=geo_for(d, f)).volume_ml for d, f, m in zip(ctx["dets"], forms, mats)])


def comp_take(ctx, vols, forms, mats, mode="hybrid", mass_thr=500.0):
    dens, T = ctx["dens"], ctx["take_thr"]
    mass = np.array([v * dens.get((f, m), dens.get(("*", m), 0.5)) for v, f, m in zip(vols, forms, mats)])
    if mode == "volume":
        take = (vols <= T).astype(int)
    elif mode == "mass":
        take = (mass <= mass_thr).astype(int)
    else:
        take = np.array([int(v <= T) if m in HEAVY else int(mc <= mass_thr) for v, m, mc in zip(vols, mats, mass)])
    return take, mass


def assemble(ctx, p_bulk, forms, mats, vols, mass, take, conf_min=0.08, nms=0.6):
    conf_min = ctx.get("conf_min", conf_min)
    nms = ctx.get("nms", nms)
    per = defaultdict(list)
    for d, pb, f, m, v, mc, t in zip(ctx["dets"], p_bulk, forms, mats, vols, mass, take):
        if d["score"] < conf_min or touches_border_px(d["box"]):
            continue
        pm = max(pb, 1 - pb)
        sc = ctx["score_fn"](d["score"], pm) if "score_fn" in ctx else d["score"] * pm
        per[d["frame"]].append({"box": d["box"], "score": sc, "form": f, "material": m,
                                "volume": float(v), "mass": float(mc), "take": int(t), "conf": d["score"], "pm": pm})
    two_box = ctx.get("two_box", 0.0)
    tb_mult = ctx.get("two_box_mult", 0.999)
    fb = ctx.get("fallback_conf", 0.0)
    if fb > 0:
        for d, pb, f, m, v, mc, t in zip(ctx["dets"], p_bulk, forms, mats, vols, mass, take):
            if fb <= d["score"] < conf_min and not touches_border_px(d["box"]):
                pm = max(pb, 1 - pb)
                per.setdefault(d["frame"] + "#fb", []).append({"box": d["box"], "score": d["score"] * pm, "form": f, "material": m,
                                                               "volume": float(v), "mass": float(mc), "take": int(t), "conf": d["score"], "pm": pm})
    rows = []
    for fr in ctx["frames"]:
        keep = []
        for c in sorted(per.get(fr, []), key=lambda o: -o["score"]):
            if all(_iou(c["box"], k["box"]) < nms for k in keep):
                keep.append(c)
        if not keep and per.get(fr + "#fb"):
            keep = [max(per[fr + "#fb"], key=lambda o: o["score"])]
            ctx.setdefault("notes", []).append(f"fallback box {fr} conf {keep[0]['conf']:.3f}")
        if two_box:
            extra = []
            for c in keep:
                p2 = 1 - c["pm"]
                if p2 >= two_box - 1e-9:
                    o = dict(c)
                    o["form"] = "solid" if c["form"] == "bulk" else "bulk"
                    o["score"] = min(c["score"], c["conf"] * p2) * tb_mult
                    extra.append(o)
            keep = keep + extra
        rows.append({"image_id": fr, "prediction": serialize(sorted(keep, key=lambda o: -o["score"])[:30])})
    return pd.DataFrame(rows)


def evaluate(sub, gt):
    pr = {str(i): S._parse(c, True, str(i)) for i, c in zip(sub.image_id, sub.prediction)}
    return S.components(gt, pr)


HOLDOUT_SPLIT = Path("/home/cleaner/CleanerVolume/holdout/configs/object_id_split.json")
HOLDOUT_DETS = C / "fold4_dets_holdout4.pkl"


def fold_context(fold, text, F, meta, prompts="rich"):
    split = json.load(open(FOLDS[fold][0] if fold in FOLDS else HOLDOUT_SPLIT))
    train_objs = sorted(split["train_object_ids"])
    val_objs = set(split["validation_object_ids"])
    assert not (set(train_objs) & val_objs)
    frames = list(split["validation_real_frames"])
    cam = json.load(open(D / "camera.json"))
    est, take_thr = fp.volume_estimator_for(train_objs, cam)
    dets, embs = pickle.load(open(C / f"fold{fold}_dets.pkl" if fold in FOLDS else HOLDOUT_DETS, "rb"))
    embs = {(t, v): embs[(t, v)] for t in MODELS for v in VIEWS}
    return {"fold": fold, "train_objs": train_objs, "val_objs": val_objs, "frames": frames, "dets": dets, "embs": embs,
            "est": est, "take_thr": take_thr, "dens": class_density(meta, set(train_objs)), "text": text, "F": F, "meta": meta,
            "prompts": prompts, "gt": gt_solution(frames)}


def run_fold(ctx, comp, opts):
    dets = ctx["dets"]
    P = comp["material"](ctx)
    p_bulk = comp["form"](ctx)
    ctx["P"], ctx["p_bulk"] = P, p_bulk
    gid = fp.link_series(dets, ctx["embs"], opts["agg"], opts.get("agg_gap", 3)) if opts["agg"] > 0 and dets else None
    ctx["gid"] = gid
    if comp.get("series"):
        P, p_bulk = comp["series"](ctx, P, p_bulk)
        ctx["P"], ctx["p_bulk"] = P, p_bulk
    thr = ctx.get("form_thr", opts.get("form_thr", 0.5))
    forms = ["bulk" if pb >= thr else "solid" for pb in p_bulk]
    mats = [MATS[int(p.argmax())] for p in P]
    vols = comp["volume"](ctx, forms, mats)
    if gid is not None:
        for g in set(gid):
            ii = np.where(gid == g)[0]
            vols[ii] = float(np.exp(np.median(np.log(np.maximum(vols[ii], 1.0)))))
    take, mass = comp["take"](ctx, vols, forms, mats)
    sub = assemble(ctx, p_bulk, forms, mats, vols, mass, take)
    return sub, (evaluate(sub, ctx["gt"]) if "gt" in ctx else None)


def summarize(res, label, out=print):
    folds = sorted(res)
    comps = [res[f] for f in folds]
    mean = {c: float(np.mean([x[c] for x in comps])) for c in ("Score", "D", "M", "V", "S")}
    sd = float(np.std([x["Score"] for x in comps]))
    f23 = float(np.mean([res[f]["Score"] for f in folds if f in (2, 3)])) if any(f in (2, 3) for f in folds) else float("nan")
    f4 = res[4]["Score"] if 4 in res else float("nan")
    for f in folds:
        c = res[f]
        out(f"  fold {f} | Score {c['Score']:6.2f}  D {c['D']:.3f}  M {c['M']:.3f}  V {c['V']:.3f}  S {c['S']:.3f}")
    out(f"  {label}: среднее {mean['Score']:.2f} ± {sd:.2f} | ф2-3 {f23:.2f} | ф4(holdout) {f4:.2f} | D {mean['D']:.3f} M {mean['M']:.3f} V {mean['V']:.3f} S {mean['S']:.3f}")
    return {"mean": mean, "std": sd, "f23": f23, "per_fold": res}


def base_components(opts):
    return {"material": comp_material, "form": comp_form, "volume": comp_volume,
            "take": lambda ctx, v, f, m: comp_take(ctx, v, f, m, opts["take"], opts.get("mass_thr", 500.0))}


def main(argv=None, comp_factory=base_components, label=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--folds", type=int, nargs="+", default=[1, 2, 3])
    ap.add_argument("--prompts", default="rich", choices=list(fp.PROMPT_SETS))
    ap.add_argument("--take", default="hybrid", choices=["hybrid", "volume", "mass"])
    ap.add_argument("--agg", type=float, default=0.90)
    ap.add_argument("--agg-gap", type=int, default=3)
    ap.add_argument("--form-thr", type=float, default=0.5)
    ap.add_argument("--mass-thr", type=float, default=500.0)
    ap.add_argument("--tag", default="base")
    a = ap.parse_args(argv)
    opts = {"take": a.take, "agg": a.agg, "agg_gap": a.agg_gap, "form_thr": a.form_thr, "mass_thr": a.mass_thr, "prompts": a.prompts}
    text = load_text()
    F, meta = fp.train_cache()
    comp = comp_factory(opts)
    res = {}
    for fold in a.folds:
        ctx = fold_context(fold, text, F, meta, a.prompts)
        sub, c = run_fold(ctx, comp, opts)
        (HERE / "cache").mkdir(exist_ok=True)
        sub.to_csv(HERE / "cache" / f"fold{fold}_{a.tag}.csv", index=False)
        res[fold] = c
        print(f"fold {fold}: T={ctx['take_thr']:.0f} мл, w_zs={ctx.get('w_zs')}, детекций {len(ctx['dets'])}, "
              f"серий {len(set(ctx['gid'])) if ctx['gid'] is not None else '-'} | Score {c['Score']:.2f} D {c['D']:.3f} M {c['M']:.3f} V {c['V']:.3f} S {c['S']:.3f}", flush=True)
    summ = summarize(res, label or a.tag)
    json.dump(summ, open(HERE / "cache" / f"report_{a.tag}.json", "w"), indent=1)
    return summ


if __name__ == "__main__":
    main()
