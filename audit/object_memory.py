"""Память предметов на тесте: находим «тот же предмет» на разных кадрах (глобально по всему набору, не только соседи),
копим информацию по кластеру и пересчитываем предсказания всех его кадров.

Ключ детекции: [CLIP-L c14 ‖ SigLIP c14] (L2) + геометрия (лог след/высота/Д/Ш). Два порога сходства:
near — кадры в пределах разрыва ≤ gap, far — любые кадры (строже, плюс геометрия в пределах tol).
Память кластера: взвешенные по уверенности log-p материала, log-объёмы, голоса формы.
Правила пересчёта (с проверкой согласия):
  объём   — медиана log-объёма кластера;
  take    — по объёму кластера;
  материал — взвешенный log-p кластера, применяется только если доля лидера ≥ gate (иначе кадр остаётся со своим);
  форма    — большинство кластера, переписываются только кадры с p_max < form_gate.
Проверка: фолды 1–3 (кэш детекций full_pipeline), чистота кластеров по GT object_id, официальный score.py."""
import argparse
import glob
import itertools
import json
import math
import pickle
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path("/home/cleaner/Cleaner")
sys.path.insert(0, str(ROOT / "audit"))
import full_pipeline as fp  # noqa: E402
from full_pipeline import C, D, FOLDS, MATS, MODELS, VIEWS, S, _iou, serialize, touches_border_px  # noqa: E402

FORMS = ["bulk", "solid"]


def det_key(dets, embs, w_geo):
    E = np.concatenate([embs[("clipL", "c14")], embs[("siglip", "c14")]], 1)
    E = E / (np.linalg.norm(E, axis=1, keepdims=True) + 1e-8)
    G = []
    for d in dets:
        g = d["new_geo"] if isinstance(d["new_geo"], dict) and "parts_ml" in d["new_geo"] else list(d["new_geo_by_form"].values())[0]
        G.append([math.log(max(g["parts_ml"], 1)), math.log(max(g["footprint_m2"], 1e-5) * 1e4), g["p95_h_m"] * 20])
    G = np.array(G)
    return E, G


def cluster(dets, embs, near=0.88, far=0.93, gap=3, geo_tol=0.35, w_geo=1.0):
    """Жадная агломерация в порядке кадров: детекция присоединяется к лучшему кластеру, если проходит порог."""
    E, G = det_key(dets, embs, w_geo)
    fnum = np.array([int(d["frame"].split("_")[1]) for d in dets])
    order = np.argsort(fnum, kind="stable")
    cid = -np.ones(len(dets), int)
    mem = []  # [sum_emb, n, frames(set), geo_mean]
    for i in order:
        best, bj = -1.0, -1
        for j, (se, n, frames, gm) in enumerate(mem):
            m = se / n
            sim = float(E[i] @ m / (np.linalg.norm(m) + 1e-8))
            is_near = any(abs(fnum[i] - f) <= gap for f in frames)
            thr = near if is_near else far
            geo_ok = np.all(np.abs(G[i] - gm) <= np.array([math.log(1 + geo_tol) * 2, math.log(1 + geo_tol) * 2, 0.06 * 20])) if not is_near else True
            if sim >= thr and geo_ok and sim > best:
                best, bj = sim, j
        if bj >= 0:
            se, n, frames, gm = mem[bj]
            mem[bj] = (se + E[i], n + 1, frames | {fnum[i]}, (gm * n + G[i]) / (n + 1))
            cid[i] = bj
        else:
            mem.append((E[i].copy(), 1, {fnum[i]}, G[i].copy()))
            cid[i] = len(mem) - 1
    return cid


def pool(dets, cid, P, p_bulk, vols, gate=0.6, form_gate=0.7, temp=1.0, use_mat=True, use_form=True, use_vol=True):
    """Возвращает (P_new, p_bulk_new, vols_new)."""
    P2, pb2, v2 = P.copy(), p_bulk.copy(), vols.copy()
    for c in set(cid):
        ii = np.where(cid == c)[0]
        if len(ii) < 2:
            continue
        if use_vol:
            v2[ii] = float(np.exp(np.median(np.log(np.maximum(vols[ii], 1.0)))))
        if use_mat:
            conf = P[ii].max(1)
            w = conf / conf.sum()
            lp = (w[:, None] * np.log(P[ii] + 1e-9) / temp).sum(0)
            pp = np.exp(lp - lp.max())
            pp /= pp.sum()
            share = (w * (P[ii].argmax(1) == pp.argmax())).sum()
            if share >= gate:
                P2[ii] = pp
        if use_form:
            votes = (p_bulk[ii] >= 0.5).mean()
            maj = 1.0 if votes >= 0.5 else 0.0
            for i in ii:
                pm = max(p_bulk[i], 1 - p_bulk[i])
                if pm < form_gate:
                    pb2[i] = 0.5 + (0.3 if maj == 1.0 else -0.3)
    return P2, pb2, v2


def gt_objects():
    lab = json.load(open(D / "labels/real_train.json"))
    ims = {i["id"]: Path(i["file_name"]).parent.name for i in lab["images"]}
    by = defaultdict(list)
    for a in lab["annotations"]:
        x, y, w, h = a["bbox"]
        by[ims[a["image_id"]]].append(((x, y, x + w, y + h), a["attributes"]["object_id"]))
    return by


def purity(dets, cid, gtb):
    oid = []
    for d in dets:
        c = gtb.get(d["frame"], [])
        best = max(c, key=lambda q: _iou(d["box"], q[0]), default=None)
        oid.append(best[1] if best and _iou(d["box"], best[0]) >= 0.5 else None)
    mixed, multi = 0, 0
    for c in set(cid):
        ids = {o for o, k in zip(oid, cid) if k == c and o}
        n = (cid == c).sum()
        if n > 1:
            multi += 1
        if len(ids) > 1:
            mixed += 1
    # recall: how many GT objects' frames ended up in one cluster
    per_obj = defaultdict(list)
    for o, k in zip(oid, cid):
        if o:
            per_obj[o].append(k)
    frag = np.mean([len(set(v)) for v in per_obj.values()]) if per_obj else 0
    return {"clusters": len(set(cid)), "multi": multi, "mixed": mixed, "frag_per_object": round(float(frag), 2)}


def run_fold(fold, enc_text, F, meta, cfgs, args):
    split_path, det_path, note = FOLDS[fold]
    split = json.load(open(split_path))
    train_objs = sorted(split["train_object_ids"])
    frames = list(split["validation_real_frames"])
    cam = json.load(open(D / "camera.json"))
    est, take_thr = fp.volume_estimator_for(train_objs, cam)
    dets, embs = pickle.load(open(C / f"fold{fold}_dets.pkl", "rb"))
    predict_mat, w = fp.material_model(F, meta, train_objs, enc_text, "generic_syn")
    embs_t = {(t, v): embs[(t, v)] for t in MODELS for v in VIEWS}
    P = predict_mat(embs[("clipL", "c14")], fp.zero_shot(embs_t, enc_text, "generic_syn")) if dets else np.zeros((0, 6))
    sys.path.insert(0, str(ROOT / "agents/form"))
    from form_cv import load_bank
    from form_model import FeatureBank, FormModel, GEO_KEYS
    fbank, fy, foid, fsrc, _ = load_bank()
    fm = FormModel(fbank, fy, foid, fsrc, grid={"spec": ["clipL_c14+zs"], "sw": [0.5], "wd": [1e-2]}).fit(train_objs)
    femb = {f"{t}_{v}": embs[(t, v)] for t in MODELS for v in VIEWS}
    femb["geo"] = np.array([[d["geo"][k] for k in GEO_KEYS] for d in dets], np.float32).reshape(len(dets), len(GEO_KEYS))
    p_bulk = fm.predict_bank(FeatureBank(femb, fbank.text))[:, 0] if dets else np.zeros(0)

    def vol_of(d, form, mat):
        geo = d.get("new_geo_by_form", {}).get(form, d["new_geo"])
        return est.estimate(None, None, None, form, mat, geometry=geo)

    sol = None
    for p in glob.glob(str(C / "val_labels_*.json")):
        s_ = S.labels_to_solution(p)
        if set(s_.image_id.astype(str)) == set(frames):
            sol = s_
    gt = {str(i): S._parse(c, False, str(i)) for i, c in zip(sol.image_id, sol.objects)}
    gtb = gt_objects()
    out = {}
    for name, cfg in cfgs.items():
        forms0 = ["bulk" if pb >= 0.5 else "solid" for pb in p_bulk]
        mats0 = [MATS[int(p.argmax())] for p in P]
        vols0 = np.array([vol_of(d, f, m).volume_ml for d, f, m in zip(dets, forms0, mats0)]) if dets else np.zeros(0)
        if cfg is None:
            P2, pb2, v2, cid = P, p_bulk, vols0, None
        else:
            cid = cluster(dets, embs, **{k: v for k, v in cfg.items() if k in ("near", "far", "gap", "geo_tol")})
            if cfg.get("off"):
                cid = np.arange(len(dets))
            P2, pb2, v2 = pool(dets, cid, P, p_bulk, vols0, **{k: v for k, v in cfg.items() if k in ("gate", "form_gate", "temp", "use_mat", "use_form", "use_vol")})
        per = defaultdict(list)
        for d, pb, p, v in zip(dets, pb2, P2, v2):
            if d["score"] < args.conf or touches_border_px(d["box"]):
                continue
            form = "bulk" if pb >= 0.5 else "solid"
            mat = MATS[int(p.argmax())]
            rr = vol_of(d, form, mat)
            vol = float(v) if cfg is not None and cfg.get("use_vol", True) else rr.volume_ml
            per[d["frame"]].append({"box": d["box"], "score": d["score"] * max(pb, 1 - pb), "form": form, "material": mat,
                                    "volume": vol, "mass": vol * rr.density_g_per_ml, "take": int(vol <= take_thr),
                                    "conf": d["score"], "pmax": max(pb, 1 - pb)})
        two = (cfg or {}).get("two", None)
        for f in per:
            keep = []
            for c in sorted(per[f], key=lambda o: -o["score"]):
                if all(_iou(c["box"], k["box"]) < 0.6 for k in keep):
                    keep.append(c)
            if two:
                extra = [dict(c, form="solid" if c["form"] == "bulk" else "bulk", score=c["conf"] * (1 - c["pmax"]))
                         for c in keep if two == "all" or c["pmax"] <= 0.65]
                keep = sorted(keep + extra, key=lambda o: -o["score"])[:30]
            per[f] = keep
        sub = pd.DataFrame({"image_id": frames, "prediction": [serialize(per.get(f, [])) for f in frames]})
        pr = {str(i): S._parse(c, True, str(i)) for i, c in zip(sub.image_id, sub.prediction)}
        comp = S.components(gt, pr)
        pur = purity(dets, cid, gtb) if cid is not None else {}
        out[name] = (comp, pur)
        sub.to_csv(C / f"om_fold{fold}_{name}.csv", index=False)
        print(f"  fold {fold} | {name:34s} Score {comp['Score']:6.2f} | D {comp['D']:.3f} M {comp['M']:.3f} V {comp['V']:.3f} S {comp['S']:.3f} | {pur}", flush=True)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--conf", type=float, default=0.08)
    ap.add_argument("--folds", type=int, nargs="+", default=[1, 2, 3])
    a = ap.parse_args()
    enc = fp.Encoders()
    F, meta = fp.train_cache()
    cfgs = {"base": None}
    cfgs["memory: vol (near .90)"] = dict(near=0.90, far=1.01, gap=3, use_mat=False, use_form=False)
    cfgs["memory + 2form unc"] = dict(near=0.90, far=1.01, gap=3, use_mat=False, use_form=False, two="unc")
    cfgs["memory + 2form all"] = dict(near=0.90, far=1.01, gap=3, use_mat=False, use_form=False, two="all")
    cfgs["memory(mat .75 far .95) + 2form all"] = dict(near=0.88, far=0.95, gap=3, gate=0.75, use_form=False, two="all")
    cfgs["no memory + 2form all"] = dict(near=0.90, far=1.01, gap=3, use_mat=False, use_form=False, use_vol=False, two="all", off=True)
    res = {}
    for fold in a.folds:
        print(f"== fold {fold}", flush=True)
        res[fold] = run_fold(fold, enc.text, F, meta, cfgs, a)
    print("\n== Среднее по фолдам ==")
    for name in cfgs:
        cs = [res[f][name][0] for f in res]
        m = {k: np.mean([c[k] for c in cs]) for k in ("Score", "D", "M", "V", "S")}
        sd = np.std([c["Score"] for c in cs])
        mixed = sum(res[f][name][1].get("mixed", 0) for f in res)
        print(f"  {name:34s} Score {m['Score']:6.2f} ± {sd:.2f} | D {m['D']:.3f} M {m['M']:.3f} V {m['V']:.3f} S {m['S']:.3f} | смешанных кластеров {mixed}")
    json.dump({f: {n: v[0] for n, v in r.items()} for f, r in res.items()}, open(C / "object_memory_report.json", "w"), indent=1, default=str)


if __name__ == "__main__":
    main()
