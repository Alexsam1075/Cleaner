"""Двухэтапная система «похожих деталей».
Этап 1 — первичная покадровая оценка (детектор, форма, материал, объём) — берётся из кэша конвейера.
Этап 2 — вторичное обобщение:
  (a) поиск похожих деталей: две детекции считаются одной деталью, если
      геометрия совпадает (длина, ширина, высота, след на полу — каждая в пределах ±tol_geo) И внешний вид похож
      (косинус CLIP-L+SigLIP ≥ thr); между кадрами с РАЗНЫМ освещением порог по внешнему виду ослаблен (thr_cross),
      т.к. освещение меняет эмбеддинг, а геометрию — нет. Объединение транзитивное (union-find).
  (b) согласование внутри группы:
      материал — взвешенное голосование: вес = уверенность × качество освещения (bright 1.0, side 0.7, dim 0.5);
                 применяется, если доля победителя ≥ gate, иначе кадры остаются со своими ответами;
      форма    — то же голосование;
      объём/масса — если все значения группы в пределах ±10% от медианы, оставляем; иначе заменяем на медиану
                 (с сохранением знака отклонения до 10%: v' = median·clip(v/median, 0.9, 1.1));
      take     — по согласованному объёму;
      размеры рамок не меняем (разный ракурс — разная рамка).
Проверка: фолды 1–3, score.py, чистота групп по GT object_id, согласованность материала внутри GT-объектов
при разных освещениях до и после."""
import argparse
import glob
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

LIGHT_W = {"bright": 1.0, "side": 0.7, "dim": 0.5}


def lighting_of(frame, split="train"):
    p = D / "real" / split / frame / "meta.json"
    if not p.exists():
        p = D / "real" / ("test" if split == "train" else "train") / frame / "meta.json"
    return json.load(open(p)).get("lighting", "bright")


def geo_vec(d, form):
    g = d.get("new_geo_by_form", {}).get(form, d["new_geo"])
    L, W_ = d["geo"]["ext1"], d["geo"]["ext2"]
    return np.array([max(L, 1e-3), max(W_, 1e-3), max(g["p95_h_m"], 3e-3), max(g["footprint_m2"], 1e-5)])


def similar_geo(a, b, tol, mode="all"):
    r = a / b
    if mode == "height":
        r = r[2:3]
    return bool(np.all((r >= 1 - tol) & (r <= 1 / (1 - tol))))


class UF:
    def __init__(self, n):
        self.p = list(range(n))

    def f(self, x):
        while self.p[x] != x:
            self.p[x] = self.p[self.p[x]]
            x = self.p[x]
        return x

    def u(self, a, b):
        self.p[self.f(a)] = self.f(b)


def find_similar(dets, embs, forms, lights, thr=0.90, thr_cross=0.80, tol_geo=0.10, gap=None, geo_mode="all"):
    E = np.concatenate([embs[("clipL", "c14")], embs[("siglip", "c14")]], 1)
    E = E / (np.linalg.norm(E, axis=1, keepdims=True) + 1e-8)
    G = np.stack([geo_vec(d, f) for d, f in zip(dets, forms)])
    fnum = np.array([int(d["frame"].split("_")[1]) for d in dets])
    n = len(dets)
    uf = UF(n)
    sim = E @ E.T
    for i in range(n):
        for j in range(i + 1, n):
            if fnum[i] == fnum[j]:
                continue
            if gap is not None and abs(fnum[i] - fnum[j]) > gap:
                continue
            t = thr if lights[i] == lights[j] else thr_cross
            if sim[i, j] >= t and similar_geo(G[i], G[j], tol_geo, geo_mode):
                uf.u(i, j)
    roots = [uf.f(i) for i in range(n)]
    remap = {r: k for k, r in enumerate(sorted(set(roots)))}
    return np.array([remap[r] for r in roots])


def harmonize(gid, P, p_bulk, vols, confs, lights, gate=0.6, tol=0.10, do_mat=True, do_form=True, do_vol=True):
    P2, pb2, v2 = P.copy(), p_bulk.copy(), vols.copy()
    changed = Counter()
    for g in set(gid):
        ii = np.where(gid == g)[0]
        if len(ii) < 2:
            continue
        w = np.array([confs[i] * LIGHT_W.get(lights[i], 0.7) for i in ii])
        w = w / w.sum()
        if do_mat:
            votes = np.zeros(6)
            for k, i in enumerate(ii):
                votes[int(P[i].argmax())] += w[k] * P[i].max()
            votes /= votes.sum()
            win = int(votes.argmax())
            if votes[win] >= gate:
                for i in ii:
                    if int(P[i].argmax()) != win:
                        changed["material"] += 1
                    lp = np.log(P[i] + 1e-9)
                    lp[win] = max(lp[win], np.log(0.6))
                    pp = np.exp(lp - lp.max())
                    P2[i] = pp / pp.sum()
                    P2[i][win] = max(P2[i][win], P2[i].max() + 1e-6)
        if do_form:
            vb = sum(w[k] for k, i in enumerate(ii) if p_bulk[i] >= 0.5)
            if vb >= gate or vb <= 1 - gate:
                maj_bulk = vb >= gate
                for i in ii:
                    if (p_bulk[i] >= 0.5) != maj_bulk:
                        changed["form"] += 1
                        pb2[i] = 0.6 if maj_bulk else 0.4
        if do_vol:
            med = float(np.exp(np.median(np.log(np.maximum(vols[ii], 1.0)))))
            ratio = vols[ii] / med
            if np.all((ratio >= 1 - tol) & (ratio <= 1 + tol)):
                pass
            else:
                for k, i in enumerate(ii):
                    v2[i] = med * float(np.clip(ratio[k], 1 - tol, 1 + tol))
                    changed["volume"] += 1
    return P2, pb2, v2, changed


def consistency(dets, gtb, mats, lights):
    """Доля GT-объектов, у которых на всех кадрах один материал; отдельно для объектов с ≥2 освещениями."""
    by = defaultdict(list)
    for d, m, l in zip(dets, mats, lights):
        c = gtb.get(d["frame"], [])
        best = max(c, key=lambda q: _iou(d["box"], q[0]), default=None)
        if best and _iou(d["box"], best[0]) >= 0.5:
            by[best[1]].append((m, l))
    allc = [len({m for m, _ in v}) == 1 for v in by.values() if len(v) > 1]
    multi = [len({m for m, _ in v}) == 1 for v in by.values() if len(v) > 1 and len({l for _, l in v}) > 1]
    return (float(np.mean(allc)) if allc else 1.0, len(allc)), (float(np.mean(multi)) if multi else 1.0, len(multi))


def gt_objects():
    lab = json.load(open(D / "labels/real_train.json"))
    ims = {i["id"]: Path(i["file_name"]).parent.name for i in lab["images"]}
    by = defaultdict(list)
    for a in lab["annotations"]:
        x, y, w, h = a["bbox"]
        by[ims[a["image_id"]]].append(((x, y, x + w, y + h), a["attributes"]["object_id"]))
    return by


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--two-form", action="store_true")
    a = ap.parse_args()
    enc = fp.Encoders()
    F, meta = fp.train_cache()
    gtb = gt_objects()
    cfgs = {
        "этап 1 (как есть)": None,
        "G: гео 20%, вид .88/.78, gate .6": dict(thr=0.88, thr_cross=0.78, tol_geo=0.20, gate=0.6),
        "H: гео 30%, вид .88/.78, gate .6": dict(thr=0.88, thr_cross=0.78, tol_geo=0.30, gate=0.6),
        "I: гео 30%, вид .85/.72, gate .6": dict(thr=0.85, thr_cross=0.72, tol_geo=0.30, gate=0.6),
        "J: только высота 30% + вид .85/.72": dict(thr=0.85, thr_cross=0.72, tol_geo=0.30, gate=0.6, geo_mode="height"),
        "K: I, gate .5": dict(thr=0.85, thr_cross=0.72, tol_geo=0.30, gate=0.5),
        "L: I, без материала (объём/форма)": dict(thr=0.85, thr_cross=0.72, tol_geo=0.30, do_mat=False),
    }
    res = defaultdict(list)
    cons = defaultdict(list)
    for fold in (1, 2, 3):
        split = json.load(open(FOLDS[fold][0]))
        train_objs = sorted(split["train_object_ids"])
        frames = list(split["validation_real_frames"])
        cam = json.load(open(D / "camera.json"))
        est, take_thr = fp.volume_estimator_for(train_objs, cam)
        dets, embs = pickle.load(open(C / f"fold{fold}_dets.pkl", "rb"))
        predict_mat, _ = fp.material_model(F, meta, train_objs, enc.text, "generic_syn")
        embs_t = {(t, v): embs[(t, v)] for t in MODELS for v in VIEWS}
        P0 = predict_mat(embs[("clipL", "c14")], fp.zero_shot(embs_t, enc.text, "generic_syn"))
        sys.path.insert(0, str(ROOT / "agents/form"))
        from form_cv import load_bank
        from form_model import FeatureBank, FormModel, GEO_KEYS
        fbank, fy, foid, fsrc, _ = load_bank()
        fm = FormModel(fbank, fy, foid, fsrc, grid={"spec": ["clipL_c14+zs"], "sw": [0.5], "wd": [1e-2]}).fit(train_objs)
        femb = {f"{t}_{v}": embs[(t, v)] for t in MODELS for v in VIEWS}
        femb["geo"] = np.array([[d["geo"][k] for k in GEO_KEYS] for d in dets], np.float32).reshape(len(dets), len(GEO_KEYS))
        pb0 = fm.predict_bank(FeatureBank(femb, fbank.text))[:, 0]
        lights = [lighting_of(d["frame"]) for d in dets]
        confs = np.array([d["score"] for d in dets])
        sol = next(S.labels_to_solution(p) for p in glob.glob(str(C / "val_labels_*.json")) if set(S.labels_to_solution(p).image_id.astype(str)) == set(frames))
        gt = {str(i): S._parse(c, False, str(i)) for i, c in zip(sol.image_id, sol.objects)}
        print(f"== fold {fold}", flush=True)
        for name, cfg in cfgs.items():
            forms0 = ["bulk" if pb >= 0.5 else "solid" for pb in pb0]
            mats0 = [MATS[int(p.argmax())] for p in P0]
            vol0 = np.array([est.estimate(None, None, None, f, m, geometry=d.get("new_geo_by_form", {}).get(f, d["new_geo"])).volume_ml for d, f, m in zip(dets, forms0, mats0)])
            if cfg is None:
                P, pb, vols, gid, changed = P0, pb0, vol0, np.arange(len(dets)), Counter()
            else:
                gid = find_similar(dets, embs, forms0, lights, **{k: v for k, v in cfg.items() if k in ("thr", "thr_cross", "tol_geo", "gap", "geo_mode")})
                P, pb, vols, changed = harmonize(gid, P0, pb0, vol0, confs, lights, **{k: v for k, v in cfg.items() if k in ("gate", "do_mat", "do_form", "do_vol")})
            forms = ["bulk" if x >= 0.5 else "solid" for x in pb]
            mats = [MATS[int(p.argmax())] for p in P]
            per = defaultdict(list)
            for d, x, f, m, v in zip(dets, pb, forms, mats, vols):
                if d["score"] < 0.08 or touches_border_px(d["box"]):
                    continue
                dn = est.estimate(None, None, None, f, m, geometry=d.get("new_geo_by_form", {}).get(f, d["new_geo"])).density_g_per_ml
                pm = max(x, 1 - x)
                per[d["frame"]].append({"box": d["box"], "score": d["score"] * pm, "form": f, "material": m, "volume": float(v), "mass": float(v * dn),
                                        "take": int(v <= take_thr), "conf": d["score"], "pmax": pm})
            for fr in per:
                keep = []
                for c in sorted(per[fr], key=lambda o: -o["score"]):
                    if all(_iou(c["box"], k["box"]) < 0.6 for k in keep):
                        keep.append(c)
                if a.two_form:
                    keep = sorted(keep + [dict(c, form="solid" if c["form"] == "bulk" else "bulk", score=c["conf"] * (1 - c["pmax"])) for c in keep], key=lambda o: -o["score"])[:30]
                per[fr] = keep
            sub = pd.DataFrame({"image_id": frames, "prediction": [serialize(per.get(f_, [])) for f_ in frames]})
            pr = {str(i): S._parse(c, True, str(i)) for i, c in zip(sub.image_id, sub.prediction)}
            comp = S.components(gt, pr)
            res[name].append(comp)
            (c_all, n_all), (c_ml, n_ml) = consistency(dets, gtb, mats, lights)
            cons[name].append((c_all, c_ml))
            # чистота групп
            mixed = 0
            for g in set(gid):
                ids = set()
                for i in np.where(gid == g)[0]:
                    cnd = gtb.get(dets[i]["frame"], [])
                    best = max(cnd, key=lambda q: _iou(dets[i]["box"], q[0]), default=None)
                    if best and _iou(dets[i]["box"], best[0]) >= 0.5:
                        ids.add(best[1])
                mixed += len(ids) > 1
            print(f"  {name:34s} Score {comp['Score']:6.2f} | D {comp['D']:.3f} M {comp['M']:.3f} V {comp['V']:.3f} S {comp['S']:.3f} | групп {len(set(gid))}, смешанных {mixed} | "
                  f"согласован материал: все объекты {c_all:.0%} (n={n_all}), с разным светом {c_ml:.0%} (n={n_ml}) | изменено {dict(changed)}", flush=True)
            sub.to_csv(C / f"ts2_fold{fold}_{name.split(':')[0].replace(' ', '_')}{'_2f' if a.two_form else ''}.csv", index=False)
    print("\n== Среднее по фолдам ==")
    for name, cs in res.items():
        m = {x: np.mean([c[x] for c in cs]) for x in ("Score", "D", "M", "V", "S")}
        ca = np.mean([c[0] for c in cons[name]])
        cm = np.mean([c[1] for c in cons[name]])
        print(f"  {name:34s} Score {m['Score']:6.2f} ± {np.std([c['Score'] for c in cs]):.2f} | D {m['D']:.3f} M {m['M']:.3f} V {m['V']:.3f} S {m['S']:.3f} | согласованность материала {ca:.0%} / при разном свете {cm:.0%}")


if __name__ == "__main__":
    main()
