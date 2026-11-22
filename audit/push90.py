"""Попытка поднять итог к 90 честными способами. Всё на фолдах 1–3, официальный score.py.
Пороги/настройки подбираются ТОЛЬКО на train-кадрах фолда (детектор фолда прогнан и на них, кэш агента формы).

D: 1) подавление вложенных рамок: рамка, площадь которой ≥ c лежит внутри более уверенной рамки, удаляется
      (ложные срабатывания на фолде 2 — части каркаса b06 внутри основной рамки);
   2) порог уверенности и NMS — подбор на train-кадрах.
Атрибуты (M, V, S) — из итогового конвейера (материал CLIP ⊕ zero-shot, объём volume/, take по калибр. объёму),
сопоставление по IoU > 0.5 c рамками до фильтрации."""
import glob
import itertools
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path("/home/cleaner/Cleaner")
W = ROOT / "agents/form"
sys.path.insert(0, str(W))
sys.path.insert(0, str(ROOT / "references"))
import score as S  # noqa: E402
from common import load_split  # noqa: E402
from form_cv import load_bank  # noqa: E402
from form_model import FormModel  # noqa: E402
from form_pipeline import build_labels, det_bank, touches_border  # noqa: E402

C = ROOT / "audit/cache"
MINE = "fold{}_generic_syn_new_parts_B_new_thr.csv"


def iou(a, b):
    w = min(a[2], b[2]) - max(a[0], b[0])
    h = min(a[3], b[3]) - max(a[1], b[1])
    if w <= 0 or h <= 0:
        return 0.0
    i = w * h
    return i / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - i)


def inside(a, b):
    w = min(a[2], b[2]) - max(a[0], b[0])
    h = min(a[3], b[3]) - max(a[1], b[1])
    if w <= 0 or h <= 0:
        return 0.0
    return w * h / max((a[2] - a[0]) * (a[3] - a[1]), 1e-6)


def build(dets, pb, conf, nms, contain, border=True, gray=0.0):
    per = defaultdict(list)
    for d, p in zip(dets, pb):
        if d["conf"] < conf or (border and touches_border(d["box"])):
            continue
        form = "bulk" if p >= 0.5 else "solid"
        pmax = max(p, 1 - p)
        per[d["frame"]].append({"box": d["box"], "score": d["conf"] * pmax, "form": form, "conf": d["conf"]})
    out = {}
    for f, L in per.items():
        keep = []
        for c in sorted(L, key=lambda x: -x["score"]):
            if any(iou(c["box"], k["box"]) >= nms for k in keep):
                continue
            if contain < 1 and any(inside(c["box"], k["box"]) >= contain for k in keep):
                continue
            keep.append(c)
        if gray > 0:
            extra = []
            for c in keep:
                p_max = c["score"] / max(c["conf"], 1e-9)
                p_other = 1 - p_max
                if p_max <= 0.5 + gray:
                    extra.append({"box": c["box"], "score": c["conf"] * p_other, "form": "solid" if c["form"] == "bulk" else "bulk", "conf": c["conf"]})
            keep = keep + extra
        out[f] = keep
    return out


def D_of(per, frames, sol):
    gt = {str(i): S._parse(c, False, str(i)) for i, c in zip(sol.image_id, sol.objects)}
    pr = {i: [{"box": tuple(o["box"]), "score": o["score"], "form": o["form"], "material": "plastic", "volume": 1, "take": 1} for o in per.get(i, [])] for i in gt}
    return S.components(gt, pr)["D"], pr


def attach(pr, mine_csv):
    m = pd.read_csv(mine_csv)
    md = {i: (S._parse(c, True, i) if str(c) != "none" else []) for i, c in zip(m.image_id.astype(str), m.prediction)}
    out = {}
    for f, L in pr.items():
        res = []
        for o in L:
            best = max(md.get(f, []), key=lambda q: iou(o["box"], q["box"]), default=None)
            if best is not None and iou(o["box"], best["box"]) > 0.5:
                res.append({**o, "material": best["material"], "volume": best["volume"], "take": best["take"]})
            else:
                res.append({**o, "material": "plastic", "volume": 300.0, "take": 1})
        out[f] = res
    return out


def solution_for(frames):
    for p in glob.glob(str(C / "val_labels_*.json")):
        s = S.labels_to_solution(p)
        if set(s.image_id.astype(str)) == set(frames):
            return s
    return S.labels_to_solution(build_labels(set(frames)))


def main():
    bank, y, oid, src, meta = load_bank()
    grid_conf = [0.05, 0.08, 0.12, 0.2, 0.3]
    grid_nms = [0.5, 0.6, 0.7]
    grid_contain = [1.0, 0.95, 0.9, 0.8, 0.7]
    res = {}
    for fold in (1, 2, 3):
        sp = load_split(fold)
        model = FormModel(bank, y, oid, src, grid={"spec": ["clipL_c14+zs"], "sw": [0.5], "wd": [1e-2]}).fit(sorted(sp["train_object_ids"]))
        rows, dbank = det_bank(fold, bank.text)
        pb_all = model.predict_bank(dbank)[:, 0]
        tr_fr, va_fr = sorted(sp["train_real_frames"]), sorted(sp["validation_real_frames"])
        sol_tr = S.labels_to_solution(build_labels(set(tr_fr)))
        sol_va = solution_for(va_fr)
        sel = lambda fr: ([r for r in rows if r["frame"] in set(fr)], pb_all[[i for i, r in enumerate(rows) if r["frame"] in set(fr)]])
        dtr, ptr = sel(tr_fr)
        dva, pva = sel(va_fr)
        # подбор на train-кадрах (на train-объектах классификатор формы переобучен, поэтому важнее всего фильтры рамок)
        best = None
        for c, n, k in itertools.product(grid_conf, grid_nms, grid_contain):
            d, _ = D_of(build(dtr, ptr, c, n, k), tr_fr, sol_tr)
            if best is None or d > best[0] + 1e-9:
                best = (d, c, n, k)
        _, c, n, k = best
        base_d, base_pr = D_of(build(dva, pva, 0.08, 0.6, 1.0), va_fr, sol_va)
        new_d, new_pr = D_of(build(dva, pva, c, n, k), va_fr, sol_va)
        g15_d, g15_pr = D_of(build(dva, pva, 0.08, 0.6, 1.0, gray=0.15), va_fr, sol_va)
        g25_d, g25_pr = D_of(build(dva, pva, 0.08, 0.6, 1.0, gray=0.25), va_fr, sol_va)
        g50_d, g50_pr = D_of(build(dva, pva, 0.08, 0.6, 1.0, gray=0.5), va_fr, sol_va)
        gt = {str(i): S._parse(cc, False, str(i)) for i, cc in zip(sol_va.image_id, sol_va.objects)}
        full = {}
        for name, pr in (("форма E (было)", base_pr), (f"+ фильтры, подобранные на train (conf {c}, nms {n}, вложенность {k})", new_pr),
                         ("+ вторая форма только для неуверенных (p_max ≤ 0.65)", g15_pr),
                         ("+ вторая форма только для неуверенных (p_max ≤ 0.75)", g25_pr),
                         ("+ вторая форма для всех рамок (эксплуатация)", g50_pr)):
            full[name] = S.components(gt, attach(pr, C / MINE.format(fold)))
        res[fold] = full
        print(f"\n== fold {fold}")
        for name, cmp_ in full.items():
            print(f"  {name:62s} Score {cmp_['Score']:6.2f} | D {cmp_['D']:.3f} M {cmp_['M']:.3f} V {cmp_['V']:.3f} S {cmp_['S']:.3f}")
    print("\n== Среднее")
    for name_i in range(5):
        names = [list(res[f].keys())[name_i] for f in res]
        m = {k: np.mean([res[f][names[j]][k] for j, f in enumerate(res)]) for k in ("Score", "D", "M", "V", "S")}
        print(f"  {names[0][:40]:40s} Score {m['Score']:6.2f} | D {m['D']:.3f} M {m['M']:.3f} V {m['V']:.3f} S {m['S']:.3f}")


if __name__ == "__main__":
    main()
