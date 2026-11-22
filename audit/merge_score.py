"""Итоговый счёт: рамки + форма от агента формы (вариант E: форма из классификатора, score = conf × p(form),
NMS 0.6, отсев рамок у края) + материал / объём `volume/` / take из full_pipeline.py (сопоставление рамок по IoU).
Всё на фолдах 1–3, официальный score.py. Обе части обучены только на train-объектах своего фолда."""
import glob
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path("/home/cleaner/Cleaner")
sys.path.insert(0, str(ROOT / "references"))
import score as S  # noqa: E402

FORM = ROOT / "agents/form/out"
MINE = ROOT / "audit/cache"
VARIANT_MINE = "generic_syn_new_parts_B_new_thr"


def iou(a, b):
    w = min(a[2], b[2]) - max(a[0], b[0])
    h = min(a[3], b[3]) - max(a[1], b[1])
    if w <= 0 or h <= 0:
        return 0.0
    i = w * h
    return i / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - i)


def solution_for(frames):
    for p in glob.glob(str(MINE / "val_labels_*.json")):
        s = S.labels_to_solution(p)
        if set(s.image_id.astype(str)) == set(frames):
            return s
    raise FileNotFoundError("labels for frames")


def merge(form_csv, mine_csv):
    f = pd.read_csv(form_csv)
    m = dict(zip(*[pd.read_csv(mine_csv)[c].astype(str) for c in ("image_id", "prediction")]))
    rows, unmatched = [], 0
    for fid, cell in zip(f.image_id.astype(str), f.prediction.astype(str)):
        if cell.strip().lower() == "none":
            rows.append((fid, "none"))
            continue
        mine = [] if m.get(fid, "none") == "none" else [t.split() for t in m[fid].split(";")]
        out = []
        for tok in (t.split() for t in cell.split(";")):
            box = [float(x) for x in tok[:4]]
            best = max(mine, key=lambda q: iou(box, [float(x) for x in q[:4]]), default=None)
            if best is not None and iou(box, [float(x) for x in best[:4]]) > 0.5:
                mat, vol, mass, take = best[6], best[7], best[8], best[9]
            else:
                unmatched += 1
                mat, vol, mass, take = "plastic", "300", "100", "1"
            out.append(" ".join(tok[:6] + [mat, vol, mass, take]))
        rows.append((fid, ";".join(out)))
    return pd.DataFrame(rows, columns=["image_id", "prediction"]), unmatched


def comps(sol, sub):
    gt = {str(i): S._parse(c, False, str(i)) for i, c in zip(sol.image_id, sol.objects)}
    sd = dict(zip(sub.image_id.astype(str), sub.prediction))
    pr = {i: S._parse(sd.get(i, "none"), True, i) for i in gt}
    c = S.components(gt, pr)
    assert abs(c["Score"] - S.score(sol, sub, "image_id")) < 1e-6
    return c


def main():
    table = {}
    for fold in (1, 2, 3):
        mine_csv = MINE / f"fold{fold}_{VARIANT_MINE}.csv"
        frames = list(pd.read_csv(mine_csv).image_id.astype(str))
        sol = solution_for(frames)
        table[(fold, "до агентов (детектор: рамки+форма)")] = comps(sol, pd.read_csv(mine_csv))
        for var, name in (("E_D+border", "форма E (честно)"), ("E_twoclass", "форма E + обе формы на рамку (эксплуатация метрики)")):
            sub, un = merge(FORM / f"fold{fold}_{var}.csv", mine_csv)
            out = MINE / f"final_fold{fold}_{var}.csv"
            sub.to_csv(out, index=False)
            table[(fold, name)] = comps(sol, sub)
            table[(fold, name)]["unmatched"] = un
    names = list(dict.fromkeys(k[1] for k in table))
    for name in names:
        print(f"\n== {name}")
        for fold in (1, 2, 3):
            c = table[(fold, name)]
            print(f"  fold {fold}: Score {c['Score']:6.2f} | D {c['D']:.3f}  M {c['M']:.3f}  V {c['V']:.3f}  S {c['S']:.3f}"
                  + (f"  (рамок без пары: {c['unmatched']})" if "unmatched" in c else ""))
        mean = {k: np.mean([table[(f, name)][k] for f in (1, 2, 3)]) for k in ("Score", "D", "M", "V", "S")}
        sd = np.std([table[(f, name)]["Score"] for f in (1, 2, 3)])
        print(f"  СРЕДНЕЕ: Score {mean['Score']:6.2f} ± {sd:.2f} | D {mean['D']:.3f}  M {mean['M']:.3f}  V {mean['V']:.3f}  S {mean['S']:.3f}")
    json.dump({f"{k[0]}|{k[1]}": v for k, v in table.items()}, open(MINE / "final_score.json", "w"), indent=1, ensure_ascii=False)


if __name__ == "__main__":
    main()
