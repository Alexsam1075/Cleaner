"""«Живая» модель с памятью предметов: валидация на фолдах 1–3, официальный score.py.

Сценарий (честный поток):
  * кадры валидации фолда идут по порядку съёмки (r_XXXX);
  * память заранее содержит train-объекты фолда (их эмбеддинги + реальные объём/масса/материал) — робот их уже видел;
  * встретив объект, робот ищет ближайшего соседа в памяти по эмбеддингу вырезки (CLIP-L + SigLIP, маска YOLO);
    если сходство ≥ τ — берёт объём/материал/take из памяти, иначе — предсказание замороженной модели;
  * обратная связь приходит ОДИН раз на объект — после первой встречи (весы на захвате + замер/оператор):
    эмбеддинг этого кадра и реальные атрибуты записываются в память. Идентичность объекта системе неизвестна —
    совпадение ищется только по эмбеддингу, поэтому ложные совпадения возможны и учитываются.
  * τ подбирается на train-объектах фолда (frame-LOO поиск соседа среди train-кадров), валидация не используется.
Сравнение: замороженная модель (итоговый конвейер, форма E) vs живая; отдельно кадры «первая встреча» и «повтор».
Папка real/test не используется."""
import glob
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path("/home/cleaner/Cleaner")
sys.path.insert(0, str(ROOT / "references"))
import score as S  # noqa: E402

C = ROOT / "audit/cache"
FOLDS = {1: ROOT / "configs/object_id_split.json",
         2: ROOT / "checkpoints/score83_cv_fold2_noval18/configs/object_id_split.json",
         3: ROOT / "checkpoints/score83_cv_fold3_noval18/configs/object_id_split.json"}
FINAL = "final_fold{}_E_D+border.csv"
L3 = math.log(3)


def l2n(x):
    return x / (np.linalg.norm(x, axis=-1, keepdims=True) + 1e-8)


def emb_pair(a, b):
    return np.concatenate([l2n(a), l2n(b)], -1) / math.sqrt(2)


def solution_for(frames):
    for p in glob.glob(str(C / "val_labels_*.json")):
        s = S.labels_to_solution(p)
        if set(s.image_id.astype(str)) == set(frames):
            return s
    raise FileNotFoundError


def comps(sol, per_frame):
    gt = {str(i): S._parse(c, False, str(i)) for i, c in zip(sol.image_id, sol.objects)}
    pr = {i: per_frame.get(i, []) for i in gt}
    return S.components(gt, pr)


def choose_tau(E, oid, frames, max_false=0.05):
    """τ на train-объектах, два сценария:
    повтор — ближайший сосед среди ДРУГИХ кадров ТОГО ЖЕ объекта (должен пройти τ);
    новый  — ближайший сосед среди кадров ДРУГИХ объектов (должен НЕ пройти τ).
    τ = минимальный порог, при котором ложных совпадений «нового» ≤ max_false. Возвращает τ, recall повтора, ложные."""
    sim = E @ E.T
    same_obj = oid[:, None] == oid[None, :]
    same_frame = frames[:, None] == frames[None, :]
    s_rep = np.where(same_obj & ~same_frame, sim, -1).max(1)
    s_new = np.where(~same_obj, sim, -1).max(1)
    has_rep = s_rep > -1
    best = 1.0
    for t in np.sort(np.unique(np.r_[s_rep, s_new]))[::-1]:
        if (s_new >= t).mean() <= max_false:
            best = t
    return float(best), float((s_rep[has_rep] >= best).mean()), float((s_new >= best).mean())


def main(max_false=0.05, soft=False, quiet=False):
    F2 = dict(np.load(C / "features2.npz"))
    G = dict(np.load(C / "overfit_gpu.npz"))
    meta = json.load(open(C / "meta.json"))
    rows = json.load(open(C / "overfit_pred_rows.json"))
    lab = json.load(open(ROOT / "datasets/raw/labels/real_train.json"))
    attrs = {a["attributes"]["object_id"]: a["attributes"] for a in lab["annotations"]}

    E_gt = emb_pair(F2["clipL_c14"], F2["siglip_c14"])
    E_pred = emb_pair(G["pred_clipL_c14"], G["pred_siglip_c14"])
    m_oid = np.array([m["oid"] for m in meta])
    m_src = np.array([m["src"] for m in meta])
    m_frame = np.array([m["frame"] for m in meta])

    summary = {}
    for fold, split_path in FOLDS.items():
        split = json.load(open(split_path))
        train_objs = set(split["train_object_ids"])
        # память: train-объекты (GT-вырезки train-кадров)
        tr_idx = np.where((m_src == "real_train") & np.isin(m_oid, list(train_objs)))[0]
        tau, nn_acc, false_new = choose_tau(E_gt[tr_idx], m_oid[tr_idx], m_frame[tr_idx], max_false)
        tau_lo, _, _ = choose_tau(E_gt[tr_idx], m_oid[tr_idx], m_frame[tr_idx], 0.30)
        mem_E = list(E_gt[tr_idx])
        mem_A = [attrs[o] for o in m_oid[tr_idx]]
        mem_O = list(m_oid[tr_idx])

        fold_rows = {(r["frame"], tuple(np.round(r["bbox"], 1))): i for i, r in enumerate(rows) if r["fold"] == fold}
        sub = pd.read_csv(C / FINAL.format(fold))
        frames = sorted(sub.image_id.astype(str))
        sol = solution_for(frames)
        gt_all = {str(i): S._parse(c, False, str(i)) for i, c in zip(sol.image_id, sol.objects)}
        pred_frozen = {str(i): S._parse(c, True, str(i)) for i, c in zip(sub.image_id, sub.prediction)}
        gt_boxes = defaultdict(list)
        img_name = {i["id"]: Path(i["file_name"]).parent.name for i in lab["images"]}
        for a in lab["annotations"]:
            gt_boxes[img_name[a["image_id"]]].append((a["bbox"], a["attributes"]["object_id"]))

        pred_live = {}
        seen = set()
        log = []
        for fid in frames:
            preds = [dict(p) for p in pred_frozen[fid]]
            gts = gt_all[fid]
            match = S._match(gts, preds)
            feedback = []
            for j, (g, p) in enumerate(zip(gts, match)):
                x, y, x2, y2 = g["box"]
                cand = [(b, o) for b, o in gt_boxes[fid] if abs(b[0] - x) < 0.6 and abs(b[1] - y) < 0.6]
                oid = cand[0][1]
                key = (fid, tuple(np.round(cand[0][0], 1)))
                ri = fold_rows.get(key)
                first = oid not in seen
                used_memory, correct_id = False, None
                if p is not None and ri is not None and rows[ri]["matched"]:
                    q = E_pred[ri]
                    sims = np.stack(mem_E) @ q
                    k = int(sims.argmax())
                    a = float(np.clip((sims[k] - tau_lo) / max(tau - tau_lo, 1e-6), 0, 1)) if soft else float(sims[k] >= tau)
                    if a > 0:
                        A = mem_A[k]
                        p["volume"] = float(math.exp((1 - a) * math.log(max(p["volume"], 1e-3)) + a * math.log(A["volume_ml"])))
                        if a >= 0.5:
                            p["material"] = A["material"]
                            p["take"] = int(A["mass_g"] <= 500)
                        used_memory, correct_id = True, mem_O[k] == oid
                    if first:
                        feedback.append((q, oid))
                gv = g["volume"]
                v = 0.0 if p is None else max(0.0, 1 - abs(math.log(max(p["volume"], 1e-3) / gv)) / L3)
                log.append({"frame": fid, "oid": oid, "first": first, "mem": used_memory, "correct": correct_id, "V": v})
            for q, oid in feedback:
                mem_E.append(q)
                mem_A.append(attrs[oid])
                mem_O.append(oid)
                seen.add(oid)
            pred_live[fid] = preds

        cf = comps(sol, pred_frozen)
        cl = comps(sol, pred_live)
        L = pd.DataFrame(log)
        # V замороженной модели по тем же объектам для разбивки
        vfz = []
        for fid in frames:
            for g, p in zip(gt_all[fid], S._match(gt_all[fid], [dict(x) for x in pred_frozen[fid]])):
                vfz.append(0.0 if p is None else max(0.0, 1 - abs(math.log(max(p["volume"], 1e-3) / g["volume"])) / L3))
        L["V_frozen"] = vfz
        rep = L[~L["first"]]
        fst = L[L["first"]]
        summary[fold] = {"frozen": cf, "live": cl, "tau": tau, "train_nn_id_acc": nn_acc,
                         "repeat_frames": len(rep), "first_frames": len(fst),
                         "V_repeat_frozen": rep.V_frozen.mean(), "V_repeat_live": rep.V.mean(),
                         "V_first_frozen": fst.V_frozen.mean(), "V_first_live": fst.V.mean(),
                         "mem_used_repeat": rep.mem.mean(), "mem_correct_repeat": rep.loc[rep.mem, "correct"].mean() if rep.mem.any() else float("nan"),
                         "mem_used_first": fst.mem.mean(), "mem_wrong_first": (fst.mem & (fst.correct == False)).sum()}  # noqa: E712
        s = summary[fold]
        if quiet:
            continue
        print(f"\n== fold {fold}  (τ={tau:.3f}; на train: повтор узнаётся {nn_acc:.0%}, новый принят за старый {false_new:.0%}; кадров-объектов: первая встреча {s['first_frames']}, повтор {s['repeat_frames']})")
        for name, c in (("замороженная", cf), ("живая (память)", cl)):
            print(f"  {name:16s} Score {c['Score']:6.2f} | D {c['D']:.3f}  M {c['M']:.3f}  V {c['V']:.3f}  S {c['S']:.3f}")
        print(f"  повторные встречи: V {s['V_repeat_frozen']:.3f} -> {s['V_repeat_live']:.3f}; память сработала в {s['mem_used_repeat']:.0%}, "
              f"верный предмет {s['mem_correct_repeat']:.0%}")
        print(f"  первые встречи:    V {s['V_first_frozen']:.3f} -> {s['V_first_live']:.3f}; ложных совпадений с другим предметом {s['mem_wrong_first']}")

    print(f"\n== Среднее по фолдам (ложных совпадений на train ≤ {max_false:.0%}, {'мягкое смешивание' if soft else 'жёсткий порог'}) ==")
    for name in ("frozen", "live"):
        m = {k: np.mean([summary[f][name][k] for f in FOLDS]) for k in ("Score", "D", "M", "V", "S")}
        print(f"  {('замороженная' if name == 'frozen' else 'живая (память)'):16s} Score {m['Score']:6.2f} | D {m['D']:.3f}  M {m['M']:.3f}  V {m['V']:.3f}  S {m['S']:.3f}")
    vr0 = np.mean([summary[f]["V_repeat_frozen"] for f in FOLDS])
    vr1 = np.mean([summary[f]["V_repeat_live"] for f in FOLDS])
    print(f"  V на повторных встречах: {vr0:.3f} -> {vr1:.3f}")
    json.dump(summary, open(C / f"memory_live_report_{max_false}_{int(soft)}.json", "w"), indent=1, default=float)
    return summary


if __name__ == "__main__":
    main(0.05, False)
    for mf, so in ((0.10, False), (0.15, False), (0.05, True), (0.10, True)):
        main(mf, so, quiet=True)
