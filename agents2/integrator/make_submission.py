"""Сабмит на тест теми же компонентами, что в eval_folds/pipeline_v2, обученными на всех 40 train-объектах.
Детекции/эмбеддинги теста — из cache/test_dets.pkl (test_cache.py). Проверка формата через score._parse,
порядок строк = sample_submission.csv, одна рамка на объект (NMS 0.6, без второй формы).
Запуск: ~/venv/bin/python make_submission.py --variant base --out ~/Cleaner/publish/submission_HHMM_tag.csv"""
import argparse
import json
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import eval_folds as E  # noqa: E402
import pipeline_v2 as P2  # noqa: E402
from eval_folds import D, MATS, MODELS, VIEWS, S, fp  # noqa: E402

ROOT = Path("/home/cleaner/Cleaner")


def test_context(text, F, meta, prompts="rich"):
    dets, embs, frames = pickle.load(open(HERE / "cache" / "test_dets.pkl", "rb"))
    all_objs = sorted({m["oid"] for m in meta if m["src"] == "real_train"})
    assert len(all_objs) == 40
    cam = json.load(open(D / "camera.json"))
    est, take_thr = fp.volume_estimator_for(all_objs, cam)
    embs = {(t, v): embs[(t, v)] for t in MODELS for v in VIEWS}
    return {"fold": "test", "train_objs": all_objs, "frames": frames, "dets": dets, "embs": embs, "est": est, "take_thr": take_thr,
            "dens": E.class_density(meta, set(all_objs)), "text": text, "F": F, "meta": meta, "prompts": prompts}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", default="base")
    ap.add_argument("--external", nargs="*", default=[])
    ap.add_argument("--combo", nargs="*", default=[])
    ap.add_argument("--agg", type=float, default=0.90)
    ap.add_argument("--agg-gap", type=int, default=3)
    ap.add_argument("--out", required=True)
    ap.add_argument("--two-box", type=float, default=0.0, help="вторая рамка с противоположной формой, если её p >= порога (0 = выкл)")
    ap.add_argument("--two-box-mult", type=float, default=0.999)
    ap.add_argument("--fallback-conf", type=float, default=0.0, help="если на кадре нет рамки ≥0.08 — взять лучшую с conf ≥ этого порога")
    a = ap.parse_args()
    text = E.load_text()
    F, meta = fp.train_cache()
    V = P2.make_variants({"a", "b", "c", "d", "e", "f"}, a.external, a.combo)
    comp = V[a.variant][0]
    ctx = test_context(text, F, meta)
    if a.two_box > 0:
        ctx["two_box"], ctx["two_box_mult"] = a.two_box, a.two_box_mult
    if a.fallback_conf > 0:
        ctx["fallback_conf"] = a.fallback_conf
    opts = {"take": "hybrid", "agg": a.agg, "agg_gap": a.agg_gap, "form_thr": 0.5, "mass_thr": 500.0, "prompts": "rich"}
    sub, _ = E.run_fold(ctx, comp, opts)
    sample = pd.read_csv(ROOT / "references/sample_submission.csv")
    assert list(sub.image_id) == list(sample.image_id.astype(str)), "порядок строк"
    n = 0
    for i, c in zip(sub.image_id, sub.prediction):
        objs = S._parse(c, True, i)
        n += len(objs)
        for x in range(len(objs)):
            for y in range(x + 1, len(objs)):
                if a.two_box > 0 and objs[x]["form"] != objs[y]["form"]:
                    continue
                assert S._iou(objs[x]["box"], objs[y]["box"]) < 0.6, (i, "дубль рамки")
    sub.to_csv(a.out, index=False)
    print(f"записано {a.out}: кадров {len(sub)}, объектов {n}, none {(sub.prediction == 'none').sum()}; "
          f"T={ctx['take_thr']:.0f} мл, w_zs={ctx.get('w_zs')}, заметки: {ctx.get('notes')}; формат score._parse OK; two_box={a.two_box}")


if __name__ == "__main__":
    main()
