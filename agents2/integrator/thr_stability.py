"""Устойчивость порога формы (вариант a): порог по вложенному CV на случайных подвыборках 34 из 40 train-объектов."""
import sys, numpy as np
sys.path.insert(0, "/home/cleaner/Cleaner/agents2/integrator")
import eval_folds as E, pipeline_v2 as P2
text = E.load_text(); F, meta = E.fp.train_cache()
allobjs = sorted({m["oid"] for m in meta if m["src"] == "real_train"})
rs = np.random.default_rng(1); thr = []
for k in range(8):
    sub = sorted(rs.choice(allobjs, 34, replace=False))
    ctx = {"fold": f"bs{k}", "train_objs": sub, "dets": [], "embs": {(t, v): np.zeros((0, 768), np.float32) for t in E.MODELS for v in E.VIEWS}, "prompts": "rich"}
    E.comp_form(ctx)
    t, s = P2.form_thr_cv(ctx); thr.append(t); print(k, "thr", t, "bal.acc", round(s, 3), flush=True)
print("пороги на подвыборках 34/40:", thr)
