"""Вероятности материала на тесте (matv2, обучен на 40) — для разбора неуверенных рамок. Только инференс."""
import json, pickle, sys
from pathlib import Path
import numpy as np, pandas as pd
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import eval_folds as E, make_submission as MS
from eval_folds import MATS, fp
import importlib.util
sp = importlib.util.spec_from_file_location("pm", "/home/cleaner/Cleaner/agents2/material/predict_material.py")
pm = importlib.util.module_from_spec(sp); sys.path.insert(0, "/home/cleaner/Cleaner/agents2/material"); sp.loader.exec_module(pm)
text = E.load_text(); F, meta = fp.train_cache()
ctx = MS.test_context(text, F, meta)
P = pm.comp_material(ctx)
P0 = E.comp_material(ctx)
rows = []
for d, p, p0 in zip(ctx["dets"], P, P0):
    if d["score"] < 0.08: continue
    o = np.argsort(-p)
    rows.append(dict(frame=d["frame"], conf=round(d["score"], 2), top=MATS[o[0]], p=round(p[o[0]], 2), second=MATS[o[1]], p2=round(p[o[1]], 2), base=MATS[int(p0.argmax())], pb=round(p0.max(), 2)))
df = pd.DataFrame(rows)
pd.set_option("display.max_rows", 300); pd.set_option("display.width", 200)
print(df.to_string())
df.to_csv(HERE / "cache" / "test_mat_probs.csv", index=False)
