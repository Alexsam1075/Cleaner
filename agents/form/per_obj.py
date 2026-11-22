import numpy as np, json
from form_cv import load_bank
from form_model import head_predict
from common import W, load_split
bank, y, oid, src, meta = load_bank()
real = src == "real_train"; objs = sorted(set(oid[real]))
X = bank.features("clipL_c14+zs")
P = np.zeros((len(y), 2))
for o in objs:
    tr = np.where((oid != o))[0]; te = np.where((oid == o) & real)[0]
    P[te] = head_predict(X, y, oid, src, tr, te, 0.5, 1e-2)
print("LOOO p_bulk per object (mean, min, max):")
for o in objs:
    te = np.where((oid == o) & real)[0]
    print(f"  {o} n={len(te):2d} p_bulk mean {P[te,0].mean():.2f} min {P[te,0].min():.2f} max {P[te,0].max():.2f}")
# fold val-frames classifier errors per object
for fold in (2, 3):
    rows = json.load(open(W/"cache"/f"fold{fold}_dets.json"))
    sp = load_split(fold); val = set(sp["validation_real_frames"])
    from form_pipeline import det_bank
    from form_model import FormModel
    m = FormModel(bank, y, oid, src, grid={"spec": ["clipL_c14+zs"], "sw": [0.5], "wd": [1e-2]}).fit(sp["train_object_ids"])
    rows, db = det_bank(fold, bank.text); Pd = m.predict_bank(db)
    err = {}
    for r, p in zip(rows, Pd):
        if r["frame"] in val and r["gt_iou"] >= 0.5 and r["conf"] >= 0.08:
            err.setdefault(r["gt_oid"], []).append(round(float(p[0]), 2))
    print(f"fold {fold} val p_bulk per TP det by object:", {k: v for k, v in sorted(err.items())})
