import json, numpy as np
from common import D, gt_by_frame
gt = gt_by_frame()
n = [len(v) for v in gt.values()]
print("objects per frame:", np.bincount(n))
touch = 0; margins = []
for f, L in gt.items():
    for o in L:
        x1,y1,x2,y2 = o["box"]
        m = min(x1, y1, 1280-x2, 720-y2); margins.append(m)
        if m <= 2: touch += 1; print("touch", f, o["oid"], [int(v) for v in o["box"]])
print("GT touching border (<=2px):", touch, "of", len(margins), "min margin pct:", np.percentile(margins, [0,1,5,10,50]))
syn = json.load(open(D/"labels/synth_train.json"))
W = {im["id"]: (im["width"], im["height"]) for im in syn["images"]}
t = 0; ms=[]
for a in syn["annotations"]:
    x,y,w,h = a["bbox"]; Wd,H = W[a["image_id"]]; m = min(x,y,Wd-x-w,H-y-h); ms.append(m); t += m<=2
print("synth touching:", t, len(ms), np.percentile(ms,[0,1,5,10]))
