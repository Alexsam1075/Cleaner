import json, numpy as np
from collections import Counter
from common import W, load_split, box_iou
for fold in (1,2,3):
    rows = json.load(open(W/"cache"/f"fold{fold}_dets.json"))
    sp = load_split(fold); val = set(sp["validation_real_frames"]); tr = set(sp["train_real_frames"])
    for name, frames in (("val", val), ("train", tr)):
        R = [r for r in rows if r["frame"] in frames]
        print(f"fold {fold} {name}: {len(R)} dets; conf hist", np.histogram([r["conf"] for r in R], bins=[0,0.05,0.1,0.2,0.3,0.5,0.7,0.9,1])[0])
        tp = [r for r in R if r["gt_iou"] >= 0.5]
        fp = [r for r in R if r["gt_iou"] < 0.5]
        print("   TP-able", len(tp), "FP", len(fp), "FP conf>0.08:", sum(r["conf"]>0.08 for r in fp), "TP conf>0.08:", sum(r["conf"]>0.08 for r in tp))
        print("   det cls vs gt form (TP):", Counter((r["cls"], r["gt_form"]) for r in tp))
        print("   FP: mask_ok", Counter(r["mask_ok"] for r in fp), "hmax<0.01:", sum(r["geo"]["hmax"]<0.01 for r in fp), "valid<0.3:", sum(r["geo"]["valid"]<0.3 for r in fp),
              "area<400:", sum(r["mask_area"]<400 for r in fp), "gt_iou in (0.1,0.5):", sum(0.1<r["gt_iou"]<0.5 for r in fp))
        print("   TP: hmax<0.01:", sum(r["geo"]["hmax"]<0.01 for r in tp), "valid<0.3:", sum(r["geo"]["valid"]<0.3 for r in tp), "area<400:", sum(r["mask_area"]<400 for r in tp))
        # duplicates: pairs of dets in same frame with IoU>0.5
        byf = {}
        for i, r in enumerate(R): byf.setdefault(r["frame"], []).append(r)
        dup = 0
        for f, L in byf.items():
            for i in range(len(L)):
                for j in range(i+1, len(L)):
                    if box_iou(L[i]["box"], L[j]["box"]) > 0.5: dup += 1
        print("   duplicate pairs IoU>0.5:", dup)
        if name == "val":
            fps = sorted(fp, key=lambda r: -r["conf"])[:12]
            for r in fps: print("     FP", r["frame"], f"conf {r['conf']:.3f} {r['cls']} iou {r['gt_iou']:.2f} area {r['mask_area']} hmax {r['geo']['hmax']:.3f} valid {r['geo']['valid']:.2f} foot {r['geo']['foot']:.1f} rough {r['geo']['rough']:.3f}")
        # missed GT
        from common import gt_by_frame
