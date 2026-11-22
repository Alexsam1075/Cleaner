import json, numpy as np, cv2
from collections import Counter
from common import W, D, load_split, box_iou, gt_by_frame
gt = gt_by_frame()
for fold in (2,3):
    rows = json.load(open(W/"cache"/f"fold{fold}_dets.json"))
    sp = load_split(fold); val = set(sp["validation_real_frames"])
    R = [r for r in rows if r["frame"] in val]
    byf = {}
    for r in R: byf.setdefault(r["frame"], []).append(r)
    print(f"== fold {fold}")
    for f, L in sorted(byf.items()):
        for i in range(len(L)):
            for j in range(i+1, len(L)):
                iou = box_iou(L[i]["box"], L[j]["box"])
                if iou > 0.3:
                    a, b = L[i], L[j]
                    print(f"  {f}: {a['cls']} {a['conf']:.2f} (gt_iou {a['gt_iou']:.2f} {a['gt_form']}) <-> {b['cls']} {b['conf']:.2f} (gt_iou {b['gt_iou']:.2f} {b['gt_form']}) IoU {iou:.2f}")
    # FP frames with high conf and iou 0
    for r in sorted(R, key=lambda r: -r["conf"]):
        if r["gt_iou"] < 0.1 and r["conf"] > 0.05:
            print(f"  FP {r['frame']} {r['cls']} {r['conf']:.2f} box {[int(v) for v in r['box']]} gt in frame: {[(o['oid'], [int(v) for v in o['box']]) for o in gt[r['frame']]]}")
# tiles of FP boxes
tiles = []
for fold in (2,3):
    rows = json.load(open(W/"cache"/f"fold{fold}_dets.json"))
    sp = load_split(fold); val = set(sp["validation_real_frames"])
    for r in sorted([r for r in rows if r["frame"] in val and r["gt_iou"] < 0.5 and r["conf"] > 0.05], key=lambda r: -r["conf"])[:16]:
        img = cv2.imread(str(D/"real/train"/r["frame"]/"color.png"))
        x1,y1,x2,y2 = [int(v) for v in r["box"]]
        cv2.rectangle(img, (x1,y1), (x2,y2), (0,0,255), 3)
        for o in gt[r["frame"]]:
            a,b,c,d = [int(v) for v in o["box"]]; cv2.rectangle(img, (a,b), (c,d), (0,255,0), 2)
        s = max(x2-x1, y2-y1)*2+40; cx, cy = (x1+x2)//2, (y1+y2)//2
        X1, Y1 = max(0, cx-s//2), max(0, cy-s//2); X2, Y2 = min(img.shape[1], cx+s//2), min(img.shape[0], cy+s//2)
        t = cv2.resize(img[Y1:Y2, X1:X2], (256, 256))
        cv2.putText(t, f"f{fold} {r['frame']} {r['cls'][0]} {r['conf']:.2f} iou{r['gt_iou']:.2f}", (3, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0,255,255), 1)
        tiles.append(t)
rows_ = [np.concatenate(tiles[i:i+8], 1) for i in range(0, len(tiles)//8*8, 8)]
cv2.imwrite(str(W/"fp_montage.jpg"), np.concatenate(rows_, 0), [cv2.IMWRITE_JPEG_QUALITY, 75])
