import json, cv2, numpy as np, sys
from pathlib import Path
D = Path("/home/cleaner/Cleaner/datasets/raw")
lab = json.load(open(D/"labels/real_train.json"))
ims = {im["id"]: im for im in lab["images"]}
seen = {}
for a in lab["annotations"]:
    seen.setdefault(a["attributes"]["object_id"], []).append((ims[a["image_id"]], a))
rows = []
for o in sys.argv[1:]:
    tiles = []
    L = seen[o]
    for im, a in [L[i] for i in np.linspace(0, len(L)-1, min(6, len(L))).astype(int)]:
        fd = D / Path(im["file_name"]).parent
        rgb = cv2.imread(str(fd/"color.png"))
        x, y, w, h = a["bbox"]; s = max(w, h)*1.3+8; cx, cy = x+w/2, y+h/2
        x1, y1 = int(max(0, cx-s/2)), int(max(0, cy-s/2)); x2, y2 = int(min(rgb.shape[1], cx+s/2)), int(min(rgb.shape[0], cy+s/2))
        c = cv2.resize(rgb[y1:y2, x1:x2], (220, 220))
        cv2.putText(c, f"{o} {fd.name}", (3, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        tiles.append(c)
    while len(tiles) < 6: tiles.append(np.zeros((220,220,3), np.uint8))
    rows.append(np.concatenate(tiles, 1))
cv2.imwrite("/home/cleaner/Cleaner/agents/form/montage2.jpg", np.concatenate(rows, 0), [cv2.IMWRITE_JPEG_QUALITY, 75])
