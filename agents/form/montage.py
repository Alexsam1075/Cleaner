import json, cv2, numpy as np
from pathlib import Path
D = Path("/home/cleaner/Cleaner/datasets/raw")
lab = json.load(open(D/"labels/real_train.json"))
ims = {im["id"]: im for im in lab["images"]}
seen = {}
for a in lab["annotations"]:
    o = a["attributes"]["object_id"]
    seen.setdefault(o, []).append((ims[a["image_id"]], a))
tiles = []
for o in sorted(seen):
    im, a = seen[o][len(seen[o])//2]
    fd = D / Path(im["file_name"]).parent
    rgb = cv2.imread(str(fd/"color.png"))
    x, y, w, h = a["bbox"]; s = max(w, h)*1.4+8; cx, cy = x+w/2, y+h/2
    x1, y1 = int(max(0, cx-s/2)), int(max(0, cy-s/2)); x2, y2 = int(min(rgb.shape[1], cx+s/2)), int(min(rgb.shape[0], cy+s/2))
    c = cv2.resize(rgb[y1:y2, x1:x2], (200, 200))
    cv2.putText(c, f"{o} {a['attributes']['form'][:1]} {a['attributes']['material'][:4]}", (3, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
    tiles.append(c)
rows = [np.concatenate(tiles[i:i+8], 1) for i in range(0, 40, 8)]
cv2.imwrite("/home/cleaner/Cleaner/agents/form/montage.jpg", np.concatenate(rows, 0), [cv2.IMWRITE_JPEG_QUALITY, 80])
