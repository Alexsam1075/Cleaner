import json,sys,cv2,numpy as np
sys.path.insert(0,"/home/cleaner/Cleaner/audit")
from attr_features import polygon_mask
from mat_features2 import crop
from pathlib import Path
D=Path("/home/cleaner/Cleaner/datasets/raw")
lab=json.load(open(D/"labels/real_train.json"))
ims={im["id"]:im for im in lab["images"]}
want=sys.argv[1:]
rows={}
for a in lab["annotations"]:
    o=a["attributes"]["object_id"]
    if o in want:
        rows.setdefault(o,[]).append((ims[a["image_id"]],a))
for o,lst in rows.items():
    tiles=[]
    for im,a in lst[:8]:
        fd=D/Path(im["file_name"]).parent
        rgb=cv2.imread(str(fd/"color.png"))
        m=polygon_mask(a["segmentation"],im["height"],im["width"])
        c,_=crop(rgb,m,a["bbox"],1.5,False,256)
        cv2.putText(c,fd.name+" "+json.load(open(fd/"meta.json"))["lighting"],(3,15),cv2.FONT_HERSHEY_SIMPLEX,0.45,(0,255,0),1)
        tiles.append(c)
    while len(tiles)%4: tiles.append(np.zeros_like(tiles[0]))
    rows_=[np.concatenate(tiles[i:i+4],1) for i in range(0,len(tiles),4)]
    cv2.imwrite(f"viz/{o}.jpg",np.concatenate(rows_,0))
    print(o,a["attributes"]["material"],a["attributes"]["form"],len(lst))
