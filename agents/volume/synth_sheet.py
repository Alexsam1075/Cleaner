import json, cv2, numpy as np
root='/home/cleaner/Cleaner/datasets/raw'
lab=json.load(open(f'{root}/labels/synth_train.json'))
imgs={i['id']:i['file_name'] for i in lab['images']}
tiles=[]
for a in lab['annotations'][:12:2]:
    fn=imgs[a['image_id']]
    img=cv2.imread(f"{root}/{fn}")
    depth=cv2.imread(f"{root}/{fn}".replace('color.png','depth.png'),cv2.IMREAD_UNCHANGED)
    x,y,w,h=[int(v) for v in a['bbox']]
    pad=40
    x0,y0,x1,y1=max(0,x-pad),max(0,y-pad),min(img.shape[1],x+w+pad),min(img.shape[0],y+h+pad)
    crop=img[y0:y1,x0:x1].copy()
    d=depth[y0:y1,x0:x1].astype(np.float32)
    dv=d[d>0]; lo,hi=np.percentile(dv,[2,98])
    dn=np.clip((d-lo)/(hi-lo+1e-6),0,1); dimg=cv2.applyColorMap((dn*255).astype(np.uint8),cv2.COLORMAP_JET); dimg[d==0]=0
    s=250/max(crop.shape[:2]); crop=cv2.resize(crop,None,fx=s,fy=s); dimg=cv2.resize(dimg,None,fx=s,fy=s)
    tile=np.zeros((270,520,3),np.uint8); tile[:crop.shape[0],:crop.shape[1]]=crop; tile[:dimg.shape[0],260:260+dimg.shape[1]]=dimg
    at=a['attributes']; cv2.putText(tile,f"{at['object_id']} {at['volume_ml']} {fn}",(2,262),cv2.FONT_HERSHEY_SIMPLEX,0.45,(255,255,255),1)
    tiles.append(tile)
cv2.imwrite('/home/cleaner/Cleaner/agents/volume/synth_sheet.jpg',np.vstack([np.hstack(tiles[:3]),np.hstack(tiles[3:6])]),[cv2.IMWRITE_JPEG_QUALITY,80])
