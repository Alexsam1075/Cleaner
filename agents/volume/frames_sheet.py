import json, cv2, numpy as np, sys
root='/home/cleaner/Cleaner/datasets/raw'
lab=json.load(open(f'{root}/labels/real_train.json'))
imgs={i['id']:i['file_name'] for i in lab['images']}
want=sys.argv[1].split(',')
tiles=[]
for a in lab['annotations']:
    fn=imgs[a['image_id']]; fr=fn.split('/')[-2]
    if fr not in want: continue
    img=cv2.imread(f"{root}/{fn}")
    depth=cv2.imread(f"{root}/{fn}".replace('color.png','depth.png'),cv2.IMREAD_UNCHANGED)
    x,y,w,h=[int(v) for v in a['bbox']]; pad=40
    x0,y0,x1,y1=max(0,x-pad),max(0,y-pad),min(img.shape[1],x+w+pad),min(img.shape[0],y+h+pad)
    crop=img[y0:y1,x0:x1].copy()
    mask=np.zeros(depth.shape,np.uint8)
    for poly in a['segmentation']: cv2.fillPoly(mask,[np.asarray(poly,np.float32).reshape(-1,2).round().astype(np.int32)],1)
    cnts,_=cv2.findContours(mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(crop,[c-np.array([x0,y0]) for c in cnts],-1,(0,255,0),1)
    d=depth[y0:y1,x0:x1].astype(np.float32); m=mask[y0:y1,x0:x1]>0
    dv=d[m&(d>0)]; lo,hi=np.percentile(dv,[2,98])
    dn=np.clip((d-lo)/(hi-lo+1e-6),0,1); dimg=cv2.applyColorMap((dn*255).astype(np.uint8),cv2.COLORMAP_JET); dimg[d==0]=0
    cv2.drawContours(dimg,[c-np.array([x0,y0]) for c in cnts],-1,(255,255,255),1)
    s=300/max(crop.shape[:2]); crop=cv2.resize(crop,None,fx=s,fy=s); dimg=cv2.resize(dimg,None,fx=s,fy=s)
    tile=np.zeros((320,620,3),np.uint8); tile[:crop.shape[0],:crop.shape[1]]=crop; tile[:dimg.shape[0],310:310+dimg.shape[1]]=dimg
    at=a['attributes']; cv2.putText(tile,f"{fr} {at['object_id']} {at['volume_ml']}",(2,312),cv2.FONT_HERSHEY_SIMPLEX,0.5,(255,255,255),1)
    tiles.append(tile)
while len(tiles)%2: tiles.append(np.zeros_like(tiles[0]))
cv2.imwrite(sys.argv[2],np.vstack([np.hstack(tiles[i:i+2]) for i in range(0,len(tiles),2)]),[cv2.IMWRITE_JPEG_QUALITY,80])
