"""Быстрый аудит без GPU: что дают геометрия, калибровка и порог take при честной валидации leave-object-out по всем 40 предметам."""
import json, math, csv, os
from collections import defaultdict
import cv2, numpy as np
from pathlib import Path
D = Path(os.environ.get('ASTRAMIS_DATA', 'dataset')); CAM = json.load(open(D/'camera.json'))
lab = json.load(open(D/'labels/real_train.json'))
ims = {im['id']: im for im in lab['images']}
dens = {f"{r['form']}_{r['material']}": float(r['density_g_per_ml']) for r in csv.DictReader(open(D/'densities.csv'))}

def mask_of(a, h, w):
    m = np.zeros((h, w), np.uint8)
    for s in a['segmentation']:
        cv2.fillPoly(m, [np.array(s, np.float32).reshape(-1, 2).round().astype(np.int32)], 1)
    return m > 0

def vol(depth, mask, plane, cell=0.004, hthr=0.0, erode=0):
    if erode:
        mask = cv2.erode(mask.astype(np.uint8), np.ones((erode, erode), np.uint8)) > 0
    v, u = np.nonzero(mask & (depth > 0))
    if len(v) < 30: return None, 0.0, 0.0
    z = depth[v, u] / 1000.0
    pts = np.stack([(u-CAM['cx'])/CAM['fx']*z, (v-CAM['cy'])/CAM['fy']*z, z], 1)
    n = np.array(plane[:3]); h = pts @ n + plane[3]
    if np.median(h) < 0: h = -h
    e1 = np.cross(n, [1., 0, 0]); e1 /= np.linalg.norm(e1); e2 = np.cross(n, e1)
    k = np.floor(pts@e1/cell).astype(np.int64)*100000 + np.floor(pts@e2/cell).astype(np.int64)
    o = np.argsort(k); k, hh = k[o], h[o]
    st = np.r_[0, np.nonzero(np.diff(k))[0]+1]
    tops = np.maximum.reduceat(hh, st)
    tops = np.clip(tops - hthr, 0, None)
    return float(tops.sum()*cell*cell*1e6), float(np.percentile(h, 95)), len(v)/mask.sum()

rows = []
for a in lab['annotations']:
    im = ims[a['image_id']]; fd = D/Path(im['file_name']).parent
    depth = cv2.imread(str(fd/'depth.png'), -1); plane = json.load(open(fd/'meta.json'))['floor_plane']
    m = mask_of(a, im['height'], im['width'])
    at = a['attributes']
    r = {'oid': at['object_id'], 'form': at['form'], 'mat': at['material'], 'vol': at['volume_ml'], 'mass': at['mass_g'], 'take': int(at['mass_g'] <= 500),
         'area_px': int(m.sum())}
    for name, kw in {'raw': {}, 'thr2': {'hthr': 0.002}, 'thr3': {'hthr': 0.003}, 'er5': {'erode': 5}, 'er5thr2': {'erode': 5, 'hthr': 0.002}}.items():
        v, h95, valid = vol(depth, m, plane, **kw)
        r[name] = v; r['h95'] = h95; r['valid'] = valid
    rows.append(r)
json.dump(rows, open('audit/geo_rows.json', 'w'))
print(len(rows))
