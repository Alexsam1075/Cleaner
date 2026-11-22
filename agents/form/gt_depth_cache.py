"""Depth features for all GT annotations (real_train + synth_train), same order as audit/cache/meta.json -> cache/gt_depth.npz"""
import json
from pathlib import Path
import cv2, numpy as np
from common import C, D, W
from attr_features import polygon_mask, records
from depth_feats import depth_features, depth_vector, DEPTH_KEYS
meta = json.load(open(C / "meta.json"))
recs = records()
assert len(recs) == len(meta)
out = []
cache = {}
for i, (src, im, a) in enumerate(recs):
    fd = D / Path(im["file_name"]).parent
    if fd not in cache:
        cache.clear()
        cache[fd] = (cv2.imread(str(fd / "depth.png"), cv2.IMREAD_UNCHANGED), json.load(open(fd / "meta.json"))["floor_plane"])
    depth, plane = cache[fd]
    assert meta[i]["frame"] == fd.name
    out.append(depth_vector(depth_features(depth, polygon_mask(a["segmentation"], im["height"], im["width"]), plane)))
    if i % 200 == 0:
        print(i, flush=True)
np.savez_compressed(W / "cache" / "gt_depth.npz", X=np.stack(out), keys=np.array(DEPTH_KEYS))
print("saved")
