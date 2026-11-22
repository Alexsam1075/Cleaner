"""Higher-resolution CLIP-L views for GT annotations: 2x2 quadrant tiles of the c14 crop (448px effective), embeddings averaged,
plus a 1.25x context crop. -> cache/gt_hires.npz (same order as meta.json)."""
import json
from pathlib import Path
import cv2, numpy as np
from common import C, D, W, PROMPT_SETS
from attr_features import polygon_mask, records
from mat_features2 import crop
from encoders import Encoders
import torch.nn.functional as Fn

def quad_views(rgb, mask, bbox, scale, masked):
    big, _ = crop(rgb, mask, bbox, scale, masked, size=448)
    return [big[:224, :224], big[:224, 224:], big[224:, :224], big[224:, 224:]], cv2.resize(big, (224, 224), interpolation=cv2.INTER_AREA)

def main():
    enc = Encoders(tags=("clipL",))
    recs = records()
    tiles = {"q_c14": [], "q_m14": [], "c12": []}
    cache = {}
    for i, (src, im, a) in enumerate(recs):
        fd = D / Path(im["file_name"]).parent
        if fd not in cache:
            cache.clear(); cache[fd] = cv2.cvtColor(cv2.imread(str(fd / "color.png")), cv2.COLOR_BGR2RGB)
        rgb = cache[fd]
        m = polygon_mask(a["segmentation"], im["height"], im["width"])
        q, _ = quad_views(rgb, m, a["bbox"], 1.5, False); tiles["q_c14"].extend(q)
        q, _ = quad_views(rgb, m, a["bbox"], 1.5, True); tiles["q_m14"].extend(q)
        tiles["c12"].append(crop(rgb, m, a["bbox"], 1.25, False)[0])
    out = {}
    for k, arr in tiles.items():
        E = enc.embed("clipL", np.stack(arr))
        if k.startswith("q_"):
            E4 = E.reshape(-1, 4, E.shape[1])
            out[k] = E4.mean(1) / np.linalg.norm(E4.mean(1), axis=1, keepdims=True)
            out[k + "_tiles"] = E4
        else:
            out[k] = E
    np.savez_compressed(W / "cache/gt_hires.npz", **out)
    print({k: v.shape for k, v in out.items()})

if __name__ == "__main__":
    main()
