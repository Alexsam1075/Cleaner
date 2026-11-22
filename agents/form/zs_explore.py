"""Zero-shot form exploration: text prompts x cached image embeddings (CLIP-L dfn2b, SigLIP)."""
import json, sys
from pathlib import Path
import numpy as np, torch, open_clip
import torch.nn.functional as Fn
C = Path("/home/cleaner/Cleaner/audit/cache"); W = Path("/home/cleaner/Cleaner/agents/form")
F = dict(np.load(C / "features2.npz")); meta = json.load(open(C / "meta.json"))
oid = np.array([m["oid"] for m in meta]); src = np.array([m["src"] for m in meta])
y = np.array([0 if m["form"] == "bulk" else 1 for m in meta]); real = src == "real_train"
DEV = "cuda"
PROMPTS = {
 "A_pile": (["a pile of small loose pieces", "scattered granular material", "a heap of many small items", "loose debris"],
            ["a single solid object", "one rigid item", "a single whole object", "one intact item"]),
 "B_many": (["many small objects", "several items", "multiple pieces", "a group of objects", "a bunch of things"],
            ["one object", "a single item", "one thing", "a single object"]),
 "C_count": (["a photo of several objects on the floor", "a photo of many pieces on the floor", "a photo of a pile on the floor", "a photo of scattered items on the floor"],
             ["a photo of one object on the floor", "a photo of a single item on the floor", "a photo of a single object lying on the floor"]),
 "D_mix": (["a pile of small loose pieces", "many small objects", "several items", "scattered granular material", "a heap of many small items", "multiple pieces", "a bunch of things"],
           ["a single solid object", "one rigid item", "one object", "a single item", "a single whole object"]),
 "E_bulk": (["bulk material", "loose bulk items", "a loose pile", "scattered pieces"], ["a solid object", "a single rigid object", "one piece"]),
}
TEMPL = ["{}", "a photo of {}.", "a close-up photo of {}.", "{} on a tiled floor."]
MODELS = {"clipL": ("ViT-L-14", "dfn2b"), "siglip": ("ViT-B-16-SigLIP", "webli")}
out = {}
with torch.no_grad():
    for tag, (name, pre) in MODELS.items():
        model = open_clip.create_model(name, pretrained=pre, device=DEV).eval(); tok = open_clip.get_tokenizer(name)
        for pname, (pb, ps) in PROMPTS.items():
            T = []
            for plist in (pb, ps):
                e = Fn.normalize(model.encode_text(tok([t.format(p) for p in plist for t in TEMPL]).to(DEV)).float(), dim=-1).mean(0)
                T.append(Fn.normalize(e, dim=0))
            T = torch.stack(T).cpu().numpy()
            out[f"{tag}_{pname}_text"] = T
            for v in ("m10", "m14", "c14", "c20"):
                s = F[f"{tag}_{v}"] @ T.T
                out[f"{tag}_{pname}_{v}"] = s
                d = s[:, 1] - s[:, 0]  # >0 -> solid
                idx = np.where(real)[0]
                # best bias (optimistic, for exploration): threshold on d
                ths = np.unique(d[idx]); best = (0, 0)
                for th in ths:
                    p = (d[idx] > th).astype(int)
                    b = ((p[y[idx]==0]==0).mean() + (p[y[idx]==1]==1).mean())/2
                    if b > best[0]: best = (b, th)
                # AUC
                from sklearn.metrics import roc_auc_score
                auc = roc_auc_score(y[idx], d[idx])
                print(f"{tag:6s} {pname:8s} {v:4s} AUC {auc:.3f} best_bal {best[0]:.3f} @ {best[1]:.4f}", flush=True)
        del model; torch.cuda.empty_cache()
np.savez_compressed(W / "zs_cache.npz", **out)
