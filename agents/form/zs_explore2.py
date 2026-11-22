import json, numpy as np, torch, open_clip, torch.nn.functional as Fn
from pathlib import Path
from sklearn.metrics import roc_auc_score
C = Path("/home/cleaner/Cleaner/audit/cache"); W = Path("/home/cleaner/Cleaner/agents/form")
F = dict(np.load(C / "features2.npz")); meta = json.load(open(C / "meta.json"))
oid = np.array([m["oid"] for m in meta]); src = np.array([m["src"] for m in meta])
y = np.array([0 if m["form"] == "bulk" else 1 for m in meta]); real = src == "real_train"; idx = np.where(real)[0]
objs = sorted(set(oid[real]))
PAIRS = {
 "multi": (["multiple separate pieces", "a collection of several loose parts", "items piled on top of each other", "a jumble of parts"],
           ["a single object", "one item", "a single piece"]),
 "crumpled": (["crumpled and scattered material", "a messy heap", "a heap of loose material", "a mess of pieces"], ["a neat single object", "one clean rigid object", "a single intact item"]),
 "countable": (["two or more objects", "three objects", "five objects", "a dozen objects", "a hundred small pieces"], ["exactly one object", "only one object", "just one thing"]),
 "shape": (["an irregular shapeless heap", "a formless pile", "an unstructured mass of material"], ["a well-defined shape", "a geometric object", "a compact rigid shape"]),
 "trash": (["a pile of trash", "scattered garbage", "spilled debris", "loose waste material"], ["a piece of trash", "a discarded object", "a single item of garbage"]),
 "parts": (["spare parts", "components", "fragments", "pieces", "chips", "granules", "scraps"], ["a product", "a tool", "a container", "a box", "a bottle", "a block"]),
}
TEMPL = ["{}", "a photo of {}.", "a close-up photo of {}.", "{} on a tiled floor."]
with torch.no_grad():
    model = open_clip.create_model("ViT-L-14", pretrained="dfn2b", device="cuda").eval(); tok = open_clip.get_tokenizer("ViT-L-14")
    for pname, (pb, ps) in PAIRS.items():
        T = []
        for plist in (pb, ps):
            e = Fn.normalize(model.encode_text(tok([t.format(p) for p in plist for t in TEMPL]).to("cuda")).float(), dim=-1).mean(0); T.append(Fn.normalize(e, dim=0))
        T = torch.stack(T).cpu().numpy()
        for v in ("c14", "m10", "m14"):
            s = F[f"clipL_{v}"] @ T.T; d = s[:, 1] - s[:, 0]
            auc = roc_auc_score(y[idx], d[idx])
            per = {o: float(d[idx][oid[idx] == o].mean()) for o in objs}
            rank = sorted(per, key=per.get)
            print(f"{pname:10s} {v} AUC {auc:.3f}  order: {' '.join(rank)}")
