import numpy as np, torch, open_clip, torch.nn.functional as Fn
from sklearn.metrics import roc_auc_score
from form_cv import load_bank
bank, y, oid, src, meta = load_bank()
real = src == "real_train"; idx = np.where(real)[0]; objs = sorted(set(oid[real]))
TEMPL = ["{}", "a photo of {}.", "a close-up photo of {}.", "{} on a tiled floor."]
PAIRS = {
 "matpile": (["a pile of screws", "a pile of stones", "a heap of sawdust", "a pile of wood chips", "a pile of plastic parts", "several crumpled paper balls", "a stack of aluminium profiles", "a bunch of small bottles", "a pile of cables and plastic parts"],
             ["a screw", "a stone", "a wooden block", "a plastic bottle", "a cardboard box", "a crumpled paper ball", "an aluminium bar", "a bottle", "a plastic part"]),
 "several": (["several", "many", "a few", "multiple", "lots of", "a pile of", "a heap of"], ["a single", "one", "a", "an individual", "a lone"]),
 "plural": (["objects", "items", "pieces", "things", "parts"], ["an object", "an item", "a piece", "a thing", "a part"]),
 "paper": (["several crumpled paper balls", "a pile of crumpled paper", "many wads of paper"], ["one crumpled paper ball", "a single wad of paper", "a crumpled sheet of paper"]),
 "profile": (["a stack of aluminium profiles", "several aluminium bars", "a bundle of metal rails"], ["a single aluminium profile", "one metal bar", "a metal bracket"]),
}
with torch.no_grad():
    model = open_clip.create_model("ViT-L-14", pretrained="dfn2b", device="cuda").eval(); tok = open_clip.get_tokenizer("ViT-L-14")
    for pname, (pb, ps) in PAIRS.items():
        T = []
        for plist in (pb, ps):
            e = Fn.normalize(model.encode_text(tok([t.format(p) for p in plist for t in TEMPL]).to("cuda")).float(), dim=-1).mean(0); T.append(Fn.normalize(e, dim=0))
        T = torch.stack(T).cpu().numpy()
        for v in ("c14", "m14"):
            s = bank.emb[f"clipL_{v}"] @ T.T; d = s[:, 1] - s[:, 0]
            per = {o: float(d[idx][oid[idx] == o].mean()) for o in objs}; rank = sorted(per, key=per.get)
            print(f"{pname:10s} {v} AUC {roc_auc_score(y[idx], d[idx]):.3f} order {' '.join(rank)}")
    # per-prompt (not pair): which single prompts separate b07 from solids?
    allp = sorted({p for pb, ps in PAIRS.values() for p in pb + ps})
    E = Fn.normalize(model.encode_text(tok([f"a photo of {p}." for p in allp]).to("cuda")).float(), dim=-1).cpu().numpy()
    s = bank.emb["clipL_c14"] @ E.T
    for j, p in enumerate(allp):
        v = s[:, j]; auc = roc_auc_score(y[idx], -v[idx])
        per = {o: float(v[idx][oid[idx] == o].mean()) for o in objs}
        rank = sorted(per, key=lambda o: -per[o])[:12]
        print(f"  prompt {p:40s} AUC(bulk) {auc:.3f} top: {' '.join(rank)}")
