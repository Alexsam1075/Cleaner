import numpy as np, torch, open_clip, torch.nn.functional as Fn
from sklearn.metrics import roc_auc_score
from form_cv import load_bank, report
from form_model import head_predict, ZS_SPECS
bank, y, oid, src, meta = load_bank()
real = src == "real_train"; objs = sorted(set(oid[real])); idx = np.where(real)[0]
TEMPL = ["{}", "a photo of {}.", "a close-up photo of {}.", "{} on a tiled floor."]
PAIRS = {
 "identical": (["many identical parts", "a set of identical items", "repeated identical objects", "a stack of identical pieces"], ["a unique single object", "one object", "a single item"]),
 "stack": (["a stack", "stacked items", "a heap", "a bundle of items", "things lying on top of each other"], ["one thing", "a single object", "an individual object"]),
 "crumpledpile": (["a pile of crumpled sheets", "several crumpled wads", "a heap of crumpled material"], ["a box", "a single package", "one rigid object"]),
 "loose": (["loose items that would scatter if pushed", "material that can be swept", "things you pick up by the handful"], ["an object you pick up with one hand", "a single object you can grab", "one rigid thing"]),
}
with torch.no_grad():
    model = open_clip.create_model("ViT-L-14", pretrained="dfn2b", device="cuda").eval(); tok = open_clip.get_tokenizer("ViT-L-14")
    for pname, (pb, ps) in PAIRS.items():
        T = []
        for plist in (pb, ps):
            e = Fn.normalize(model.encode_text(tok([t.format(p) for p in plist for t in TEMPL]).to("cuda")).float(), dim=-1).mean(0); T.append(Fn.normalize(e, dim=0))
        T = torch.stack(T).cpu().numpy(); bank.text[("clipL", pname)] = T
        for v in ("c14", "m10"):
            s = bank.emb[f"clipL_{v}"] @ T.T; d = s[:, 1] - s[:, 0]
            per = {o: float(d[idx][oid[idx] == o].mean()) for o in objs}; rank = sorted(per, key=per.get)
            print(f"{pname:12s} {v} AUC {roc_auc_score(y[idx], d[idx]):.3f} order {' '.join(rank)}")
    del model; torch.cuda.empty_cache()
def looo(X, sw, wd):
    P = np.zeros((len(y), 2))
    for o in objs:
        tr = np.where((oid != o) & (real | (sw > 0)))[0]; te = np.where((oid == o) & real)[0]
        P[te] = head_predict(X, y, oid, src, tr, te, sw, wd)
    return P
base = bank.features("clipL_c14")
Z = bank.zs_features("zs") / 3.0
for sc in (1, 6, 10, 20):
    report(looo(np.concatenate([base, Z * sc], 1), 0.5, 1e-2), y, oid, real, f"clipL_c14 + zs*{sc}")
ZS_SPECS["zsx"] = (("clipL",), ("pile", "many", "count", "mix", "multi", "crumpled", "parts", "trash", "identical", "stack", "crumpledpile", "loose"), ("c14", "m10", "m14"))
report(looo(bank.features("clipL_c14+zsx"), 0.5, 1e-2), y, oid, real, "clipL_c14+zsx")
report(looo(bank.features("zsx"), 0.5, 1e-1), y, oid, real, "zsx")
