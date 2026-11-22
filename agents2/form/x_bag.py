"""Does adding generic 'single soft/packed item' families to the prior fix bag-like solids (s01, s11) without hurting bulk?
Quick fixed-lambda LOOO (head clipL_c14+zs + lam * prior)."""
import numpy as np
import torch
import torch.nn.functional as Fn
import open_clip

from bank import gt_bank
from form_cv import report
from form_model2 import head_predict, Prior, logit, sigmoid
from prompts import FAMILIES, TEMPL

EXTRA = {
    "bag": (["a pile of plastic bags", "several plastic bags", "shredded plastic film", "scattered plastic wrappers", "torn packaging pieces"],
            ["a plastic bag", "a folded plastic bag", "a single packed bag", "a pouch", "a sealed package"]),
    "soft": (["a heap of soft loose material", "scattered rags", "a pile of cloth scraps"],
             ["a single soft object", "a folded cloth", "a cushion", "a wrapped bundle"]),
    "one_vs_pile_generic": (["a pile of them", "a heap of them", "lots of them scattered"], ["just one of them", "a single one", "only one"]),
}
bank, y, oid, src, meta = gt_bank()
real = src == "real_train"
objs = sorted(set(oid[real]))
with torch.no_grad():
    model = open_clip.create_model("ViT-L-14", pretrained="dfn2b", device="cuda").eval()
    tok = open_clip.get_tokenizer("ViT-L-14")
    def enc(plist):
        e = Fn.normalize(model.encode_text(tok([t.format(p) for p in plist for t in TEMPL]).to("cuda")).float(), dim=-1).mean(0)
        return Fn.normalize(e, dim=0).cpu().numpy()
    extra_cols = {}
    for fam, (pb, ps) in EXTRA.items():
        tb, ts = enc(pb), enc(ps)
        extra_cols[fam] = np.stack([bank.emb[f"clipL_{v}"] @ ts - bank.emb[f"clipL_{v}"] @ tb for v in ("c14", "m10", "m14")], 1)
    del model
    torch.cuda.empty_cache()
M0 = bank.zs_matrix()
X = bank.features("clipL_c14+zs")
head = {}
for o in objs:
    tr_all = np.where(oid != o)[0]
    te = np.where((oid == o) & real)[0]
    head[o] = logit(head_predict(X, y, oid, src, tr_all, te, 0.5, 1e-2)[:, 1])


def run(M, lam, label):
    P = np.zeros(len(y))
    for o in objs:
        tr_all = np.where(oid != o)[0]
        tr = np.where((oid != o) & real)[0]
        te = np.where((oid == o) & real)[0]
        pr = Prior().fit(M, tr_all, y[tr], tr)
        P[te] = sigmoid(head[o] + lam * pr.z(M)[te])
    return report(P, y, oid, real, label)


for lam in (4.0, 8.0):
    run(M0, lam, f"lam {lam} base families")
    for fam in EXTRA:
        run(np.concatenate([M0, extra_cols[fam]], 1), lam, f"lam {lam} + {fam}")
    run(np.concatenate([M0] + list(extra_cols.values()), 1), lam, f"lam {lam} + all extra")
    # prior only on families: drop one family at a time (ablation)
    names = list(FAMILIES)
    for i, f in enumerate(names):
        keep = [j for j in range(M0.shape[1]) if j // 3 != i]
        run(M0[:, keep], lam, f"lam {lam} - {f}")
