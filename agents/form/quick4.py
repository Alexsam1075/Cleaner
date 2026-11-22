"""Try: (a) per-prompt similarity features (linear head restricted to text span), (b) other classifiers (SVM, kNN, LDA)."""
import numpy as np, torch, open_clip, torch.nn.functional as Fn
from form_cv import load_bank, report
from form_model import head_predict, sample_weights
from common import PROMPT_SETS, TEMPL, W
bank, y, oid, src, meta = load_bank()
real = src == "real_train"; objs = sorted(set(oid[real]))
prompts = sorted({p for pb, ps in PROMPT_SETS.values() for p in pb + ps})
print(len(prompts), "prompts")
with torch.no_grad():
    model = open_clip.create_model("ViT-L-14", pretrained="dfn2b", device="cuda").eval(); tok = open_clip.get_tokenizer("ViT-L-14")
    T = []
    for p in prompts:
        e = Fn.normalize(model.encode_text(tok([t.format(p) for t in TEMPL]).to("cuda")).float(), dim=-1).mean(0); T.append(Fn.normalize(e, dim=0))
    T = torch.stack(T).cpu().numpy()
    del model; torch.cuda.empty_cache()
np.savez(W / "prompt_text.npz", T=T, prompts=np.array(prompts))
def looo(X, sw, wd):
    P = np.zeros((len(y), 2))
    for o in objs:
        tr = np.where((oid != o) & (real | (sw > 0)))[0]; te = np.where((oid == o) & real)[0]
        P[te] = head_predict(X, y, oid, src, tr, te, sw, wd)
    return P
def psims(views):
    return np.concatenate([bank.emb[f"clipL_{v}"] @ T.T for v in views], 1).astype(np.float32) * 10
for views in (("c14",), ("c14", "m10", "m14")):
    for wd in (1e-2, 1e-1, 1.0):
        report(looo(psims(views), 0.5, wd), y, oid, real, f"psims {views} wd={wd}")
    report(looo(np.concatenate([bank.features("clipL_c14"), psims(views)], 1), 0.5, 1e-2), y, oid, real, f"clipL_c14 + psims {views}")
    report(looo(np.concatenate([bank.features("clipL_c14+zs"), psims(views)], 1), 0.5, 1e-2), y, oid, real, f"clipL_c14+zs + psims {views}")
# other classifiers on clipL_c14+zs
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
X = bank.features("clipL_c14+zs")
for name, mk in [("svc_lin C=0.1", lambda: SVC(kernel="linear", C=0.1, probability=True, class_weight="balanced")),
                 ("svc_rbf C=1", lambda: SVC(kernel="rbf", C=1.0, probability=True, class_weight="balanced")),
                 ("lda shrink", lambda: LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto")),
                 ("sk logreg C=0.1", lambda: LogisticRegression(C=0.1, max_iter=2000, class_weight="balanced"))]:
    P = np.zeros((len(y), 2))
    for o in objs:
        tr = np.where(oid != o)[0]; te = np.where((oid == o) & real)[0]
        mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-6
        clf = mk()
        w = sample_weights(tr, oid, src, y, 0.5)
        try: clf.fit((X[tr] - mu) / sd, y[tr], sample_weight=w)
        except TypeError: clf.fit((X[tr] - mu) / sd, y[tr])
        P[te] = clf.predict_proba((X[te] - mu) / sd)
    report(P, y, oid, real, name)
