import sys, numpy as np, pandas as pd, math, collections
sys.path.insert(0,'/home/cleaner/Cleaner')
from evaluation.volume_parts_cv import volume_score
real=pd.read_csv('/home/cleaner/Cleaner/reports/volume_parts_gt_rows.csv')
y=np.log(real.volume_ml.to_numpy()); ids=real.object_id.to_numpy()
MATS=['glass','metal','mineral','paper','plastic','wood']
def w_of(ids):
    c=collections.Counter(ids); return np.array([1/c[o] for o in ids])
def ridge(X,y,w,a):
    mu=np.average(X,0,weights=w); sd=np.sqrt(np.average((X-mu)**2,0,weights=w)); sd[sd<1e-9]=1
    Z=(X-mu)/sd; ym=np.average(y,weights=w)
    coef=np.linalg.solve(Z.T@(Z*w[:,None])+a*np.eye(Z.shape[1]),Z.T@(w*(y-ym)))
    return lambda Xn: ym+((Xn-mu)/sd)@coef
def looo(X,alphas=(0.3,1,3,10,30,100),inner=5,select=True):
    pred=np.zeros(len(y)); w=w_of(ids)
    for o in sorted(set(ids)):
        tr=ids!=o; Xt,yt,it,wt=X[tr],y[tr],ids[tr],w[tr]
        if select and len(alphas)>1:
            ut=np.array(sorted(set(it))); np.random.default_rng(0).shuffle(ut); fo={u:k%inner for k,u in enumerate(ut)}; f=np.array([fo[u] for u in it])
            best=None
            for a in alphas:
                p=np.zeros(len(yt))
                for k in range(inner): p[f==k]=ridge(Xt[f!=k],yt[f!=k],wt[f!=k],a)(Xt[f==k])
                s=np.sum(wt*np.maximum(0,1-np.abs(p-yt)/math.log(3)))/wt.sum()
                if best is None or s>best[0]: best=(s,a)
            a=best[1]
        else: a=alphas[0]
        pred[~tr]=ridge(Xt,yt,wt,a)(X[~tr])
    return pred
L=lambda c: np.log(np.maximum(real[c].to_numpy(),1.0))
cands=['columns_ml','parts_ml','hull_ml','footprint_ml','old_geom_ml']
form=(real.form=='bulk').to_numpy().astype(float)[:,None]
matoh=np.stack([(real.material==m).to_numpy() for m in MATS],1).astype(float)
fp=np.log(np.maximum(real.fp_area_0.to_numpy(),1e-5)); h=np.log(np.maximum(real.p95_h_m.to_numpy(),1e-3))
for c in cands:
    X=L(c)[:,None]; print(f'{c:13s} single: V={volume_score(np.exp(looo(X)),real.volume_ml):.4f}', f'+form={volume_score(np.exp(looo(np.hstack([X,form]))),real.volume_ml):.4f}', f'+form+mat={volume_score(np.exp(looo(np.hstack([X,form,matoh]))),real.volume_ml):.4f}')
Xall=np.stack([L(c) for c in cands],1)
print('all5', volume_score(np.exp(looo(Xall)),real.volume_ml), '+form', volume_score(np.exp(looo(np.hstack([Xall,form]))),real.volume_ml), '+form+mat', volume_score(np.exp(looo(np.hstack([Xall,form,matoh]))),real.volume_ml))
X2=np.hstack([Xall,form,matoh,fp[:,None],h[:,None],real.valid_frac.to_numpy()[:,None]])
print('all5+form+mat+fp+h+vf', volume_score(np.exp(looo(X2)),real.volume_ml))
knots=np.stack([np.maximum(0,L('columns_ml')-math.log(t)) for t in (50,300)],1)
print('+knots', volume_score(np.exp(looo(np.hstack([X2,knots]))),real.volume_ml))
# geometric mean of candidates
gm=Xall.mean(1)[:,None]; print('geomean', volume_score(np.exp(looo(gm)),real.volume_ml), volume_score(np.exp(looo(np.hstack([gm,form]))),real.volume_ml))
