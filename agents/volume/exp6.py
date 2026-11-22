import sys, warnings, json, numpy as np, pandas as pd, math, collections
warnings.filterwarnings('ignore')
sys.path.insert(0,'/home/cleaner/Cleaner/agents/volume')
from exp1 import *
f2=pd.read_csv('/home/cleaner/Cleaner/agents/volume/cache/feat2_real.csv')
f2=f2.rename(columns={c:c+'_ml' for c in f2.columns if c.endswith(('colmax','colmean'))})
df=pd.concat([real,f2.drop(columns=['split','frame','object_id'])],axis=1).copy()
cols=['columns_ml','parts_ml','hull_ml','footprint_ml','gi_env_ml']
X=design(df,cols,inter='gi_env_ml',knots=(50,300))
p0=looo(X); report('ref',p0)
r=p0-y; om=pd.Series(r).groupby(ids).transform('median').values
print('ceiling if per-object bias removed: V=%.4f'%np.mean(np.maximum(0,1-np.abs(r-om)/math.log(3))))
print('ceiling if within-object noise removed: V=%.4f'%np.mean(np.maximum(0,1-np.abs(om)/math.log(3))))
# IRLS truncated-L1 (metric) refinement, nested
def irls_looo(X, alphas=(1,3,10,30), iters=8, floor=0.05):
    pred=np.zeros(len(y)); w0=w_of(ids)
    def fit(Xt,yt,wt,a):
        mdl=ridge(Xt,yt,wt,a); 
        for _ in range(iters):
            res=np.abs(mdl(Xt)-yt)
            wi=np.where(res<math.log(3), 1/np.maximum(res,floor), 1/math.log(3)*0.1)
            mdl=ridge(Xt,yt,wt*wi,a)
        return mdl
    for o in sorted(set(ids)):
        tr=ids!=o; Xt,yt,it,wt=X[tr],y[tr],ids[tr],w0[tr]
        ut=np.array(sorted(set(it))); np.random.default_rng(0).shuffle(ut); fo={u:k%5 for k,u in enumerate(ut)}; f=np.array([fo[u] for u in it])
        best=None
        for a in alphas:
            p=np.zeros(len(yt))
            for k in range(5): p[f==k]=fit(Xt[f!=k],yt[f!=k],wt[f!=k],a)(Xt[f==k])
            s=np.sum(wt*np.maximum(0,1-np.abs(p-yt)/math.log(3)))/wt.sum()
            if best is None or s>best[0]: best=(s,a)
        pred[~tr]=fit(Xt,yt,wt,best[1])(X[~tr])
    return pred
for fl in (0.05,0.15,0.3):
    report(f'IRLS metric floor={fl}', irls_looo(X,floor=fl))
# retrieval analysis
C='/home/cleaner/Cleaner/agents/volume/shared_cache/'
meta=json.load(open(C+'meta.json')); z2=np.load(C+'features2.npz')
key={(m['frame'],m['oid']):i for i,m in enumerate(meta)}
ridx=np.array([key[(f,o)] for f,o in zip(df.frame,df.object_id)])
for name in ['clipL_c14','siglip_c14','dinoL_c14','clipL_m14']:
    E=z2[name][ridx].astype(np.float64); E/=np.linalg.norm(E,axis=1,keepdims=True)
    S=E@E.T; np.fill_diagonal(S,-9)
    same=(ids[:,None]==ids[None,:])
    best_same=np.where(same,S,-9).max(1); best_diff=np.where(~same,S,-9).max(1)
    print(name,'same-obj NN sim quantiles',np.round(np.quantile(best_same[np.isfinite(best_same)&(best_same>-9)],[0.05,0.25,0.5]),3),' diff-obj NN sim quantiles',np.round(np.quantile(best_diff,[0.5,0.9,0.99,1.0]),3))
