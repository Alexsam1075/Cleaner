import sys, warnings, json, numpy as np, pandas as pd, math, collections
warnings.filterwarnings('ignore')
sys.path.insert(0,'/home/cleaner/Cleaner/agents/volume')
from exp1 import *
from sklearn.linear_model import HuberRegressor, QuantileRegressor
from sklearn.decomposition import PCA
f2=pd.read_csv('/home/cleaner/Cleaner/agents/volume/cache/feat2_real.csv')
f2=f2.rename(columns={c:c+'_ml' for c in f2.columns if c.endswith(('colmax','colmean'))})
df=pd.concat([real,f2.drop(columns=['split','frame','object_id'])],axis=1).copy()
f2s=pd.read_csv('/home/cleaner/Cleaner/agents/volume/cache/feat2_synth.csv')
f2s=f2s.rename(columns={c:c+'_ml' for c in f2s.columns if c.endswith(('colmax','colmean'))})
dfs=pd.concat([synth,f2s.drop(columns=['split','frame','object_id'])],axis=1).copy()
C='/home/cleaner/Cleaner/agents/volume/shared_cache/'
meta=json.load(open(C+'meta.json')); z2=np.load(C+'features2.npz')
key={(m['frame'],m['oid']):i for i,m in enumerate(meta)}
ridx=np.array([key[(f,o)] for f,o in zip(df.frame,df.object_id)])
E=z2['clipL_c14'][ridx].astype(np.float64); E/=np.linalg.norm(E,axis=1,keepdims=True)
Es=z2['siglip_c14'][ridx].astype(np.float64); Es/=np.linalg.norm(Es,axis=1,keepdims=True)
cols=['columns_ml','parts_ml','hull_ml','footprint_ml','gi_env_ml']
X=design(df,cols,inter='gi_env_ml',knots=(50,300))
p0=looo(X); report('ridge5 env-inter knots',p0)
# synth
ys_=np.log(dfs.volume_ml.to_numpy()); ids_s=dfs.object_id.to_numpy()
for sw in (0.2,0.5):
    report(f'  +synth w={sw}', looo(X,Xs=design(dfs,cols,inter='gi_env_ml',knots=(50,300)),ys=ys_,ids_s=ids_s,syn_w=sw))

def robust_looo(X, kind='huber', eps=1.35, alphas=(1e-3,1e-2,1e-1,1)):
    pred=np.zeros(len(y)); w=w_of(ids)
    for o in sorted(set(ids)):
        tr=ids!=o
        mu=X[tr].mean(0); sd=X[tr].std(0); sd[sd<1e-9]=1; Z=(X-mu)/sd
        if kind=='huber':
            m=HuberRegressor(epsilon=eps,alpha=1.0,max_iter=500).fit(Z[tr],y[tr],sample_weight=w[tr])
        else:
            m=QuantileRegressor(quantile=0.5,alpha=0.01,solver='highs').fit(Z[tr],y[tr],sample_weight=w[tr])
        pred[~tr]=m.predict(Z[~tr])
    return pred
report('huber eps1.35 a1', robust_looo(X))
report('L1 a0.01', robust_looo(X,'l1'))

def knn_feat(Etr,ytr,idtr,Eq,k=3,tau=0.05,exclude_self=None):
    """object-level kNN prediction of target ytr (per-frame values, averaged per object)."""
    uniq=sorted(set(idtr)); M=np.stack([Etr[idtr==u].mean(0) for u in uniq]); M/=np.linalg.norm(M,axis=1,keepdims=True)
    T=np.array([ytr[idtr==u].mean() for u in uniq])
    S=Eq@M.T
    out=np.zeros(len(Eq))
    for i,s in enumerate(S):
        if exclude_self is not None: s=s.copy(); s[uniq.index(exclude_self[i])]=-9 if exclude_self[i] in uniq else s[0]
        nn=np.argsort(-s)[:k]; ww=np.exp(s[nn]/tau); ww/=ww.sum(); out[i]=np.sum(ww*T[nn])
    return out

def stack_looo(X, E, base_col, k=3, tau=0.05, mode='feature'):
    """nested: kNN residual feature computed leave-one-object-out inside training."""
    pred=np.zeros(len(y)); w=w_of(ids); base=np.log(np.maximum(df[base_col].to_numpy(),1))
    resid=y-base
    for o in sorted(set(ids)):
        tr=ids!=o
        ftr=knn_feat(E[tr],resid[tr],ids[tr],E[tr],k,tau,exclude_self=ids[tr])
        fte=knn_feat(E[tr],resid[tr],ids[tr],E[~tr],k,tau)
        Xtr=np.hstack([X[tr],(base[tr]+ftr)[:,None]]); Xte=np.hstack([X[~tr],(base[~tr]+fte)[:,None]])
        # alpha inner selection
        it=ids[tr]; ut=np.array(sorted(set(it))); np.random.default_rng(0).shuffle(ut); fo={u:kk%5 for kk,u in enumerate(ut)}; f=np.array([fo[u] for u in it])
        best=None
        for a in (0.3,1,3,10,30,100):
            p=np.zeros(tr.sum())
            for kk in range(5): p[f==kk]=ridge(Xtr[f!=kk],y[tr][f!=kk],w[tr][f!=kk],a)(Xtr[f==kk])
            s=np.sum(w[tr]*np.maximum(0,1-np.abs(p-y[tr])/math.log(3)))/w[tr].sum()
            if best is None or s>best[0]: best=(s,a)
        pred[~tr]=ridge(Xtr,y[tr],w[tr],best[1])(Xte)
    return pred
for k in (3,5):
    for tau in (0.03,0.1):
        report(f'stack clip knn-resid k={k} tau={tau}', stack_looo(X,E,'gi_env_ml',k,tau))
report('stack siglip knn-resid k=3 tau=0.05', stack_looo(X,Es,'gi_env_ml',3,0.05))
# PCA emb
for nc in (4,8,16):
    pred=np.zeros(len(y)); w=w_of(ids)
    for o in sorted(set(ids)):
        tr=ids!=o; pca=PCA(nc,random_state=0).fit(E[tr]); Xa=np.hstack([X,pca.transform(E)*3])
        pred[~tr]=ridge(Xa[tr],y[tr],w[tr],3.0)(Xa[~tr])
    report(f'ridge + PCA{nc}(clip) a=3',pred)
