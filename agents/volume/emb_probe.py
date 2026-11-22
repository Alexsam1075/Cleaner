import sys, json, math, numpy as np, pandas as pd
sys.path.insert(0,'/home/cleaner/Cleaner')
from evaluation.volume_parts_cv import volume_score
C='/home/cleaner/Cleaner/agents/volume/shared_cache/'
meta=json.load(open(C+'meta.json'))
z1=np.load(C+'features.npz'); z2=np.load(C+'features2.npz')
real_idx=np.array([i for i,r in enumerate(meta) if r['src']=='real_train'])
syn_idx=np.array([i for i,r in enumerate(meta) if r['src']!='real_train'])
oid=np.array([r['oid'] for r in meta]); vol=np.array([r['vol'] for r in meta]); form=np.array([r['form'] for r in meta]); mat=np.array([r['mat'] for r in meta])
rows=pd.read_csv('/home/cleaner/Cleaner/reports/volume_parts_gt_rows.csv')
# align rows to meta real order
key={(r.image_id,r.object_id):i for i,r in enumerate(rows.itertuples())}
ridx=np.array([key[(meta[i]['frame'],meta[i]['oid'])] for i in real_idx])
geo_log=np.log(np.maximum(rows.parts_ml.to_numpy()[ridx],1.0))
y=np.log(vol)
def ridge_fit(X,y,w,a):
    mu=np.average(X,0,weights=w); Z=X-mu; ym=np.average(y,weights=w)
    A=Z.T@(Z*w[:,None])+a*np.eye(X.shape[1]); coef=np.linalg.solve(A,Z.T@(w*(y-ym))); return mu,ym,coef
def weights(ids):
    import collections; c=collections.Counter(ids); return np.array([1/c[o] for o in ids])
def looo(X, use_syn=False, alphas=(1,10,100,1000,10000), inner=5):
    Xr=X[real_idx]; yr=y[real_idx]; idr=oid[real_idx]
    uniq=sorted(set(idr)); pred=np.zeros(len(real_idx)); chosen=[]
    for o in uniq:
        tr=idr!=o
        Xt,yt,it=Xr[tr],yr[tr],idr[tr]
        if use_syn:
            s=np.isin(oid[syn_idx],[u for u in uniq if u!=o])
            Xt=np.vstack([Xt,X[syn_idx][s]]); yt=np.concatenate([yt,y[syn_idx][s]]); it=np.concatenate([it,oid[syn_idx][s]])
        wt=weights(it)
        # inner selection of alpha: grouped 5 fold
        ut=np.array(sorted(set(it))); rng=np.random.default_rng(0); rng.shuffle(ut); fo={u:k%inner for k,u in enumerate(ut)}; f=np.array([fo[u] for u in it])
        best=None
        for a in alphas:
            p=np.zeros(len(yt))
            for k in range(inner):
                mu,ym,coef=ridge_fit(Xt[f!=k],yt[f!=k],wt[f!=k],a); p[f==k]=ym+(Xt[f==k]-mu)@coef
            s=np.sum(wt*np.maximum(0,1-np.abs(p-yt)/math.log(3)))/wt.sum()
            if best is None or s>best[0]: best=(s,a)
        mu,ym,coef=ridge_fit(Xt,yt,wt,best[1]); chosen.append(best[1])
        pred[idr==o]=ym+(Xr[idr==o]-mu)@coef
    return pred, chosen
def norm(X): X=X/np.linalg.norm(X,axis=1,keepdims=True); return X
res={}
for name,Xa in [('parts_only',None),('dinov2_b_m',z1['dinov2_b_m']),('dinov2_b_c',z1['dinov2_b_c']),('clip_b_m',z1['clip_b_m']),('convnext_m',z1['convnext_m']),('clipL_c14',z2['clipL_c14']),('clipL_m14',z2['clipL_m14']),('dinoL_m14',z2['dinoL_m14']),('dinoL_c14',z2['dinoL_c14']),('siglip_c14',z2['siglip_c14']),('siglip_m14',z2['siglip_m14'])]:
    cls=np.stack([form=='bulk']+[mat==m for m in ['glass','metal','mineral','paper','plastic','wood']],1).astype(float)
    if Xa is None:
        X=np.zeros((len(meta),0))
    else:
        X=norm(Xa)*10
    G=np.zeros(len(meta)); G[real_idx]=geo_log
    for variant in ['emb','emb+geo']:
        if Xa is None and variant=='emb': continue
        Xv=X if variant=='emb' else np.hstack([X,G[:,None],cls,G[:,None]*cls])
        if variant=='emb+geo' and Xa is None: Xv=np.hstack([G[:,None],cls,G[:,None]*cls])
        use_syn=False
        p,ch=looo(Xv,use_syn)
        v=volume_score(np.exp(p),vol[real_idx])
        print(f'{name:12s} {variant:8s} V={v:.4f} alphas={sorted(set(ch))}',flush=True)
