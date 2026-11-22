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
# ensemble over feature designs
designs=[design(df,cols,inter='gi_env_ml',knots=(50,300)), design(df,cols,inter='parts_ml',knots=(50,300)), design(df,cols,inter='gi_env_ml',knots=(30,100,300,1000)), design(df,cols+['i0_colmax_ml','gi_sym_ml'],inter='gi_env_ml',knots=(50,300)), design(df,['columns_ml','parts_ml','gi_env_ml'],inter='gi_env_ml',knots=(50,300))]
ps=[looo(D) for D in designs]
for i,p in enumerate(ps): report(f'design{i}',p)
report('ensemble mean', np.mean(ps,0)); report('ensemble median', np.median(ps,0))
# material noise augmentation in training: duplicate rows with wrong material at weight q
def looo_noise(X_fn, q, alphas=(0.3,1,3,10,30,100)):
    pred=np.zeros(len(y)); predn=np.zeros(len(y)); w=w_of(ids); rng=np.random.default_rng(0)
    for o in sorted(set(ids)):
        tr=ids!=o
        dtr=df[tr]; rows=[dtr]; ws=[w[tr]*(1-q)]
        for m in MATS:
            d2=dtr.copy(); d2['material']=m; rows.append(d2); ws.append(w[tr]*q/6)
        Xt=X_fn(pd.concat(rows)); yt=np.tile(y[tr],7); wt=np.concatenate(ws); it=np.tile(ids[tr],7)
        ut=np.array(sorted(set(it))); np.random.default_rng(0).shuffle(ut); fo={u:k%5 for k,u in enumerate(ut)}; f=np.array([fo[u] for u in it])
        best=None
        for a in alphas:
            p=np.zeros(len(yt))
            for k in range(5): p[f==k]=ridge(Xt[f!=k],yt[f!=k],wt[f!=k],a)(Xt[f==k])
            s=np.sum(wt*np.maximum(0,1-np.abs(p-yt)/math.log(3)))/wt.sum()
            if best is None or s>best[0]: best=(s,a)
        mdl=ridge(Xt,yt,wt,best[1])
        pred[~tr]=mdl(X_fn(df[~tr]))
        dte=df[~tr].copy(); dte['material']=[m if rng.random()>=0.08 else str(rng.choice([x for x in MATS if x!=m])) for m in dte.material]
        predn[~tr]=mdl(X_fn(dte))
    return pred,predn
fn=lambda d: design(d,cols,inter='gi_env_ml',knots=(50,300))
for q in (0.0,0.05,0.1,0.2):
    p,pn=looo_noise(fn,q); report(f'noise-aug q={q} clean',p); report(f'noise-aug q={q} noisy8%',pn)
# seen-object scenario: frame-level LOO, kNN k=1 on clip embeddings
C='/home/cleaner/Cleaner/agents/volume/shared_cache/'
meta=json.load(open(C+'meta.json')); z2=np.load(C+'features2.npz')
key={(m['frame'],m['oid']):i for i,m in enumerate(meta)}
ridx=np.array([key[(f,o)] for f,o in zip(df.frame,df.object_id)])
for name in ['clipL_c14','dinoL_c14','siglip_c14']:
    E=z2[name][ridx].astype(np.float64); E/=np.linalg.norm(E,axis=1,keepdims=True)
    S=E@E.T; np.fill_diagonal(S,-9); nn=S.argmax(1)
    print(name,'seen-object frame-LOO kNN1: V=%.4f  id-acc=%.3f'%(volume_score(df.volume_ml.values[nn],df.volume_ml),(ids[nn]==ids).mean()))
