import sys, warnings, json, numpy as np, pandas as pd, math, collections
warnings.filterwarnings('ignore')
sys.path.insert(0,'/home/cleaner/Cleaner/agents/volume')
from exp1 import *
import exp1
f2=pd.read_csv('/home/cleaner/Cleaner/agents/volume/cache/feat2_real.csv')
f2=f2.rename(columns={c:c+'_ml' for c in f2.columns if c.endswith(('colmax','colmean'))})
df=pd.concat([real,f2.drop(columns=['split','frame','object_id'])],axis=1).copy()
C='/home/cleaner/Cleaner/agents/volume/shared_cache/'
meta=json.load(open(C+'meta.json')); z1=np.load(C+'features.npz')
key={(m['frame'],m['oid']):i for i,m in enumerate(meta)}
ridx=np.array([key[(f,o)] for f,o in zip(df.frame,df.object_id)])
photo=z1['photo'][ridx]  # 17 dims: spec, white, Smean, Vmean, Vstd, V99, lap, lap95, contrast, hue hist 8
for i,n in enumerate(['spec','white','Smean','Vmean','Vstd','V99','lap','lap95','contrast']): df['ph_'+n]=photo[:,i]
cols=['columns_ml','parts_ml','hull_ml','footprint_ml','gi_env_ml']
def D(extra=[],inter='gi_env_ml',knots=(50,300),d=df):
    X=design(d,cols,inter=inter,knots=knots)
    if extra: X=np.hstack([X,d[extra].to_numpy(float)])
    return X
p0=looo(D()); report('ref',p0)
for extra in [['ph_Vmean'],['ph_Vmean','ph_Smean'],['ph_spec','ph_white'],['ph_lap'],['ph_contrast'],['hole_inner_frac'],['flying_frac'],['valid_frac'],['mask_solidity'],['ph_Vmean','hole_inner_frac','mask_solidity']]:
    report('+'+','.join(extra), looo(D(extra)))
# weighting schemes
def looo_w(X,wfun,alphas=(0.3,1,3,10,30,100)):
    pred=np.zeros(len(y)); w=wfun(ids)
    for o in sorted(set(ids)):
        tr=ids!=o; Xt,yt,it,wt=X[tr],y[tr],ids[tr],w[tr]
        ut=np.array(sorted(set(it))); np.random.default_rng(0).shuffle(ut); fo={u:k%5 for k,u in enumerate(ut)}; f=np.array([fo[u] for u in it])
        best=None
        for a in alphas:
            p=np.zeros(len(yt))
            for k in range(5): p[f==k]=ridge(Xt[f!=k],yt[f!=k],wt[f!=k],a)(Xt[f==k])
            s=np.sum(wt*np.maximum(0,1-np.abs(p-yt)/math.log(3)))/wt.sum()
            if best is None or s>best[0]: best=(s,a)
        pred[~tr]=ridge(Xt,yt,wt,best[1])(X[~tr])
    return pred
report('w=1 (frame)', looo_w(D(),lambda i: np.ones(len(i))))
report('w=1/sqrt(n)', looo_w(D(),lambda i: np.sqrt(w_of(i))))
# residual per object after ref
r=p0-y; g=pd.DataFrame({'o':ids,'r':r,'vol':df.volume_ml,'form':df.form,'mat':df.material,'Vmean':df.ph_Vmean,'hole':df.hole_inner_frac,'env_over_col':np.log(df.gi_env_ml/df.i0_colmax_ml),'hull_over_env':np.log(df.hull_ml/df.gi_env_ml)}).groupby('o').agg(r=('r','median'),n=('r','size'),vol=('vol','first'),form=('form','first'),mat=('mat','first'),Vmean=('Vmean','mean'),hole=('hole','mean'),eoc=('env_over_col','median'),hoe=('hull_over_env','median'))
pd.set_option('display.width',200); print(g.sort_values('r').round(3).to_string())
print(g[['r','Vmean','hole','eoc','hoe']].corr().round(2))
