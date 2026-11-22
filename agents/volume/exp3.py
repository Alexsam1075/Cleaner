import sys, warnings, numpy as np, pandas as pd, math
warnings.filterwarnings('ignore')
sys.path.insert(0,'/home/cleaner/Cleaner/agents/volume')
from exp1 import *
f2=pd.read_csv('/home/cleaner/Cleaner/agents/volume/cache/feat2_real.csv')
assert (f2.frame.values==real.frame.values).all()
f2=f2.rename(columns={c:c+'_ml' for c in f2.columns if c.endswith(('colmax','colmean'))})
df=pd.concat([real,f2.drop(columns=['split','frame','object_id'])],axis=1).copy()
cands=[c for c in f2.columns if c.endswith('_ml')]
res=[]
for c in cands:
    l=np.log(np.maximum(df[c],1)/df.volume_ml)
    dev=l-df.assign(l=l).groupby('object_id').l.transform('median')
    v=volume_score(np.exp(looo(design(df,[c]))),df.volume_ml)
    v2=volume_score(np.exp(looo(design(df,[c],inter=c,knots=(50,300)))),df.volume_ml)
    res.append((c,dev.std(),v,v2))
for r in sorted(res,key=lambda r:-r[3]): print('%-18s within sd=%.3f  V(+cls)=%.4f  V(+cls+inter+knots)=%.4f'%r, flush=True)
base=['columns_ml','parts_ml','hull_ml','footprint_ml']
for extra in [['gi_env_ml'],['gi_env_ml','i0_colmax_ml'],['gi_env_ml','gi_sym_ml'],['gi_env_ml','i0_colmax_ml','gi_sym_ml','i0_core_colmax_ml']]:
    for inter in ['parts_ml','gi_env_ml']:
        print(extra, inter, 'V=%.4f'%volume_score(np.exp(looo(design(df,base+extra,inter=inter,knots=(50,300)))),df.volume_ml), flush=True)
