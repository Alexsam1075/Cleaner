import sys, warnings, json, numpy as np, pandas as pd, math, collections
warnings.filterwarnings('ignore')
sys.path.insert(0,'/home/cleaner/Cleaner/agents/volume')
from exp1 import *
f2=pd.read_csv('/home/cleaner/Cleaner/agents/volume/cache/feat2_real.csv')
f2=f2.rename(columns={c:c+'_ml' for c in f2.columns if c.endswith(('colmax','colmean'))})
f3=pd.read_csv('/home/cleaner/Cleaner/agents/volume/cache/feat3_real.csv')
df=pd.concat([real,f2.drop(columns=['split','frame','object_id']),f3.drop(columns=['split','frame','object_id'])],axis=1).copy()
for c in ['vh_ml','vhc_ml','vhd_ml','vh15_ml','vhc15_ml']:
    l=np.log(np.maximum(df[c],1)/df.volume_ml)
    dev=l-df.assign(l=l).groupby('object_id').l.transform('median')
    print('%-10s within sd=%.3f between sd=%.3f V(+cls)=%.4f V(+cls+inter+knots)=%.4f'%(c,dev.std(),(l-dev).std(),volume_score(np.exp(looo(design(df,[c]))),df.volume_ml),volume_score(np.exp(looo(design(df,[c],inter=c,knots=(50,300)))),df.volume_ml)),flush=True)
cols=['columns_ml','parts_ml','hull_ml','footprint_ml','gi_env_ml']
report('ref',looo(design(df,cols,inter='gi_env_ml',knots=(50,300))))
for extra in [['vhc_ml'],['vhd_ml'],['vhc_ml','vhd_ml'],['vh_ml','vhc_ml','vhd_ml']]:
    report('ref+'+'+'.join(extra),looo(design(df,cols+extra,inter='gi_env_ml',knots=(50,300))))
report('ref-hull-fp+vhc+vhd',looo(design(df,['columns_ml','parts_ml','gi_env_ml','vhc_ml','vhd_ml'],inter='gi_env_ml',knots=(50,300))))
