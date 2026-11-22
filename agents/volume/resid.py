import sys, numpy as np, pandas as pd
sys.path.insert(0,'/home/cleaner/Cleaner/agents/volume')
from exp1 import *
base=['columns_ml','parts_ml','hull_ml','footprint_ml']
p=looo(design(real,base,inter='parts_ml',knots=(50,300)))
real['lr']=p-y
real['raw_lr']=np.log(np.maximum(real.inp_colmax_ml,1))-y
g=real.groupby('object_id').agg(form=('form','first'),mat=('material','first'),vol=('volume_ml','first'),mass=('mass_g','first'),n=('lr','size'),lr=('lr','median'),lr_sd=('lr','std'),raw=('raw_lr','median'),h=('inp_h_p90_m','median'),hm=('inp_h_mean_m','median'),A=('inp_cells_m2','median'),hole=('hole_frac','mean'),L=('inp_L_m','median'),W=('inp_W_m','median'))
g['Vobj']=np.maximum(0,1-np.abs(g.lr)/np.log(3))
pd.set_option('display.width',250)
print(g.sort_values('vol').round(3).to_string())
print('mean |lr| between', g.lr.abs().mean())
