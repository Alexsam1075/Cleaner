import sys, numpy as np, pandas as pd, math
sys.path.insert(0,'/home/cleaner/Cleaner')
from evaluation.volume_parts_cv import volume_score
real=pd.read_csv('/home/cleaner/Cleaner/reports/volume_parts_gt_rows.csv')
for col in ['parts_ml','columns_ml','hull_ml','footprint_ml','old_geom_ml']:
    r=np.log(real[col]/real.volume_ml)
    # global scale corrected
    g=r-np.median(r)
    obj_med=real.assign(r=r).groupby('object_id').r.transform('median')
    within=r-obj_med
    print(f'{col:13s} raw V={volume_score(real[col],real.volume_ml):.3f} scale-corrected V={np.mean(np.maximum(0,1-np.abs(g)/math.log(3))):.3f}  between-obj sd={obj_med.std():.3f} within-obj sd={within.std():.3f}  V if only within noise={np.mean(np.maximum(0,1-np.abs(within)/math.log(3))):.3f}')
