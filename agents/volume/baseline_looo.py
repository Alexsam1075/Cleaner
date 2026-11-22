import sys, math, numpy as np, pandas as pd
sys.path.insert(0,'/home/cleaner/Cleaner')
from evaluation.volume_parts_cv import fit_calibrator, apply_calibrator, volume_score, MATERIALS, BINS
real=pd.read_csv('/home/cleaner/Cleaner/reports/volume_parts_gt_rows.csv')
ids=real.object_id.to_numpy()
pred=np.zeros(len(real)); predn=np.zeros(len(real))
rng=np.random.default_rng(0)
for o in sorted(set(ids)):
    tr=real[ids!=o]; va=real[ids==o]
    cal=fit_calibrator(tr)
    for i,r in zip(np.nonzero(ids==o)[0], va.to_dict('records')):
        pred[i]=apply_calibrator(cal,r,r['form'],r['material'])
        m=r['material'] if rng.random()>=0.08 else str(rng.choice([x for x in MATERIALS if x!=r['material']]))
        predn[i]=apply_calibrator(cal,r,r['form'],m)
real['pred']=pred
print('LOOO baseline V=%.4f noisy=%.4f'%(volume_score(pred,real.volume_ml),volume_score(predn,real.volume_ml)))
for lo,hi in BINS:
    g=real[(real.volume_ml>=lo)&(real.volume_ml<hi)]
    print(f'{lo}-{hi}: n={len(g)} V={volume_score(g.pred,g.volume_ml):.3f}')
for f,g in real.groupby('form'): print(f, len(g), round(volume_score(g.pred,g.volume_ml),3))
for f,g in real.groupby('material'): print(f, len(g), round(volume_score(g.pred,g.volume_ml),3))
real['ratio']=real.pred/real.volume_ml
print(real.groupby('object_id').agg(vol=('volume_ml','first'),ratio=('ratio','median'),n=('ratio','size')).sort_values('vol').round(2).to_string())
