import sys, warnings, json, numpy as np, pandas as pd, math, collections
warnings.filterwarnings('ignore')
sys.path.insert(0,'/home/cleaner/Cleaner/agents/volume')
from exp1 import *
from sklearn.kernel_ridge import KernelRidge
from sklearn.ensemble import GradientBoostingRegressor
f2=pd.read_csv('/home/cleaner/Cleaner/agents/volume/cache/feat2_real.csv')
f2=f2.rename(columns={c:c+'_ml' for c in f2.columns if c.endswith(('colmax','colmean'))})
f3=pd.read_csv('/home/cleaner/Cleaner/agents/volume/cache/feat3_real.csv')
df=pd.concat([real,f2.drop(columns=['split','frame','object_id']),f3.drop(columns=['split','frame','object_id'])],axis=1).copy()
cols=['columns_ml','parts_ml','hull_ml','footprint_ml','gi_env_ml']
X=design(df,cols,inter='gi_env_ml',knots=(50,300))
w=w_of(ids)
def inner_folds(it,n=5):
    ut=np.array(sorted(set(it))); np.random.default_rng(0).shuffle(ut); fo={u:k%n for k,u in enumerate(ut)}; return np.array([fo[u] for u in it])
def nested(X, fit, grid):
    pred=np.zeros(len(y))
    for o in sorted(set(ids)):
        tr=ids!=o; Xt,yt,it,wt=X[tr],y[tr],ids[tr],w[tr]; f=inner_folds(it)
        best=None
        for g in grid:
            p=np.zeros(len(yt))
            for k in range(5): p[f==k]=fit(Xt[f!=k],yt[f!=k],wt[f!=k],g)(Xt[f==k])
            s=np.sum(wt*np.maximum(0,1-np.abs(p-yt)/math.log(3)))/wt.sum()
            if best is None or s>best[0]: best=(s,g)
        pred[~tr]=fit(Xt,yt,wt,best[1])(X[~tr])
    return pred
def std_fit(Xt,yt,wt,mk):
    mu=Xt.mean(0); sd=Xt.std(0); sd[sd<1e-9]=1
    m=mk().fit((Xt-mu)/sd,yt,sample_weight=wt); return lambda Xn: m.predict((Xn-mu)/sd)
# kernel ridge rbf
grid=[(a,g) for a in (0.1,0.3,1,3) for g in (0.01,0.03,0.1)]
report('KRR rbf', nested(X, lambda Xt,yt,wt,g: std_fit(Xt,yt,wt,lambda: KernelRidge(alpha=g[0],kernel='rbf',gamma=g[1])), grid))
# ridge + residual GBM (nested over n_estimators)
def rgbm(Xt,yt,wt,g):
    base=ridge(Xt,yt,wt,3.0); r=yt-base(Xt)
    m=GradientBoostingRegressor(n_estimators=g[0],max_depth=2,learning_rate=0.05,subsample=0.8,min_samples_leaf=g[1],loss='huber',random_state=0).fit(Xt,r,sample_weight=wt)
    return lambda Xn: base(Xn)+m.predict(Xn)
report('ridge+resGBM', nested(X, rgbm, [(0,10)]+[(n,l) for n in (30,80,150) for l in (10,25)]) if False else nested(X, lambda Xt,yt,wt,g: (ridge(Xt,yt,wt,3.0) if g[0]==0 else rgbm(Xt,yt,wt,g)), [(0,10),(30,10),(80,10),(150,10),(80,25)]))
# offset model: target y - log(env) with env excluded from X? keep X but target offset
off=np.log(np.maximum(df.gi_env_ml.to_numpy(),1))
pred=np.zeros(len(y))
for o in sorted(set(ids)):
    tr=ids!=o
    pred[~tr]=off[~tr]+ridge(X[tr],(y-off)[tr],w[tr],3.0)(X[~tr])
report('offset ridge a=3',pred)
# per-form models
pred=np.zeros(len(y))
for fm in ('bulk','solid'):
    s=(df.form==fm).to_numpy()
    Xf=design(df[s],cols,inter='gi_env_ml',knots=(50,300))
    pred[s]=looo(Xf,y=y[s],ids=ids[s])
report('per-form ridge',pred)
# IRLS + knn-stack combination
from exp4 import stack_looo as _s
