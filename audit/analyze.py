import json, math, csv, os
import numpy as np, pandas as pd
from collections import Counter
rows = pd.DataFrame(json.load(open('audit/geo_rows.json')))
dens = {f"{r['form']}_{r['material']}": float(r['density_g_per_ml']) for r in csv.DictReader(open(os.environ.get('ASTRAMIS_DATA','dataset')+'/densities.csv'))}
Vs = lambda p, g: np.clip(1 - np.abs(np.log(p / g)) / math.log(3), 0, 1)
def Sbal(pt, t):
    return np.mean([(pt[t == v] == v).mean() for v in (0, 1) if (t == v).any()])
print("== Объём: сырой расчёт по глубине (без обучения), V по всем 214 объектам ==")
for c in ['raw', 'thr2', 'thr3', 'er5', 'er5thr2']:
    p = rows[c].fillna(rows['vol'].median()).clip(lower=1)
    print(f"{c:8s} V={Vs(p, rows.vol).mean():.3f}   медиана отношения pred/gt={np.median(p/rows.vol):.2f}")
print("\n== Константы для S ==")
t = rows['take'].values
for name, pt in [('всегда take=1', np.ones_like(t)), ('всегда 0', np.zeros_like(t))]:
    print(name, 'S=', Sbal(pt, t))
print("\n== Leave-one-object-out (40 фолдов) ==")
objs = sorted(rows.oid.unique())
feat = lambda d: np.c_[np.log1p(d.raw.fillna(0)), np.log1p(d.er5thr2.fillna(0)), d.h95.fillna(0)*10, np.log(d.area_px), d.valid]
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.ensemble import HistGradientBoostingRegressor as HGB
res = {k: [] for k in ['ridge', 'ridge_mat', 'lgb', 'density_take', 'cal_take', 'mass_ridge_take']}
for o in objs:
    tr, te = rows[rows.oid != o], rows[rows.oid == o]
    Xtr, Xte = feat(tr), feat(te)
    r = Ridge(1.0).fit(Xtr, np.log(tr.vol)); pv = np.exp(r.predict(Xte))
    res['ridge'] += list(Vs(pv, te.vol))
    M = lambda d: pd.get_dummies(d.form + '_' + d.mat).reindex(columns=sorted((rows.form + '_' + rows.mat).unique()), fill_value=0).values
    r2 = Ridge(1.0).fit(np.c_[Xtr, M(tr)], np.log(tr.vol)); pv2 = np.exp(r2.predict(np.c_[Xte, M(te)]))
    res['ridge_mat'] += list(Vs(pv2, te.vol))
    g = HGB(max_iter=200, learning_rate=0.05, max_leaf_nodes=7, min_samples_leaf=5).fit(Xtr, np.log(tr.vol))
    res['lgb'] += list(Vs(np.exp(g.predict(Xte)), te.vol))
    mass = pv2 * np.array([dens[f"{a}_{b}"] for a, b in zip(te.form, te.mat)])
    res['density_take'] += list(zip((mass <= 500).astype(int), te['take']))
    rm = Ridge(1.0).fit(np.c_[Xtr, M(tr)], np.log(tr.mass)); pm = np.exp(rm.predict(np.c_[Xte, M(te)]))
    res['mass_ridge_take'] += list(zip((pm <= 500).astype(int), te['take']))
    lr = LogisticRegression(C=1.0, class_weight='balanced', max_iter=2000).fit(np.c_[Xtr, M(tr)], tr['take'])
    res['cal_take'] += list(zip(lr.predict(np.c_[Xte, M(te)]), te['take']))
for k in ['ridge', 'ridge_mat', 'lgb']:
    print(f"V {k:10s} = {np.mean(res[k]):.3f}")
for k in ['density_take', 'mass_ridge_take', 'cal_take']:
    a = np.array(res[k]); print(f"S {k:16s} = {Sbal(a[:,0], a[:,1]):.3f}")
print("\n(в M-варианте форма/материал — истинные; это верхняя граница при идеальном классификаторе материала)")
print("\n== Распределение объектов по take и массе ==")
u = rows.drop_duplicates('oid')
print(u.groupby('take').size().to_dict(), ' тяжёлые:', sorted(u[u['take'] == 0].oid))
print("\n== Плотность класса vs реальная плотность предмета ==")
u = u.assign(real_d=u.mass/u.vol, table_d=[dens[f"{a}_{b}"] for a, b in zip(u.form, u.mat)])
print(u[['oid','form','mat','vol','mass','real_d','table_d']].sort_values('mat').to_string(index=False))

print("\n== Порог массы для take (LOO, объём ridge_mat, плотность из таблицы) ==")
from sklearn.linear_model import Ridge as _R
preds=[]
for o in objs:
    tr, te = rows[rows.oid != o], rows[rows.oid == o]
    r2 = _R(1.0).fit(np.c_[feat(tr), M(tr)], np.log(tr.vol)); pv2 = np.exp(r2.predict(np.c_[feat(te), M(te)]))
    preds += list(zip(pv2 * np.array([dens[f"{a}_{b}"] for a, b in zip(te.form, te.mat)]), te['take']))
a = np.array(preds)
for thr in [200, 250, 300, 350, 400, 450, 500, 600]:
    print(thr, round(Sbal((a[:,0] <= thr).astype(int), a[:,1]), 3))
