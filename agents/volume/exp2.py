"""Non-linear / embedding variants, all LOOO over objects."""
import sys, math, json, collections, numpy as np, pandas as pd
sys.path.insert(0, '/home/cleaner/Cleaner/agents/volume')
from exp1 import real, synth, y, ids, w_of, ridge, looo, report, design, MATS
from sklearn.ensemble import HistGradientBoostingRegressor, GradientBoostingRegressor, RandomForestRegressor

C = '/home/cleaner/Cleaner/agents/volume/shared_cache/'
meta = json.load(open(C + 'meta.json'))
z1 = np.load(C + 'features.npz'); z2 = np.load(C + 'features2.npz')
key = {(m['frame'], m['oid']): i for i, m in enumerate(meta)}
ridx = np.array([key[(f, o)] for f, o in zip(real.frame, real.object_id)])
sidx = np.array([key[(f, o)] for f, o in zip(synth.frame, synth.object_id)])


def emb(name, idx=ridx):
    z = z2 if name in z2.files else z1
    X = z[name][idx].astype(np.float64)
    return X / np.linalg.norm(X, axis=1, keepdims=True)


def gbm_looo(X, y=y, ids=ids, mono=None, params=None):
    pred = np.zeros(len(y)); w = w_of(ids)
    for o in sorted(set(ids)):
        tr = ids != o
        m = GradientBoostingRegressor(loss='huber', alpha=0.9, random_state=0, **(params or {}))
        m.fit(X[tr], y[tr], sample_weight=w[tr])
        pred[~tr] = m.predict(X[~tr])
    return pred


def rf_looo(X, y=y, ids=ids, **kw):
    pred = np.zeros(len(y)); w = w_of(ids)
    for o in sorted(set(ids)):
        tr = ids != o
        m = RandomForestRegressor(n_estimators=300, random_state=0, n_jobs=8, **kw)
        m.fit(X[tr], y[tr], sample_weight=w[tr])
        pred[~tr] = m.predict(X[~tr])
    return pred


def knn_obj_looo(E, k=3, target=None, ids=ids):
    """Object-level kNN in embedding space: mean embedding per object, predict target of nearest train objects."""
    target = y if target is None else target
    uniq = sorted(set(ids))
    Eo = {o: E[ids == o].mean(0) for o in uniq}
    To = {o: np.average(target[ids == o]) for o in uniq}
    pred = np.zeros(len(ids))
    for o in uniq:
        others = [u for u in uniq if u != o]
        M = np.stack([Eo[u] for u in others]); M /= np.linalg.norm(M, axis=1, keepdims=True)
        q = E[ids == o]; q = q / np.linalg.norm(q, axis=1, keepdims=True)
        sims = q @ M.T  # (n_frames, n_others)
        for j, s in zip(np.nonzero(ids == o)[0], sims):
            nn = np.argsort(-s)[:k]
            ww = np.exp(s[nn] / 0.05); ww /= ww.sum()
            pred[j] = float(np.sum(ww * np.array([To[others[i]] for i in nn])))
    return pred


if __name__ == '__main__':
    base = ['columns_ml', 'parts_ml', 'hull_ml', 'footprint_ml']
    Xb = design(real, base, inter='parts_ml', knots=(50, 300))
    p_ridge = looo(Xb)
    report('ridge baseline', p_ridge)
    geo_cols = ['columns_ml', 'parts_ml', 'hull_ml', 'footprint_ml', 'inp_colmean_ml', 'inp_colmax_ml', 'inp_sym_colmax_ml', 'inp_box_ml', 'inp_sym_hull_ml',
                'inp_h_mean_m', 'inp_h_p90_m', 'inp_L_m', 'inp_W_m', 'inp_cells_m2', 'fp_area_0', 'valid_frac', 'hole_inner_frac', 'mask_solidity']
    Xg = design(real, geo_cols)
    for params in [dict(n_estimators=150, max_depth=2, learning_rate=0.05, subsample=0.8, min_samples_leaf=10),
                   dict(n_estimators=300, max_depth=3, learning_rate=0.03, subsample=0.7, min_samples_leaf=15),
                   dict(n_estimators=100, max_depth=2, learning_rate=0.1, subsample=0.8, min_samples_leaf=20)]:
        report(f'GBM geo {params}', gbm_looo(Xg, params=params))
    report('RF geo', rf_looo(Xg, min_samples_leaf=5, max_features=0.5))
    # residual GBM on top of ridge
    # nested: ridge OOF inside training is complex; approximate by using ridge prediction as feature (fit on train only each fold)
    # kNN embedding
    for name in ['clipL_c14', 'clipL_m14', 'dinoL_m14', 'dinov2_b_m', 'siglip_c14', 'siglip_m14']:
        E = emb(name)
        for k in (1, 3, 5):
            p = knn_obj_looo(E, k)
            report(f'kNN-obj {name} k={k} (vol)', p)
        # residual target: log(gt) - log(colmax)
        resid = y - np.log(np.maximum(real.inp_colmax_ml.to_numpy(), 1))
        pr = knn_obj_looo(E, 3, target=resid) + np.log(np.maximum(real.inp_colmax_ml.to_numpy(), 1))
        report(f'kNN-obj {name} k=3 (resid of colmax)', pr)
        report(f'  blend 0.5 ridge + 0.5 knn-resid', 0.5 * p_ridge + 0.5 * pr)
        report(f'  blend 0.7 ridge + 0.3 knn-vol(k3)', 0.7 * p_ridge + 0.3 * knn_obj_looo(E, 3))
