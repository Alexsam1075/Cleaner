import sys, math, collections, numpy as np, pandas as pd
sys.path.insert(0, '/home/cleaner/Cleaner')
from evaluation.volume_parts_cv import volume_score, BINS
C = '/home/cleaner/Cleaner/agents/volume/cache/'
real = pd.read_csv(C + 'feat_real.csv'); synth = pd.read_csv(C + 'feat_synth.csv')
MATS = ['glass', 'metal', 'mineral', 'paper', 'plastic', 'wood']
y = np.log(real.volume_ml.to_numpy()); ids = real.object_id.to_numpy()


def w_of(ids):
    c = collections.Counter(ids); return np.array([1 / c[o] for o in ids])


def ridge(X, y, w, a):
    mu = np.average(X, 0, weights=w); sd = np.sqrt(np.average((X - mu) ** 2, 0, weights=w)); sd[sd < 1e-9] = 1
    Z = (X - mu) / sd; ym = np.average(y, weights=w)
    coef = np.linalg.solve(Z.T @ (Z * w[:, None]) + a * np.eye(Z.shape[1]), Z.T @ (w * (y - ym)))
    return lambda Xn: ym + ((Xn - mu) / sd) @ coef


def looo(X, y=y, ids=ids, alphas=(0.3, 1, 3, 10, 30, 100), inner=5, Xs=None, ys=None, ids_s=None, syn_w=0.3):
    pred = np.zeros(len(y)); w = w_of(ids)
    for o in sorted(set(ids)):
        tr = ids != o; Xt, yt, it, wt = X[tr], y[tr], ids[tr], w[tr]
        if Xs is not None:
            s = ids_s != o
            Xt = np.vstack([Xt, Xs[s]]); yt = np.concatenate([yt, ys[s]]); it = np.concatenate([it, ids_s[s]])
            wt = np.concatenate([wt, syn_w * w_of(ids_s[s])])
        if len(alphas) > 1:
            ut = np.array(sorted(set(it))); np.random.default_rng(0).shuffle(ut); fo = {u: k % inner for k, u in enumerate(ut)}; f = np.array([fo[u] for u in it])
            best = None
            for a in alphas:
                p = np.zeros(len(yt))
                for k in range(inner): p[f == k] = ridge(Xt[f != k], yt[f != k], wt[f != k], a)(Xt[f == k])
                s_ = np.sum(wt * np.maximum(0, 1 - np.abs(p - yt) / math.log(3))) / wt.sum()
                if best is None or s_ > best[0]: best = (s_, a)
            a = best[1]
        else:
            a = alphas[0]
        pred[~tr] = ridge(Xt, yt, wt, a)(X[~tr])
    return pred


def report(name, pred, df=real):
    v = volume_score(np.exp(pred), df.volume_ml)
    bins = []
    for lo, hi in BINS:
        g = (df.volume_ml >= lo) & (df.volume_ml < hi)
        bins.append(volume_score(np.exp(pred[g.to_numpy()]), df.volume_ml[g]))
    print(f'{name:40s} V={v:.4f}  bins={[round(b, 3) for b in bins]}', flush=True)
    return v


def design(df, cols, cls=True, inter=None, knots=()):
    X = [np.log(np.maximum(df[c].to_numpy(dtype=float), 1e-6 if 'm2' in c or '_m' in c else 1.0)) if (c.endswith('_ml') or c.endswith('_m2') or c.endswith('_m')) else df[c].to_numpy(dtype=float) for c in cols]
    X = np.stack(X, 1)
    parts = [X]
    if cls:
        form = (df.form == 'bulk').to_numpy().astype(float)[:, None]
        mat = np.stack([(df.material == m).to_numpy() for m in MATS], 1).astype(float)
        parts += [form, mat, form * mat]
        if inter is not None:
            main = X[:, cols.index(inter)][:, None]
            parts += [main * mat, main * form]
    for t in knots:
        main = X[:, cols.index(inter)]
        parts.append(np.maximum(0, main - math.log(t))[:, None])
    return np.hstack(parts)


if __name__ == '__main__':
    base = ['columns_ml', 'parts_ml', 'hull_ml', 'footprint_ml']
    report('baseline4 +cls', looo(design(real, base)))
    report('baseline4 +cls+inter', looo(design(real, base, inter='parts_ml')))
    report('baseline4 +cls+inter+knots', looo(design(real, base, inter='parts_ml', knots=(50, 300))))
    for c in ['colmax_ml', 'colmean_ml', 'inp_colmax_ml', 'inp_colmean_ml', 'sym_colmax_ml', 'inp_sym_colmax_ml', 'box_ml', 'ell_ml', 'cyl_ml', 'sym_box_ml', 'inp_box_ml', 'inp_ell_ml', 'sym_hull_ml', 'inp_sym_hull_ml']:
        report(f'single {c} +cls', looo(design(real, [c])))
    rich = ['columns_ml', 'parts_ml', 'hull_ml', 'footprint_ml', 'inp_colmean_ml', 'inp_colmax_ml', 'inp_sym_colmax_ml', 'inp_box_ml', 'inp_ell_ml', 'inp_sym_hull_ml']
    report('rich +cls', looo(design(real, rich)))
    report('rich +cls+inter', looo(design(real, rich, inter='inp_colmean_ml')))
    rich2 = rich + ['inp_h_mean_m', 'inp_h_p90_m', 'inp_L_m', 'inp_W_m', 'inp_cells_m2', 'hull2d_m2', 'fp_area_0', 'mask_area_at_z_m2', 'rect_long_m', 'rect_short_m', 'sil_height_m', 'valid_frac', 'hole_inner_frac', 'hole_edge_frac', 'mask_solidity']
    report('rich2 +cls', looo(design(real, rich2)))
    report('rich2 +cls+inter', looo(design(real, rich2, inter='inp_colmean_ml')))
    report('rich2 +cls+inter+knots', looo(design(real, rich2, inter='inp_colmean_ml', knots=(50, 300))))
    # with synth
    ys_ = np.log(synth.volume_ml.to_numpy()); ids_s = synth.object_id.to_numpy()
    for sw in (0.1, 0.3, 1.0):
        report(f'rich2 +cls+inter +synth w={sw}', looo(design(real, rich2, inter='inp_colmean_ml'), Xs=design(synth, rich2, inter='inp_colmean_ml'), ys=ys_, ids_s=ids_s, syn_w=sw))
