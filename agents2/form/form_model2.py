"""Form model v2 = learned head + zero-shot ensemble prior.
  head : L2-regularised logistic regression on a feature block combination (CLIP-L crop embedding, zero-shot family
         differences, optional depth / DINO / FastSAM statistics), trained on real+synth annotations of the training objects.
  prior: equal-weight average of standardized zero-shot differences over ALL prompt families x 3 crop views (no learned
         weights -> cannot overfit to 30-40 objects); its decision point t is chosen on the training real frames.
  blend: logit = logit(head) + lam * 2 * (prior - t).
All choices (feature spec, synth weight, weight decay, lam, final threshold) are made by grouped inner CV over the
training objects only; optional top-k averaging over the best configs."""
from collections import defaultdict

import numpy as np
import torch
import mat_cv
from mat_cv import fit_predict  # LBFGS logreg (audit); run on CPU: faster than the shared GPU for these sizes

mat_cv.DEV = "cpu"
torch.set_num_threads(8)

GRID = {
    "spec": ["clipL_c14+zs", "clipL_c14+zs+depth", "clipL_c14+dino+zs", "clipL_c14+zs+depth+sam"],
    "sw": [0.5],
    "wd": [1e-2, 1e-1],
    "lam": [0.0, 1.0, 2.0, 4.0, 8.0],
}
PRIOR_SCALE = 2.0


def sample_weights(idx, oid, src, y, synth_w):
    cnt = defaultdict(int)
    for i in idx:
        cnt[(oid[i], src[i])] += 1
    w = np.array([(1.0 if src[i] == "real_train" else synth_w) / cnt[(oid[i], src[i])] for i in idx])
    per = defaultdict(float)
    for j, i in enumerate(idx):
        per[y[i]] += w[j]
    w = np.array([w[j] / per[y[i]] for j, i in enumerate(idx)])
    return w / w.mean()


def head_predict(X, y, oid, src, tr, te, sw, wd):
    mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-6
    Xs = (X - mu) / sd
    return fit_predict(Xs, y, sample_weights(tr, oid, src, y, sw), tr, te, wd=wd, n_cls=2)


def bal_acc(p_solid, y, thr=0.5):
    pred = (p_solid > thr).astype(int)
    return float(np.mean([(pred[y == c] == c).mean() for c in (0, 1) if (y == c).any()]))


def obj_bal_acc(p_solid, y, oid, thr=0.5):
    objs = sorted(set(oid))
    po = np.array([p_solid[oid == o].mean() for o in objs])
    yo = np.array([y[oid == o][0] for o in objs])
    return bal_acc(po, yo, thr)


def logit(p):
    return np.log(p + 1e-6) - np.log(1 - p + 1e-6)


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


class Prior:
    """Standardization stats + decision point of the zero-shot ensemble, fitted on training samples."""

    def fit(self, M, fit_idx, y_real, real_idx):
        self.mu, self.sd = M[fit_idx].mean(0), M[fit_idx].std(0) + 1e-6
        s = self.score(M)[real_idx]
        cand = np.quantile(s, np.linspace(0.05, 0.95, 91))
        best = max(bal_acc(s, y_real, t) for t in cand)
        ts = [t for t in cand if bal_acc(s, y_real, t) >= best - 1e-9]
        self.t = float(np.median(ts))
        return self

    def score(self, M):
        return ((M - self.mu) / self.sd).mean(1)

    def z(self, M):
        return PRIOR_SCALE * (self.score(M) - self.t)


class FormModel2:
    def __init__(self, bank, y, oid, src, grid=GRID, inner_folds=5, seed=0, topk=1, thr_grid=(0.5,), repeats=3, verbose=False):
        self.bank, self.y, self.oid, self.src = bank, y, oid, src
        self.grid, self.inner_folds, self.seed, self.topk, self.thr_grid, self.verbose = grid, inner_folds, seed, topk, thr_grid, verbose
        self.repeats = repeats
        self.M = bank.zs_matrix()

    def _idx(self, objs, sw):
        real = self.src == "real_train"
        return np.where(np.isin(self.oid, list(objs)) & (real | (sw > 0)))[0]

    def inner_oof(self, train_objs, seed):
        real = self.src == "real_train"
        rs = np.random.default_rng(seed)
        groups = np.array_split(rs.permutation(train_objs), self.inner_folds)
        oof = {}
        zprior = np.full(len(self.y), np.nan)
        for g in groups:
            rest = [o for o in train_objs if o not in set(g)]
            a_all = self._idx(rest, 1.0)
            a_real = self._idx(rest, 0.0)
            b = np.where(np.isin(self.oid, g) & real)[0]
            pr = Prior().fit(self.M, a_all, self.y[a_real], a_real)
            zprior[b] = pr.z(self.M)[b]
        for spec in self.grid["spec"]:
            X = self.bank.features(spec)
            for sw in self.grid["sw"]:
                for wd in self.grid["wd"]:
                    zh = np.full(len(self.y), np.nan)
                    for g in groups:
                        rest = [o for o in train_objs if o not in set(g)]
                        a = self._idx(rest, sw)
                        b = np.where(np.isin(self.oid, g) & real)[0]
                        zh[b] = logit(head_predict(X, self.y, self.oid, self.src, a, b, sw, wd)[:, 1])
                    for lam in self.grid["lam"]:
                        oof[(spec, sw, wd, lam)] = sigmoid(zh + lam * zprior)
        return oof

    def fit(self, train_objs):
        train_objs = sorted(train_objs)
        real = self.src == "real_train"
        tr_real = np.where(np.isin(self.oid, train_objs) & real)[0]
        yt, ot = self.y[tr_real], self.oid[tr_real]
        # repeated inner CV (different object groupings): criteria averaged over repeats -> stable selection
        oofs = [self.inner_oof(train_objs, self.seed + r) for r in range(self.repeats)]
        scored = []
        for cfg in oofs[0]:
            crit = []
            for oof in oofs:
                p = oof[cfg][tr_real]
                ll = float(np.mean(np.log(np.where(yt == 1, p, 1 - p) + 1e-9)))
                crit.append((bal_acc(p, yt), obj_bal_acc(p, yt, ot), ll))
            c = np.mean(crit, 0)
            scored.append(((round(float(c[0]), 4), round(float(c[1]), 4), float(c[2])), cfg))
        scored.sort(key=lambda t: t[0], reverse=True)
        self.cfgs = [c for _, c in scored[:self.topk]]
        self.inner_score = scored[0][0]
        self.inner_table = scored
        pe = np.mean([oof[c][tr_real] for c in self.cfgs for oof in oofs], 0)
        self.thr = float(max(self.thr_grid, key=lambda t: (bal_acc(pe, yt, t), -abs(t - 0.5))))
        self.inner_oof_p = pe
        if self.verbose:
            print(f"  selected {self.cfgs} inner bal/obj/ll {self.inner_score} thr {self.thr}")
        self.train_objs = train_objs
        all_idx = self._idx(train_objs, 1.0)
        self.prior = Prior().fit(self.M, all_idx, yt, tr_real)
        self._fitted = {}
        for spec, sw, wd, lam in self.cfgs:
            self._fitted[(spec, sw, wd, lam)] = (self.bank.features(spec), self._idx(train_objs, sw))
        return self

    def predict_bank(self, bank_te):
        """P(solid) for all samples of another Bank."""
        zp = self.prior.z(bank_te.zs_matrix())
        ps = []
        for (spec, sw, wd, lam), (X, tr) in self._fitted.items():
            Xte = bank_te.features(spec)
            Xall = np.concatenate([X, Xte])
            yall = np.concatenate([self.y, np.zeros(len(Xte), int)])
            oid_all = np.concatenate([self.oid, np.array(["_te"] * len(Xte))])
            src_all = np.concatenate([self.src, np.array(["_te"] * len(Xte))])
            te = np.arange(len(X), len(Xall))
            zh = logit(head_predict(Xall, yall, oid_all, src_all, tr, te, sw, wd)[:, 1])
            ps.append(sigmoid(zh + lam * zp))
        return np.mean(ps, 0)

    def predict_idx(self, te):
        zp = self.prior.z(self.M)[te]
        ps = []
        for (spec, sw, wd, lam), (X, tr) in self._fitted.items():
            zh = logit(head_predict(X, self.y, self.oid, self.src, tr, te, sw, wd)[:, 1])
            ps.append(sigmoid(zh + lam * zp))
        return np.mean(ps, 0)
