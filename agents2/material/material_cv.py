"""Честная оценка MaterialClassifier: leave-one-object-out по 40 train-объектам (только real-кадры отложенного предмета;
все гиперпараметры — внутри train-объектов фолда). Метрики: M (макро по материалам, по кадрам), M_obj, по освещению,
per-class, bootstrap-интервал по объектам, перестановочный тест (материалы перемешаны между предметами).
Запуск: ~/venv/bin/python material_cv.py [--cfg '{"zs_mode": "mean"}'] [--perm 5] [--tag name]"""
import argparse
import json
import time
from collections import defaultdict

import numpy as np

from predict_material import CACHE, HERE, MATS, MODELS, VIEWS, M_frame, MaterialClassifier, TextBank


def evaluate(P, mat, oid, light, idx):
    out = {"M": M_frame(P[idx], mat[idx])}
    pred = P[idx].argmax(1)
    out["per_class"] = {MATS[c]: f"{int((pred[mat[idx] == c] == c).sum())}/{int((mat[idx] == c).sum())}" for c in range(6)}
    out["light"] = {lg: round(M_frame(P[idx[light[idx] == lg]], mat[idx[light[idx] == lg]]), 3) for lg in ("bright", "dim", "side")}
    hits, wrong = defaultdict(list), {}
    for o in sorted(set(oid[idx])):
        ii = idx[oid[idx] == o]
        p = int(np.log(P[ii] + 1e-9).mean(0).argmax())
        hits[mat[ii[0]]].append(int(p == mat[ii[0]]))
        if p != mat[ii[0]]:
            wrong[o] = MATS[p]
    out["M_obj"] = float(np.mean([np.mean(v) for v in hits.values()]))
    out["per_obj"] = {MATS[c]: f"{sum(v)}/{len(v)}" for c, v in sorted(hits.items())}
    out["wrong_objects"] = wrong
    return out


def looo(cfg, mat_override=None, verbose=True):
    bank = TextBank()
    F = dict(np.load(CACHE / "features2.npz"))
    meta = json.load(open(CACHE / "meta.json"))
    if mat_override is not None:
        meta = [dict(m, mat=mat_override[m["oid"]]) for m in meta]
    clf0 = MaterialClassifier(cfg, bank, F, meta)
    oid, mat, real = clf0.oid, clf0.mat, clf0.real
    objs = sorted(set(oid[real]))
    P = np.zeros((len(mat), 6))
    ws = []
    t0 = time.time()
    for k, o in enumerate(objs):
        clf = MaterialClassifier(cfg, bank, F, meta).fit([q for q in objs if q != o])
        te = np.where((oid == o) & real)[0]
        P[te] = clf.predict(F["clipL_c14"][te], {(t, v): F[f"{t}_{v}"][te] for t in MODELS for v in VIEWS})
        ws.append(clf.w)
        if verbose and k % 10 == 9:
            print(f"  фолд {k + 1}/{len(objs)} ({time.time() - t0:.0f}s)", flush=True)
    return P, np.array(ws), clf0


def bootstrap(P, mat, oid, idx, n=1000, seed=0):
    rs = np.random.default_rng(seed)
    objs = sorted(set(oid[idx]))
    by = {o: idx[oid[idx] == o] for o in objs}
    vals = []
    for _ in range(n):
        pick = rs.choice(objs, len(objs), replace=True)
        ii = np.concatenate([by[o] for o in pick])
        vals.append(M_frame(P[ii], mat[ii]))
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfg", default="{}")
    ap.add_argument("--perm", type=int, default=0)
    ap.add_argument("--tag", default="final")
    a = ap.parse_args()
    cfg = json.loads(a.cfg)
    P, ws, clf = looo(cfg)
    light = np.array([m.get("lighting") or "" for m in clf.meta])
    idx = np.where(clf.real)[0]
    r = evaluate(P, clf.mat, clf.oid, light, idx)
    r["w_zs_choices"] = {str(k): int(v) for k, v in zip(*np.unique(ws, return_counts=True))}
    r["bootstrap95"] = bootstrap(P, clf.mat, clf.oid, idx)
    r["cfg"] = clf.cfg
    print(f"\n== MaterialClassifier LOOO ({a.tag}) ==")
    print(f"  M={r['M']:.3f}  M_obj={r['M_obj']:.3f}  95% CI {r['bootstrap95']}")
    print(f"  по классам (кадры): {r['per_class']}")
    print(f"  по классам (предметы): {r['per_obj']}; ошибки: {r['wrong_objects']}")
    print(f"  по свету: {r['light']}")
    print(f"  выбор w_zs: {r['w_zs_choices']}")
    np.save(HERE / f"oof_{a.tag}.npy", P)
    if a.perm > 0:
        print("\n== Перестановочный тест (материалы перемешаны между предметами) ==")
        objs = sorted(set(clf.oid[clf.real]))
        true = {o: clf.meta[int(np.where(clf.oid == o)[0][0])]["mat"] for o in objs}
        perms = []
        for s in range(a.perm):
            rs = np.random.default_rng(100 + s)
            vals = [true[o] for o in objs]
            rs.shuffle(vals)
            ov = dict(zip(objs, vals))
            Pp, _, c2 = looo(cfg, mat_override=ov, verbose=False)
            m = M_frame(Pp[idx], c2.mat[idx])
            perms.append(m)
            print(f"  перестановка {s}: M={m:.3f}", flush=True)
        r["permutation"] = {"values": perms, "mean": float(np.mean(perms))}
        print(f"  среднее {np.mean(perms):.3f} (случайный уровень ≈ 0.167)")
    json.dump(r, open(HERE / f"material_cv_{a.tag}.json", "w"), indent=1, default=str)


if __name__ == "__main__":
    main()
