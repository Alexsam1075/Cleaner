"""Эксперимент 3: какие виды использовать в zero-shot (вложенный выбор), разбор ошибок по кадрам."""
import argparse
import json
import itertools

import numpy as np

from common import HERE, MATS, MODELS, VIEWS, report, softmax, M_frame
from exp1 import Runner, SET_KEYS, W_GRID, fuse, m_base

GEN = [s for s in SET_KEYS if not s.startswith("rich") and not s.startswith("generic_syn")]
VIEW_SETS = [("c14",), ("c20",), ("c14", "c20"), ("m10", "c14", "c20"), tuple(VIEWS), ("m10", "c20")]


def m_select_views(R, train_objs, te, skeys=("rich|floor",), mode="mean", temp=0.01, view_sets=VIEW_SETS):
    tr = np.where(np.isin(R.oid, train_objs) & R.real)[0]
    Pin = R.inner_oof(train_objs)
    best, best_s = None, -1
    for vs in view_sets:
        L = np.mean([R.logits(s, mode, views=list(vs)) for s in skeys], 0)
        Z = softmax(L / temp)
        for w in W_GRID:
            s = M_frame(fuse(Pin, Z, w), R.mat, tr)
            if s > best_s:
                best, best_s = (vs, w), s
    vs, w = best
    L = np.mean([R.logits(s, mode, views=list(vs)) for s in skeys], 0)
    Pte = R.head_full(train_objs, te)
    return fuse(Pte, softmax(L[te] / temp), w), {"views": vs, "w": w}


def dump_errors(R, P, name):
    rows = []
    for i in R.idx:
        p = int(P[i].argmax())
        if p != R.mat[i]:
            rows.append({"frame": R.meta[i]["frame"], "oid": R.meta[i]["oid"], "true": MATS[R.mat[i]], "pred": MATS[p], "light": R.light[i],
                         "p": np.round(P[i], 2).tolist(), "bbox": R.meta[i]["bbox"]})
    json.dump(rows, open(HERE / f"errors_{name}.json", "w"), indent=1)
    from collections import Counter
    print(f"  ошибки ({len(rows)}): по предметам {Counter((r['oid'], r['true'], r['pred']) for r in rows).most_common()}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", default="all")
    a = ap.parse_args()
    R = Runner()
    results = {}
    print("== Виды в zero-shot ==")
    for vs in VIEW_SETS:
        P, results[f"rich_views_{'+'.join(vs)}"] = R.run(m_base, f"голова ⊕ zs rich|floor, виды {'+'.join(vs)}", skeys=("rich|floor",), views=list(vs))
        if vs == ("c14",):
            dump_errors(R, P, "base")
        if vs == ("c14", "c20"):
            dump_errors(R, P, "rich_c14c20")
    for vs in VIEW_SETS:
        _, results[f"rich_lse_views_{'+'.join(vs)}"] = R.run(m_base, f"голова ⊕ zs rich|floor lse, виды {'+'.join(vs)}", skeys=("rich|floor",), mode="lse", views=list(vs))
    for vs in (("c14", "c20"), ("c20",)):
        _, results[f"gen_views_{'+'.join(vs)}"] = R.run(m_base, f"голова ⊕ zs все общие, виды {'+'.join(vs)}", skeys=tuple(GEN), views=list(vs))
        _, results[f"gen_lse_views_{'+'.join(vs)}"] = R.run(m_base, f"голова ⊕ zs все общие lse, виды {'+'.join(vs)}", skeys=tuple(GEN), mode="lse", views=list(vs))
    _, results["sel_views_rich"] = R.run(m_select_views, "голова ⊕ zs rich|floor, виды вложенно", skeys=("rich|floor",))
    _, results["sel_views_gen"] = R.run(m_select_views, "голова ⊕ zs все общие, виды вложенно", skeys=tuple(GEN))
    json.dump(results, open(HERE / f"exp3_{a.part}.json", "w"), indent=1, default=str)


if __name__ == "__main__":
    main()
