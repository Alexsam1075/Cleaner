"""Идея: классифицировать геометрическую фигуру объекта и считать объём её формулой.
Фигуры и формулы (L ≥ W — габариты следа на полу по якорю пол+камера, H — высота):
  box        L·W·H                 (коробка, брусок)
  cylinder   π/4·L·W·H             (лежащая бутылка/банка: L — длина, W≈H — диаметр → π/4·L·D²)
  ellipsoid  π/6·L·W·H             (камень, комок)
  prism      1/2·L·W·H             (клин, треугольная призма)
  cone/pile  1/3·L·W·H             (горка, насыпная груда)
  flat       S_следа·H             (плоский предмет: пакет, лист, щепа)
  columns    интеграл карты высот  (как есть, без формулы)
Шаги:
 1) «Оракул»: для каждого предмета берём формулу, лучшую для него (верхняя граница, если фигура угадана идеально).
 2) Честно: фигура предсказывается (а) по геометрии (отношения H/W, W/L, заполнение следа, solidity, профиль высот)
    или (б) по картинке zero-shot CLIP/SigLIP («a box», «a bottle lying», «a pile» …), а формула — с одним
    множителем на фигуру, подобранным на обучающих предметах; LOOO по 40 предметам.
 3) Гибрид: объём по фигуре как дополнительный признак калибратора volume/.
Только real_train, real/test не используется."""
import json
import math
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path("/home/cleaner/Cleaner")
sys.path.insert(0, "/home/cleaner/Cleaner/agents/volume")
from exp1 import design, ridge, w_of  # noqa: E402

L3 = math.log(3)
CV = "/home/cleaner/Cleaner/agents/volume/cache/"
FIG = ["box", "cylinder", "ellipsoid", "prism", "pile", "flat", "columns"]
COEF = {"box": 1.0, "cylinder": math.pi / 4, "ellipsoid": math.pi / 6, "prism": 0.5, "pile": 1 / 3}


def Vs(p, g):
    return np.clip(1 - np.abs(np.log(np.clip(p, 1e-3, None) / g)) / L3, 0, 1)


def fig_volumes(d):
    L = d["inp_L_m"].fillna(d["L_m"]).clip(lower=0.005).values
    W = d["inp_W_m"].fillna(d["W_m"]).clip(lower=0.005).values
    H = d["inp_h_max_m"].fillna(d["h_max_m"]).fillna(d["sil_height_m"]).clip(lower=0.003).values
    Hp = d["inp_h_p90_m"].fillna(d["h_p90_m"]).clip(lower=0.003).values
    S = d["inp_cells_m2"].fillna(d["cells_m2"]).clip(lower=1e-5).values
    out = {f: COEF[f] * L * W * H * 1e6 for f in COEF}
    out["cylinder"] = math.pi / 4 * L * np.minimum(W, np.maximum(H, W * 0.5)) * H * 1e6
    out["flat"] = S * Hp * 1e6
    out["columns"] = d["inp_colmean_ml"].fillna(d["columns_ml"]).clip(lower=1).values
    return pd.DataFrame(out), L, W, H, S


def geo_shape_feats(d, L, W, H, S):
    return np.c_[np.log(H / W), np.log(W / L), S / (L * W), d["mask_solidity"].fillna(0.8).values,
                 d["inp_h_mean_m"].fillna(d["h_mean_m"]).values / H, d["inp_h_std_m"].fillna(d["h_std_m"]).values / H,
                 d["hole_inner_frac"].fillna(0).values, (d["form"] == "bulk").astype(float).values]


def main():
    d = pd.read_csv(CV + "feat_real.csv")
    g = d["volume_ml"].values
    ids = d["object_id"].values
    objs = sorted(set(ids))
    FV, L, W, H, S = fig_volumes(d)

    print("== Сырые формулы (без множителя), V по всем 214 кадрам ==")
    for f in FIG:
        print(f"  {f:10s} V={Vs(FV[f].values, g).mean():.3f}   медиана pred/gt={np.median(FV[f].values / g):.2f}")

    # множитель на фигуру по train-объектам (медиана log-отношения по объектам)
    def k_for(f, tr):
        lr = pd.Series(np.log(g[tr] / FV[f].values[tr])).groupby(ids[tr]).median()
        return math.exp(float(np.median(lr)))

    print("\n== 1) Оракул: для каждого предмета своя лучшая фигура (множители — LOOO) ==")
    best_fig, pred_or = {}, np.zeros(len(d))
    for o in objs:
        tr, te = ids != o, ids == o
        sc = {f: Vs(FV[f].values[te] * k_for(f, tr), g[te]).mean() for f in FIG}
        bf = max(sc, key=sc.get)
        best_fig[o] = bf
        pred_or[te] = FV[bf].values[te] * k_for(bf, tr)
    print(f"  V оракула = {Vs(pred_or, g).mean():.3f}   (выбор фигуры подсмотрен по ответу — верхняя граница)")
    print("  лучшая фигура по предметам:", dict(Counter(best_fig.values())))
    meta = d.groupby("object_id").agg(form=("form", "first"), mat=("material", "first"), gt=("volume_ml", "first"))
    print("  ", ", ".join(f"{o}({meta.loc[o,'form'][0]}/{meta.loc[o,'mat'][:4]}):{best_fig[o]}" for o in objs))

    print("\n== 2a) Фигура по геометрии (обучаемый выбор): для каждого объекта предсказываем «лучшую фигуру» ==")
    X = geo_shape_feats(d, L, W, H, S)
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    pred_g = np.zeros(len(d))
    for o in objs:
        tr, te = ids != o, ids == o
        # метка фигуры для train-объектов — их «оракульная» фигура, посчитанная БЕЗ объекта o
        lab = {}
        for q in sorted(set(ids[tr])):
            trq, teq = tr & (ids != q), ids == q
            lab[q] = max(FIG, key=lambda f: Vs(FV[f].values[teq] * k_for(f, trq), g[teq]).mean())
        y = np.array([lab[q] for q in ids[tr]])
        sc = StandardScaler().fit(X[tr])
        if len(set(y)) == 1:
            fig = np.array([y[0]] * te.sum())
        else:
            clf = LogisticRegression(C=0.3, max_iter=2000, class_weight="balanced").fit(sc.transform(X[tr]), y)
            fig = clf.predict(sc.transform(X[te]))
        pred_g[te] = [FV[f].values[i] * k_for(f, tr) for f, i in zip(fig, np.where(te)[0])]
    print(f"  V = {Vs(pred_g, g).mean():.3f}")

    print("\n== 2b) Фигура по картинке zero-shot CLIP/SigLIP ==")
    import open_clip
    import torch
    F2 = dict(np.load(ROOT / "audit/cache/features2.npz"))
    meta_all = json.load(open(ROOT / "audit/cache/meta.json"))
    key = {(m["frame"], m["oid"]): i for i, m in enumerate(meta_all)}
    ridx = np.array([key[(f, o)] for f, o in zip(d.frame, d.object_id)])
    prompts = {"box": ["a box", "a rectangular block", "a brick-shaped object", "a cardboard box"],
               "cylinder": ["a bottle lying on its side", "a can lying on the floor", "a cylinder", "a tube"],
               "ellipsoid": ["a rock", "a round lump", "a crumpled ball", "an egg-shaped object"],
               "prism": ["a wedge", "a triangular object", "a triangular prism"],
               "pile": ["a pile of small pieces", "a heap of debris", "a mound of loose material"],
               "flat": ["a flat bag lying on the floor", "a flat sheet", "a thin flat object", "a flat package"]}
    tmpl = ["a photo of {} on the floor.", "{} lying on a tiled floor.", "a photo of {}."]
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    logits = 0
    for tag, (name, pre) in {"clipL": ("ViT-L-14", "dfn2b"), "siglip": ("ViT-B-16-SigLIP", "webli")}.items():
        model, _, _ = open_clip.create_model_and_transforms(name, pretrained=pre, device=dev)
        tok = open_clip.get_tokenizer(name)
        with torch.no_grad():
            T = []
            for f in prompts:
                e = model.encode_text(tok([t.format(p) for p in prompts[f] for t in tmpl]).to(dev)).float()
                e = torch.nn.functional.normalize(e, dim=-1).mean(0)
                T.append(torch.nn.functional.normalize(e, dim=0).cpu().numpy())
        T = np.stack(T)
        for v in ("c14", "m10"):
            logits = logits + F2[f"{tag}_{v}"][ridx] @ T.T
        del model
    zs_fig = np.array(list(prompts))[np.asarray(logits).argmax(1)]
    pred_z = np.zeros(len(d))
    for o in objs:
        tr, te = ids != o, ids == o
        pred_z[te] = [FV[f].values[i] * k_for(f, tr) for f, i in zip(zs_fig[te], np.where(te)[0])]
    print(f"  V = {Vs(pred_z, g).mean():.3f}   распределение фигур: {dict(Counter(zs_fig))}")
    agree = np.mean([zs_fig[ids == o][0] == best_fig[o] for o in objs])
    print(f"  совпадение zero-shot фигуры с «оракульной»: {agree:.0%} предметов")

    print("\n== 3) Гибрид: объёмы по фигурам как признаки калибратора (LOOO, alpha во внутреннем CV) ==")
    base = ["columns_ml", "parts_ml", "hull_ml", "footprint_ml"]
    Xb = design(d, base, inter="parts_ml", knots=(50, 300))
    zs_onehot = np.stack([(zs_fig == f).astype(float) for f in prompts], 1)
    figlog = np.log(FV[[f for f in FIG if f != "columns"]].values.clip(min=1))
    variants = {"калибратор volume/ (база)": Xb,
                "+ объёмы по формулам": np.c_[Xb, figlog],
                "+ фигура zero-shot (one-hot)": np.c_[Xb, zs_onehot],
                "+ объём по zero-shot фигуре": np.c_[Xb, np.log(np.array([FV[f].values[i] for i, f in enumerate(zs_fig)]).clip(min=1))]}
    y = np.log(g)
    w0 = w_of(ids)
    for name, Xv in variants.items():
        pred = np.zeros(len(d))
        for o in objs:
            tr = ids != o
            ut = np.array(sorted(set(ids[tr])))
            np.random.default_rng(0).shuffle(ut)
            fo = {u: k % 5 for k, u in enumerate(ut)}
            fold = np.array([fo[u] for u in ids[tr]])
            best = None
            for a in (1, 3, 10, 30, 100):
                p = np.zeros(tr.sum())
                for k in range(5):
                    p[fold == k] = ridge(Xv[tr][fold != k], y[tr][fold != k], w0[tr][fold != k], a)(Xv[tr][fold == k])
                s = np.sum(w0[tr] * np.maximum(0, 1 - np.abs(p - y[tr]) / L3)) / w0[tr].sum()
                if best is None or s > best[0]:
                    best = (s, a)
            pred[~tr] = ridge(Xv[tr], y[tr], w0[tr], best[1])(Xv[~tr])
        v = Vs(np.exp(pred), g)
        print(f"  {name:32s} V={v.mean():.3f}  bulk {v[d.form == 'bulk'].mean():.2f} solid {v[d.form == 'solid'].mean():.2f}", flush=True)


if __name__ == "__main__":
    main()
