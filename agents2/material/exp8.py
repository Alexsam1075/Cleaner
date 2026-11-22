"""Эксперимент 8: глубина как «дыры» (стекло/металл) и шероховатость поверх финального метода; λ вложенно."""
import json

import numpy as np

from common import HERE, MATS, softmax, M_frame
from exp1 import W_GRID, fuse
from exp5 import Runner5


def m_depth(R, train_objs, te, feat="holes", lam_grid=(0.0, 0.25, 0.5, 1.0, 2.0), classes=("glass", "metal")):
    gk = list(R.F["geo_keys"])
    G = R.F["geo"]
    if feat == "holes":
        h = np.log1p(G[:, gk.index("holes_inner")] * 20) + np.log1p(G[:, gk.index("holes_edge")] * 5)
    elif feat == "holes_inner":
        h = np.log1p(G[:, gk.index("holes_inner")] * 20)
    else:
        h = np.log(np.maximum(G[:, gk.index("rough")], 1e-4))
    tr = np.where(np.isin(R.oid, train_objs) & R.real)[0]
    Z = softmax(R.L(("rich|floor",), ("c14", "c20"), 0.01) / 0.01)
    Pin = R.inner_oof(train_objs)
    mu, sd = h[tr].mean(), h[tr].std() + 1e-6
    z = (h - mu) / sd
    cls = [MATS.index(c) for c in classes]

    def adj(P, lam, zz):
        Lg = np.log(P + 1e-9)
        for c in cls:
            Lg[:, c] += lam * zz
        return softmax(Lg)

    best = max(((w, lam) for w in W_GRID for lam in lam_grid), key=lambda t: M_frame(adj(fuse(Pin, Z, t[0]), t[1], z), R.mat, tr))
    w, lam = best
    Pte = R.head_full(train_objs, te)
    return adj(fuse(Pte, Z[te], w), lam, z[te]), {"w": w, "lam": lam}


def main():
    R = Runner5()
    res = {}
    gk = list(R.F["geo_keys"])
    print("geo_keys:", gk)
    idx = R.idx
    for k in ("holes_inner", "holes_edge", "rough"):
        v = R.F["geo"][:, gk.index(k)]
        print(f"  {k}: " + ", ".join(f"{MATS[c]}={np.median(v[idx][R.mat[idx] == c]):.3f}" for c in range(6)))
    print("== Глубина поверх финала ==")
    _, res["holes_gm"] = R.run(m_depth, "дыры → glass,metal")
    _, res["holes_metal"] = R.run(m_depth, "дыры → metal", classes=("metal",))
    _, res["holes_inner_gm"] = R.run(m_depth, "дыры inner → glass,metal", feat="holes_inner")
    _, res["rough_metal_neg"] = R.run(m_depth, "шероховатость → metal/glass (знак подбирается λ∈±)", feat="rough", lam_grid=(-1.0, -0.5, 0.0, 0.5, 1.0))
    json.dump(res, open(HERE / "exp8.json", "w"), indent=1, default=str)


if __name__ == "__main__":
    main()
