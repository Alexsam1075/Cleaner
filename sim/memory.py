"""«Живая» память: ближайшие соседи в пространстве эмбеддингов физической головы.
Новый объект добавляется в память без переобучения сети. Предсказание = смесь головы сети и голосования соседей,
вес соседей растёт с их близостью. Объект далеко от всех — помечается как «неизвестный» (запрос на метку).

Эксперимент (python memory.py): поток реальных кадров с новыми предметами (val-объекты, которых модель не видела).
Предметы «встречаются» по очереди; после первой встречи робот получает обратную связь (вес на захвате -> take,
материал/объём от оператора или датчиков) и кладёт эмбеддинг в память. Сравнивается качество на следующих кадрах
этих предметов: замороженная модель vs модель с памятью."""
import argparse
import json
import math
from collections import defaultdict

import numpy as np
import torch
from torch.utils.data import DataLoader

from data import RealFrames, collate
from geom import ROOT
from model import TwoHeadNet, masks_from_inst
from scenes import object_split
from train import DEV


class Memory:
    def __init__(self, k=5, tau=0.07, novelty=0.75):
        self.E, self.Y = [], []
        self.k, self.tau, self.novelty = k, tau, novelty

    def add(self, emb, y):
        self.E.append(np.asarray(emb, np.float32))
        self.Y.append(np.asarray(y, np.float32))

    def query(self, emb):
        if not self.E:
            return None
        E = np.stack(self.E)
        Y = np.stack(self.Y)
        sim = E @ emb
        idx = np.argsort(-sim)[: self.k]
        w = np.exp((sim[idx] - sim[idx].max()) / self.tau)
        w /= w.sum()
        mat = np.bincount(Y[idx, 1].astype(int), weights=w, minlength=6)
        form = np.bincount(Y[idx, 0].astype(int), weights=w, minlength=2)
        return {"sim": float(sim[idx[0]]), "mat": mat, "form": form, "vol": float((w * Y[idx, 2]).sum()),
                "take": float((w * Y[idx, 3]).sum()), "novel": bool(sim[idx[0]] < self.novelty)}


def fuse(net_out, mem, alpha_max=0.9):
    if mem is None:
        return net_out
    a = alpha_max * float(np.clip((mem["sim"] - 0.5) / 0.4, 0, 1))
    return {"form": (1 - a) * net_out["form"] + a * mem["form"], "mat": (1 - a) * net_out["mat"] + a * mem["mat"],
            "vol": (1 - a) * net_out["vol"] + a * mem["vol"], "take": (1 - a) * net_out["take"] + a * mem["take"]}


@torch.no_grad()
def embed_all(net, ds):
    rows = []
    for b in DataLoader(ds, batch_size=8, collate_fn=collate):
        out = net(b["x"].to(DEV))
        r = net.objects(out, masks_from_inst(b["inst"].to(DEV)))
        if r is None:
            continue
        A = [t for t in b["attrs"] if len(t)]
        oids = [o for t, oo in zip(b["attrs"], b["object_ids"]) if len(t) for o in oo]
        ids = [i for t, i in zip(b["attrs"], b["id"]) if len(t) for _ in range(len(t))]
        A = torch.cat(A).numpy()
        for j in range(len(A)):
            rows.append({"id": ids[j], "oid": oids[j], "y": A[j], "emb": r["emb"][j].cpu().numpy(),
                         "net": {"form": torch.softmax(r["form"][j], 0).cpu().numpy(), "mat": torch.softmax(r["mat"][j], 0).cpu().numpy(),
                                 "vol": float(r["vol"][j]), "take": float(torch.sigmoid(r["take"][j]))}})
    return rows


def score(preds, ys):
    P = np.array([[p["form"].argmax(), p["mat"].argmax(), p["vol"], p["take"]] for p in preds])
    Y = np.array(ys)
    per_mat = [float((P[Y[:, 1] == m, 1] == m).mean()) for m in range(6) if (Y[:, 1] == m).any()]
    V = float(np.clip(1 - np.abs(P[:, 2] - Y[:, 2]) / math.log(3), 0, 1).mean())
    tk = (P[:, 3] > 0.5).astype(int)
    S = float(np.mean([(tk[Y[:, 3] == v] == v).mean() for v in (0, 1) if (Y[:, 3] == v).any()]))
    return {"form": float((P[:, 0] == Y[:, 0]).mean()), "M": float(np.mean(per_mat)), "V": V, "S": S, "n": len(Y)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=str(ROOT / "sim/checkpoints/twohead.pt"))
    ap.add_argument("--shots", type=int, default=1)
    a = ap.parse_args()
    ck = torch.load(a.ckpt, map_location=DEV, weights_only=False)
    net = TwoHeadNet(pretrained=False).to(DEV)
    net.load_state_dict(ck["model"])
    net.eval()
    tr_ids, va_ids = object_split()
    train_rows = embed_all(net, RealFrames(set(tr_ids)))
    val_rows = embed_all(net, RealFrames(set(va_ids)))
    mem = Memory()
    for r in train_rows:
        mem.add(r["emb"], r["y"])
    by_obj = defaultdict(list)
    for r in val_rows:
        by_obj[r["oid"]].append(r)
    rng = np.random.default_rng(0)
    order = list(by_obj)
    rng.shuffle(order)
    frozen, live, ys, curve = [], [], [], []
    novel_flags = {"first": [], "later": []}
    for step, oid in enumerate(order):
        rows = by_obj[oid]
        rng.shuffle(rows)
        shots, rest = rows[: a.shots], rows[a.shots:]
        for r in shots:
            q = mem.query(r["emb"])
            novel_flags["first"].append(q["novel"])
            mem.add(r["emb"], r["y"])
        for r in rest:
            q = mem.query(r["emb"])
            novel_flags["later"].append(q["novel"])
            frozen.append(r["net"])
            live.append(fuse(r["net"], q))
            ys.append(r["y"])
        if ys:
            curve.append({"objects_seen": step + 1, "frozen": score(frozen, ys), "live": score(live, ys)})
    res = {"frozen": score(frozen, ys), "live": score(live, ys),
           "novelty_rate_first_encounter": float(np.mean(novel_flags["first"])),
           "novelty_rate_after_memory": float(np.mean(novel_flags["later"])) if novel_flags["later"] else None,
           "curve": curve}
    print(json.dumps({k: v for k, v in res.items() if k != "curve"}, indent=1, ensure_ascii=False))
    (ROOT / "sim/out/memory_result.json").write_text(json.dumps(res, indent=1, default=float), encoding="utf-8")


if __name__ == "__main__":
    main()
