"""Обучение модели с двумя головами на полигоне (sim) + реальных кадрах (только физическая голова).
Валидация честная: предметы из val-части (каждый 5-й object_id) не участвуют ни в рендере train-сцен, ни в real train.
Отчёт: навигация (IoU классов, MAE slip) на sim/val; физика (форма, материал, объём, take) на real val (новые предметы)."""
import argparse
import json
import math
import random

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import ConcatDataset, DataLoader

from data import MATERIALS, RealFrames, SimFrames, collate
from geom import ROOT
from model import TwoHeadNet, masks_from_inst
from scenes import object_split

DEV = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
NAV_W = torch.tensor([1.0, 3.0, 3.0, 3.0])


def losses(net, batch, has_nav):
    x = batch["x"].to(DEV)
    out = net(x)
    L = {}
    nav = batch["nav"].to(DEV)
    lg = F.interpolate(out["nav_logits"], size=nav.shape[-2:], mode="bilinear", align_corners=False)
    sl = F.interpolate(out["slip"][:, None], size=nav.shape[-2:], mode="bilinear", align_corners=False)[:, 0]
    hn = has_nav.to(DEV)
    if hn.any():
        L["nav"] = F.cross_entropy(lg[hn], nav[hn], weight=NAV_W.to(DEV))
        L["slip"] = F.l1_loss(sl[hn], batch["slip"].to(DEV)[hn]) * 5
    if (~hn).any():
        objm = nav[~hn] > 0
        if objm.any():
            L["nav_real"] = F.cross_entropy(lg[~hn].permute(0, 2, 3, 1)[objm], nav[~hn][objm], weight=NAV_W.to(DEV)) * 0.5
    masks = masks_from_inst(batch["inst"].to(DEV))
    r = net.objects(out, masks)
    if r is not None:
        a = torch.cat([t for t in batch["attrs"] if len(t)]).to(DEV)
        mat_w = torch.ones(len(MATERIALS), device=DEV)
        L["form"] = F.cross_entropy(r["form"], a[:, 0].long())
        L["mat"] = F.cross_entropy(r["mat"], a[:, 1].long(), weight=mat_w)
        L["vol"] = F.smooth_l1_loss(r["vol"], a[:, 2]) * 0.5
        L["take"] = F.binary_cross_entropy_with_logits(r["take"], a[:, 3], pos_weight=torch.tensor(0.3, device=DEV))
    return L


class Tagged(torch.utils.data.Dataset):
    def __init__(self, ds, nav):
        self.ds, self.nav = ds, nav

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, i):
        d = dict(self.ds[i])
        d.pop("id", None)
        d.pop("object_ids", None)
        d["has_nav"] = self.nav
        return d


def collate_t(b):
    out = collate(b)
    out["has_nav"] = torch.tensor([x["has_nav"] for x in b])
    return out


@torch.no_grad()
def evaluate(net, sim_val, real_val):
    net.eval()
    res = {}
    if sim_val is not None and len(sim_val):
        inter = np.zeros(4)
        union = np.zeros(4)
        mae = []
        for b in DataLoader(sim_val, batch_size=8, collate_fn=collate):
            out = net(b["x"].to(DEV))
            nav = b["nav"].to(DEV)
            p = F.interpolate(out["nav_logits"], size=nav.shape[-2:], mode="bilinear").argmax(1)
            for c in range(4):
                inter[c] += ((p == c) & (nav == c)).sum().item()
                union[c] += ((p == c) | (nav == c)).sum().item()
            sl = F.interpolate(out["slip"][:, None], size=nav.shape[-2:], mode="bilinear")[:, 0]
            mae.append((sl - b["slip"].to(DEV)).abs().mean().item())
        res["nav_iou"] = dict(zip(["floor", "heavy_surface", "avoid", "take"], (inter / np.maximum(union, 1)).round(3).tolist()))
        res["slip_mae"] = float(np.mean(mae))
    if real_val is not None:
        P, A = [], []
        for b in DataLoader(real_val, batch_size=8, collate_fn=collate):
            out = net(b["x"].to(DEV))
            r = net.objects(out, masks_from_inst(b["inst"].to(DEV)))
            if r is None:
                continue
            P.append(torch.stack([r["form"].argmax(1).float(), r["mat"].argmax(1).float(), r["vol"], torch.sigmoid(r["take"])], 1).cpu())
            A.append(torch.cat([t for t in b["attrs"] if len(t)]))
        P, A = torch.cat(P).numpy(), torch.cat(A).numpy()
        per_mat = [float((P[A[:, 1] == m, 1] == m).mean()) for m in range(6) if (A[:, 1] == m).any()]
        V = np.clip(1 - np.abs(P[:, 2] - A[:, 2]) / math.log(3), 0, 1).mean()
        tk = (P[:, 3] > 0.5).astype(int)
        parts = [float((tk[A[:, 3] == v] == v).mean()) for v in (0, 1) if (A[:, 3] == v).any()]
        res.update({"form_acc": float((P[:, 0] == A[:, 0]).mean()), "M": float(np.mean(per_mat)), "V": float(V), "S": float(np.mean(parts)),
                    "n_obj": int(len(A))})
    net.train()
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--bs", type=int, default=8)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--out", default=str(ROOT / "sim/checkpoints/twohead.pt"))
    a = ap.parse_args()
    torch.manual_seed(0)
    random.seed(0)
    tr_ids, va_ids = object_split()
    sim_tr = SimFrames("train", train=True)
    sim_va = SimFrames("val", train=False)
    real_tr = RealFrames(set(tr_ids))
    real_va = RealFrames(set(va_ids))
    print(f"sim train {len(sim_tr)} | sim val {len(sim_va)} | real train {len(real_tr)} | real val {len(real_va)} (новые предметы: {va_ids})")
    train = ConcatDataset([Tagged(sim_tr, True), Tagged(real_tr, False), Tagged(real_tr, False)])
    dl = DataLoader(train, batch_size=a.bs, shuffle=True, collate_fn=collate_t, num_workers=4, persistent_workers=True)
    net = TwoHeadNet().to(DEV)
    opt = torch.optim.AdamW(net.parameters(), lr=a.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=a.lr, total_steps=a.epochs * len(dl), pct_start=0.1)
    hist = []
    for ep in range(a.epochs):
        agg = {}
        for b in dl:
            L = losses(net, b, b["has_nav"])
            loss = sum(L.values())
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 5.0)
            opt.step()
            sched.step()
            for k, v in L.items():
                agg.setdefault(k, []).append(v.item())
        ev = evaluate(net, sim_va, real_va)
        row = {"epoch": ep + 1, **{k: round(float(np.mean(v)), 4) for k, v in agg.items()}, **ev}
        hist.append(row)
        print(json.dumps(row, ensure_ascii=False), flush=True)
    import pathlib
    pathlib.Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": net.state_dict(), "hist": hist, "val_ids": va_ids}, a.out)
    print("saved", a.out)


if __name__ == "__main__":
    main()
