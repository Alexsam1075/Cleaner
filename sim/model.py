"""Модель с общим RGB-D энкодером и двумя головами.
Навигационная голова (по пикселю): класс проходимости {пол, тяжёлая поверхность, объехать, забрать} + прогноз slip.
Физическая голова (по объекту, пулинг признаков по маске): форма, материал, log(объём), take + эмбеддинг для памяти."""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision


class Encoder(nn.Module):
    def __init__(self, in_ch=5, pretrained=True):
        super().__init__()
        r = torchvision.models.resnet18(weights="IMAGENET1K_V1" if pretrained else None)
        w = r.conv1.weight.data
        self.conv1 = nn.Conv2d(in_ch, 64, 7, 2, 3, bias=False)
        with torch.no_grad():
            self.conv1.weight[:, :3] = w
            self.conv1.weight[:, 3:] = w.mean(1, keepdim=True) * 0.5
        self.stem = nn.Sequential(self.conv1, r.bn1, r.relu, r.maxpool)
        self.l1, self.l2, self.l3, self.l4 = r.layer1, r.layer2, r.layer3, r.layer4
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406, 0.0, 0.5]).view(1, 5, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225, 1.0, 0.5]).view(1, 5, 1, 1))

    def forward(self, x):
        x = (x - self.mean) / self.std
        x = self.stem(x)
        c1 = self.l1(x)
        c2 = self.l2(c1)
        c3 = self.l3(c2)
        c4 = self.l4(c3)
        return c1, c2, c3, c4


class FPN(nn.Module):
    def __init__(self, chs=(64, 128, 256, 512), out=128):
        super().__init__()
        self.lat = nn.ModuleList(nn.Conv2d(c, out, 1) for c in chs)
        self.smooth = nn.Sequential(nn.Conv2d(out, out, 3, padding=1), nn.BatchNorm2d(out), nn.ReLU(inplace=True))

    def forward(self, feats):
        p = self.lat[3](feats[3])
        for i in (2, 1, 0):
            p = F.interpolate(p, size=feats[i].shape[-2:], mode="bilinear", align_corners=False) + self.lat[i](feats[i])
        return self.smooth(p)


class TwoHeadNet(nn.Module):
    def __init__(self, n_nav=4, n_forms=2, n_mat=6, emb=128, pretrained=True):
        super().__init__()
        self.enc = Encoder(pretrained=pretrained)
        self.fpn = FPN()
        self.nav = nn.Sequential(nn.Conv2d(128, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True), nn.Conv2d(64, n_nav + 1, 1))
        self.obj_proj = nn.Sequential(nn.Linear(128 + 512 + 6, 256), nn.ReLU(inplace=True), nn.Dropout(0.2), nn.Linear(256, emb))
        self.form = nn.Linear(emb, n_forms)
        self.mat = nn.Linear(emb, n_mat)
        self.vol = nn.Linear(emb, 1)
        self.take = nn.Linear(emb, 1)

    def forward(self, x):
        feats = self.enc(x)
        p = self.fpn(feats)
        nav = self.nav(p)
        return {"p": p, "c4": feats[3], "nav_logits": nav[:, :-1], "slip": torch.sigmoid(nav[:, -1]), "x": x}

    @staticmethod
    def geometry(x, mask):
        """Геометрические признаки объекта из канала высоты (в метрах ×5) и маски на полном разрешении."""
        h = x[3] / 5.0
        v = x[4]
        m = mask & (v > 0)
        if m.sum() < 5:
            return torch.zeros(6, device=x.device)
        hv = h[m].clamp(min=0)
        area = mask.float().mean()
        return torch.stack([hv.mean() * 10, hv.max() * 10, torch.quantile(hv, 0.9) * 10, area * 10,
                            1 - m.float().sum() / mask.float().sum().clamp(min=1), torch.log1p(mask.float().sum()) / 10])

    def objects(self, out, masks_per_image):
        """masks_per_image: список (для каждого изображения) тензоров масок [K, H, W] на разрешении входа."""
        embs = []
        for b, masks in enumerate(masks_per_image):
            if len(masks) == 0:
                continue
            p, c4, x = out["p"][b], out["c4"][b], out["x"][b]
            mp = F.interpolate(masks[None].float(), size=p.shape[-2:], mode="area")[0]
            m4 = F.interpolate(masks[None].float(), size=c4.shape[-2:], mode="area")[0]
            fp = (mp[:, None] * p[None]).sum((2, 3)) / mp.sum((1, 2)).clamp(min=1e-3)[:, None]
            f4 = (m4[:, None] * c4[None]).sum((2, 3)) / m4.sum((1, 2)).clamp(min=1e-3)[:, None]
            g = torch.stack([self.geometry(x, m.bool()) for m in masks])
            embs.append(torch.cat([fp, f4, g], 1))
        if not embs:
            return None
        e = self.obj_proj(torch.cat(embs))
        return {"emb": F.normalize(e, dim=1), "form": self.form(e), "mat": self.mat(e), "vol": self.vol(e)[:, 0], "take": self.take(e)[:, 0]}


def masks_from_inst(inst):
    return [torch.stack([inst[b] == k for k in range(1, int(inst[b].max()) + 1)]) if inst[b].max() > 0 else []
            for b in range(inst.shape[0])]
