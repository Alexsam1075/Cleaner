"""Датасеты для модели с двумя головами: кадры полигона (sim/data) и реальные кадры датасета (для проверки).
Вход: RGB + высота над полом (из глубины и плоскости пола), разрешение 640x360.
Метки навигации (по пикселю): 0 пол-ок, 1 тяжёлая поверхность (ковёр), 2 препятствие/объехать, 3 забрать.
Метки физики (по объекту): форма, материал, log(объём), take."""
import json
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from geom import CAM, DATA, ROOT, floor_frame

FORMS = ["bulk", "solid"]
MATERIALS = ["metal", "wood", "plastic", "glass", "paper", "mineral"]
SURF_SLIP = {0: 0.05, 1: 0.06, 2: 0.40}
W, H = 640, 360


def height_map(depth_m, plane, scale):
    h, w = depth_m.shape
    v, u = np.mgrid[0:h, 0:w].astype(np.float32)
    fx, fy, cx, cy = CAM["fx"] * scale, CAM["fy"] * scale, CAM["cx"] * scale, CAM["cy"] * scale
    z = depth_m.astype(np.float32)
    P = np.stack([(u - cx) / fx * z, (v - cy) / fy * z, z], -1)
    R, t = floor_frame(plane)
    hgt = P @ R[2].astype(np.float32) + np.float32(t[2])
    hgt[depth_m <= 0] = 0
    return np.clip(hgt, -0.05, 0.4), (depth_m > 0).astype(np.float32)


def to_tensor(rgb, hgt, valid):
    x = np.concatenate([rgb.astype(np.float32) / 255.0, hgt[..., None] * 5.0, valid[..., None]], -1)
    return torch.from_numpy(x).permute(2, 0, 1).contiguous()


def augment(rgb, rng):
    rgb = rgb.astype(np.float32)
    rgb = rgb * rng.uniform(0.75, 1.25) + rng.uniform(-20, 20)
    rgb = rgb * rng.uniform(0.9, 1.1, 3)
    if rng.random() < 0.5:
        rgb = cv2.GaussianBlur(rgb, (0, 0), rng.uniform(0.3, 1.2))
    rgb = rgb + rng.normal(0, rng.uniform(1, 6), rgb.shape)
    return np.clip(rgb, 0, 255).astype(np.uint8)


def depth_noise(z, rng, holes_mask=None):
    z = z * (1 + rng.normal(0, 0.003, z.shape)).astype(np.float32) + (rng.normal(0, 0.0015, z.shape) * z ** 2).astype(np.float32)
    drop = rng.random(z.shape) < rng.uniform(0.02, 0.1)
    drop = cv2.dilate(drop.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
    z[drop] = 0
    if holes_mask is not None:
        z[holes_mask & (rng.random(z.shape) < 0.7)] = 0
    return z


class SimFrames(Dataset):
    def __init__(self, split, root=ROOT / "sim/data", train=True):
        from run_blender import read_frame
        self.read = read_frame
        self.items = []
        for sd in sorted((Path(root) / split).glob("scene_*")):
            if not (sd / "scene.json").exists():
                continue
            spec = json.loads((sd / "scene.json").read_text())
            for i, f in enumerate(sorted(sd.glob("f_*"))):
                self.items.append((f, spec, i))
        self.train = train

    def __len__(self):
        return len(self.items)

    def __getitem__(self, k):
        f, spec, i = self.items[k]
        rng = np.random.default_rng(None if self.train else k)
        rgb, z, idx = self.read(f)
        z = z.copy()
        z[z > 20] = 0
        pose = spec["camera"]["poses"][i]
        R = np.array(pose["R"])
        t = np.array(pose["t"])
        n_cam = R[2]
        plane = [*n_cam, t[2]]
        objs = spec["objects"]
        glassy = np.zeros(idx.shape, bool)
        for j, o in enumerate(objs, start=1):
            if o["material"] in ("glass", "metal"):
                glassy |= idx == j
        if self.train:
            rgb = augment(rgb, rng)
            z = depth_noise(z, rng, glassy)
        hgt, valid = height_map(z, plane, spec.get("scale", 0.5))
        x = to_tensor(rgb, hgt, valid)
        nav = np.zeros(idx.shape, np.int64)
        slip = np.full(idx.shape, SURF_SLIP[0], np.float32)
        for sid, sv in SURF_SLIP.items():
            slip[idx == 100 + sid] = sv
        nav[idx == 102] = 1
        nav[idx == 99] = 2
        inst = np.zeros(idx.shape, np.int64)
        attrs = []
        for j, o in enumerate(objs, start=1):
            m = idx == j
            if m.sum() < 30:
                continue
            nav[m] = 3 if o["take"] else 2
            slip[m] = 0.0
            inst[m] = len(attrs) + 1
            attrs.append([FORMS.index(o["form"]), MATERIALS.index(o["material"]), np.log(max(o["volume_ml"], 1.0)),
                          float(o["take"]), float(o.get("unknown", False))])
        slip[idx == 99] = 0.0
        return {"x": x, "nav": torch.from_numpy(nav), "slip": torch.from_numpy(slip),
                "inst": torch.from_numpy(inst), "attrs": torch.tensor(attrs, dtype=torch.float32).reshape(-1, 5)}


def polygon_mask(seg, h, w):
    m = np.zeros((h, w), np.uint8)
    for s in seg:
        cv2.fillPoly(m, [np.array(s, np.float32).reshape(-1, 2).round().astype(np.int32)], 1)
    return m > 0


class RealFrames(Dataset):
    """Реальные кадры: только физическая голова (атрибуты объектов) и навигационная метка объекта."""

    def __init__(self, object_ids=None, labels="real_train", include_empty=False):
        d = json.loads((DATA / "labels" / f"{labels}.json").read_text(encoding="utf-8"))
        anns = {}
        for a in d["annotations"]:
            anns.setdefault(a["image_id"], []).append(a)
        self.items = []
        for im in d["images"]:
            aa = anns.get(im["id"], [])
            if object_ids is not None:
                aa = [a for a in aa if a["attributes"]["object_id"] in object_ids]
                if not aa:
                    continue
            if not aa and not include_empty:
                continue
            self.items.append((DATA / Path(im["file_name"]).parent, im, aa))

    def __len__(self):
        return len(self.items)

    def __getitem__(self, k):
        fd, im, aa = self.items[k]
        rgb = cv2.resize(cv2.imread(str(fd / "color.png")), (W, H), interpolation=cv2.INTER_AREA)
        depth = cv2.imread(str(fd / "depth.png"), cv2.IMREAD_UNCHANGED).astype(np.float32) / 1000.0
        depth = cv2.resize(depth, (W, H), interpolation=cv2.INTER_NEAREST)
        plane = json.loads((fd / "meta.json").read_text())["floor_plane"]
        hgt, valid = height_map(depth, plane, W / CAM["width"])
        inst = np.zeros((H, W), np.int64)
        nav = np.zeros((H, W), np.int64)
        attrs = []
        for a in aa:
            m = polygon_mask(a["segmentation"], im["height"], im["width"])
            m = cv2.resize(m.astype(np.uint8), (W, H), interpolation=cv2.INTER_NEAREST) > 0
            at = a["attributes"]
            inst[m] = len(attrs) + 1
            nav[m] = 3 if at["take"] else 2
            attrs.append([FORMS.index(at["form"]), MATERIALS.index(at["material"]), np.log(at["volume_ml"]), float(at["take"]), 0.0])
        return {"x": to_tensor(rgb, hgt, valid), "nav": torch.from_numpy(nav), "slip": torch.zeros(H, W),
                "inst": torch.from_numpy(inst), "attrs": torch.tensor(attrs, dtype=torch.float32).reshape(-1, 5),
                "id": fd.name, "object_ids": [a["attributes"]["object_id"] for a in aa]}


def collate(batch):
    out = {k: torch.stack([b[k] for b in batch]) for k in ("x", "nav", "slip", "inst")}
    out["attrs"] = [b["attrs"] for b in batch]
    if "id" in batch[0]:
        out["id"] = [b["id"] for b in batch]
        out["object_ids"] = [b["object_ids"] for b in batch]
    return out
