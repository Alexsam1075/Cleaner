"""Shared constants/helpers for the form agent (workspace ~/Cleaner/agents/form)."""
import json
import math
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path("/home/cleaner/Cleaner")
W = ROOT / "agents/form"
C = ROOT / "audit/cache"
D = Path(os.environ.setdefault("ASTRAMIS_DATA", str(ROOT / "datasets/raw")))
sys.path.insert(0, str(ROOT / "audit"))
sys.path.insert(0, str(ROOT / "references"))
FORMS = ["bulk", "solid"]
VIEWS = {"m10": (1.1, True), "m14": (1.5, True), "c14": (1.5, False), "c20": (2.2, False)}
MODELS = {"clipL": ("ViT-L-14", "dfn2b"), "siglip": ("ViT-B-16-SigLIP", "webli")}
FOLDS = {
    1: (ROOT / "configs/object_id_split.json", ROOT / "checkpoints/experiments/D_domain_randomized/detector.pt"),
    2: (ROOT / "checkpoints/score83_cv_fold2_noval18/configs/object_id_split.json",
        ROOT / "checkpoints/score83_cv_fold2_noval18/checkpoints/experiments/C_joint/detector.pt"),
    3: (ROOT / "checkpoints/score83_cv_fold3_noval18/configs/object_id_split.json",
        ROOT / "checkpoints/score83_cv_fold3_noval18/checkpoints/experiments/C_joint/detector.pt"),
}
# zero-shot prompt sets (bulk list, solid list)
PROMPT_SETS = {
    "pile": (["a pile of small loose pieces", "scattered granular material", "a heap of many small items", "loose debris"],
             ["a single solid object", "one rigid item", "a single whole object", "one intact item"]),
    "many": (["many small objects", "several items", "multiple pieces", "a group of objects", "a bunch of things"],
             ["one object", "a single item", "one thing", "a single object"]),
    "count": (["a photo of several objects on the floor", "a photo of many pieces on the floor", "a photo of a pile on the floor",
               "a photo of scattered items on the floor"],
              ["a photo of one object on the floor", "a photo of a single item on the floor", "a photo of a single object lying on the floor"]),
    "mix": (["a pile of small loose pieces", "many small objects", "several items", "scattered granular material",
             "a heap of many small items", "multiple pieces", "a bunch of things"],
            ["a single solid object", "one rigid item", "one object", "a single item", "a single whole object"]),
    "multi": (["multiple separate pieces", "a collection of several loose parts", "items piled on top of each other", "a jumble of parts"],
              ["a single object", "one item", "a single piece"]),
    "crumpled": (["crumpled and scattered material", "a messy heap", "a heap of loose material", "a mess of pieces"],
                 ["a neat single object", "one clean rigid object", "a single intact item"]),
    "parts": (["spare parts", "components", "fragments", "pieces", "chips", "granules", "scraps"],
              ["a product", "a tool", "a container", "a box", "a bottle", "a block"]),
    "trash": (["a pile of trash", "scattered garbage", "spilled debris", "loose waste material"],
              ["a piece of trash", "a discarded object", "a single item of garbage"]),
    "matpile": (["a pile of screws", "a pile of stones", "a heap of sawdust", "a pile of wood chips", "a pile of plastic parts",
                 "several crumpled paper balls", "a stack of aluminium profiles", "a bunch of small bottles", "a pile of cables and plastic parts"],
                ["a screw", "a stone", "a wooden block", "a plastic bottle", "a cardboard box", "a crumpled paper ball", "an aluminium bar",
                 "a bottle", "a plastic part"]),
    "several": (["several", "many", "a few", "multiple", "lots of", "a pile of", "a heap of"], ["a single", "one", "a", "an individual", "a lone"]),
}
ZS_TAGS = ("clipL", "siglip")
ZS_VIEWS = ("c14", "m10", "m14")
TEMPL = ["{}", "a photo of {}.", "a close-up photo of {}.", "{} on a tiled floor."]


def l2n(X):
    return X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-8)


def load_split(fold):
    return json.load(open(FOLDS[fold][0]))


def gt_cache():
    F = dict(np.load(C / "features.npz"))
    F.update(dict(np.load(C / "features2.npz")))
    meta = json.load(open(C / "meta.json"))
    return F, meta


def box_iou(a, b):
    w = min(a[2], b[2]) - max(a[0], b[0])
    h = min(a[3], b[3]) - max(a[1], b[1])
    if w <= 0 or h <= 0:
        return 0.0
    inter = w * h
    return inter / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter)


def gt_by_frame():
    lab = json.load(open(D / "labels/real_train.json"))
    ims = {im["id"]: Path(im["file_name"]).parent.name for im in lab["images"]}
    out = {f: [] for f in ims.values()}
    for a in lab["annotations"]:
        x, y, w, h = a["bbox"]
        out[ims[a["image_id"]]].append({"box": [x, y, x + w, y + h], "form": a["attributes"]["form"], "oid": a["attributes"]["object_id"],
                                        "mat": a["attributes"]["material"]})
    return out
