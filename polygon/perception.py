"""Run the project's real perception pipeline on what the simulated robot's D435 records.

The frame is rendered with the dataset intrinsics (camera.json) at the robot camera pose, gets the
RealSense depth model, and is written with a dataset-style meta.json (floor_plane in camera
coordinates). Then inference.predictor.predict_frame runs the detector, the attribute model and
the parts-based volume module exactly as for a submission.
"""
from __future__ import annotations

import json
import sys
import threading
import time
from pathlib import Path

import cv2
import numpy as np
import torch

from .fastrt import FastCam, render, to_u8
from .sensor import simulate_depth

REPO = Path(__file__).resolve().parents[1]
RUNS = Path.home() / "Cleaner/checkpoints"


class Perception:
    def __init__(self, run="score83_cv_fold2_noval18", experiment="C_joint", attributes="attributes_rgb_only.pt",
                 calibrator=REPO / "checkpoints/volume_parts/calibrator.json", data=REPO / "datasets/raw",
                 workdir=Path("/dev/shm/polygon_live"), min_score=0.25):
        sys.path.insert(0, str(REPO.parent))
        pkg = __import__(f"{REPO.name}.inference.predictor", fromlist=["predict_frame"])
        attr = __import__(f"{REPO.name}.training.attributes", fromlist=["load_attribute_model"])
        vb = __import__(f"{REPO.name}.inference.volume_backend", fromlist=["load_calibrator"])
        from ultralytics import YOLO
        wdir = RUNS / run / "checkpoints/experiments" / experiment
        self.predict_frame = pkg.predict_frame
        self.attr_model, self.ckpt, self.device = attr.load_attribute_model(wdir / attributes, "cuda:0")
        self.detector = YOLO(str(wdir / "detector.pt"))
        self.calibrator = vb.load_calibrator(calibrator)
        self.densities = vb.load_densities(data / "densities.csv")
        self.camera = json.loads((data / "camera.json").read_text())
        self.workdir = workdir / "frame"
        self.workdir.mkdir(parents=True, exist_ok=True)
        self.min_score = min_score
        self.name = f"{experiment} {run} + {attributes.replace('.pt', '')} + volume parts"
        self.latest = None            # dict: time, latency_ms, frame (BGR), detections, pose
        self.busy = False
        self.lock = threading.Lock()

    def capture(self, sc, eye, target, props, frame_idx, floor):
        """Render the D435 frame (call from the render thread). Returns inputs for run()."""
        cam = FastCam.from_pose(self.camera, eye, target)
        r = render(sc, cam, floor=floor, spp=2, seed=frame_idx)
        img = to_u8(r["rgb"])
        depth_true = r["depth"].cpu().numpy()
        ids = r["id"].cpu().numpy()
        spec = r["spec"].cpu().numpy()
        return cam, img, depth_true, ids, spec, props

    def run(self, cam, img, depth_true, ids, spec, props, rng):
        """Sensor model + real pipeline (call from a worker thread)."""
        t0 = time.time()
        floor_mm = np.where(ids < 0, depth_true * 1000, 0).astype(np.float32)
        sig = 1.3 * (floor_mm / 550.0) ** 2
        noisy_floor = floor_mm + sig * rng.standard_normal(floor_mm.shape).astype(np.float32)
        noisy_floor[(floor_mm == 0) | (rng.random(floor_mm.shape) < 0.01)] = 0
        bg_u16 = np.clip(np.round(noisy_floor), 0, 65535).astype(np.uint16)
        depth = simulate_depth(depth_true, bg_u16, ids, spec, props, cam.fx, 1.3, rng)
        noise = rng.normal(0, 2.2, img.shape)
        frame = np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)
        frame = cv2.GaussianBlur(frame, (0, 0), 0.6)
        cv2.imwrite(str(self.workdir / "color.png"), frame)
        cv2.imwrite(str(self.workdir / "depth.png"), depth)
        (self.workdir / "meta.json").write_text(json.dumps({"floor_plane": cam.floor_plane()}))
        preds = self.predict_frame(self.workdir / "color.png", self.detector, self.attr_model, self.ckpt, self.device,
                                   self.camera, conf=0.08, volume_backend="parts", volume_calibrator=self.calibrator,
                                   densities=self.densities)
        dets = []
        for p in preds:
            if p["score"] < self.min_score:
                continue
            m = p["mask"]
            ys, xs = np.nonzero(m & (depth_true > 0))
            if len(xs) < 20:
                continue
            z = depth_true[ys, xs]
            dc = np.stack([(xs - cam.cx) / cam.fx * z, (ys - cam.cy) / cam.fy * z, z], 1)
            w = dc @ cam.R.T + cam.origin
            pos = np.median(w, 0)
            dets.append({"box": p["box"], "mask": m, "score": p["score"], "form": p["form"], "material": p["material"],
                         "volume": p["volume_ml"], "mass": p["mass_g"], "take": int(p["take"]), "world": pos,
                         "top": float(np.percentile(w[:, 2], 95))})
        lat = (time.time() - t0) * 1000
        with self.lock:
            self.latest = {"time": time.time(), "latency_ms": lat, "frame": frame, "dets": dets}
        return dets


def overlay(frame, dets, W, H):
    """Model view: the exact frame the model saw, with its masks, boxes and decisions."""
    img = frame.copy()
    for d in dets:
        col = (90, 210, 110) if d["take"] else (70, 70, 240)
        tint = np.zeros_like(img)
        tint[d["mask"]] = col
        img = cv2.addWeighted(img, 1.0, tint, 0.35, 0)
        x1, y1, x2, y2 = (int(v) for v in d["box"])
        cv2.rectangle(img, (x1, y1), (x2, y2), col, 2)
        txt = f"{d['material']} {d['volume']:.0f} ml {d['mass']:.0f} g {'TAKE' if d['take'] else 'AVOID'} {d['score']:.2f}"
        (tw, th), _ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
        cv2.rectangle(img, (x1, max(0, y1 - th - 10)), (x1 + tw + 8, y1), (22, 22, 24), -1)
        cv2.putText(img, txt, (x1 + 4, y1 - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (240, 240, 240), 1, cv2.LINE_AA)
    return cv2.resize(img, (W, H), interpolation=cv2.INTER_AREA)
