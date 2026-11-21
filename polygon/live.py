"""Live polygon: real-time renderer streamed to a browser (MJPEG), driven by the project's real model.

  python -m polygon.live --port 8790          # on the GPU server
  ssh -L 8790:localhost:8790 ...              # on the laptop, then open http://localhost:8790

The robot's D435 frame (dataset intrinsics, RealSense depth model, dataset meta.json) goes through
inference.predictor.predict_frame: detector + attribute model + parts volume. The autopilot drives to
what the model calls TAKE and around what it calls AVOID; collisions with truly heavy trash are counted.

Mouse drag orbits, wheel zooms, W/A/S/D drive, P autopilot, V model view, G ground truth labels,
Space drops trash, R resets.
"""
from __future__ import annotations

import argparse
import json
import math
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import cv2
import numpy as np
import torch

from .brain import Brain
from .fast_scene import add_robot, random_trash, robot_camera
from .fastrt import FastCam, FloorTex, Scene, render, to_u8

TILE = Path(__file__).resolve().parent / "assets/floor_tile.png"
WHEEL_R, TRACK = 0.085, 0.27

PAGE = """<!doctype html><html lang="ru"><head><meta charset="utf-8"><title>Полигон в реальном времени</title>
<meta name="viewport" content="width=device-width, initial-scale=1">
<style>html,body{margin:0;height:100%;background:#111315;color:#e9ebee;font:13px system-ui,sans-serif;overflow:hidden}
#v{position:fixed;inset:0;width:100%;height:100%;object-fit:contain;cursor:grab;user-select:none}
#h{position:fixed;left:14px;bottom:14px;background:rgba(18,20,22,.85);border:1px solid rgba(255,255,255,.1);
border-radius:8px;padding:8px 12px;line-height:1.6}#h b{color:#fff}</style></head><body>
<img id="v" src="/stream" draggable="false">
<div id="h"><b>Полигон Astramis</b> · свой рендер на GPU + наша модель (детектор, атрибуты, объём)<br>
мышь — камера, колесо — зум, <b>W A S D</b> — робот, <b>P</b> — автопилот по модели, <b>V</b> — что видит модель,
<b>G</b> — истина, <b>пробел</b> — мусор, <b>R</b> — сброс</div>
<script>
const keys={w:0,a:0,s:0,d:0};let orbit={daz:0,del:0,dz:0},drag=null,act=null;
addEventListener('keydown',e=>{const k=e.key.toLowerCase();if(k in keys)keys[k]=1;
 if(k===' ')act='spawn';if(k==='r')act='reset';if(k==='v')act='view';if(k==='p')act='auto';if(k==='g')act='gt';});
addEventListener('keyup',e=>{const k=e.key.toLowerCase();if(k in keys)keys[k]=0;});
const v=document.getElementById('v');
v.addEventListener('mousedown',e=>{drag=[e.clientX,e.clientY];v.style.cursor='grabbing'});
addEventListener('mouseup',()=>{drag=null;v.style.cursor='grab'});
addEventListener('mousemove',e=>{if(!drag)return;orbit.daz-=(e.clientX-drag[0])*0.006;orbit.del+=(e.clientY-drag[1])*0.004;drag=[e.clientX,e.clientY]});
addEventListener('wheel',e=>{orbit.dz+=e.deltaY*0.0012});
setInterval(()=>{fetch('/input',{method:'POST',body:JSON.stringify({keys,orbit,act})});orbit={daz:0,del:0,dz:0};act=null},50);
</script></body></html>"""


def chip(img, txt, x, y, col, scale=0.42):
    (tw, th), _ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, scale, 1)
    x0, y0 = int(x - tw / 2), int(y)
    cv2.rectangle(img, (x0 - 4, y0 - th - 6), (x0 + tw + 4, y0 + 3), (22, 22, 24), -1)
    cv2.rectangle(img, (x0 - 4, y0 - th - 6), (x0 + tw + 4, y0 + 3), col, 1)
    cv2.putText(img, txt, (x0, y0 - 2), cv2.FONT_HERSHEY_SIMPLEX, scale, (235, 235, 235), 1, cv2.LINE_AA)


class Live:
    def __init__(self, W, H, spp, fps, perception=None):
        self.W, self.H, self.spp, self.fps = W, H, spp, fps
        self.floor = FloorTex(TILE)
        self.rng = np.random.default_rng(0)
        self.lock = threading.Lock()
        self.keys = {"w": 0, "a": 0, "s": 0, "d": 0}
        self.az, self.el, self.dist = math.pi + 0.6, 0.55, 2.2
        self.robot_view = False
        self.show_gt = False
        self.auto = False
        self.jpeg = None
        self.frame_ms = 0.0
        self.perc = perception
        self.perc_busy = False
        self.dets = []
        self.brain = Brain()
        self.reset()

    # ---------------------------------------------------------------- world
    def reset(self):
        self.x, self.y, self.yaw = 0.0, 0.0, 0.0
        self.wl = self.wr = 0.0
        self.trash = []
        for _ in range(8):
            self.spawn(near=False)
        self.collected = 0
        self.collisions = 0
        self.dets = []
        if hasattr(self, 'brain'):
            self.brain.reset()

    def spawn(self, near=True):
        t = random_trash(self.rng)
        for _ in range(50):
            if near:
                a, r = self.yaw + self.rng.uniform(-0.6, 0.6), self.rng.uniform(0.7, 1.6)
                px, py = self.x + r * math.cos(a), self.y + r * math.sin(a)
            else:
                px, py = self.rng.uniform(0.5, 3.6), self.rng.uniform(-1.3, 1.3)
            if all(math.hypot(px - o[1], py - o[2]) > t.radius + o[0].radius + 0.05 for o in self.trash):
                self.trash.append([t, px, py, float(self.rng.uniform(0, 6.28)), -1, False])
                return

    def handle(self, msg):
        with self.lock:
            self.keys.update({k: int(v) for k, v in msg.get("keys", {}).items() if k in self.keys})
            o = msg.get("orbit", {})
            self.az += o.get("daz", 0.0)
            self.el = min(1.35, max(0.12, self.el + o.get("del", 0.0)))
            self.dist = min(5.0, max(0.7, self.dist * math.exp(o.get("dz", 0.0))))
            act = msg.get("act")
            if act == "spawn":
                for _ in range(3):
                    self.spawn()
            elif act == "reset":
                self.reset()
            elif act == "view":
                self.robot_view = not self.robot_view
            elif act == "gt":
                self.show_gt = not self.show_gt
            elif act == "auto":
                self.auto = not self.auto

    # ---------------------------------------------------------------- frame
    def step(self, dt, frame):
        with self.lock:
            if self.auto and not any(self.keys.values()):
                v, w = self.brain.act(self.x, self.y, self.yaw)
            else:
                v = 0.5 * (self.keys["w"] - self.keys["s"])
                w = 1.6 * (self.keys["a"] - self.keys["d"])
            self.yaw += w * dt
            self.x += v * math.cos(self.yaw) * dt
            self.y += v * math.sin(self.yaw) * dt
            self.wl += (v - w * TRACK) * dt / WHEEL_R
            self.wr += (v + w * TRACK) * dt / WHEEL_R
            fx, fy = self.x + 0.42 * math.cos(self.yaw), self.y + 0.42 * math.sin(self.yaw)
            self.brain.passed(fx, fy, self.x, self.y, self.yaw)
            for o in self.trash:
                t, px, py, yaw, gone, hit = o
                near = math.hypot(px - fx, py - fy) < 0.22
                if gone < 0 and t.take and near:
                    o[4] = frame
                    self.collected += 1
                if not t.take and near and not hit:
                    o[5] = True
                    self.collisions += 1
            self.trash = [o for o in self.trash if o[4] < 0 or frame - o[4] < 8]
            sc = Scene()
            add_robot(sc, self.x, self.y, self.yaw, brush_angle=frame * 0.35 * (1 + abs(v)), wheel_angle=(self.wl, self.wr))
            vis, props = [], {}
            for k, (t, px, py, yaw, gone, hit) in enumerate(self.trash):
                if gone >= 0:
                    a = (frame - gone) / 8
                    px, py = px + (self.x - px) * a, py + (self.y - py) * a
                t.place(sc, px, py, yaw, k)
                props[k] = {"transmission": t.mat.transmission, "metallic": t.mat.metallic, "roughness": t.mat.roughness}
                if gone < 0:
                    vis.append((t, px, py, hit))
            eye_r, tgt_r = robot_camera(self.x, self.y, self.yaw)
            cx, cy = self.x + 0.3 * math.cos(self.yaw), self.y + 0.3 * math.sin(self.yaw)
            eye = (cx + self.dist * math.cos(self.el) * math.cos(self.az), cy + self.dist * math.cos(self.el) * math.sin(self.az),
                   0.15 + self.dist * math.sin(self.el))
            cam = FastCam.look_at(eye, (cx, cy, 0.15), W=self.W, H=self.H, fov_deg=55)
            dets = [dict(m, world=(m['x'], m['y'])) for m in self.brain.memory]
            target = self.brain.target
            mode = self.brain.mode
        # model: capture the robot's D435 frame when the worker is free
        if self.perc is not None and not self.perc_busy:
            self.perc_busy = True
            inputs = self.perc.capture(sc, eye_r, tgt_r, props, frame, self.floor)
            threading.Thread(target=self._perceive, args=(inputs,), daemon=True).start()
        torch.cuda.synchronize()
        t0 = time.time()
        if self.robot_view and self.perc is not None and self.perc.latest is not None:
            from .perception import overlay
            img = overlay(self.perc.latest["frame"], self.perc.latest["dets"], self.W, self.H)
        else:
            img = to_u8(render(sc, cam, floor=self.floor, spp=self.spp, seed=frame)["rgb"])
            if self.auto and target is not None:
                (ax, ay), za = cam.project((fx, fy, 0.01))
                (bx, by), zb = cam.project((target["x"], target["y"], 0.01))
                if za > 0 and zb > 0:
                    cv2.line(img, (int(ax), int(ay)), (int(bx), int(by)), (255, 190, 80), 2, cv2.LINE_AA)
                    cv2.circle(img, (int(bx), int(by)), 9, (255, 190, 80), 2, cv2.LINE_AA)
            for d in dets:
                (sx, sy), z = cam.project((d["world"][0], d["world"][1], d["top"] + 0.06))
                if z > 0 and 0 < sx < self.W and 0 < sy < self.H:
                    chip(img, f"model: {d['material']} {d['volume']:.0f} ml {d['mass']:.0f} g {'TAKE' if d['take'] else 'AVOID'}",
                         sx, sy, (110, 220, 120) if d["take"] else (80, 80, 245))
            if self.show_gt:
                for t, px, py, hit in vis:
                    (sx, sy), z = cam.project((px, py, 0.0))
                    if z > 0 and 0 < sx < self.W and 0 < sy < self.H:
                        chip(img, f"true: {t.material} {t.volume_ml:.0f} ml {t.mass_g:.0f} g {'TAKE' if t.take else 'AVOID'}",
                             sx, sy + 18, (200, 200, 200), 0.38)
        self.frame_ms = 0.9 * self.frame_ms + 0.1 * (time.time() - t0) * 1000
        lat = self.perc.latest["latency_ms"] if self.perc is not None and self.perc.latest else 0
        model = self.perc.name if self.perc is not None else "no model"
        cv2.putText(img, f"render {self.frame_ms:.1f} ms  |  model {lat:.0f} ms  |  collected {self.collected}  "
                         f"collisions {self.collisions}  |  autopilot {(self.brain.mode if self.auto else 'off')}  memory {len(self.brain.memory)}",
                    (12, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(img, model, (12, 46), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1, cv2.LINE_AA)
        self.last_img = img
        ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 82])
        self.jpeg = buf.tobytes()

    def _perceive(self, inputs):
        try:
            dets = self.perc.run(*inputs, rng=np.random.default_rng())
            with self.lock:
                self.dets = dets
                self.brain.observe(dets, inputs[0])
        except Exception as e:  # keep the loop alive; show the error in the log
            print("perception error:", repr(e), flush=True)
        finally:
            self.perc_busy = False

    def loop(self):
        frame, last = 0, time.time()
        while True:
            now = time.time()
            self.step(min(now - last, 0.1), frame)
            last = now
            frame += 1
            time.sleep(max(0.0, 1 / self.fps - (time.time() - now)))


def serve(live: Live, port: int):
    class H(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def do_GET(self):
            if self.path.startswith("/stream"):
                self.send_response(200)
                self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
                self.send_header("Cache-Control", "no-cache")
                self.end_headers()
                last = None
                try:
                    while True:
                        j = live.jpeg
                        if j is not None and j is not last:
                            self.wfile.write(b"--frame\r\nContent-Type: image/jpeg\r\nContent-Length: " + str(len(j)).encode()
                                             + b"\r\n\r\n" + j + b"\r\n")
                            last = j
                        time.sleep(0.005)
                except (BrokenPipeError, ConnectionResetError):
                    return
            body = PAGE.encode()
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_POST(self):
            n = int(self.headers.get("Content-Length", 0))
            try:
                live.handle(json.loads(self.rfile.read(n) or b"{}"))
            except json.JSONDecodeError:
                pass
            self.send_response(204)
            self.end_headers()

    ThreadingHTTPServer(("127.0.0.1", port), H).serve_forever()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8790)
    ap.add_argument("--W", type=int, default=960)
    ap.add_argument("--H", type=int, default=540)
    ap.add_argument("--spp", type=int, default=2)
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--no-model", action="store_true")
    ap.add_argument("--run", default="score83_cv_fold2_noval18")
    args = ap.parse_args()
    perc = None
    if not args.no_model:
        from .perception import Perception
        perc = Perception(run=args.run)
    live = Live(args.W, args.H, args.spp, args.fps, perc)
    live.step(0.0, 0)
    threading.Thread(target=live.loop, daemon=True).start()
    print(f"live polygon on http://127.0.0.1:{args.port}  model: {perc.name if perc else 'none'}", flush=True)
    serve(live, args.port)


if __name__ == "__main__":
    main()
