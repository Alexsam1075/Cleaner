"""Live polygon: the real-time renderer streamed to a browser (MJPEG) with interactive control.

  python -m polygon.live --port 8790          # on the GPU server
  ssh -L 8790:localhost:8790 ...              # on the laptop, then open http://localhost:8790

Mouse drag orbits the camera, wheel zooms, W/A/S/D drive the robot, Space drops new trash, R resets,
V toggles the robot's D435 view. Light trash is swept when the robot drives over it; heavy trash
(mass > 500 g) is marked AVOID.
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

from .fast_scene import add_robot, random_trash, robot_camera
from .fastrt import FastCam, FloorTex, Scene, render, to_u8

TILE = Path(__file__).resolve().parent / "assets/floor_tile.png"

PAGE = """<!doctype html><html lang="ru"><head><meta charset="utf-8"><title>Полигон в реальном времени</title>
<meta name="viewport" content="width=device-width, initial-scale=1">
<style>html,body{margin:0;height:100%;background:#111315;color:#e9ebee;font:13px system-ui,sans-serif;overflow:hidden}
#v{position:fixed;inset:0;width:100%;height:100%;object-fit:contain;cursor:grab;user-select:none}
#h{position:fixed;left:14px;bottom:14px;background:rgba(18,20,22,.85);border:1px solid rgba(255,255,255,.1);
border-radius:8px;padding:8px 12px;line-height:1.6}#h b{color:#fff}</style></head><body>
<img id="v" src="/stream" draggable="false">
<div id="h"><b>Полигон Astramis</b> · свой рендер на GPU, кадр считается целиком в CUDA<br>
мышь — камера, колесо — зум, <b>W A S D</b> — робот, <b>пробел</b> — новый мусор, <b>V</b> — вид камеры робота, <b>R</b> — сброс</div>
<script>
const keys={w:0,a:0,s:0,d:0};let orbit={daz:0,del:0,dz:0},drag=null,act=null;
addEventListener('keydown',e=>{const k=e.key.toLowerCase();if(k in keys)keys[k]=1;if(k===' ')act='spawn';if(k==='r')act='reset';if(k==='v')act='view';});
addEventListener('keyup',e=>{const k=e.key.toLowerCase();if(k in keys)keys[k]=0;});
const v=document.getElementById('v');
v.addEventListener('mousedown',e=>{drag=[e.clientX,e.clientY];v.style.cursor='grabbing'});
addEventListener('mouseup',()=>{drag=null;v.style.cursor='grab'});
addEventListener('mousemove',e=>{if(!drag)return;orbit.daz-=(e.clientX-drag[0])*0.006;orbit.del+=(e.clientY-drag[1])*0.004;drag=[e.clientX,e.clientY]});
addEventListener('wheel',e=>{orbit.dz+=e.deltaY*0.0012});
setInterval(()=>{fetch('/input',{method:'POST',body:JSON.stringify({keys,orbit,act})});orbit={daz:0,del:0,dz:0};act=null},50);
</script></body></html>"""


class Live:
    def __init__(self, W, H, spp, fps):
        self.W, self.H, self.spp, self.fps = W, H, spp, fps
        self.floor = FloorTex(TILE)
        self.rng = np.random.default_rng(0)
        self.lock = threading.Lock()
        self.keys = {"w": 0, "a": 0, "s": 0, "d": 0}
        self.az, self.el, self.dist = math.pi + 0.6, 0.55, 2.0
        self.robot_view = False
        self.jpeg = None
        self.frame_ms = 0.0
        self.reset()

    def reset(self):
        self.x, self.y, self.yaw = 0.0, 0.0, 0.0
        self.trash = []
        for _ in range(7):
            self.spawn(near=False)
        self.collected = 0

    def spawn(self, near=True):
        t = random_trash(self.rng)
        for _ in range(50):
            if near:
                a, r = self.yaw + self.rng.uniform(-0.6, 0.6), self.rng.uniform(0.7, 1.6)
                px, py = self.x + r * math.cos(a), self.y + r * math.sin(a)
            else:
                px, py = self.rng.uniform(0.6, 3.5), self.rng.uniform(-1.0, 1.0)
            if all(math.hypot(px - o[1], py - o[2]) > t.radius + o[0].radius + 0.05 for o in self.trash):
                self.trash.append([t, px, py, float(self.rng.uniform(0, 6.28)), -1])
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

    def step(self, dt, frame):
        with self.lock:
            v = 0.5 * (self.keys["w"] - self.keys["s"])
            w = 1.6 * (self.keys["a"] - self.keys["d"])
            self.yaw += w * dt
            self.x += v * math.cos(self.yaw) * dt
            self.y += v * math.sin(self.yaw) * dt
            fx, fy = self.x + 0.42 * math.cos(self.yaw), self.y + 0.42 * math.sin(self.yaw)
            for o in self.trash:
                t, px, py, yaw, gone = o
                if gone < 0 and t.take and math.hypot(px - fx, py - fy) < 0.22:
                    o[4] = frame
                    self.collected += 1
            self.trash = [o for o in self.trash if o[4] < 0 or frame - o[4] < 8]
            sc = Scene()
            add_robot(sc, self.x, self.y, self.yaw, brush_angle=frame * 0.35 * (1 + abs(v)))
            vis = []
            for k, (t, px, py, yaw, gone) in enumerate(self.trash):
                if gone >= 0:
                    a = (frame - gone) / 8
                    px, py = px + (self.x - px) * a, py + (self.y - py) * a
                t.place(sc, px, py, yaw, k)
                if gone < 0:
                    vis.append((t, px, py))
            if self.robot_view:
                e, tg = robot_camera(self.x, self.y, self.yaw)
                cam = FastCam.look_at(e, tg, W=self.W, H=self.H, fov_deg=70)
            else:
                cx, cy = self.x + 0.3 * math.cos(self.yaw), self.y + 0.3 * math.sin(self.yaw)
                eye = (cx + self.dist * math.cos(self.el) * math.cos(self.az), cy + self.dist * math.cos(self.el) * math.sin(self.az),
                       0.15 + self.dist * math.sin(self.el))
                cam = FastCam.look_at(eye, (cx, cy, 0.15), W=self.W, H=self.H, fov_deg=55)
        torch.cuda.synchronize()
        t0 = time.time()
        img = to_u8(render(sc, cam, floor=self.floor, spp=self.spp, seed=frame)["rgb"])
        self.frame_ms = 0.9 * self.frame_ms + 0.1 * (time.time() - t0) * 1000
        for t, px, py in vis:
            (sx, sy), z = cam.project((px, py, t.height + 0.05))
            if z > 0 and 0 < sx < self.W and 0 < sy < self.H:
                txt = f"{t.material} {t.volume_ml:.0f} ml {t.mass_g:.0f} g {'TAKE' if t.take else 'AVOID'}"
                (tw, th), _ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, 0.42, 1)
                x0, y0 = int(sx - tw / 2), int(sy)
                cv2.rectangle(img, (x0 - 4, y0 - th - 6), (x0 + tw + 4, y0 + 3), (22, 22, 24), -1)
                cv2.rectangle(img, (x0 - 4, y0 - th - 6), (x0 + tw + 4, y0 + 3), (110, 220, 120) if t.take else (80, 80, 245), 1)
                cv2.putText(img, txt, (x0, y0 - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (235, 235, 235), 1, cv2.LINE_AA)
        cv2.putText(img, f"render {self.frame_ms:.1f} ms ({1000 / max(self.frame_ms, 0.1):.0f} FPS)  {self.W}x{self.H} spp{self.spp}"
                         f"  collected {self.collected}", (12, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
        ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 82])
        self.jpeg = buf.tobytes()

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
                            self.wfile.write(b"--frame\r\nContent-Type: image/jpeg\r\nContent-Length: " + str(len(j)).encode() + b"\r\n\r\n" + j + b"\r\n")
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
    args = ap.parse_args()
    live = Live(args.W, args.H, args.spp, args.fps)
    live.step(0.0, 0)
    threading.Thread(target=live.loop, daemon=True).start()
    print(f"live polygon on http://127.0.0.1:{args.port}", flush=True)
    serve(live, args.port)


if __name__ == "__main__":
    main()
