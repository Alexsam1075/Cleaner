"""Демо «живого» обучения навигации на полигоне.
Сценарий: на пути робота лежит ворсистый ковёр (shag) — поверхность, которой не было в обучении.
  Проезд 1: замороженная модель считает ковёр проходимым -> робот въезжает, одометрия по полу фиксирует
            большой slip (робот вязнет). Признаки пола под колёсами + измеренный slip кладутся в память поверхностей.
  Проезд 2: новая сцена, такой же ковёр в другом месте. Замороженная модель снова едет прямо;
            модель с памятью узнаёт поверхность, повышает прогноз slip и планировщик выбирает объезд.
Выход: sim/out/demo/demo.mp4, summary.json."""
import json
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from data import height_map, to_tensor
from drive import SURF, surface_at
from geom import CAM, ROOT, floor_ray_hit, meta, world_to_pixel
from model import TwoHeadNet
from odometry import estimate_motion, floor_mask
from run_blender import LIGHTING, camera_pose, read_frame, render
from train import DEV

OUT = ROOT / "sim/out/demo"
SCALE = 0.5
PLANE = meta("r_0001")["floor_plane"]
NAV_COLORS = np.array([[0, 0, 0], [0, 140, 255], [0, 0, 255], [0, 220, 0]], np.uint8)
ROBOT_W = 0.35


class SurfaceMemory:
    def __init__(self, thr=0.85, max_items=2000):
        self.F, self.S = [], []
        self.thr, self.max = thr, max_items

    def add(self, feats, slip):
        self.F.append(feats)
        self.S.append(np.full(len(feats), slip, np.float32))

    def apply(self, p, slip_pred):
        """p: [C,h,w] нормированные признаки; возвращает скорректированную карту slip и карту «узнавания»."""
        if not self.F:
            return slip_pred, np.zeros_like(slip_pred)
        Fm = torch.from_numpy(np.concatenate(self.F)[-self.max:]).to(p.device)
        Sm = torch.from_numpy(np.concatenate(self.S)[-self.max:]).to(p.device)
        C, h, w = p.shape
        q = p.reshape(C, -1).T
        sim = q @ Fm.T
        top, idx = sim.topk(min(10, len(Fm)), dim=1)
        wgt = torch.softmax((top - top.max(1, keepdim=True).values) / 0.05, 1)
        s_mem = (wgt * Sm[idx]).sum(1).reshape(h, w)
        conf = ((top[:, 0] - self.thr) / (1 - self.thr)).clamp(0, 1).reshape(h, w)
        out = slip_pred * (1 - conf) + s_mem * conf
        return out, conf


def load_net(path=ROOT / "sim/checkpoints/twohead.pt"):
    ck = torch.load(path, map_location=DEV, weights_only=False)
    net = TwoHeadNet(pretrained=False).to(DEV)
    net.load_state_dict(ck["model"])
    return net.eval()


@torch.no_grad()
def infer(net, rgb, z, plane):
    hgt, valid = height_map(z, plane, SCALE)
    x = to_tensor(rgb, hgt, valid)[None].to(DEV)
    out = net(x)
    p = F.normalize(out["p"][0], dim=0)
    nav = out["nav_logits"][0].argmax(0)
    return p, out["slip"][0], nav


def near_band_features(p, n=200, rng=None):
    """Признаки пола, по которому робот сейчас поедет: нижняя центральная полоса кадра (~0.25–0.4 м впереди)."""
    C, h, w = p.shape
    band = p[:, int(h * 0.82):, int(w * 0.35):int(w * 0.65)].reshape(C, -1).T
    rng = rng or np.random.default_rng(0)
    idx = rng.choice(len(band), min(n, len(band)), replace=False)
    return band[idx].cpu().numpy()


def floor_cost_grid(slip_map, plane, nav):
    """Переводит карту slip/навигации из пикселей в сетку на полу (x: -1..1, y: 0.2..2.0 м), шаг 5 см."""
    xs = np.arange(-1.0, 1.0, 0.05)
    ys = np.arange(0.2, 2.0, 0.05)
    X, Y = np.meshgrid(xs, ys)
    uv, z = world_to_pixel(np.stack([X, Y, np.zeros_like(X)], -1), plane)
    h, w = slip_map.shape
    u = (uv[..., 0] * SCALE * w / (CAM["width"] * SCALE)).round().astype(int)
    v = (uv[..., 1] * SCALE * h / (CAM["height"] * SCALE)).round().astype(int)
    ok = (z > 0) & (u >= 0) & (u < w) & (v >= 0) & (v < h)
    cost = np.full(X.shape, np.nan)
    cost[ok] = slip_map[v[ok], u[ok]]
    obst = np.zeros(X.shape, bool)
    obst[ok] = nav[v[ok], u[ok]] == 2
    return xs, ys, cost, obst


def plan(xs, ys, cost, obst, n_headings=21, length=1.4):
    """Выбор курса: интеграл прогнозного slip вдоль коридора шириной робота + штраф за препятствия и отклонение."""
    best, scores = None, []
    for yaw in np.linspace(-0.6, 0.6, n_headings):
        c = 0.0
        for s in np.arange(0.2, length, 0.05):
            for lat in np.linspace(-ROBOT_W / 2, ROBOT_W / 2, 5):
                x = -np.sin(yaw) * s + np.cos(yaw) * lat
                y = np.cos(yaw) * s + np.sin(yaw) * lat
                i, j = np.searchsorted(ys, y) - 1, np.searchsorted(xs, x) - 1
                if 0 <= i < len(ys) and 0 <= j < len(xs) and not np.isnan(cost[i, j]):
                    c += cost[i, j] * 0.05 + (5.0 if obst[i, j] else 0.0)
        c += 0.15 * abs(yaw)
        scores.append(c)
        if best is None or c < best[1]:
            best = (yaw, c)
    return float(best[0]), scores


def overlay(rgb, nav, slip, conf=None, text=""):
    h, w = slip.shape
    img = cv2.resize(rgb, (w, h)) if rgb.shape[:2] != (h, w) else rgb.copy()
    col = NAV_COLORS[nav]
    m = nav > 0
    img[m] = (0.5 * img[m] + 0.5 * col[m]).astype(np.uint8)
    heat = cv2.applyColorMap((np.clip(slip / 0.5, 0, 1) * 255).astype(np.uint8), cv2.COLORMAP_INFERNO)
    hm = slip > 0.15
    img[hm] = (0.55 * img[hm] + 0.45 * heat[hm]).astype(np.uint8)
    if conf is not None:
        cnt, _ = cv2.findContours((conf > 0.5).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(img, cnt, -1, (255, 255, 0), 2)
    cv2.putText(img, text, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    return img


def draw_heading(img, yaw, color, plane):
    pts = []
    for s in np.arange(0.25, 1.6, 0.05):
        uv, z = world_to_pixel(np.array([[-np.sin(yaw) * s, np.cos(yaw) * s, 0.0]]), plane)
        pts.append((uv[0] * SCALE).astype(int))
    cv2.polylines(img, [np.array(pts)], False, color, 3)


def pass1(net, mem, writer):
    rug = {"type": "shag", "rect": [-1.5, 0.45, 1.5, 1.4], "thickness": 0.025}
    scene = {"surfaces": [{"type": "tile", "rect": [-6, -6, 6, 6], "rot": 0.12}, rug], "objects": [], "props": [], **LIGHTING["bright"]}
    rng = np.random.default_rng(0)
    x, y, traj = 0.0, -0.45, []
    for i in range(34):
        surf = surface_at(scene["surfaces"], x, y + 0.17)
        s = float(np.clip(SURF[surf]["rolling_resistance"] * 5 + rng.normal(0, 0.01), 0, 0.9))
        traj.append({"x": x, "y": y, "surface": surf, "slip": s})
        y += 0.04 * (1 - s)
    scene["camera"] = {"poses": [camera_pose(PLANE, t["x"], t["y"]) for t in traj]}
    frames = render(scene, OUT / "pass1", samples=16, scale=SCALE)
    log, pending = [], []
    for i in range(len(frames) - 1):
        c0, z0, _ = read_frame(frames[i])
        c1, _, _ = read_frame(frames[i + 1])
        p, slip, nav = infer(net, c0, z0, PLANE)
        m = estimate_motion(c0, c1, PLANE, mask0=floor_mask(z0, PLANE, K_scale=SCALE), scale=SCALE)
        meas = 1 - m["forward"] / 0.04
        pending.append((traj[i]["y"] + 0.3, near_band_features(p, rng=rng)))
        while pending and pending[0][0] <= traj[i]["y"] + 0.17:
            _, feats = pending.pop(0)
            mem.add(feats, float(np.clip(meas, 0, 1)))
        pred_here = float(slip[int(slip.shape[0] * 0.9):, slip.shape[1] // 3: 2 * slip.shape[1] // 3].mean())
        log.append({"y": traj[i]["y"], "surface": traj[i]["surface"], "slip_true": traj[i]["slip"], "slip_measured": meas, "slip_pred_frozen": pred_here})
        sl = F.interpolate(slip[None, None], size=c0.shape[:2], mode="bilinear")[0, 0].cpu().numpy()
        nv = F.interpolate(nav[None, None].float(), size=c0.shape[:2], mode="nearest")[0, 0].cpu().numpy().astype(int)
        img = overlay(c0, nv, sl, text=f"PASS 1 (frozen)  surface={traj[i]['surface']}  slip measured={meas:.2f}")
        writer(img, log)
    return log


def pass2(net, mem, writer):
    scene = {"surfaces": [{"type": "tile", "rect": [-6, -6, 6, 6], "rot": 0.5},
                          {"type": "shag", "rect": [-0.35, 0.55, 0.75, 1.6], "rot": 0.2, "thickness": 0.025}],
             "objects": [{"id": "s03", "x": -0.55, "y": 1.3, "material": "plastic"}], "props": [], **LIGHTING["dim"]}
    scene["camera"] = {"poses": [camera_pose(PLANE, 0.0, 0.0)]}
    fr = render(scene, OUT / "pass2", samples=32, scale=SCALE)[0]
    c0, z0, _ = read_frame(fr)
    p, slip, nav = infer(net, c0, z0, PLANE)
    slip_live, conf = mem.apply(p, slip)
    res = {}
    imgs = []
    for name, sm, cf in (("frozen", slip, None), ("live", slip_live, conf)):
        sl = F.interpolate(sm[None, None], size=c0.shape[:2], mode="bilinear")[0, 0].cpu().numpy()
        nv = F.interpolate(nav[None, None].float(), size=c0.shape[:2], mode="nearest")[0, 0].cpu().numpy().astype(int)
        cfu = None if cf is None else F.interpolate(cf[None, None], size=c0.shape[:2], mode="bilinear")[0, 0].cpu().numpy()
        xs, ys, cost, obst = floor_cost_grid(sl, PLANE, nv)
        yaw, _ = plan(xs, ys, cost, obst)
        through = [surface_at(scene["surfaces"], -np.sin(yaw) * s, np.cos(yaw) * s) for s in np.arange(0.3, 1.5, 0.05)]
        res[name] = {"heading_deg": round(float(np.degrees(yaw)), 1), "share_path_on_shag": round(float(np.mean([t == "shag" for t in through])), 2)}
        img = overlay(c0, nv, sl, cfu, text=f"PASS 2 {name}: heading {np.degrees(yaw):+.0f} deg, path on rug {res[name]['share_path_on_shag']:.0%}")
        draw_heading(img, yaw, (255, 255, 255) if name == "frozen" else (0, 255, 0), PLANE)
        imgs.append(img)
    for _ in range(45):
        writer(imgs[0], None)
    for _ in range(45):
        writer(imgs[1], None)
    cv2.imwrite(str(OUT / "pass2_compare.jpg"), np.hstack(imgs))
    return res


def main():
    import sys
    OUT.mkdir(parents=True, exist_ok=True)
    net = load_net(sys.argv[1]) if len(sys.argv) > 1 else load_net()
    mem = SurfaceMemory()
    W, H = int(CAM["width"] * SCALE), int(CAM["height"] * SCALE)
    vw = cv2.VideoWriter(str(OUT / "demo.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), 10, (W + 360, H))

    def writer(img, log):
        panel = np.full((H, 360, 3), 30, np.uint8)
        if log:
            ys_ = [r["slip_measured"] for r in log]
            ps_ = [r["slip_pred_frozen"] for r in log]
            for k in range(1, len(log)):
                x0, x1 = 20 + (k - 1) * 9, 20 + k * 9
                cv2.line(panel, (x0, int(300 - ys_[k - 1] * 500)), (x1, int(300 - ys_[k] * 500)), (0, 200, 255), 2)
                cv2.line(panel, (x0, int(300 - ps_[k - 1] * 500)), (x1, int(300 - ps_[k] * 500)), (200, 200, 200), 1)
            cv2.putText(panel, "slip: measured (orange)", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 255), 1)
            cv2.putText(panel, "slip: frozen model (grey)", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
            cv2.putText(panel, f"memory: {sum(len(f) for f in mem.F)} vectors", (10, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
        else:
            cv2.putText(panel, "new scene, same rug type", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            cv2.putText(panel, "white = frozen plan", (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            cv2.putText(panel, "green = live plan (memory)", (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            cv2.putText(panel, "cyan contour = recognised", (10, 105), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
        vw.write(np.hstack([img, panel]))

    log = pass1(net, mem, writer)
    res2 = pass2(net, mem, writer)
    vw.release()
    on = [r for r in log if r["surface"] == "shag"]
    off = [r for r in log if r["surface"] != "shag"]
    summary = {"pass1": {"slip_measured_on_rug": float(np.mean([r["slip_measured"] for r in on])),
                         "slip_true_on_rug": float(np.mean([r["slip_true"] for r in on])),
                         "slip_pred_frozen_on_rug": float(np.mean([r["slip_pred_frozen"] for r in on])),
                         "slip_measured_on_tile": float(np.mean([r["slip_measured"] for r in off]))},
               "pass2": res2}
    (OUT / "summary.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
