"""Own GPU renderer: sphere tracing of SDF parts with physically based shading.

Two modes:
* ``render_over_frame`` — synthetic objects composited into a real dataset frame. The floor, its
  noise and its depth stay the real ones; objects get light estimated from the floor highlight,
  irradiance matched to the local floor brightness, soft shadows, contact occlusion, a faint
  reflection in the glossy tile, environment reflections and glass refraction sampled from the
  real photo.
* ``render_cg`` — a fully synthetic scene on a textured floor (robot animation, map views).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from . import sdf as S

DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ---------------------------------------------------------------- colour helpers

def srgb_to_lin(x):
    return torch.where(x <= 0.04045, x / 12.92, ((x + 0.055) / 1.055) ** 2.4)


def lin_to_srgb(x):
    x = x.clamp(0, 1)
    return torch.where(x <= 0.0031308, x * 12.92, 1.055 * x ** (1 / 2.4) - 0.055)


# ---------------------------------------------------------------- materials and parts

@dataclass
class Material:
    color: tuple = (0.5, 0.5, 0.5)       # linear albedo
    roughness: float = 0.5
    metallic: float = 0.0
    transmission: float = 0.0
    ior: float = 1.5
    tint: tuple = (0.9, 0.95, 0.9)        # glass absorption colour per 5 mm
    texture: str = "plain"                # plain|fbm|wood|stone|brushed|paper|label|bristle
    color2: tuple = (0.2, 0.2, 0.2)
    tex_scale: float = 40.0
    tex_amp: float = 0.25
    clearcoat: float = 0.0
    color_fn: Callable | None = None      # custom albedo(p_local, n_local) -> (N,3)

    def albedo(self, p, n):
        if self.color_fn is not None:
            return self.color_fn(p, n)
        c1 = torch.tensor(self.color, device=p.device)
        c2 = torch.tensor(self.color2, device=p.device)
        t = self.texture
        if t == "plain":
            return c1.expand(len(p), 3)
        if t == "fbm":
            k = (S.fbm(p * self.tex_scale, 4) * 0.5 + 0.5)
            return c1 * (1 - self.tex_amp + 2 * self.tex_amp * k[:, None])
        if t == "stone":
            k = (S.fbm(p * self.tex_scale, 5) * 0.5 + 0.5).clamp(0, 1) ** 1.5
            grain = S.value_noise(p * self.tex_scale * 6) * 0.5 + 0.5
            spots = (S.value_noise(p * self.tex_scale * 2.5 + 7) > 0.6).float()
            return (c1 * (1 - k[:, None]) + c2 * k[:, None]) * (0.75 + 0.5 * grain[:, None]) * (1 + 0.6 * spots[:, None])
        if t == "wood":
            r = (p[:, 1] ** 2 + p[:, 2] ** 2).sqrt() * self.tex_scale + 3 * S.fbm(p * 8, 3)
            g = (torch.sin(r * 6.283) * 0.5 + 0.5) ** 3
            return c1 * (1 - g[:, None]) + c2 * g[:, None]
        if t == "brushed":
            k = S.value_noise(torch.stack([p[:, 0] * 900, p[:, 1] * 6, p[:, 2] * 6], -1)) * 0.5 + 0.5
            return c1 * (0.85 + 0.3 * k[:, None])
        if t == "paper":
            k = S.fbm(p * self.tex_scale, 3) * 0.5 + 0.5
            return c1 * (0.9 + 0.2 * k[:, None])
        if t == "label":       # a printed band around the middle of a bottle (local z)
            band = ((p[:, 0].abs() < self.tex_amp)).float()
            stripes = (torch.sin(torch.atan2(p[:, 1], p[:, 2]) * 7) > 0.6).float()
            return c1 * (1 - band[:, None]) + (c2 * (0.7 + 0.3 * stripes[:, None])) * band[:, None]
        if t == "bristle":
            ang = torch.atan2(p[:, 1], p[:, 0])
            k = S.value_noise(torch.stack([ang * 60, p[:, 2] * 5, ang * 3], -1)) * 0.5 + 0.5
            return c1 * (0.5 + 0.9 * k[:, None])
        raise ValueError(t)


@dataclass
class Part:
    sdf: Callable                         # local points (N,3) -> (N,)
    material: Material
    radius: float                         # bounding sphere radius around the local origin
    pos: tuple = (0.0, 0.0, 0.0)          # world position of the local origin
    yaw: float = 0.0
    obj_id: int = -1                      # instance id written to the mask (-1 = not labelled)
    no_shadow: bool = False

    def to_local(self, p):
        return S.rot_z(p - torch.tensor(self.pos, device=p.device, dtype=p.dtype), self.yaw)

    def dir_to_world(self, v):
        return S.rot_z(v, -self.yaw)


def scene_dist(parts: list[Part], p: torch.Tensor, skip_shadowless: bool = False):
    """Distance to the nearest part and its index; far parts use their bounding sphere."""
    best = torch.full((len(p),), 1e3, device=p.device)
    idx = torch.full((len(p),), -1, device=p.device, dtype=torch.long)
    for k, part in enumerate(parts):
        if skip_shadowless and part.no_shadow:
            continue
        q = part.to_local(p)
        bd = q.norm(dim=-1) - part.radius
        near = bd < 0.02
        d = bd + 0.01
        if near.any():
            d = d.clone()
            d[near] = part.sdf(q[near]).to(d.dtype)
        better = d < best
        best = torch.where(better, d, best)
        idx = torch.where(better, torch.full_like(idx, k), idx)
    return best, idx


def march(parts, o, d, tmax=6.0, steps=160, inside=False):
    t = torch.full((len(o),), 1e-3, device=o.device)
    hit = torch.zeros(len(o), dtype=torch.bool, device=o.device)
    alive = torch.ones(len(o), dtype=torch.bool, device=o.device)
    idx = torch.full((len(o),), -1, dtype=torch.long, device=o.device)
    for _ in range(steps):
        if not alive.any():
            break
        a = alive.nonzero().squeeze(1)
        dist, k = scene_dist(parts, o[a] + d[a] * t[a, None])
        if inside:
            dist = -dist
        done = dist.abs() < 2e-4 + 1e-4 * t[a]
        hit[a[done]] = True
        idx[a[done]] = k[done]
        t[a] = t[a] + dist.clamp(min=2e-4) * 0.9
        far = t[a] > tmax
        alive[a[done | far]] = False
    return t, hit, idx


def normals(parts, p):
    e = 2e-4
    ks = torch.tensor([[1, -1, -1], [-1, -1, 1], [-1, 1, -1], [1, 1, 1]], device=p.device, dtype=p.dtype)
    n = sum(k * scene_dist(parts, p + k * e)[0][:, None] for k in ks)
    return F.normalize(n, dim=-1)


def soft_shadow(parts, p, l, tmax, k=12.0, steps=48):
    res = torch.ones(len(p), device=p.device)
    t = 0.004 + torch.rand(len(p), device=p.device) * 0.01          # jitter hides banding
    for _ in range(steps):
        d, _ = scene_dist(parts, p + l * t[:, None], skip_shadowless=True)
        res = torch.minimum(res, (k * d / t).clamp(0, 1))
        t = t + d.clamp(0.002, 0.03) * (0.8 + 0.4 * torch.rand(len(p), device=p.device))
        if bool((t > tmax).all()):
            break
    return res.clamp(0, 1) ** 1.5


def ambient_occlusion(parts, p, n, scale=0.03):
    occ = torch.zeros(len(p), device=p.device)
    for i in range(1, 6):
        h = scale * i / 5
        d, _ = scene_dist(parts, p + n * h, skip_shadowless=True)
        occ = occ + (h - d).clamp(min=0) / (2 ** i)
    return (1 - 3.0 * occ / scale).clamp(0.15, 1)


def refract(d, n, eta):
    cosi = -(d * n).sum(-1, keepdim=True)
    k = 1 - eta * eta * (1 - cosi * cosi)
    t = eta * d + (eta * cosi - k.clamp(min=0).sqrt()) * n
    refl = d + 2 * cosi * n
    return F.normalize(torch.where(k < 0, refl, t), dim=-1)


# ---------------------------------------------------------------- camera

@dataclass
class Camera:
    fx: float
    fy: float
    cx: float
    cy: float
    W: int
    H: int
    origin: torch.Tensor           # (3,) world
    R: torch.Tensor                # (3,3) camera -> world

    @classmethod
    def from_plane(cls, cam: dict, plane) -> "Camera":
        p = np.asarray(plane, dtype=np.float64)
        nrm = np.linalg.norm(p[:3])
        n, d = p[:3] / nrm, p[3] / nrm
        if d < 0:
            n, d = -n, -d
        fwd = np.array([0, 0, 1.0]) - n * n[2]
        fwd /= np.linalg.norm(fwd)
        side = np.cross(n, fwd)
        R = np.stack([fwd, side, n])                       # rows: world axes in camera coords
        return cls(cam["fx"], cam["fy"], cam["cx"], cam["cy"], cam["width"], cam["height"],
                   torch.tensor([0.0, 0.0, d], device=DEV, dtype=torch.float32),
                   torch.tensor(R, device=DEV, dtype=torch.float32))

    @classmethod
    def look_at(cls, eye, target, W=1280, H=720, fov_deg=60.0, up=(0, 0, 1)) -> "Camera":
        eye, target, up = (np.asarray(v, float) for v in (eye, target, up))
        z = target - eye
        z /= np.linalg.norm(z)
        x = np.cross(z, up)
        x /= np.linalg.norm(x)
        y = np.cross(z, x)
        f = 0.5 * W / math.tan(math.radians(fov_deg) / 2)
        R = np.stack([x, y, z], axis=1)                    # columns: camera axes in world
        return cls(f, f, W / 2, H / 2, W, H, torch.tensor(eye, device=DEV, dtype=torch.float32),
                   torch.tensor(R, device=DEV, dtype=torch.float32))

    def rays(self, pix: torch.Tensor):
        """pix (N,2) float pixel coords -> world origins, unit dirs, and z-depth factor."""
        dc = torch.stack([(pix[:, 0] - self.cx) / self.fx, (pix[:, 1] - self.cy) / self.fy,
                          torch.ones(len(pix), device=pix.device)], -1)
        dw = dc @ self.R.T
        norm = dw.norm(dim=-1, keepdim=True)
        return self.origin.expand(len(pix), 3), dw / norm, 1.0 / norm.squeeze(1)

    def project(self, pw: torch.Tensor):
        pc = (pw - self.origin) @ self.R
        z = pc[:, 2].clamp(min=1e-6)
        return torch.stack([pc[:, 0] / z * self.fx + self.cx, pc[:, 1] / z * self.fy + self.cy], -1), pc[:, 2]


# ---------------------------------------------------------------- background (real frame)

@dataclass
class Background:
    rgb_lin: torch.Tensor          # (H,W,3)
    depth_m: torch.Tensor          # (H,W), 0 = hole
    blur_s: torch.Tensor           # (H,W,3) light blur
    blur_l: torch.Tensor           # (H,W,3) heavy blur (rough reflections, local irradiance)
    floor: torch.Tensor            # (H,W) bool: valid floor pixels (height < 1 cm)
    room: torch.Tensor             # (3,) colour above the horizon
    light: torch.Tensor            # (3,) main light position (world)
    noise: float                   # sRGB sensor noise std

    @classmethod
    def from_frame(cls, rgb_u8: np.ndarray, depth_u16: np.ndarray, cam: Camera) -> "Background":
        rgb = torch.tensor(rgb_u8[..., ::-1].copy(), device=DEV, dtype=torch.float32) / 255
        lin = srgb_to_lin(rgb)
        bs = torch.tensor(cv2.GaussianBlur(lin.cpu().numpy(), (0, 0), 2.5), device=DEV)
        bl = torch.tensor(cv2.GaussianBlur(lin.cpu().numpy(), (0, 0), 14), device=DEV)
        depth = torch.tensor(depth_u16.astype(np.float32) / 1000, device=DEV)
        H, W = depth.shape
        yy, xx = torch.meshgrid(torch.arange(H, device=DEV), torch.arange(W, device=DEV), indexing="ij")
        pix = torch.stack([xx.reshape(-1), yy.reshape(-1)], -1).float()
        o, d, zf = cam.rays(pix)
        pts = o + d * (depth.reshape(-1) / zf)[:, None]
        floor = ((pts[:, 2].abs() < 0.01) & (depth.reshape(-1) > 0)).reshape(H, W)
        room = bl[: H // 8].reshape(-1, 3).mean(0) * 1.3
        light = cls._estimate_light(bl, floor, cam)
        g = cv2.cvtColor(rgb_u8, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255
        hp = g - cv2.GaussianBlur(g, (0, 0), 1.5)
        fl = floor.cpu().numpy()
        noise = float(np.std(hp[fl])) if fl.any() else 0.01
        return cls(lin, depth, bs, bl, floor, room, light, noise)

    @staticmethod
    def _estimate_light(bl, floor, cam: Camera):
        """Main light from the brightest floor reflection: mirror the camera in the floor."""
        lum = bl.mean(-1).clone()
        lum[~floor] = 0
        H = lum.shape[0]
        lum[: H // 10] = 0
        flat = lum.reshape(-1)
        k = int(flat.argmax())
        y, x = divmod(k, lum.shape[1])
        med = float(bl.mean(-1)[floor].median()) if floor.any() else 1.0
        if float(flat[k]) < 1.25 * med:
            return torch.tensor([1.6, 0.2, 2.6], device=DEV)
        o, d, _ = cam.rays(torch.tensor([[float(x), float(y)]], device=DEV))
        t = -o[0, 2] / d[0, 2]
        P = o[0] + d[0] * t
        mirror = torch.tensor([0.0, 0.0, -float(cam.origin[2])], device=DEV)
        v = P - mirror
        s = (2.6 - mirror[2]) / v[2]
        return mirror + v * s

    def sample(self, img, pix):
        H, W = img.shape[:2]
        g = torch.stack([pix[:, 0] / (W - 1) * 2 - 1, pix[:, 1] / (H - 1) * 2 - 1], -1)
        out = F.grid_sample(img.permute(2, 0, 1)[None], g[None, None], align_corners=True, mode="bilinear")
        return out[0, :, 0].T

    def env(self, cam: Camera, o, d, rough):
        """Environment seen along rays: the real floor below the horizon, the room above."""
        col = self.room.expand(len(o), 3).clone()
        down = d[:, 2] < -1e-3
        if down.any():
            t = -o[down, 2] / d[down, 2]
            q = o[down] + d[down] * t[:, None]
            pix, z = cam.project(q)
            inside = (pix[:, 0] >= 0) & (pix[:, 0] < cam.W - 1) & (pix[:, 1] >= 0) & (pix[:, 1] < cam.H - 1) & (z > 0)
            fallback = self.blur_l[self.floor].mean(0) if self.floor.any() else self.room
            c = fallback.expand(len(q), 3).clone()
            if inside.any():
                r = rough[down][inside][:, None]
                c[inside] = self.sample(self.blur_s, pix[inside]) * (1 - r) + self.sample(self.blur_l, pix[inside]) * r
            col[down] = c
        return col


# ---------------------------------------------------------------- shading

def ggx_spec(n, v, l, rough, f0):
    h = F.normalize(v + l, dim=-1)
    nl = (n * l).sum(-1).clamp(min=0)
    nv = (n * v).sum(-1).clamp(min=1e-3)
    nh = (n * h).sum(-1).clamp(min=0)
    vh = (v * h).sum(-1).clamp(min=0)
    a = (rough * rough).clamp(min=0.02)
    a2 = a * a
    D = a2 / (math.pi * (nh * nh * (a2 - 1) + 1) ** 2)
    k = (rough + 1) ** 2 / 8
    G = nl / (nl * (1 - k) + k) * nv / (nv * (1 - k) + k)
    Fr = f0 + (1 - f0) * ((1 - vh) ** 5)[:, None]
    return (D * G / (4 * nv))[:, None] * Fr * nl[:, None]


def shade_hits(parts, cam: Camera, bg: Background | None, o, d, t, idx, irradiance, light_pos,
               floor_tex=None, depth_level=0):
    """Linear colour for ray hits. ``irradiance`` (N,3) is the floor irradiance near each hit."""
    p = o + d * t[:, None]
    n = normals(parts, p)
    v = -d
    col = torch.zeros(len(p), 3, device=p.device)
    lvec = light_pos[None] - p
    ldist = lvec.norm(dim=-1)
    l = lvec / ldist[:, None]
    up_l = l[:, 2].clamp(min=0.05)
    sh = soft_shadow(parts, p + n * 2e-3, l, tmax=float(ldist.max().clamp(max=3.0)))
    ao = ambient_occlusion(parts, p, n)
    i_amb, i_dir = 0.45 * irradiance, 0.55 * irradiance / up_l[:, None]
    for k in idx.unique().tolist():
        sel = idx == k
        part = parts[k]
        m = part.material
        pl = part.to_local(p[sel])
        nl_ = S.rot_z(n[sel], part.yaw)
        alb = m.albedo(pl, nl_)
        rough = torch.full((int(sel.sum()),), m.roughness, device=p.device)
        nlv = (n[sel] * l[sel]).sum(-1).clamp(min=0)
        diffuse = alb * (i_amb[sel] * ao[sel, None] + i_dir[sel] * (nlv * sh[sel])[:, None])
        f0 = torch.full_like(alb, 0.04) * (1 - m.metallic) + alb * m.metallic
        spec = ggx_spec(n[sel], v[sel], l[sel], rough, f0) * i_dir[sel] * sh[sel, None] * 1.4
        cosv = (n[sel] * v[sel]).sum(-1).clamp(0, 1)
        fres = f0 + (1 - f0) * ((1 - cosv) ** 5)[:, None]
        r = F.normalize(d[sel] - 2 * (d[sel] * n[sel]).sum(-1, keepdim=True) * n[sel], dim=-1)
        if bg is not None:
            env = bg.env(cam, p[sel] + n[sel] * 1e-3, r, rough)
        else:
            env = sky_env(r, floor_tex, p[sel], rough)
        env = env * ao[sel, None]
        gloss = (1 - m.roughness) ** 2
        refl = env * fres * (gloss * (1 - m.metallic) + m.metallic)
        c = diffuse * (1 - m.metallic) * (1 - m.transmission) + spec + refl
        if m.clearcoat > 0:
            c = c + env * 0.04 * m.clearcoat
        if m.transmission > 0 and depth_level == 0:
            behind = transmit(parts, cam, bg, p[sel], d[sel], n[sel], m, floor_tex)
            c = c + behind * (1 - fres.mean(-1, keepdim=True)) * m.transmission
        col[sel] = c
    return col, n, sh


def transmit(parts, cam, bg, p, d, n, m: Material, floor_tex):
    t1 = refract(d, n, 1 / m.ior)
    o2 = p - n * 2e-3
    tt, hit, _ = march(parts, o2, t1, tmax=0.5, steps=80, inside=True)
    q = o2 + t1 * tt[:, None]
    n2 = normals(parts, q)
    t2 = refract(t1, -n2, m.ior)
    tint = torch.tensor(m.tint, device=p.device) ** (tt.clamp(max=0.2) / 0.005)[:, None]
    rough = torch.full((len(p),), 0.05, device=p.device)
    if bg is not None:
        behind = bg.env(cam, q + t2 * 2e-3, t2, rough)
    else:
        behind = sky_env(t2, floor_tex, q, rough)
    return behind * tint


# ---------------------------------------------------------------- composite into a real frame

def render_over_frame(parts: list[Part], cam: Camera, bg: Background, ss: int = 2, seed: int = 0):
    """Return dict(color_u8 HxWx3 BGR, depth_true_m HxW, ids HxW, spec HxW) with objects composited."""
    H, W = cam.H, cam.W
    gen = torch.Generator(device=DEV).manual_seed(seed)
    yy, xx = torch.meshgrid(torch.arange(H, device=DEV), torch.arange(W, device=DEV), indexing="ij")
    pix = torch.stack([xx.reshape(-1), yy.reshape(-1)], -1).float()
    o, d, zf = cam.rays(pix)
    t_floor = torch.where(d[:, 2] < -1e-4, -o[:, 2] / d[:, 2], torch.full_like(zf, 1e3))
    fpts = o + d * t_floor[:, None]
    centers = torch.tensor([pt.pos for pt in parts], device=DEV, dtype=torch.float32)
    radii = torch.tensor([pt.radius for pt in parts], device=DEV)
    # pixels that may show an object, its shadow or its reflection
    dxy = (fpts[:, None, :2] - centers[None, :, :2]).norm(dim=-1)
    region = (dxy < radii[None] * 3.0 + 0.06).any(1)
    cpix, cz = cam.project(centers)
    rpix = radii / cz.clamp(min=0.05) * cam.fx
    for c, r in zip(cpix, rpix):
        region |= ((pix - c).norm(dim=-1) < r * 1.3 + 4)
    ridx = region.nonzero().squeeze(1)

    color = bg.rgb_lin.reshape(-1, 3).clone()
    depth = bg.depth_m.reshape(-1).clone()
    ids = torch.full((H * W,), -1, dtype=torch.long, device=DEV)
    spec_map = torch.zeros(H * W, device=DEV)
    light = bg.light

    # local floor irradiance (no objects in the background): median-albedo assumption 0.5
    def irr_at(q):
        pp, _ = cam.project(q)
        pp[:, 0] = pp[:, 0].clamp(0, W - 1)
        pp[:, 1] = pp[:, 1].clamp(0, H - 1)
        return bg.sample(bg.blur_l, pp) / 0.5

    # 1. floor in the region: shadow, contact occlusion, glossy reflection of objects
    bgd = depth[ridx]
    fl = bg.floor.reshape(-1)[ridx] | (bgd == 0)
    fr = ridx[fl]
    if len(fr):
        P = fpts[fr]
        lv = F.normalize(light[None] - P, dim=-1)
        sh = soft_shadow(parts, P + torch.tensor([0, 0, 1e-3], device=DEV), lv, tmax=2.5, k=10)
        ao = ambient_occlusion(parts, P, torch.tensor([[0.0, 0, 1]], device=DEV).expand(len(P), 3), 0.04)
        fac = (0.45 * ao + 0.55 * sh)
        color[fr] = color[fr] * fac[:, None]
        rd = d[fr] * torch.tensor([1.0, 1.0, -1.0], device=DEV)
        tr, hr, kr = march(parts, P + rd * 1e-3, rd, tmax=1.5, steps=90)
        if hr.any():
            q = P[hr] + rd[hr] * tr[hr, None]
            alb = torch.stack([parts[k].material.albedo(parts[k].to_local(q[j:j + 1]), q[j:j + 1])[0]
                               for j, k in enumerate(kr[hr].tolist())]) if hr.sum() < 64 else \
                torch.tensor([parts[k].material.color for k in kr[hr].tolist()], device=DEV)
            w = 0.10 * torch.exp(-tr[hr] * 6)
            cr = alb * irr_at(q) * 0.6
            sub = fr[hr]
            color[sub] = color[sub] * (1 - w[:, None]) + cr * w[:, None]

    # 2. objects, supersampled
    offs = [(0.25, 0.25), (0.75, 0.25), (0.25, 0.75), (0.75, 0.75)] if ss >= 2 else [(0.5, 0.5)]
    acc = torch.zeros(len(ridx), 3, device=DEV)
    cover = torch.zeros(len(ridx), device=DEV)
    zbest = torch.full((len(ridx),), 1e3, device=DEV)
    idbest = torch.full((len(ridx),), -1, dtype=torch.long, device=DEV)
    specbest = torch.zeros(len(ridx), device=DEV)
    bg_z = depth[ridx]
    for ox, oy in offs:
        pr = pix[ridx] + torch.tensor([ox - 0.5, oy - 0.5], device=DEV)
        oo, dd, zz = cam.rays(pr)
        t, hit, k = march(parts, oo, dd)
        zc = t * zz
        visible = hit & ((bg_z == 0) | (zc < bg_z + 0.004) | fl)
        if visible.any():
            hv = visible.nonzero().squeeze(1)
            irr = irr_at(oo[hv] + dd[hv] * t[hv, None])
            c, nrm, sh = shade_hits(parts, cam, bg, oo[hv], dd[hv], t[hv], k[hv], irr, light)
            acc[hv] += c
            cover[hv] += 1
            first = zc[hv] < zbest[hv]
            zbest[hv] = torch.where(first, zc[hv], zbest[hv])
            idbest[hv] = torch.where(first, torch.tensor([parts[j].obj_id for j in k[hv].tolist()], device=DEV), idbest[hv])
            lv = F.normalize(light[None] - (oo[hv] + dd[hv] * t[hv, None]), dim=-1)
            hvec = F.normalize(lv - dd[hv], dim=-1)
            spec = (nrm * hvec).sum(-1).clamp(min=0) ** 60 * sh
            specbest[hv] = torch.maximum(specbest[hv], spec)
    a = cover / len(offs)
    obj = acc / cover.clamp(min=1)[:, None]
    noise = torch.randn(len(ridx), 3, device=DEV, generator=gen) * bg.noise * 0.8
    obj_srgb = (lin_to_srgb(obj) + noise).clamp(0, 1)
    base = lin_to_srgb(color[ridx])
    color_srgb = lin_to_srgb(color)
    color_srgb[ridx] = base * (1 - a[:, None]) + obj_srgb * a[:, None]
    half = a >= 0.5
    depth[ridx[half]] = zbest[half]
    ids[ridx[half]] = idbest[half]
    spec_map[ridx[half]] = specbest[half]
    img = (color_srgb.reshape(H, W, 3).cpu().numpy() * 255 + 0.5).astype(np.uint8)[..., ::-1]
    # optics: the real photo is slightly soft; apply the same softness to rendered pixels only
    cov = np.zeros(H * W, np.float32)
    cov[ridx.cpu().numpy()] = a.cpu().numpy()
    cov = cv2.dilate(cov.reshape(H, W), np.ones((5, 5), np.uint8))
    soft = cv2.GaussianBlur(img, (0, 0), 0.75)
    img = np.where(cov[..., None] > 0, soft, img)
    return {"color": np.ascontiguousarray(img), "depth_true": depth.reshape(H, W).cpu().numpy(),
            "ids": ids.reshape(H, W).cpu().numpy(), "spec": spec_map.reshape(H, W).cpu().numpy()}


# ---------------------------------------------------------------- fully synthetic scenes

@dataclass
class FloorTexture:
    """Floor albedo on a mirrored repeat (no visible seams on a tile grid) with distance-based MIP."""
    levels: list                   # [(h,w,3) linear] blurred copies, sigma 0,1,2,4,8 texels
    size_m: tuple                  # (x, y) period of one texture copy on the floor
    res_m: float                   # texel size
    gloss: float = 0.18

    @classmethod
    def load(cls, path, size_m=None, gloss=0.18):
        import json as _json
        from pathlib import Path as _P
        im = cv2.imread(str(path))[..., ::-1].astype(np.float32) / 255
        meta = _P(path).with_suffix(".json")
        if size_m is None and meta.exists():
            size_m = tuple(_json.loads(meta.read_text())["size_m"])
        size_m = size_m if isinstance(size_m, tuple) else (size_m or 0.43, size_m or 0.43)
        levels = [im] + [cv2.GaussianBlur(im, (0, 0), s) for s in (1, 2, 4, 8, 16)]
        levels = [srgb_to_lin(torch.tensor(l.copy(), device=DEV)) for l in levels]
        return cls(levels, size_m, size_m[0] / im.shape[1], gloss)

    def at(self, xy, footprint_m=None):
        sx, sy = self.size_m
        u = torch.remainder(xy[:, 0] / sx, 2.0)
        v = torch.remainder(xy[:, 1] / sy, 2.0)
        u = torch.where(u > 1, 2 - u, u)
        v = torch.where(v > 1, 2 - v, v)
        g = torch.stack([v * 2 - 1, u * 2 - 1], -1)[None, None]     # image x = floor y, image y = floor x

        def samp(img):
            return F.grid_sample(img.permute(2, 0, 1)[None], g, align_corners=False, mode="bilinear")[0, :, 0].T
        if footprint_m is None:
            return samp(self.levels[2])
        lvl = torch.log2((footprint_m / self.res_m).clamp(min=1.0)).clamp(0, len(self.levels) - 1.001)
        lo = lvl.floor().long()
        out = torch.zeros(len(xy), 3, device=xy.device)
        for k in lo.unique().tolist():
            sel = lo == k
            a = (lvl[sel] - k)[:, None]
            gs = g[:, :, sel]
            s0 = F.grid_sample(self.levels[k].permute(2, 0, 1)[None], gs, align_corners=False)[0, :, 0].T
            s1 = F.grid_sample(self.levels[k + 1].permute(2, 0, 1)[None], gs, align_corners=False)[0, :, 0].T
            out[sel] = s0 * (1 - a) + s1 * a
        return out


ROOM = torch.tensor([0.55, 0.56, 0.58], device=DEV)


def sky_env(r, floor_tex, p, rough):
    col = ROOM.expand(len(r), 3).clone() * (0.8 + 0.4 * r[:, 2:3].clamp(0, 1))
    down = r[:, 2] < -1e-3
    if down.any() and floor_tex is not None:
        t = -p[down, 2] / r[down, 2]
        col[down] = floor_tex.at((p[down] + r[down] * t[:, None])[:, :2],
                                 0.004 + rough[down] * 0.04 + t * 0.01) * 0.8
    return col


def sphere_entry(parts, o, d):
    """First entry distance into any part's bounding sphere (inf if the ray misses all)."""
    best = torch.full((len(o),), float("inf"), device=o.device)
    for pt in parts:
        c = torch.tensor(pt.pos, device=o.device, dtype=o.dtype)
        oc = o - c
        b = (oc * d).sum(-1)
        cc = (oc * oc).sum(-1) - pt.radius ** 2
        disc = b * b - cc
        hit = disc > 0
        t0 = (-b - disc.clamp(min=0).sqrt()).clamp(min=0)
        best = torch.where(hit & (t0 < best) & (-b + disc.clamp(min=0).sqrt() > 0), t0, best)
    return best


def render_cg(parts: list[Part], cam: Camera, floor_tex: FloorTexture, light_pos=(1.0, -0.8, 2.6),
              irradiance=(1.6, 1.58, 1.52), ss: int = 2):
    """Fully synthetic frame: textured glossy floor + parts. Returns BGR uint8."""
    H, W = cam.H, cam.W
    yy, xx = torch.meshgrid(torch.arange(H, device=DEV), torch.arange(W, device=DEV), indexing="ij")
    base_pix = torch.stack([xx.reshape(-1), yy.reshape(-1)], -1).float()
    light = torch.tensor(light_pos, device=DEV, dtype=torch.float32)
    irr = torch.tensor(irradiance, device=DEV, dtype=torch.float32)
    offs = [(0.25, 0.25), (0.75, 0.25), (0.25, 0.75), (0.75, 0.75)] if ss >= 2 else [(0.5, 0.5)]
    out = torch.zeros(H * W, 3, device=DEV)
    for ox, oy in offs:
        o, d, _ = cam.rays(base_pix + torch.tensor([ox - 0.5, oy - 0.5], device=DEV))
        col = sky_env(d, None, o, torch.ones(len(o), device=DEV))
        t_floor = torch.where(d[:, 2] < -1e-4, -o[:, 2] / d[:, 2], torch.full((len(o),), 1e3, device=DEV))
        entry = sphere_entry(parts, o, d)
        cand = (entry < t_floor) & torch.isfinite(entry)
        t = torch.full((len(o),), 1e3, device=DEV)
        hit = torch.zeros(len(o), dtype=torch.bool, device=DEV)
        k = torch.full((len(o),), -1, dtype=torch.long, device=DEV)
        if cand.any():
            ci = cand.nonzero().squeeze(1)
            oc = o[ci] + d[ci] * entry[ci, None]
            tc, hc, kc = march(parts, oc, d[ci], tmax=3.0, steps=160)
            t[ci], hit[ci], k[ci] = tc + entry[ci], hc, kc
        obj = hit & (t < t_floor)
        fl = (~obj) & (t_floor < 1e3)
        if fl.any():
            fi = fl.nonzero().squeeze(1)
            P = o[fi] + d[fi] * t_floor[fi, None]
            foot = t_floor[fi] / cam.fx / d[fi, 2].abs().clamp(min=0.05)
            alb = floor_tex.at(P[:, :2], foot)
            lv = F.normalize(light[None] - P, dim=-1)
            cxy = torch.tensor([pt.pos[:2] for pt in parts], device=DEV, dtype=torch.float32)
            crad = torch.tensor([pt.radius for pt in parts], device=DEV)
            near = ((P[:, None, :2] - cxy[None]).norm(dim=-1) < crad[None] * 1.6 + 0.25).any(1)
            sh = torch.ones(len(P), device=DEV)
            ao = torch.ones(len(P), device=DEV)
            if near.any():
                ni = near.nonzero().squeeze(1)
                sh[ni] = soft_shadow(parts, P[ni] + torch.tensor([0, 0, 1e-3], device=DEV), lv[ni], tmax=3.0, k=10)
                ao[ni] = ambient_occlusion(parts, P[ni], torch.tensor([[0.0, 0, 1]], device=DEV).expand(len(ni), 3), 0.05)
            c = alb * irr * (0.45 * ao + 0.55 * sh)[:, None]
            # glossy floor: light hotspot and blurry reflection of objects
            v = -d[fi]
            n = torch.tensor([[0.0, 0, 1]], device=DEV).expand(len(P), 3)
            c = c + ggx_spec(n, v, lv, torch.full((len(P),), 0.25, device=DEV),
                             torch.full_like(c, 0.04)) * irr * 3.0 * sh[:, None]
            rd = d[fi] * torch.tensor([1.0, 1.0, -1.0], device=DEV)
            re = sphere_entry(parts, P, rd)
            rcand = torch.isfinite(re) & (re < 2.0)
            if rcand.any():
                ri = rcand.nonzero().squeeze(1)
                tr, hr, kr = march(parts, P[ri] + rd[ri] * re[ri, None], rd[ri], tmax=2.0, steps=80)
                if hr.any():
                    rc = torch.tensor([parts[j].material.color for j in kr[hr].tolist()], device=DEV) * irr * 0.6
                    w = floor_tex.gloss * torch.exp(-(tr[hr] + re[ri][hr]) * 3)
                    sub = ri[hr]
                    c[sub] = c[sub] * (1 - w[:, None]) + rc * w[:, None]
            dist = t_floor[fi]
            fog = torch.exp(-((dist - 3).clamp(min=0)) * 0.15)
            col[fi] = c * fog[:, None] + ROOM * (1 - fog[:, None])
        if obj.any():
            oi = obj.nonzero().squeeze(1)
            c, _, _ = shade_hits(parts, cam, None, o[oi], d[oi], t[oi], k[oi], irr.expand(len(oi), 3),
                                 light, floor_tex)
            col[oi] = c
        out += col
    out = out / len(offs)
    img = (lin_to_srgb(out).reshape(H, W, 3).cpu().numpy() * 255 + 0.5).astype(np.uint8)[..., ::-1]
    return np.ascontiguousarray(img)
