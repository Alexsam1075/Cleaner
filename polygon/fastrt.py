"""Real-time renderer: the fused CUDA kernel in fastrt_kernel.cu, compiled at runtime with NVRTC
(from the torch wheel) and launched through the CUDA driver API via ctypes. Works on torch tensors,
so no extra packages are needed. Scene = table of primitives + table of materials.
"""
from __future__ import annotations

import ctypes
import glob
import hashlib
import math
import os
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch

DEV = torch.device("cuda")
HERE = Path(__file__).resolve().parent
PS, MS = 32, 24
TYPES = {"sphere": 0, "rbox": 1, "rcyl": 2, "capsule": 3, "ellipsoid": 4, "torus": 5}
TEXTURES = {"plain": 0, "fbm": 1, "stone": 2, "wood": 3, "brushed": 4, "paper": 5, "label": 6, "bristle": 7, "robot": 8, "wheel": 9}


# ---------------------------------------------------------------- NVRTC + driver
def _find_nvrtc():
    import torch as _t
    roots = [Path(_t.__file__).parent.parent / "nvidia" / "cuda_nvrtc" / "lib"]
    for r in roots:
        hits = sorted(glob.glob(str(r / "libnvrtc.so.12*")))
        if hits:
            return hits[0]
    raise FileNotFoundError("libnvrtc.so.12 not found next to torch")


class _Cuda:
    _inst = None

    def __init__(self):
        torch.zeros(1, device=DEV)                       # make torch's primary context current
        self.nvrtc = ctypes.CDLL(_find_nvrtc())
        self.cu = ctypes.CDLL("libcuda.so.1")
        src = (HERE / "fastrt_kernel.cu").read_text()
        major, minor = torch.cuda.get_device_capability()
        arch = f"compute_{major}{minor}"
        key = hashlib.sha1((src + arch).encode()).hexdigest()[:12]
        cache = Path.home() / ".cache" / "polygon_fastrt" / f"{key}.ptx"
        if cache.exists():
            ptx = cache.read_bytes()
        else:
            ptx = self._compile(src, arch)
            cache.parent.mkdir(parents=True, exist_ok=True)
            cache.write_bytes(ptx)
        self.module = ctypes.c_void_p()
        self._chk(self.cu.cuModuleLoadData(ctypes.byref(self.module), ctypes.c_char_p(ptx)), "cuModuleLoadData")
        self.fn = {}
        for name in ("render", "sdf_points"):
            f = ctypes.c_void_p()
            self._chk(self.cu.cuModuleGetFunction(ctypes.byref(f), self.module, name.encode()), name)
            self.fn[name] = f

    def _compile(self, src, arch):
        prog = ctypes.c_void_p()
        r = self.nvrtc.nvrtcCreateProgram(ctypes.byref(prog), src.encode(), b"fastrt.cu", 0, None, None)
        if r:
            raise RuntimeError(f"nvrtcCreateProgram {r}")
        opts = [f"--gpu-architecture={arch}".encode(), b"--use_fast_math", b"-std=c++14"]
        arr = (ctypes.c_char_p * len(opts))(*opts)
        r = self.nvrtc.nvrtcCompileProgram(prog, len(opts), arr)
        size = ctypes.c_size_t()
        self.nvrtc.nvrtcGetProgramLogSize(prog, ctypes.byref(size))
        log = ctypes.create_string_buffer(size.value)
        self.nvrtc.nvrtcGetProgramLog(prog, log)
        if r:
            raise RuntimeError("NVRTC compile failed:\n" + log.value.decode(errors="replace"))
        self.nvrtc.nvrtcGetPTXSize(prog, ctypes.byref(size))
        ptx = ctypes.create_string_buffer(size.value)
        self.nvrtc.nvrtcGetPTX(prog, ptx)
        return ptx.raw

    def _chk(self, r, what):
        if r:
            raise RuntimeError(f"CUDA driver error {r} in {what}")

    @classmethod
    def get(cls):
        if cls._inst is None:
            cls._inst = cls()
        return cls._inst

    def launch(self, name, grid, block, args):
        holders = []
        for a in args:
            if isinstance(a, torch.Tensor):
                holders.append(ctypes.c_uint64(a.data_ptr()))
            elif isinstance(a, bool):
                holders.append(ctypes.c_int(int(a)))
            elif isinstance(a, int):
                holders.append(ctypes.c_int(a))
            elif isinstance(a, np.uint32):
                holders.append(ctypes.c_uint(int(a)))
            else:
                holders.append(ctypes.c_float(float(a)))
        ptrs = (ctypes.c_void_p * len(holders))(*[ctypes.cast(ctypes.byref(h), ctypes.c_void_p) for h in holders])
        stream = ctypes.c_void_p(torch.cuda.current_stream().cuda_stream)
        self._chk(self.cu.cuLaunchKernel(self.fn[name], grid[0], grid[1], 1, block[0], block[1], 1, 0, stream, ptrs, None),
                  "cuLaunchKernel")


# ---------------------------------------------------------------- scene description
def rot_z(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1.0]])


def rot_x(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])


def rot_y(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


LAY_X = np.array([[0, 0, 1.0], [0, 1, 0], [-1, 0, 0]])   # local z axis -> world x (lay a cylinder down)


@dataclass
class Mat:
    color: tuple = (0.5, 0.5, 0.5)
    color2: tuple = (0.2, 0.2, 0.2)
    roughness: float = 0.5
    metallic: float = 0.0
    transmission: float = 0.0
    ior: float = 1.5
    tint: tuple = (0.9, 0.95, 0.9)
    texture: str = "plain"
    tex_scale: float = 40.0
    tex_amp: float = 0.25
    clearcoat: float = 0.0

    def row(self):
        r = np.zeros(MS, np.float32)
        r[0:3], r[3:6] = self.color, self.color2
        r[6:10] = self.roughness, self.metallic, self.transmission, self.ior
        r[10:13] = self.tint
        r[13:17] = TEXTURES[self.texture], self.tex_scale, self.tex_amp, self.clearcoat
        return r


@dataclass
class Prim:
    type: str
    params: tuple
    pos: tuple = (0.0, 0.0, 0.0)
    rot: np.ndarray = field(default_factory=lambda: np.eye(3))   # local -> world
    mat: int = 0
    obj: int = -1
    group: int = -1
    smooth: float = 0.0
    disp: tuple = (0.0, 1.0, 0.0)                                # amplitude, frequency, seed
    flags: int = 0                                                # 1 crumple, 2 clip at floor, 4 cheap noise
    bound: float = 0.1
    shadowless: bool = False

    def row(self):
        r = np.zeros(PS, np.float32)
        r[0], r[1], r[2], r[3] = TYPES[self.type], self.mat, self.obj, self.group
        r[4:7] = self.pos
        r[7:16] = np.asarray(self.rot, np.float64).T.reshape(-1)     # world -> local rows
        p = list(self.params) + [0.0] * (8 - len(self.params))
        r[16:24] = p
        r[24:27] = self.disp
        r[27], r[28], r[29], r[30] = self.smooth, self.bound, self.flags, float(self.shadowless)
        return r


class Scene:
    def __init__(self):
        self.prims: list[Prim] = []
        self.mats: list[Mat] = []
        self._group = 0

    def mat(self, m: Mat) -> int:
        self.mats.append(m)
        return len(self.mats) - 1

    def new_group(self) -> int:
        self._group += 1
        return self._group

    def add(self, p: Prim):
        self.prims.append(p)
        return p

    def tensors(self):
        P = torch.tensor(np.stack([p.row() for p in self.prims]) if self.prims else np.zeros((1, PS), np.float32), device=DEV)
        M = torch.tensor(np.stack([m.row() for m in self.mats]) if self.mats else np.zeros((1, MS), np.float32), device=DEV)
        return P, M


def transform(prims: list[Prim], pos, yaw: float) -> list[Prim]:
    """Place prims built in an object frame (origin on the floor) at world pos with yaw."""
    R = rot_z(yaw)
    out = []
    for p in prims:
        q = Prim(**{**p.__dict__})
        q.pos = tuple(np.asarray(pos, float) + R @ np.asarray(p.pos, float))
        q.rot = R @ np.asarray(p.rot, float)
        out.append(q)
    return out


# ---------------------------------------------------------------- camera / textures
@dataclass
class FastCam:
    fx: float
    fy: float
    cx: float
    cy: float
    W: int
    H: int
    origin: np.ndarray
    R: np.ndarray        # camera -> world

    @classmethod
    def from_plane(cls, cam: dict, plane, W=None, H=None):
        p = np.asarray(plane, float)
        n, d = p[:3] / np.linalg.norm(p[:3]), p[3] / np.linalg.norm(p[:3])
        if d < 0:
            n, d = -n, -d
        fwd = np.array([0, 0, 1.0]) - n * n[2]
        fwd /= np.linalg.norm(fwd)
        side = np.cross(n, fwd)
        s = (W or cam["width"]) / cam["width"]
        return cls(cam["fx"] * s, cam["fy"] * s, cam["cx"] * s, cam["cy"] * s, W or cam["width"],
                   H or cam["height"], np.array([0, 0, d]), np.stack([fwd, side, n]))

    @classmethod
    def look_at(cls, eye, target, W=1280, H=720, fov_deg=60.0):
        eye, target = np.asarray(eye, float), np.asarray(target, float)
        z = target - eye
        z /= np.linalg.norm(z)
        x = np.cross(z, [0, 0, 1.0])
        x /= np.linalg.norm(x)
        y = np.cross(z, x)
        f = 0.5 * W / math.tan(math.radians(fov_deg) / 2)
        return cls(f, f, W / 2, H / 2, W, H, eye, np.stack([x, y, z], 1))

    @classmethod
    def from_pose(cls, cam: dict, eye, target):
        """Dataset intrinsics (camera.json) at a world pose: what the robot's D435 would record."""
        c = cls.look_at(eye, target, W=cam["width"], H=cam["height"])
        c.fx, c.fy, c.cx, c.cy = cam["fx"], cam["fy"], cam["cx"], cam["cy"]
        return c

    def floor_plane(self):
        """Dataset-style floor plane in camera coordinates: n . X + d = 0, n toward the camera."""
        up_c = self.R.T @ np.array([0.0, 0.0, 1.0])
        return [float(v) for v in up_c] + [float(self.origin[2])]

    def project(self, p):
        q = (np.asarray(p, float) - self.origin) @ self.R
        return np.array([q[0] / q[2] * self.fx + self.cx, q[1] / q[2] * self.fy + self.cy]), q[2]

    def ray(self, px, py):
        d = self.R @ np.array([(px - self.cx) / self.fx, (py - self.cy) / self.fy, 1.0])
        return self.origin, d / np.linalg.norm(d)


class FloorTex:
    def __init__(self, path, size_m=None):
        import cv2
        import json
        im = cv2.imread(str(path))[..., ::-1].astype(np.float32) / 255
        meta = Path(path).with_suffix(".json")
        if size_m is None:
            size_m = tuple(json.loads(meta.read_text())["size_m"]) if meta.exists() else (0.43, 0.43)
        self.size = size_m
        levels = [im] + [cv2.GaussianBlur(im, (0, 0), s) for s in (1, 2, 4, 8, 16)]
        lin = [np.where(l <= 0.04045, l / 12.92, ((l + 0.055) / 1.055) ** 2.4) for l in levels]
        self.t = torch.tensor(np.stack(lin), device=DEV, dtype=torch.float32).contiguous()
        self.h, self.w = im.shape[:2]


def srgb_to_lin(x):
    return torch.where(x <= 0.04045, x / 12.92, ((x + 0.055) / 1.055) ** 2.4)


def lin_to_srgb(x):
    x = x.clamp(0, 1)
    return torch.where(x <= 0.0031308, x * 12.92, 1.055 * x ** (1 / 2.4) - 0.055)


# ---------------------------------------------------------------- render entry points
_dummy = None


def _dummy_t():
    global _dummy
    if _dummy is None:
        _dummy = torch.zeros(8, device=DEV)
    return _dummy


def render(scene: Scene, cam: FastCam, *, floor: FloorTex | None = None, bg=None, light=(1.0, -0.8, 2.6),
           irradiance=(1.6, 1.58, 1.52), room=(0.55, 0.56, 0.58), spp=1, seed=0, gloss=0.18):
    """Return dict of torch tensors: rgb (H,W,3) linear, depth (H,W), id (H,W) int32, spec, cov."""
    cu = _Cuda.get()
    P, M = scene.tensors()
    H, W = cam.H, cam.W
    out_rgb = torch.empty(H, W, 3, device=DEV)
    out_depth = torch.empty(H, W, device=DEV)
    out_id = torch.empty(H, W, device=DEV, dtype=torch.int32)
    out_spec = torch.empty(H, W, device=DEV)
    out_cov = torch.empty(H, W, device=DEV)
    R = torch.tensor(cam.R.reshape(-1), device=DEV, dtype=torch.float32)
    d = _dummy_t()
    mode = 1 if bg is not None else 0
    ft = floor.t if floor is not None else d
    fw, fh, fl = (floor.w, floor.h, floor.t.shape[0]) if floor is not None else (1, 1, 1)
    fsx, fsy = floor.size if floor is not None else (1.0, 1.0)
    if bg is not None:
        bgl, bgs, bgb, bgd, bgf = bg["lin"], bg["blur_s"], bg["blur_l"], bg["depth"], bg["floor"]
        room = tuple(bg["room"])
        light = tuple(bg["light"])
    else:
        bgl = bgs = bgb = bgd = bgf = d
    args = [P, len(scene.prims), M, cam.fx, cam.fy, cam.cx, cam.cy, W, H, *map(float, cam.origin), R,
            mode, ft, fw, fh, fl, fsx, fsy, gloss, bgl, bgs, bgb, bgd, bgf, *room, *light, *irradiance,
            int(spp), np.uint32(seed), out_rgb, out_depth, out_id, out_spec, out_cov]
    cu.launch("render", ((W + 15) // 16, (H + 7) // 8), (16, 8), args)
    return {"rgb": out_rgb, "depth": out_depth, "id": out_id, "spec": out_spec, "cov": out_cov}


def sdf_points(scene: Scene, pts: torch.Tensor) -> torch.Tensor:
    cu = _Cuda.get()
    P, _ = scene.tensors()
    pts = pts.contiguous().float()
    out = torch.empty(len(pts), device=DEV)
    cu.launch("sdf_points", ((len(pts) + 255) // 256, 1), (256, 1), [P, len(scene.prims), pts, len(pts), out])
    return out


def volume_ml(scene: Scene, center_xy, radius, height, step=None) -> float:
    step = step or max(radius, height) / 100
    xs = torch.arange(-radius, radius, step, device=DEV) + step / 2
    zs = torch.arange(0, height, step, device=DEV) + step / 2
    total = 0
    for z0 in range(0, len(zs), 32):
        g = torch.stack(torch.meshgrid(xs + center_xy[0], xs + center_xy[1], zs[z0:z0 + 32], indexing="ij"), -1).reshape(-1, 3)
        total += int((sdf_points(scene, g) < 0).sum())
    return total * step ** 3 * 1e6


def background(rgb_u8: np.ndarray, depth_u16: np.ndarray, cam: FastCam):
    """Real frame as the kernel's background: linear colour, blurs, depth, floor mask, room, light."""
    import cv2
    lin = srgb_to_lin(torch.tensor(rgb_u8[..., ::-1].copy(), device=DEV, dtype=torch.float32) / 255).contiguous()
    ln = lin.cpu().numpy()
    bs = torch.tensor(cv2.GaussianBlur(ln, (0, 0), 2.5), device=DEV).contiguous()
    bl = torch.tensor(cv2.GaussianBlur(ln, (0, 0), 14), device=DEV).contiguous()
    depth = torch.tensor(depth_u16.astype(np.float32) / 1000, device=DEV).contiguous()
    H, W = depth.shape
    yy, xx = np.mgrid[0:H, 0:W]
    dc = np.stack([(xx - cam.cx) / cam.fx, (yy - cam.cy) / cam.fy, np.ones_like(xx, dtype=float)], -1)
    pts = (dc * (depth_u16 / 1000.0)[..., None]) @ cam.R.T + cam.origin
    floor = (np.abs(pts[..., 2]) < 0.01) & (depth_u16 > 0)
    room = bl[: H // 8].reshape(-1, 3).mean(0).cpu().numpy() * 1.3
    lum = cv2.GaussianBlur(ln.mean(-1), (0, 0), 14).copy()
    lum[~floor] = 0
    lum[: H // 10] = 0
    y, x = np.unravel_index(int(lum.argmax()), lum.shape)
    med = float(np.median(ln.mean(-1)[floor])) if floor.any() else 1.0
    light = np.array([1.6, 0.2, 2.6])
    if lum[y, x] > 1.25 * med:
        o, dvec = cam.ray(x, y)
        P = o + dvec * (-o[2] / dvec[2])
        mirror = np.array([0, 0, -cam.origin[2]])
        v = P - mirror
        light = mirror + v * ((2.6 - mirror[2]) / v[2])
    g = cv2.cvtColor(rgb_u8, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255
    noise = float(np.std((g - cv2.GaussianBlur(g, (0, 0), 1.5))[floor])) if floor.any() else 0.01
    return {"lin": lin, "blur_s": bs, "blur_l": bl, "depth": depth,
            "floor": torch.tensor(floor.astype(np.float32), device=DEV).contiguous(),
            "floor_np": floor, "room": room, "light": light, "noise": noise}


def to_u8(rgb_lin: torch.Tensor) -> np.ndarray:
    return (lin_to_srgb(rgb_lin) * 255 + 0.5).to(torch.uint8).cpu().numpy()[..., ::-1].copy()
