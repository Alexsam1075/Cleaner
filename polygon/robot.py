"""Astramis cleaning robot digitised from photos (white body, red trim, black base, two side brushes,
front D435 at 0.25 m looking down ~30°, lidar, tablet and e-stop on top, removable rear bin).

Robot frame: x forward, y left, z up, origin on the floor under the body centre. Sizes in metres
are estimated from the photos and the dataset camera (height 0.25 m, robot width ~0.48 m).
"""
from __future__ import annotations

import math

import torch

from . import sdf as S
from .render import Material, Part

WHITE = (0.78, 0.78, 0.77)
RED = (0.83, 0.055, 0.04)
BLACK = (0.018, 0.018, 0.02)
GREY = (0.35, 0.35, 0.36)
ORANGE = (0.9, 0.18, 0.05)

HX, HY, HZ = 0.31, 0.24, 0.20          # body half sizes
BODY_Z = 0.17 + HZ                    # body centre height
TOP = BODY_Z + HZ
CAM_POS = (HX + 0.02, 0.0, 0.25)      # D435 in the front bar
CAM_PITCH_DEG = 30.0


def _star_outline(y, z, r=0.085, thick=0.006):
    """Distance to a five-pointed star outline (iq's sdStar5) in the (y, z) plane."""
    k1x, k1y = 0.809016994375, -0.587785252292
    k2x, k2y = -k1x, k1y
    px, py = y.abs(), z.clone()
    d1 = (k1x * px + k1y * py).clamp(min=0)
    px, py = px - 2 * d1 * k1x, py - 2 * d1 * k1y
    d2 = (k2x * px + k2y * py).clamp(min=0)
    px, py = px - 2 * d2 * k2x, py - 2 * d2 * k2y
    px = px.abs()
    py = py - r
    rf = 0.42
    bax, bay = rf * (-k1y) - 0.0, rf * k1x - 1.0
    h = ((px * bax + py * bay) / (bax * bax + bay * bay)).clamp(0, r)
    dist = ((px - bax * h) ** 2 + (py - bay * h) ** 2).sqrt() * torch.sign(py * bax - px * bay)
    return dist.abs() - thick


def _body_color(p, n):
    """White panels, red trim along every edge, black star logo and rivets on the front panel."""
    q = p - torch.tensor([0.0, 0.0, BODY_Z], device=p.device)
    e = 0.04
    near = torch.stack([(q[:, 0].abs() > HX - e), (q[:, 1].abs() > HY - e), (q[:, 2].abs() > HZ - e)], -1).float().sum(-1)
    col = torch.tensor(WHITE, device=p.device).expand(len(p), 3).clone()
    col[near >= 2] = torch.tensor(RED, device=p.device)
    front = q[:, 0] > HX - 0.006
    star = _star_outline(q[:, 1] + 0.01, q[:, 2] - 0.02) < 0
    col[front & star] = torch.tensor(BLACK, device=p.device)
    dot = ((q[:, 1] + 0.12) ** 2 + (q[:, 2] - 0.06) ** 2).sqrt() < 0.012
    col[front & dot] = torch.tensor(RED, device=p.device)
    # rivets: small black dots on a grid near the trim
    g = (torch.remainder(q[:, :3] + 0.5, 0.09) - 0.045)
    rivet = (near == 1) & ((g[:, 0] ** 2 + g[:, 1] ** 2 + g[:, 2] ** 2).sqrt() < 0.006)
    col[rivet] = torch.tensor(BLACK, device=p.device)
    return col


def build_robot(x: float, y: float, yaw: float, brush_angle: float = 0.0, intake_open: bool = True) -> list[Part]:
    parts: list[Part] = []

    def add(fn, mat, radius, local=(0.0, 0.0, 0.0), no_shadow=False):
        c, s = math.cos(yaw), math.sin(yaw)
        lx, ly, lz = local
        parts.append(Part(fn, mat, radius, (x + c * lx - s * ly, y + s * lx + c * ly, lz), yaw, -1, no_shadow))

    white_red = Material(WHITE, 0.35, color_fn=_body_color, clearcoat=0.5)
    add(lambda p: S.round_box(p - torch.tensor([0.0, 0.0, BODY_Z], device=p.device), (HX, HY, HZ), 0.035),
        white_red, 0.76)
    # black base with rounded ends
    add(lambda p: S.round_box(p - torch.tensor([0.0, 0.0, 0.12], device=p.device), (HX + 0.02, HY + 0.015, 0.05), 0.04),
        Material(BLACK, 0.6, texture="fbm", tex_scale=200, tex_amp=0.3), 0.45)
    # wheels
    for wx in (-0.2, 0.2):
        for wy in (-1, 1):
            add(lambda p, wx=wx, wy=wy: S.round_cylinder_z(S.rot_x(p - torch.tensor([wx, wy * (HY + 0.03), 0.085], device=p.device), math.pi / 2), 0.085, 0.03, 0.012),
                Material(BLACK, 0.8, texture="fbm", tex_scale=300, tex_amp=0.4), 0.95)
    # front bar with the D435
    add(lambda p: S.round_box(p - torch.tensor([HX + 0.012, 0.0, 0.25], device=p.device), (0.014, 0.10, 0.028), 0.008),
        Material(RED, 0.4), 0.45)
    add(lambda p: S.round_box(p - torch.tensor([HX + 0.027, 0.0, 0.25], device=p.device), (0.004, 0.045, 0.012), 0.004),
        Material(BLACK, 0.15, clearcoat=1.0), 0.45)
    # side brushes: bristle discs ahead of the body, hubs and arms
    for side in (-1, 1):
        bx, by = HX + 0.10, side * 0.15
        ang = brush_angle * side

        def brush(p, bx=bx, by=by, ang=ang):
            q = S.rot_z(p - torch.tensor([bx, by, 0.025], device=p.device), ang)
            rr = q[:, :2].norm(dim=-1)
            fuzz = 0.004 * S.value_noise(torch.stack([torch.atan2(q[:, 1], q[:, 0]) * 40, rr * 30, q[:, 2] * 30], -1))
            cone = S.round_cylinder_z(q, 0.16, 0.022, 0.01) + (0.16 - rr).clamp(min=0) * 0.08
            return cone + fuzz
        add(brush, Material((0.03, 0.03, 0.035), 0.75, texture="bristle"), 0.8)
        add(lambda p, bx=bx, by=by: S.round_cylinder_z(p - torch.tensor([bx, by, 0.065], device=p.device), 0.04, 0.025, 0.006),
            Material(BLACK, 0.3, clearcoat=0.5), 0.8)
        add(lambda p, bx=bx, by=by: S.capsule(p, (HX - 0.05, by * 0.6, 0.13), (bx, by, 0.08), 0.018),
            Material(BLACK, 0.45), 0.8)
    # top: lidar puck, tablet, e-stop
    add(lambda p: S.round_cylinder_z(p - torch.tensor([0.12, 0.0, TOP + 0.03], device=p.device), 0.045, 0.03, 0.006),
        Material((0.06, 0.06, 0.065), 0.35), 0.75)
    add(lambda p: S.round_box(p - torch.tensor([-0.06, 0.02, TOP + 0.006], device=p.device), (0.13, 0.09, 0.006), 0.004),
        Material((0.04, 0.045, 0.05), 0.2, clearcoat=1.0), 0.82)
    add(lambda p: S.round_cylinder_z(p - torch.tensor([-0.21, -0.15, TOP + 0.016], device=p.device), 0.022, 0.016, 0.008),
        Material(RED, 0.3, clearcoat=1.0), 0.8)
    # rear removable bin with orange rings
    add(lambda p: S.round_box(p - torch.tensor([-HX - 0.06, 0.0, 0.12], device=p.device), (0.06, 0.17, 0.09), 0.012),
        Material(WHITE, 0.45), 0.6)
    for ry in (-0.08, 0.08):
        add(lambda p, ry=ry: S.torus_z(S.rot_y(p - torch.tensor([-HX - 0.122, ry, 0.11], device=p.device), math.pi / 2), 0.03, 0.009),
            Material(ORANGE, 0.4), 0.6)
    return parts


def robot_camera_pose(x: float, y: float, yaw: float):
    """World eye and look-at target of the robot's front D435."""
    c, s = math.cos(yaw), math.sin(yaw)
    cx, cy, cz = CAM_POS
    eye = (x + c * cx - s * cy, y + s * cx + c * cy, cz)
    pitch = math.radians(CAM_PITCH_DEG)
    fwd = (c * math.cos(pitch), s * math.cos(pitch), -math.sin(pitch))
    return eye, tuple(e + f for e, f in zip(eye, fwd))
