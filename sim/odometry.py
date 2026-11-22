"""Визуальная одометрия по полу.
Якорь — сам пол: плоскость и интринсики известны, поэтому точку пикселя на полу можно перевести
в метры в системе робота. Оптический поток между кадрами -> пары точек на полу -> жёсткое 2D-преобразование
(RANSAC + Кабш) -> смещение и поворот робота за кадр -> скорость. Сравнение с колёсной одометрией даёт slip."""
import cv2
import numpy as np

from geom import floor_ray_hit


def floor_mask(depth_m, plane, K_scale=1.0, tol=0.01):
    """Пиксели пола: глубина совпадает с пересечением луча и плоскости (нет объекта)."""
    h, w = depth_m.shape
    v, u = np.mgrid[0:h, 0:w].astype(np.float64) / K_scale
    _, t = floor_ray_hit(u, v, plane)
    z_expected = t
    return (depth_m > 0) & (np.abs(depth_m - z_expected) < tol + 0.02 * z_expected) & (z_expected < 3.0)


def rigid2d(a, b):
    ca, cb = a.mean(0), b.mean(0)
    H = (a - ca).T @ (b - cb)
    U, _, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T
    if np.linalg.det(R) < 0:
        Vt[1] *= -1
        R = Vt.T @ U.T
    return R, cb - R @ ca


def estimate_motion(img0, img1, plane, mask0=None, scale=1.0, max_pts=600, iters=200, thr=0.004, rng=None):
    """Возвращает (dx_forward, dx_lateral, dyaw, inliers, n) — движение робота между кадрами в метрах/радианах."""
    rng = rng or np.random.default_rng(0)
    g0 = cv2.cvtColor(img0, cv2.COLOR_BGR2GRAY)
    g1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    m = None if mask0 is None else (mask0.astype(np.uint8) * 255)
    p0 = cv2.goodFeaturesToTrack(g0, max_pts, 0.005, 6, mask=m, blockSize=7)
    if p0 is None or len(p0) < 8:
        return None
    p1, st, _ = cv2.calcOpticalFlowPyrLK(g0, g1, p0, None, winSize=(21, 21), maxLevel=4)
    pb, st2, _ = cv2.calcOpticalFlowPyrLK(g1, g0, p1, None, winSize=(21, 21), maxLevel=4)
    good = (st[:, 0] == 1) & (st2[:, 0] == 1) & (np.linalg.norm((pb - p0)[:, 0], axis=1) < 0.8)
    p0, p1 = p0[good, 0] / scale, p1[good, 0] / scale
    if len(p0) < 8:
        return None
    a, t0 = floor_ray_hit(p0[:, 0], p0[:, 1], plane)
    b, t1 = floor_ray_hit(p1[:, 0], p1[:, 1], plane)
    ok = (t0 > 0) & (t1 > 0) & (t0 < 3) & (t1 < 3)
    a, b = a[ok, :2], b[ok, :2]
    if len(a) < 8:
        return None
    best = None
    for _ in range(iters):
        idx = rng.choice(len(a), 2, replace=False)
        R, t = rigid2d(a[idx], b[idx])
        err = np.linalg.norm(a @ R.T + t - b, axis=1)
        inl = err < thr
        if best is None or inl.sum() > best.sum():
            best = inl
    R, t = rigid2d(a[best], b[best])
    Rinv = R.T
    tinv = -Rinv @ t
    dyaw = float(np.arctan2(Rinv[1, 0], Rinv[0, 0]))
    return {"forward": float(tinv[1]), "lateral": float(tinv[0]), "dyaw": dyaw,
            "inliers": int(best.sum()), "n": int(len(a))}
