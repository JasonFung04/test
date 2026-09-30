"""Celestial bodies: the red giant, the senders' planet, Earth-like globes."""
import cv2
import numpy as np

from ..color import hex_lin
from ..noise import fbm, hash01


def fib_sphere(n, jitter=0.35, seed=0):
    i = np.arange(n) + 0.5
    phi = np.arccos(1 - 2 * i / n)
    th = np.pi * (1 + 5 ** 0.5) * i
    rng = np.random.default_rng(seed)
    phi = phi + rng.normal(0, jitter * np.sqrt(4 * np.pi / n), n)
    th = th + rng.normal(0, jitter * np.sqrt(4 * np.pi / n), n) / np.maximum(np.sin(phi), 0.05)
    return np.stack([np.sin(phi) * np.cos(th), np.cos(phi), np.sin(phi) * np.sin(th)], axis=1).astype(np.float32)


def sphere_mask(cam, center, radius, W, H, feather=1.0):
    """Anti-aliased screen mask of a sphere (1 inside)."""
    c = np.asarray(center, float)
    v = c - cam.pos
    d = np.linalg.norm(v)
    if d <= radius:
        return np.ones((H, W), np.float32)
    v /= d
    # limb circle: tangent points form a circle of radius r' at distance d' from the eye
    a = np.arcsin(radius / d)
    u1 = np.cross(v, [0, 1, 0])
    if np.linalg.norm(u1) < 1e-6:
        u1 = np.cross(v, [1, 0, 0])
    u1 /= np.linalg.norm(u1)
    u2 = np.cross(v, u1)
    th = np.linspace(0, 2 * np.pi, 720, endpoint=False)
    dirs = np.cos(a) * v[None] + np.sin(a) * (np.cos(th)[:, None] * u1 + np.sin(th)[:, None] * u2)
    P = cam.pos + dirs * d
    x, y, z, _, ok = cam.project(P)
    m = np.zeros((H * 4, W * 4), np.uint8)
    if ok.all():
        pts = np.stack([x * 4, y * 4], 1).astype(np.int32)
        cv2.fillPoly(m, [pts], 255, lineType=cv2.LINE_AA)
    m = cv2.resize(m.astype(np.float32) / 255.0, (W, H), interpolation=cv2.INTER_AREA)
    if feather > 1:
        m = cv2.GaussianBlur(m, (0, 0), feather)
    return m


class RedGiant:
    """A boiling, breathing red giant. Points live on the unit sphere; scale/translate per frame."""

    def __init__(self, n=900_000, seed=21):
        self.n = n
        self.N = fib_sphere(n, seed=seed)
        self.rng = np.random.default_rng(seed)
        self.core = hex_lin("#FF5A1F")
        self.limb = hex_lin("#8A1406")
        self.hi = hex_lin("#FFB36B")
        self.white = hex_lin("#FFE2B0")
        # prominence loops: footpoints on the sphere
        k = 7
        self.loops = []
        for j in range(k):
            a = self.rng.normal(size=3)
            a /= np.linalg.norm(a)
            b = a + self.rng.normal(size=3) * 0.12
            b /= np.linalg.norm(b)
            self.loops.append((a, b, self.rng.uniform(0.04, 0.11), self.rng.uniform(0, 1)))

    def radius(self, t, R0, period=4.0, amp=0.006, phase=0.0):
        return R0 * (1 + amp * np.sin(2 * np.pi * (t / period) + phase))

    def surface(self, t, center, R, cam_pos, energy=3.0, n_vis=None):
        C = np.asarray(center, float)
        view = cam_pos - C
        view /= np.linalg.norm(view)
        mu = self.N @ view
        vis = mu > -0.02
        Nv = self.N[vis]
        mu = np.clip(mu[vis], 0, 1)
        # slow boiling granulation: large convection cells + fine grain
        big = fbm(Nv * 2.2, octaves=3, offset=(t * 0.05, 0, t * 0.03))
        fine = fbm(Nv * 9.0, octaves=2, offset=(0, t * 0.21, 0))
        g = np.clip(0.55 + 1.35 * big + 0.35 * fine, 0, 1.6).astype(np.float32)
        limb = 0.28 + 0.72 * mu ** 0.65
        lum = (g * limb)[:, None]
        col = self.limb * (1 - np.clip(lum, 0, 1)) + self.core * np.clip(lum, 0, 1)
        col = col + self.hi * np.clip(lum - 0.85, 0, 1) * 1.5 + self.white * np.clip(lum - 1.2, 0, 1) * 2.0
        col *= (0.55 + 0.9 * lum) * energy
        P = C + Nv * R
        return P.astype(np.float32), col.astype(np.float32)

    def corona(self, t, center, R, cam_pos, n=60_000, energy=0.18):
        rng = np.random.default_rng(99)
        d = rng.standard_normal((n, 3))
        d /= np.linalg.norm(d, axis=1, keepdims=True)
        h = rng.exponential(0.05, n)
        drift = fbm(d * 3, octaves=2, offset=(t * 0.1, 0, 0)) * 0.03
        P = np.asarray(center) + d * (R * (1.0 + h + drift))[:, None]
        col = (self.core * 0.7 + self.hi * 0.3) * (energy * np.exp(-h / 0.05))[:, None]
        return P.astype(np.float32), col.astype(np.float32)

    def prominences(self, t, center, R, n_per=2500, energy=1.2):
        Ps, Cs = [], []
        C = np.asarray(center, float)
        for j, (a, b, hgt, ph) in enumerate(self.loops):
            s = hash01(np.arange(n_per), seed=j + 3)
            flow = (s + t * 0.03 + ph) % 1.0
            mid = (a + b)
            mid /= np.linalg.norm(mid)
            # quadratic Bezier arc lifted above the surface
            top = mid * (1 + hgt * 2.0)
            p = ((1 - flow) ** 2)[:, None] * a + (2 * (1 - flow) * flow)[:, None] * top + (flow ** 2)[:, None] * b
            jit = np.random.default_rng(j).normal(0, 0.004, (n_per, 3))
            P = C + (p + jit) * R
            fade = np.sin(np.pi * flow) ** 0.7
            col = (self.hi * 0.6 + self.core * 0.4) * (energy * fade)[:, None]
            Ps.append(P)
            Cs.append(col)
        return np.concatenate(Ps).astype(np.float32), np.concatenate(Cs).astype(np.float32)


class Planet:
    """The senders' world: dark body, city-light filigree on the night side, thin backlit atmosphere."""

    def __init__(self, n_lights=260_000, seed=8):
        rng = np.random.default_rng(seed)
        N = fib_sphere(n_lights * 3, jitter=0.9, seed=seed)
        # cities: phyllotaxis-ring clusters around a set of centres, plus webbing along noise ridges
        centres = fib_sphere(40, jitter=2.0, seed=seed + 1)
        d = N @ centres.T
        best = d.max(axis=1)
        idx = d.argmax(axis=1)
        ang = np.arccos(np.clip(best, -1, 1))
        rings = 0.5 + 0.5 * np.cos(ang * 180.0 + idx)
        ridge = 1.0 - np.abs(fbm(N * 3.0, octaves=4))
        dens = np.clip(np.exp(-ang / 0.22) * (0.4 + 0.6 * rings) + 0.55 * ridge ** 6, 0, 1)
        keep = rng.random(N.shape[0]) < dens
        self.N = N[keep][:n_lights]
        self.b = (0.3 + 0.7 * rng.random(self.N.shape[0]) ** 2).astype(np.float32)
        self.gold = hex_lin("#FFC46B")
        self.rim_col = hex_lin("#FF6A2A")

    def lights(self, center, R, cam_pos, star_dir, on=1.0, energy=1.0, alt=1.002):
        C = np.asarray(center, float)
        view = cam_pos - C
        view /= np.linalg.norm(view)
        mu = self.N @ view
        night = (self.N @ star_dir) < 0.08
        vis = (mu > 0.02) & night
        P = C + self.N[vis] * R * alt
        col = self.gold * (self.b[vis] * energy * on * (0.35 + 0.65 * mu[vis]))[:, None]
        return P.astype(np.float32), col.astype(np.float32), vis

    def rim(self, center, R, cam_pos, star_dir, n=40_000, energy=0.9, thick=0.025):
        rng = np.random.default_rng(5)
        C = np.asarray(center, float)
        v = cam_pos - C
        v /= np.linalg.norm(v)
        d = rng.standard_normal((n, 3))
        d -= np.outer(d @ v, v)
        d /= np.linalg.norm(d, axis=1, keepdims=True)
        h = np.abs(rng.normal(0, thick, n))
        forward = np.clip(d @ star_dir, -1, 1)
        w = np.clip(0.25 + 0.75 * (forward * 0.5 + 0.5), 0, 1) ** 2.2
        P = C + d * (R * (1 + h))[:, None] - v * R * 0.02
        col = self.rim_col * (energy * w * np.exp(-h / thick))[:, None]
        return P.astype(np.float32), col.astype(np.float32)
