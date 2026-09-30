"""Pinhole camera with an art-directable depth of field.

Focal length is expressed in mm on a 36 mm-wide sensor (see SHOTLIST.md).
Depth of field: `bokeh` is the circle-of-confusion radius (px @1920 wide) that an
object at infinity gets when the lens is focused at `focus`.  An object at depth
z gets  coc = bokeh * |1 - focus / z|, i.e. the thin-lens shape.
"""
import numpy as np

from .config import W as FULL_W, H as FULL_H, REF_W


def _norm(v):
    v = np.asarray(v, dtype=np.float64)
    return v / (np.linalg.norm(v) + 1e-12)


class Camera:
    def __init__(self, pos, target, up=(0.0, 1.0, 0.0), focal=35.0, roll=0.0,
                 focus=None, bokeh=0.0, W=FULL_W, H=FULL_H, near=1e-4, shift=(0.0, 0.0)):
        self.pos = np.asarray(pos, dtype=np.float64)
        self.target = np.asarray(target, dtype=np.float64)
        self.W, self.H = W, H
        self.s = W / REF_W
        self.focal = float(focal)
        self.fpx = self.focal / 36.0 * W
        self.near = near
        self.bokeh = float(bokeh) * self.s
        self.shift = (shift[0] * self.s, shift[1] * self.s)   # lens shift in px
        f = _norm(self.target - self.pos)
        upv = _norm(up)
        if abs(np.dot(f, upv)) > 0.999:
            upv = _norm(np.array([0.0, 0.0, 1.0]) if abs(f[2]) < 0.9 else np.array([1.0, 0.0, 0.0]))
        r = _norm(np.cross(f, upv))
        u = np.cross(r, f)
        if roll:
            c, s = np.cos(roll), np.sin(roll)
            r, u = r * c + u * s, -r * s + u * c
        self.fwd, self.right, self.up = f, r, u
        self.focus = float(focus) if focus is not None else float(np.linalg.norm(self.target - self.pos))

    # -- projection -----------------------------------------------------------
    def to_cam(self, P):
        d = np.asarray(P, dtype=np.float64) - self.pos
        return d @ self.right, d @ self.up, d @ self.fwd

    def project(self, P):
        """World points (N,3) -> x_px, y_px, depth, coc_px, valid mask (float32 arrays)."""
        xc, yc, zc = self.to_cam(P)
        valid = zc > self.near
        zs = np.where(valid, zc, 1.0)
        x = self.W * 0.5 + self.shift[0] + self.fpx * xc / zs
        y = self.H * 0.5 - self.shift[1] - self.fpx * yc / zs
        coc = self.bokeh * np.abs(1.0 - self.focus / zs) if self.bokeh > 0 else np.zeros_like(zs)
        return (x.astype(np.float32), y.astype(np.float32), zs.astype(np.float32),
                coc.astype(np.float32), valid)

    def project_dirs(self, D):
        """Directions at infinity (N,3) -> x, y, coc, valid."""
        D = np.asarray(D, dtype=np.float64)
        xc, yc, zc = D @ self.right, D @ self.up, D @ self.fwd
        valid = zc > 1e-6
        zs = np.where(valid, zc, 1.0)
        x = self.W * 0.5 + self.shift[0] + self.fpx * xc / zs
        y = self.H * 0.5 - self.shift[1] - self.fpx * yc / zs
        coc = np.full(zs.shape, self.bokeh, dtype=np.float32)
        return x.astype(np.float32), y.astype(np.float32), coc, valid

    def unproject(self, x, y, depth):
        """Screen px + depth -> world points."""
        xc = (np.asarray(x) - self.W * 0.5 - self.shift[0]) / self.fpx * depth
        yc = -(np.asarray(y) - self.H * 0.5 + self.shift[1]) / self.fpx * depth
        return (self.pos + np.outer(xc, self.right) + np.outer(yc, self.up)
                + np.outer(np.broadcast_to(depth, np.shape(xc)), self.fwd))

    @property
    def hfov(self):
        return 2 * np.arctan(18.0 / self.focal)


def orbit(center, radius, yaw, pitch):
    """Position on a sphere around `center` (yaw about +Y, pitch up from the XZ plane)."""
    c = np.asarray(center, dtype=np.float64)
    return c + radius * np.array([np.cos(pitch) * np.sin(yaw), np.sin(pitch), np.cos(pitch) * np.cos(yaw)])


def ease(t):
    """Smoothstep-ish cubic ease in/out on [0,1]."""
    t = np.clip(t, 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def ease5(t):
    t = np.clip(t, 0.0, 1.0)
    return t * t * t * (t * (t * 6 - 15) + 10)


def ease_out(t, p=3.0):
    t = np.clip(t, 0.0, 1.0)
    return 1.0 - (1.0 - t) ** p


def ease_in(t, p=3.0):
    t = np.clip(t, 0.0, 1.0)
    return t ** p


def lerp(a, b, t):
    return np.asarray(a, dtype=np.float64) + (np.asarray(b, dtype=np.float64) - np.asarray(a, dtype=np.float64)) * t


def seg(t, t0, t1):
    """Normalised position of t inside [t0, t1], clipped."""
    return float(np.clip((t - t0) / max(t1 - t0, 1e-9), 0.0, 1.0))


def handheld(t, amp=1.0, seed=0):
    """Smooth pseudo-random 3-vector drift for 'human' camera breathing."""
    rng = np.random.default_rng(seed)
    fr = rng.uniform(0.11, 0.37, size=(3, 3))
    ph = rng.uniform(0, 2 * np.pi, size=(3, 3))
    v = np.array([np.sum(np.sin(2 * np.pi * fr[i] * t + ph[i]) / np.arange(1, 4)) for i in range(3)])
    return amp * v / 1.83
