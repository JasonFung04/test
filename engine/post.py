"""HDR -> display pipeline: bloom, filmic tone map, grade, vignette, grain, dither."""
import cv2
import numpy as np

from .color import hex_srgb


def bloom(hdr, strength=1.0, threshold=0.0, tint=(1.0, 0.97, 0.92), levels=(0.12, 0.06, 0.03, 0.02)):
    """Multi-scale glow approximating radii ~4/16/48/96 px (at 1920 wide)."""
    if strength <= 0:
        return hdr
    H, W = hdr.shape[:2]
    s = W / 1920.0
    src = hdr if threshold <= 0 else np.maximum(hdr - threshold, 0)
    out = hdr.copy()
    radii = (4, 16, 48, 96)
    for rad, w in zip(radii, levels):
        rp = rad * s
        # blur on a downsampled copy for speed
        d = max(1, int(rp / 4))
        small = cv2.resize(src, (max(1, W // d), max(1, H // d)), interpolation=cv2.INTER_AREA)
        sig = max(rp / d / 2.0, 0.5)
        small = cv2.GaussianBlur(small, (0, 0), sig, borderType=cv2.BORDER_CONSTANT)
        up = cv2.resize(small, (W, H), interpolation=cv2.INTER_LINEAR)
        out += up * np.float32(w * strength)
    return out * np.asarray(tint, np.float32) if tint is not None else out


def aces(x):
    a, b, c, d, e = 2.51, 0.03, 2.43, 0.59, 0.14
    return np.clip((x * (a * x + b)) / (x * (c * x + d) + e), 0.0, 1.0)


def tonemap(hdr, exposure=0.0):
    x = hdr * np.float32(2.0 ** exposure)
    y = aces(x)
    return np.power(np.clip(y, 0, 1), 1 / 2.2).astype(np.float32)


class Grade:
    """Per-shot look. All params optional."""

    def __init__(self, exposure=0.0, wb=(1, 1, 1), sat=1.0, lift=(0, 0, 0), gamma=(1, 1, 1),
                 gain=(1, 1, 1), contrast=1.0, vignette=0.18, grain=0.006, bloom=1.0,
                 bloom_threshold=0.0, black=0.0, fade=1.0):
        self.__dict__.update(locals())
        del self.__dict__["self"]


def apply_grade(hdr, g: Grade, frame_idx=0, overlays=None):
    x = hdr * np.asarray(g.wb, np.float32)
    x = bloom(x, g.bloom, g.bloom_threshold)
    if g.sat != 1.0:
        lum = (x @ np.array([0.2126, 0.7152, 0.0722], np.float32))[..., None]
        x = lum + (x - lum) * np.float32(g.sat)
        x = np.maximum(x, 0)
    y = tonemap(x, g.exposure)
    lift = np.asarray(g.lift, np.float32)
    gain = np.asarray(g.gain, np.float32)
    gam = np.asarray(g.gamma, np.float32)
    y = np.clip(gain * (y + lift * (1 - y)), 0, 1) ** (1.0 / gam)
    if g.contrast != 1.0:
        y = np.clip(0.5 + (y - 0.5) * g.contrast, 0, 1)
    if g.black:
        y = np.clip((y - g.black) / (1 - g.black), 0, 1)
    H, W = y.shape[:2]
    if g.vignette:
        yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
        nx = (xx - W / 2) / (W / 2)
        ny = (yy - H / 2) / (W / 2)
        v = 1.0 - g.vignette * np.clip((nx * nx + ny * ny) * 0.9, 0, 1.5) ** 1.3
        y = y * v[..., None]
    if overlays:
        for ov in overlays:
            y = ov(y)
    if g.fade != 1.0:
        y = y * np.float32(g.fade)
    if g.grain:
        rng = np.random.default_rng(1000003 + frame_idx)
        n = rng.standard_normal((H, W), dtype=np.float32) * np.float32(g.grain)
        # grain is strongest in midtones, invisible in pure black (keeps black clean for the encoder)
        lum = y.mean(axis=2)
        y = y + (n * np.clip(lum * 4.0, 0, 1) * (1.0 - lum * 0.5))[..., None]
    return y


def to_uint8(y, frame_idx=0):
    rng = np.random.default_rng(7919 + frame_idx)
    d = rng.random(y.shape[:2], dtype=np.float32)[..., None] - 0.5
    return np.clip(y * 255.0 + d, 0, 255).astype(np.uint8)


def composite(y, rgba_color, alpha):
    """Display-space alpha composite. rgba_color: (3,) or HxWx3 in 0..1; alpha: HxW."""
    a = alpha[..., None]
    return y * (1 - a) + np.asarray(rgba_color, np.float32) * a


def solid(H, W, hexcol):
    return np.broadcast_to(hex_srgb(hexcol), (H, W, 3)).astype(np.float32)
