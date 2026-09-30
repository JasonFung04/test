"""ACT II — ON THE WAY (II1–II8).  II9 lives in act3.

II1  nebula: the shell front sweeps through pillars of gas (light echo), camera rides alongside
II2  the galaxy from 32 deg above the disk; a gold sphere grows from the halo toward the Sun
II3  black card
II4  Blombos Cave, 73,000 years ago
II5  the six parallel strokes (hand rhyme)
II6  her profile in firelight
II7  the three crossing strokes
II8  push-in on the nine lines -> match cut to the city (crosshatch.MATCH_SPAN contract)
"""
import numpy as np

from .. import timeline as TL
from ..assets import sky
from ..assets.shell import Shell
from ..camera import Camera, ease, ease5, lerp, seg
from ..color import blackbody, hex_lin
from ..post import Grade
from .common import black, dispatch

_C = {}


def _get(name, fn):
    if name not in _C:
        _C[name] = fn()
    return _C[name]


def shell():
    return _get("shell", lambda: Shell(seed=7, axis=(0.25, 0.94, -0.22)))


# ============================================================================ II2 — the galaxy
# Galaxy frame: disk in XZ, radius ~1 (= 50,000 ly). The Sun sits 26,000 ly (0.52) from the
# centre in an inter-arm gap; the source is on the far side, 0.40 above the plane in the halo,
# 73,000 ly (1.46) from the Sun (BIBLE 6).
SUN_TH = np.radians(22.0)
SUN_R = 0.52
SRC_TH = SUN_TH + np.pi
SRC_R = 0.884
SRC_H = 0.40
GAL_SPIN = np.radians(-1.6)          # total disk rotation over the shot (arms trail)


def _galaxy():
    g = sky.spiral_galaxy()
    rng = np.random.default_rng(12)
    pos, rgb, kind = g["pos"], g["rgb"], g["kind"]
    n = pos.shape[0]
    u = rng.random(n)
    # a sparse subset becomes crisp resolved stars; the rest is soft unresolved glow
    star = (u < 0.06) | (kind == 3)
    return dict(pos=pos.astype(np.float32), rgb=rgb.astype(np.float32), kind=kind, star=star,
                glow_idx=np.nonzero(~star)[0], star_idx=np.nonzero(star)[0])


def _halo():
    """Old stars of the halo + a few globular clusters (the senders' neighbourhood)."""
    rng = np.random.default_rng(31)
    n = 26_000
    rr = 0.08 * (rng.random(n) ** -0.55)                  # steep power law
    rr = rr[rr < 1.7]
    d = rng.standard_normal((rr.size, 3))
    d /= np.linalg.norm(d, axis=1, keepdims=True)
    d[:, 1] *= 0.8
    P = d * rr[:, None]
    T = rng.uniform(3300, 5200, rr.size)
    col = blackbody(T) * (0.25 + 0.75 * rng.random(rr.size) ** 3)[:, None]
    # globular clusters
    Ps, Cs = [P], [col]
    for k in range(14):
        c = rng.standard_normal(3)
        c /= np.linalg.norm(c)
        c *= rng.uniform(0.35, 1.3)
        c[1] = np.sign(c[1] or 1) * max(abs(c[1]), 0.12)
        m = 500
        pl = rng.standard_normal((m, 3))
        pl /= np.linalg.norm(pl, axis=1, keepdims=True)
        rad = 0.012 * (rng.random(m) ** (-2 / 3) - 1) ** -0.5
        rad = np.clip(rad, 0, 0.05)
        Ps.append(c + pl * rad[:, None])
        Cs.append(blackbody(rng.uniform(3800, 5600, m)) * (0.4 + 0.6 * rng.random(m))[:, None])
    return np.concatenate(Ps).astype(np.float32), np.concatenate(Cs).astype(np.float32)


def _distant_galaxies():
    rng = np.random.default_rng(77)
    Ds, Cs = [], []
    for k in range(60):
        c = rng.standard_normal(3)
        c /= np.linalg.norm(c)
        a = np.cross(c, [0, 1, 0])
        a /= np.linalg.norm(a)
        b = np.cross(c, a)
        m = 60
        ang = rng.uniform(0, np.pi)
        e = rng.uniform(0.2, 1.0)
        size = rng.uniform(0.0006, 0.0025)
        x, y = rng.standard_normal(m) * size, rng.standard_normal(m) * size * e
        xa = x * np.cos(ang) - y * np.sin(ang)
        ya = x * np.sin(ang) + y * np.cos(ang)
        D = c + a * xa[:, None] + b * ya[:, None]
        Ds.append(D / np.linalg.norm(D, axis=1, keepdims=True))
        tint = hex_lin("#FFE6C8") if rng.random() < 0.6 else hex_lin("#C8D8FF")
        Cs.append(np.broadcast_to(tint * rng.uniform(0.002, 0.012), (m, 3)))
    return np.concatenate(Ds).astype(np.float32), np.concatenate(Cs).astype(np.float32)


def _yrot(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


def _gal_frame():
    """World = galaxy frame. Source/Sun and the fixed camera."""
    sun0 = np.array([SUN_R * np.cos(SUN_TH), 0.0, SUN_R * np.sin(SUN_TH)])
    src = np.array([SRC_R * np.cos(SRC_TH), SRC_H, SRC_R * np.sin(SRC_TH)])
    # camera looks across the source->Sun line from 32 deg above the disk
    axis = sun0 - src
    axis[1] = 0
    axis /= np.linalg.norm(axis)
    side = np.cross(axis, [0, 1, 0])                     # horizontal, perpendicular to the flight
    el = np.radians(32.0)
    look = np.array([0.05, 0.10, 0.0]) + axis * 0.02
    pos = look + 4.5 * (np.cos(el) * side + np.sin(el) * np.array([0, 1.0, 0]))
    return sun0, src, pos, look


def II2(tl, tg, R, W, H):
    G = _get("galaxy", _galaxy)
    hP, hC = _get("halo", _halo)
    dD, dC = _get("distgal", _distant_galaxies)
    sun0, src, cpos, look = _gal_frame()
    cam = Camera(cpos, look, focal=50, W=W, H=H)
    u = tl / 6.0
    Rm = _yrot(GAL_SPIN * u)                              # very slow rotation of the disk
    # distant galaxies (the only 'background')
    R.draw_dirs(cam, dD, dC, size_px=0.9, soft=True)
    pos = G["pos"] @ Rm.T.astype(np.float32)
    gi, si = G["glow_idx"], G["star_idx"]
    R.draw(cam, pos[gi], G["rgb"][gi] * 0.010, size_px=1.6, soft=True)
    R.draw(cam, pos[si], G["rgb"][si] * 0.05, size_px=0.7)
    R.draw(cam, hP, hC * 0.05, size_px=0.6)
    # the Sun: a small yellow star, slightly warmer and steadier than its neighbours
    sun = Rm @ sun0
    R.draw(cam, sun[None], hex_lin("#FFE7A0", 0.9)[None], size_px=1.0)
    # the source: a red giant in the halo
    R.draw(cam, src[None], hex_lin("#FF7A3A", 0.5)[None], size_px=0.9)
    # the message shell: constant speed of light (compressed), already launched at the cut
    rad = 0.035 + 1.28 * u
    shell().draw(R, cam, src, rad, t=tg, energy=0.010, size_px=0.8, spacing_px=3.2, limb=3.0,
                 thickness=0.004, beaming=0.55)
    return R.resolve(), Grade(wb=(1.0, 0.98, 1.0), sat=1.05, bloom=1.2, vignette=0.2, grain=0.005)


# ============================================================================ II3 — black card
def II3(tl, tg, R, W, H):
    return black(W, H), Grade(bloom=0, grain=0, vignette=0)


TABLE = {"II2": II2, "II3": II3}


def render(sid, tl, tg, R, W, H):
    return dispatch(TABLE, sid, tl, tg, R, W, H)
