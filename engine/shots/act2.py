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
from ..assets import nebula, sky
from ..assets.shell import Shell
from ..raster import Renderer
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


def shell_ii1():
    # same membrane design, oriented so the nebula sees its equator (rings run horizontally)
    return _get("shell1", lambda: Shell(seed=7, axis=(-0.30, 0.90, 0.32), ref=(1.0, 0.0, 0.3)))


# ============================================================================ II1 — the nebula
# The front travels along +x at V (units/s); the camera rides alongside it, DELTA just ahead of
# the front (outside the shell), looking across the nebula (-z), yawed back toward the lit wake.
# The shell's limb is then a crisp gold 'horizon' separating the lit wake (left, seen through the
# gauze of the membrane) from the dark nebula ahead (right).
V1 = 10.0
DELTA1 = 4.0
R1_0 = 2600.0                      # shell radius at the cut (huge: the front is a gently curved veil)
ZC1 = -170.0                       # the front's apex line runs through the middle of the nebula
YAW1 = np.radians(19.0)


def _neb():
    d = nebula.build()
    C0 = np.array([-R1_0, 0.0, ZC1])
    d = dict(d)
    d["shadow"] = nebula.shadow_mask(d["P"])
    d["r"] = np.linalg.norm(d["P"].astype(np.float64) - C0, axis=1)    # distance to the shell centre
    k = d["kind"]
    d["bg"] = np.nonzero((k == 0) | (k == 5))[0]
    d["fg"] = np.nonzero((k == 1) | (k == 2) | (k == 3) | (k == 4))[0]
    d["wisp"] = np.nonzero(k == 6)[0]
    d["occ"] = np.nonzero((k == 1) | (k == 2))[0]
    return d


def _field():
    return sky.field()


def II1(tl, tg, R, W, H):
    d = _get("nebula", _neb)
    fld = _get("field", _field)
    Rf = R1_0 + V1 * tl                                  # front radius (x of the apex = V1 * tl)
    C0 = np.array([-R1_0, 0.0, ZC1])
    pos = np.array([V1 * tl + DELTA1, 6.0, 0.0])
    fwd = np.array([-np.sin(YAW1), np.sin(np.radians(2.0)), -np.cos(YAW1)])
    cam = Camera(pos, pos + fwd * 100.0, focal=24, W=W, H=H)
    E = nebula.light(d, d["r"], Rf, shadow=d["shadow"])
    P, size = d["P"], d["size"]
    # --- far layer: stars + diffuse wall + filaments, attenuated by the pillars in front of them
    fl = fld["rgb"].max(axis=1) > 0.004
    R.draw_dirs(cam, fld["dirs"][fl], fld["rgb"][fl] * 0.55, size_px=0.6)
    bi = d["bg"]
    R.draw(cam, P[bi], E[bi] * 4.0, size_px=size[bi], soft=True)
    far = R.new_layer()
    Rd = Renderer(W, H)
    oi = d["occ"]
    Rd.draw(cam, P[oi], np.full((1, 3), 0.02, np.float32), size_px=4.0, soft=True)
    dens = Rd.resolve()[..., 0]
    far *= np.exp(-dens * 3.0)[..., None]
    # --- pillars (lit rims, dark cores), streamers, embedded stars
    fi = d["fg"]
    R.draw(cam, P[fi], E[fi] * 1.2, size_px=size[fi], soft=True)
    wi = d["wisp"]
    R.draw(cam, P[wi], E[wi] * 1.5, size=0.35, size_px=2.0, soft=True)
    # --- the front
    shell_ii1().draw(R, cam, C0, Rf, t=tg, energy=0.022, size_px=0.7, spacing_px=2.6, limb=1.0,
                     thickness=0.0004, beaming=0.6)
    return far + R.resolve(), Grade(wb=(1.0, 0.97, 1.02), sat=1.1, bloom=1.3, vignette=0.22, grain=0.005)


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
    """Per-point look of the spiral: clumpy star clouds, dust lanes, sparse pink HII knots.

    Two layers: a soft unresolved glow (all points, faint) and crisp resolved stars (a subset with a
    heavy-tailed brightness distribution) so the disk reads as dots, not as a smooth painting."""
    from ..noise import fbm
    g = sky.spiral_galaxy()
    rng = np.random.default_rng(12)
    pos, rgb, kind = g["pos"].astype(np.float64), g["rgb"].copy(), g["kind"]
    n = pos.shape[0]
    rr = np.linalg.norm(pos[:, [0, 2]], axis=1)
    clump = fbm(pos * np.array([7.0, 7.0, 7.0]), octaves=4, offset=(3.1, 0.0, 7.7))
    f = np.exp(2.0 * clump) * (0.35 + 0.65 * np.clip(rr / 0.25, 0, 1))
    dust = fbm(pos * np.array([16.0, 16.0, 16.0]), octaves=3, offset=(11.0, 5.0, 2.0))
    lane = np.clip((dust - 0.02) / 0.22, 0, 1) * np.clip((rr - 0.12) / 0.2, 0, 1) * (kind != 0)
    f *= 1.0 - 0.85 * lane
    young = (kind == 1)
    rgb[young] = rgb[young] * 0.7 + hex_lin("#A9C2FF") * 0.45 * rgb[young].mean(axis=1, keepdims=True)
    hii = kind == 3
    f[hii] = 0.28 * (0.3 + rng.random(hii.sum()) ** 3 * 2.5)
    rgb = rgb * f[:, None].astype(np.float32)
    u = rng.random(n)
    star = ((u < 0.10) & (kind != 3)) | (hii & (rng.random(n) < 0.5))
    sb = np.where(star, (rng.pareto(2.2, n) + 0.3) * 0.9, 0.0)
    return dict(pos=pos.astype(np.float32), glow=(rgb * 0.0075).astype(np.float32),
                star_rgb=(rgb * sb[:, None] * 0.030).astype(np.float32)[star],
                star_idx=np.nonzero(star)[0])


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
        m = 400
        pl = rng.standard_normal((m, 3))
        pl /= np.linalg.norm(pl, axis=1, keepdims=True)
        rad = 0.005 * (rng.random(m) ** (-2 / 3) - 1) ** -0.5
        rad = np.clip(rad, 0, 0.03)
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
    pos = look + 3.8 * (np.cos(el) * side + np.sin(el) * np.array([0, 1.0, 0]))
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
    R.draw(cam, pos, G["glow"], size_px=1.3, soft=True)
    R.draw(cam, pos[G["star_idx"]], G["star_rgb"], size_px=0.6)
    R.draw(cam, hP, hC * 0.035, size_px=0.6)
    # the Sun: a small yellow star, slightly warmer and steadier than its neighbours
    sun = Rm @ sun0
    R.draw(cam, sun[None], hex_lin("#FFE08A", 2.2)[None], size_px=0.9)
    R.draw(cam, sun[None], hex_lin("#FFD890", 0.5)[None], size_px=5.0, soft=True)
    # the source: a red giant in the halo
    R.draw(cam, src[None], hex_lin("#FF7A3A", 0.5)[None], size_px=0.9)
    # the message shell: constant speed of light (compressed), already launched at the cut
    rad = 0.06 + 1.20 * u
    shell().draw(R, cam, src, rad, t=tg, energy=0.005, size_px=0.8, spacing_px=3.4, limb=5.0,
                 thickness=0.004, beaming=0.7)
    return R.resolve(), Grade(wb=(1.0, 0.98, 1.0), sat=1.05, bloom=1.2, vignette=0.2, grain=0.005)


# ============================================================================ II3 — black card
def II3(tl, tg, R, W, H):
    return black(W, H), Grade(bloom=0, grain=0, vignette=0)


TABLE = {"II1": II1, "II2": II2, "II3": II3}


def render(sid, tl, tg, R, W, H):
    return dispatch(TABLE, sid, tl, tg, R, W, H)
