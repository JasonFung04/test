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
from ..assets import crosshatch as XH
from ..assets import drawer as DR
from ..assets import fire as FIRE
from ..assets import flake as FL
from ..assets import nebula, sky
from ..assets.shell import Shell
from ..config import FPS
from ..raster import Renderer
from ..solid import Light, SolidCloud, draw_solid, visibility
from ..camera import Camera, ease, ease5, handheld, lerp, seg
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


# ============================================================================ Blombos: shared
# Flake space (see assets/flake.py): drawing face = plane y=0, canonical +y = -z (away from her).
# She kneels on the +z side; the hearth is beyond the flake (-z), a little to the left.
FIRE_IN_FLAKE = np.array([-0.16, 0.20, -0.82])
SKY_DIR = np.array([0.25, 0.55, 1.0]) / np.linalg.norm([0.25, 0.55, 1.0])   # the mouth is behind her
GRADE_CAVE = dict(wb=(0.97, 0.98, 1.03), sat=0.98, bloom=1.0, vignette=0.26, grain=0.006)


def flake():
    return _get("flake", FL.Flake)


def fire_lights(tg, fire_pos, k=1.0, fill=1.0):
    fl = FIRE.flicker(tg)
    p = np.asarray(fire_pos, float) + FIRE.light_offset(tg)
    return [Light("point", FIRE.LIGHT, 2.2 * k * fl, vec=p, radius=0.08),
            Light("dir", hex_lin("#8FA6D8"), 0.035 * fill, vec=SKY_DIR),
            Light("amb", hex_lin("#40302A"), 0.010 * fill)]


def ochre_glints(cl, cov, cam, fire_pos, tg, strength=6.0):
    """Specular-hematite sparkle inside the ochre (tiny crystal facets catching the firelight)."""
    n = len(cov)
    h = np.asarray(cl.key, np.float64)
    sel = np.nonzero((cov > 0.45) & (h < 0.05))[0]
    if sel.size == 0:
        return
    P = cl.P[sel].astype(np.float64)
    rng = np.random.default_rng(5)
    fac = np.stack([np.cos(h[sel] * 977.0) * 0.45, np.ones(sel.size), np.sin(h[sel] * 613.0) * 0.45], 1)
    fac /= np.linalg.norm(fac, axis=1, keepdims=True)
    L = fire_pos[None, :] - P
    L /= np.linalg.norm(L, axis=1, keepdims=True)
    V = cam.pos[None, :] - P
    V /= np.linalg.norm(V, axis=1, keepdims=True)
    Hh = L + V
    Hh /= np.linalg.norm(Hh, axis=1, keepdims=True)
    sp = np.clip(np.sum(fac * Hh, axis=1), 0, 1) ** 60
    e = strength * FIRE.flicker(tg) * sp
    m = e > 0.02
    return P[m], (hex_lin("#FFB27A")[None, :] * e[m, None]).astype(np.float32)


def draw_ground(R, cam, cl, lights, energy=1.0, fill=0.55, max_px=14.0):
    """A dim, calm ground: every sample is a dot sized to its own footprint (no bright specks where
    the sampling is sparse), shaded like a solid."""
    from ..solid import shade
    x, y, z, coc, valid = cam.project(cl.P)
    keep = valid & (x > -30) & (x < R.W + 30) & (y > -30) & (y < R.H + 30)
    idx = np.nonzero(keep)[0]
    if idx.size == 0:
        return
    sub = cl.subset(idx)
    rad = shade(sub, cam.pos, lights)
    native = np.sqrt(sub.area) * cam.fpx / np.maximum(z[idx], 1e-6)          # px at this resolution
    area_px = (native / R.s) ** 2
    e = rad * area_px[:, None] * energy
    r = np.minimum(native / R.s * fill, max_px)
    R.draw(cam, sub.P, e.astype(np.float32), size_px=r.astype(np.float32))


def draw_flake(R, cam, tg, lights, spacing=2.2, occlude=True, floor=True):
    top, side, flo, cov = flake().cloud(tg)
    cl = SolidCloud.concat([top, side])
    draw_solid(R, cam, cl, lights, spacing_px=spacing, seurat=0.25, p_min=0.35, jitter=0.08,
               size_var=0.3, occlude=occlude)
    if floor:
        # the floor: calm, dark, a touch out of focus -- never glitter
        draw_ground(R, cam, flo, lights, energy=0.8)
    return top, cov


# ============================================================================ II5 / II7 — the hand
RHYME_ANCHOR = (0.60, 0.56)            # same screen anchor as the elder's palm in I5 (act1.palm_screen_anchor)
WRIST_DIR = np.array([0.55, 0.40, 0.73])         # from the crayon tip toward her wrist (flake space)
PSI = np.radians(280.0)                          # roll of the grip about the wrist direction
HAND_CAM = np.array([-0.300, 0.550, 0.350])      # camera position (flake space), her left-front
WRIST_SCREEN_DEG = 38.0                          # the wrist leaves the frame toward the lower right


def hand():
    return _get("hand", DR.Hand)


def hand_pose(tg):
    """Rm, T of the hand at tg: the crayon tip rides the scheduled stroke path (flake.tip)."""
    k, q, lift = FL.tip(tg)
    p = FL.to_flake(q)[0] + np.array([0.004, 0.010, 0.008]) * lift
    wd = WRIST_DIR / np.linalg.norm(WRIST_DIR)
    psi = PSI
    # the wrist leans into each stroke; a living hand trembles a little
    if k >= 0:
        u = float(np.clip((tg - FL.stroke_times()[k][0]) / FL.stroke_times()[k][1], 0, 1))
        psi += np.radians(4.0) * np.sin(np.pi * u)
        wd = wd + (np.array([0.0, 0.0, 0.06]) if k < 6 else np.array([0.06, 0.0, 0.0])) * np.sin(np.pi * u)
    wd = wd + np.array([FIRE._n1(tg, 1.7, 21), FIRE._n1(tg, 1.3, 22), FIRE._n1(tg, 1.1, 23)]) * 0.010
    return hand().placement_wrist(p, wd, psi), k, p, lift


def _smooth_tip(tg, half=0.7, n=21):
    ts = np.linspace(tg - half, tg + half, n)
    w = np.cos(np.linspace(-np.pi / 2, np.pi / 2, n)) ** 2
    Q = np.array([FL.to_flake(FL.tip(t)[1])[0] for t in ts])
    return (Q * w[:, None]).sum(0) / w.sum()


def _hand_roll():
    """Camera roll that puts the tip->wrist direction at WRIST_SCREEN_DEG below screen-right (the I5
    rhyme: wrist toward the lower-right corner, fingers toward the upper-left). Constant per setup."""
    def make():
        Rm, T = hand().placement_wrist(np.zeros(3), WRIST_DIR, PSI)
        wrist = Rm @ hand().WRIST + T
        cam = Camera(HAND_CAM, np.zeros(3), up=(0.0, 1.0, 0.0), focal=100)
        x, y, _, _, _ = cam.project(np.stack([np.zeros(3), wrist]))
        ang = np.degrees(np.arctan2(y[1] - y[0], x[1] - x[0]))     # screen angle, y down
        return np.radians(WRIST_SCREEN_DEG - ang)
    return _get("hand_roll", make)


def hand_camera(tg, W, H, seed, pinch=None):
    """100 mm from her left-front, looking down ~55 deg, rolled for the hand rhyme; the view gently
    follows the crayon; focus on the fingertips (the pinch); lens shift puts the followed point on the
    shared hand-rhyme anchor."""
    centre = np.array([0.0, 0.0, -0.004])
    follow = centre + 0.5 * (_smooth_tip(tg) - centre)
    pos = HAND_CAM + handheld(tg, 0.0016, seed=seed)
    look = follow + handheld(tg + 3.0, 0.0008, seed=seed + 1)
    if pinch is None:
        pinch = FL.to_flake(FL.tip(tg)[1])[0]
    focus = float(np.linalg.norm(pos - pinch))
    kw = dict(up=(0.0, 1.0, 0.0), focal=100, focus=focus, bokeh=60.0, W=W, H=H, roll=_hand_roll())
    cam = Camera(pos, look, **kw)
    ax, ay = RHYME_ANCHOR[0] * W, RHYME_ANCHOR[1] * H
    return Camera(pos, look, shift=((ax - W / 2) / cam.s, -(ay - H / 2) / cam.s), **kw)


def shadow_on_plane(P, light, res=0.00025, extent=0.10, blur_k=0.02):
    """Soft shadow of points P (occluders above y=0) cast by a point light onto the plane y=0.
    Returns a lookup function (x, z) -> darkness 0..1."""
    Ly = light[1]
    y = np.clip(P[:, 1], 1e-5, Ly - 1e-3)
    t = y / (Ly - y)
    S = P + (P - light) * t[:, None]
    n = int(2 * extent / res)
    ix = ((S[:, 0] + extent) / res).astype(np.int64)
    iz = ((S[:, 2] + extent) / res).astype(np.int64)
    ok = (ix >= 0) & (ix < n) & (iz >= 0) & (iz < n)
    grid = np.zeros((n, n), np.float32)
    np.add.at(grid, (iz[ok], ix[ok]), 1.0)
    import cv2
    grid = cv2.GaussianBlur(grid, (0, 0), 2.0)
    cov = np.clip(grid * 3.0, 0, 1)
    # penumbra grows with the occluder height (fire is ~15 cm across at ~0.8 m)
    hmed = float(np.median(y))
    sig = max(1.0, blur_k * hmed / res * 6.0)
    cov = cv2.GaussianBlur(cov, (0, 0), sig)

    def look(x, z):
        jx = np.clip(((x + extent) / res).astype(np.int64), 0, n - 1)
        jz = np.clip(((z + extent) / res).astype(np.int64), 0, n - 1)
        inside = (np.abs(x) < extent) & (np.abs(z) < extent)
        return np.where(inside, cov[jz, jx], 0.0)
    return look


def coverage_dof(R, cam, clouds, spacing_px=1.6):
    """DOF-aware occlusion mask of solid clouds: each visible front-facing sample splats its projected
    area with its circle of confusion, so defocused edges become soft and semi-transparent."""
    Rc = Renderer(R.W, R.H)
    P = np.concatenate([c.P for c in clouds])
    N = np.concatenate([c.N for c in clouds])
    A = np.concatenate([np.broadcast_to(c.area, (len(c.P),)) for c in clouds])
    x, y, z, coc, valid = cam.project(P)
    V = cam.pos[None, :] - P
    V /= np.linalg.norm(V, axis=1, keepdims=True)
    ndv = np.sum(N * V, axis=1)
    keep = valid & (ndv > -0.05) & (x > -60) & (x < R.W + 60) & (y > -60) & (y < R.H + 60)
    idx = np.nonzero(keep)[0]
    native = np.sqrt(A[idx]) * cam.fpx / np.maximum(z[idx], 1e-6)
    vis = visibility(x[idx], y[idx], z[idx], R.W, R.H, np.clip(native * 1.2 + 1, 1, 12).astype(np.int32))
    idx, native = idx[vis], native[vis]
    area_px = native ** 2 * np.clip(ndv[idx], 0.05, 1.0)
    r = np.sqrt(coc[idx] ** 2 + (np.maximum(native, spacing_px * R.s) * 0.9) ** 2)
    Rc.acc.splat(x[idx], y[idx], r.astype(np.float32),
                 np.repeat(area_px[:, None], 3, 1).astype(np.float32))
    return np.clip(Rc.resolve()[..., 0] * 1.1, 0.0, 1.0)


def _drawing_shot(tl, tg, R, W, H, seed):
    (Rm, T), k, tipw, lift = hand_pose(tg)
    pinch = Rm @ hand().g["K"] + T
    cam = hand_camera(tg, W, H, seed, pinch=pinch)
    hcl, ccl = hand().clouds(Rm, T)
    fire_pos = FIRE_IN_FLAKE + FIRE.light_offset(tg)
    lights = fire_lights(tg, FIRE_IN_FLAKE)
    # the flake, darkened where the hand shades it from the fire
    top, side, flo, cov = flake().cloud(tg)
    sh = shadow_on_plane(np.concatenate([hcl.P[::3], ccl.P]).astype(np.float64), fire_pos)
    dark = sh(top.P[:, 0].astype(np.float64), top.P[:, 2].astype(np.float64))
    top.albedo *= (1.0 - 0.85 * dark)[:, None].astype(np.float32)
    dark_f = sh(flo.P[:, 0].astype(np.float64), flo.P[:, 2].astype(np.float64))
    flo.albedo *= (1.0 - 0.85 * dark_f)[:, None].astype(np.float32)
    draw_solid(R, cam, SolidCloud.concat([top, side]), lights, spacing_px=2.2, seurat=0.25, p_min=0.35,
               jitter=0.08, size_var=0.3)
    draw_ground(R, cam, flo, lights, energy=0.8)
    g = ochre_glints(top, cov, cam, fire_pos, tg)
    if g is not None:
        R.draw(cam, g[0], g[1], size_px=0.7)
    bg = R.new_layer()
    mask = coverage_dof(R, cam, [hcl, ccl])
    # the hand: dark skin, backlit by the fire -> rim + sheen; the crayon
    rim = (FIRE.LIGHT * 0.05 * FIRE.flicker(tg), 3.0, 1.0)
    draw_solid(R, cam, hcl, lights, spacing_px=1.8, seurat=0.35, p_min=0.30, jitter=0.08, spec=(0.09, 16.0),
               rim=None, size_var=0.3)
    draw_solid(R, cam, ccl, lights, spacing_px=1.6, seurat=0.3, p_min=0.3, jitter=0.12, spec=(0.03, 8.0))
    return bg * (1.0 - mask)[..., None] + R.resolve(), Grade(**GRADE_CAVE)


def II5(tl, tg, R, W, H):
    return _drawing_shot(tl, tg, R, W, H, seed=51)


def II7(tl, tg, R, W, H):
    return _drawing_shot(tl, tg, R, W, H, seed=71)


# ============================================================================ II8 — the nine lines
D_MATCH = 100.0 * FL.S_C / (36.0 * XH.MATCH_SPAN)       # camera height that satisfies MATCH_SPAN
II8_LAST = TL.frame_range("II8")[1] - 1                  # index of the last frame of II8


def ii8_camera(tl, W, H):
    """Slow push-in + rotation; at the last frame: straight down, canonical frame centred, x right,
    y up, 1 canonical unit = MATCH_SPAN * W (crosshatch.py contract)."""
    t_last = II8_LAST / FPS - TL.shot("II8")[1]
    u = float(np.clip(tl / t_last, 0.0, 1.2))
    w = max(1.0 - u, 0.0)
    # log-distance: eases in fast, then keeps a slow 2 %/s push that carries over the cut
    l_end = np.log(D_MATCH)
    B = 0.02 * t_last
    A = np.log(0.52) - l_end - B
    dist = float(np.exp(l_end + A * w ** 3 + B * (1.0 - u)))
    el = np.radians(90.0 - 34.0 * w ** 2.5)
    az = np.radians(-26.0) * w ** 3
    look = FL.to_flake(np.array([0.10, 0.06]) * w ** 2)[0]
    horiz = np.array([np.sin(az), 0.0, np.cos(az)])
    pos = look + dist * (np.cos(el) * horiz + np.sin(el) * np.array([0.0, 1.0, 0.0]))
    up = -horiz                                  # far side of the flake is 'up' on screen
    return Camera(pos, look, up=up, focal=100, bokeh=38.0, W=W, H=H)


def II8(tl, tg, R, W, H):
    cam = ii8_camera(tl, W, H)
    fire_pos = FIRE_IN_FLAKE + FIRE.light_offset(tg)
    lights = fire_lights(tg, FIRE_IN_FLAKE)
    top, cov = draw_flake(R, cam, tg, lights)
    g = ochre_glints(top, cov, cam, fire_pos, tg)
    if g is not None:
        R.draw(cam, g[0], g[1], size_px=0.7)
    return R.resolve(), Grade(**GRADE_CAVE)


# ============================================================================ II3 — black card
def II3(tl, tg, R, W, H):
    return black(W, H), Grade(bloom=0, grain=0, vignette=0)


TABLE = {"II1": II1, "II2": II2, "II3": II3, "II5": II5, "II7": II7, "II8": II8}


def render(sid, tl, tg, R, W, H):
    return dispatch(TABLE, sid, tl, tg, R, W, H)
