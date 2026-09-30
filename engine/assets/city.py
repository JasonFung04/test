"""THE CITY: a modern Asian megacity at night, its blackout and its return of power (Act III).

World frame: +y up, +x = EAST, +z = SOUTH (north = -z). 1 unit = 1 m. Ground at y = 0.

Layout (see BIBLE 6):
  * the core district's main avenues ARE the canonical '#' (assets/crosshatch.py), 1 canonical unit =
    SCALE m, centred on the world origin (canonical x -> east, canonical y -> north); a finer street grid
    follows the nine lines; the CBD towers stand in the crossing field; a park + the river keep the
    south-east corner of the '#' calm;
  * a curving river runs down the east side of the core and bends west south of it;
  * other districts (Voronoi cells) have their own grid orientation and building mix;
  * elevated expressways with traffic, bridges, a substation ~700 m from the girl's building;
  * the girl's building: an old 7-storey slab (rooftop.py set on top) just north of the core.

Public API (used by act3.py and by the lead's shots):
  ROOF_ORIGIN, ROOF_YAW, GIRL_WORLD, GIRL_YAW, SUBSTATION_POS, LAMP_WORLD, MW_BASIS, GC_DIR, NGP_DIR
  power(tg) -> dict                       electrical state (rings, sky glow, stars, lamp, flash ...)
  env_lights(tg, pw=None) -> [Light]      solid.Light list for lighting characters/props at the rooftop
  render_env(R, cam, tg, W, H, sky=True, stars=True, city=True, roof=True, girl=False) -> (hdr, mask)
  crowd_draw(...), girl_cloud(...)        helpers (see below)
"""
import numpy as np

from .. import timeline as TL
from ..color import blackbody, hex_lin
from ..config import CACHE
from ..noise import fbm, hash01
from . import crosshatch as X
from . import rooftop

VERSION = 1

# ------------------------------------------------------------------------------------------ constants
SCALE = 1600.0                                   # metres per canonical '#' unit
ROOF_YAW = float(np.radians(-8.0))               # the girl's block is aligned with her district's grid
ROOF_ORIGIN = np.array([-380.0, 21.0, -1290.0])  # centre of her roof deck (7 storeys x 3 m)
GIRL_WORLD = rooftop.to_world(rooftop.GIRL_SEAT, ROOF_ORIGIN, ROOF_YAW)
GIRL_YAW = ROOF_YAW                              # she faces the rooftop's local +z (south, 8 deg west)
LAMP_WORLD = rooftop.to_world(rooftop.LAMP_POS, ROOF_ORIGIN, ROOF_YAW)
ROOF_HALF = (12.0, 6.0)                          # her building footprint half-extents (local x, z)

# substation: ~700 m from her roof, bearing ~200 deg (SSW), inside a block of the core district
SUBSTATION_POS = np.array([-612.0, 0.0, -630.0])

CITY_R = 16500.0                                 # modelled extent (m) around the origin
RIVER_W = 440.0

# 14 rings of the blackout: distance bands from the substation (geometric radii, noisy edges)
RING_R = np.array([0.0] + [60.0 * 1.47 ** k for k in range(1, 14)] + [1e9])
N_RING = 14

PAL = {k: hex_lin(v) for k, v in dict(sodium="#FF9F3A", led="#EAF2FF", tail="#FF3B2F", head="#FFF6E0",
                                      skyglow="#3A2412", night="#02040C", star="#CFE0FF",
                                      milkyway="#F3EAD7").items()}


def _cache_path(name):
    return CACHE / f"city{VERSION}_{name}.npz"


def _cached(name, fn):
    p = _cache_path(name)
    if p.exists():
        d = np.load(p, allow_pickle=False)
        return {k: d[k] for k in d.files}
    d = fn()
    np.savez(p, **d)
    return d


def canon_to_world(c):
    """Canonical '#' coords (N,2) -> world (N,3) at ground level."""
    c = np.asarray(c, float)
    return np.stack([c[:, 0] * SCALE, np.zeros(len(c)), -c[:, 1] * SCALE], 1)


def _dir(theta):
    """Map direction at angle theta (CCW from east, north up) as world (x, z)."""
    return np.array([np.cos(theta), -np.sin(theta)])


def _perp(theta):
    """Unit vector 90 deg CCW of _dir(theta) (i.e. 'north' of the grid's u axis), world (x, z)."""
    return np.array([-np.sin(theta), -np.cos(theta)])


# ------------------------------------------------------------------------------------------ river
_RIVER_CTRL = np.array([[7600, -12000], [5600, -8200], [4300, -5200], [3300, -2700], [2720, -900],
                        [2560, 700], [2250, 2150], [1300, 3250], [-500, 3900], [-2900, 4350],
                        [-5300, 5500], [-7600, 7700], [-10200, 10600], [-12500, 13500]], float)


def _catmull(P, n_per=40):
    P = np.asarray(P, float)
    Q = np.concatenate([P[:1] * 2 - P[1:2], P, P[-1:] * 2 - P[-2:-1]])
    out = []
    t = np.linspace(0, 1, n_per, endpoint=False)[:, None]
    for i in range(1, len(Q) - 2):
        p0, p1, p2, p3 = Q[i - 1], Q[i], Q[i + 1], Q[i + 2]
        out.append(0.5 * ((2 * p1) + (-p0 + p2) * t + (2 * p0 - 5 * p1 + 4 * p2 - p3) * t * t +
                          (-p0 + 3 * p1 - 3 * p2 + p3) * t * t * t))
    out.append(Q[-2:-1])
    return np.concatenate(out)


def river_line():
    """Dense centreline (M,2) world (x, z) and per-vertex half width."""
    C = _catmull(_RIVER_CTRL, 60)
    s = np.concatenate([[0], np.cumsum(np.linalg.norm(np.diff(C, axis=0), axis=1))])
    hw = 0.5 * RIVER_W * (1.0 + 0.12 * np.sin(s / 2300.0) + 0.06 * np.sin(s / 900.0 + 1.0))
    return C, hw


def _seg_dist(Q, A, B):
    """Distance from points Q (N,2) to segments A->B (M,2): returns (N,) min distance and argmin."""
    best = np.full(len(Q), np.inf)
    arg = np.zeros(len(Q), np.int64)
    AB = B - A
    L2 = np.maximum((AB * AB).sum(1), 1e-9)
    for i in range(len(A)):
        t = np.clip(((Q - A[i]) @ AB[i]) / L2[i], 0, 1)
        d = np.linalg.norm(Q - (A[i] + t[:, None] * AB[i]), axis=1)
        m = d < best
        best[m] = d[m]
        arg[m] = i
    return best, arg


# ------------------------------------------------------------------------------------------ districts
# type codes
T_CORE, T_OLD, T_NEW, T_CBD, T_MIX, T_IND, T_SUB, T_PARK = range(8)

# hand-placed districts near the story locations + procedural rings of others
_DISTRICTS_FIXED = [
    # x, z, angle_deg, type, su, sv
    (ROOF_ORIGIN[0], ROOF_ORIGIN[2], -8.0, T_OLD, 176.0, 150.0),   # the girl's 1990s housing estate (grid on her block)
    (-3400.0, -600.0, 24.0, T_MIX, 150.0, 170.0),       # west of the core
    (4600.0, -900.0, 31.0, T_NEW, 230.0, 210.0),        # east bank: new high-rise compounds
    (3900.0, 1900.0, 18.0, T_CBD, 150.0, 150.0),        # east-bank secondary CBD at the river bend
    (600.0, 2600.0, -14.0, T_MIX, 160.0, 150.0),        # south of the core, north of the bend
    (-2200.0, 2900.0, 9.0, T_OLD, 170.0, 150.0),
    (2600.0, -3600.0, -21.0, T_OLD, 170.0, 160.0),
    (-2600.0, -3400.0, 14.0, T_NEW, 220.0, 200.0),
    (-600.0, -3600.0, -3.0, T_OLD, 170.0, 150.0),
]


def _districts():
    rng = np.random.default_rng(77)
    D = list(_DISTRICTS_FIXED)
    # procedural seeds on a jittered polar layout
    for ring_r, n in ((6200.0, 9), (9500.0, 13), (13500.0, 17)):
        for k in range(n):
            a = 2 * np.pi * (k + rng.uniform(-0.3, 0.3)) / n + ring_r * 0.0003
            r = ring_r * rng.uniform(0.85, 1.12)
            x, z = r * np.cos(a), r * np.sin(a)
            u = rng.random()
            if ring_r < 7000:
                t = T_NEW if u < 0.45 else (T_MIX if u < 0.75 else (T_OLD if u < 0.92 else T_IND))
            elif ring_r < 10000:
                t = T_NEW if u < 0.35 else (T_IND if u < 0.6 else (T_MIX if u < 0.85 else T_SUB))
            else:
                t = T_SUB if u < 0.55 else (T_IND if u < 0.8 else T_NEW)
            ang = rng.uniform(-40, 40)
            su, sv = rng.uniform(170, 260), rng.uniform(150, 240)
            if t == T_IND:
                su, sv = su * 1.6, sv * 1.6
            if t == T_SUB:
                su, sv = su * 1.3, sv * 1.3
            D.append((x, z, ang, t, su, sv))
    return np.array(D, float)


CORE_VMAX = 0.78


def core_region(xz):
    """Boolean: inside the '#' core district."""
    u = xz[:, 0] / SCALE
    v = -xz[:, 1] / SCALE
    return (np.abs(u) <= 1.30) & (v >= -0.80) & (v <= CORE_VMAX)


PARK_C = np.array([1.00, -0.45])       # canonical, the calm lower-right of II9


def park_region(xz):
    u = xz[:, 0] / SCALE
    v = -xz[:, 1] / SCALE
    q = np.stack([u, v, np.zeros_like(u)], 1)
    wob = fbm(q * 3.0, octaves=2, offset=(3.0, 1.0, 0.0)) * 0.08
    r = np.sqrt(((u - PARK_C[0]) / 1.05) ** 2 + ((v - PARK_C[1]) / 0.80) ** 2)
    return r < 0.26 + wob


def lake_region(xz):
    u = xz[:, 0] / SCALE
    v = -xz[:, 1] / SCALE
    r = np.sqrt(((u - 1.04) / 1.2) ** 2 + ((v + 0.47) / 0.75) ** 2)
    return r < 0.085


# ------------------------------------------------------------------------------------------ grid rasters
GRID_RES = 5.0                          # street/river mask resolution (m)
GRID_N = int(2 * CITY_R / GRID_RES)


def _to_px(xz, res=GRID_RES):
    return ((np.asarray(xz)[..., 0] + CITY_R) / res, (np.asarray(xz)[..., 1] + CITY_R) / res)


def _lookup(mask, xz, res=GRID_RES):
    px, pz = _to_px(xz, res)
    n = mask.shape[0]
    i = np.clip(pz.astype(np.int64), 0, n - 1)
    j = np.clip(px.astype(np.int64), 0, n - 1)
    return mask[i, j]


# ------------------------------------------------------------------------------------------ streets
# classes: 0 '#' avenue, 1 arterial, 2 minor, 3 lane, 4 expressway (elevated), 5 park path, 6 bridge (ground road over river)
ST_W = np.array([52.0, 32.0, 16.0, 8.0, 26.0, 4.0, 28.0])


def _core_streets():
    """The nine avenues + the minor grid interpolated between them (canonical coords)."""
    out = []   # (poly canonical (n,2), class)
    for p in X.lines(n=400, wobble=0.004):
        out.append((p, 0))
    # extensions of the avenues beyond their canonical ends (ordinary minor streets)
    for fam, D, Nn in ((X.SIX, X.D6, X.N6), (X.THREE, X.D3, X.N3)):
        for (off, s0, s1, sag, seed) in fam:
            for a, b in ((-1.9, s0), (s1, 1.9)):
                s = np.linspace(a, b, 120)
                out.append(((s[:, None] * D[None] + off * Nn[None]), 2))
    # minor streets: interpolate offsets/sagittas between consecutive lines of each family
    for fam, D, Nn, step in ((X.SIX, X.D6, X.N6, 3), (X.THREE, X.D3, X.N3, 3)):
        offs = [f[0] for f in fam]
        sags = [f[3] for f in fam]
        items = []
        for i in range(len(fam) - 1):
            for k in range(1, step):
                f = k / step
                items.append((offs[i] + f * (offs[i + 1] - offs[i]), sags[i] + f * (sags[i + 1] - sags[i])))
        gap = (offs[-1] - offs[0]) / (len(offs) - 1) / step
        for k in range(1, 9):
            items.append((offs[0] - k * gap, sags[0] * max(0, 1 - k / 3)))
            items.append((offs[-1] + k * gap, sags[-1] * max(0, 1 - k / 3)))
        for off, sag in items:
            s = np.linspace(-1.9, 1.9, 400)
            t = (s + 1.9) / 3.8
            bow = sag * 4.0 * t * (1 - t)
            out.append((s[:, None] * D[None] + (off + bow)[:, None] * Nn[None], 2))
    return out


def _clip_runs(pts, inside, min_len=3):
    """Split a polyline (n,2) into runs where inside is True."""
    runs = []
    idx = np.nonzero(inside)[0]
    if idx.size == 0:
        return runs
    br = np.nonzero(np.diff(idx) > 1)[0]
    starts = np.concatenate([[0], br + 1])
    ends = np.concatenate([br, [idx.size - 1]])
    for a, b in zip(starts, ends):
        if b - a + 1 >= min_len:
            runs.append(pts[idx[a]:idx[b] + 1])
    return runs


# expressways (elevated, world x, z control points) and their deck height
_EXPRESSWAYS = [
    ([[-15500, 2300], [-9000, 1950], [-5000, 1700], [-2500, 1560], [0, 1500], [2000, 1560], [3600, 1500],
      [6000, 1250], [9500, 900], [15500, 600]], 18.0),                                   # E-W south of core
    ([[-3150, -15500], [-3200, -9000], [-3100, -3000], [-3000, 200], [-3350, 4200], [-4200, 9500],
      [-5200, 15500]], 16.0),                                                              # N-S west
    ([[-3100, -2050], [-1500, -2350], [0, -2450], [1700, -2250], [3050, -1650], [3900, -700],
      [4150, 400], [3950, 1500]], 21.0),                                                   # inner ring N + E
    ([[4300, -15500], [4450, -8000], [4400, -2600], [4150, 400], [4500, 5200], [5600, 10000],
      [6800, 15500]], 17.0),                                                               # N-S east bank
    ([[-3000, 200], [-6000, -1500], [-10000, -3500], [-15500, -6000]], 15.0),              # radial W
    ([[3950, 1500], [7000, 3800], [10500, 6800], [15500, 9500]], 15.0),                    # radial SE
]


# bridges: arclength fractions along the river centreline, deck height
_BRIDGES = [(0.12, 11.0), (0.205, 12.0), (0.285, 12.0), (0.335, 13.0), (0.395, 12.0), (0.47, 12.0),
            (0.545, 12.0), (0.63, 11.0), (0.72, 11.0), (0.83, 10.0)]

# special buildings: the girl's block and the tall block the III3 camera stands on
PERCH_C = np.array([-362.0, -1348.0])      # (x, z)
SPECIAL = [
    # x, z, a, b, yaw, floors, floor_h, type, tag   (tag 1 = girl's block, 2 = perch)
    (ROOF_ORIGIN[0], ROOF_ORIGIN[2], 12.0, 6.0, ROOF_YAW, 7, 3.0, T_OLD, 1),
    (PERCH_C[0], PERCH_C[1], 16.0, 7.0, ROOF_YAW, 12, 3.0, T_NEW, 2),
]


def _core_zone(xz):
    """Building mix inside the core: 0 old residential, 1 mixed mid-rise, 2 CBD towers."""
    u = xz[:, 0] / SCALE
    v = -xz[:, 1] / SCALE
    cbd = ((u + 0.14) / 0.46) ** 2 + ((v - 0.06) / 0.30) ** 2 < 1.0
    n = fbm(np.stack([u * 4.0, v * 4.0, np.full_like(u, 2.0)], 1), octaves=2)
    z = np.where(n > 0.05, 1, 0)
    return np.where(cbd, 2, z)


def _build_layout():
    rng = np.random.default_rng(2024)
    import cv2
    N = GRID_N
    # --- river raster
    RC, RH = river_line()
    river = np.zeros((N, N), np.uint8)
    for i in range(len(RC) - 1):
        a = np.array(_to_px(RC[i])).astype(np.int32)
        b = np.array(_to_px(RC[i + 1])).astype(np.int32)
        cv2.line(river, tuple(int(v) for v in a), tuple(int(v) for v in b), 1,
                 thickness=max(1, int(2 * RH[i] / GRID_RES)))
    # --- district raster (20 m)
    DR = 20.0
    ND = int(2 * CITY_R / DR)
    D = _districts()
    g = (np.arange(ND) + 0.5) * DR - CITY_R
    gx, gz = np.meshgrid(g, g)
    Q = np.stack([gx.ravel(), gz.ravel()], 1)
    wob = fbm(np.stack([Q[:, 0], Q[:, 1], np.zeros(len(Q))], 1) * 0.0006, octaves=3) * 700.0
    best = np.full(len(Q), np.inf)
    did = np.zeros(len(Q), np.int32)
    for k in range(len(D)):
        d = np.hypot(Q[:, 0] - D[k, 0], Q[:, 1] - D[k, 1]) + wob * (1 + (k % 3))
        m = d < best
        best[m] = d[m]
        did[m] = k + 1
    did[core_region(Q)] = 0
    dmap = did.reshape(ND, ND).astype(np.int16)
    # --- streets
    streets = []            # (poly world xz (n,2), class, district)
    for poly, cls in _core_streets():
        xz = canon_to_world(poly)[:, [0, 2]]
        s = np.concatenate([[0], np.cumsum(np.linalg.norm(np.diff(xz, axis=0), axis=1))])
        ss = np.linspace(0, s[-1], max(3, int(s[-1] / 8)))
        xz = np.stack([np.interp(ss, s, xz[:, 0]), np.interp(ss, s, xz[:, 1])], 1)
        inside = core_region(xz) & (_lookup(river, xz) == 0)
        if cls != 0:
            inside &= ~park_region(xz)
            inside &= np.hypot(xz[:, 0] - SUBSTATION_POS[0], xz[:, 1] - SUBSTATION_POS[2]) > 60
        for run in _clip_runs(xz, inside):
            streets.append((run, cls, 0))
    for k in range(len(D)):
        x0, z0, ang, typ, su, sv = D[k]
        cells = np.nonzero(dmap == k + 1)
        if cells[0].size == 0:
            continue
        zmin, zmax = g[cells[0].min()] - DR, g[cells[0].max()] + DR
        xmin, xmax = g[cells[1].min()] - DR, g[cells[1].max()] + DR
        th = np.radians(ang)
        eu, ev = _dir(th), _perp(th)
        corners = np.array([[xmin, zmin], [xmax, zmin], [xmin, zmax], [xmax, zmax]]) - [x0, z0]
        cu, cv = corners @ eu, corners @ ev
        art_u = int(rng.integers(3, 5))
        art_v = int(rng.integers(3, 5))
        jit = 0.0 if k == 0 else 0.04
        for (e1, e2, sp, cmin, cmax, lmin, lmax, art) in (
                (eu, ev, su, cu.min(), cu.max(), cv.min(), cv.max(), art_u),
                (ev, eu, sv, cv.min(), cv.max(), cu.min(), cu.max(), art_v)):
            i0, i1 = int(np.floor(cmin / sp)) - 1, int(np.ceil(cmax / sp)) + 1
            for i in range(i0, i1 + 1):
                c = (i + 0.5) * sp + rng.normal(0, sp * jit)
                s = np.arange(lmin, lmax, 8.0)
                pts = np.array([x0, z0])[None] + c * e1[None] + s[:, None] * e2[None]
                inside = (_lookup(dmap, pts, DR) == k + 1) & (_lookup(river, pts) == 0)
                inside &= np.hypot(pts[:, 0], pts[:, 1]) < CITY_R - 300
                cls = 1 if (i % art == 0) else 2
                if typ in (T_SUB, T_IND) and cls == 2 and rng.random() < 0.35:
                    continue
                for run in _clip_runs(pts, inside, 4):
                    streets.append((run, cls, k + 1))
    # core boundary arterial (canonical rectangle)
    ring_c = np.array([[-1.30, -0.80], [1.30, -0.80], [1.30, CORE_VMAX], [-1.30, CORE_VMAX], [-1.30, -0.80]])
    ring_w = canon_to_world(ring_c)[:, [0, 2]]
    for i in range(4):
        s = np.linspace(0, 1, 400)[:, None]
        pts = ring_w[i] * (1 - s) + ring_w[i + 1] * s
        for run in _clip_runs(pts, _lookup(river, pts) == 0):
            streets.append((run, 1, 0))
    # park paths
    for k in range(5):
        th = np.linspace(0, 2 * np.pi, 240)
        r = 0.10 + 0.045 * k + 0.02 * np.sin(th * (2 + k) + k)
        c = np.stack([PARK_C[0] + r * np.cos(th), PARK_C[1] + r * np.sin(th) * 0.72], 1)
        xz = canon_to_world(c)[:, [0, 2]]
        for run in _clip_runs(xz, park_region(xz) & ~lake_region(xz)):
            streets.append((run, 5, 0))
    # bridges (ground roads over the river, deck above the water)
    sR = np.concatenate([[0], np.cumsum(np.linalg.norm(np.diff(RC, axis=0), axis=1))])
    bridges = []
    for f, hdeck in _BRIDGES:
        s0 = f * sR[-1]
        i = int(np.searchsorted(sR, s0))
        i = min(max(i, 1), len(RC) - 2)
        t = RC[i + 1] - RC[i - 1]
        t /= np.linalg.norm(t)
        nrm = np.array([-t[1], t[0]])
        c = RC[i]
        half = RH[i] + 90.0
        s = np.linspace(-half, half, int(2 * half / 8))
        pts = c[None] + s[:, None] * nrm[None]
        if np.hypot(c[0], c[1]) < CITY_R - 400:
            bridges.append((pts, hdeck))
    # expressways (elevated)
    exprs = []
    for ctrl, hdeck in _EXPRESSWAYS:
        C = _catmull(np.array(ctrl, float), 30)
        s = np.concatenate([[0], np.cumsum(np.linalg.norm(np.diff(C, axis=0), axis=1))])
        ss = np.arange(0, s[-1], 8.0)
        xz = np.stack([np.interp(ss, s, C[:, 0]), np.interp(ss, s, C[:, 1])], 1)
        for run in _clip_runs(xz, np.hypot(xz[:, 0], xz[:, 1]) < CITY_R - 100):
            exprs.append((run, hdeck))
    # --- street mask (for building placement)
    smask = np.zeros((N, N), np.uint8)
    for poly, cls, dd in streets:
        pts = np.stack(_to_px(poly), 1).astype(np.int32)
        cv2.polylines(smask, [pts], False, 1, thickness=max(1, int(round((ST_W[cls] + 6) / GRID_RES))))
    for poly, hdeck in exprs + bridges:
        pts = np.stack(_to_px(poly), 1).astype(np.int32)
        cv2.polylines(smask, [pts], False, 1, thickness=int(round((ST_W[4] + 8) / GRID_RES)))
    B = _place_buildings(rng, D, dmap, DR, river, smask)
    sp = [s[0] for s in streets]
    ep = [e[0] for e in exprs] + [b[0] for b in bridges]
    out = dict(
        st_pts=np.concatenate(sp).astype(np.float32), st_len=np.array([len(p) for p in sp], np.int64),
        st_cls=np.array([s[1] for s in streets], np.int8), st_dist=np.array([s[2] for s in streets], np.int16),
        ex_pts=np.concatenate(ep).astype(np.float32), ex_len=np.array([len(p) for p in ep], np.int64),
        ex_h=np.array([e[1] for e in exprs] + [b[1] for b in bridges], np.float32),
        ex_bridge=np.array([0] * len(exprs) + [1] * len(bridges), np.int8),
        dist=D.astype(np.float32), dmap=dmap, river_c=RC.astype(np.float32), river_hw=RH.astype(np.float32),
    )
    out.update(B)
    return out


def _footprint_ok(C, a, b, eu, ev, smask, river):
    ok = np.ones(len(C), bool)
    for fu, fv in ((0, 0), (1, 1), (1, -1), (-1, 1), (-1, -1), (1, 0), (-1, 0), (0, 1), (0, -1),
                   (0.5, 1), (-0.5, 1), (0.5, -1), (-0.5, -1)):
        q = C + (fu * a)[:, None] * eu[None] + (fv * b)[:, None] * ev[None]
        ok &= (_lookup(smask, q) == 0) & (_lookup(river, q) == 0)
    return ok


def _gen_type(rng, typ, n):
    u = rng.random(n)
    if typ == T_OLD:
        a = rng.uniform(17, 27, n)
        b = rng.uniform(5.2, 6.6, n)
        fl = np.where(u < 0.72, 7, np.where(u < 0.9, 6, 5))
        fh = np.full(n, 3.0)
        tower = rng.random(n) < 0.05
        a[tower], b[tower] = rng.uniform(10, 15, tower.sum()), rng.uniform(9, 12, tower.sum())
        fl[tower] = rng.integers(14, 25, tower.sum())
        bt = np.where(tower, T_NEW, T_OLD)
    elif typ == T_NEW:
        a = rng.uniform(11, 17, n)
        b = rng.uniform(7, 10, n)
        fl = rng.integers(16, 34, n)
        fh = np.full(n, 3.0)
        bt = np.full(n, T_NEW)
    elif typ == T_CBD:
        a = rng.uniform(15, 30, n)
        b = rng.uniform(15, 28, n)
        fl = (rng.pareto(1.5, n) * 12 + 16).astype(int).clip(10, 72)
        fh = np.full(n, 4.0)
        bt = np.full(n, T_CBD)
    elif typ == T_MIX:
        a = rng.uniform(9, 20, n)
        b = rng.uniform(7, 14, n)
        fl = np.where(u < 0.72, rng.integers(3, 9, n), rng.integers(10, 26, n))
        fh = np.full(n, 3.2)
        bt = np.where(fl > 9, T_NEW, T_MIX)
    elif typ == T_IND:
        a = rng.uniform(22, 60, n)
        b = rng.uniform(14, 36, n)
        fl = rng.integers(1, 4, n)
        fh = np.full(n, 5.0)
        bt = np.full(n, T_IND)
    else:
        a = rng.uniform(10, 24, n)
        b = rng.uniform(6, 11, n)
        fl = np.where(u < 0.75, rng.integers(3, 7, n), rng.integers(12, 28, n))
        fh = np.full(n, 3.0)
        bt = np.where(fl > 9, T_NEW, T_MIX)
    return a, b, fl, fh, bt


_PITCH = {T_OLD: (58.0, 29.0), T_NEW: (52.0, 58.0), T_CBD: (82.0, 82.0), T_MIX: (44.0, 36.0),
          T_IND: (130.0, 90.0), T_SUB: (60.0, 55.0)}


def _place_buildings(rng, D, dmap, DR, river, smask):
    """Footprints: centre (x,z), half extents (a along the grid u axis, b along v), yaw, floors, floor
    height, type.  Candidates on a jittered lattice per district, rejected where they touch streets,
    water, parks or leave their district."""
    recs = []
    jobs = [(0, T_OLD, 0), (0, T_MIX, 1), (0, T_CBD, 2)] + [(k + 1, int(D[k, 3]), -1) for k in range(len(D))]
    gpos = lambda i: (i + 0.5) * DR - CITY_R
    for did, typ, zone in jobs:
        cells = np.nonzero(dmap == did)
        if cells[0].size == 0:
            continue
        zmin, zmax = gpos(cells[0].min()) - DR, gpos(cells[0].max()) + DR
        xmin, xmax = gpos(cells[1].min()) - DR, gpos(cells[1].max()) + DR
        if did == 0:
            th, x0, z0 = X.A3, 0.0, 0.0
        else:
            x0, z0, th = D[did - 1, 0], D[did - 1, 1], np.radians(D[did - 1, 2])
        eu, ev = _dir(th), _perp(th)
        corners = np.array([[xmin, zmin], [xmax, zmin], [xmin, zmax], [xmax, zmax]]) - [x0, z0]
        cu, cv = corners @ eu, corners @ ev
        pu, pv = _PITCH[typ]
        uu = np.arange(np.floor(cu.min() / pu) * pu, cu.max(), pu)
        vv = np.arange(np.floor(cv.min() / pv) * pv, cv.max(), pv)
        U, V = np.meshgrid(uu, vv)
        U = U.ravel()
        V = V.ravel()
        if did != 1:          # the girl's estate keeps a clean lattice (rows of slabs)
            U = U + rng.uniform(-0.12, 0.12, U.size) * pu
            V = V + rng.uniform(-0.10, 0.10, V.size) * pv
        else:
            U = U + rng.uniform(-0.06, 0.06, U.size) * pu
        C = np.array([x0, z0])[None] + U[:, None] * eu[None] + V[:, None] * ev[None]
        n = len(C)
        if n == 0:
            continue
        a, b, fl, fh, bt = _gen_type(rng, typ, n)
        keep = _lookup(dmap, C, DR) == did
        if zone >= 0:
            keep &= _core_zone(C) == zone
        r = np.hypot(C[:, 0], C[:, 1])
        dens = np.clip(1.25 - (r - 6500) / 7000, 0.12, 1.0)
        keep &= rng.random(n) < dens
        if typ == T_SUB:
            keep &= rng.random(n) < 0.55
        keep &= _footprint_ok(C, a, b, eu, ev, smask, river)
        keep &= ~park_region(C)
        keep &= np.hypot(C[:, 0] - SUBSTATION_POS[0], C[:, 1] - SUBSTATION_POS[2]) > 80
        keep &= r < CITY_R - 200
        for sx, sz, sa, sb, syaw, sfl, sfh, styp, tag in SPECIAL:
            keep &= np.hypot(C[:, 0] - sx, C[:, 1] - sz) > (sa + a + 4.0) * 0.8 + 0 * sb
        idx = np.nonzero(keep)[0]
        if idx.size == 0:
            continue
        recs.append(dict(x=C[idx, 0], z=C[idx, 1], a=a[idx], b=b[idx], yaw=np.full(idx.size, th),
                         fl=fl[idx].astype(np.int16), fh=fh[idx], typ=bt[idx].astype(np.int8),
                         dist=np.full(idx.size, did, np.int16), tag=np.zeros(idx.size, np.int8)))
    for sx, sz, sa, sb, syaw, sfl, sfh, styp, tag in SPECIAL:
        recs.append(dict(x=np.array([sx]), z=np.array([sz]), a=np.array([sa]), b=np.array([sb]),
                         yaw=np.array([syaw]), fl=np.array([sfl], np.int16), fh=np.array([sfh]),
                         typ=np.array([styp], np.int8), dist=np.array([1], np.int16), tag=np.array([tag], np.int8)))
    B = {k: np.concatenate([r[k] for r in recs]) for k in recs[0]}
    B = {("b_" + k): v.astype(np.float32) if v.dtype == np.float64 else v for k, v in B.items()}
    return B


_L = {}


def layout():
    if "lay" not in _L:
        _L["lay"] = _cached("layout", _build_layout)
    return _L["lay"]
