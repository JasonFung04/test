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
    (3900.0, 1900.0, 18.0, T_NEW, 190.0, 170.0),        # east bank at the bend
    (700.0, 2350.0, -14.0, T_CBD, 150.0, 140.0),        # supertall cluster inside the river bend
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
PERCH_C = np.array([-362.0, -1356.0])      # (x, z)
SPECIAL = [
    # x, z, a, b, yaw, floors, floor_h, type, tag   (tag 1 = girl's block, 2 = perch)
    (ROOF_ORIGIN[0], ROOF_ORIGIN[2], 12.0, 6.0, ROOF_YAW, 7, 3.0, T_OLD, 1),
    (PERCH_C[0], PERCH_C[1], 14.0, 7.0, ROOF_YAW, 13, 3.0, T_NEW, 2),
]


def _core_zone(xz):
    """Building mix inside the core: 0 old residential, 1 mixed mid-rise, 2 CBD towers."""
    u = xz[:, 0] / SCALE
    v = -xz[:, 1] / SCALE
    cbd = ((u + 0.14) / 0.40) ** 2 + ((v - 0.04) / 0.26) ** 2 < 1.0
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
        a = rng.uniform(12, 22, n)
        b = rng.uniform(12, 20, n)
        fl = (rng.pareto(1.5, n) * 10 + 16).astype(int).clip(10, 62)
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


_PITCH = {T_OLD: (58.0, 29.0), T_NEW: (52.0, 58.0), T_CBD: (56.0, 56.0), T_MIX: (44.0, 36.0),
          T_IND: (130.0, 90.0), T_SUB: (60.0, 55.0)}


def _place_buildings(rng, D, dmap, DR, river, smask):
    """Footprints: centre (x,z), half extents (a along the grid u axis, b along v), yaw, floors, floor
    height, type.  Candidates on a jittered lattice per district, rejected where they touch streets,
    water, parks or leave their district."""
    recs = []
    jobs = [(0, T_OLD, 0), (0, T_MIX, 1), (0, T_CBD, 2)]
    for k in range(len(D)):
        if int(D[k, 3]) == T_CBD:
            jobs += [(k + 1, T_CBD, 10), (k + 1, T_MIX, 11)]
        else:
            jobs.append((k + 1, int(D[k, 3]), -1))
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
        if did == 0 and typ == T_CBD:
            pu, pv = 72.0, 72.0
        if did == 1:
            pu, pv = 52.0, 29.0              # the girl's estate: rows of slabs fitting its 176 m blocks
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
        if did == 1:
            # the girl's estate stays a homogeneous low-rise carpet around her block (open view south)
            a = np.minimum(a, 21.5)
            near_girl = np.hypot(C[:, 0] - ROOF_ORIGIN[0], C[:, 1] - ROOF_ORIGIN[2]) < 420.0
            tw = near_girl & (bt != T_OLD)
            a[tw], b[tw], fl[tw], bt[tw] = rng.uniform(17, 21.5, tw.sum()), rng.uniform(5.2, 6.6, tw.sum()), 7, T_OLD
        # the III3 / III5 view corridor south of the girl's roof stays low-rise (the skyline is the CBD
        # cluster beyond ~1 km and the supertalls at the river bend)
        rel = C - np.array([ROOF_ORIGIN[0], ROOF_ORIGIN[2]])
        brg = np.degrees(np.arctan2(rel[:, 0], -rel[:, 1])) % 360
        dist_g = np.hypot(rel[:, 0], rel[:, 1])
        corridor = (dist_g < 950.0) & (brg > 138.0) & (brg < 232.0)
        if typ != T_CBD:
            tw = corridor & (fl * fh > 22.0)
            fl = np.where(tw, np.minimum(fl, 7), fl)
            fh = np.where(tw, 3.0, fh)
            bt = np.where(tw, np.where(bt == T_NEW, T_OLD, bt), bt)
            a = np.where(tw & (typ == T_MIX), np.minimum(a, 20.0), a)
        keep = _lookup(dmap, C, DR) == did
        if 0 <= zone < 10:
            keep &= _core_zone(C) == zone
        elif zone >= 10:
            dd = np.hypot(C[:, 0] - x0, C[:, 1] - z0) + fbm(np.stack([C[:, 0], C[:, 1], np.zeros(n)], 1) * 0.004,
                                                           octaves=2) * 250.0
            keep &= (dd < 850.0) if zone == 10 else (dd >= 850.0)
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
            su_, sv_ = _dir(syaw), _perp(syaw)
            du = np.abs((C[:, 0] - sx) * su_[0] + (C[:, 1] - sz) * su_[1])
            dv = np.abs((C[:, 0] - sx) * sv_[0] + (C[:, 1] - sz) * sv_[1])
            r_c = np.hypot(a, b)
            keep &= ~((du < sa + r_c + 5.0) & (dv < sb + r_c * 0.5 + 9.0))
        idx = np.nonzero(keep)[0]
        if idx.size == 0:
            continue
        if typ == T_CBD and did == 0:
            # the old core: a modest mid-rise office cluster (the skyline belongs to the river bend)
            fl[idx] = np.clip((fl[idx] * 0.55).astype(int), 9, 34)
        if typ == T_CBD:
            ctr = np.array([-0.14 * SCALE, -0.06 * SCALE]) if did == 0 else np.array([x0, z0])
            tall = [38, 35, 33] if did == 0 else [118, 102, 90, 76, 70, 66]
            order = idx[np.argsort(np.hypot(C[idx, 0] - ctr[0], C[idx, 1] - ctr[1]))]
            for j, f in zip(order, tall):
                fl[j] = f
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


# ================================================================================================ rings
_RING = {}


def ring_dist(xz):
    """Noisy distance from the substation (m) -> sets the blackout ring of anything at xz (N,2)."""
    xz = np.asarray(xz, np.float64)
    d = np.hypot(xz[:, 0] - SUBSTATION_POS[0], xz[:, 1] - SUBSTATION_POS[2])
    q = np.stack([xz[:, 0] * 0.0021, xz[:, 1] * 0.0021, np.full(len(xz), 5.0)], 1)
    n = fbm(q, octaves=3)
    return d * (1.0 + 0.30 * n) + 25.0 * fbm(q * 6.0, octaves=2)


def ring_of(xz):
    return np.clip(np.searchsorted(RING_R, ring_dist(xz), side="right") - 1, 0, N_RING - 1).astype(np.uint8)


def _ringmap():
    """20 m raster of ring_dist for fast per-frame lookups (cars, haze, sky)."""
    if "map" not in _RING:
        def make():
            DR = 20.0
            ND = int(2 * CITY_R / DR)
            g = (np.arange(ND) + 0.5) * DR - CITY_R
            gx, gz = np.meshgrid(g, g)
            return dict(rd=ring_dist(np.stack([gx.ravel(), gz.ravel()], 1)).reshape(ND, ND).astype(np.float32))
        _RING["map"] = _cached("ringmap", make)["rd"]
    return _RING["map"]


def ring_fast(xz):
    rd = _lookup(_ringmap(), np.asarray(xz), 20.0)
    return np.clip(np.searchsorted(RING_R, rd, side="right") - 1, 0, N_RING - 1).astype(np.int64), rd


# ================================================================================================ lights
# window-set kinds
K_WIN, K_SHOP, K_STAIR, K_CROWN, K_AVI, K_EMERG, K_FLICK, K_YARD = range(8)
# lamp-set kinds
L_STREET, L_EXPR, L_BRIDGE, L_BANK, L_PARK, L_DECO = range(6)

TILE = 500.0
NT = int(2 * CITY_R / TILE)


def _tile_of(xz):
    i = np.clip(((xz[:, 1] + CITY_R) / TILE).astype(np.int64), 0, NT - 1)
    j = np.clip(((xz[:, 0] + CITY_R) / TILE).astype(np.int64), 0, NT - 1)
    return i * NT + j


def _h1(i, seed):
    return float(hash01(np.array([i]), seed)[0])


def _ncode(nx, nz):
    ang = np.mod(np.arctan2(nz, nx), 2 * np.pi)
    return np.round(ang / (2 * np.pi) * 254).astype(np.int64) % 254


def _bb(T, desat=0.25):
    c = blackbody(T)
    lum = c @ np.array([0.2126, 0.7152, 0.0722])
    return c + (lum[..., None] - c) * desat


def _resample(poly, pitch, offset=0.0):
    """Points every `pitch` m along a polyline (n,2); returns pts (m,2), tangents (m,2)."""
    d = np.linalg.norm(np.diff(poly, axis=0), axis=1)
    s = np.concatenate([[0], np.cumsum(d)])
    if s[-1] < pitch * 0.5:
        return np.zeros((0, 2)), np.zeros((0, 2))
    ss = np.arange(offset % pitch, s[-1], pitch)
    x = np.interp(ss, s, poly[:, 0])
    z = np.interp(ss, s, poly[:, 1])
    x2 = np.interp(np.minimum(ss + 2.0, s[-1]), s, poly[:, 0]) - np.interp(np.maximum(ss - 2.0, 0), s, poly[:, 0])
    z2 = np.interp(np.minimum(ss + 2.0, s[-1]), s, poly[:, 1]) - np.interp(np.maximum(ss - 2.0, 0), s, poly[:, 1])
    t = np.stack([x2, z2], 1)
    t /= np.linalg.norm(t, axis=1, keepdims=True) + 1e-9
    return np.stack([x, z], 1), t


def _polys(L, pre):
    pts, ln = L[pre + "_pts"], L[pre + "_len"]
    off = np.concatenate([[0], np.cumsum(ln)])
    return [pts[off[i]:off[i + 1]].astype(np.float64) for i in range(len(ln))]


SODIUM = hex_lin("#FF9F3A")
LED = hex_lin("#EAF2FF")
WARM_LED = hex_lin("#FFE3B8")


def _build_lamps():
    L = layout()
    rng = np.random.default_rng(11)
    P, E, K, H = [], [], [], []

    def add(xz, y, col, kind, pool_h):
        n = len(xz)
        if n == 0:
            return
        P.append(np.stack([xz[:, 0], np.broadcast_to(y, n), xz[:, 1]], 1))
        E.append(np.broadcast_to(np.asarray(col, np.float64), (n, 3)) * (0.8 + 0.4 * rng.random(n))[:, None])
        K.append(np.full(n, kind, np.uint8))
        H.append(np.broadcast_to(np.asarray(pool_h, np.float64), (n,)))

    D = L["dist"]
    led_frac = {T_OLD: 0.15, T_NEW: 0.55, T_CBD: 0.8, T_MIX: 0.35, T_IND: 0.25, T_SUB: 0.4}
    polys = _polys(L, "st")
    for i, poly in enumerate(polys):
        cls = int(L["st_cls"][i])
        dd = int(L["st_dist"][i])
        typ = T_MIX if dd == 0 else int(D[dd - 1, 3])
        if cls == 0:
            pitch, hh, inten, sides = 20.0, 11.0, 1.0, (-23.0, 23.0, 0.0)
            col = SODIUM
        elif cls == 1:
            pitch, hh, inten, sides = 30.0, 10.0, 0.62, (-14.0, 14.0)
            col = LED if _h1(i, 5) < led_frac.get(typ, 0.3) else SODIUM
        elif cls == 2:
            pitch, hh, inten, sides = 34.0, 8.0, 0.30, (-7.0, 7.0)
            col = LED * 0.9 if _h1(i, 6) < led_frac.get(typ, 0.3) else SODIUM
        elif cls == 5:
            pitch, hh, inten, sides = 24.0, 3.5, 0.10, (1.5,)
            col = hex_lin("#FFE2B0")
        else:
            continue
        for k, sd in enumerate(sides):
            pts, t = _resample(poly, pitch, offset=(pitch * 0.5 * k + 7 * i) % pitch)
            if len(pts) == 0:
                continue
            nrm = np.stack([-t[:, 1], t[:, 0]], 1)
            xz = pts + nrm * sd
            add(xz, hh, col * inten * (0.8 if sd == 0.0 else 1.0), L_PARK if cls == 5 else L_STREET,
                hh if cls != 5 else 0.0)
    # expressways + bridges (on their decks)
    polys = _polys(L, "ex")
    for i, poly in enumerate(polys):
        hdeck = float(L["ex_h"][i])
        br = bool(L["ex_bridge"][i])
        pts, t = _resample(poly, 30.0 if not br else 24.0)
        nrm = np.stack([-t[:, 1], t[:, 0]], 1)
        col = (LED if i % 2 == 0 else SODIUM) if not br else hex_lin("#F4F0FF")
        for sd in (-12.0, 12.0) if not br else (-13.0, 13.0):
            add(pts + nrm * sd, hdeck + 9.0, col * (0.62 if not br else 0.7), L_BRIDGE if br else L_EXPR, 9.0)
        if br:
            # decorative railing lights (dense, dim), both sides
            pts2, t2 = _resample(poly, 4.0)
            n2 = np.stack([-t2[:, 1], t2[:, 0]], 1)
            for sd in (-14.5, 14.5):
                add(pts2 + n2 * sd, hdeck + 1.2, hex_lin("#9FD8FF") * 0.05, L_DECO, 0.0)
    # river promenades (reflecting)
    RC, RH = L["river_c"].astype(np.float64), L["river_hw"].astype(np.float64)
    pts, t = _resample(RC, 22.0)
    hw = np.interp(np.arange(len(pts)), np.linspace(0, len(pts) - 1, len(RH)), RH)
    nrm = np.stack([-t[:, 1], t[:, 0]], 1)
    for sg in (-1.0, 1.0):
        xz = pts + nrm * (sg * (hw + 14.0))[:, None]
        ok = np.hypot(xz[:, 0], xz[:, 1]) < CITY_R - 600
        add(xz[ok], 4.5, hex_lin("#FFD9A0") * 0.22, L_BANK, 0.0)
    # lake shore in the park
    th = np.linspace(0, 2 * np.pi, 70, endpoint=False)
    c = np.stack([1.04 + 0.087 * 1.2 * np.cos(th), -0.47 + 0.087 * 0.75 * np.sin(th)], 1)
    add(canon_to_world(c)[:, [0, 2]], 3.0, hex_lin("#FFE2B0") * 0.10, L_BANK, 0.0)
    P = np.concatenate(P)
    ring = ring_of(P[:, [0, 2]])
    return dict(P=P.astype(np.float32), E=np.concatenate(E).astype(np.float32), kind=np.concatenate(K),
                pool=np.concatenate(H).astype(np.float32), ring=ring)


def _build_windows():
    L = layout()
    rng = np.random.default_rng(12)
    x, z, a, b, yaw = [L["b_" + k].astype(np.float64) for k in ("x", "z", "a", "b", "yaw")]
    fl, fh, typ, tag = L["b_fl"].astype(np.int64), L["b_fh"].astype(np.float64), L["b_typ"], L["b_tag"]
    nb = len(x)
    r = np.hypot(x, z)
    eu = np.stack([np.cos(yaw), -np.sin(yaw)], 1)
    ev = np.stack([-np.sin(yaw), -np.cos(yaw)], 1)
    pitch = np.select([typ == T_OLD, typ == T_NEW, typ == T_CBD, typ == T_IND], [3.6, 3.5, 3.2, 10.0], 3.7)
    p_lit = np.select([typ == T_OLD, typ == T_NEW, typ == T_CBD, typ == T_IND], [0.36, 0.33, 0.40, 0.12], 0.30)
    p_lit = p_lit * (0.7 + 0.6 * rng.random(nb)) * np.clip(1.15 - r / 30000.0, 0.6, 1.0)
    warm = np.select([typ == T_OLD, typ == T_NEW, typ == T_CBD, typ == T_IND], [0.78, 0.62, 0.22, 0.3], 0.6)
    # emergency buildings (hospital generators) seen from the rooftop, bearing ~195 / ~168 deg
    emerg = np.zeros(nb, bool)
    for brg, dist in ((196.0, 4300.0), (171.0, 6200.0)):
        tgt = np.array([ROOF_ORIGIN[0] + dist * np.sin(np.radians(brg)), ROOF_ORIGIN[2] - dist * np.cos(np.radians(brg))])
        cand = np.nonzero((fl >= 8) & (fl <= 20))[0]
        j = cand[np.argmin(np.hypot(x[cand] - tgt[0], z[cand] - tgt[1]))]
        emerg[j] = True
        p_lit[j] = 0.75
    Ps, Es, Ns, Ks, Bs = [], [], [], [], []
    # facades: 0 +ev, 1 -ev, 2 +eu, 3 -eu
    for f in range(4):
        n_vec = [ev, -ev, eu, -eu][f]
        t_vec = [eu, -eu, -ev, ev][f]
        half_n = [b, b, a, a][f]
        length = 2 * [a, a, b, b][f]
        ncol = np.maximum(1, np.floor(length / pitch)).astype(np.int64)
        if f >= 2:
            # slab ends: a single column of small windows (often none)
            slab = typ == T_OLD
            ncol = np.where(slab, (rng.random(nb) < 0.5).astype(np.int64), ncol)
        slots = ncol * fl
        B0 = 0
        CH = 6000
        for c0 in range(0, nb, CH):
            sl = slice(c0, min(nb, c0 + CH))
            ns = slots[sl]
            tot = int(ns.sum())
            if tot == 0:
                continue
            bi = np.repeat(np.arange(sl.start, sl.stop), ns)
            first = np.repeat(np.cumsum(ns) - ns, ns)
            k = np.arange(tot) - first                   # slot index inside the building facade
            colj = k % ncol[bi]
            floor = k // ncol[bi]
            gid = bi * 131071 + f * 7919 + k * 3
            u = hash01(gid, 21)
            office = typ[bi] == T_CBD
            floor_on = hash01(bi * 977 + floor * 13 + f, 22) < np.where(office, 0.62, 1.0)
            lit = (u < p_lit[bi] * np.where(office, 1.35, 1.0)) & floor_on
            lit |= (floor == 0) & (typ[bi] != T_CBD) & (u < 0.30)          # ground-floor shops / lobbies
            idx = np.nonzero(lit)[0]
            if idx.size == 0:
                continue
            bi, colj, floor, u, office = bi[idx], colj[idx], floor[idx], u[idx], office[idx]
            nc = ncol[bi]
            w = length[bi] / nc
            off_t = (colj - (nc - 1) * 0.5) * w
            jit = (hash01(gid[idx], 23) - 0.5) * w * 0.25
            y = (floor + 0.55) * fh[bi] + (hash01(gid[idx], 24) - 0.5) * 0.3
            c = np.stack([x[bi], z[bi]], 1)
            xz = c + n_vec[bi] * (half_n[bi] + 0.3)[:, None] + t_vec[bi] * (off_t + jit)[:, None]
            Ps.append(np.stack([xz[:, 0], y, xz[:, 1]], 1).astype(np.float32))
            # colour: warm incandescent / cool LED-fluorescent, curtains dim some
            h2 = hash01(gid[idx], 25)
            iswarm = h2 < warm[bi]
            T = np.where(iswarm, 2500 + 900 * hash01(gid[idx], 26), 4200 + 2600 * hash01(gid[idx], 27))
            col = _bb(T, 0.35 if not True else 0.30)
            inten = 0.35 + 0.9 * hash01(gid[idx], 28) ** 2
            kind = np.full(idx.size, K_WIN, np.uint8)
            shop = (floor == 0) & (typ[bi] != T_CBD)
            if shop.any():
                hs = hash01(gid[idx][shop], 29)
                pal = np.array([hex_lin(h) for h in ("#FF3B5C", "#FFB23A", "#3AE0FF", "#FF5AE0", "#FFF2D0",
                                                     "#7CFF8A", "#FF7A3A", "#EAF2FF")])
                col[shop] = pal[(hs * len(pal)).astype(int) % len(pal)]
                inten[shop] = 1.1 + 1.2 * hash01(gid[idx][shop], 30)
                kind[shop] = K_SHOP
            fk = hash01(gid[idx], 31) < 0.02
            kind[fk & ~shop] = K_FLICK
            em = emerg[bi]
            kind[em] = K_EMERG
            Es.append((col * inten[:, None] * np.where(office, 0.8, 1.0)[:, None]).astype(np.float32))
            Ns.append(np.full(idx.size, _ncode(n_vec[bi[0], 0], n_vec[bi[0], 1]), np.int64) if False else
                      _ncode(n_vec[bi, 0], n_vec[bi, 1]))
            Ks.append(kind)
            Bs.append(bi.astype(np.int32))
    # stairwell lights (voice-activated) on the north face of old slabs: one column per ~25 m
    old = np.nonzero(typ == T_OLD)[0]
    nst = np.maximum(1, np.round(2 * a[old] / 25.0)).astype(np.int64)
    tot = nst * fl[old]
    bi = np.repeat(old, tot)
    first = np.repeat(np.cumsum(tot) - tot, tot)
    k = np.arange(int(tot.sum())) - first
    sc = k % np.repeat(nst, tot)
    floor = k // np.repeat(nst, tot)
    off_t = (sc - (np.repeat(nst, tot) - 1) * 0.5) * (2 * a[bi] / np.repeat(nst, tot))
    c = np.stack([x[bi], z[bi]], 1)
    xz = c + ev[bi] * (b[bi] + 0.3)[:, None] + eu[bi] * off_t[:, None]
    y = (floor + 0.5) * 3.0 + 1.5
    Ps.append(np.stack([xz[:, 0], y, xz[:, 1]], 1).astype(np.float32))
    Es.append((np.broadcast_to(_bb(np.array(3000.0), 0.2), (len(bi), 3)) * 0.35).astype(np.float32))
    Ns.append(_ncode(ev[bi, 0], ev[bi, 1]))
    Ks.append(np.full(len(bi), K_STAIR, np.uint8))
    Bs.append(bi.astype(np.int32))
    # crowns + aviation lights on tall buildings
    tall = np.nonzero(fl * fh > 95.0)[0]
    for j in tall:
        top = fl[j] * fh[j]
        cxz = np.array([x[j], z[j]])
        corners = np.array([cxz + sa * a[j] * eu[j] + sb * b[j] * ev[j] for sa, sb in ((1, 1), (1, -1), (-1, -1), (-1, 1))])
        # aviation: red, at the corners of the roof (+ mid-height pairs on supertalls)
        av = [np.c_[corners[:, 0], np.full(4, top + 1.5), corners[:, 1]]]
        if top > 250:
            av.append(np.c_[corners[::2, 0], np.full(2, top * 0.5), corners[::2, 1]])
        av = np.concatenate(av)
        Ps.append(av.astype(np.float32))
        Es.append(np.broadcast_to(hex_lin("#FF2A1A") * 2.2, (len(av), 3)).astype(np.float32))
        Ns.append(np.full(len(av), 255))
        Ks.append(np.full(len(av), K_AVI, np.uint8))
        Bs.append(np.full(len(av), j, np.int32))
        if top > 150 or _h1(j, 40) < 0.25:
            # LED crown: dense points along the roof perimeter (+ a second band for supertalls)
            per = np.concatenate([np.linspace(corners[q], corners[(q + 1) % 4], int(np.linalg.norm(corners[q] - corners[(q + 1) % 4]) / 1.6), endpoint=False) for q in range(4)])
            ys = [top - 0.5] + ([top - 12.0] if top > 250 else [])
            hc = _h1(j, 41)
            ccol = hex_lin("#EAF2FF") if hc < 0.5 else (hex_lin("#FFD9A0") if hc < 0.8 else hex_lin("#7FE3FF"))
            for yy in ys:
                Ps.append(np.c_[per[:, 0], np.full(len(per), yy), per[:, 1]].astype(np.float32))
                Es.append(np.broadcast_to(ccol * 0.28, (len(per), 3)).astype(np.float32))
                Ns.append(np.full(len(per), 255))
                Ks.append(np.full(len(per), K_CROWN, np.uint8))
                Bs.append(np.full(len(per), j, np.int32))
    # substation yard floodlights
    th = np.linspace(0, 2 * np.pi, 9, endpoint=False)
    yard = np.c_[SUBSTATION_POS[0] + 38 * np.cos(th), np.full(9, 9.0), SUBSTATION_POS[2] + 30 * np.sin(th)]
    Ps.append(yard.astype(np.float32))
    Es.append(np.broadcast_to(SODIUM * 1.4, (9, 3)).astype(np.float32))
    Ns.append(np.full(9, 255))
    Ks.append(np.full(9, K_YARD, np.uint8))
    Bs.append(np.full(9, -1, np.int32))
    P = np.concatenate(Ps)
    E = np.concatenate(Es)
    Nc = np.concatenate(Ns).astype(np.uint8)
    K = np.concatenate(Ks)
    Bi = np.concatenate(Bs)
    # ring: per building (whole block switches together), yard/specials by position
    bring = ring_of(np.stack([x, z], 1))
    ring = np.where(Bi >= 0, bring[np.maximum(Bi, 0)], ring_of(P[:, [0, 2]])).astype(np.uint8)
    # sort by tile, then by a random key (LOD = prefix of each tile)
    tile = _tile_of(P[:, [0, 2]].astype(np.float64))
    key = hash01(np.arange(len(P)), 33)
    order = np.lexsort((key, tile))
    tile = tile[order]
    starts = np.searchsorted(tile, np.arange(NT * NT + 1))
    return dict(P=P[order], E=E[order].astype(np.float16), nrm=Nc[order], kind=K[order], ring=ring[order],
                bid=Bi[order], tile_start=starts.astype(np.int64))


def lamps():
    if "lamps" not in _L:
        _L["lamps"] = _cached("lamps", _build_lamps)
    return _L["lamps"]


def windows():
    if "win" not in _L:
        _L["win"] = _cached("windows", _build_windows)
    return _L["win"]


# ================================================================================================ traffic
def _build_traffic():
    L = layout()
    rng = np.random.default_rng(13)
    lane_pts, lane_len, cars_lane, cars_s, cars_v = [], [], [], [], []
    spec = {0: ((5.0, 8.5, 12.0), 20.0, 11.0), 1: ((4.0, 7.5), 38.0, 10.0), 2: ((3.0,), 190.0, 7.0)}

    def add_lane(pts2, y, gap, v):
        d = np.linalg.norm(np.diff(pts2, axis=0), axis=1)
        Ls = float(d.sum())
        if Ls < 60:
            return
        li = len(lane_len)
        lane_pts.append(np.c_[pts2[:, 0], np.broadcast_to(y, len(pts2)), pts2[:, 1]])
        lane_len.append(Ls)
        n = rng.poisson(Ls / gap)
        if n == 0:
            return
        cars_lane.append(np.full(n, li, np.int32))
        cars_s.append(rng.random(n) * Ls)
        cars_v.append(v * rng.uniform(0.75, 1.2, n))

    polys = _polys(L, "st")
    for i, poly in enumerate(polys):
        cls = int(L["st_cls"][i])
        if cls not in spec:
            continue
        offs, gap, v = spec[cls]
        pts, t = _resample(poly, 10.0)
        if len(pts) < 3:
            continue
        nrm = np.stack([-t[:, 1], t[:, 0]], 1)
        for o in offs:
            add_lane(pts + nrm * o, 0.8, gap, v)                    # forward lanes (right-hand traffic)
            add_lane((pts - nrm * o)[::-1], 0.8, gap, v)            # opposite direction
    polys = _polys(L, "ex")
    for i, poly in enumerate(polys):
        br = bool(L["ex_bridge"][i])
        hdeck = float(L["ex_h"][i])
        pts, t = _resample(poly, 10.0)
        nrm = np.stack([-t[:, 1], t[:, 0]], 1)
        offs, gap, v = ((3.5, 7.0, 10.5), 24.0, 17.0) if not br else ((4.0, 8.0), 30.0, 10.0)
        for o in offs:
            add_lane(pts + nrm * o, hdeck + 0.8, gap, v)
            add_lane((pts - nrm * o)[::-1], hdeck + 0.8, gap, v)
    # concatenate with a global arclength (lanes separated by 5 m gaps)
    cum, start = [], []
    s0 = 0.0
    for p in lane_pts:
        d = np.concatenate([[0], np.cumsum(np.linalg.norm(np.diff(p[:, [0, 2]], axis=0), axis=1))])
        start.append(s0)
        cum.append(d + s0)
        s0 += d[-1] + 5.0
    return dict(pts=np.concatenate(lane_pts).astype(np.float32), cum=np.concatenate(cum),
                start=np.array(start), length=np.array(lane_len),
                car_lane=np.concatenate(cars_lane), car_s=np.concatenate(cars_s).astype(np.float32),
                car_v=np.concatenate(cars_v).astype(np.float32))


def traffic():
    if "traffic" not in _L:
        _L["traffic"] = _cached("traffic", _build_traffic)
    return _L["traffic"]


def cars_at(tg):
    """Car positions (N,3) and unit driving directions (N,3) at global time tg."""
    T = traffic()
    ln = T["car_lane"]
    Ls = T["length"][ln]
    s = np.mod(T["car_s"].astype(np.float64) + T["car_v"] * tg, Ls - 3.0)
    sg = T["start"][ln] + s
    cum, pts = T["cum"], T["pts"]
    P = np.stack([np.interp(sg, cum, pts[:, k]) for k in range(3)], 1)
    P2 = np.stack([np.interp(sg + 2.5, cum, pts[:, k]) for k in range(3)], 1)
    d = P2 - P
    d /= np.linalg.norm(d, axis=1, keepdims=True) + 1e-9
    return P, d


# ================================================================================================ power
EV = TL.EVENTS
T_FLASH = float(EV["transformer_flash"])
T_OFF = np.array(TL.BLACKOUT_OFF, float)          # ring k starts to die at T_OFF[k]
T_ON = np.array(TL.POWER_ON, float)[::-1]         # tide from the far edge: ring 13 first ... ring 0 last
T_LAMP_OFF = float(EV["rooftop_lamp_off"])
T_LAMP_ON = float(TL.POWER_ON[-1])
T_STARS = float(EV["stars_begin"])
T_MID = 0.5 * (T_OFF[-1] + T_ON.min())

# flicker of a dying circuit (s after the switch-off starts) and of a returning one
_FX = np.array([0.0, 0.02, 0.04, 0.06, 0.08, 0.10, 0.12, 0.14, 0.16, 0.18, 0.20, 0.22, 0.25])
_FY = np.array([1.0, 0.45, 0.9, 0.3, 0.75, 0.12, 0.55, 0.06, 0.3, 0.02, 0.1, 0.0, 0.0])
_OX = np.array([0.0, 0.03, 0.06, 0.09, 0.12, 0.16, 0.20, 0.26, 0.34, 0.5])
_OY = np.array([0.0, 0.55, 0.08, 0.8, 0.25, 0.95, 0.6, 0.9, 1.0, 1.0])


def level(tg, ring, jit=0.0):
    """Per-light supply level 0..1 (vectorised over ring/jit arrays)."""
    ring = np.asarray(ring)
    if tg < T_MID:
        x = tg - T_OFF[ring] - jit
        return np.interp(x, _FX, _FY, left=1.0, right=0.0)
    y = tg - T_ON[ring] - jit
    return np.interp(y, _OX, _OY, left=0.0, right=1.0)


def _ring_weights():
    rmid = np.sqrt(np.maximum(RING_R[:-1], 1.0) * np.minimum(RING_R[1:], 20000.0))
    cnt = np.array([50, 665, 263, 2199, 6005, 11261, 21160, 41668, 93395, 246266, 599960, 1469055, 2361055,
                    1668561], float)
    w = cnt / (1.0 + (rmid / 1200.0) ** 2.2)
    return w / w.sum()


_RW = _ring_weights()


def _flash(tg):
    """Substation arc: a few blue-white pulses, then a dull orange burn that fades."""
    t = tg - T_FLASH
    if t < -0.01 or t > 6.0:
        return 0.0, 0.0
    arc = 0.0
    for t0, dur, amp in ((0.0, 0.10, 1.0), (0.13, 0.07, 0.6), (0.245, 0.05, 0.35), (0.37, 0.04, 0.2),
                         (0.62, 0.03, 0.12)):
        x = (t - t0) / dur
        if 0 <= x <= 1:
            arc = max(arc, amp * (1 - x) ** 1.5 * min(1.0, x * 8 + 0.3))
    burn = 0.0 if t < 0.1 else 0.25 * np.exp(-(t - 0.1) / 1.6) * (0.8 + 0.2 * np.sin(t * 23.0) * np.sin(t * 7.3))
    return float(arc), float(burn)


def power(tg):
    """Electrical state of the city at global time tg.

    ring   (14,) supply level of each blackout ring (with flicker), ring k = distance band k from the
           substation (RING_R); off at TL.BLACKOUT_OFF[k], back on at TL.POWER_ON[13-k]
    glow   sky-glow (light pollution) level 0..1;  glow_near / glow_far: zenith / horizon parts
    stars  star visibility 0..1 (dark adaptation; faintest stars need the highest value)
    mw     Milky Way visibility 0..1
    lamp   the rooftop bulb 0..1 (dies at rooftop_lamp_off, returns with the last ring)
    flash, burn   substation arc intensity (blue-white) and the smouldering glow after it
    """
    lv = level(tg, np.arange(N_RING))
    glow = float(np.dot(_RW, lv))
    near = float(lv[:9].mean())
    far = float(np.dot(_RW[9:], lv[9:]) / _RW[9:].sum())
    # dark adaptation after the blackout, drowned again by the returning glow
    if tg < T_STARS:
        adapt = 0.0
    else:
        a = np.clip((tg - T_STARS) / 10.5, 0, 1)
        adapt = float(a * (1.6 - 0.6 * a))
    stars = adapt * float(np.clip(1.0 - glow * 1.15, 0, 1) ** 1.6)
    mw = float(np.clip((stars - 0.52) / 0.40, 0, 1))
    mw = mw * mw * (3 - 2 * mw)
    if tg < T_MID:
        lamp = float(np.interp(tg - T_LAMP_OFF, [0.0, 0.03, 0.07, 0.10, 0.16, 0.2, 0.26, 0.3, 0.42],
                               [1.0, 0.35, 0.95, 0.5, 0.8, 0.15, 0.45, 0.12, 0.0], left=1.0, right=0.0))
    else:
        lamp = float(np.interp(tg - T_LAMP_ON, _OX, _OY, left=0.0, right=1.0))
    arc, burn = _flash(tg)
    return dict(ring=lv, glow=glow, glow_near=near, glow_far=far, stars=stars, mw=mw, lamp=lamp,
                flash=arc, burn=burn, t=tg)


# ================================================================================================ the sky
LATITUDE = 31.0
LST_H = 18.5          # local sidereal time: galactic centre low in the south, band arching overhead (ESE)


def _eq_to_world(ra_deg, dec_deg, lst_h=LST_H, lat=LATITUDE):
    H = np.radians(lst_h * 15.0 - ra_deg)
    d, ph = np.radians(dec_deg), np.radians(lat)
    U = np.sin(d) * np.sin(ph) + np.cos(d) * np.cos(ph) * np.cos(H)
    Nn = np.sin(d) * np.cos(ph) - np.cos(d) * np.sin(ph) * np.cos(H)
    E = -np.cos(d) * np.sin(H)
    return np.array([E, U, -Nn])


GC_DIR = _eq_to_world(266.405, -28.936)           # galactic centre (world direction)
NGP_DIR = _eq_to_world(192.859, 27.128)           # north galactic pole


def _gbasis():
    from .sky import galactic_basis
    return galactic_basis(GC_DIR, NGP_DIR)


MW_BASIS = _gbasis()                              # world_dirs = galactic_dirs @ MW_BASIS.T

_SKY = {}


def sky_assets():
    if "s" not in _SKY:
        from . import sky as SK
        f = SK.field()
        mw = SK.milky_way()
        M = MW_BASIS.astype(np.float32)
        fd = f["dirs"].astype(np.float32)
        s_d = (mw["s_dirs"] @ M.T).astype(np.float32)
        g_d = (mw["g_dirs"] @ M.T).astype(np.float32)
        f_flux = f["rgb"].max(axis=1)
        s_flux = mw["s_rgb"].max(axis=1)
        _SKY["s"] = dict(f_dirs=fd, f_rgb=f["rgb"], f_flux=f_flux, s_dirs=s_d, s_rgb=mw["s_rgb"], s_flux=s_flux,
                         g_dirs=g_d, g_rgb=mw["g_rgb"])
    return _SKY["s"]


def _extinction(el):
    """Atmospheric + haze extinction factor for a direction of elevation el (radians)."""
    am = 1.0 / np.maximum(np.sin(np.maximum(el, 0.0)) + 0.025, 0.03)
    return 10 ** (-0.4 * 0.42 * (am - 1.0)) * np.clip(el / np.radians(1.5), 0, 1)


def star_weight(flux, vis, width=0.9):
    """Dark-adaptation visibility per star: limiting magnitude rises with vis; faint stars last."""
    if vis <= 0:
        return np.zeros_like(flux)
    m = -2.5 * np.log10(np.maximum(flux, 1e-12))
    # the field's brightest stars have flux ~ 10**(0.4*11) ... map vis to a limiting magnitude
    m_lim = M_BRIGHT + (M_FAINT - M_BRIGHT) * vis
    return np.clip((m_lim - m) / width + 0.5, 0, 1) ** 1.5


M_BRIGHT, M_FAINT = -7.6, 2.9           # in units of -2.5 log10(flux) of sky.field()/milky_way()


# ================================================================================================ z-buffer
from numba import njit  # noqa: E402


@njit(cache=True)
def _tri(zb, x0, y0, w0, x1, y1, w1, x2, y2, w2):
    H, W = zb.shape
    minx = max(int(np.floor(min(x0, min(x1, x2)))), 0)
    maxx = min(int(np.ceil(max(x0, max(x1, x2)))), W - 1)
    miny = max(int(np.floor(min(y0, min(y1, y2)))), 0)
    maxy = min(int(np.ceil(max(y0, max(y1, y2)))), H - 1)
    if minx > maxx or miny > maxy:
        return
    area = (x1 - x0) * (y2 - y0) - (x2 - x0) * (y1 - y0)
    if abs(area) < 1e-12:
        return
    inv = 1.0 / area
    for py in range(miny, maxy + 1):
        cy = py + 0.5
        for px in range(minx, maxx + 1):
            cx = px + 0.5
            b0 = ((x1 - cx) * (y2 - cy) - (x2 - cx) * (y1 - cy)) * inv
            b1 = ((x2 - cx) * (y0 - cy) - (x0 - cx) * (y2 - cy)) * inv
            b2 = 1.0 - b0 - b1
            if b0 < -1e-6 or b1 < -1e-6 or b2 < -1e-6:
                continue
            w = b0 * w0 + b1 * w1 + b2 * w2
            if w <= 0:
                continue
            z = 1.0 / w
            if z < zb[py, px]:
                zb[py, px] = z


@njit(cache=True)
def _zboxes(zb, B, pos, R, U, F, fpx, cx0, cy0, near, tanx, tany):
    """Rasterise oriented boxes into a depth buffer (camera-space z).

    B: (n, 8) rows [cx, cz, ux, uz, a, b, y0, y1]; u axis (ux, uz) in the ground plane, v = (-uz, ux)."""
    corners = np.empty((8, 3))
    cc = np.empty((8, 3))
    face_idx = np.array([[4, 5, 6, 7], [0, 1, 5, 4], [2, 3, 7, 6], [1, 2, 6, 5], [3, 0, 4, 7]])
    poly = np.empty((10, 3))
    tmp = np.empty((10, 3))
    for i in range(B.shape[0]):
        cx, cz, ux, uz, a, b, y0, y1 = B[i, 0], B[i, 1], B[i, 2], B[i, 3], B[i, 4], B[i, 5], B[i, 6], B[i, 7]
        vx, vz = -uz, ux
        # quick frustum cull with a bounding sphere
        mx, my, mz = cx - pos[0], 0.5 * (y0 + y1) - pos[1], cz - pos[2]
        rad = np.sqrt(a * a + b * b + 0.25 * (y1 - y0) ** 2)
        zc = mx * F[0] + my * F[1] + mz * F[2]
        if zc < -rad:
            continue
        xc = mx * R[0] + my * R[1] + mz * R[2]
        yc = mx * U[0] + my * U[1] + mz * U[2]
        if abs(xc) > (max(zc, 0.0) * tanx + rad * 1.5) or abs(yc) > (max(zc, 0.0) * tany + rad * 1.5):
            continue
        k = 0
        for yy in (y0, y1):
            for (su, sv) in ((-1.0, -1.0), (1.0, -1.0), (1.0, 1.0), (-1.0, 1.0)):
                corners[k, 0] = cx + su * a * ux + sv * b * vx
                corners[k, 1] = yy
                corners[k, 2] = cz + su * a * uz + sv * b * vz
                k += 1
        for k in range(8):
            dx, dy, dz = corners[k, 0] - pos[0], corners[k, 1] - pos[1], corners[k, 2] - pos[2]
            cc[k, 0] = dx * R[0] + dy * R[1] + dz * R[2]
            cc[k, 1] = dx * U[0] + dy * U[1] + dz * U[2]
            cc[k, 2] = dx * F[0] + dy * F[1] + dz * F[2]
        # faces: top, and the four sides; cull those facing away from the camera
        for f in range(5):
            if f == 0:
                if pos[1] <= y1:
                    continue
            else:
                # outward normal of side f
                if f == 1:
                    nx, nz, px_, pz_ = -vx, -vz, cx - b * vx, cz - b * vz
                elif f == 2:
                    nx, nz, px_, pz_ = vx, vz, cx + b * vx, cz + b * vz
                elif f == 3:
                    nx, nz, px_, pz_ = ux, uz, cx + a * ux, cz + a * uz
                else:
                    nx, nz, px_, pz_ = -ux, -uz, cx - a * ux, cz - a * uz
                if nx * (pos[0] - px_) + nz * (pos[2] - pz_) <= 0:
                    continue
            m = 0
            for q in range(4):
                c = face_idx[f, q]
                poly[m, 0], poly[m, 1], poly[m, 2] = cc[c, 0], cc[c, 1], cc[c, 2]
                m += 1
            # clip against z > near
            n2 = 0
            for q in range(m):
                a0 = poly[q]
                a1 = poly[(q + 1) % m]
                in0 = a0[2] > near
                in1 = a1[2] > near
                if in0:
                    tmp[n2, 0], tmp[n2, 1], tmp[n2, 2] = a0[0], a0[1], a0[2]
                    n2 += 1
                if in0 != in1:
                    t = (near - a0[2]) / (a1[2] - a0[2])
                    tmp[n2, 0] = a0[0] + t * (a1[0] - a0[0])
                    tmp[n2, 1] = a0[1] + t * (a1[1] - a0[1])
                    tmp[n2, 2] = near
                    n2 += 1
            if n2 < 3:
                continue
            # project + fan
            sx0 = cx0 + fpx * tmp[0, 0] / tmp[0, 2]
            sy0 = cy0 - fpx * tmp[0, 1] / tmp[0, 2]
            w0 = 1.0 / tmp[0, 2]
            for q in range(1, n2 - 1):
                sx1 = cx0 + fpx * tmp[q, 0] / tmp[q, 2]
                sy1 = cy0 - fpx * tmp[q, 1] / tmp[q, 2]
                sx2 = cx0 + fpx * tmp[q + 1, 0] / tmp[q + 1, 2]
                sy2 = cy0 - fpx * tmp[q + 1, 1] / tmp[q + 1, 2]
                _tri(zb, sx0, sy0, w0, sx1, sy1, 1.0 / tmp[q, 2], sx2, sy2, 1.0 / tmp[q + 1, 2])


def _cam_params(cam):
    return (cam.pos.astype(np.float64), cam.right.astype(np.float64), cam.up.astype(np.float64),
            cam.fwd.astype(np.float64), float(cam.fpx), float(cam.W * 0.5 + cam.shift[0]),
            float(cam.H * 0.5 - cam.shift[1]))


def zbuffer(cam, boxes, W, H):
    """Depth buffer (H,W) of camera-space z (inf = empty) for boxes (n, 8)."""
    zb = np.full((H, W), np.inf, np.float64)
    pos, Rv, Uv, Fv, fpx, cx0, cy0 = _cam_params(cam)
    tanx = (W * 0.5 + abs(cam.shift[0])) / fpx
    tany = (H * 0.5 + abs(cam.shift[1])) / fpx
    _zboxes(zb, np.ascontiguousarray(boxes, np.float64), pos, Rv, Uv, Fv, fpx, cx0, cy0, 0.3, tanx, tany)
    return zb


def building_boxes(exclude_tag=()):
    """All buildings as boxes (n, 8): [cx, cz, ux, uz, a, b, y0, y1]."""
    key = ("boxes",) + tuple(exclude_tag)
    if key not in _L:
        L = layout()
        yaw = L["b_yaw"].astype(np.float64)
        h = L["b_fl"] * L["b_fh"].astype(np.float64)
        B = np.stack([L["b_x"], L["b_z"], np.cos(yaw), -np.sin(yaw), L["b_a"], L["b_b"], np.zeros(len(yaw)), h], 1)
        keep = np.ones(len(B), bool)
        for tg in exclude_tag:
            keep &= L["b_tag"] != tg
        _L[key] = np.ascontiguousarray(B[keep], np.float64)
    return _L[key]


# ================================================================================================ drawing
def _view_dirs(cam, n=24):
    """World directions of rays through the frame border (and centre)."""
    W, H = cam.W, cam.H
    t = np.linspace(0, 1, n)
    xs = np.concatenate([t * W, t * W, np.zeros(n), np.full(n, W), [W / 2]])
    ys = np.concatenate([np.zeros(n), np.full(n, H), t * H, t * H, [H / 2]])
    xc = (xs - W * 0.5 - cam.shift[0]) / cam.fpx
    yc = -(ys - H * 0.5 + cam.shift[1]) / cam.fpx
    D = xc[:, None] * cam.right[None] + yc[:, None] * cam.up[None] + cam.fwd[None]
    return D / np.linalg.norm(D, axis=1, keepdims=True)


def _zenith_in_view(cam):
    x, y, c, ok = cam.project_dirs(np.array([[0.0, 1.0, 0.0]]))
    return bool(ok[0] and -2 < x[0] < cam.W + 2 and -2 < y[0] < cam.H + 2)


def sky_dots(cam, spacing_px=3.0, el_min=-0.03):
    """World-anchored stipple of sky directions with ~constant screen spacing.
    Returns dirs (N,3), el (N,), az (N,), u (N,) random in [0,1)."""
    focal_px1920 = cam.focal / 36.0 * 1920.0
    dth = spacing_px / focal_px1920
    D = _view_dirs(cam)
    el = np.arcsin(np.clip(D[:, 1], -1, 1))
    az = np.arctan2(D[:, 0], -D[:, 2])
    e0 = max(el.min() - 2 * dth, el_min)
    e1 = min(el.max() + 2 * dth, np.pi / 2)
    if _zenith_in_view(cam):
        e1 = np.pi / 2
        a_lo, a_hi = -np.pi, np.pi
    else:
        ac = np.arctan2(cam.fwd[0], -cam.fwd[2])
        rel = np.mod(az - ac + np.pi, 2 * np.pi) - np.pi
        a_lo, a_hi = ac + rel.min() - 0.05, ac + rel.max() + 0.05
        if cam.fwd[1] > 0.2:      # looking up: rows near the top of the frame span more azimuth
            a_lo, a_hi = ac - min(np.pi, (a_hi - a_lo) * 1.2), ac + min(np.pi, (a_hi - a_lo) * 1.2)
    if e1 <= e0:
        return np.zeros((0, 3)), np.zeros(0), np.zeros(0), np.zeros(0)
    i0, i1 = int(np.floor(e0 / dth)), int(np.ceil(e1 / dth))
    rows = np.arange(i0, i1 + 1)
    elr = (rows + 0.5) * dth
    naz = np.maximum(1, np.round(2 * np.pi * np.cos(np.clip(elr, -1.5, 1.5)) / dth)).astype(np.int64)
    step = 2 * np.pi / naz
    j0 = np.floor(a_lo / step).astype(np.int64)
    j1 = np.ceil(a_hi / step).astype(np.int64)
    j1 = np.minimum(j1, j0 + naz - 1)
    cnt = np.maximum(j1 - j0 + 1, 0)
    tot = int(cnt.sum())
    if tot == 0:
        return np.zeros((0, 3)), np.zeros(0), np.zeros(0), np.zeros(0)
    ri = np.repeat(np.arange(len(rows)), cnt)
    jj = np.repeat(j0, cnt) + (np.arange(tot) - np.repeat(np.cumsum(cnt) - cnt, cnt))
    jm = np.mod(jj, naz[ri])
    key = (rows[ri] + 100000) * 1000003 + jm
    h1 = hash01(key, 51)
    h2 = hash01(key, 52)
    e = (rows[ri] + 0.5 + (h1 - 0.5) * 0.8) * dth
    a = (jm + 0.5 + (h2 - 0.5) * 0.8) * step[ri]
    ce = np.cos(e)
    dirs = np.stack([ce * np.sin(a), np.sin(e), -ce * np.cos(a)], 1)
    return dirs, e, a, hash01(key, 53)


# dome colours (linear HDR radiance) — through ACES these land on the BIBLE's murky #3A2412 lid:
# horizon ~#5A3A22, zenith ~#2A1C16; the blacked-out night sky ~#02040C
DOME_HOR = np.array([0.085, 0.046, 0.021])
DOME_ZEN = np.array([0.028, 0.017, 0.012])
NIGHT_ZEN = np.array([0.0026, 0.0052, 0.0155])
NIGHT_HOR = np.array([0.0045, 0.0062, 0.0125])
RESID_HOR = np.array([0.010, 0.0055, 0.0025])


def dome_radiance(dirs, el, az, cam_pos, pw):
    """Sky brightness per direction: city glow (by the rings lit along that bearing), night sky,
    the flash of the substation."""
    se = np.sin(np.maximum(el, 0.0))
    hor = np.exp(-np.maximum(el, 0) / np.radians(9.0))
    base = DOME_ZEN[None] * (1 - hor[:, None]) + DOME_HOR[None] * hor[:, None]
    base = base * (1.0 + 0.35 * np.exp(-np.maximum(el, 0) / np.radians(2.5)))[:, None]
    # which rings light this part of the sky: ground point where the line of sight crosses ~1.2 km
    D = np.clip(1200.0 / np.maximum(np.tan(np.maximum(el, np.radians(0.4))), 1e-3), 0, 15000.0)
    lv = np.zeros(len(el))
    for f in (0.55, 1.0, 1.8):
        g = cam_pos[[0, 2]][None] + (D * f)[:, None] * np.stack([np.sin(az), -np.cos(az)], 1)
        rk, rd = ring_fast(g)
        lv += level(pw["t"], rk)
    lv /= 3.0
    L = base * (0.35 * pw["glow"] + 0.65 * lv)[:, None]
    night = NIGHT_ZEN[None] * (1 - hor[:, None]) + NIGHT_HOR[None] * hor[:, None]
    L = L + night + RESID_HOR[None] * np.exp(-np.maximum(el, 0) / np.radians(4.0))[:, None] * (1 - pw["glow"])
    if pw["flash"] > 0 or pw["burn"] > 0:
        v = SUBSTATION_POS + np.array([0, 20.0, 0]) - cam_pos
        v /= np.linalg.norm(v)
        cosang = np.clip(dirs @ v, -1, 1)
        ang = np.arccos(cosang)
        fl = pw["flash"] * (0.9 * np.exp(-ang / 0.10) + 0.25 * np.exp(-ang / 0.6)) * (0.5 + 0.5 * hor)
        L = L + hex_lin("#BFE0FF")[None] * (fl * 0.9)[:, None]
        L = L + hex_lin("#FF7A30")[None] * (pw["burn"] * 0.05 * np.exp(-ang / 0.08))[:, None]
    return L


def draw_sky(R, cam, pw, spacing_px=2.6, energy=1.0):
    dirs, el, az, u = sky_dots(cam, spacing_px)
    if len(dirs) == 0:
        return
    L = dome_radiance(dirs, el, az, cam.pos, pw)
    # pointillist stipple: a fine grain of dots with mild brightness spread + rare complementary flecks
    m = 0.55 + 0.9 * u
    h = hash01(np.arange(len(u)) + 7, 54)
    tint = np.ones((len(u), 3))
    tint[h < 0.05] = [0.85, 0.9, 1.3]
    tint[(h > 0.95)] = [1.15, 1.05, 0.8]
    # energy per dot = radiance * the solid angle it stands for (px^2 @1920)
    E = L * (m[:, None] * tint) * (spacing_px ** 2)
    R.draw_dirs(cam, dirs, E.astype(np.float32), size_px=1.0, energy=energy)


def _twinkle(n_idx, tg, el, amp=0.22):
    ph = hash01(n_idx, 61) * 6.283
    fr = 2.0 + 5.0 * hash01(n_idx, 62)
    k = amp * (1.0 + 2.0 * np.exp(-np.maximum(el, 0) / 0.25))
    return 1.0 + k * np.sin(tg * fr + ph) * np.sin(tg * fr * 0.37 + ph * 1.7)


def draw_stars(R, cam, pw, energy=1.0, mw_energy=1.0):
    vis, mwv = pw["stars"], pw["mw"]
    if vis <= 0:
        return
    S = sky_assets()
    tg = pw["t"]
    for dk, ck, fk, e_mul, tag in (("f_dirs", "f_rgb", "f_flux", 1.0, 0), ("s_dirs", "s_rgb", "s_flux", 0.9, 1)):
        D, C, F = S[dk], S[ck], S[fk]
        el = np.arcsin(np.clip(D[:, 1], -1, 1))
        w = star_weight(F, vis)
        keep = (w > 0.002) & (el > 0)
        if not keep.any():
            continue
        idx = np.nonzero(keep)[0]
        ext = _extinction(el[idx])
        tw = _twinkle(idx + tag * 1000000, tg, el[idx])
        col = C[idx] * (w[idx] * ext * tw * e_mul)[:, None]
        # the brightest stars are drawn a touch larger (they bloom on film)
        br = np.clip(-2.5 * np.log10(np.maximum(F[idx], 1e-9)) , -8, 3)
        size = 0.55 + 0.22 * np.clip(-br - 2.0, 0, 5) ** 0.7
        R.draw_dirs(cam, D[idx], col.astype(np.float32), size_px=size.astype(np.float32), energy=energy)
    if mwv > 0:
        D, C = S["g_dirs"], S["g_rgb"]
        el = np.arcsin(np.clip(D[:, 1], -1, 1))
        keep = el > 0.0
        idx = np.nonzero(keep)[0]
        ext = _extinction(el[idx])
        col = C[idx] * (ext * mwv * 0.0055)[:, None]
        R.draw_dirs(cam, D[idx], col.astype(np.float32), size_px=1.5, soft=True, energy=mw_energy)


# ------------------------------------------------------------------------------------------ lights
D_REF, ALPHA, HAZE_L = 900.0, 1.25, 9000.0
WARM_SHIFT = np.array([1.0, 0.90, 0.74])


def _atten(d):
    a = (D_REF / np.maximum(d, 40.0)) ** ALPHA * np.exp(-d / HAZE_L)
    return a


def _warm(d):
    return WARM_SHIFT[None, :] ** np.clip(d / 7000.0, 0, 2.5)[:, None]


def _ztest(zb, x, y, z, tol=0.004, bias=0.8):
    H, W = zb.shape
    xi = np.clip(x.astype(np.int64), 0, W - 1)
    yi = np.clip(y.astype(np.int64), 0, H - 1)
    return z <= zb[yi, xi] * (1.0 + tol) + bias


def _splat(R, x, y, r_px, E, soft=False):
    if len(x) == 0:
        return
    R.acc.splat(np.ascontiguousarray(x, np.float32), np.ascontiguousarray(y, np.float32),
                np.ascontiguousarray(np.broadcast_to(np.asarray(r_px, np.float32) * R.s, np.shape(x)), np.float32),
                np.ascontiguousarray(E * R.e, np.float32), soft=soft)


def _in_frame(cam, x, y, ok, m=4.0):
    return ok & (x > -m) & (x < cam.W + m) & (y > -m) & (y < cam.H + m)


def _select_tiles(cam, s_target=2.2, f_min=0.015):
    Wd = windows()
    ts = Wd["tile_start"]
    cnt = np.diff(ts)
    g = (np.arange(NT) + 0.5) * TILE - CITY_R
    tz, tx = np.meshgrid(g, g, indexing="ij")
    C = np.stack([tx.ravel(), np.full(NT * NT, 80.0), tz.ravel()], 1)
    rad = 470.0
    d = C - cam.pos
    zc = d @ cam.fwd
    xc = d @ cam.right
    yc = d @ cam.up
    tanx = (cam.W * 0.5) / cam.fpx
    tany = (cam.H * 0.5) / cam.fpx
    ok = (cnt > 0) & (zc > -rad) & (np.abs(xc) < np.maximum(zc, 0) * tanx + rad * 1.5) & \
         (np.abs(yc) < np.maximum(zc, 0) * tany + rad * 1.5)
    dist = np.maximum(np.linalg.norm(d, axis=1) - rad * 0.6, 30.0)
    focal1920 = cam.focal / 36.0 * 1920.0
    d_lod = 3.5 * focal1920 / s_target
    f = np.clip((d_lod / dist) ** 2, f_min, 1.0)
    t = np.nonzero(ok)[0]
    n = np.ceil(f[t] * cnt[t]).astype(np.int64)
    tot = int(n.sum())
    if tot == 0:
        return np.zeros(0, np.int64), np.zeros(0)
    base = np.repeat(ts[t], n)
    off = np.arange(tot) - np.repeat(np.cumsum(n) - n, n)
    w = np.repeat(cnt[t] / np.maximum(n, 1), n)
    return base + off, w


def _nvec(code):
    ang = code.astype(np.float64) * (2 * np.pi / 254.0)
    return np.cos(ang), np.sin(ang)


def draw_windows(R, cam, zb, pw, energy=1.0, s_target=2.2):
    Wd = windows()
    idx, wlod = _select_tiles(cam, s_target)
    if len(idx) == 0:
        return
    P = Wd["P"][idx]
    code = Wd["nrm"][idx]
    kind = Wd["kind"][idx]
    omni = code == 255
    nx, nz = _nvec(code)
    vx = cam.pos[0] - P[:, 0]
    vz = cam.pos[2] - P[:, 2]
    vn = np.sqrt(vx * vx + vz * vz) + 1e-6
    face = (nx * vx + nz * vz) / vn
    keep = omni | (face > 0.02)
    x, y, z, coc, ok = cam.project(P)
    keep &= _in_frame(cam, x, y, ok)
    keep &= _ztest(zb, x, y, z)
    k = np.nonzero(keep)[0]
    if len(k) == 0:
        return
    gi = idx[k]
    E = Wd["E"][gi].astype(np.float64)
    kd = kind[k]
    ring = Wd["ring"][gi]
    tg = pw["t"]
    lv = level(tg, ring, hash01(gi, 71) * 0.10)
    # special behaviours
    stair = kd == K_STAIR
    if stair.any():
        ph = hash01(gi[stair], 72) * 30.0
        per = 9.0 + 14.0 * hash01(gi[stair], 73)
        on = hash01(gi[stair] * 7 + np.floor((tg + ph) / per).astype(np.int64), 74) < 0.33
        E[stair] *= on[:, None]
    flick = kd == K_FLICK
    if flick.any():
        fl = hash01(gi[flick] * 131 + np.floor(tg * 9.0).astype(np.int64), 75)
        E[flick] *= (0.35 + 0.9 * fl)[:, None] * np.array([0.8, 0.95, 1.35])
    avi = kd == K_AVI
    if avi.any():
        phase = np.mod(tg / 1.5 + hash01(Wd["bid"][gi[avi]], 76), 1.0)
        E[avi] *= (phase < 0.42)[:, None] * 1.0
        lv[avi] = 1.0                                # battery-backed obstruction lights
    em = kd == K_EMERG
    if em.any():
        lv[em] = np.maximum(lv[em], 0.55 * (hash01(gi[em], 77) < 0.55))
    f = np.where(omni[k], 1.0, 0.3 + 0.7 * np.clip(face[k], 0, 1))
    d = z[k].astype(np.float64)
    E = E * (wlod[k] * lv * f * _atten(d))[:, None] * _warm(d)
    live = lv > 0.002
    size = 0.55 + 0.5 * np.clip(1.0 - d / 2500.0, 0, 1)
    _splat(R, x[k][live], y[k][live], size[live], (E[live] * energy).astype(np.float32))


POOL_L = 0.07          # peak road radiance under a unit lamp


def draw_lamps(R, cam, zb, pw, energy=1.0, pools=True):
    Lm = lamps()
    P = Lm["P"]
    x, y, z, coc, ok = cam.project(P)
    keep = _in_frame(cam, x, y, ok, 30.0)
    k = np.nonzero(keep)[0]
    if len(k) == 0:
        return
    tg = pw["t"]
    lv = level(tg, Lm["ring"][k], hash01(k, 81) * 0.10)
    E = Lm["E"][k].astype(np.float64) * lv[:, None]
    d = z[k].astype(np.float64)
    vis = _ztest(zb, x[k], y[k], z[k]) & (lv > 0.002) & _in_frame(cam, x[k], y[k], ok[k])
    Ed = E * (_atten(d))[:, None] * _warm(d)
    size = 0.6 + 0.7 * np.clip(1.0 - d / 1500.0, 0, 1)
    _splat(R, x[k][vis], y[k][vis], size[vis], (Ed[vis] * energy).astype(np.float32))
    if pools:
        ph = Lm["pool"][k]
        pk = (ph > 0) & (lv > 0.002)
        if pk.any():
            kk = k[pk]
            Pp = P[kk].astype(np.float64).copy()
            Pp[:, 1] -= ph[pk] - 0.3
            xp, yp, zp, _, okp = cam.project(Pp)
            v = cam.pos[None] - Pp
            dist = np.linalg.norm(v, axis=1) + 1e-6
            sin_el = np.clip(v[:, 1] / dist, 0.02, 1.0)
            rw = 0.85 * ph[pk]
            r1920 = rw * (cam.focal / 36.0 * 1920.0) / np.maximum(zp, 1e-3)     # pool radius, px @1920
            rr = r1920 * np.sqrt(sin_el)                                           # foreshortened blob
            m = _in_frame(cam, xp, yp, okp, 60) & _ztest(zb, xp, yp, zp, 0.01, 3.0) & (rr > 0.35)
            if m.any():
                Ek = E[pk][m]
                big = rr[m] > 2.5
                # small (distant) pools: one soft blob each; energy = radiance * blob area
                sm = ~big
                if sm.any():
                    dp = zp[m][sm].astype(np.float64)
                    rad = rr[m][sm]
                    Ep = Ek[sm] * (POOL_L * 1.57 * rad ** 2 * np.exp(-dp / HAZE_L))[:, None] * _warm(dp)
                    _splat(R, xp[m][sm], yp[m][sm], np.maximum(rad, 0.6), (Ep * energy).astype(np.float32), soft=True)
                if big.any():
                    # near pools: a stipple of dots on the asphalt (true perspective, pointillist)
                    bi = np.nonzero(m)[0][big]
                    area_px = np.pi * r1920[bi] ** 2 * sin_el[bi]
                    nd = np.clip((area_px / 26.0).astype(np.int64), 4, 60)
                    tot = int(nd.sum())
                    rep = np.repeat(np.arange(len(bi)), nd)
                    kid = np.arange(tot) - np.repeat(np.cumsum(nd) - nd, nd)
                    gid = np.repeat(kk[bi], nd) * 97 + kid
                    rad_ = np.sqrt(hash01(gid, 86)) * rw[bi][rep] * 1.25
                    th = hash01(gid, 87) * 2 * np.pi
                    Q = Pp[bi][rep].copy()
                    Q[:, 0] += rad_ * np.cos(th)
                    Q[:, 2] += rad_ * np.sin(th)
                    fall = np.exp(-(rad_ / (rw[bi][rep] * 0.62)) ** 2)
                    xq, yq, zq, _, okq = cam.project(Q)
                    mq = _in_frame(cam, xq, yq, okq) & _ztest(zb, xq, yq, zq, 0.01, 2.0)
                    dq = zq.astype(np.float64)
                    # each dot carries (pool radiance) x (its share of the pool's screen area)
                    share = (area_px[rep] * 1.9 / nd[rep]) * fall * (0.6 + 0.8 * hash01(gid, 88))
                    Eq = Ek[big][rep] * (POOL_L * share * np.exp(-dq / HAZE_L))[:, None] * _warm(dq)
                    sz = np.clip(np.sqrt(area_px[rep] / nd[rep]) * 0.35, 0.8, 3.0)
                    _splat(R, xq[mq], yq[mq], sz[mq], (Eq[mq] * energy).astype(np.float32), soft=True)


def _river_mask():
    if "rmask" not in _L:
        def make():
            import cv2
            res = 10.0
            n = int(2 * CITY_R / res)
            m = np.zeros((n, n), np.uint8)
            RC, RH = river_line()
            for i in range(len(RC) - 1):
                a = ((RC[i] + CITY_R) / res).astype(np.int32)
                b = ((RC[i + 1] + CITY_R) / res).astype(np.int32)
                cv2.line(m, (int(a[0]), int(a[1])), (int(b[0]), int(b[1])), 1, thickness=max(1, int(2 * RH[i] / res)))
            g = (np.arange(n) + 0.5) * res - CITY_R
            gx, gz = np.meshgrid(g, g)
            lake = lake_region(np.stack([gx.ravel(), gz.ravel()], 1)).reshape(n, n)
            m[lake] = 1
            return dict(m=m)
        _L["rmask"] = _cached("rivermask", make)["m"]
    return _L["rmask"]


def draw_reflections(R, cam, zb, pw, energy=1.0, n_streak=7):
    """Glitter paths of lamps (banks, bridges) and near-river windows on the water."""
    Lm = lamps()
    kind = Lm["kind"]
    sel = np.nonzero((kind == L_BANK) | (kind == L_BRIDGE) | (kind == L_DECO))[0]
    P = Lm["P"][sel].astype(np.float64)
    tg = pw["t"]
    lv = level(tg, Lm["ring"][sel], hash01(sel, 81) * 0.10)
    E0 = Lm["E"][sel].astype(np.float64) * lv[:, None]
    live = lv > 0.002
    sel, P, E0 = sel[live], P[live], E0[live]
    if len(sel) == 0:
        return
    hv = cam.pos[[0, 2]][None] - P[:, [0, 2]]
    dist = np.linalg.norm(hv, axis=1) + 1e-6
    hv /= dist[:, None]
    # specular point on the water between lamp and camera (flat mirror)
    hc = max(cam.pos[1], 1.0)
    s_spec = dist * P[:, 1] / (P[:, 1] + hc)
    spread = 4.0 + 0.10 * dist * (hc / (hc + 60.0)) ** 0.3
    k = np.arange(n_streak)
    frac = (k + 0.5) / n_streak
    offs = (frac * 2.2 - 0.6)[None, :] * spread[:, None]
    base = P[:, [0, 2]] + hv * s_spec[:, None]
    jit = (hash01(np.arange(len(sel))[:, None] * 17 + k[None, :], 83) - 0.5) * 3.0
    side = np.stack([-hv[:, 1], hv[:, 0]], 1)
    Q = base[:, None, :] + hv[:, None, :] * offs[:, :, None] + side[:, None, :] * jit[:, :, None]
    Q = Q.reshape(-1, 2)
    wet = _lookup(_river_mask(), Q, 10.0) > 0
    Pw = np.stack([Q[:, 0], np.full(len(Q), 0.05), Q[:, 1]], 1)
    # sparkle: each streak point flickers (ripples)
    ids = (np.repeat(sel, n_streak) * 31 + np.tile(k, len(sel))).astype(np.int64)
    sp = hash01(ids * 7 + np.floor(tg * 7.0 + hash01(ids, 84) * 7).astype(np.int64), 85)
    w = np.tile(np.exp(-((frac * 2.2 - 0.6) - 0.2) ** 2 / 0.5), len(sel)) * (0.2 + 1.6 * sp ** 3)
    x, y, z, coc, ok = cam.project(Pw)
    m = wet & _in_frame(cam, x, y, ok) & _ztest(zb, x, y, z, 0.01, 2.0)
    if not m.any():
        return
    Er = np.repeat(E0, n_streak, axis=0)[m] * (w[m] * 0.55 / n_streak * 3.0)[:, None]
    d = z[m].astype(np.float64)
    Er = Er * (_atten(d))[:, None] * _warm(d)
    _splat(R, x[m], y[m], 0.6, (Er * energy).astype(np.float32))


HEAD, TAIL = hex_lin("#FFF6E0") * 1.1, hex_lin("#FF3B2F") * 0.55


def _car_level(tg, ring, surv):
    if tg < T_MID:
        a = np.clip(1.0 - (tg - T_OFF[ring] - 0.4) / 1.6, 0, 1)
    else:
        a = np.clip((tg - T_ON[ring] - 0.3) / 2.2, 0, 1)
    return np.where(surv, 1.0, a)


def draw_traffic(R, cam, zb, pw, energy=1.0):
    tg = pw["t"]
    P, dvec = cars_at(tg)
    x, y, z, coc, ok = cam.project(P)
    keep = _in_frame(cam, x, y, ok, 20)
    k = np.nonzero(keep)[0]
    if len(k) == 0:
        return
    P, dvec = P[k], dvec[k]
    ring, _ = ring_fast(P[:, [0, 2]])
    surv = hash01(k, 91) < 0.05
    cl = _car_level(tg, ring, surv)
    live = cl > 0.01
    P, dvec, cl, kk = P[live], dvec[live], cl[live], k[live]
    v = cam.pos[None] - P
    dist = np.linalg.norm(v, axis=1)
    v /= dist[:, None] + 1e-9
    c = np.sum(dvec * v, axis=1)
    fh = 0.05 + 0.95 * np.clip(c, 0, 1) ** 4
    ft = 0.16 + 0.84 * np.clip(-c, 0, 1) ** 1.5
    for off, col, f, sz in ((2.2, HEAD, fh, 0.7), (-2.2, TAIL, ft, 0.6)):
        Q = P + dvec * off
        xq, yq, zq, _, okq = cam.project(Q)
        m = _in_frame(cam, xq, yq, okq) & _ztest(zb, xq, yq, zq, 0.006, 1.5)
        d = zq[m].astype(np.float64)
        E = col[None] * (f[m] * cl[m] * _atten(d))[:, None] * _warm(d)
        _splat(R, xq[m], yq[m], sz, (E * energy).astype(np.float32))
    # headlight pools on the asphalt (reads from above)
    Q = P + dvec * 14.0
    Q[:, 1] = np.maximum(Q[:, 1] - 0.7, 0.1)
    xq, yq, zq, _, okq = cam.project(Q)
    rpx = 7.0 * cam.fpx / np.maximum(zq, 1e-3) / R.s
    m = _in_frame(cam, xq, yq, okq, 20) & (rpx > 0.8) & _ztest(zb, xq, yq, zq, 0.01, 3.0)
    if m.any():
        d = zq[m].astype(np.float64)
        E = HEAD[None] * (0.35 * cl[m] * _atten(d))[:, None] * _warm(d)
        _splat(R, xq[m], yq[m], np.minimum(rpx[m], 60), (E * energy).astype(np.float32), soft=True)


# ================================================================================================ render_env
def ground_mask(cam, W, H, far=12000.0):
    """1 where the view ray hits the ground plane within `far` metres."""
    ys = np.arange(H) + 0.5
    xs = np.arange(W) + 0.5
    # the ground is covered where the ray's downward slope beats cam_height / far; test per pixel row/col
    xc = (xs - W * 0.5 - cam.shift[0]) / cam.fpx
    yc = -(ys - H * 0.5 + cam.shift[1]) / cam.fpx
    dy = (xc[None, :] * cam.right[1] + yc[:, None] * cam.up[1] + cam.fwd[1])
    dn = np.sqrt((xc[None, :] ** 2 + yc[:, None] ** 2 + 1.0))
    return (dy / dn < -max(cam.pos[1], 0.5) / far).astype(np.float32)


# ================================================================================================ near solids
NEAR_R = 330.0          # buildings within this radius of the girl's roof get real surfaces
CONCRETE_N = np.array([0.30, 0.285, 0.265])
WALL_N = np.array([0.24, 0.225, 0.21])


def _face_points(rng, c, e1, e2, n_out, L1, L2, dens, jitter=0.0):
    """Uniform samples on a rectangle centred c spanned by unit e1 (half L1) and e2 (half L2)."""
    area = 4 * L1 * L2
    n = max(4, int(area * dens))
    u = rng.uniform(-1, 1, n)
    v = rng.uniform(-1, 1, n)
    P = c[None] + (u * L1)[:, None] * e1[None] + (v * L2)[:, None] * e2[None]
    if jitter:
        P = P + n_out[None] * rng.normal(0, jitter, (n, 1))
    return P, np.broadcast_to(n_out, (n, 3)).copy(), area / n, u, v


def _box_surface(rng, c, ax, ay, az, hx, hy, hz, dens, top=True):
    """Oriented box: centre c, unit axes ax (x), ay (up), az; half sizes. Returns P, N, area_per_pt."""
    Ps, Ns, As = [], [], []
    faces = [(ax, ay, az, hx, hy, hz), (-ax, ay, az, hx, hy, hz), (az, ay, ax, hz, hy, hx), (-az, ay, ax, hz, hy, hx)]
    for n, e1, e2, hn, h1, h2 in faces:
        P, N, a, _, _ = _face_points(rng, c + n * hn, e2, e1, n, h2, h1, dens)
        Ps.append(P)
        Ns.append(N)
        As.append(np.full(len(P), a))
    if top:
        P, N, a, _, _ = _face_points(rng, c + ay * hy, ax, az, ay, hx, hz, dens)
        Ps.append(P)
        Ns.append(N)
        As.append(np.full(len(P), a))
    return np.concatenate(Ps), np.concatenate(Ns), np.concatenate(As)


def _build_near():
    """Surfaces of the buildings around the girl's roof (walls with window recesses, roof decks,
    parapets, stair enclosures + water tanks, solar heaters, antennas), and her own building's body.
    Also the clutter boxes for the z-buffer."""
    L = layout()
    rng = np.random.default_rng(21)
    x, z, a, b, yaw = [L["b_" + k].astype(np.float64) for k in ("x", "z", "a", "b", "yaw")]
    fl, fh, typ, tag = L["b_fl"], L["b_fh"].astype(np.float64), L["b_typ"], L["b_tag"]
    d = np.hypot(x - ROOF_ORIGIN[0], z - ROOF_ORIGIN[2])
    sel = np.nonzero(d < NEAR_R)[0]
    Ps, Ns, As, Cs, Ks = [], [], [], [], []
    boxes = []
    up = np.array([0.0, 1.0, 0.0])

    def add(P, N, A, col, kind=0):
        Ps.append(P)
        Ns.append(N)
        As.append(np.broadcast_to(A, (len(P),)))
        Cs.append(np.broadcast_to(np.asarray(col, np.float64), (len(P), 3)) if np.ndim(col) == 1 else col)
        Ks.append(np.full(len(P), kind, np.int8))

    for j in sel:
        c2 = np.array([x[j], z[j]])
        ux = np.array([np.cos(yaw[j]), 0.0, -np.sin(yaw[j])])
        uz = np.array([np.sin(yaw[j]), 0.0, np.cos(yaw[j])])       # local +z (south face normal)
        H = float(fl[j] * fh[j])
        dd = d[j]
        dens_w = 2.2 if dd < 160 else 1.1
        dens_r = 5.0 if dd < 160 else 2.2
        girl = tag[j] == 1
        if girl:
            dens_w = 9.0
        cc = np.array([c2[0], H * 0.5, c2[1]])
        # walls with a window grid (dark glass recesses) and floor bands
        for n, e1, hn, h1 in ((uz, ux, b[j], a[j]), (-uz, ux, b[j], a[j]), (ux, uz, a[j], b[j]), (-ux, uz, a[j], b[j])):
            P, N, A, u, v = _face_points(rng, cc + n * hn, e1, up, n, h1, H * 0.5, dens_w)
            s_ = (u * h1)
            y_ = (v + 1) * H * 0.5
            wx = np.abs(((s_ / 3.6) % 1.0) - 0.5) < 0.22
            wy = ((y_ / fh[j]) % 1.0 > 0.35) & ((y_ / fh[j]) % 1.0 < 0.85) & (y_ > 1.0)
            win = wx & wy & (hn > 3.0 if typ[j] == T_OLD else True)
            col = np.where(win[:, None], WALL_N * 0.35, WALL_N * (0.85 + 0.3 * rng.random((len(P), 1))))
            add(P, N, A, col, 0)
        if girl:
            continue
        # roof deck + parapet
        top = np.array([c2[0], H, c2[1]])
        P, N, A, u, v = _face_points(rng, top, ux, uz, up, a[j], b[j], dens_r)
        col = CONCRETE_N * (0.75 + 0.5 * rng.random((len(P), 1)))
        add(P, N, A, col, 1)
        if typ[j] in (T_OLD, T_MIX):
            for n, e1, hn, h1 in ((uz, ux, b[j], a[j]), (-uz, ux, b[j], a[j]), (ux, uz, a[j], b[j]), (-ux, uz, a[j], b[j])):
                P, N, A, _, _ = _face_points(rng, top + n * hn + up * 0.5, e1, up, n, h1, 0.5, dens_r)
                add(P, N, A, CONCRETE_N * 0.9, 1)
                P, N, A, _, _ = _face_points(rng, top + n * (hn - 0.12) + up * 1.0, e1, n, up, h1, 0.12, dens_r * 2)
                add(P, N, A, CONCRETE_N, 1)
            boxes.append([c2[0], c2[1], ux[0], ux[2], a[j], b[j], H, H + 1.0])
            # stair enclosures with water tanks: one per ~20 m of slab
            nst = max(1, int(round(2 * a[j] / 22.0)))
            for q in range(nst):
                off = (q - (nst - 1) * 0.5) * (2 * a[j] / nst) + rng.uniform(-2, 2)
                zoff = rng.uniform(-0.3, 0.3) * b[j]
                ec = top + ux * off + uz * zoff
                eh = rng.uniform(2.4, 2.9)
                P, N, A = _box_surface(rng, ec + up * eh * 0.5, ux, up, uz, 1.8, eh * 0.5, 1.4, dens_r)
                add(P, N, A, CONCRETE_N * 0.95, 2)
                boxes.append([ec[0], ec[2], ux[0], ux[2], 1.8, 1.4, H, H + eh])
                if rng.random() < 0.8:
                    th_ = rng.uniform(1.0, 1.4)
                    P, N, A = _box_surface(rng, ec + up * (eh + th_ * 0.5), ux, up, uz, 1.5, th_ * 0.5, 1.2, dens_r)
                    add(P, N, A, CONCRETE_N * 1.1, 2)
                    boxes.append([ec[0], ec[2], ux[0], ux[2], 1.5, 1.2, H + eh, H + eh + th_])
            # solar water heaters (tilted racks) and a TV antenna or two
            for q in range(rng.integers(1, 4)):
                sc = top + ux * rng.uniform(-a[j] + 3, a[j] - 3) + uz * rng.uniform(-b[j] + 2.5, b[j] - 2.5)
                P, N, A = _box_surface(rng, sc + up * 0.8, ux, up, uz, 1.1, 0.6, 0.6, dens_r)
                add(P, N, A, np.array([0.12, 0.13, 0.15]), 2)
            for q in range(rng.integers(0, 3)):
                mc = top + ux * rng.uniform(-a[j] + 1, a[j] - 1) + uz * rng.uniform(-b[j] + 1, b[j] - 1)
                mh = rng.uniform(3.0, 6.0)
                n = int(mh * 40)
                t_ = rng.random(n)
                P = mc[None] + up[None] * (t_ * mh)[:, None]
                add(P, np.tile(uz, (n, 1)), 0.004, np.array([0.18, 0.18, 0.2]), 3)
                for rr in range(3):
                    yy = mh * (0.7 + 0.1 * rr)
                    Lr = 0.9 - rr * 0.2
                    t_ = rng.uniform(-1, 1, 20)
                    P = mc[None] + up[None] * yy + ux[None] * (t_ * Lr)[:, None]
                    add(P, np.tile(uz, (20, 1)), 0.004, np.array([0.18, 0.18, 0.2]), 3)
        else:
            # towers: machine room on the roof
            P, N, A = _box_surface(rng, top + up * 2.2, ux, up, uz, min(5.0, a[j] * 0.4), 2.2, min(4.0, b[j] * 0.4), dens_r)
            add(P, N, A, CONCRETE_N * 0.9, 2)
            boxes.append([c2[0], c2[1], ux[0], ux[2], min(5.0, a[j] * 0.4), min(4.0, b[j] * 0.4), H, H + 4.4])
    P = np.concatenate(Ps).astype(np.float32)
    N = np.concatenate(Ns).astype(np.float32)
    return dict(P=P, N=N, area=np.concatenate(As).astype(np.float32), alb=np.concatenate(Cs).astype(np.float32),
                kind=np.concatenate(Ks), boxes=np.array(boxes, np.float64))


def near():
    if "near" not in _L:
        _L["near"] = _cached("near", _build_near)
    return _L["near"]


def roof_boxes():
    """The rooftop set's big occluders (stair enclosure + tank, parapets) and her building's body."""
    c, s = np.cos(ROOF_YAW), np.sin(ROOF_YAW)
    ux, uz = c, -s
    out = []
    for C, Bh in ((rooftop.ENCL_C, rooftop.ENCL_B), (rooftop.TANK_C, rooftop.TANK_B)):
        w = rooftop.to_world(C, ROOF_ORIGIN, ROOF_YAW)
        out.append([w[0], w[2], ux, uz, Bh[0], Bh[2], w[1] - Bh[1], w[1] + Bh[1]])
    for C, Bh in (((0, 0.5, 6.0), (12.0, 0.5, 0.12)), ((0, 0.5, -6.0), (12.0, 0.5, 0.12)),
                  ((12.0, 0.5, 0), (0.12, 0.5, 6.0)), ((-12.0, 0.5, 0), (0.12, 0.5, 6.0))):
        w = rooftop.to_world(np.array(C), ROOF_ORIGIN, ROOF_YAW)
        out.append([w[0], w[2], ux, uz, Bh[0], Bh[2], w[1] - Bh[1], w[1] + Bh[1]])
    out.append([ROOF_ORIGIN[0], ROOF_ORIGIN[2], ux, uz, ROOF_HALF[0], ROOF_HALF[1], 0.0, ROOF_ORIGIN[1] - 0.05])
    return np.array(out, np.float64)


_ROOF = {}


def roof_cloud():
    if "c" not in _ROOF:
        _ROOF["c"] = rooftop.cloud(ROOF_ORIGIN, ROOF_YAW)
    return _ROOF["c"]


BULB = hex_lin("#FFB870")        # a bare 40 W incandescent bulb


def env_lights(tg=None, pw=None):
    """solid.Light list for anything standing on/near the girl's roof at time tg (characters, props):
    the orange lid (sky glow, from above), city glow from the south horizon, the bare bulb, starlight
    after the blackout, and the substation flash."""
    from ..solid import Light
    pw = pw or power(tg)
    g = pw["glow"]
    lights = [
        Light("dir", hex_lin("#FFB27A"), 0.050 * g, vec=(0.1, 1.0, 0.25), wrap=0.9),     # the orange lid
        Light("dir", hex_lin("#FF9F5A"), 0.040 * g, vec=(0.15, 0.25, 1.0), wrap=0.6),    # city glow (south)
        Light("amb", hex_lin("#FFB080"), 0.008 * g),
        Light("dir", hex_lin("#9FB8FF"), 0.004 + 0.010 * pw["stars"], vec=(-0.2, 1.0, 0.3), wrap=0.8),  # starlight
        Light("amb", hex_lin("#6F86C8"), 0.0015),
    ]
    if pw["lamp"] > 0:
        lights.append(Light("point", BULB, 2.4 * pw["lamp"], vec=LAMP_WORLD, radius=0.25))
    if pw["flash"] > 0 or pw["burn"] > 0:
        sp = SUBSTATION_POS + np.array([0, 25.0, 0])
        lights.append(Light("point", hex_lin("#C8E4FF"), 9.0e4 * pw["flash"] + 1.5e3 * pw["burn"], vec=sp, radius=5.0))
    return lights


_GIRL = {}


def girl_cloud(**pose):
    """The girl (figures.Girl) posed and placed on the water tank, world frame. Returns SolidCloud."""
    from .figures import Girl
    if "g" not in _GIRL:
        _GIRL["g"] = Girl()
    key = tuple(sorted(pose.items()))
    if _GIRL.get("key") != key:
        cl, info = _GIRL["g"].pose(**pose)
        Rm = rooftop.yaw_matrix(ROOF_YAW)
        _GIRL["cl"] = cl.transformed(Rm, GIRL_WORLD)
        _GIRL["key"] = key
    return _GIRL["cl"]


def _solids(roof=True, near_=True):
    from ..solid import SolidCloud
    key = ("solids", roof, near_)
    if key not in _L:
        parts = []
        if roof:
            parts.append(roof_cloud())
        if near_:
            d = near()
            parts.append(SolidCloud(d["P"], d["N"], d["alb"], area=d["area"]))
        _L[key] = SolidCloud.concat(parts) if len(parts) > 1 else parts[0]
    return _L[key]


def draw_bulb(R, cam, zb, pw, energy=1.0):
    if pw["lamp"] <= 0:
        return
    P = LAMP_WORLD[None]
    x, y, z, coc, ok = cam.project(P)
    if not (ok[0] and -50 < x[0] < cam.W + 50 and -50 < y[0] < cam.H + 50):
        return
    lv = pw["lamp"]
    # filament afterglow is orange; a healthy bulb is warm white
    col = BULB * lv + hex_lin("#FF7A20") * (1 - lv) * 0.5 * (lv > 0)
    vis = bool(_ztest(zb, x, y, z, 0.02, 0.4)[0])
    if vis:
        R.draw(cam, P, col[None] * 9.0 * lv, size_px=1.6, energy=energy)
        R.draw(cam, P, col[None] * 1.2 * lv, size_px=9.0, soft=True, energy=energy)
    # the bulb's glow in the damp air (seen around the enclosure even when the bulb is hidden)
    R.draw(cam, P, col[None] * 0.9 * lv, size=1.6, soft=True, energy=energy)
    R.draw(cam, P, col[None] * 0.6 * lv, size=5.0, soft=True, energy=energy)


def render_env(R, cam, tg, W, H, sky=True, stars=True, city=True, roof=True, girl=False, pw=None,
               sky_energy=1.0, city_energy=1.0, star_energy=1.0, solid_energy=1.0, girl_pose=None,
               spacing_px=2.2):
    """Sky (glow dome / stars + Milky Way), the city and the rooftop set at global time tg.

    Returns (hdr, mask): mask = coverage of solid geometry (buildings, ground, rooftop set) so the
    caller can put sky-only elements behind it and composite characters on top. R is left empty.
    girl=True draws the girl (figures.Girl, pose kwargs in girl_pose) seated on the tank."""
    import cv2
    from ..solid import SolidCloud, draw_solid
    pw = pw or power(tg)
    R.new_layer()                                  # start clean
    parts = []
    if city:
        parts += [building_boxes(exclude_tag=(1,)), near()["boxes"]]
    if roof:
        parts.append(roof_boxes())
    boxes = np.concatenate(parts) if parts else np.zeros((0, 8))
    zb = zbuffer(cam, boxes, W, H) if len(boxes) else np.full((H, W), np.inf)
    cover = np.isfinite(zb).astype(np.float32)
    if city:
        cover = np.maximum(cover, ground_mask(cam, W, H))
    soft_cover = np.clip(cv2.GaussianBlur(cover, (0, 0), 0.5) * 1.15, 0, 1)
    if sky:
        draw_sky(R, cam, pw, energy=sky_energy)
    if stars:
        draw_stars(R, cam, pw, energy=star_energy)
    sky_img = R.new_layer() * (1.0 - soft_cover)[..., None]
    if city:
        draw_lamps(R, cam, zb, pw, energy=city_energy)
        draw_windows(R, cam, zb, pw, energy=city_energy)
        draw_traffic(R, cam, zb, pw, energy=city_energy)
        draw_reflections(R, cam, zb, pw, energy=city_energy)
    if roof or city:
        draw_bulb(R, cam, zb, pw, energy=city_energy)
    city_img = R.new_layer()
    hdr = sky_img + city_img
    mask = soft_cover
    if roof or city or girl:
        parts = []
        if roof or city:
            parts.append(_solids(roof, city))
        if girl:
            parts.append(girl_cloud(**(girl_pose or {})))
        cl = SolidCloud.concat(parts) if len(parts) > 1 else parts[0]
        lights = env_lights(pw=pw)
        m = draw_solid(R, cam, cl, lights, spacing_px=spacing_px, return_mask=True, energy=solid_energy,
                       rim=(hex_lin("#FFB27A") * 0.02 * pw["glow"] + hex_lin("#8FA8FF") * 0.01 * pw["stars"], 3.0, 1.0),
                       seurat=0.6, p_min=0.06, flecks=(hex_lin("#4A5A9A"), 0.05))
        solid_img = R.new_layer()
        hdr = hdr * (1.0 - m)[..., None] + solid_img
        mask = np.maximum(mask, m)
    return hdr, mask
