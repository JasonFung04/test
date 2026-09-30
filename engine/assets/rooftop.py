"""The girl's rooftop: an old 7-storey residential block (Chinese megacity, 1990s).

Local frame: origin at the centre of the roof deck, +y up (deck at y=0), footprint 24 m (x) by 12 m (z).
The city module places this frame in the world (ROOF_ORIGIN, ROOF_YAW live in assets/city.py).

Set dressing (authentic details): parapet, stair enclosure with a door and a bare bulb above it,
a concrete water tank on top of the enclosure (THE girl sits on its flat top), steel ladder,
solar water heaters (rows of vacuum tubes), a TV antenna, a clothesline with laundry, styrofoam
boxes with vegetables.
"""
import numpy as np

from ..config import CACHE
from ..noise import fbm
from ..sdf import sample_surface, sd_capsule, sd_round_box
from ..solid import SolidCloud

ENCL_C = np.array([-6.0, 1.3, -2.0])        # stair enclosure centre (3.6 x 2.6 x 2.8)
ENCL_B = np.array([1.8, 1.3, 1.4])
TANK_C = np.array([-6.0, 3.2, -2.0])        # water tank on top (3.0 x 1.2 x 2.4)
TANK_B = np.array([1.5, 0.6, 1.2])
TANK_TOP_Y = 3.8
GIRL_SEAT = np.array([-6.15, TANK_TOP_Y, -2.55])   # girl figure origin (she faces +z)
ELDER_SPOT = np.array([-6.15, TANK_TOP_Y, -1.30])  # elder stands here facing -z (toward her)
LAMP_POS = np.array([-6.0, 2.25, -0.50])            # bare bulb above the stair door (+z face)
DOOR_C = np.array([-6.0, 1.0, -0.59])

CONCRETE = np.array([0.30, 0.29, 0.27], np.float32)
TANK_ALB = np.array([0.36, 0.35, 0.32], np.float32)
METAL = np.array([0.20, 0.21, 0.23], np.float32)
GLASS = np.array([0.10, 0.12, 0.14], np.float32)
FOAM = np.array([0.55, 0.55, 0.52], np.float32)
LEAF = np.array([0.05, 0.12, 0.04], np.float32)
CLOTH = [np.array(c, np.float32) for c in ([0.45, 0.10, 0.08], [0.10, 0.18, 0.40], [0.55, 0.52, 0.45],
                                           [0.12, 0.30, 0.18], [0.60, 0.45, 0.10])]


def _box_cloud(c, b, r, n, seed, alb, noise=0.004, only_top=False):
    def f(p):
        d = sd_round_box(p, c, b, r)
        return d + noise * fbm(p * 6.0, octaves=3)
    lo = np.asarray(c) - np.asarray(b) - r - 0.05
    hi = np.asarray(c) + np.asarray(b) + r + 0.05
    P, N, a = sample_surface(f, lo, hi, n, seed=seed)
    if only_top:
        k = N[:, 1] > 0.5
        P, N = P[k], N[k]
    tint = 0.75 + 0.5 * (fbm(P * 1.7, octaves=3) * 0.5 + 0.5)[:, None]   # water stains, patches
    return P, N, np.float32(a), (alb * tint).astype(np.float32)


def build(density=1.0):
    path = CACHE / f"rooftop_{density}.npz"
    if path.exists():
        d = np.load(path)
        return {k: d[k] for k in d.files}
    rng = np.random.default_rng(3)
    Ps, Ns, As, Cs = [], [], [], []

    def add(P, N, a, C):
        Ps.append(P)
        Ns.append(N)
        As.append(np.full(len(P), a, np.float32))
        Cs.append(np.broadcast_to(C, P.shape).astype(np.float32))

    # roof deck (top face only) with a tiled texture
    n = int(260_000 * density)
    x = rng.uniform(-12, 12, n)
    z = rng.uniform(-6, 6, n)
    P = np.stack([x, np.zeros(n), z], 1)
    tile = ((np.abs((x % 0.5) - 0.25) < 0.012) | (np.abs((z % 0.5) - 0.25) < 0.012))
    C = CONCRETE * (0.8 + 0.4 * (fbm(P * 0.9, octaves=3) * 0.5 + 0.5))[:, None]
    C[tile] *= 0.55
    add(P.astype(np.float32), np.tile([0, 1.0, 0], (n, 1)).astype(np.float32), 288.0 / n, C)
    # parapet walls (1.0 m) around the edge
    for c, b in (((0, 0.5, 6.0), (12.0, 0.5, 0.12)), ((0, 0.5, -6.0), (12.0, 0.5, 0.12)),
                 ((12.0, 0.5, 0), (0.12, 0.5, 6.0)), ((-12.0, 0.5, 0), (0.12, 0.5, 6.0))):
        P, N, a, C = _box_cloud(c, b, 0.02, int(60_000 * density), rng.integers(1e6), CONCRETE)
        add(P, N, a, C)
    # stair enclosure + water tank
    P, N, a, C = _box_cloud(ENCL_C, ENCL_B, 0.04, int(120_000 * density), 11, CONCRETE)
    add(P, N, a, C)
    P, N, a, C = _box_cloud(TANK_C, TANK_B, 0.05, int(110_000 * density), 12, TANK_ALB)
    add(P, N, a, C)
    # door (dark recess) on the +z face of the enclosure
    n = int(8000 * density)
    P = np.stack([rng.uniform(-6.45, -5.55, n), rng.uniform(0.0, 2.0, n), np.full(n, -0.585)], 1)
    add(P.astype(np.float32), np.tile([0, 0, 1.0], (n, 1)).astype(np.float32), 1.8 / n, METAL * 0.6)
    # ladder (two rails + rungs) on the +x side of enclosure/tank
    for dz in (-0.25, 0.25):
        P, N, a = sample_surface(lambda p, dz=dz: sd_capsule(p, (-4.15, 0.0, -2.0 + dz), (-4.15, 3.9, -2.0 + dz), 0.02),
                                 (-4.3, -0.1, -2.4), (-4.0, 4.0, -1.6), int(4000 * density), seed=13)
        add(P, N, a, METAL)
    for k in range(12):
        y = 0.3 + k * 0.3
        P, N, a = sample_surface(lambda p, y=y: sd_capsule(p, (-4.15, y, -2.25), (-4.15, y, -1.75), 0.012),
                                 (-4.3, y - 0.05, -2.35), (-4.0, y + 0.05, -1.65), int(500 * density), seed=14 + k)
        add(P, N, a, METAL)
    # solar water heaters: 3 units, each a row of 14 tubes + a tank cylinder
    for u, (ux, uz) in enumerate(((3.0, -3.5), (6.5, -3.5), (3.0, 2.5))):
        for k in range(14):
            x0 = ux - 1.0 + k * 0.15
            P, N, a = sample_surface(lambda p, x0=x0, uz=uz: sd_capsule(p, (x0, 0.25, uz + 0.7), (x0, 1.35, uz - 0.4), 0.028),
                                     (x0 - 0.05, 0.15, uz - 0.5), (x0 + 0.05, 1.45, uz + 0.8), int(1500 * density),
                                     seed=100 + u * 20 + k)
            add(P, N, a, GLASS)
        P, N, a = sample_surface(lambda p, ux=ux, uz=uz: sd_capsule(p, (ux - 1.1, 1.45, uz - 0.5), (ux + 1.1, 1.45, uz - 0.5), 0.2),
                                 (ux - 1.4, 1.2, uz - 0.8), (ux + 1.4, 1.7, uz - 0.2), int(9000 * density), seed=150 + u)
        add(P, N, a, METAL * 1.6)
    # TV antenna mast
    P, N, a = sample_surface(lambda p: sd_capsule(p, (8.5, 0, 3.5), (8.5, 4.5, 3.5), 0.03), (8.3, -0.1, 3.3),
                             (8.7, 4.6, 3.7), int(3000 * density), seed=160)
    add(P, N, a, METAL)
    for k in range(5):
        y = 3.2 + k * 0.28
        L = 0.9 - k * 0.12
        P, N, a = sample_surface(lambda p, y=y, L=L: sd_capsule(p, (8.5 - L, y, 3.5), (8.5 + L, y, 3.5), 0.012),
                                 (8.5 - L - 0.1, y - 0.05, 3.4), (8.5 + L + 0.1, y + 0.05, 3.6), int(900 * density),
                                 seed=170 + k)
        add(P, N, a, METAL)
    # clothesline with laundry
    n = int(3000 * density)
    t = rng.random(n)
    P = np.stack([-1.0 + t * 8.0, 1.9 - 0.15 * np.sin(np.pi * t), np.full(n, 4.5)], 1)
    add(P.astype(np.float32), np.tile([0, 1.0, 0], (n, 1)).astype(np.float32), 0.0004, METAL * 0.5)
    for k in range(7):
        cx = -0.4 + k * 1.1 + rng.uniform(-0.2, 0.2)
        w, h = rng.uniform(0.35, 0.7), rng.uniform(0.4, 0.8)
        n = int(5000 * density)
        u, v = rng.random(n), rng.random(n)
        sag = 1.9 - 0.15 * np.sin(np.pi * (cx + 1.0) / 8.0)
        P = np.stack([cx + (u - 0.5) * w, sag - v * h, 4.5 + 0.03 * np.sin(u * 9 + k)], 1)
        add(P.astype(np.float32), np.tile([0, 0, 1.0], (n, 1)).astype(np.float32), w * h / n * 2, CLOTH[k % 5])
    # styrofoam planters with greens along the south parapet
    for k in range(6):
        cx = -10.5 + k * 1.3
        P, N, a, C = _box_cloud((cx, 0.18, 5.4), (0.5, 0.18, 0.3), 0.02, int(6000 * density), 200 + k, FOAM)
        add(P, N, a, C)
        n = int(4000 * density)
        P = np.stack([cx + rng.normal(0, 0.25, n), 0.36 + np.abs(rng.normal(0, 0.12, n)), 5.4 + rng.normal(0, 0.12, n)], 1)
        add(P.astype(np.float32), rng.normal(0, 1, (n, 3)).astype(np.float32) * 0.2 + [0, 1, 0], 0.3 / n, LEAF)
    Nn = np.concatenate(Ns)
    Nn /= np.linalg.norm(Nn, axis=1, keepdims=True) + 1e-9
    d = dict(P=np.concatenate(Ps).astype(np.float32), N=Nn.astype(np.float32),
             area=np.concatenate(As).astype(np.float32), alb=np.concatenate(Cs).astype(np.float32))
    np.savez(path, **d)
    return d


_R = {}


def cloud(origin=(0, 0, 0), yaw=0.0):
    """SolidCloud of the rooftop placed in the world."""
    if "d" not in _R:
        _R["d"] = build()
    d = _R["d"]
    c, s = np.cos(yaw), np.sin(yaw)
    Rm = np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]], np.float32)
    return SolidCloud(d["P"] @ Rm.T + np.asarray(origin, np.float32), d["N"] @ Rm.T, d["alb"], area=d["area"])


def to_world(p_local, origin=(0, 0, 0), yaw=0.0):
    c, s = np.cos(yaw), np.sin(yaw)
    Rm = np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])
    return np.asarray(p_local) @ Rm.T + np.asarray(origin)


def yaw_matrix(yaw):
    c, s = np.cos(yaw), np.sin(yaw)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])
