"""THE DRAWER (BIBLE 5): the young woman of Blombos Cave, 73,000 years ago.

Assets: the right hand holding the ochre crayon (II5/II7 extreme close-ups), her head in profile
(II6) and her kneeling figure (II4).  Sculpted with the shared SDF tools (engine/sdf.py), sampled
once and cached; lit per frame by the fire through engine/solid.py.

Hand frame (right hand): origin at the wrist centre, +y distal (toward the fingers), +z dorsal
(back of the hand), +x ulnar (toward the little finger); the thumb is on -x.
"""
import numpy as np

from ..config import CACHE
from ..noise import fbm, hash01
from ..sdf import rot, sample_surface, sd_ellipsoid, sd_round_box, sd_round_cone, sd_sphere, smin
from ..solid import SolidCloud

SKIN = np.array([0.105, 0.058, 0.036], np.float32)        # dark skin (linear albedo)
SKIN_PALM = np.array([0.23, 0.13, 0.09], np.float32)      # lighter palmar skin
NAIL = np.array([0.30, 0.19, 0.15], np.float32)
CRAYON = np.array([0.36, 0.055, 0.025], np.float32)       # red ochre, a little darker than the drawn line
CRAYON_FACET = np.array([0.46, 0.08, 0.035], np.float32)

HAND_VERSION = 2


def _cache(name, fn):
    path = CACHE / f"drawer_{name}.npz"
    if path.exists():
        d = np.load(path)
        return {k: d[k] for k in d.files}
    d = fn()
    np.savez(path, **d)
    return d


def _rx(a):
    return rot("x", a)


def _rz(a):
    return rot("z", a)


# ------------------------------------------------------------------------------------ the hand
# finger: MCP position, splay (deg about z), phalanx lengths, radii at joints, flexions (MCP, PIP, DIP deg)
FINGERS = {
    "index": ((-0.0225, 0.090, 0.000), -6.0, (0.037, 0.022, 0.0175), (0.0092, 0.0085, 0.0078, 0.0070),
              (38.0, 38.0, 4.0)),
    "middle": ((-0.0030, 0.094, 0.001), -1.0, (0.041, 0.026, 0.0185), (0.0094, 0.0088, 0.0080, 0.0072),
               (52.0, 70.0, 28.0)),
    "ring": ((0.0160, 0.089, -0.001), 5.0, (0.038, 0.024, 0.0175), (0.0088, 0.0082, 0.0075, 0.0068),
             (66.0, 88.0, 42.0)),
    "little": ((0.0320, 0.080, -0.004), 13.0, (0.030, 0.019, 0.0160), (0.0079, 0.0074, 0.0068, 0.0062),
               (72.0, 92.0, 48.0)),
}
THUMB_CMC = np.array([-0.019, 0.024, -0.011])


def finger_joints(name):
    """Joint positions (MCP, PIP, DIP, tip) and per-segment rotation matrices, hand frame."""
    base, splay, L, rad, flex = FINGERS[name]
    Rm = _rz(np.radians(-splay))
    pts = [np.asarray(base, float)]
    Rs = []
    for k in range(3):
        Rm = Rm @ _rx(-np.radians(flex[k]))
        Rs.append(Rm.copy())
        pts.append(pts[-1] + Rm @ np.array([0.0, L[k], 0.0]))
    return pts, Rs, rad


def thumb_joints():
    """Thumb: CMC -> MCP -> IP -> tip, opposed to reach the index pad (tripod grip)."""
    d0 = np.array([-0.52, 0.66, -0.54])
    d0 /= np.linalg.norm(d0)
    mcp = THUMB_CMC + d0 * 0.041
    # flexion plane: the thumb curls toward the index tip
    target = finger_joints("index")[0][3] + np.array([-0.004, -0.004, -0.017])
    d1 = target - mcp
    d1 /= np.linalg.norm(d1)
    d1 = 0.55 * d1 + 0.45 * d0
    d1 /= np.linalg.norm(d1)
    ip = mcp + d1 * 0.030
    d2 = target - ip
    d2 /= np.linalg.norm(d2)
    tip = ip + d2 * 0.0235
    return [THUMB_CMC, mcp, ip, tip], (0.0125, 0.0112, 0.0100, 0.0088)


def crayon_axis():
    """The crayon: pinched between the index pad and the thumb pad, resting on the middle finger.

    Returns (tip point, unit axis pointing from the tip back into the hand), hand frame."""
    ip, _, irad = finger_joints("index")
    tp, trad = thumb_joints()
    mp, _, mrad = finger_joints("middle")
    # contact points: under the index pad, inside the thumb pad, on the radial side of the middle DIP
    pad_i = ip[3] * 0.72 + ip[2] * 0.28 + np.array([0.0, 0.0, -0.0105])
    pad_t = tp[3] * 0.75 + tp[2] * 0.25 + np.array([0.0085, 0.002, 0.001])
    rest_m = mp[2] * 0.6 + mp[3] * 0.4 + np.array([-0.0105, 0.0, 0.0])
    grip = (pad_i + pad_t) * 0.5
    back = np.array([-0.010, 0.050, 0.030])                   # the crayon's end rests in the thumb web
    ax = back - grip
    ax /= np.linalg.norm(ax)
    # nudge so it also touches the middle finger's side
    tip = grip - ax * 0.019
    return tip, ax, grip, rest_m


def hand_sdf(p):
    d = sd_round_box(p, (0.006, 0.050, -0.001), (0.031, 0.040, 0.0075), 0.0085)          # palm
    d = smin(d, sd_ellipsoid(p, (-0.020, 0.036, -0.011), (0.017, 0.027, 0.013), _rz(0.35)), 0.010)  # thenar
    d = smin(d, sd_ellipsoid(p, (0.029, 0.040, -0.009), (0.012, 0.030, 0.010)), 0.010)  # hypothenar
    # wrist + forearm (slightly flattened)
    q = p * np.array([1.0, 1.0, 1.25])
    d = smin(d, sd_round_cone(q, (0.004, -0.20, -0.004), (0.002, 0.004, 0.0), 0.030, 0.026), 0.018)
    # knuckles
    for name in ("index", "middle", "ring", "little"):
        pts, Rs, rad = finger_joints(name)
        d = smin(d, sd_sphere(p, pts[0] + np.array([0, -0.002, 0.004]), rad[0] * 1.02), 0.008)
        for k in range(3):
            d = smin(d, sd_round_cone(p, pts[k], pts[k + 1], rad[k], rad[k + 1]), 0.0035)
        # finger pad: a soft bulge on the palmar side of the distal phalanx
        pad = pts[2] * 0.4 + pts[3] * 0.6 + Rs[2] @ np.array([0.0, 0.0, -0.0028])
        d = smin(d, sd_ellipsoid(p, pad, (rad[3] * 0.92, 0.0085, rad[3] * 0.75), Rs[2]), 0.003)
    tp, trad = thumb_joints()
    for k in range(3):
        d = smin(d, sd_round_cone(p, tp[k], tp[k + 1], trad[k], trad[k + 1]), 0.006 if k == 0 else 0.003)
    return d


def crayon_sdf(p):
    tip, ax, _, _ = crayon_axis()
    back = tip + ax * 0.040
    body = sd_round_cone(p, tip + ax * 0.014, back, 0.0032, 0.0046)
    point = sd_round_cone(p, tip + ax * 0.0006, tip + ax * 0.014, 0.0005, 0.0032)
    d = smin(body, point, 0.002)
    # ground facets: flatten slightly along two planes (a crayon rubbed to a point)
    n1 = np.cross(ax, [0.0, 0.0, 1.0])
    n1 /= np.linalg.norm(n1)
    for sgn in (1.0, -1.0):
        plane = (p - (tip + ax * 0.02)) @ (n1 * sgn) - 0.0036
        d = np.maximum(d, plane)
    return d


class Hand:
    """The drawing hand + crayon, sampled once. `cloud(M, t)` places it with a 4x4-like (Rm, T)."""

    def __init__(self, density=1.0):
        def make():
            out = {}
            P, N, a = sample_surface(hand_sdf, (-0.075, -0.20, -0.06), (0.075, 0.16, 0.06),
                                     int(900_000 * density), seed=51)
            keep = P[:, 1] > -0.11                              # the forearm leaves the frame anyway
            out["P"], out["N"], out["a"] = P[keep], N[keep], np.float32(a)
            P, N, a = sample_surface(crayon_sdf, (-0.06, -0.02, -0.06), (0.06, 0.12, 0.06),
                                     int(60_000 * density), seed=52, band=0.0004)
            out["cP"], out["cN"], out["ca"] = P, N, np.float32(a)
            return out
        self.d = _cache(f"hand_v{HAND_VERSION}_{density}", make)
        self.tip, self.ax, _, _ = crayon_axis()
        P, N = self.d["P"].astype(np.float64), self.d["N"].astype(np.float64)
        alb = np.broadcast_to(SKIN, P.shape).copy()
        # palmar skin is lighter (sides of the fingers, the thenar, the pads)
        palm = np.clip((-N[:, 2] - 0.15) / 0.6, 0, 1)
        alb = alb * (1 - palm[:, None]) + SKIN_PALM * palm[:, None]
        # nails on the dorsal distal phalanx of each finger and the thumb
        nail = np.zeros(len(P), bool)
        crease = np.zeros(len(P))
        for name in ("index", "middle", "ring", "little"):
            pts, Rs, rad = finger_joints(name)
            loc = (P - pts[2]) @ Rs[2]                          # distal phalanx frame
            L = np.linalg.norm(pts[3] - pts[2])
            nail |= (loc[:, 1] > L * 0.35) & (loc[:, 1] < L * 1.02) & (loc[:, 2] > rad[3] * 0.55) & \
                (np.abs(loc[:, 0]) < rad[3] * 0.72)
            for j in (1, 2):                                     # dorsal creases at PIP and DIP
                lj = (P - pts[j]) @ Rs[j - 1]
                band = np.exp(-(lj[:, 1] / 0.0018) ** 2) * (lj[:, 2] > 0)
                crease = np.maximum(crease, band * (0.5 + 0.5 * np.sin(lj[:, 0] * 900.0)))
        tp, trad = thumb_joints()
        dt = tp[3] - tp[2]
        Lt = np.linalg.norm(dt)
        dt /= Lt
        rel = P - tp[2]
        along = rel @ dt
        perp = rel - along[:, None] * dt
        nrm_nail = np.cross(dt, [0, 0, 1.0])
        nrm_nail = np.cross(nrm_nail, dt)
        nrm_nail /= np.linalg.norm(nrm_nail)
        nail |= (along > Lt * 0.35) & (along < Lt * 1.05) & ((perp @ nrm_nail) > trad[3] * 0.5)
        alb[nail] = NAIL
        # skin texture: fine mottling, darker creases
        tex = fbm(P * 900.0, octaves=2)
        alb = alb * (0.85 + 0.30 * tex)[:, None] * (1.0 - 0.45 * crease)[:, None]
        self.alb = alb.astype(np.float32)
        self.nail = nail
        self.key = hash01(np.arange(len(P)), 61).astype(np.float32)
        cP = self.d["cP"]
        tip, ax = self.tip, self.ax
        along = (cP - tip) @ ax
        facet = along < 0.02
        self.calb = np.where(facet[:, None], CRAYON_FACET, CRAYON).astype(np.float32)
        self.calb *= (0.8 + 0.4 * hash01(np.arange(len(cP)), 62)[:, None]).astype(np.float32)
        self.ckey = hash01(np.arange(len(cP)), 63).astype(np.float32)

    def placement(self, tip_world, axis_world, dorsal_hint):
        """Rotation Rm and translation T that put the crayon tip at tip_world with its axis along
        axis_world, rolling the hand so its back faces dorsal_hint."""
        a_h = self.ax / np.linalg.norm(self.ax)
        d_h = np.array([0.0, 0.0, 1.0]) - a_h * a_h[2]
        d_h /= np.linalg.norm(d_h)
        B_h = np.stack([a_h, d_h, np.cross(a_h, d_h)], 1)
        a_w = np.asarray(axis_world, float)
        a_w /= np.linalg.norm(a_w)
        d_w = np.asarray(dorsal_hint, float) - a_w * (np.asarray(dorsal_hint, float) @ a_w)
        d_w /= np.linalg.norm(d_w)
        B_w = np.stack([a_w, d_w, np.cross(a_w, d_w)], 1)
        Rm = B_w @ B_h.T
        T = np.asarray(tip_world, float) - Rm @ self.tip
        return Rm, T

    def clouds(self, Rm, T):
        Rm32 = np.asarray(Rm, np.float32)
        P = self.d["P"] @ Rm32.T + np.asarray(T, np.float32)
        N = self.d["N"] @ Rm32.T
        hand = SolidCloud(P, N, self.alb, area=self.d["a"], key=self.key, part=np.zeros(len(P), np.int16))
        cP = self.d["cP"] @ Rm32.T + np.asarray(T, np.float32)
        cN = self.d["cN"] @ Rm32.T
        cray = SolidCloud(cP, cN, self.calb, area=self.d["ca"], key=self.ckey, part=np.ones(len(cP), np.int16))
        return hand, cray


# ------------------------------------------------------------------------------------ the head
# Head frame: origin between the ear canals (tragion), +z forward (the face), +y up, +x = her left.
# Proportions: adult woman, head height 22.5 cm (vertex 0.12 .. menton -0.105), length 18.7 cm.
EYE_C = np.array([0.0315, 0.0120, 0.0680])          # left eyeball centre (mirror for the right)
EYE_R = 0.0120
HEAD_VERSION = 3
NECK_PIVOT = np.array([0.0, -0.035, -0.012])        # the head nods about this point
LID_DOWN = np.radians(38.0)                         # upper lid margin tilt: she is looking down


def _mirror(fn):
    """Evaluate a left-side feature on both sides (|x|)."""
    def f(p):
        q = p.copy()
        q[:, 0] = np.abs(q[:, 0])
        return fn(q)
    return f


def _eye_region(q):
    """Left-side eye socket carve, upper lid (lowered) and lower lid, on |x| coordinates."""
    c = EYE_C
    sock = sd_ellipsoid(q, c + np.array([0.0, 0.001, 0.011]), (0.0185, 0.0125, 0.0105))
    # upper lid: a shell over the eyeball, cut by a plane tilted forward/down (gaze down)
    lid = sd_sphere(q, c, EYE_R + 0.0016)
    n = np.array([0.0, np.cos(LID_DOWN), np.sin(LID_DOWN)])
    lid = np.maximum(lid, -((q - c) @ n) - 0.0035)
    low = sd_sphere(q, c, EYE_R + 0.0012)
    n2 = np.array([0.0, -np.cos(0.25), np.sin(0.25)])
    low = np.maximum(low, -((q - c) @ n2) - 0.0072)
    return sock, lid, low


def head_sdf(p):
    d = sd_ellipsoid(p, (0.0, 0.032, -0.006), (0.071, 0.089, 0.094))                  # cranium
    d = smin(d, sd_ellipsoid(p, (0.0, -0.028, 0.046), (0.056, 0.052, 0.050)), 0.020)  # midface
    q = p.copy()
    q[:, 0] = np.abs(q[:, 0])
    # mandible: gonion -> chin, cheekbones, brow
    d = smin(d, sd_round_cone(q, (0.050, -0.066, 0.008), (0.013, -0.094, 0.078), 0.013, 0.011), 0.022)
    d = smin(d, sd_ellipsoid(p, (0.0, -0.089, 0.074), (0.021, 0.016, 0.015)), 0.012)     # chin
    d = smin(d, sd_ellipsoid(q, (0.047, -0.008, 0.053), (0.020, 0.015, 0.021)), 0.014)  # cheekbone
    d = smin(d, sd_ellipsoid(p, (0.0, 0.027, 0.077), (0.047, 0.011, 0.015)), 0.012)     # brow ridge
    sock, lid, low = _eye_region(q)
    d = -smin(-d, sock, 0.006)                                                         # eye sockets
    d = smin(d, lid, 0.0025)
    d = smin(d, low, 0.0025)
    # nose: bridge, tip, broad alae; nostrils
    d = smin(d, sd_round_cone(p, (0.0, 0.017, 0.086), (0.0, -0.016, 0.0985), 0.0075, 0.0088), 0.006)
    d = smin(d, sd_sphere(p, (0.0, -0.0235, 0.0965), 0.0112), 0.006)
    d = smin(d, sd_sphere(q, (0.0172, -0.0305, 0.0855), 0.0102), 0.005)
    d = -smin(-d, sd_ellipsoid(q, (0.0095, -0.0368, 0.0905), (0.0042, 0.0030, 0.0058)), 0.0015)
    # full lips + mouth line
    d = smin(d, sd_ellipsoid(p, (0.0, -0.0485, 0.0875), (0.0255, 0.0088, 0.0112), rot("x", -0.25)), 0.005)
    d = smin(d, sd_ellipsoid(p, (0.0, -0.0635, 0.0858), (0.0228, 0.0098, 0.0118), rot("x", 0.20)), 0.005)
    d = -smin(-d, sd_ellipsoid(p, (0.0, -0.0558, 0.0945), (0.0235, 0.0011, 0.0080)), 0.0012)
    # ears (helix rim around a concha)
    ear = sd_ellipsoid(q, (0.0715, -0.004, -0.010), (0.0085, 0.030, 0.0185), rot("x", 0.18))
    ear = -smin(-ear, sd_ellipsoid(q, (0.0775, -0.002, -0.006), (0.0065, 0.019, 0.011), rot("x", 0.18)), 0.002)
    d = smin(d, ear, 0.004)
    # neck
    d = smin(d, sd_round_cone(p, (0.0, -0.055, -0.018), (0.0, -0.215, -0.035), 0.051, 0.058), 0.028)
    return d


def eyes_sdf(p):
    q = p.copy()
    q[:, 0] = np.abs(q[:, 0])
    ball = sd_sphere(q, EYE_C, EYE_R)
    cornea = sd_sphere(q, EYE_C + np.array([0.0, -0.0025, 0.0060]), 0.0079)       # gaze tilted down
    return smin(ball, cornea, 0.002)


def hair_region(P):
    """Scalp region covered by short hair: above a hairline, around (not over) the ears."""
    x, y, z = P[:, 0], P[:, 1], P[:, 2]
    ax = np.abs(x)
    front = y > 0.058 + 0.35 * np.clip(z - 0.035, 0, None) - 0.12 * np.clip(0.035 - z, 0, None) * (z > -0.02)
    temple = (z < 0.040) & (y > -0.004 - 0.4 * np.clip(0.015 - z, 0, None))
    around_ear = ~((np.abs(y + 0.004) < 0.036) & (z > -0.034) & (z < 0.016) & (ax > 0.055))
    nape = y > -0.050 + 0.25 * np.clip(z + 0.03, 0, None)
    return (front | temple) & around_ear & nape


class Head:
    """Head + eyes + coiled hair + lashes, sampled once (head frame)."""

    def __init__(self, density=1.0):
        def make():
            out = {}
            P, N, a = sample_surface(head_sdf, (-0.10, -0.23, -0.12), (0.10, 0.14, 0.125),
                                     int(700_000 * density), seed=81)
            out["P"], out["N"], out["a"] = P, N, np.float32(a)
            P, N, a = sample_surface(eyes_sdf, (-0.05, -0.005, 0.05), (0.05, 0.03, 0.085),
                                     int(30_000 * density), seed=82, band=0.0003)
            out["eP"], out["eN"], out["ea"] = P, N, np.float32(a)
            return out
        self.d = _cache(f"head_v{HEAD_VERSION}_{density}", make)
        P = self.d["P"].astype(np.float64)
        N = self.d["N"].astype(np.float64)
        rng = np.random.default_rng(83)
        # skin albedo: dark skin, lips slightly darker/redder, fine texture
        alb = np.broadcast_to(SKIN * 1.05, P.shape).copy()
        lips = (P[:, 2] > 0.074) & (P[:, 1] < -0.041) & (P[:, 1] > -0.074) & (np.abs(P[:, 0]) < 0.025)
        alb[lips] = np.array([0.085, 0.036, 0.030])
        alb *= (0.88 + 0.24 * fbm(P * 700.0, octaves=2))[:, None]
        hair = hair_region(P) & (N[:, 1] > -0.6)
        self.skin_P, self.skin_N = P[~hair], N[~hair]
        self.skin_alb = alb[~hair].astype(np.float32)
        # coiled hair: every scalp sample grows a tiny tight coil standing off the scalp
        base, bn = P[hair], N[hair]
        n = len(base)
        t1 = np.cross(bn, rng.normal(size=(n, 3)))
        t1 /= np.linalg.norm(t1, axis=1, keepdims=True)
        t2 = np.cross(bn, t1)
        lift = 0.0045 + 0.0060 * rng.random(n) ** 0.7              # hair thickness ~1 cm
        coils = []
        m = 5
        ph = rng.random(n) * 2 * np.pi
        rc = 0.0007 + 0.0005 * rng.random(n)
        for k in range(m):
            a = ph + k * 2 * np.pi / m * 1.3
            h = lift * (0.55 + 0.45 * k / (m - 1))
            coils.append(base + bn * h[:, None] + (t1 * np.cos(a)[:, None] + t2 * np.sin(a)[:, None]) * rc[:, None])
        self.hair_P = np.concatenate(coils).astype(np.float32)
        self.hair_N = np.concatenate([bn] * m).astype(np.float32)
        tone = 0.010 + 0.012 * rng.random(n * m)
        self.hair_alb = np.stack([tone * 1.1, tone * 0.95, tone * 0.85], 1).astype(np.float32)
        self.hair_area = np.float32(self.d["a"] * 0.8)
        # lashes: short strands along the lowered upper lid margin, pointing down/forward
        L = []
        for sx in (1.0, -1.0):
            u = rng.uniform(-1, 1, 260)
            ang = u * 0.95
            c = EYE_C * np.array([sx, 1, 1])
            nlid = np.array([0.0, np.cos(LID_DOWN), np.sin(LID_DOWN)])
            fwd = np.array([0.0, -np.sin(LID_DOWN), np.cos(LID_DOWN)])
            side = np.array([1.0, 0.0, 0.0]) * sx
            root = c + (EYE_R + 0.0016) * (np.cos(ang)[:, None] * fwd + np.sin(ang)[:, None] * side) - nlid * 0.0035
            ln = rng.uniform(0.004, 0.0085, 260) * (1 - 0.4 * np.abs(u))
            for s_ in np.linspace(0.0, 1.0, 6):
                L.append(root + (fwd * 0.9 - nlid * 0.35) * (ln * s_)[:, None] + nlid * (ln * s_ ** 2 * 0.25)[:, None])
        self.lash_P = np.concatenate(L).astype(np.float32)
        eP = self.d["eP"].astype(np.float64)
        q = eP.copy()
        q[:, 0] = np.abs(q[:, 0])
        iris = ((q - EYE_C) @ np.array([0.0, -0.21, 0.98])) > EYE_R * 0.80
        ealb = np.where(iris[:, None], np.array([0.035, 0.022, 0.016]), np.array([0.40, 0.35, 0.31]))
        self.eye_alb = ealb.astype(np.float32)

    def clouds(self, Rm, T):
        """Head parts in world space: skin, hair, eyes, lashes (SolidClouds)."""
        Rm = np.asarray(Rm)
        R32 = Rm.astype(np.float32)
        T32 = np.asarray(T, np.float32)

        def tf(P):
            return (np.asarray(P, np.float32) @ R32.T) + T32

        skin = SolidCloud(tf(self.skin_P), self.skin_N.astype(np.float32) @ R32.T, self.skin_alb,
                          area=self.d["a"], key=hash01(np.arange(len(self.skin_P)), 84).astype(np.float32))
        hair = SolidCloud(tf(self.hair_P), self.hair_N @ R32.T, self.hair_alb, area=self.hair_area,
                          key=hash01(np.arange(len(self.hair_P)), 85).astype(np.float32))
        eyes = SolidCloud(tf(self.d["eP"]), self.d["eN"] @ R32.T, self.eye_alb, area=self.d["ea"],
                          key=hash01(np.arange(len(self.d["eP"])), 86).astype(np.float32))
        lash = SolidCloud(tf(self.lash_P), np.tile(R32 @ np.array([0, 0, 1.0], np.float32), (len(self.lash_P), 1)),
                          np.array([0.012, 0.010, 0.009], np.float32), area=np.float32(1.5e-7),
                          key=hash01(np.arange(len(self.lash_P)), 87).astype(np.float32))
        return skin, hair, eyes, lash


def head_pose(pitch, yaw=0.0, roll=0.0):
    """Rotation of the head about the neck pivot (pitch > 0: looking down)."""
    return rot("y", yaw) @ rot("x", pitch) @ rot("z", roll)


def necklace(n_beads=21, seed=17):
    """~20 perforated Nassarius kraussianus shells on a sinew cord, around the base of the neck.

    Returns bead centres, orientation frames (3x3 each) and sizes in the head/neck frame (unposed)."""
    rng = np.random.default_rng(seed)
    # the cord: an ellipse around the neck base, dipping at the front
    t = np.linspace(-0.62, 0.62, n_beads) * np.pi + rng.normal(0, 0.03, n_beads)
    rx, rz = 0.066, 0.060
    cx = rx * np.sin(t)
    cz = -0.030 + rz * np.cos(t)
    cy = -0.175 - 0.030 * np.cos(t) ** 2
    C = np.stack([cx, cy, cz], 1)
    frames = []
    for k in range(n_beads):
        tang = np.array([np.cos(t[k]) * rx, 0.0, -np.sin(t[k]) * rz])
        tang /= np.linalg.norm(tang)
        out = np.array([np.sin(t[k]), 0.25, np.cos(t[k])])
        out -= tang * (out @ tang)
        out /= np.linalg.norm(out)
        b = np.cross(tang, out)
        tw = rng.normal(0, 0.35)
        c, s = np.cos(tw), np.sin(tw)
        frames.append(np.stack([tang * c + b * s, -tang * s + b * c, out], 1))
    size = 0.0050 + 0.0006 * rng.standard_normal(n_beads)
    return C, np.array(frames), size


def bead_cloud(n_per=900, seed=19):
    """One Nassarius shell (local frame: spire along +y, aperture facing -z, perforation on +z)."""
    rng = np.random.default_rng(seed)
    u = rng.normal(size=(n_per * 2, 3))
    u /= np.linalg.norm(u, axis=1, keepdims=True)
    P = u * np.array([0.75, 1.0, 0.70])
    P[:, 1] += 0.35 * np.clip(P[:, 1], 0, None) ** 2                   # pointed spire
    hole = (u[:, 2] > 0.80) & (np.abs(u[:, 1]) < 0.35)                 # the drilled hole
    ap = (u[:, 2] < -0.55) & (u[:, 1] < 0.2)                           # aperture (dark)
    keep = ~hole
    P, u, ap = P[keep][:n_per], u[keep][:n_per], ap[keep][:n_per]
    N = u / np.linalg.norm(u, axis=1, keepdims=True)
    alb = np.where(ap[:, None], np.array([0.10, 0.08, 0.06]), np.array([0.62, 0.56, 0.47]))
    band = 0.5 + 0.5 * np.sin(P[:, 1] * 18.0)
    alb = alb * (0.8 + 0.2 * band)[:, None]
    return P.astype(np.float32), N.astype(np.float32), alb.astype(np.float32)
