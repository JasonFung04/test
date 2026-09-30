"""THE DRAWER (BIBLE 5): the young woman of Blombos Cave, 73,000 years ago.

Assets: the right hand holding the ochre crayon (II5/II7 extreme close-ups), her head in profile
(II6) and her kneeling figure (II4).  Sculpted with the shared SDF tools (engine/sdf.py), sampled
once and cached; lit per frame by the fire through engine/solid.py.

Hand frame (right hand): origin at the wrist centre, +y distal (toward the fingers), +z dorsal
(back of the hand), +x ulnar (toward the little finger); the thumb is on -x.
"""
import numpy as np

from numba import njit, prange

from ..config import CACHE
from ..noise import fbm, hash01
from ..sdf import rot, sample_surface, sd_ellipsoid, sd_round_box, sd_round_cone, sd_sphere, smin
from ..solid import SolidCloud

SKIN = np.array([0.078, 0.042, 0.027], np.float32)        # dark skin (linear albedo)
SKIN_PALM = np.array([0.17, 0.095, 0.066], np.float32)    # lighter palmar skin
NAIL = np.array([0.20, 0.13, 0.105], np.float32)
CRAYON = np.array([0.36, 0.055, 0.025], np.float32)       # red ochre, a little darker than the drawn line
CRAYON_FACET = np.array([0.46, 0.08, 0.035], np.float32)

HAND_VERSION = 5


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


# ------------------------------------------------------------------------------------ sculpting
@njit(cache=True, inline="always")
def _smin(a, b, k):
    if k <= 0.0:
        return min(a, b)
    h = min(1.0, max(0.0, 0.5 + 0.5 * (b - a) / k))
    return b + (a - b) * h - k * h * (1.0 - h)


@njit(cache=True, inline="always")
def _prim(kind, q, x, y, z):
    if kind == 0:                                   # round cone a->b, r1, r2
        bax, bay, baz = q[3] - q[0], q[4] - q[1], q[5] - q[2]
        l2 = bax * bax + bay * bay + baz * baz
        rr = q[6] - q[7]
        a2 = l2 - rr * rr
        il2 = 1.0 / l2
        pax, pay, paz = x - q[0], y - q[1], z - q[2]
        yy = pax * bax + pay * bay + paz * baz
        zz = yy - l2
        wx, wy, wz = pax * l2 - bax * yy, pay * l2 - bay * yy, paz * l2 - baz * yy
        x2 = wx * wx + wy * wy + wz * wz
        y2 = yy * yy * l2
        z2 = zz * zz * l2
        sg = 1.0 if rr > 0 else (-1.0 if rr < 0 else 0.0)
        k = sg * rr * rr * x2
        sz = 1.0 if zz > 0 else (-1.0 if zz < 0 else 0.0)
        sy = 1.0 if yy > 0 else (-1.0 if yy < 0 else 0.0)
        if sz * a2 * z2 > k:
            return np.sqrt(x2 + z2) * il2 - q[7]
        if sy * a2 * y2 < k:
            return np.sqrt(x2 + y2) * il2 - q[6]
        return (np.sqrt(max(x2 * a2 * il2, 0.0)) + yy * rr) * il2 - q[6]
    if kind == 1:                                   # ellipsoid c, r, R (row-major, columns = axes)
        dx, dy, dz = x - q[0], y - q[1], z - q[2]
        lx = (dx * q[6] + dy * q[9] + dz * q[12]) / q[3]
        ly = (dx * q[7] + dy * q[10] + dz * q[13]) / q[4]
        lz = (dx * q[8] + dy * q[11] + dz * q[14]) / q[5]
        k0 = np.sqrt(lx * lx + ly * ly + lz * lz)
        k1 = np.sqrt((lx / q[3]) ** 2 + (ly / q[4]) ** 2 + (lz / q[5]) ** 2)
        return k0 * (k0 - 1.0) / max(k1, 1e-9)
    dx, dy, dz = x - q[0], y - q[1], z - q[2]       # sphere
    return np.sqrt(dx * dx + dy * dy + dz * dz) - q[3]


@njit(cache=True)
def _csg(x, y, z, kinds, prm, ks, subs):
    d = 1e9
    for j in range(kinds.shape[0]):
        di = _prim(kinds[j], prm[j], x, y, z)
        if j == 0:
            d = di
        elif subs[j]:
            d = -_smin(-d, di, ks[j]) if ks[j] > 0 else max(d, -di)
        else:
            d = _smin(d, di, ks[j])
    return d


@njit(cache=True, parallel=True)
def _csg_many(P, kinds, prm, ks, subs, out):
    for i in prange(P.shape[0]):
        out[i] = _csg(P[i, 0], P[i, 1], P[i, 2], kinds, prm, ks, subs)


@njit(cache=True, parallel=True)
def _project(P, kinds, prm, ks, subs, iters, D, G):
    e = 1e-5
    for i in prange(P.shape[0]):
        x, y, z = P[i, 0], P[i, 1], P[i, 2]
        for it in range(iters + 1):
            d = _csg(x, y, z, kinds, prm, ks, subs)
            gx = (_csg(x + e, y, z, kinds, prm, ks, subs) - _csg(x - e, y, z, kinds, prm, ks, subs)) / (2 * e)
            gy = (_csg(x, y + e, z, kinds, prm, ks, subs) - _csg(x, y - e, z, kinds, prm, ks, subs)) / (2 * e)
            gz = (_csg(x, y, z + e, kinds, prm, ks, subs) - _csg(x, y, z - e, kinds, prm, ks, subs)) / (2 * e)
            g2 = max(gx * gx + gy * gy + gz * gz, 1e-12)
            if it < iters:
                x -= d * gx / g2
                y -= d * gy / g2
                z -= d * gz / g2
            else:
                gn = np.sqrt(g2)
                D[i] = d
                G[i, 0] = gx / gn
                G[i, 1] = gy / gn
                G[i, 2] = gz / gn
        P[i, 0], P[i, 1], P[i, 2] = x, y, z


class Sculpt:
    """Ordered smooth CSG of primitives with a fast surface sampler.

    Seeds are drawn on the primitives themselves (area-weighted), Newton-projected onto the
    blended zero set and thinned on a voxel grid -> even density, no wasted candidates."""

    def __init__(self):
        self.ops = []

    def cone(self, a, b, r1, r2, k=0.0, sub=False):
        self.ops.append(("cone", (np.asarray(a, float), np.asarray(b, float), float(r1), float(r2)), k, sub))

    def ell(self, c, r, R=None, k=0.0, sub=False):
        self.ops.append(("ell", (np.asarray(c, float), np.asarray(r, float), None if R is None else np.asarray(R)),
                         k, sub))

    def sph(self, c, r, k=0.0, sub=False):
        self.ops.append(("sph", (np.asarray(c, float), float(r)), k, sub))

    @staticmethod
    def _eval(kind, prm, p):
        if kind == "cone":
            return sd_round_cone(p, *prm)
        if kind == "ell":
            return sd_ellipsoid(p, prm[0], prm[1], prm[2])
        return sd_sphere(p, prm[0], prm[1])

    def pack(self):
        kinds = np.zeros(len(self.ops), np.int64)
        prm = np.zeros((len(self.ops), 15))
        ks = np.zeros(len(self.ops))
        subs = np.zeros(len(self.ops), np.bool_)
        for j, (kind, q, k, sub) in enumerate(self.ops):
            if kind == "cone":
                kinds[j] = 0
                prm[j, :8] = np.concatenate([q[0], q[1], [q[2], q[3]]])
            elif kind == "ell":
                kinds[j] = 1
                R = np.eye(3) if q[2] is None else np.asarray(q[2], float)
                prm[j, :15] = np.concatenate([q[0], q[1], R.ravel()])
            else:
                kinds[j] = 2
                prm[j, :4] = np.concatenate([q[0], [q[1]]])
            ks[j] = k
            subs[j] = sub
        return kinds, prm, ks, subs

    def sdf_fast(self, p):
        out = np.empty(len(p))
        _csg_many(np.ascontiguousarray(p, np.float64), *self.pack(), out)
        return out

    def sdf(self, p):
        d = None
        for kind, prm, k, sub in self.ops:
            di = self._eval(kind, prm, p)
            if d is None:
                d = di
            elif sub:
                d = -smin(-d, di, k) if k > 0 else np.maximum(d, -di)
            else:
                d = smin(d, di, k) if k > 0 else np.minimum(d, di)
        return d

    def _seeds(self, n, rng):
        items, areas = [], []
        for kind, prm, k, sub in self.ops:
            if sub:
                continue
            if kind == "cone":
                a, b, r1, r2 = prm
                L = np.linalg.norm(b - a)
                areas.append(np.pi * (r1 + r2) * L + 2 * np.pi * (r1 * r1 + r2 * r2))
            elif kind == "ell":
                r = prm[1]
                areas.append(4 * np.pi * (((r[0] * r[1]) ** 1.6 + (r[0] * r[2]) ** 1.6 + (r[1] * r[2]) ** 1.6) / 3)
                             ** (1 / 1.6))
            else:
                areas.append(4 * np.pi * prm[1] ** 2)
            items.append((kind, prm))
        areas = np.array(areas)
        cnt = rng.multinomial(n, areas / areas.sum())
        out = []
        for (kind, prm), m in zip(items, cnt):
            if m == 0:
                continue
            u = rng.normal(size=(m, 3))
            u /= np.linalg.norm(u, axis=1, keepdims=True)
            if kind == "cone":
                a, b, r1, r2 = prm
                t = rng.random(m)[:, None]
                c = a + (b - a) * t
                r = r1 + (r2 - r1) * t
                out.append(c + u * r)
            elif kind == "ell":
                c, r, R = prm
                q = u * r
                out.append(c + (q @ R.T if R is not None else q))
            else:
                out.append(prm[0] + u * prm[1])
        return np.concatenate(out)

    def sample(self, spacing, seed=0, max_seeds=4_000_000):
        rng = np.random.default_rng(seed)
        P = self._seeds(max_seeds, rng)
        P = np.ascontiguousarray(P + rng.normal(0, spacing * 0.5, P.shape))
        D = np.empty(len(P))
        G = np.empty_like(P)
        _project(P, *self.pack(), 4, D, G)
        ok = np.abs(D) < spacing * 0.15
        P, G = P[ok], G[ok]
        # voxel thinning: one sample per cell -> even density
        key = np.floor(P / spacing).astype(np.int64)
        kk = (key[:, 0] * 73856093) ^ (key[:, 1] * 19349663) ^ (key[:, 2] * 83492791)
        order = rng.permutation(len(P))
        _, first = np.unique(kk[order], return_index=True)
        sel = order[first]
        return P[sel].astype(np.float32), G[sel].astype(np.float32), np.float32(spacing * spacing * 0.9)


def _ik2(root, end, l1, l2, bend):
    """Middle joint of a 2-link chain root->mid->end, bending toward `bend`."""
    root, end = np.asarray(root, float), np.asarray(end, float)
    v = end - root
    d = np.linalg.norm(v)
    d = min(d, (l1 + l2) * 0.999)
    v = v / (np.linalg.norm(v) + 1e-12)
    a = (l1 * l1 - l2 * l2 + d * d) / (2 * d)
    h = np.sqrt(max(l1 * l1 - a * a, 0.0))
    w = np.asarray(bend, float) - v * (np.asarray(bend, float) @ v)
    w /= np.linalg.norm(w) + 1e-12
    return root + v * a + w * h


def _unit(v):
    v = np.asarray(v, float)
    return v / np.linalg.norm(v)


# ------------------------------------------------------------------------------------ the hand
# Right hand in a tripod grip. Hand frame: wrist centre origin, +y distal, +z dorsal, +x ulnar.
MCP = {"index": np.array([-0.0230, 0.0880, -0.0010]), "middle": np.array([-0.0035, 0.0920, 0.0015]),
       "ring": np.array([0.0150, 0.0870, -0.0010]), "little": np.array([0.0310, 0.0790, -0.0055])}
CARPAL = {"index": np.array([-0.0120, 0.0200, -0.0010]), "middle": np.array([-0.0020, 0.0210, 0.0010]),
          "ring": np.array([0.0080, 0.0200, -0.0010]), "little": np.array([0.0170, 0.0180, -0.0040])}
PHAL = {"index": (0.0370, 0.0220, 0.0175), "middle": (0.0410, 0.0260, 0.0185),
        "ring": (0.0380, 0.0240, 0.0175), "little": (0.0300, 0.0190, 0.0160)}
RAD = {"index": (0.0092, 0.0084, 0.0076, 0.0068), "middle": (0.0094, 0.0087, 0.0079, 0.0070),
       "ring": (0.0088, 0.0081, 0.0074, 0.0066), "little": (0.0078, 0.0073, 0.0067, 0.0060)}
THUMB_CMC = np.array([-0.0215, 0.0300, -0.0135])
THUMB_L = (0.0470, 0.0320, 0.0260)
THUMB_R = (0.0125, 0.0112, 0.0100, 0.0086)
CRAYON_R = (0.0006, 0.0034, 0.0047)          # tip, shoulder of the ground point, back end
CRAYON_LEN = 0.040
FLEX = {"index": ((40.0, 48.0, 12.0), -5.0), "middle": ((46.0, 54.0, 16.0), 0.0),
        "ring": ((64.0, 84.0, 42.0), 6.0), "little": ((72.0, 90.0, 48.0), 12.0)}


def _fk(name):
    flex, splay = FLEX[name]
    Rm = rot("z", np.radians(-splay))
    pts = [MCP[name]]
    for k in range(3):
        Rm = Rm @ rot("x", -np.radians(flex[k]))
        pts.append(pts[-1] + Rm @ np.array([0.0, PHAL[name][k], 0.0]))
    return pts


def grip():
    """Chalk-like pinch: crayon between the thumb, index and middle pads (hand frame)."""
    ch = {n: _fk(n) for n in ("index", "middle", "ring", "little")}
    ip, mp = ch["index"], ch["middle"]
    di = _unit(ip[3] - ip[2])
    dm = _unit(mp[3] - mp[2])
    ax = -_unit(di + dm)                                     # crayon axis (tip -> back) ~ along the pads
    # pads (palmar side of the distal phalanges)
    pal_i = _unit(np.cross(di, [1.0, 0.0, 0.0]))
    pal_i = pal_i if pal_i[2] < 0 else -pal_i
    pad_i = ip[3] * 0.7 + ip[2] * 0.3 + pal_i * RAD["index"][3]
    pad_m = mp[3] * 0.7 + mp[2] * 0.3 + np.array([-1.0, 0.0, 0.0]) * RAD["middle"][3]
    K = (pad_i + pad_m) * 0.5
    # push the axis off both pads by the crayon radius (it sits in the groove between them)
    for _ in range(8):
        for pts, rad in ((ip, RAD["index"]), (mp, RAD["middle"])):
            a, b = pts[2], pts[3]
            ab = b - a
            t = np.clip(((K - a) @ ab) / (ab @ ab), 0, 1)
            c = a + t * ab
            dv = K - c
            dist = np.linalg.norm(dv)
            need = rad[3] + 0.0040
            if dist < need:
                K = c + dv / (dist + 1e-12) * need
    # make sure the crayon clears both pads by its radius
    tip = K - ax * 0.019
    n_d = _unit(np.array([0.2, 0.2, 1.0]) - ax * (ax @ _unit([0.2, 0.2, 1.0])))
    n_r = np.cross(ax, n_d)
    if n_r[0] > 0:
        n_r = -n_r
    # thumb: its pad presses the crayon from the radial side, opposite the middle finger
    ttip = K + ax * 0.001 + _unit(n_r * 0.85 - n_d * 0.35) * (THUMB_R[3] + 0.0038)
    # distribute the bend evenly over the MCP and IP joints (a gentle arc, not a hook)
    base = _unit(THUMB_CMC - ttip)
    best = None
    for side_w in (n_r, -n_d, n_d * 0.5 + n_r, -n_d * 0.5 + n_r):
        side = _unit(side_w - base * (side_w @ base))
        for a in np.radians(np.linspace(0.0, 70.0, 141)):
            tdir = base * np.cos(a) + side * np.sin(a)
            ip_ = ttip + tdir * THUMB_L[2]
            if np.linalg.norm(ip_ - THUMB_CMC) > (THUMB_L[0] + THUMB_L[1]) * 0.999:
                continue
            mcp_ = _ik2(THUMB_CMC, ip_, THUMB_L[0], THUMB_L[1], side)
            d0 = _unit(mcp_ - THUMB_CMC)
            d1 = _unit(ip_ - mcp_)
            d2 = _unit(ttip - ip_)
            f1 = np.arccos(np.clip(d0 @ d1, -1, 1))
            f2 = np.arccos(np.clip(d1 @ d2, -1, 1))
            err = abs(f1 - f2) + 0.3 * (f1 + f2)
            if best is None or err < best[0]:
                best = (err, ip_, mcp_)
    _, tip_ip, tmcp = best
    ch["thumb"] = [THUMB_CMC, tmcp, tip_ip, ttip]
    return dict(K=K, ax=ax, tip=tip, n_d=n_d, n_r=n_r, chains=ch)


def hand_sculpt():
    g = grip()
    S = Sculpt()
    # palm from four blended metacarpals -> natural transverse arch and knuckle ridges
    for name in ("index", "middle", "ring", "little"):
        S.cone(CARPAL[name], MCP[name], 0.0118 if name != "little" else 0.0108, RAD[name][0] * 1.08, k=0.014)
    S.ell((-0.0215, 0.040, -0.0125), (0.0165, 0.0270, 0.0125), rot("z", 0.40), k=0.012)   # thenar
    S.ell((0.0265, 0.045, -0.0105), (0.0115, 0.0300, 0.0100), k=0.012)                   # hypothenar
    S.cone((0.0020, -0.13, -0.0030), (0.0020, 0.018, -0.0010), 0.0265, 0.0235, k=0.016)  # wrist/forearm
    S.ell((0.0020, 0.004, -0.0020), (0.0270, 0.0160, 0.0165), k=0.012)                  # carpal mass
    for name in ("index", "middle", "ring", "little"):
        pts, rad = g["chains"][name], RAD[name]
        for k in range(3):
            S.cone(pts[k], pts[k + 1], rad[k], rad[k + 1], k=0.004)
        # pad bulge on the palmar side of the distal phalanx
        d = _unit(pts[3] - pts[2])
        S.sph(pts[3] - d * 0.004, rad[3] * 1.02, k=0.003)
    pts = g["chains"]["thumb"]
    for k in range(3):
        S.cone(pts[k], pts[k + 1], THUMB_R[k], THUMB_R[k + 1], k=0.009 if k == 0 else 0.004)
    S.sph(pts[3] - _unit(pts[3] - pts[2]) * 0.004, THUMB_R[3] * 1.05, k=0.003)
    return S, g


def crayon_sculpt():
    g = grip()
    S = Sculpt()
    tip, ax = g["tip"], g["ax"]
    S.cone(tip + ax * 0.0006, tip + ax * 0.013, CRAYON_R[0], CRAYON_R[1], k=0.0)
    S.cone(tip + ax * 0.013, tip + ax * CRAYON_LEN, CRAYON_R[1], CRAYON_R[2], k=0.002)
    return S, g


class Hand:
    """The drawing hand + crayon, sampled once (hand frame); `placement` + `clouds` pose it."""

    def __init__(self, spacing=0.00032):
        def make():
            S, g = hand_sculpt()
            P, N, a = S.sample(spacing, seed=51, max_seeds=3_000_000)
            keep = P[:, 1] > -0.095
            C, _ = crayon_sculpt()
            cP, cN, ca = C.sample(spacing * 0.8, seed=52, max_seeds=400_000)
            return dict(P=P[keep], N=N[keep], a=a, cP=cP, cN=cN, ca=ca)
        self.d = _cache(f"hand_v{HAND_VERSION}", make)
        g = grip()
        self.g = g
        self.tip, self.ax = g["tip"], g["ax"]
        P, N = self.d["P"].astype(np.float64), self.d["N"].astype(np.float64)
        alb = np.broadcast_to(SKIN, P.shape).copy()
        palm = np.clip((-N[:, 2] - 0.15) / 0.6, 0, 1) * np.clip((P[:, 1] + 0.01) / 0.03, 0, 1)
        alb = alb * (1 - palm[:, None]) + SKIN_PALM * palm[:, None]
        nail = np.zeros(len(P), bool)
        crease = np.zeros(len(P))
        for name in ("index", "middle", "ring", "little", "thumb"):
            pts = g["chains"][name]
            rad = RAD[name] if name != "thumb" else THUMB_R
            d = _unit(pts[3] - pts[2])
            L = np.linalg.norm(pts[3] - pts[2])
            # dorsal direction of this phalanx: away from the palm side (approx. from the finger's bend)
            bend = pts[1] - (pts[0] + pts[2]) * 0.5
            dors = _unit(bend - d * (bend @ d)) if np.linalg.norm(bend) > 1e-6 else np.array([0, 0, 1.0])
            if name == "thumb":
                dors = _unit(np.cross(d, [0.0, 0.0, 1.0]) * -1.0)
                dors = _unit(dors - d * (dors @ d))
            rel = P - pts[2]
            along = rel @ d
            perp = rel - along[:, None] * d
            up = perp @ dors
            side = np.linalg.norm(perp - up[:, None] * dors, axis=1)
            nail |= (along > L * 0.38) & (along < L * 1.08) & (up > rad[3] * 0.50) & (side < rad[3] * 0.72)
            for j in (1, 2):
                dj = _unit(pts[j + 1] - pts[j - 1])
                lj = (P - pts[j]) @ dj
                bj = pts[j] - (pts[j - 1] + pts[j + 1]) * 0.5
                dj2 = _unit(bj - dj * (bj @ dj)) if np.linalg.norm(bj) > 1e-7 else dors
                upj = (P - pts[j]) @ dj2
                band = np.exp(-(lj / 0.0016) ** 2) * (upj > 0.3 * rad[j])
                crease = np.maximum(crease, band * (0.55 + 0.45 * np.sin((P - pts[j]) @ np.cross(dj, dj2) * 2600.0)))
        alb[nail] = NAIL
        tex = fbm(P * 900.0, octaves=2)
        alb = alb * (0.86 + 0.28 * tex)[:, None] * (1.0 - 0.5 * crease)[:, None]
        self.alb = alb.astype(np.float32)
        self.nail = nail
        self.key = hash01(np.arange(len(P)), 61).astype(np.float32)
        cP = self.d["cP"]
        along = (cP - self.tip) @ self.ax
        facet = along < 0.013
        self.calb = np.where(facet[:, None], CRAYON_FACET, CRAYON).astype(np.float32)
        self.calb *= (0.75 + 0.5 * hash01(np.arange(len(cP)), 62)[:, None]).astype(np.float32)
        self.ckey = hash01(np.arange(len(cP)), 63).astype(np.float32)

    def placement(self, tip_world, axis_world, dorsal_hint):
        """Rm, T putting the crayon tip at tip_world, crayon axis along axis_world (tip -> back),
        rolled so the back of the hand faces dorsal_hint."""
        a_h = _unit(self.ax)
        d_h = self.g["n_d"] - a_h * (self.g["n_d"] @ a_h)
        d_h = _unit(d_h)
        B_h = np.stack([a_h, d_h, np.cross(a_h, d_h)], 1)
        a_w = _unit(axis_world)
        dh = np.asarray(dorsal_hint, float)
        d_w = _unit(dh - a_w * (dh @ a_w))
        B_w = np.stack([a_w, d_w, np.cross(a_w, d_w)], 1)
        Rm = B_w @ B_h.T
        T = np.asarray(tip_world, float) - Rm @ self.tip
        return Rm, T

    WRIST = np.array([0.002, 0.0, -0.002])

    def placement_wrist(self, tip_world, wrist_dir, psi):
        """Rm, T with the crayon tip at tip_world, the hand's tip->wrist direction along wrist_dir, and
        the crayon rolled by psi (rad) about that direction (psi=0: crayon leans toward +x x wrist)."""
        w_h = _unit(self.WRIST - self.tip)
        a_h = _unit(self.ax)
        ang = np.arccos(np.clip(w_h @ a_h, -1, 1))

        def frame(w, a):
            a2 = _unit(a - w * (a @ w))
            return np.stack([w, a2, np.cross(w, a2)], 1)
        w_w = _unit(wrist_dir)
        u1 = _unit(np.cross(w_w, [0.0, 1.0, 0.0]))
        u2 = np.cross(w_w, u1)
        a_w = np.cos(ang) * w_w + np.sin(ang) * (np.cos(psi) * u1 + np.sin(psi) * u2)
        Rm = frame(w_w, a_w) @ frame(w_h, a_h).T
        T = np.asarray(tip_world, float) - Rm @ self.tip
        return Rm, T

    def clouds(self, Rm, T):
        Rm32 = np.asarray(Rm, np.float32)
        T32 = np.asarray(T, np.float32)
        P = self.d["P"] @ Rm32.T + T32
        N = self.d["N"] @ Rm32.T
        hand = SolidCloud(P, N, self.alb, area=self.d["a"], key=self.key, part=np.zeros(len(P), np.int16))
        cP = self.d["cP"] @ Rm32.T + T32
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


# ------------------------------------------------------------------------------------ the figure
# Body frame: origin on the floor under her pelvis, +z = the way she faces, +y up, +x = her left.
LEATHER = np.array([0.16, 0.085, 0.045], np.float32)
CLOAK = np.array([0.13, 0.075, 0.045], np.float32)
BODY_VERSION = 2
HEAD_AT = np.array([0.0, 0.770, 0.170])       # where the head frame origin sits (upright head)
HEAD_PITCH = np.radians(34.0)                 # she looks down at the flake
NECK_BASE = np.array([0.0, 0.628, 0.118])


def body_sculpt():
    S = Sculpt()
    S.ell((0.0, 0.25, -0.02), (0.165, 0.110, 0.130))                                    # pelvis
    S.cone((0.0, 0.30, 0.0), (0.0, 0.55, 0.085), 0.125, 0.130, k=0.05)                   # torso
    S.ell((0.0, 0.49, 0.075), (0.150, 0.110, 0.095), k=0.05)                             # chest
    S.ell((0.0, 0.600, 0.095), (0.185, 0.050, 0.075), k=0.05)                            # shoulders
    S.cone((0.0, 0.595, 0.105), (0.0, 0.700, 0.150), 0.056, 0.050, k=0.03)               # neck
    for sx in (1.0, -1.0):
        S.cone((0.090 * sx, 0.25, 0.0), (0.100 * sx, 0.080, 0.360), 0.085, 0.056, k=0.04)     # thigh
        S.sph((0.100 * sx, 0.070, 0.370), 0.056, k=0.02)                                     # knee
        S.cone((0.100 * sx, 0.055, 0.370), (0.085 * sx, 0.048, -0.050), 0.050, 0.037, k=0.03)  # shin
        S.cone((0.085 * sx, 0.045, -0.050), (0.080 * sx, 0.030, -0.200), 0.036, 0.024, k=0.02)  # foot
    # right arm (her right = -x) reaching down to the flake, left hand resting on the left knee
    S.cone((-0.170, 0.595, 0.095), (-0.190, 0.405, 0.255), 0.046, 0.037, k=0.03)
    S.cone((-0.190, 0.405, 0.255), (-0.090, 0.150, 0.440), 0.036, 0.027, k=0.015)
    S.ell((-0.070, 0.090, 0.480), (0.030, 0.050, 0.020), _rx(-0.6), k=0.012)
    S.cone((0.170, 0.595, 0.095), (0.205, 0.400, 0.235), 0.046, 0.037, k=0.03)
    S.cone((0.205, 0.400, 0.235), (0.120, 0.190, 0.390), 0.036, 0.027, k=0.015)
    S.ell((0.105, 0.140, 0.430), (0.032, 0.048, 0.020), _rx(-0.9), k=0.012)
    return S


def wrap_sculpt():
    """Hide wrapped around the hips and thighs."""
    S = Sculpt()
    S.ell((0.0, 0.215, 0.07), (0.192, 0.108, 0.245))
    return S


def cloak_sculpt():
    """A hide cloak over the shoulders and back, open at the front."""
    S = Sculpt()
    S.ell((0.0, 0.47, -0.035), (0.205, 0.215, 0.150))
    S.ell((0.0, 0.600, 0.060), (0.215, 0.070, 0.105), k=0.04)
    S.ell((0.0, 0.56, 0.21), (0.30, 0.30, 0.10), k=0.02, sub=True)                       # open front
    return S


class Figure:
    """The kneeling drawer (II4 wide shot and the shoulders/necklace in II6)."""

    def __init__(self, spacing=0.0022):
        def make():
            out = {}
            P, N, a = body_sculpt().sample(spacing, seed=91, max_seeds=2_000_000)
            out["bP"], out["bN"], out["ba"] = P, N, a
            P, N, a = wrap_sculpt().sample(spacing, seed=92, max_seeds=400_000)
            out["wP"], out["wN"], out["wa"] = P, N, a
            P, N, a = cloak_sculpt().sample(spacing, seed=93, max_seeds=600_000)
            out["cP"], out["cN"], out["ca"] = P, N, a
            return out
        self.d = _cache(f"figure_v{BODY_VERSION}", make)
        d = self.d
        rng = np.random.default_rng(94)
        self.balb = (SKIN * (0.85 + 0.3 * rng.random(len(d["bP"])))[:, None]).astype(np.float32)
        wsd = wrap_sculpt().sdf_fast(d["bP"].astype(np.float64))
        csd = cloak_sculpt().sdf_fast(d["bP"].astype(np.float64))
        self.bkeep = (wsd > 0.003) & (csd > 0.003)                  # skin hidden under the hides
        self.walb = (LEATHER * (0.8 + 0.4 * fbm(d["wP"] * 30.0, octaves=3) + 0.2)[:, None]).astype(np.float32)
        self.calb = (CLOAK * (0.75 + 0.5 * rng.random(len(d["cP"])) ** 2)[:, None]).astype(np.float32)
        bsd = body_sculpt().sdf_fast(d["cP"].astype(np.float64))
        self.ckeep = bsd > 0.002

    @staticmethod
    def head_placement(pitch=HEAD_PITCH):
        """Rotation and translation of the head frame into the body frame."""
        Rh = rot("x", pitch)
        piv = HEAD_AT + NECK_PIVOT
        T = piv - Rh @ NECK_PIVOT
        return Rh, T

    def clouds(self, Rm, T):
        """Body, wrap, cloak as SolidClouds placed by (Rm, T)."""
        d = self.d
        R32 = np.asarray(Rm, np.float32)
        T32 = np.asarray(T, np.float32)
        out = []
        for P, N, a, alb, keep, seed in ((d["bP"], d["bN"], d["ba"], self.balb, self.bkeep, 95),
                                         (d["wP"], d["wN"], d["wa"], self.walb, None, 96),
                                         (d["cP"], d["cN"], d["ca"], self.calb, self.ckeep, 97)):
            if keep is not None:
                P, N, alb = P[keep], N[keep], alb[keep]
            out.append(SolidCloud(P @ R32.T + T32, N @ R32.T, alb, area=a,
                                  key=hash01(np.arange(len(P)), seed).astype(np.float32)))
        return out


def beads_world(Rm, T, n_per=260):
    """The necklace (body frame -> placed): bead points, normals, albedo, bead centres."""
    C, F, S = necklace()
    # necklace() is in the (upright) neck frame around y=-0.175; move it onto the body's neck base
    C = C - np.array([0.0, -0.175, -0.030]) + NECK_BASE + np.array([0.0, -0.012, 0.0])
    bp, bn, ba = bead_cloud(n_per)
    Ps, Ns, As = [], [], []
    for c, f, s in zip(C, F, S):
        Ps.append(c + (bp * s) @ f.T)
        Ns.append(bn @ f.T)
        As.append(ba)
    P = np.concatenate(Ps) @ np.asarray(Rm).T + T
    N = np.concatenate(Ns) @ np.asarray(Rm).T
    return P.astype(np.float32), N.astype(np.float32), np.concatenate(As).astype(np.float32), C @ np.asarray(Rm).T + T
