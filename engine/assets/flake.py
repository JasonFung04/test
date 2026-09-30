"""The Blombos silcrete flake and its ochre drawing (II4-II8).

Flake space (metres): origin at the canonical origin of the '#' ON the ground drawing face,
+y up (the face is the plane y = 0), canonical (x, y) -> (x * S_C, 0, -y * S_C), exactly as
crosshatch.to_world(plane='xz'). The flake rests on the cave floor at y = -THICK.

Strokes are deposited progressively along the canonical lines of assets/crosshatch.py, with the
start times of timeline.EVENTS (strokes_parallel / strokes_cross):
  six parallel lines : pulled toward the drawer (from the far end to the near end), ~0.45 s each
  three crossing     : left to right, ~0.6 s each
"""
import numpy as np

from .. import timeline as TL
from ..color import hex_lin
from ..config import CACHE
from ..noise import fbm, hash01
from ..solid import SolidCloud
from . import crosshatch as X

S_C = 0.030            # metres per canonical unit (the drawing is ~6.6 x 4.5 cm)
THICK = 0.0115         # flake thickness at the rim (m)
DUR_SIX = 0.45
DUR_THREE = 0.60
LINE_W = 0.052         # ochre line width, canonical units (~1.6 mm)
VERSION = 6

# outline of the flake, canonical units (the lower-right edge leaves the caption corner dark in II8)
OUTLINE = np.array([(-2.05, 0.10), (-1.80, 0.82), (-1.20, 1.22), (-0.35, 1.38), (0.55, 1.30), (1.35, 1.02),
                    (1.92, 0.52), (2.02, -0.02), (1.62, -0.33), (1.10, -0.44), (0.66, -0.80), (-0.05, -1.18),
                    (-0.95, -1.22), (-1.62, -0.85)])

STONE = hex_lin("#A89886")      # grey-beige silcrete ...
STONE_RED = hex_lin("#B08070")  # ... with a reddish tint
OCHRE = hex_lin("#B5391E")
OCHRE_DK = hex_lin("#7A2412")
SAND = hex_lin("#8A6E55")
SHELLC = hex_lin("#D8CCBA")

_LINES = X.lines(n=400)


def _chaikin(P, it=3):
    for _ in range(it):
        Q = np.roll(P, -1, axis=0)
        P = np.concatenate([0.75 * P + 0.25 * Q, 0.25 * P + 0.75 * Q], axis=1).reshape(-1, 2)
    return P


def outline(n_iter=4):
    """Smoothed outline with fine irregular chipping (canonical units, closed polygon)."""
    P = _chaikin(OUTLINE.astype(float), n_iter)
    c = P.mean(axis=0)
    d = P - c
    ang = np.arctan2(d[:, 1], d[:, 0])
    k = np.arange(len(P))
    chip = 0.035 * np.sin(ang * 23 + 1.3) + 0.02 * np.sin(ang * 57 + 0.4) + 0.012 * (hash01(k, 3) - 0.5)
    r = np.linalg.norm(d, axis=1, keepdims=True)
    return c + d * (1 + chip[:, None] / np.maximum(r, 1e-6))


def _inside(poly, pts):
    """Even-odd point-in-polygon, vectorised."""
    x, y = pts[:, 0], pts[:, 1]
    inside = np.zeros(len(pts), bool)
    n = len(poly)
    for i in range(n):
        x1, y1 = poly[i]
        x2, y2 = poly[(i + 1) % n]
        cond = ((y1 > y) != (y2 > y)) & (x < (x2 - x1) * (y - y1) / (y2 - y1 + 1e-12) + x1)
        inside ^= cond
    return inside


def _edge_dist(poly, pts):
    """Distance from points to the polygon boundary."""
    best = np.full(len(pts), np.inf)
    n = len(poly)
    for i in range(n):
        a, b = poly[i], poly[(i + 1) % n]
        ab = b - a
        t = np.clip(((pts - a) @ ab) / (ab @ ab), 0, 1)
        d = np.linalg.norm(pts - (a + t[:, None] * ab), axis=1)
        best = np.minimum(best, d)
    return best


def _line_attrs(q, lines, idxs):
    """For points q (N,2): nearest line among `idxs` -> (index, param u along the polyline, distance)."""
    best_d = np.full(len(q), np.inf)
    best_i = np.full(len(q), -1, np.int8)
    best_u = np.zeros(len(q))
    for k in idxs:
        L = lines[k]
        seg = L[1:] - L[:-1]
        cum = np.concatenate([[0], np.cumsum(np.linalg.norm(seg, axis=1))])
        tot = cum[-1]
        # coarse candidate prune
        lo, hi = L.min(axis=0) - 0.15, L.max(axis=0) + 0.15
        cand = np.nonzero((q[:, 0] > lo[0]) & (q[:, 0] < hi[0]) & (q[:, 1] > lo[1]) & (q[:, 1] < hi[1]))[0]
        if cand.size == 0:
            continue
        qq = q[cand]
        dmin = np.full(len(cand), np.inf)
        umin = np.zeros(len(cand))
        for j in range(len(seg)):
            a = L[j]
            ab = seg[j]
            t = np.clip(((qq - a) @ ab) / (ab @ ab), 0, 1)
            d = np.linalg.norm(qq - (a + t[:, None] * ab), axis=1)
            m = d < dmin
            dmin[m] = d[m]
            umin[m] = (cum[j] + t[m] * np.linalg.norm(ab)) / tot
        upd = dmin < best_d[cand]
        ci = cand[upd]
        best_d[ci] = dmin[upd]
        best_i[ci] = k
        best_u[ci] = umin[upd]
    return best_i, best_u, best_d


def stroke_param(k, u):
    """Position along the stroke (0 = touch-down, 1 = lift) for polyline parameter u of line k."""
    return 1.0 - u if k < 6 else u


def stroke_times():
    t6 = TL.EVENTS["strokes_parallel"]
    t3 = TL.EVENTS["strokes_cross"]
    return [(t, DUR_SIX) for t in t6] + [(t, DUR_THREE) for t in t3]


def _ease_stroke(x):
    x = np.clip(x, 0.0, 1.0)
    return x * x * (3 - 2 * x) * 0.35 + x * 0.65          # confident, slightly eased pull


def progress(tg):
    """Array (9,) of stroke progress in [0,1] at global time tg."""
    return np.array([_ease_stroke((tg - t0) / d) for t0, d in stroke_times()])


def _at(k, p):
    """Canonical point of line k at stroke position p (0 = touch-down, 1 = lift-off)."""
    L = _LINES[k]
    u = 1.0 - p if k < 6 else p
    f = u * (len(L) - 1)
    i = int(min(f, len(L) - 2))
    a = f - i
    return L[i] * (1 - a) + L[i + 1] * a


def tip(tg):
    """Where the crayon tip is: (line index or -1, canonical xy, lift 0..1 above the face).

    Within a group the tip hops (lifted) from the end of one stroke to the start of the next;
    across long gaps (before the first stroke, between the two groups, after the last) it lifts,
    rests, and comes down again onto the next start."""
    st = stroke_times()
    for k, (t0, d) in enumerate(st):
        if t0 <= tg <= t0 + d:
            return k, _at(k, float(_ease_stroke((tg - t0) / d))), 0.0
    prev = [k for k, (t0, d) in enumerate(st) if t0 + d < tg]
    nxt = [k for k, (t0, d) in enumerate(st) if t0 > tg]

    def sm(x):
        x = float(np.clip(x, 0, 1))
        return x * x * (3 - 2 * x)

    if prev and nxt and st[nxt[0]][0] - (st[prev[-1]][0] + st[prev[-1]][1]) < 1.2:
        kp, kn = prev[-1], nxt[0]
        tend = st[kp][0] + st[kp][1]
        x = (tg - tend) / (st[kn][0] - tend)
        s = sm(x)
        return -1, _at(kp, 1.0) * (1 - s) + _at(kn, 0.0) * s, float(np.sin(np.pi * np.clip(x, 0, 1)) ** 0.7)
    if nxt and st[nxt[0]][0] - tg < 0.6:
        kn = nxt[0]
        return -1, _at(kn, 0.0), 1.0 - sm((tg - (st[kn][0] - 0.6)) / 0.6)
    if prev:
        kp = prev[-1]
        tend = st[kp][0] + st[kp][1]
        return -1, _at(kp, 1.0), sm((tg - tend) / 0.5)
    return -1, _at(nxt[0], 0.0), 1.0


class Flake:
    def __init__(self, density=1.0):
        path = CACHE / f"flake_v{VERSION}_{density}.npz"
        if path.exists():
            d = np.load(path)
            self.d = {k: d[k] for k in d.files}
        else:
            self.d = self._build(density)
            np.savez(path, **self.d)
        d = self.d
        self.n_top = int(d["n_top"])

    def _build(self, density):
        rng = np.random.default_rng(41)
        poly = outline()
        lo, hi = poly.min(axis=0), poly.max(axis=0)
        # ---- ground drawing face: uniform samples inside the outline
        n = int(900_000 * density)
        q = lo + rng.random((n, 2)) * (hi - lo)
        q = q[_inside(poly, q)]
        area_c = (hi - lo).prod() * len(q) / n                 # canonical units^2
        ed = _edge_dist(poly, q)
        # bevel: the face rolls off toward the rim over ~1.2 mm
        bev = np.clip(1.0 - ed / 0.045, 0, 1) ** 2
        P = np.stack([q[:, 0] * S_C, -bev * 0.0012, -q[:, 1] * S_C], 1)
        # micro-relief normals (ground silcrete: fine grain + grinding striations along ~35 deg)
        g1 = fbm(np.stack([q[:, 0] * 60, q[:, 1] * 60, np.zeros(len(q))], 1), octaves=3)
        sd = q @ np.array([np.cos(0.6), np.sin(0.6)])
        striae = np.sin(sd * 900.0 + 3.0 * fbm(np.stack([q[:, 0] * 8, q[:, 1] * 8, np.ones(len(q))], 1), octaves=2))
        nx = 0.10 * g1 + 0.03 * striae * np.cos(0.6)
        nz = -0.10 * fbm(np.stack([q[:, 0] * 60, q[:, 1] * 60, np.full(len(q), 3.0)], 1), octaves=3) \
            - 0.03 * striae * np.sin(0.6)
        # bevel tilts the normal outward
        c = q.mean(axis=0)
        outw = (q - c) / (np.linalg.norm(q - c, axis=1, keepdims=True) + 1e-9)
        nx += bev * outw[:, 0] * 0.8
        nz += -bev * outw[:, 1] * 0.8
        N = np.stack([nx, np.ones(len(q)), nz], 1)
        N /= np.linalg.norm(N, axis=1, keepdims=True)
        # albedo: mottled grey-beige with a reddish tint, dark specks, faint iron staining
        m1 = fbm(np.stack([q[:, 0] * 2.2, q[:, 1] * 2.2, np.full(len(q), 7.0)], 1), octaves=4)
        m2 = fbm(np.stack([q[:, 0] * 14, q[:, 1] * 14, np.full(len(q), 1.0)], 1), octaves=2)
        red = np.clip(0.35 + 1.1 * m1, 0, 1)[:, None] * 0.6
        alb = STONE * (1 - red) + STONE_RED * red
        alb = alb * (0.80 + 0.35 * m2 + 0.12 * rng.standard_normal(len(q)))[:, None]
        speck = rng.random(len(q)) < 0.012
        alb[speck] *= 0.35
        alb = np.clip(alb, 0.02, 1.0) * 0.50
        # line attributes for the drawing: nearest SIX and nearest THREE
        i6, u6, d6 = _line_attrs(q, _LINES, range(6))
        i3, u3, d3 = _line_attrs(q, _LINES, range(6, 9))
        grain = fbm(np.stack([q[:, 0] * 90, q[:, 1] * 90, np.full(len(q), 5.0)], 1), octaves=2)
        top = dict(P=P, N=N, alb=alb, q=q, i6=i6, u6=u6, d6=d6, i3=i3, u3=u3, d3=d3, grain=grain)
        # ---- sides: fracture surfaces sloping from the rim down to the floor (conchoidal scallops)
        poly2 = outline(5)
        seg = np.roll(poly2, -1, axis=0) - poly2
        L = np.linalg.norm(seg, axis=1)
        cum = np.concatenate([[0], np.cumsum(L)])
        m = int(160_000 * density)
        s = rng.random(m) * cum[-1]
        j = np.searchsorted(cum, s) - 1
        j = np.clip(j, 0, len(seg) - 1)
        a = (s - cum[j]) / L[j]
        e = poly2[j] + seg[j] * a[:, None]
        tang = seg[j] / L[j][:, None]
        out = np.stack([tang[:, 1], -tang[:, 0]], 1)
        if (_inside(poly2, poly2.mean(axis=0, keepdims=True) + out[:1] * 0.01)).any():
            out = -out
        h = rng.random(m)                                      # 0 = rim, 1 = floor
        h = rng.random(m)                                      # 0 = rim, 1 = floor
        scal = 0.5 + 0.5 * np.sin(h * 14.0 + 1.5 * np.sin(s * 5.0))    # ripples parallel to the rim
        spread = (0.16 + 0.08 * scal) * h ** 0.9               # slope outward (canonical units)
        qe = e + out * spread[:, None]
        Ps = np.stack([qe[:, 0] * S_C, -0.0012 - h * (THICK - 0.0012), -qe[:, 1] * S_C], 1)
        slope = np.arctan2(THICK, (0.20 * S_C))
        ox, oz = out[:, 0], -out[:, 1]
        Ns = np.stack([ox * np.sin(slope), np.full(m, np.cos(slope)) * 0.6, oz * np.sin(slope)], 1)
        Ns += rng.normal(0, 0.04, Ns.shape)
        Ns[:, 1] += (scal - 0.5) * 0.35
        Ns /= np.linalg.norm(Ns, axis=1, keepdims=True)
        m3 = fbm(Ps * 180.0, octaves=3)
        alb_s = (STONE * 0.8 + STONE_RED * 0.2) * (0.62 + 0.3 * m3)[:, None] * 0.38
        # ---- the floor around: sand with crushed shell (Blombos midden), radius 0.55 m, denser near
        k = int(900_000 * density)
        rr = 0.55 * np.sqrt(rng.random(k))
        th = rng.random(k) * 2 * np.pi
        dens = 1.0 / (1.0 + (rr / 0.13) ** 2)
        kp = rng.random(k) < dens
        rr, th, dens = rr[kp], th[kp], dens[kp]
        k = len(rr)
        Pf = np.stack([rr * np.cos(th), np.full(k, -THICK), rr * np.sin(th)], 1)
        qf = np.stack([Pf[:, 0] / S_C, -Pf[:, 2] / S_C], 1)
        keepf = ~_inside(outline(3) * 1.01, qf)
        Pf = Pf[keepf]
        dens = dens[keepf]
        k = len(Pf)
        hgt = fbm(Pf * 70.0, octaves=3)
        Pf[:, 1] += hgt * 0.0025
        Nf = np.stack([-fbm(Pf * 70 + 3.3, octaves=2) * 0.15, np.ones(k), -fbm(Pf * 70 + 7.7, octaves=2) * 0.15], 1)
        Nf /= np.linalg.norm(Nf, axis=1, keepdims=True)
        alb_f = SAND * (0.75 + 0.3 * rng.random(k))[:, None] * 0.22
        shell = rng.random(k) < 0.02
        alb_f[shell] = SHELLC * (0.5 + 0.5 * rng.random(shell.sum()))[:, None] * 0.30
        side = dict(P=Ps, N=Ns, alb=alb_s)
        floor = dict(P=Pf, N=Nf, alb=alb_f)
        area_top = area_c * S_C * S_C / len(q)
        area_side = cum[-1] * S_C * (THICK / 0.8) / m * 1.6
        area_floor = (np.pi * 0.55 ** 2 / (900_000 * density)) / dens
        out = {}
        for name, dd in (("top", top), ("side", side), ("floor", floor)):
            for kk, v in dd.items():
                out[f"{name}_{kk}"] = np.asarray(v)
        out["n_top"] = np.int64(len(q))
        out["area_top"] = np.float32(area_top)
        out["area_side"] = np.float32(area_side)
        out["area_floor"] = np.float32(area_floor)
        # ochre powder crumbs next to each line (revealed as the stroke passes)
        crumbs = []
        for kline in range(9):
            Lk = _LINES[kline]
            nc = 700
            u = rng.random(nc)
            f = u * (len(Lk) - 1)
            i0 = np.minimum(f.astype(int), len(Lk) - 2)
            a = f - i0
            pt = Lk[i0] * (1 - a[:, None]) + Lk[i0 + 1] * a[:, None]
            off = rng.normal(0, LINE_W * 0.9, (nc, 2))
            crumbs.append(np.column_stack([pt + off, np.full(nc, kline), u, rng.random(nc)]))
        out["crumbs"] = np.concatenate(crumbs)
        return out

    # ------------------------------------------------------------------ per frame
    def ink(self, tg):
        """Per-top-point ochre coverage in [0,1] at global time tg."""
        d = self.d
        prog = progress(tg)
        cov = np.zeros(self.n_top)
        for fam in ("6", "3"):
            i = d[f"top_i{fam}"]
            u = d[f"top_u{fam}"]
            dist = d[f"top_d{fam}"]
            ok = i >= 0
            k = np.where(ok, i, 0)
            sp = np.where(k < 6, 1.0 - u, u)                   # position along the stroke
            drawn = ok & (sp <= prog[k]) & (prog[k] > 0)
            # pressure: thin touch-down, full body, tapering lift-off
            w = LINE_W * (0.55 + 0.45 * np.clip(sp / 0.12, 0, 1)) * (0.75 + 0.25 * np.clip((1 - sp) / 0.1, 0, 1))
            g = d["top_grain"]
            edge = dist / (0.5 * w * (0.75 + 0.9 * (g + 0.5)))
            c = np.clip(1.6 - edge * 1.2, 0, 1) * drawn
            # skips: the crayon misses a few low spots inside the stroke
            c *= np.where(g < -0.33, 0.35, 1.0)
            cov = np.maximum(cov, c)
        return cov

    def cloud(self, tg, lift_red=1.0):
        """SolidCloud of face + sides + floor, with the ochre drawn so far."""
        d = self.d
        cov = self.ink(tg)[:, None]
        alb = d["top_alb"]
        g = d["top_grain"][:, None]
        och = OCHRE * (0.85 + 0.5 * (g + 0.4)) * 0.95 + OCHRE_DK * 0.10
        alb_t = alb * (1 - cov) + och * cov
        top = SolidCloud(d["top_P"], d["top_N"], alb_t, area=d["area_top"], key=hash01(np.arange(self.n_top), 71))
        side = SolidCloud(d["side_P"], d["side_N"], d["side_alb"], area=d["area_side"],
                          key=hash01(np.arange(len(d["side_P"])), 72))
        flo = SolidCloud(d["floor_P"], d["floor_N"], d["floor_alb"], area=d["area_floor"],
                         key=hash01(np.arange(len(d["floor_P"])), 73))
        return top, side, flo, cov[:, 0]

    def crumbs(self, tg):
        """Loose ochre powder grains beside the drawn strokes: (P, albedo)."""
        c = self.d["crumbs"]
        prog = progress(tg)
        k = c[:, 2].astype(int)
        sp = np.where(k < 6, 1.0 - c[:, 3], c[:, 3])
        on = (sp <= prog[k]) & (prog[k] > 0) & (c[:, 4] < 0.55)
        q = c[on, :2]
        P = np.stack([q[:, 0] * S_C, np.full(len(q), 0.00015), -q[:, 1] * S_C], 1)
        return P.astype(np.float32), on.sum()


def to_flake(q):
    """Canonical (N,2) -> flake space (N,3) on the drawing face."""
    q = np.atleast_2d(q)
    return np.stack([q[:, 0] * S_C, np.zeros(len(q)), -q[:, 1] * S_C], 1)
