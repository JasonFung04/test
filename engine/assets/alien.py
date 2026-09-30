"""The senders' world at ground level (I2–I6): the sunflower city, its terraces, the crowd,
the lights that rise, and the red giant filling half the sky.

World frame for the ground scenes: y up, metres. City centre at the origin. The giant rises in
the +z direction ("south" of this world).
"""
import numpy as np

from .. import timeline as TL
from ..color import hex_lin
from ..config import CACHE
from ..noise import curl, fbm, hash01
from .bodies import RedGiant

GOLDEN = np.radians(137.50776)
RC = 12000.0            # city radius
RING_W = 240.0          # terrace ring width
STEP_H = 11.0           # terrace step height
GOLD = hex_lin("#FFC46B")
GOLD_HI = hex_lin("#FFE2A6")
EMBER = hex_lin("#FF8A3A")
E = TL.EVENTS


def terrace_h(r):
    """Stepped mound: rings rise toward the centre (a ziggurat of light)."""
    k = np.clip((RC - r) / RING_W, 0, None)
    frac = k - np.floor(k)
    step = np.floor(k) + np.clip((frac - 0.9) / 0.1, 0, 1)      # short ramps between terraces
    return step * STEP_H


def city(n_nodes=7000, seed=2):
    """Structured sunflower city: plazas on a Vogel spiral, streets along the 34 and 55
    parastichies (two families of crossing spirals -- an alien echo of the '#'), terrace rings,
    and sparse window lights in between."""
    path = CACHE / f"alien_city4_{n_nodes}_{seed}.npz"
    if path.exists():
        d = np.load(path)
        return {k: d[k] for k in d.files}
    rng = np.random.default_rng(seed)
    k = np.arange(n_nodes) + 0.5
    r = RC * np.sqrt(k / n_nodes)
    th = k * GOLDEN
    nodes = np.stack([r * np.cos(th), r * np.sin(th)], 1)
    Ps, Cs, kinds = [], [], []
    # plazas
    Ps.append(nodes)
    Cs.append(np.tile(GOLD_HI * 2.2, (n_nodes, 1)) * (0.5 + rng.random((n_nodes, 1))))
    kinds.append(np.full(n_nodes, 2))
    # streets: two opposite families of logarithmic spirals (sunflower parastichies), crossing ~80 deg
    r0 = 250.0
    for N, sgn, gain, psi in ((34, 1, 1.0, 40.0), (55, -1, 1.0, 40.0), (89, 1, 0.45, 40.0), (144, -1, 0.3, 40.0)):
        cot = 1.0 / np.tan(np.radians(psi))
        # arc length along a log spiral from r0 to r: (r - r0) / cos(psi)
        L = (RC - r0) / np.cos(np.radians(psi))
        m = int(L / 24.0)
        u = (np.arange(m) + 0.5) / m
        rr_ = r0 + (RC - r0) * u
        for j in range(N):
            th_ = 2 * np.pi * j / N + sgn * np.log(rr_ / r0) * cot
            pts = np.stack([rr_ * np.cos(th_), rr_ * np.sin(th_)], 1) + rng.normal(0, 1.5, (m, 2))
            Ps.append(pts)
            Cs.append(np.tile(GOLD * 0.9 * gain, (m, 1)) * (0.6 + 0.8 * rng.random((m, 1))))
            kinds.append(np.full(m, 1))
    # terrace rings (lips)
    ring_r = RC - (np.arange(1, int(RC / RING_W)) * RING_W) + RING_W * 0.05
    ring_r = ring_r[::3]
    per = (2 * np.pi * ring_r / 40.0).astype(int)
    rr = np.repeat(ring_r, per)
    tt = rng.random(rr.size) * 2 * np.pi
    Ps.append(np.stack([rr * np.cos(tt), rr * np.sin(tt)], 1))
    Cs.append(np.tile(GOLD * 0.45, (rr.size, 1)))
    kinds.append(np.full(rr.size, 3))
    # windows: sparse dim lights in the blocks
    nw = 180_000
    rw = RC * np.sqrt(rng.random(nw))
    tw = rng.random(nw) * 2 * np.pi
    Ps.append(np.stack([rw * np.cos(tw), rw * np.sin(tw)], 1))
    Cs.append(np.tile(GOLD * 0.35, (nw, 1)) * (rng.random((nw, 1)) ** 2 * 1.5 + 0.1))
    kinds.append(np.zeros(nw))
    # far field: sparse settlements (small sunflowers) out to the horizon
    nf = 90
    fr = RC * 1.3 + rng.random(nf) ** 0.7 * 60000.0
    ft = rng.random(nf) * 2 * np.pi
    for cx, cz, sz in zip(fr * np.cos(ft), fr * np.sin(ft), rng.uniform(400, 2200, nf)):
        m = int(sz * 1.2)
        kk = np.arange(m) + 0.5
        rr2 = sz * np.sqrt(kk / m)
        tt2 = kk * GOLDEN
        Ps.append(np.stack([cx + rr2 * np.cos(tt2), cz + rr2 * np.sin(tt2)], 1))
        Cs.append(np.tile(GOLD * 1.1, (m, 1)) * (0.4 + rng.random((m, 1))))
        kinds.append(np.full(m, 1))
    xz = np.concatenate(Ps)
    rad = np.hypot(xz[:, 0], xz[:, 1])
    y = np.where(rad < RC, terrace_h(rad), 0.0) + 1.5
    P = np.stack([xz[:, 0], y, xz[:, 1]], 1).astype(np.float32)
    key = rng.random(len(P)).astype(np.float32)
    col = np.concatenate(Cs).astype(np.float32)
    # districts: low-frequency brightness variation; brighter towards the centre mound
    dist = np.clip(fbm(np.stack([xz[:, 0], np.zeros(len(xz)), xz[:, 1]], 1) * 0.00035, octaves=3) * 1.6 + 0.9, 0.25, 1.8)
    core = 1.0 + 1.2 * np.exp(-(rad / 2500.0) ** 2)
    col *= (dist * core)[:, None]
    d = dict(P=P, rgb=col.astype(np.float32), key=key, kind=np.concatenate(kinds).astype(np.int8))
    np.savez(path, **d)
    return d


def lift_times(key, P):
    """When each light leaves: a few early sparks, then everyone; a slow wave from the centre out."""
    r = np.sqrt(P[:, 0] ** 2 + P[:, 2] ** 2) / RC
    t0, t1 = E["lift_first"], E["lift_all"]
    return t0 + (t1 - t0 + 1.2) * (0.25 * r + 0.75 * key ** 0.6) - 0.1


def lifted(P, rgb, key, tg, rise=1.0):
    """Positions/energies of lights that have left the ground by tg."""
    tl = lift_times(key, P)
    age = np.maximum(tg - tl, 0.0)
    up = age > 0
    if not up.any():
        return P, rgb, up
    Pn = P.copy()
    h = (4.0 * age + 38.0 * age ** 2.2) * rise
    Pn[up, 1] += h[up]
    drift = curl(P[up] * 0.0015, scale=1.0, t=tg * 0.2) * (h[up] * 0.10)[:, None]
    Pn[up] += drift
    e = rgb.copy()
    e[up] *= (1.6 + 2.2 * np.exp(-age[up] / 0.6))[:, None]       # a flare as it leaves the hand
    return Pn, e, up


class Crowd:
    """Rows of slender light-beings (same species as the elder, low-density instances) on a
    terrace, facing +z (the giant). Rendered with the elder's translucent glow."""

    def __init__(self, center, rows=6, per_row=34, spacing=1.35, row_gap=2.2, seed=5, pts=5000):
        from .figures import Elder
        rng = np.random.default_rng(seed)
        e = Elder()
        d = e.d
        sel = rng.choice(len(d["body_P"]), pts, replace=False)
        selh = rng.choice(len(d["head_P"]), pts // 5, replace=False)
        self.proto_P = np.concatenate([d["body_P"][sel], d["head_P"][selh]]).astype(np.float32)
        self.proto_N = np.concatenate([d["body_N"][sel], d["head_N"][selh]]).astype(np.float32)
        a_b = d["body_a"] * len(d["body_P"]) / pts
        a_h = d["head_a"] * len(d["head_P"]) / (pts // 5)
        self.proto_a = np.concatenate([np.full(pts, a_b), np.full(pts // 5, a_h)]).astype(np.float32)
        self.chest = (self.proto_P[:, 1] > 1.18) & (self.proto_P[:, 1] < 1.42)
        self.bodies = []
        c = np.asarray(center, float)
        base_y = terrace_h(np.hypot(c[0], c[2]))
        for i in range(rows):
            for j in range(per_row):
                x = c[0] + (j - per_row / 2) * spacing + rng.normal(0, 0.22) + (i % 2) * spacing * 0.5
                z = c[2] - i * row_gap + rng.normal(0, 0.3)
                sc = rng.uniform(0.86, 1.02)
                yaw = rng.normal(0, 0.12)
                self.bodies.append((x, base_y, z, sc, yaw, rng.random() * 6.28))

    def pose(self, tg, skip=None):
        """A pose dict (P, N, col, area, key, part) for all bodies (compatible with elder_points)."""
        Ps, Ns, Cs, As, Ks = [], [], [], [], []
        for n, (x, y, z, sc, yaw, ph) in enumerate(self.bodies):
            if skip is not None and np.hypot(x - skip[0], z - skip[2]) < 0.9:
                continue
            c, s_ = np.cos(yaw), np.sin(yaw)
            Rm = np.array([[c, 0, s_], [0, 1, 0], [-s_, 0, c]])
            P = (self.proto_P * sc) @ Rm.T + np.array([x, y, z])
            breath = 0.55 + 0.45 * np.sin(2 * np.pi * tg / 3.2 + ph)
            col = np.tile((GOLD * 0.55 + EMBER * 0.45), (len(P), 1))
            col[self.chest] = GOLD_HI * (0.8 + 2.6 * breath)
            Ps.append(P)
            Ns.append(self.proto_N @ Rm.T)
            Cs.append(col)
            As.append(self.proto_a * sc * sc)
            Ks.append(hash01(np.arange(len(P)) + n * 10007, 17))
        return dict(P=np.concatenate(Ps).astype(np.float32), N=np.concatenate(Ns).astype(np.float32),
                    col=np.concatenate(Cs).astype(np.float32), area=np.concatenate(As).astype(np.float32),
                    key=np.concatenate(Ks).astype(np.float32), part=np.zeros(sum(len(p) for p in Ps), np.int16))


_G = {}


def giant():
    if "g" not in _G:
        _G["g"] = RedGiant(n=700_000, seed=33)
    return _G["g"]


def draw_giant(R, cam, tg, elev_deg=-8.0, ang_radius_deg=48.0, energy=1.6, az_deg=0.0, mask_stars=None):
    """The red giant rising on the horizon: a wall of boiling red filling half the sky."""
    d = 2.0e7
    a = np.radians(az_deg)
    el = np.radians(elev_deg)
    dirv = np.array([np.sin(a) * np.cos(el), np.sin(el), np.cos(a) * np.cos(el)])
    C = cam.pos + dirv * d
    Rs = d * np.sin(np.radians(ang_radius_deg)) * (1 + 0.004 * np.cos(2 * np.pi * (tg - E["star_breath_first"]) / 4.0))
    g = giant()
    P, col = g.surface(tg, C, Rs, cam.pos, energy=energy)
    R.draw(cam, P, col, size_px=1.2)
    P, col = g.corona(tg, C, Rs, cam.pos, n=40_000, energy=0.35)
    R.draw(cam, P, col, size_px=4.0, soft=True)
    return C, Rs


def horizon_mask(cam, W, H, dip_deg=0.9, soft=2.0):
    """1 above the (slightly dipped) horizon, 0 below, for a flat-ish planet under the camera."""
    import cv2
    yy = np.arange(H, dtype=np.float64)
    xs = np.linspace(0, W - 1, 9)
    # sample the horizon direction across the frame
    pts = []
    for x in xs:
        xc = (x - cam.W / 2 - cam.shift[0]) / cam.fpx
        # direction for this column on the horizon: solve for y where the ray elevation = -dip
        best = None
        for y in np.linspace(-H, 2 * H, 600):
            yc = -(y - cam.H / 2 + cam.shift[1]) / cam.fpx
            dvec = cam.fwd + xc * cam.right + yc * cam.up
            elev = np.degrees(np.arcsin(dvec[1] / np.linalg.norm(dvec)))
            if best is None or abs(elev + dip_deg) < best[0]:
                best = (abs(elev + dip_deg), y)
        pts.append(best[1])
    hy = np.interp(np.arange(W), xs, pts)
    m = (yy[:, None] < hy[None, :]).astype(np.float32)
    return cv2.GaussianBlur(m, (0, 0), soft), hy


def sky_glow(W, H, hy, strength=1.0):
    """Atmosphere lit by the giant: deep red at the horizon fading to plum-black overhead."""
    yy = np.arange(H, dtype=np.float32)[:, None]
    d = np.clip((hy[None, :] - yy) / H, 0, 2)
    red = hex_lin("#FF4A1A")
    plum = hex_lin("#2A0710")
    g = np.exp(-d / 0.10)[..., None] * red * 0.35 + np.exp(-d / 0.5)[..., None] * plum * 0.25
    g *= (yy < hy[None, :])[..., None]
    # ground haze just below the horizon (lit by the giant)
    b = np.clip((yy - hy[None, :]) / H, 0, 2)
    g += (np.exp(-b / 0.035) * (yy >= hy[None, :]))[..., None] * red * 0.10
    return (g * strength).astype(np.float32)
