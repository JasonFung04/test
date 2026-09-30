"""Blombos Cave, 73,000 years ago, at night (II4).

Cave space (metres): the camera sits low inside the cave near the origin and looks toward -z,
where the arched mouth opens on the sea (~35 m below) and the Milky Way. +y up, +x to the right.
The rock is a noisy arch-shaped tunnel that widens toward the mouth; the floor slopes gently down
to a short ledge outside the mouth. Sampling density falls with distance from the camera so the
pointillist grain stays even on screen.
"""
import numpy as np

from ..config import CACHE
from ..noise import fbm, hash01

Z_MOUTH = -11.5
Z_BACK = 1.2
LEDGE = 1.6                  # the floor continues this far outside the mouth
CAM0 = np.array([0.0, 0.36, 0.0])
VERSION = 2

ROCK = np.array([0.36, 0.27, 0.19], np.float32)      # warm calcarenite
ROCK_DK = np.array([0.20, 0.14, 0.10], np.float32)
SAND = np.array([0.40, 0.31, 0.22], np.float32)
ASH = np.array([0.20, 0.19, 0.18], np.float32)
SHELL = np.array([0.62, 0.58, 0.52], np.float32)


def floor_y(x, z):
    """Floor height: level near the camera, sloping down toward the mouth, with gentle undulation."""
    x = np.asarray(x, float)
    z = np.asarray(z, float)
    base = 0.028 * np.minimum(z, 0.0) + 0.03 * np.sin(x * 0.9 + 0.4)
    return base + 0.06 * fbm(np.stack([x * 0.6, np.zeros_like(x), z * 0.6], -1).reshape(-1, 3),
                            octaves=3).reshape(np.shape(x))


def arch(z):
    """Tunnel cross-section at depth z: centre x, half width, apex height above the floor."""
    u = np.clip((Z_BACK - z) / (Z_BACK - Z_MOUTH), 0, 1)
    xc = 0.25 * u
    a = 3.2 + 1.7 * u ** 1.6
    b = 2.5 + 2.15 * u ** 2.2
    return xc, a, b


def mouth_outline(n=240, seed=3):
    """Closed outline (N,3) of the opening at the mouth, incl. the ledge edge along the bottom."""
    th = np.linspace(0.0, np.pi, n)
    xc, a, b = arch(Z_MOUTH)
    x = xc - a * np.cos(th)
    y0 = floor_y(x, np.full(n, Z_MOUTH))
    rough = 0.22 * fbm(np.stack([np.cos(th) * 3, np.sin(th) * 3, np.full(n, 1.0)], 1), octaves=4)
    y = y0 + b * np.sin(th) ** 0.85 * (1 + rough * 0.3)
    x = x - rough * np.cos(th) * 0.6
    top = np.stack([x, y, np.full(n, Z_MOUTH)], 1)
    # bottom: the ledge edge outside (the floor drops away to the sea)
    xs = np.linspace(x[-1], x[0], 60)
    zl = Z_MOUTH - LEDGE + 0.25 * np.sin(xs * 1.3)
    yl = floor_y(xs, zl) - 0.05
    ledge = np.stack([xs, yl, zl], 1)
    return np.concatenate([top, ledge])


def _density_sample(rng, n, sampler, cam=CAM0, dmin=1.2):
    """Draw n candidates from `sampler`, keep with prob ~ (dmin/d)^2 (denser near the camera)."""
    P, extra = sampler(n)
    d = np.linalg.norm(P - cam, axis=1)
    p = np.clip((dmin / np.maximum(d, 1e-3)) ** 2, 0.02, 1.0)
    k = rng.random(n) < p
    return P[k], [e[k] for e in extra], p[k]


def build(density=1.0):
    path = CACHE / f"cave_v{VERSION}_{density}.npz"
    if path.exists():
        d = np.load(path)
        return {k: d[k] for k in d.files}
    rng = np.random.default_rng(7)

    # ---------------------------------------------------------------- walls + ceiling
    def wall_sampler(n):
        z = rng.uniform(Z_MOUTH, Z_BACK, n)
        th = rng.uniform(0.02, np.pi - 0.02, n)
        xc, a, b = arch(z)
        x = xc - a * np.cos(th)
        yf = floor_y(x, z)
        y = yf + b * np.sin(th) ** 0.85
        P = np.stack([x, y, z], 1)
        # inward normal of the arch (pointing into the cave)
        nx = np.cos(th) / a
        ny = -np.sin(th) / b
        N = np.stack([nx, ny, np.zeros(n)], 1)
        N /= np.linalg.norm(N, axis=1, keepdims=True)
        big = fbm(P * 0.45, octaves=3, offset=(1.0, 2.0, 3.0))
        small = fbm(P * 2.2, octaves=3, offset=(4.0, 5.0, 6.0))
        disp = 0.45 * big + 0.10 * small
        # the rim of the mouth: a thicker lip of rock
        lip = np.exp(-((z - Z_MOUTH) / 0.6) ** 2) * 0.25
        P = P - N * (disp + lip)[:, None]
        # perturbed normals from the displacement field
        e = 0.05
        gx = (fbm((P + [e, 0, 0]) * 2.2, octaves=2, offset=(4, 5, 6)) - fbm((P - [e, 0, 0]) * 2.2, octaves=2,
                                                                             offset=(4, 5, 6))) / (2 * e)
        gy = (fbm((P + [0, e, 0]) * 2.2, octaves=2, offset=(4, 5, 6)) - fbm((P - [0, e, 0]) * 2.2, octaves=2,
                                                                             offset=(4, 5, 6))) / (2 * e)
        gz = (fbm((P + [0, 0, e]) * 2.2, octaves=2, offset=(4, 5, 6)) - fbm((P - [0, 0, e]) * 2.2, octaves=2,
                                                                             offset=(4, 5, 6))) / (2 * e)
        G = np.stack([gx, gy, gz], 1) * 0.25
        N = N + G - N * np.sum(G * N, axis=1, keepdims=True)
        N /= np.linalg.norm(N, axis=1, keepdims=True)
        # area element of this parametrisation (per unit z per unit theta)
        dl = np.sqrt((a * np.sin(th)) ** 2 + (b * 0.85 * np.sin(th) ** -0.15 * np.cos(th)) ** 2)
        return P, [N, dl, big, small]

    n = int(3_800_000 * density)
    Pw, (Nw, dlw, bigw, smallw), pw = _density_sample(rng, n, wall_sampler)
    area_w = (Z_BACK - Z_MOUTH) * (np.pi - 0.04) * dlw / (n * pw)
    t = np.clip(0.5 + 1.3 * bigw, 0, 1)[:, None]
    alb_w = ROCK * t + ROCK_DK * (1 - t)
    alb_w = alb_w * (0.75 + 0.5 * np.clip(smallw + 0.5, 0, 1))[:, None]
    # soot above the hearth (blackened ceiling) and pale flowstone streaks
    soot = np.exp(-((Pw[:, 0] + 0.85) ** 2 + (Pw[:, 2] + 3.2) ** 2) / 2.5) * (Pw[:, 1] > 1.5)
    alb_w *= (1 - 0.6 * soot)[:, None]

    # ---------------------------------------------------------------- floor (incl. the ledge outside)
    def floor_sampler(n):
        z = rng.uniform(Z_MOUTH - LEDGE, 0.9, n)
        xc, a, b = arch(np.maximum(z, Z_MOUTH))
        x = xc + rng.uniform(-1, 1, n) * (a - 0.05)
        y = floor_y(x, z)
        P = np.stack([x, y, z], 1)
        g = fbm(P * 3.0, octaves=3, offset=(9.0, 1.0, 1.0))
        P[:, 1] += 0.02 * g
        N = np.stack([-fbm(P * 3 + 2, octaves=2) * 0.3, np.ones(n), -fbm(P * 3 + 5, octaves=2) * 0.3], 1)
        N /= np.linalg.norm(N, axis=1, keepdims=True)
        return P, [N, a, g]

    n = int(2_600_000 * density)
    Pf, (Nf, af, gf), pf = _density_sample(rng, n, floor_sampler)
    area_f = (0.9 - (Z_MOUTH - LEDGE)) * 2 * af / (n * pf)
    alb_f = SAND * (0.7 + 0.5 * np.clip(gf + 0.5, 0, 1))[:, None]
    hearth = np.exp(-((Pf[:, 0] + 0.85) ** 2 + (Pf[:, 2] + 3.2) ** 2) / 0.35)
    alb_f = alb_f * (1 - hearth[:, None]) + ASH * hearth[:, None]
    shells = hash01(np.arange(len(Pf)), 5) < 0.035 * (1 - hearth)
    alb_f[shells] = SHELL * (0.6 + 0.4 * hash01(np.arange(shells.sum()), 6))[:, None]
    # outside the mouth the ledge is only lit by the sky
    outside = Pf[:, 2] < Z_MOUTH

    P = np.concatenate([Pw, Pf]).astype(np.float32)
    N = np.concatenate([Nw, Nf]).astype(np.float32)
    alb = np.concatenate([alb_w, alb_f]).astype(np.float32)
    area = np.concatenate([area_w, area_f]).astype(np.float32)
    kind = np.concatenate([np.zeros(len(Pw), np.int8), np.ones(len(Pf), np.int8)])
    out = np.concatenate([np.zeros(len(Pw), bool), outside])
    d = dict(P=P, N=N, alb=alb, area=area, kind=kind, outside=out)
    np.savez(path, **d)
    return d


def hearth_stones(seed=4):
    """A few stones and sticks around the fire (fire space: centre of the hearth on the floor)."""
    rng = np.random.default_rng(seed)
    Ps, Ns, As = [], [], []
    for k in range(9):
        ang = k / 9 * 2 * np.pi + rng.normal(0, 0.15)
        c = np.array([np.cos(ang) * 0.30, 0.03, np.sin(ang) * 0.24])
        r = rng.uniform(0.035, 0.06, 3) * np.array([1.2, 0.7, 1.0])
        m = 1500
        u = rng.normal(size=(m, 3))
        u /= np.linalg.norm(u, axis=1, keepdims=True)
        Ps.append(c + u * r)
        Ns.append(u / r)
        As.append(np.full(m, 4 * np.pi * r.mean() ** 2 / m))
    P = np.concatenate(Ps)
    N = np.concatenate(Ns)
    N /= np.linalg.norm(N, axis=1, keepdims=True)
    return P.astype(np.float32), N.astype(np.float32), np.concatenate(As).astype(np.float32)


def logs(seed=6):
    """Charred sticks in the hearth with glowing coals: P, N, albedo, glow (0..1)."""
    rng = np.random.default_rng(seed)
    Ps, Ns, G = [], [], []
    for k in range(6):
        ang = k / 6 * np.pi * 2 + rng.normal(0, 0.3)
        a = np.array([np.cos(ang) * 0.26, 0.02, np.sin(ang) * 0.22])
        b = np.array([np.cos(ang) * 0.03, 0.09 + rng.uniform(0, 0.05), np.sin(ang) * 0.03])
        m = 2500
        s = rng.random(m)
        th = rng.random(m) * 2 * np.pi
        ax = b - a
        L = np.linalg.norm(ax)
        ax /= L
        e1 = np.cross(ax, [0, 1, 0])
        e1 /= np.linalg.norm(e1)
        e2 = np.cross(ax, e1)
        r = 0.018 + 0.006 * rng.random()
        nrm = np.cos(th)[:, None] * e1 + np.sin(th)[:, None] * e2
        Ps.append(a + ax * (s * L)[:, None] + nrm * r)
        Ns.append(nrm)
        G.append(np.clip((s - 0.45) / 0.55, 0, 1) ** 1.5)       # glowing toward the fire
    return (np.concatenate(Ps).astype(np.float32), np.concatenate(Ns).astype(np.float32),
            np.concatenate(G).astype(np.float32))
