"""Star fields, the Milky Way band (seen from inside) and spiral galaxies (seen from outside)."""
import numpy as np

from ..color import blackbody, desaturate, hex_lin
from ..config import CACHE
from ..noise import fbm


def _cached(name, fn):
    path = CACHE / f"{name}.npz"
    if path.exists():
        d = np.load(path)
        return {k: d[k] for k in d.files}
    d = fn()
    np.savez(path, **d)
    return d


def _unit(v):
    return v / np.linalg.norm(v, axis=1, keepdims=True)


def _star_colors(rng, n, desat=0.45):
    u = rng.random(n)
    T = np.where(u < 0.06, rng.uniform(2900, 3900, n),
        np.where(u < 0.62, rng.uniform(4000, 6500, n),
        np.where(u < 0.88, rng.uniform(6500, 10000, n), rng.uniform(10000, 22000, n))))
    return desaturate(blackbody(T), desat).astype(np.float32)


def _star_flux(rng, n, mmin=-1.0, mmax=8.5, slope=0.33):
    a, b = 10 ** (slope * mmin), 10 ** (slope * mmax)
    m = np.log10(rng.random(n) * (b - a) + a) / slope
    return (10 ** (-0.4 * (m - mmax + 2.5))).astype(np.float32), m


def field(n=90000, seed=11):
    """Uniform all-sky star field. Returns dirs (n,3) and energy (n,3)."""
    def make():
        rng = np.random.default_rng(seed)
        d = _unit(rng.standard_normal((n, 3)))
        flux, _ = _star_flux(rng, n)
        col = _star_colors(rng, n) * flux[:, None]
        return dict(dirs=d.astype(np.float32), rgb=col.astype(np.float32))
    return _cached(f"field_{n}_{seed}", make)


def galactic_basis(center_dir, north_dir):
    """Rotation matrix mapping galactic (x toward centre, y toward north pole, z = x cross y) to world."""
    c = np.asarray(center_dir, float)
    c /= np.linalg.norm(c)
    nth = np.asarray(north_dir, float)
    nth = nth - c * np.dot(nth, c)
    nth /= np.linalg.norm(nth)
    z = np.cross(c, nth)
    return np.stack([c, nth, z], axis=1)   # columns


def milky_way(n_glow=1_400_000, n_stars=260_000, seed=5):
    """The band as seen from Earth, in galactic coordinates (x -> centre, y -> north pole).

    Returns dict with glow dirs/rgb (soft, faint) and star dirs/rgb. Dust lanes (incl. a Great
    Rift-like split) are carved by rejecting points."""
    def make():
        rng = np.random.default_rng(seed)

        def sample(n, core_frac, bscale):
            l = np.where(rng.random(n) < core_frac, rng.normal(0, 32, n), rng.uniform(-180, 180, n))
            l = (l + 180) % 360 - 180
            bulge = np.exp(-(l / 16.0) ** 2)
            bs = bscale * (1.0 + 1.6 * bulge)
            b = rng.laplace(0, 1, n) * bs
            return np.radians(l), np.radians(b)

        def dust(l, b):
            # lane wobbles along the plane; Great Rift = extra split between l=+10 and +80 deg
            ld, bd = np.degrees(l), np.degrees(b)
            P = np.stack([np.cos(l) * 3, np.sin(l) * 3, bd / 6.0], axis=1)
            wob = fbm(P, octaves=3, scale=1.0) * 3.0
            lane = np.exp(-((bd - wob * 0.8) / 1.6) ** 2)
            rift_on = np.clip(1 - np.abs(ld - 45) / 40, 0, 1)
            rift = rift_on * np.exp(-((bd - 1.8 - wob) / 1.4) ** 2)
            clumps = np.clip(fbm(P * 2.3, octaves=4, scale=1.0, offset=(5, 9, 2)) * 2.2 + 0.1, 0, 1)
            d = np.clip(0.85 * lane + 0.9 * rift + 0.55 * clumps * np.exp(-(bd / 9) ** 2), 0, 1)
            return d

        # diffuse glow
        l, b = sample(int(n_glow * 1.6), 0.45, 3.6)
        dd = dust(l, b)
        keep = rng.random(l.size) > dd
        l, b = l[keep][:n_glow], b[keep][:n_glow]
        g_dirs = np.stack([np.cos(b) * np.cos(l), np.sin(b), np.cos(b) * np.sin(l)], axis=1)
        core = np.exp(-(np.degrees(l) / 40) ** 2)
        warm, cool = hex_lin("#FFE2B8"), hex_lin("#C9D6FF")
        g_rgb = (warm * core[:, None] + cool * (1 - core[:, None]))
        g_rgb *= (0.55 + 0.45 * rng.random(l.size))[:, None] * (0.6 + 1.2 * core)[:, None]
        # sparse HII pink knots
        pink = rng.random(l.size) < 0.004
        g_rgb[pink] = hex_lin("#FF7FA8") * 2.5
        # resolved stars inside the band
        l2, b2 = sample(int(n_stars * 1.4), 0.4, 5.0)
        keep2 = rng.random(l2.size) > dust(l2, b2) * 0.8
        l2, b2 = l2[keep2][:n_stars], b2[keep2][:n_stars]
        s_dirs = np.stack([np.cos(b2) * np.cos(l2), np.sin(b2), np.cos(b2) * np.sin(l2)], axis=1)
        flux, _ = _star_flux(rng, s_dirs.shape[0], mmin=2.0, mmax=9.5, slope=0.38)
        s_rgb = _star_colors(rng, s_dirs.shape[0]) * flux[:, None]
        return dict(g_dirs=g_dirs.astype(np.float32), g_rgb=g_rgb.astype(np.float32),
                    s_dirs=s_dirs.astype(np.float32), s_rgb=s_rgb.astype(np.float32))
    return _cached(f"milkyway_{n_glow}_{n_stars}_{seed}", make)


def spiral_galaxy(n=2_000_000, seed=3, arms=4, pitch_deg=12.0):
    """Face-on spiral in the XZ plane, radius 1 (units of ~50,000 ly), thickness along Y.

    Returns pos (n,3), rgb (n,3), kind (n,) where kind 0=bulge 1=arm 2=disk 3=HII."""
    def make():
        rng = np.random.default_rng(seed)
        nb, na, nd, nh = int(n * 0.16), int(n * 0.58), int(n * 0.24), int(n * 0.02)
        k = 1.0 / np.tan(np.radians(pitch_deg))
        # bulge + bar
        pb = rng.standard_normal((nb, 3)) * np.array([0.09, 0.045, 0.09])
        bar = rng.random(nb) < 0.35
        pb[bar] = rng.standard_normal((bar.sum(), 3)) * np.array([0.20, 0.03, 0.05])
        # arms
        r = np.clip(rng.exponential(0.33, na) + 0.08, 0.08, 1.05)
        arm = rng.integers(0, arms, na)
        th0 = np.log(r / 0.08) * k * 0.5
        spread = rng.normal(0, 1, na) * (0.10 + 0.22 * r)
        th = th0 + arm * 2 * np.pi / arms + spread
        pa = np.stack([r * np.cos(th), rng.normal(0, 0.012, na), r * np.sin(th)], axis=1)
        # smooth disk
        rd = np.clip(rng.exponential(0.30, nd), 0, 1.1)
        thd = rng.uniform(0, 2 * np.pi, nd)
        pd = np.stack([rd * np.cos(thd), rng.normal(0, 0.02, nd), rd * np.sin(thd)], axis=1)
        # HII knots along arms
        rh = np.clip(rng.exponential(0.35, nh) + 0.15, 0.15, 1.0)
        armh = rng.integers(0, arms, nh)
        thh = np.log(rh / 0.08) * k * 0.5 + armh * 2 * np.pi / arms + rng.normal(0, 0.05, nh)
        ph = np.stack([rh * np.cos(thh), rng.normal(0, 0.006, nh), rh * np.sin(thh)], axis=1)
        pos = np.concatenate([pb, pa, pd, ph]).astype(np.float32)
        kind = np.concatenate([np.zeros(nb), np.ones(na), np.full(nd, 2), np.full(nh, 3)]).astype(np.int8)
        rr = np.linalg.norm(pos[:, [0, 2]], axis=1)
        warm, blue, pink = hex_lin("#FFE6B0"), hex_lin("#9DB7FF"), hex_lin("#FF6FA3")
        t = np.clip(rr / 0.45, 0, 1)[:, None]
        rgb = warm * (1 - t) + blue * t
        rgb = rgb * (0.4 + 0.9 * rng.random(pos.shape[0]))[:, None]
        rgb[kind == 0] *= 1.6
        rgb[kind == 3] = pink * 3.0
        # dust lanes: darken the inner (trailing) edge of each arm
        dark = (kind == 1) & (spread < -0.02) & (spread > -0.14)
        rgb[dark] *= 0.25
        return dict(pos=pos, rgb=rgb.astype(np.float32), kind=kind)
    return _cached(f"galaxy_{n}_{seed}_{arms}_{pitch_deg}", make)
