"""Vectorised signed-distance primitives + surface sampler (for sculpting characters and props)."""
import numpy as np


def length(p):
    return np.sqrt(np.sum(p * p, axis=-1))


def sd_sphere(p, c, r):
    return length(p - np.asarray(c)) - r


def sd_ellipsoid(p, c, r, Rm=None):
    q = p - np.asarray(c)
    if Rm is not None:
        q = q @ np.asarray(Rm)          # rotate into the ellipsoid frame (Rm columns = local axes)
    r = np.asarray(r, float)
    k0 = length(q / r)
    k1 = length(q / (r * r))
    return k0 * (k0 - 1.0) / np.maximum(k1, 1e-9)


def sd_capsule(p, a, b, r):
    a, b = np.asarray(a, float), np.asarray(b, float)
    pa, ba = p - a, b - a
    h = np.clip((pa @ ba) / (ba @ ba), 0, 1)
    return length(pa - h[:, None] * ba) - r


def sd_round_cone(p, a, b, r1, r2):
    a, b = np.asarray(a, float), np.asarray(b, float)
    ba = b - a
    l2 = ba @ ba
    rr = r1 - r2
    a2 = l2 - rr * rr
    il2 = 1.0 / l2
    pa = p - a
    y = pa @ ba
    z = y - l2
    q = pa * l2 - y[:, None] * ba
    x2 = np.sum(q * q, axis=1)
    y2 = y * y * l2
    z2 = z * z * l2
    k = np.sign(rr) * rr * rr * x2
    d1 = np.sqrt(x2 + z2) * il2 - r2
    d2 = np.sqrt(x2 + y2) * il2 - r1
    d3 = (np.sqrt(np.maximum(x2 * a2 * il2, 0)) + y * rr) * il2 - r1
    return np.where(np.sign(z) * a2 * z2 > k, d1, np.where(np.sign(y) * a2 * y2 < k, d2, d3))


def sd_round_box(p, c, b, r, Rm=None):
    q = p - np.asarray(c)
    if Rm is not None:
        q = q @ np.asarray(Rm)
    q = np.abs(q) - np.asarray(b)
    return length(np.maximum(q, 0)) + np.minimum(np.max(q, axis=1), 0) - r


def sd_torus_y(p, c, R, r):
    q = p - np.asarray(c)
    xz = np.sqrt(q[:, 0] ** 2 + q[:, 2] ** 2) - R
    return np.sqrt(xz ** 2 + q[:, 1] ** 2) - r


def smin(a, b, k):
    if k <= 0:
        return np.minimum(a, b)
    h = np.clip(0.5 + 0.5 * (b - a) / k, 0, 1)
    return b + (a - b) * h - k * h * (1 - h)


def smax(a, b, k):
    return -smin(-a, -b, k)


def rot(ax, ang):
    """Rotation matrix about a principal axis ('x','y','z') by angle (rad)."""
    c, s = np.cos(ang), np.sin(ang)
    if ax == "x":
        return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])
    if ax == "y":
        return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


def grad(f, p, e=1e-4):
    ex, ey, ez = np.array([e, 0, 0]), np.array([0, e, 0]), np.array([0, 0, e])
    g = np.stack([f(p + ex) - f(p - ex), f(p + ey) - f(p - ey), f(p + ez) - f(p - ez)], axis=1) / (2 * e)
    return g


def sample_surface(f, lo, hi, n, band=None, seed=0, batch=1_500_000, max_batches=200):
    """Uniformly sample ~n points on the zero set of SDF f inside the box [lo, hi].

    Returns P (n,3), N (n,3) unit normals, area_per_point (float)."""
    lo, hi = np.asarray(lo, float), np.asarray(hi, float)
    ext = hi - lo
    band = band or float(ext.min()) * 0.01
    rng = np.random.default_rng(seed)
    got, tried = [], 0
    count = 0
    for _ in range(max_batches):
        c = lo + rng.random((batch, 3)) * ext
        d = f(c)
        m = np.abs(d) < band
        tried += batch
        if m.any():
            got.append(c[m])
            count += int(m.sum())
        if count >= n:
            break
    P = np.concatenate(got)
    total_area = count / tried * float(np.prod(ext)) / (2 * band)
    P = P[rng.permutation(P.shape[0])[:n]]
    for _ in range(3):
        g = grad(f, P)
        gn = np.sum(g * g, axis=1, keepdims=True) + 1e-12
        P = P - f(P)[:, None] * g / gn
    N = grad(f, P)
    N /= np.linalg.norm(N, axis=1, keepdims=True) + 1e-12
    return P.astype(np.float32), N.astype(np.float32), float(total_area / P.shape[0])
