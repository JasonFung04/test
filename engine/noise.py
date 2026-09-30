"""Vectorised 3-D gradient noise (Perlin 'improved'), fbm and curl, compiled with numba."""
import numpy as np
from numba import njit, prange

_rng = np.random.default_rng(1729)
_P = _rng.permutation(256).astype(np.int32)
PERM = np.concatenate([_P, _P]).astype(np.int32)

_GRAD = np.array([[1, 1, 0], [-1, 1, 0], [1, -1, 0], [-1, -1, 0],
                  [1, 0, 1], [-1, 0, 1], [1, 0, -1], [-1, 0, -1],
                  [0, 1, 1], [0, -1, 1], [0, 1, -1], [0, -1, -1],
                  [1, 1, 0], [-1, 1, 0], [0, -1, 1], [0, -1, -1]], dtype=np.float64)


@njit(cache=True, fastmath=True, inline="always")
def _fade(t):
    return t * t * t * (t * (t * 6.0 - 15.0) + 10.0)


@njit(cache=True, fastmath=True, inline="always")
def _lerp(a, b, t):
    return a + t * (b - a)


@njit(cache=True, fastmath=True, inline="always")
def _grad(h, x, y, z, G):
    h = h & 15
    return G[h, 0] * x + G[h, 1] * y + G[h, 2] * z


@njit(cache=True, fastmath=True)
def _noise1(x, y, z, P, G):
    xf = np.floor(x)
    yf = np.floor(y)
    zf = np.floor(z)
    X = int(xf) & 255
    Y = int(yf) & 255
    Z = int(zf) & 255
    x -= xf
    y -= yf
    z -= zf
    u = _fade(x)
    v = _fade(y)
    w = _fade(z)
    A = P[X] + Y
    AA = P[A] + Z
    AB = P[A + 1] + Z
    B = P[X + 1] + Y
    BA = P[B] + Z
    BB = P[B + 1] + Z
    return _lerp(_lerp(_lerp(_grad(P[AA], x, y, z, G), _grad(P[BA], x - 1, y, z, G), u),
                       _lerp(_grad(P[AB], x, y - 1, z, G), _grad(P[BB], x - 1, y - 1, z, G), u), v),
                 _lerp(_lerp(_grad(P[AA + 1], x, y, z - 1, G), _grad(P[BA + 1], x - 1, y, z - 1, G), u),
                       _lerp(_grad(P[AB + 1], x, y - 1, z - 1, G), _grad(P[BB + 1], x - 1, y - 1, z - 1, G), u),
                       v), w)


@njit(cache=True, fastmath=True, parallel=True)
def _fbm(xs, ys, zs, octaves, lac, gain, P, G):
    n = xs.shape[0]
    out = np.empty(n, dtype=np.float32)
    for i in prange(n):
        a = 1.0
        f = 1.0
        s = 0.0
        norm = 0.0
        for o in range(octaves):
            s += a * _noise1(xs[i] * f + o * 17.13, ys[i] * f + o * 3.71, zs[i] * f + o * 11.07, P, G)
            norm += a
            a *= gain
            f *= lac
        out[i] = s / norm
    return out


def fbm(P3, octaves=4, lac=2.0, gain=0.5, scale=1.0, offset=(0.0, 0.0, 0.0)):
    """P3: (N,3) array. Returns float32 (N,) roughly in [-0.7, 0.7]."""
    P3 = np.asarray(P3, dtype=np.float64)
    x = np.ascontiguousarray(P3[:, 0] * scale + offset[0])
    y = np.ascontiguousarray(P3[:, 1] * scale + offset[1])
    z = np.ascontiguousarray(P3[:, 2] * scale + offset[2])
    return _fbm(x, y, z, int(octaves), float(lac), float(gain), PERM, _GRAD)


def noise(P3, scale=1.0, offset=(0.0, 0.0, 0.0)):
    return fbm(P3, octaves=1, scale=scale, offset=offset)


def curl(P3, scale=1.0, t=0.0, eps=1e-3, octaves=2):
    """Divergence-free flow field (N,3) from three offset noise potentials."""
    P3 = np.asarray(P3, dtype=np.float64)
    offs = [(0.0, 0.0, t), (31.4, 47.2, t * 1.1), (-19.8, 73.1, t * 0.9)]

    def pot(k, d):
        return fbm(P3 + d, octaves=octaves, scale=scale, offset=offs[k])

    ex = np.array([eps, 0, 0])
    ey = np.array([0, eps, 0])
    ez = np.array([0, 0, eps])
    # curl of (psi1, psi2, psi3)
    dp3_dy = (pot(2, ey) - pot(2, -ey)) / (2 * eps)
    dp2_dz = (pot(1, ez) - pot(1, -ez)) / (2 * eps)
    dp1_dz = (pot(0, ez) - pot(0, -ez)) / (2 * eps)
    dp3_dx = (pot(2, ex) - pot(2, -ex)) / (2 * eps)
    dp2_dx = (pot(1, ex) - pot(1, -ex)) / (2 * eps)
    dp1_dy = (pot(0, ey) - pot(0, -ey)) / (2 * eps)
    return np.stack([dp3_dy - dp2_dz, dp1_dz - dp3_dx, dp2_dx - dp1_dy], axis=1).astype(np.float32)


def hash01(i, seed=0):
    """Deterministic per-index uniform random in [0,1)."""
    i = np.asarray(i, dtype=np.uint64)
    x = (i * np.uint64(0x9E3779B97F4A7C15) + np.uint64(seed) * np.uint64(0xBF58476D1CE4E5B9)) & np.uint64(
        0xFFFFFFFFFFFFFFFF)
    x ^= x >> np.uint64(30)
    x = (x * np.uint64(0xBF58476D1CE4E5B9)) & np.uint64(0xFFFFFFFFFFFFFFFF)
    x ^= x >> np.uint64(27)
    x = (x * np.uint64(0x94D049BB133111EB)) & np.uint64(0xFFFFFFFFFFFFFFFF)
    x ^= x >> np.uint64(31)
    return (x >> np.uint64(11)).astype(np.float64) / float(1 << 53)
