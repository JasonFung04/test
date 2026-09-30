"""The Blombos fire: one flicker rhythm for every shot of the cave (II4-II8), particle flames,
embers and smoke.

Fire space: origin at the centre of the hearth on the floor, +y up. All randomness is hashed from
particle indices, so any frame can be rendered on its own.
"""
import numpy as np

from ..color import hex_lin
from ..noise import fbm, hash01

FIRE = hex_lin("#FF8A2A")
LIGHT = np.array([1.0, 0.60, 0.32], np.float32)       # colour of the light the fire casts (graded ~2600 K)
CORE = hex_lin("#FFE3A0")
EMBER = hex_lin("#FF5A1A")
SMOKE = hex_lin("#6B5A4E")


def _n1(t, f, seed):
    """Smooth 1-D noise from a few incommensurate sines (deterministic, cheap)."""
    rng = np.random.default_rng(seed)
    ph = rng.uniform(0, 2 * np.pi, 4)
    fr = f * np.array([1.0, 1.618, 2.414, 3.303])
    a = np.array([1.0, 0.6, 0.35, 0.2])
    return float(np.sum(a * np.sin(2 * np.pi * fr * t + ph)) / a.sum())


def flicker(tg):
    """Intensity multiplier of the firelight (~0.75..1.2), shared by every shot of the cave."""
    return 1.0 + 0.10 * _n1(tg, 0.7, 1) + 0.06 * _n1(tg, 2.3, 2) + 0.035 * _n1(tg, 6.1, 3)


def light_offset(tg, scale=0.03):
    """The flames sway: the effective light centre wanders a few cm (moving shadows)."""
    return np.array([_n1(tg, 0.9, 4), 0.4 * _n1(tg, 1.3, 5), _n1(tg, 1.1, 6)]) * scale


def flames(tg, n_rate=9000.0, life=(0.35, 0.9), radius=0.16, height=0.55):
    """Flame particles alive at tg. Returns P (N,3) fire space, rgb (N,3) energy, size (N,) px."""
    lmax = life[1]
    k0 = int(np.floor((tg - lmax) * n_rate))
    k1 = int(np.floor(tg * n_rate))
    k = np.arange(k0, k1 + 1, dtype=np.int64)
    kk = (k % (1 << 40)).astype(np.uint64)
    birth = k / n_rate + hash01(kk, 11) / n_rate
    L = life[0] + (life[1] - life[0]) * hash01(kk, 12) ** 1.5
    age = tg - birth
    ok = (age >= 0) & (age < L)
    k, kk, birth, L, age = k[ok], kk[ok], birth[ok], L[ok], age[ok]
    u = age / L
    r = radius * np.sqrt(hash01(kk, 13))
    th = 2 * np.pi * hash01(kk, 14)
    x0 = r * np.cos(th)
    z0 = r * np.sin(th) * 0.8
    # rise with acceleration; tongues: particles are drawn toward the axis as they climb
    h = height * (0.55 + 0.9 * hash01(kk, 15)) * (u ** 0.85)
    pull = 1.0 - 0.75 * u
    # turbulent sway from a time-varying noise field (tongues lick sideways)
    q = np.stack([x0 * 4.0, h * 3.0 - tg * 2.2, z0 * 4.0], 1)
    sway_x = fbm(q, octaves=2, offset=(1.0, 0.0, 0.0)) * 0.10 * u
    sway_z = fbm(q, octaves=2, offset=(7.0, 3.0, 0.0)) * 0.08 * u
    P = np.stack([x0 * pull + sway_x, h, z0 * pull + sway_z], 1)
    # colour by age: white-yellow core -> orange -> deep red, fading out
    core = np.exp(-u / 0.18)[:, None]
    mid = np.exp(-((u - 0.35) / 0.25) ** 2)[:, None]
    col = CORE * core * 2.5 + FIRE * mid * 1.4 + hex_lin("#B8300C") * (u[:, None] ** 1.5) * 0.8
    fade = (1.0 - u) ** 1.3 * np.clip(u / 0.06, 0, 1)
    col = col * fade[:, None] * (0.6 + 0.8 * hash01(kk, 16))[:, None]
    size = 1.5 + 5.0 * u
    return P.astype(np.float32), col.astype(np.float32), size.astype(np.float32)


def embers(tg, n_rate=28.0, life=(1.6, 4.0), radius=0.14):
    """Rising, drifting sparks. Returns P, rgb, size."""
    lmax = life[1]
    k0 = int(np.floor((tg - lmax) * n_rate))
    k1 = int(np.floor(tg * n_rate))
    k = np.arange(k0, k1 + 1, dtype=np.int64)
    kk = (k % (1 << 40)).astype(np.uint64) + np.uint64(99991)
    birth = k / n_rate + hash01(kk, 21) / n_rate
    L = life[0] + (life[1] - life[0]) * hash01(kk, 22)
    age = tg - birth
    ok = (age >= 0) & (age < L)
    kk, age, L = kk[ok], age[ok], L[ok]
    u = age / L
    r = radius * np.sqrt(hash01(kk, 23))
    th = 2 * np.pi * hash01(kk, 24)
    v = 0.35 + 0.45 * hash01(kk, 25)
    y = 0.15 + v * age - 0.02 * age ** 2
    ph = 2 * np.pi * hash01(kk, 26)
    wob = 0.05 + 0.08 * hash01(kk, 27)
    x = r * np.cos(th) + wob * np.sin(age * (2.0 + 2 * hash01(kk, 28)) + ph) + 0.05 * age
    z = r * np.sin(th) + wob * np.cos(age * (1.7 + 2 * hash01(kk, 29)) + ph)
    P = np.stack([x, y, z], 1)
    tw = 0.6 + 0.4 * np.sin(age * 23.0 + ph * 3.0)
    col = EMBER[None, :] * ((1 - u) ** 1.2 * tw * (1.5 + 2.0 * hash01(kk, 30)))[:, None]
    return P.astype(np.float32), col.astype(np.float32), np.full(len(P), 0.8, np.float32)


def smoke(tg, n_rate=900.0, life=(3.0, 6.0), radius=0.12):
    """Thin smoke rising above the flames (lit from below). Returns P, rgb, size."""
    lmax = life[1]
    k0 = int(np.floor((tg - lmax) * n_rate))
    k1 = int(np.floor(tg * n_rate))
    k = np.arange(k0, k1 + 1, dtype=np.int64)
    kk = (k % (1 << 40)).astype(np.uint64) + np.uint64(777)
    birth = k / n_rate + hash01(kk, 31) / n_rate
    L = life[0] + (life[1] - life[0]) * hash01(kk, 32)
    age = tg - birth
    ok = (age >= 0) & (age < L)
    kk, age, L = kk[ok], age[ok], L[ok]
    u = age / L
    r = radius * np.sqrt(hash01(kk, 33)) * (1 + 2.5 * u)
    th = 2 * np.pi * hash01(kk, 34)
    y = 0.45 + 0.30 * age
    q = np.stack([np.cos(th) * 2, y * 2.0 - tg * 0.3, np.sin(th) * 2], 1)
    sx = fbm(q, octaves=2, offset=(3.0, 0.0, 0.0)) * 0.25 * u + 0.06 * age
    sz = fbm(q, octaves=2, offset=(9.0, 1.0, 0.0)) * 0.20 * u
    P = np.stack([r * np.cos(th) + sx, y, r * np.sin(th) * 0.8 + sz], 1)
    lit = np.exp(-(y - 0.45) / 0.5)[:, None]
    col = (SMOKE * 0.4 + FIRE * 0.6 * lit) * (np.sin(np.pi * u) * 0.022)[:, None]
    return P.astype(np.float32), col.astype(np.float32), (6.0 + 14.0 * u).astype(np.float32)
