"""THE GIRL's eye for the climax extreme close-up (III8b).

Eye-local frame, metres: origin at the eyeball centre, +z = gaze (out of the pupil), +y up.
Dark-adapted: the pupil is wide (3.3 mm). Iris grey-brown #6B5646 with a warm collarette.
"""
import numpy as np

from ..color import hex_lin
from ..noise import fbm, hash01

R_EYE = 0.0120
R_IRIS = 0.0059
R_PUPIL = 0.0033
Z_IRIS = 0.0094
CORNEA_C = np.array([0.0, 0.0, 0.0046])
CORNEA_R = 0.0078
HIGHLIGHT = np.array([-0.0016, 0.0019, 0.0])     # on the cornea, up-left of the pupil (xy offset)

IRIS_OUT = hex_lin("#3E3028")
IRIS_MID = hex_lin("#6B5646")
IRIS_IN = hex_lin("#8E6A3C")
SCLERA = hex_lin("#C9CED8")
SKIN = hex_lin("#B98E78")
LASH = hex_lin("#120D0B")


def cornea_point(xy):
    """Point on the cornea dome above a given (x, y) offset."""
    x, y = xy
    z = CORNEA_C[2] + np.sqrt(max(CORNEA_R ** 2 - x * x - y * y, 0.0))
    return np.array([x, y, z])


def build(seed=4, n_iris=240_000, n_sclera=60_000, n_skin=160_000, n_lash=6000):
    rng = np.random.default_rng(seed)
    Ps, Cs, kinds = [], [], []
    # ---- iris: many wavy radial fibres, crypts as darker patches, amber collarette, outer furrows
    nf = 1800
    fib_a = rng.uniform(0, 2 * np.pi, nf)
    fib_len = rng.uniform(0.45, 1.0, nf)
    fib_w = rng.uniform(0.2, 1.0, nf)
    per = n_iris // nf
    a0 = np.repeat(fib_a, per)
    t = rng.random(nf * per) * np.repeat(fib_len, per)
    t = np.where(rng.random(nf * per) < 0.5, t, 1 - t)          # fibres from both ends
    a = a0 + 0.035 * np.sin(t * 7.0 + np.repeat(rng.uniform(0, 6.3, nf), per)) + rng.normal(0, 0.004, nf * per)
    r = R_PUPIL + (R_IRIS - R_PUPIL) * t
    x, y = r * np.cos(a), r * np.sin(a)
    z = Z_IRIS + 0.00035 * np.sin(np.pi * t) + 0.00012 * np.cos(a * 11)
    P = np.stack([x, y, z], 1)
    crypt = fbm(np.stack([x * 1400, y * 1400, np.zeros_like(x)], 1), octaves=3)
    shade = np.clip(0.75 + 1.4 * crypt, 0.25, 1.4)
    furrow = 1.0 - 0.35 * np.exp(-((t - 0.78) / 0.02) ** 2) - 0.3 * np.exp(-((t - 0.9) / 0.02) ** 2)
    base = np.where(t[:, None] < 0.26, IRIS_IN * 1.25, np.where(t[:, None] < 0.75, IRIS_MID, IRIS_OUT * 1.1))
    col = base * (shade * furrow * np.repeat(0.6 + 0.6 * fib_w, per))[:, None]
    coll = np.abs(t - 0.27) < 0.03
    col[coll] *= 1.6
    Ps.append(P)
    Cs.append(col)
    kinds.append(np.zeros(len(P)))
    # limbal ring (dark, dense)
    n = 30_000
    a = rng.uniform(0, 2 * np.pi, n)
    r = R_IRIS + rng.normal(0, 0.00012, n)
    Ps.append(np.stack([r * np.cos(a), r * np.sin(a), np.full(n, Z_IRIS - 0.0002)], 1))
    Cs.append(np.tile(IRIS_OUT * 0.35, (n, 1)))
    kinds.append(np.zeros(n))
    # ---- sclera on the eyeball, outside the iris
    d = rng.normal(0, 1, (n_sclera * 3, 3))
    d /= np.linalg.norm(d, axis=1, keepdims=True)
    d = d[(d[:, 2] > 0.2) & (np.hypot(d[:, 0], d[:, 1]) * R_EYE > R_IRIS * 1.02)][:n_sclera]
    Ps.append(d * R_EYE)
    Cs.append(np.tile(SCLERA * 0.9, (len(d), 1)) * (0.7 + 0.3 * rng.random((len(d), 1))))
    kinds.append(np.ones(len(d)))
    # ---- skin around the eye (a gently curved mask surface), with the lid opening cut out
    x = rng.uniform(-0.030, 0.030, n_skin)
    y = rng.uniform(-0.020, 0.024, n_skin)
    lid_up = 0.0068 * np.sqrt(np.clip(1 - (x / 0.0135) ** 2, 0, 1)) + 0.0006
    lid_dn = -0.0056 * np.sqrt(np.clip(1 - (x / 0.0135) ** 2, 0, 1)) - 0.0004
    outside = (y > lid_up) | (y < lid_dn) | (np.abs(x) > 0.0135)
    x, y = x[outside], y[outside]
    z = 0.0112 - 0.9 * (x * x + y * y) / 0.06 + 0.0012 * np.exp(-((y - 0.0075) / 0.002) ** 2) * (np.abs(x) < 0.013)
    Ps.append(np.stack([x, y, z], 1))
    Cs.append(np.tile(SKIN, (len(x), 1)) * (0.5 + 0.5 * rng.random((len(x), 1))))
    kinds.append(np.full(len(x), 2))
    # ---- lashes along the upper lid (curling up/out toward the camera)
    u = rng.uniform(-1, 1, n_lash)
    base_x = u * 0.0132
    base_y = 0.0068 * np.sqrt(np.clip(1 - u ** 2, 0, 1)) + 0.0007
    L = rng.uniform(0.3, 1.0, n_lash) * (0.0045 + 0.0035 * (1 - np.abs(u)))
    s = rng.random(n_lash)
    Ps.append(np.stack([base_x + u * 0.0012 * s, base_y + L * s * 0.9 + (s * L) ** 2 * 40, 0.0118 + L * s * 0.5], 1))
    Cs.append(np.tile(LASH, (n_lash, 1)))
    kinds.append(np.full(n_lash, 3))
    P = np.concatenate(Ps).astype(np.float32)
    return dict(P=P, rgb=np.concatenate(Cs).astype(np.float32), kind=np.concatenate(kinds).astype(np.int8),
                key=hash01(np.arange(len(P)), 31).astype(np.float32))
