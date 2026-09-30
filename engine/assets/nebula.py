"""The nebula of II1: pillars, globules and a vast gas wall lit by the passing shell (a light echo).

Nebula frame: light travels along +x (the shell's centre is far away at -x), +y up; the camera
rides along the front at z ~ 0 looking toward -z. Every point has a precomputed emission colour,
amplitude and 'faces the light' factor; `light()` evaluates the echo per frame:

  ahead of the front  : unlit -- a faint ambient glow; dense pillars read as dark silhouettes
  at the front        : a thin flash of the messenger's gold
  behind the front    : magenta/teal emission that rises quickly and fades slowly
"""
import numpy as np

from ..color import hex_lin
from ..config import CACHE
from ..noise import fbm, hash01

MAGENTA = hex_lin("#D6336C")
TEAL = hex_lin("#2BB3A8")
GOLD = hex_lin("#FFD27A")
DEEP = hex_lin("#5A1636")
ROSE = hex_lin("#F07AA0")
STARW = hex_lin("#DDE8FF")
RIMGOLD = hex_lin("#FFC98A")

# (base xyz, tip xyz, base radius, neck radius, head radius) -- x relative to the front at t=0
PILLARS = [
    ((-70.0, -150.0, -185.0), (-50.0, 22.0, -172.0), 32.0, 15.0, 20.0),
    ((6.0, -160.0, -245.0), (24.0, 50.0, -236.0), 40.0, 18.0, 26.0),
    ((42.0, -135.0, -128.0), (54.0, -10.0, -134.0), 20.0, 9.0, 12.0),
    ((95.0, -175.0, -340.0), (80.0, 66.0, -332.0), 48.0, 24.0, 32.0),
    ((-150.0, -160.0, -310.0), (-130.0, 4.0, -298.0), 34.0, 16.0, 21.0),
    ((150.0, -150.0, -205.0), (141.0, -18.0, -212.0), 24.0, 12.0, 15.0),
]
VERSION = 6


def _frame(base, tip):
    ax = tip - base
    L = np.linalg.norm(ax)
    ax = ax / L
    e1 = np.cross(ax, [0.0, 0.0, 1.0])
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(ax, e1)
    return ax, L, e1, e2


def _radius(s, rb, rn, rh):
    r = rb + (rn - rb) * np.clip(s / 0.7, 0, 1) ** 0.8
    head = np.exp(-((s - 0.86) / 0.09) ** 2)
    r = r * (1 - head) + rh * head
    tipcap = np.sqrt(np.clip((1.0 - s) / 0.08, 0, 1))
    return r * np.where(s > 0.92, tipcap, 1.0)


def _pillar_points(base, tip, rb, rn, rh, seed, n, offset):
    """Points on (offset>0: outside, <0: inside) a lumpy pillar surface. Returns P, normals, s, ridge."""
    rng = np.random.default_rng(seed)
    base, tip = np.asarray(base, float), np.asarray(tip, float)
    ax, L, e1, e2 = _frame(base, tip)
    s = rng.random(n) ** 0.8
    # importance: more samples on the side facing the source (-x)
    th = rng.random(n) * 2 * np.pi
    d = np.cos(th)[:, None] * e1 + np.sin(th)[:, None] * e2
    lit = np.clip(-d[:, 0], 0, 1)
    keep = rng.random(n) < 0.25 + 0.75 * lit ** 0.7
    s, d = s[keep], d[keep]
    bend = (np.sin(s * 2.4 + seed) * 0.05)[:, None] * e1 * L
    c = base + ax * (s * L)[:, None] + bend
    r0 = _radius(s, rb, rn, rh)
    Pn = c + d * r0[:, None]
    lump = fbm(Pn * 0.028, octaves=5, offset=(seed * 3.3, 1.0, 2.0))
    fine = fbm(Pn * 0.16, octaves=3, offset=(2.0, seed * 1.7, 5.0))
    ridge = 1.0 - np.abs(fbm(Pn * 0.07, octaves=3, offset=(9.0, 4.0, seed * 0.9))) * 2.2
    r = r0 * (1.0 + 0.38 * lump + 0.14 * fine)
    P = c + d * (r + offset(len(s), r))[:, None]
    return P, d, s, np.clip(ridge, 0, 1)


def build(seed=5):
    path = CACHE / f"nebula_v{VERSION}_{seed}.npz"
    if path.exists():
        d = np.load(path)
        return {k: d[k] for k in d.files}
    rng = np.random.default_rng(seed)
    out = {k: [] for k in ("P", "col", "amp", "face", "kind", "size")}

    def add(P, col, amp, face, kind, size):
        n = len(P)
        out["P"].append(np.asarray(P, np.float32))
        out["col"].append(np.broadcast_to(np.asarray(col, np.float32), (n, 3)).astype(np.float32))
        out["amp"].append(np.broadcast_to(np.asarray(amp, np.float32), (n,)).astype(np.float32))
        out["face"].append(np.broadcast_to(np.asarray(face, np.float32), (n,)).astype(np.float32))
        out["kind"].append(np.full(n, kind, np.int8))
        out["size"].append(np.broadcast_to(np.asarray(size, np.float32), (n,)).astype(np.float32))

    # ------------------------------------------------------------------ pillars
    for k, (b, t, rb, rn, rh) in enumerate(PILLARS):
        sc = (rb / 40.0) ** 1.5
        sd = seed + k * 11
        # ionised skin: thin, bright where it faces the source, brightest on ridges
        P, Nn, S, Rg = _pillar_points(b, t, rb, rn, rh, sd, int(260_000 * sc),
                                      lambda n, r: -np.abs(rng.normal(0, 0.012, n)) * r)
        face = np.clip(-Nn[:, 0] * 0.95 + 0.12 * Nn[:, 1], 0, 1) ** 1.8 * (0.35 + 0.65 * Rg)
        hue = np.clip(0.5 + 1.2 * fbm(P * 0.03, octaves=2, offset=(k, 0, 0)), 0, 1)[:, None]
        col = MAGENTA * (0.45 + 0.55 * hue) + ROSE * (1 - hue) * 0.35
        edge = (np.clip(-Nn[:, 0], 0, 1) ** 6 * Rg)[:, None]              # the very rim turns gold
        col = col * (1 - 0.7 * edge) + RIMGOLD * 0.9 * edge
        amp = (0.35 + 1.1 * rng.random(len(P)) ** 3) * (1.0 + 0.9 * np.exp(-((S - 0.86) / 0.07) ** 2))
        add(P, col, amp, face, 1, 1.4)
        # soft halo of evaporating gas just outside the lit skin
        P, Nn, S, Rg = _pillar_points(b, t, rb, rn, rh, sd + 1, int(90_000 * sc),
                                      lambda n, r: rng.exponential(0.10, n) * r)
        face = np.clip(-Nn[:, 0], 0, 1) ** 1.5
        add(P, MAGENTA * 0.6 + TEAL * 0.25, 0.30, face, 3, 3.5)
        # dark dusty core (occluder only, a faint deep glow)
        P, Nn, S, Rg = _pillar_points(b, t, rb, rn, rh, sd + 2, int(60_000 * sc),
                                      lambda n, r: -rng.random(n) ** 0.7 * r * 0.9)
        add(P, DEEP, 0.05, 0.05, 2, 3.0)
        # an embedded young star in the head
        head = np.asarray(b) + (np.asarray(t) - np.asarray(b)) * 0.84
        add(head[None] + rng.normal(0, 2.0, (1, 3)), STARW, 4.0, 1.0, 4, 0.6)

    # ------------------------------------------------------------------ the vast gas wall (behind)
    # (a) smooth large-scale glow: big soft sprites following a low-frequency density
    n = 900_000
    cand = np.stack([rng.uniform(-560, 560, n), rng.uniform(-260, 260, n), rng.uniform(-640, -430, n)], 1)
    env = fbm(cand * 0.0030, octaves=4, offset=(2.0, 9.0, 4.0))
    p = np.clip(env * 1.5 + 0.40, 0, 1) ** 2.0
    keep = rng.random(n) < p * 0.5
    Pg = cand[keep][:220_000]
    hue = np.clip(0.55 + 1.6 * fbm(Pg * 0.0045, octaves=3, offset=(5.0, 5.0, 5.0)), 0, 1)[:, None]
    add(Pg, TEAL * hue + MAGENTA * (1 - hue) * 0.8, 0.55, 1.0, 0, 7.0)
    # (b) filaments and sheets: ridged noise, crisp small dots
    n = 3_600_000
    cand = np.stack([rng.uniform(-560, 560, n), rng.uniform(-260, 260, n), rng.uniform(-620, -420, n)], 1)
    env = fbm(cand * 0.0030, octaves=4, offset=(2.0, 9.0, 4.0))
    rid = 1.0 - np.abs(fbm(cand * np.array([0.008, 0.011, 0.008]), octaves=4, offset=(7.0, 1.0, 3.0))) * 2.0
    p = np.clip(env * 1.5 + 0.40, 0, 1) ** 1.2 * np.clip(rid, 0, 1) ** 6.0
    keep = rng.random(n) < np.clip(p * 1.6, 0, 1)
    Pg = cand[keep][:650_000]
    hue = np.clip(0.55 + 1.6 * fbm(Pg * 0.0045, octaves=3, offset=(5.0, 5.0, 5.0)), 0, 1)[:, None]
    col = TEAL * hue + MAGENTA * (1 - hue) * 0.8
    amp = 0.22 + 0.40 * rng.random(len(Pg)) ** 2
    add(Pg, col, amp, 1.0, 0, 1.6)

    # ------------------------------------------------------------------ mid-depth veils between pillars
    n = 1_200_000
    cand = np.stack([rng.uniform(-420, 420, n), rng.uniform(-200, 120, n), rng.uniform(-240, -90, n)], 1)
    rid = 1.0 - np.abs(fbm(cand * np.array([0.012, 0.02, 0.012]), octaves=4, offset=(3.0, 6.0, 1.0))) * 2.0
    env = fbm(cand * 0.006, octaves=2, offset=(8.0, 2.0, 6.0))
    p = np.clip(rid, 0, 1) ** 5.0 * np.clip(env * 1.8 + 0.2, 0, 1)
    keep = rng.random(n) < p
    Pm = cand[keep][:260_000]
    hue = hash01(np.arange(len(Pm)), 5)[:, None]
    add(Pm, TEAL * (0.6 + 0.4 * hue) + MAGENTA * 0.25, 0.30, 1.0, 5, 1.8)

    # ------------------------------------------------------------------ foreground wisps (parallax)
    n = 400_000
    cand = np.stack([rng.uniform(-260, 260, n), rng.uniform(-70, 60, n), rng.uniform(-80, -16, n)], 1)
    rid = 1.0 - np.abs(fbm(cand * np.array([0.02, 0.035, 0.02]), octaves=3, offset=(1.0, 3.0, 8.0))) * 2.0
    keep = rng.random(n) < np.clip(rid, 0, 1) ** 6 * 0.5
    Pw = cand[keep][:50_000]
    add(Pw, TEAL * 0.8 + MAGENTA * 0.2, 0.10, 1.0, 6, 3.0)

    d = {k: np.concatenate(v) for k, v in out.items()}
    np.savez(path, **d)
    return d


def shadow_mask(P):
    """Transmission (0..1) for gas downstream (+x) of a pillar, i.e. in its shadow."""
    T = np.ones(len(P), np.float32)
    for (b, t, rb, rn, rh) in PILLARS:
        b, t = np.asarray(b), np.asarray(t)
        s = np.clip((P[:, 1] - b[1]) / (t[1] - b[1]), 0, 1)
        c = b + (t - b) * s[:, None]
        r = (rb + (rn - rb) * s) * 0.9
        dz = np.abs(P[:, 2] - c[:, 2])
        inside = np.clip(1.0 - dz / r, 0, 1) * (P[:, 1] < t[1] + rh * 0.5)
        down = np.clip((P[:, 0] - c[:, 0] - r * 0.5) / (r * 0.8), 0, 1)
        T *= (1.0 - 0.85 * inside * down).astype(np.float32)
    return T


def light(d, r_pt, R_front, ambient=0.10, flash_len=4.0, rise=5.0, fade=190.0, shadow=None):
    """Emission (N,3) of the light echo. r_pt: distance of each point from the shell centre."""
    dlt = R_front - r_pt                                   # >0: the front has passed
    passed = dlt > 0
    x = np.maximum(dlt, 0.0)
    flash = np.exp(-x / flash_len) * passed
    glow = (1.0 - np.exp(-x / rise)) * (0.25 * np.exp(-x / 40.0) + 0.75 * np.exp(-x / fade)) * passed
    face = d["face"]
    kind = d["kind"]
    tr = 1.0 if shadow is None else shadow
    lit = glow * face * tr
    fl = flash * (0.2 + 0.8 * face) * tr
    amb = ambient * ((kind == 0) + 0.6 * (kind == 5) + 0.15 * (kind == 1) * face)
    e = d["amp"][:, None] * (d["col"] * (lit + amb)[:, None] + GOLD[None, :] * (2.4 * fl)[:, None])
    return e.astype(np.float32)
