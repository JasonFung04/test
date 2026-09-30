"""The canonical '#': the nine lines of the Blombos drawing.

ONE geometry drives three things: the ochre strokes on the silcrete flake (II5, II7, II8),
the avenues of the modern city (II9, Act III), and (loosely) the terminal '#'.
The II8 -> II9 match cut depends on both shots using these exact coordinates.

Canonical frame: 2-D, x right, y up, centred on the crossing field, roughly [-1,1] x [-0.62,0.62].
Six sub-parallel lines ('SIX') crossed obliquely by three slightly curved lines ('THREE'),
as described for the Blombos silcrete flake (Henshilwood et al., Nature 2018).
"""
import numpy as np

A6 = np.radians(72.0)     # direction of the six lines (from +x, counter-clockwise)
A3 = np.radians(10.0)     # direction of the three crossing lines
D6 = np.array([np.cos(A6), np.sin(A6)])
N6 = np.array([np.sin(A6), -np.cos(A6)])
D3 = np.array([np.cos(A3), np.sin(A3)])
N3 = np.array([-np.sin(A3), np.cos(A3)])

# per-line: offset along the normal, start/end along the direction, curvature (sagitta), wobble seed
SIX = [(-0.62, -0.66, 0.58, 0.012, 1), (-0.37, -0.70, 0.64, -0.008, 2), (-0.13, -0.72, 0.66, 0.010, 3),
       (0.11, -0.68, 0.70, -0.012, 4), (0.36, -0.64, 0.62, 0.006, 5), (0.60, -0.58, 0.52, -0.010, 6)]
THREE = [(-0.30, -1.02, 0.98, 0.035, 7), (0.02, -1.08, 1.04, 0.030, 8), (0.33, -0.96, 0.94, 0.040, 9)]


def _line(off, s0, s1, sag, seed, D, N, n=200, wobble=0.004):
    t = np.linspace(0.0, 1.0, n)
    s = s0 + (s1 - s0) * t
    bow = sag * 4.0 * t * (1 - t)
    rng = np.random.default_rng(seed)
    k = np.arange(1, 6)
    amp = rng.normal(0, wobble, 5) / k
    ph = rng.uniform(0, 2 * np.pi, 5)
    wob = (amp[None, :] * np.sin(2 * np.pi * k[None, :] * t[:, None] + ph[None, :])).sum(1)
    return (s[:, None] * D[None, :] + (off + bow + wob)[:, None] * N[None, :])


def lines(n=200, wobble=0.004):
    """List of 9 polylines (n,2): six first (drawing order), then three."""
    out = [_line(*p, D6, N6, n, wobble) for p in SIX]
    out += [_line(*p, D3, N3, n, wobble) for p in THREE]
    return out


def to_world(poly, scale, origin=(0.0, 0.0, 0.0), plane="xz"):
    """Map canonical 2-D coords to world. plane 'xz': canonical y -> world -z (so top-down view
    with camera up = -z reads the same way as the canonical frame)."""
    o = np.asarray(origin, float)
    P = np.zeros((poly.shape[0], 3))
    if plane == "xz":
        P[:, 0] = poly[:, 0] * scale
        P[:, 2] = -poly[:, 1] * scale
    else:  # 'xy'
        P[:, 0] = poly[:, 0] * scale
        P[:, 1] = poly[:, 1] * scale
    return P + o


# Match-cut contract: in the LAST frame of II8 and the FIRST frame of II9 the canonical frame
# is centred on screen, x to the right, y up, and 1 canonical unit spans MATCH_SPAN of the
# image WIDTH (at 1920 px -> 0.30 * 1920 = 576 px). Both shots must place their cameras so this holds.
MATCH_SPAN = 0.30
MATCH_ROLL = 0.0
