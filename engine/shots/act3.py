"""ACT III — ARRIVAL: the city shots (II9, III1, III3, III5, III6, III11).

The megacity, its blackout and the return of power live in assets/city.py; this module only places
cameras and grades.  (III2a/b, III4, III7-III10, III12 are in act3_girl.py / climax.py.)

II9   top-down match cut on the '#': the nine avenues are the canonical crosshatch (MATCH_SPAN contract)
III1  24mm tilt from straight down to the horizon: river, elevated roads, the orange lid of sky
III3  35mm locked, high vantage behind the girl's block: substation flash, rings of darkness, her bulb
III5  18mm from her roof, low: the sky surfaces from depth (dark adaptation), the Milky Way overhead
III6  50mm lateral truck across rooftops: hundreds of people looking up; one phone lights a face
III11 same camera as III3: the tide of light rolls back in; the stars drown
"""
import numpy as np

from .. import timeline as TL
from ..assets import city as C
from ..assets import crosshatch as X
from ..camera import Camera, ease, ease5, ease_in, handheld, lerp, seg
from ..post import Grade
from .common import dispatch

EV = TL.EVENTS
GRADE_CITY = dict(wb=(1.0, 0.95, 0.88), sat=1.0, bloom=1.1, vignette=0.22, grain=0.006)
GRADE_DARK = dict(wb=(0.94, 0.97, 1.06), sat=1.0, bloom=1.15, vignette=0.24, grain=0.005)


def _mix_grade(a, b, t):
    t = float(np.clip(t, 0, 1))
    out = {}
    for k in a:
        va, vb = np.asarray(a[k], float), np.asarray(b[k], float)
        v = va + (vb - va) * t
        out[k] = tuple(v) if v.ndim else float(v)
    return Grade(**out)


# ============================================================================ II9 — the match cut
# Contract (crosshatch.py): first frame = canonical frame centred, x right, y up, 1 unit = MATCH_SPAN*W.
# Top-down, screen-up = north (-z): canonical y -> -z (crosshatch.to_world 'xz').
LAMP_H = 11.0                                         # the avenues' lamp heads sit at the match plane


def ii9_height(focal=35.0):
    fpx1920 = focal / 36.0 * 1920.0
    return fpx1920 * C.SCALE / (X.MATCH_SPAN * 1920.0) + LAMP_H


def II9(tl, tg, R, W, H):
    h0 = ii9_height()
    p = ease_in(tl / 6.0, 2.0)
    h = h0 * (1.0 - 0.055 * p)                      # an extremely slow push (descent), centred
    cam = Camera((0.0, h, 0.0), (0.0, 0.0, 0.0), up=(0.0, 0.0, -1.0), focal=35, W=W, H=H)
    hdr, _ = C.render_env(R, cam, tg, W, H, sky=False, stars=False, roof=False)
    return hdr, Grade(**dict(GRADE_CITY, exposure=1.3, bloom=1.35))


# ============================================================================ III1 — the city tilt
III1_POS0 = np.array([2750.0, 820.0, -3350.0])
III1_POS1 = np.array([2700.0, 960.0, -3500.0])
III1_AZ = np.radians(190.0)


def III1(tl, tg, R, W, H):
    u = seg(tl, 0.0, 7.0)
    p = ease5(u)
    pos = lerp(III1_POS0, III1_POS1, ease(u))
    pitch = np.radians(lerp(-89.5, -7.5, p))
    fwd = np.array([np.sin(III1_AZ) * np.cos(pitch), np.sin(pitch), -np.cos(III1_AZ) * np.cos(pitch)])
    up = np.array([np.sin(III1_AZ) * -np.sin(pitch), np.cos(pitch), -np.cos(III1_AZ) * -np.sin(pitch)])
    cam = Camera(pos, pos + fwd * 100.0, up=up, focal=24, W=W, H=H)
    hdr, _ = C.render_env(R, cam, tg, W, H, stars=False, roof=False)
    return hdr, Grade(**GRADE_CITY)


# ============================================================================ III3 / III11 — the blackout
III3_BEARING_TO_GIRL = 199.0
III3_DIST = 34.0
III3_HEIGHT = 32.3
III3_VIEW_BEARING = 183.0
III3_PITCH = -5.0


def iii3_camera(W, H):
    g = C.GIRL_WORLD
    b = np.radians(III3_BEARING_TO_GIRL)
    pos = g + III3_DIST * np.array([-np.sin(b), 0.0, np.cos(b)])
    pos[1] = III3_HEIGHT
    v = np.radians(III3_VIEW_BEARING)
    tgt = pos + np.array([np.sin(v), np.tan(np.radians(III3_PITCH)), -np.cos(v)]) * 100.0
    return Camera(pos, tgt, focal=35, W=W, H=H)


GIRL_III3 = dict(head_pitch=0.30, head_yaw=-0.08)


def III3(tl, tg, R, W, H):
    cam = iii3_camera(W, H)
    pw = C.power(tg)
    hdr, _ = C.render_env(R, cam, tg, W, H, pw=pw, girl=True, girl_pose=GIRL_III3)
    dark = 1.0 - pw["glow"]
    return hdr, _mix_grade(GRADE_CITY, GRADE_DARK, dark)


def III11(tl, tg, R, W, H):
    cam = iii3_camera(W, H)
    pw = C.power(tg)
    hdr, _ = C.render_env(R, cam, tg, W, H, pw=pw, girl=True, girl_pose=dict(head_pitch=0.62, head_yaw=-0.1))
    dark = 1.0 - pw["glow"]
    return hdr, _mix_grade(GRADE_CITY, GRADE_DARK, dark)


# ============================================================================ III5 — the sky surfaces
def iii5_camera(tl, W, H):
    # low on the roof deck near the south parapet, looking south over it; a very slow tilt up
    base = C.rooftop.to_world(np.array([-4.6, 1.35, 3.0]), C.ROOF_ORIGIN, C.ROOF_YAW)
    # linger low while the sky surfaces (the core of the Milky Way rises over the skyline), then follow
    # the band up to the zenith
    el = np.radians(5.0 + 10.0 * ease(tl / 7.5) + 63.0 * ease5(float(np.clip((tl - 5.0) / 8.0, 0, 1))))
    az = np.radians(186.0)
    fwd = np.array([np.sin(az) * np.cos(el), np.sin(el), -np.cos(az) * np.cos(el)])
    hh = handheld(tl, 0.012, seed=35)
    pos = base + hh * 0.3
    return Camera(pos, pos + fwd * 50.0 + hh, focal=18, W=W, H=H)


def III5(tl, tg, R, W, H):
    cam = iii5_camera(tl, W, H)
    hdr, _ = C.render_env(R, cam, tg, W, H)
    return hdr, Grade(**GRADE_DARK)


# ============================================================================ III6 — the people on the roofs
def iii6_camera(tl, W, H):
    a = C.rooftop.to_world(np.array([4.5, 1.55, 5.2]), C.ROOF_ORIGIN, C.ROOF_YAW)
    b = C.rooftop.to_world(np.array([9.5, 1.55, 5.2]), C.ROOF_ORIGIN, C.ROOF_YAW)
    u = ease(seg(tl, 0.0, 4.5))
    pos = lerp(a, b, u) + handheld(tl, 0.006, seed=36)
    az = np.radians(163.0)
    el = np.radians(9.0)
    fwd = np.array([np.sin(az) * np.cos(el), np.sin(el), -np.cos(az) * np.cos(el)])
    return Camera(pos, pos + fwd * 120.0, focal=50, W=W, H=H, focus=110.0, bokeh=3.0)


def III6(tl, tg, R, W, H):
    cam = iii6_camera(tl, W, H)
    hdr, _ = C.render_env(R, cam, tg, W, H, crowd=True)
    return hdr, Grade(**GRADE_DARK)


TABLE = {"II9": II9, "III1": III1, "III3": III3, "III5": III5, "III6": III6, "III11": III11}


def render(sid, tl, tg, R, W, H):
    return dispatch(TABLE, sid, tl, tg, R, W, H)
