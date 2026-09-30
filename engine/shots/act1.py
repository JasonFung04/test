"""ACT I — THE SENDERS (I1–I9)."""
import numpy as np

from .. import timeline as TL
from ..assets import bodies, sky
from ..camera import Camera, ease
from ..post import Grade
from .common import dispatch, stars_layer

_C = {}


def giant():
    if "giant" not in _C:
        _C["giant"] = bodies.RedGiant()
    return _C["giant"]


def planet():
    if "planet" not in _C:
        _C["planet"] = bodies.Planet()
    return _C["planet"]


def field():
    if "field" not in _C:
        _C["field"] = sky.field()
    return _C["field"]


BREATH0 = TL.EVENTS["star_breath_first"]
GRADE_I = dict(wb=(1.0, 0.96, 0.9), sat=1.08, bloom=1.25, vignette=0.22, grain=0.006)


def star_radius(tg, R0=100.0):
    return R0 * (1 + 0.006 * np.cos(2 * np.pi * (tg - BREATH0) / 4.0))


# ---------------------------------------------------------------------------- I1
STAR_C = np.array([-47.5, 12.0, 0.0])
PLANET_C = np.array([26.5, -3.1, 111.0])
PLANET_R = 5.0


def I1(tl, tg, R, W, H):
    p = ease(tl / 7.0)
    cam = Camera(pos=(0.0, 0.0, 231.0 - 7.0 * p), target=(0.0, 0.0, 0.0), focal=35, W=W, H=H)
    g = giant()
    Rs = star_radius(tg)
    stars_layer(R, cam, field(), energy=0.8)
    bg = R.new_layer() * (1.0 - bodies.sphere_mask(cam, STAR_C, Rs * 1.004, W, H))[..., None]
    P, C = g.surface(tg, STAR_C, Rs, cam.pos, energy=2.6)
    R.draw(cam, P, C, size_px=1.1)
    P, C = g.corona(tg, STAR_C, Rs, cam.pos)
    R.draw(cam, P, C, size_px=3.0, soft=True)
    P, C = g.prominences(tg, STAR_C, Rs, cam=cam)
    R.draw(cam, P, C, size_px=1.4, soft=True)
    back = R.new_layer() + bg
    m = bodies.sphere_mask(cam, PLANET_C, PLANET_R, W, H)
    back *= (1.0 - m)[..., None]
    star_dir = STAR_C - PLANET_C
    star_dir /= np.linalg.norm(star_dir)
    pl = planet()
    P, C, _ = pl.lights(PLANET_C, PLANET_R, cam.pos, star_dir, energy=0.55, frac=0.035)
    R.draw(cam, P, C, size_px=0.5)
    P, C = pl.rim(PLANET_C, PLANET_R, cam.pos, star_dir, energy=1.4)
    R.draw(cam, P, C, size_px=0.8, soft=True)
    return back + R.resolve(), Grade(**GRADE_I)


TABLE = {"I1": I1}


def render(sid, tl, tg, R, W, H):
    return dispatch(TABLE, sid, tl, tg, R, W, H)
