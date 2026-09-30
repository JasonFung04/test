"""ACT III — the girl on her rooftop: III2a, III2b, III4, III7a, III7b, III10, III12.

Everything in city-world coordinates (+x east, +y up, +z south). The rooftop set and the girl are
drawn here (opaque, draw_solid); the sky and the city come from assets/city.render_env."""
import numpy as np

from .. import timeline as TL
from ..assets import rooftop as RT
from ..assets.figures import Girl
from ..camera import Camera, ease, ease5, ease_in, ease_out, handheld, lerp, seg
from ..color import hex_lin
from ..post import Grade
from ..solid import Light, SolidCloud, draw_solid
from .common import dispatch

E = TL.EVENTS
TORCH = hex_lin("#F4F8FF")
GOLD = hex_lin("#FFD27A")
_C = {}


def city():
    from ..assets import city as C
    return C


def girl():
    if "girl" not in _C:
        _C["girl"] = Girl()
    return _C["girl"]


def roof_world():
    if "roof" not in _C:
        C = city()
        _C["roof"] = RT.cloud(C.ROOF_ORIGIN, C.ROOF_YAW)
    return _C["roof"]


def W_(p_local):
    C = city()
    return RT.to_world(np.asarray(p_local, float), C.ROOF_ORIGIN, C.ROOF_YAW)


def Rw():
    return RT.yaw_matrix(city().ROOF_YAW)


def girl_world(pose_kw):
    """Posed girl as a SolidCloud in world coordinates + info (eyes, torch tip/dir in world)."""
    gc, gi = girl().pose(**pose_kw)
    Rm = Rw()
    seat = W_(RT.GIRL_SEAT)
    cloud = gc.transformed(Rm, seat)
    info = dict(gi)
    info["eyes"] = [Rm @ e + seat for e in gi["eyes"]]
    if "torch_tip" in gi:
        info["torch_tip"] = Rm @ gi["torch_tip"] + seat
        info["torch_dir"] = Rm @ gi["torch_dir"]
    return cloud, info


def framed(cam_pos, look, point, sx, sy, W, H, **kw):
    """Camera looking along (look - cam_pos) with a lens shift that puts `point` at (sx*W, sy*H)."""
    c0 = Camera(cam_pos, look, W=W, H=H, **kw)
    x, y, _, _, _ = c0.project(np.asarray(point)[None])
    dx = (sx * W - x[0]) / c0.s
    dy = -(sy * H - y[0]) / c0.s
    kw2 = dict(kw)
    return Camera(cam_pos, look, W=W, H=H, shift=(dx, dy), **kw2)


def env_lights(tg):
    C = city()
    pw = C.power(tg)
    glow = pw["glow"]
    lamp = pw["lamp"]
    L = [Light("dir", (1.0, 0.55, 0.22), 0.55 * glow, vec=(0.3, 0.15, 1.0)),
         Light("dir", (1.0, 0.58, 0.28), 0.40 * glow, vec=(0.0, 1.0, 0.0)),
         Light("dir", (0.55, 0.65, 1.0), 0.10 * pw["stars"], vec=(0.2, 0.8, 0.4)),
         Light("amb", (0.55, 0.33, 0.18), 0.07 * glow), Light("amb", (0.25, 0.3, 0.5), 0.006)]
    if lamp > 0:
        L.append(Light("point", (1.0, 0.78, 0.5), 0.35 * lamp, vec=W_(RT.LAMP_POS + np.array([0, 0, 0.12])),
                       radius=0.3))
    return L, pw


def star_dir():
    from .climax import gold_star_dir
    return gold_star_dir()


def beam(R, cam, tip, direction, level, length=60.0, n=110_000, gain=2.5, spread=0.05):
    from .epilogue import beam_points
    Pb, Eb = beam_points(tip, direction, level, length=length, n=n, spread=spread)
    R.draw(cam, Pb, Eb * gain, size_px=0.8)
    if level > 0:
        R.draw(cam, np.asarray(tip)[None], TORCH[None] * (20.0 * level), size_px=3.0, soft=True)


def compose(R, cam, tg, W, H, cloud, extra_draw=None, env_kw=None, rim=((1.0, 0.6, 0.3), 3.0, 0.3)):
    """Sky + city (render_env) behind, rooftop + girl (opaque) in front, extras (beam...) on top."""
    C = city()
    env, m_env = C.render_env(R, cam, tg, W, H, roof=False, girl=False, **(env_kw or {}))
    L, pw = env_lights(tg)
    m = draw_solid(R, cam, SolidCloud.concat([roof_world(), cloud]), L, rim=rim, spacing_px=2.0, return_mask=True,
                   flecks=((0.3, 0.3, 0.8), 0.04), seed=3)
    solid = R.new_layer()
    if pw["lamp"] > 0:
        lp = W_(RT.LAMP_POS)
        R.draw(cam, lp[None], np.array([[6.0, 4.6, 2.8]]) * pw["lamp"], size_px=1.6)
        R.draw(cam, lp[None], np.array([[0.5, 0.35, 0.18]]) * pw["lamp"], size_px=30, soft=True)
    if extra_draw is not None:
        extra_draw(R, cam)
    return env * (1 - m)[..., None] + solid + R.resolve()


GRADE_CITY = dict(wb=(1.0, 0.96, 0.9), sat=1.0, bloom=1.2, vignette=0.25, grain=0.006)
GRADE_DARK = dict(wb=(0.92, 0.97, 1.08), sat=0.95, bloom=1.2, vignette=0.28, grain=0.006)


# ---------------------------------------------------------------------------- III2a / III12 (static wide)
def wide_cam(W, H):
    # from the far corner of the roof, standing on the parapet ledge: she sits small on the tank,
    # the huge sky above her (Ozu: locked off, horizon low)
    return Camera(W_([9.5, 3.4, 5.4]), W_([-6.0, 6.6, -2.6]), focal=35, W=W, H=H)


def torch_state_2a(tg):
    """Unreliable torch in III2a/b: two failed clicks, two taps, flicker, on; off at torch_off."""
    on = E["torch_on"]
    off = E["torch_off"]
    if tg < on - 0.25 or tg > off + 0.05:
        return 0.0
    x = tg - on
    lv = float(np.interp(x, [-0.25, -0.2, -0.14, -0.08, 0.0, 0.04, 0.1], [0, 0.6, 0.0, 0.3, 0.95, 0.6, 1.0]))
    if tg > off:
        lv *= max(0.0, 1 - (tg - off) / 0.05)
    return lv


def pose_2(tg):
    """Girl pose through III2a/III2b: fiddling with the torch, then pointing it at the sky."""
    sd_l = Rw().T @ star_dir()
    taps = E["torch_tap"]
    tap = sum(np.exp(-((tg - t) / 0.07) ** 2) for t in taps)
    raise_k = ease5(seg(tg, E["torch_on"] + 0.2, E["torch_on"] + 1.3))
    look_up = ease5(seg(tg, E["torch_on"] + 0.4, E["torch_on"] + 1.6))
    lower = ease5(seg(tg, E["torch_off"] + 0.3, E["torch_off"] + 1.5))
    aim = lerp(np.array([0.1, -0.2, 1.0]), sd_l, raise_k * (1 - 0.6 * lower))
    armR = (lerp(0.45, 1.75, raise_k * (1 - 0.7 * lower)) - 0.35 * tap, 0.15, lerp(1.45, 0.35, raise_k * (1 - 0.7 * lower)))
    armL = (0.55 + 0.25 * min(tap, 1), 0.25, 1.55)
    return dict(head_pitch=lerp(-0.25, 0.62, look_up) - 0.08 * tap, head_yaw=0.0, armL=armL, armR=armR,
                torch_hand="R", torch_aim=aim)


def III2a(tl, tg, R, W, H):
    cam = wide_cam(W, H)
    cloud, info = girl_world(pose_2(tg))
    lv = torch_state_2a(tg)

    def extra(R, cam):
        if lv > 0:
            beam(R, cam, info["torch_tip"], info["torch_dir"], lv, length=45.0, gain=1.6, spread=0.06)
    return compose(R, cam, tg, W, H, cloud, extra), Grade(**GRADE_CITY)


def III2b(tl, tg, R, W, H):
    # (rim light from the sky glow: she reads as a dark shape edged in orange against the haze)
    # MCU from behind-left, three-quarter: she looks up at nothing; the beam into the haze
    seat = W_(RT.GIRL_SEAT)
    p = ease(seg(tg, 138.5, 145.0))
    head = seat + Rw() @ np.array([0.0, 0.52, 0.0])
    # low behind her right shoulder, looking up past her head at the sky she is looking at
    cam_pos = seat + Rw() @ np.array([-0.70, 0.30 + 0.02 * p, -1.05]) + handheld(tg, 0.004, seed=12)
    look = seat + Rw() @ np.array([0.35, 1.30, 1.40])
    cam = framed(cam_pos, look, head, 0.30, 0.74, W, H, focal=32, focus=float(np.linalg.norm(cam_pos - head)),
                 bokeh=6)
    cloud, info = girl_world(pose_2(tg))
    lv = torch_state_2a(tg)

    def extra(R, cam):
        if lv > 0:
            beam(R, cam, info["torch_tip"], info["torch_dir"], lv, length=70.0, gain=3.0)
    return compose(R, cam, tg, W, H, cloud, extra), Grade(**GRADE_CITY)


def III12(tl, tg, R, W, H):
    cam = wide_cam(W, H)
    sd_l = Rw().T @ star_dir()
    cloud, info = girl_world(dict(head_pitch=0.62, armL=(0.45, 0.2, 1.5), armR=(0.5, 0.15, 1.3), torch_hand="R",
                                  torch_aim=np.array([0.2, -0.4, 1.0])))
    return compose(R, cam, tg, W, H, cloud), Grade(**GRADE_CITY)


# ---------------------------------------------------------------------------- III4 (near black)
def III4(tl, tg, R, W, H):
    seat = W_(RT.GIRL_SEAT)
    cloud, info = girl_world(dict(head_pitch=0.35, armL=(0.45, 0.2, 1.5), armR=(0.45, 0.2, 1.5), torch_hand="R"))
    eye = info["eyes"][0]
    cam_pos = eye + Rw() @ np.array([0.85, -0.10, 0.25])
    cam = Camera(cam_pos, eye + Rw() @ np.array([0.0, 0.02, 0.1]), focal=50, W=W, H=H)
    return compose(R, cam, tg, W, H, cloud, rim=((0.35, 0.42, 0.7), 3.0, 0.22)), Grade(**GRADE_DARK)


# ---------------------------------------------------------------------------- III7a / III10 (profile CU)
def profile_cam(info, W, H, dist=0.95, focal=85, drift=0.0, tg=0.0):
    eye = info["eyes"][0]
    side = Rw() @ np.array([1.0, 0.0, 0.0])
    cam_pos = eye + side * dist + Rw() @ np.array([0.0, -0.12, 0.10]) + handheld(tg, 0.003, seed=21)
    return Camera(cam_pos, eye + Rw() @ np.array([0.0, 0.0, -0.06 + drift]), focal=focal, W=W, H=H,
                  focus=dist, bokeh=24)


def III7a(tl, tg, R, W, H):
    stop = ease5(seg(tg, 176.6, 177.4))
    cloud, info = girl_world(dict(head_pitch=lerp(0.55, 0.66, stop), head_yaw=lerp(0.06, -0.02, stop),
                                  armL=(0.45, 0.2, 1.5), armR=(0.45, 0.2, 1.5), torch_hand="R"))
    cam = profile_cam(info, W, H, tg=tg)
    return compose(R, cam, tg, W, H, cloud, rim=((0.55, 0.68, 1.0), 3.0, 1.1)), Grade(**GRADE_DARK)


def III7b(tl, tg, R, W, H):
    """Her POV, telephoto: the gold star lights up beside the Milky Way and trembles."""
    sd = star_dir()
    cloud, info = girl_world(dict(head_pitch=0.66, armL=(0.45, 0.2, 1.5), armR=(0.45, 0.2, 1.5), torch_hand="R"))
    eye = info["eyes"][0]
    cam = Camera(eye + handheld(tg, 0.02, seed=81), eye + sd * 1000.0 + np.array([0, -60.0, 0]), focal=135, W=W, H=H)
    C = city()
    env, m = C.render_env(R, cam, tg, W, H, city=False, roof=False, girl=False)
    on = ease(seg(tg, E["gold_star_on"], E["gold_star_on"] + 0.5))
    if on > 0:
        from .climax import messenger
        tw = 0.75 + 0.25 * np.sin(tg * 9.0) * np.sin(tg * 5.3)
        messenger(R, cam, eye + sd * 1e7, 0.10 * on * tw, tg, spikes=False)
    return env + R.resolve(), Grade(**GRADE_DARK)


def III10(tl, tg, R, W, H):
    b = E["blink"]
    blink = float(np.interp(tg, [b - 0.08, b, b + 0.06, b + 0.16], [0, 1, 1, 0]))
    cloud, info = girl_world(dict(head_pitch=0.66, armL=(0.45, 0.2, 1.5), armR=(0.45, 0.2, 1.5), torch_hand="R",
                                  blink=blink))
    cam = profile_cam(info, W, H, dist=0.85, tg=tg)
    eye = info["eyes"][0]

    def extra(R, cam):
        # the last trace of gold in her eye fades after the blink
        k = (1 - blink) * (1 - ease(seg(tg, b, b + 1.2)))
        if k > 0:
            R.draw(cam, eye[None], GOLD[None] * (1.5 * k), size_px=0.9)
    return compose(R, cam, tg, W, H, cloud, extra, rim=((0.55, 0.68, 1.0), 3.0, 1.1)), Grade(**GRADE_DARK)


TABLE = {"III2a": III2a, "III2b": III2b, "III4": III4, "III7a": III7a, "III7b": III7b, "III10": III10,
         "III12": III12}


def render(sid, tl, tg, R, W, H):
    return dispatch(TABLE, sid, tl, tg, R, W, H)
