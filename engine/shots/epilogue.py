"""EPILOGUE (E1–E3), BLACK, CREDITS."""
import numpy as np

from .. import timeline as TL
from ..assets import earth as EA
from ..assets import sky
from ..assets.bodies import sphere_mask
from ..assets.figures import girl_torch_hand
from ..camera import Camera, ease, ease5, ease_in, ease_out, handheld, lerp, seg
from ..color import hex_lin
from ..noise import hash01
from ..post import Grade
from ..solid import Light, SolidCloud, draw_solid
from ..text import draw_line, glow_composite
from .common import dispatch

E = TL.EVENTS
TORCH = hex_lin("#F4F8FF")
_C = {}


def star_dir():
    from .climax import gold_star_dir
    return gold_star_dir()


# ---------------------------------------------------------------------------- flashlight state
def torch_level(tg):
    """0..1 output of the (unreliable) flashlight in the epilogue."""
    on = E["torch_on_final"]
    if tg < on - 0.35:
        return 0.0
    x = tg - on
    # a stutter before it holds: flicker, dip, then steady
    return float(np.interp(x, [-0.35, -0.3, -0.22, -0.16, -0.08, 0.0, 0.05, 0.12, 0.3],
                           [0.0, 0.5, 0.0, 0.25, 0.0, 0.9, 0.55, 1.0, 1.0]))


def beam_points(tip, direction, level, length=90.0, n=90_000, seed=3, haze=1.0, spread=0.055):
    """Scattering points along a flashlight cone in hazy air."""
    if level <= 0:
        return np.zeros((0, 3), np.float32), np.zeros((0, 3), np.float32)
    rng = np.random.default_rng(seed)
    d = np.asarray(direction, float)
    d /= np.linalg.norm(d)
    a = np.cross(d, [0, 1.0, 0])
    if np.linalg.norm(a) < 1e-6:
        a = np.cross(d, [1.0, 0, 0])
    a /= np.linalg.norm(a)
    b = np.cross(d, a)
    s = length * rng.random(n) ** 1.8
    r = spread * s * np.sqrt(rng.random(n)) + 0.006
    t = rng.random(n) * 2 * np.pi
    P = tip + np.outer(s, d) + np.outer(r * np.cos(t), a) + np.outer(r * np.sin(t), b)
    # brightness: bright core near the lens, fading with distance (spreading + haze extinction)
    w = np.exp(-s / (length * 0.35)) / (1.0 + (s / 4.0) ** 1.2)
    core = np.exp(-(r / (spread * s + 0.006)) ** 2 * 2.0)
    E_ = TORCH * (level * haze * 0.9 * w * (0.35 + 0.65 * core))[:, None]
    return P.astype(np.float32), E_.astype(np.float32)


# ---------------------------------------------------------------------------- E1: the hand
def hand_cloud():
    if "hand" not in _C:
        d = girl_torch_hand()
        skin = np.array([0.50, 0.33, 0.25], np.float32)
        sleeve = np.array([0.80, 0.56, 0.09], np.float32)
        metal = np.array([0.55, 0.57, 0.60], np.float32)
        parts = [SolidCloud(d["skin_P"], d["skin_N"], skin, area=d["skin_a"]),
                 SolidCloud(d["torch_P"], d["torch_N"], metal, area=d["torch_a"]),
                 SolidCloud(d["sleeve_P"], d["sleeve_N"], sleeve, area=d["sleeve_a"])]
        _C["hand"] = SolidCloud.concat(parts)
        _C["lens"] = (d["lens_P"], d["lens_N"])
    return _C["hand"], _C["lens"]


def _rot_to(v_from, v_to):
    a = np.asarray(v_from, float)
    b = np.asarray(v_to, float)
    a /= np.linalg.norm(a)
    b /= np.linalg.norm(b)
    v = np.cross(a, b)
    c = float(a @ b)
    if np.linalg.norm(v) < 1e-9:
        return np.eye(3)
    K = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    return np.eye(3) + K + K @ K / (1 + c)


def E1(tl, tg, R, W, H):
    hc, (lensP, lensN) = hand_cloud()
    sd = star_dir()
    # the hand rises (217.8 -> 219.3) from a lazy angle to aiming at the star; a jolt at the tap
    raise_k = ease5(seg(tg, E["torch_raise"], E["torch_raise"] + 1.5))
    aim0 = np.array([0.35, 0.35, 0.85])
    aim = lerp(aim0 / np.linalg.norm(aim0), sd, raise_k)
    aim /= np.linalg.norm(aim)
    tap = np.exp(-max(tg - E["torch_tap_final"], 0) / 0.08) * (tg >= E["torch_tap_final"])
    Rm = _rot_to([0, 1.0, 0], aim)
    # roll so the palm side faces the camera side
    base = np.array([0.0, -0.10, 0.0]) * (1 - raise_k) + np.array([0, -0.012, 0]) * tap
    P = hc.P @ Rm.T + base
    N = hc.N @ Rm.T
    cloud = SolidCloud(P, N, hc.albedo, area=hc.area, key=hc.key)
    tip = base + Rm @ np.array([0, 0.061, 0])
    # camera: hand-rhyme framing -- side-on, a little below, the hand in the lower-right third and the
    # torch pointing up-left toward the sky; 100mm, focus on the fingers
    focus_pt = base + Rm @ np.array([0.004, -0.028, 0.0])
    side = np.cross(aim, [0, 1.0, 0])
    side /= np.linalg.norm(side) + 1e-9
    cam_pos = focus_pt - side * 0.62 - aim * 0.16 + np.array([0.0, -0.10, 0.0]) + handheld(tg, 0.004, seed=7)
    tilt = ease_in(seg(tg, E["torch_on_final"] + 0.6, 224.0), 1.5)
    tgt = lerp(focus_pt, tip + aim * 3.0, tilt)
    cam = Camera(cam_pos, tgt, focal=lerp(100, 50, tilt), W=W, H=H,
                 focus=float(np.linalg.norm(focus_pt - cam_pos)), bokeh=55,
                 shift=(0.15 * 1920 * (1 - tilt), -0.12 * 804 * (1 - tilt)))
    # background: the city's glowing sky, defocused, with a few out-of-focus city lights low in frame
    yy = np.linspace(0, 1, H, dtype=np.float32)[:, None, None]
    env = (hex_lin("#3A2412") * (0.20 + 0.25 * yy) + hex_lin("#1A0E08") * 0.1).astype(np.float32) * np.ones((1, W, 1),
                                                                                                            np.float32)
    rng = np.random.default_rng(71)
    nb = 60
    bx = rng.uniform(-0.2, 1.2, nb) * W
    by = (0.80 + rng.uniform(0, 0.35, nb)) * H - tilt * 0.9 * H
    bc = np.where(rng.random(nb)[:, None] < 0.7, hex_lin("#FF9F3A"), hex_lin("#EAF2FF")) * rng.uniform(1, 4, (nb, 1))
    R.draw2d(bx, by, bc, r_px=rng.uniform(18, 42, nb))
    env = env + R.new_layer()
    L = [Light("dir", (1.0, 0.55, 0.22), 0.55, vec=(0.3, 0.2, 1.0)), Light("dir", (1.0, 0.6, 0.3), 0.35, vec=(0, 1, 0)),
         Light("amb", (0.6, 0.35, 0.18), 0.06)]
    m = draw_solid(R, cam, cloud, L, rim=((1.0, 0.7, 0.4), 3.0, 0.45), spacing_px=2.0, return_mask=True,
                   flecks=((0.3, 0.3, 0.8), 0.04), seed=5, seurat=0.35, p_min=0.3, spec=(0.35, 30.0))
    fg = R.new_layer()
    lvl = torch_level(tg)
    # the lens: a small disc that lights up when the torch finally works
    lensP = lensP @ Rm.T + base
    lx, ly, lz, _, lok = cam.project(lensP)
    R.draw(cam, lensP, TORCH[None] * (0.004 + 0.9 * lvl), size_px=0.7)
    if lvl > 0:
        Pb, Eb = beam_points(tip, aim, lvl, length=40.0, n=160_000, spread=0.05)
        # as we tilt up the focus racks out along the beam, so the dust in it becomes crisp
        cam_b = Camera(cam.pos, cam.target, focal=cam.focal, W=W, H=H,
                       focus=float(np.linalg.norm(focus_pt - cam.pos)) * (1 + 6.0 * tilt),
                       bokeh=55 * (1 - 0.8 * tilt), shift=(cam.shift[0] / cam.s, cam.shift[1] / cam.s))
        R.draw(cam_b, Pb, Eb * 3.0, size_px=0.8)
        R.draw(cam, (tip + aim * 0.002)[None], TORCH[None] * (40.0 * lvl), size_px=4.0, soft=True)
        R.draw(cam, (tip + aim * 0.002)[None], TORCH[None] * (8.0 * lvl), size_px=18.0, soft=True)
    img = env * (1 - m)[..., None] + fg + R.resolve()
    return img, Grade(wb=(1.0, 0.96, 0.92), sat=1.0, bloom=1.25, vignette=0.28, grain=0.006)


# ---------------------------------------------------------------------------- E2: the ascent
def girl_tip_world():
    from ..assets import city
    from .climax import scene_geometry
    geo = scene_geometry()
    eye_local = geo["eyec"]
    from ..assets import rooftop as RT
    hand_local = eye_local + np.array([0.12, -0.28, 0.18])
    return RT.to_world(hand_local, city.ROOF_ORIGIN, city.ROOF_YAW)


def E2(tl, tg, R, W, H):
    sd = star_dir()
    tip = girl_tip_world()
    u = seg(tg, 224.0, 237.0)
    # altitude along the beam: exponential from 3 m to 2 500 km
    s = 3.0 * (2.5e6 / 3.0) ** ease5(u) if u > 0 else 3.0
    lateral = np.cross(sd, [0, 1.0, 0])
    lateral /= np.linalg.norm(lateral)
    cam_pos = tip + sd * s + lateral * (0.22 * s) + np.array([0, 0.3, 0])
    look = tip + sd * (0.25 * s)
    lvl = 1.0 - ease(seg(tg, E["beam_fades"], E["space"]))
    earth_w = ease(seg(s, 9000.0, 60000.0))
    city_w = 1.0 - ease(seg(s, 12000.0, 90000.0))
    img = np.zeros((H, W, 3), np.float32)
    if city_w > 0:
        cam = Camera(cam_pos, look, focal=24, W=W, H=H, near=0.05)
        try:
            from ..assets import city
            env, m_env = city.render_env(R, cam, tg, W, H, roof=True, girl=True)
        except TypeError:
            from ..assets import city
            env, m_env = city.render_env(R, cam, tg, W, H)
        Pb, Eb = beam_points(tip, sd, lvl, length=min(s * 1.6 + 30.0, 4000.0), n=160_000, spread=0.03)
        R.draw(cam, Pb, Eb * (0.6 + 0.4 * (s > 50)), size_px=0.9)
        img += (env + R.resolve()) * city_w
    if earth_w > 0:
        e, n, up = EA.enu(*EA.HOME)
        to_g = lambda p: EA.city_to_globe(np.asarray(p)[None])[0]
        gpos = to_g(cam_pos)
        glook = to_g(look)
        gup = to_g(cam_pos + sd) - gpos
        camg = Camera(gpos, glook, focal=24, W=W, H=H, near=1e-3)
        f = sky.field()
        R.draw_dirs(camg, f["dirs"], f["rgb"], energy=0.8 * earth_w)
        mw = sky.milky_way()
        from .climax import mw_basis
        M = mw_basis()
        R.draw_dirs(camg, mw["g_dirs"] @ M.T, mw["g_rgb"] * 0.010, size_px=1.4, soft=True, energy=earth_w)
        R.draw_dirs(camg, mw["s_dirs"] @ M.T, mw["s_rgb"], energy=0.8 * earth_w)
        stars = R.new_layer() * (1 - sphere_mask(camg, np.zeros(3), EA.R_E, W, H))[..., None]
        L = EA.lights()
        D = EA.land_dots()
        vis = (L["P"] @ (camg.pos / np.linalg.norm(camg.pos))) > 0
        R.draw(camg, L["P"][vis], L["rgb"][vis] * 1.1 * earth_w, size_px=0.6)
        visd = (D["P"] @ (camg.pos / np.linalg.norm(camg.pos))) > 0
        R.draw(camg, D["P"][visd], D["rgb"][visd] * earth_w, size_px=0.8)
        Pa, Ca = EA.atmosphere(camg)
        R.draw(camg, Pa, Ca * earth_w, size_px=2.0, soft=True)
        img += stars + R.resolve()
    return img, Grade(wb=(0.97, 0.98, 1.03), sat=1.0, bloom=1.2, vignette=0.22, grain=0.006)


# ---------------------------------------------------------------------------- E3: the pale point
def E3(tl, tg, R, W, H):
    e, n, up = EA.enu(*EA.HOME)
    sd = star_dir()
    p = ease(seg(tg, 237.0, 244.0))
    dist = 1.2e5 * (12.0 ** p)
    # we look back at Earth from along the beam; Earth sits in the lower-left third
    to_g_dir = sd[0] * e + sd[1] * up - sd[2] * n
    cam_pos = to_g_dir * dist
    fwd_target = -to_g_dir
    cam0 = Camera(cam_pos, cam_pos + fwd_target, focal=50, W=W, H=H)
    # aim so the Earth lands at (0.30 W, 0.68 H)
    shift = ((0.30 - 0.5) * 1920, -(0.68 - 0.5) * 804)
    cam = Camera(cam_pos, cam_pos + fwd_target, focal=50, W=W, H=H, shift=shift)
    f = sky.field()
    R.draw_dirs(cam, f["dirs"], f["rgb"], energy=0.9)
    mw = sky.milky_way()
    from .climax import mw_basis
    M = mw_basis()
    R.draw_dirs(cam, mw["g_dirs"] @ M.T, mw["g_rgb"] * 0.012, size_px=1.4, soft=True)
    R.draw_dirs(cam, mw["s_dirs"] @ M.T, mw["s_rgb"], energy=0.8)
    stars = R.new_layer() * (1 - sphere_mask(cam, np.zeros(3), EA.R_E, W, H))[..., None]
    L = EA.lights()
    vis = (L["P"] @ (cam.pos / np.linalg.norm(cam.pos))) > 0
    R.draw(cam, L["P"][vis], L["rgb"][vis] * 1.2, size_px=0.5)
    Pa, Ca = EA.atmosphere(cam, energy=1.6)
    R.draw(cam, Pa, Ca, size_px=1.5, soft=True)
    fade = 1.0 - ease(seg(tg, 243.2, 244.0))
    return (stars + R.resolve()) * fade, Grade(wb=(0.96, 0.98, 1.04), sat=1.0, bloom=1.2, vignette=0.2,
                                               grain=0.005)


def BLACK(tl, tg, R, W, H):
    return np.zeros((H, W, 3), np.float32), Grade(bloom=0, grain=0, vignette=0)


# ---------------------------------------------------------------------------- credits
CREDIT_CARDS = [
    # (t0, t1, [ (text, font, size, color, y_frac, tracking) ... ])
    (246.3, 249.3, [("光　年", "serif_semibold", 96, "#F2ECE0", 0.47, 0.35),
                    ("LIGHT-YEARS", "plex_light", 26, "#D9D1C3", 0.60, 0.6)]),
    (249.8, 253.3, [("编剧 · 导演 · 渲染 · 配乐", "serif", 26, "#CFC7B8", 0.40, 0.12),
                    ("Written · Directed · Rendered · Scored by", "garamond_it", 22, "#A9A194", 0.47, 0.02),
                    ("Claude Opus 5.5", "plex_light", 40, "#F2ECE0", 0.60, 0.08)]),
    (253.8, 257.3, "CONSULTANTS"),
    (257.8, 260.8, [("cameras 0 · photographs 0 · points of light 11,407,288", "mono_light", 22, "#B9B2A5", 0.40, 0.02),
                    ("每一帧都由代码写成。", "serif", 28, "#EDE7DB", 0.53, 0.10),
                    ("Every frame was written in code.", "garamond_it", 24, "#B9B2A5", 0.60, 0.02)]),
    (261.2, 263.8, [("献给每一个在黑暗里抬起头的人。", "serif", 28, "#EDE7DB", 0.47, 0.10),
                    ("For everyone who ever looked up in the dark.", "garamond_it", 24, "#B9B2A5", 0.55, 0.02)]),
]


def consultant_lines():
    rest = TL.CREDIT_LINES[1:]
    lines = []
    y = 0.50 - 0.08 * (len(rest) - 1)
    for name, cn, en in rest:
        lines.append((f"{cn}　{name.capitalize()}", "serif", 28, "#EDE7DB", y, 0.08))
        lines.append((en.title(), "garamond_it", 22, "#A9A194", y + 0.06, 0.02))
        y += 0.16
    return lines


def CR(tl, tg, R, W, H):
    f = sky.field()
    cam = Camera(np.zeros(3), np.array([np.sin(tg * 0.004), 0.1, np.cos(tg * 0.004)]), focal=35, W=W, H=H)
    R.draw_dirs(cam, f["dirs"], f["rgb"], energy=0.12)
    return R.resolve(), Grade(bloom=0.6, grain=0.004, vignette=0.2)


def overlay(sid, tl, tg, y):
    if sid != "CR":
        return y
    H, W = y.shape[:2]
    s = W / 1920.0
    from ..text import envelope
    for t0, t1, lines in CREDIT_CARDS:
        op = envelope(tg, t0, t1, 0.7, 0.7)
        if op <= 0:
            continue
        if lines == "CONSULTANTS":
            lines = consultant_lines()
        for text, key, size, col, yf, trk in lines:
            a = np.zeros((H, W), np.float32)
            draw_line(a, text, key, size, W / 2, yf * H, trk, "center", s)
            y = glow_composite(y, a, col, op, glow=0.25)
    return y


TABLE = {"E1": E1, "E2": E2, "E3": E3, "BLACK": BLACK, "CR": CR}


def render(sid, tl, tg, R, W, H):
    return dispatch(TABLE, sid, tl, tg, R, W, H)
