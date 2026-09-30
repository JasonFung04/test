"""CLIMAX (III8a–III8c): the photon falls, touches her eye, and for the light there is no distance.

III8c (rooftop, elder + girl, 180-degree orbit) is built on the city/rooftop environment."""
import numpy as np

from .. import timeline as TL
from ..assets import eye as EYE
from ..assets import sky
from ..camera import Camera, ease, ease5, ease_in, ease_out, handheld, lerp, seg
from ..color import hex_lin
from ..noise import hash01
from ..post import Grade
from .common import dispatch, stars_layer

E = TL.EVENTS
GOLD = hex_lin("#FFD27A")
_C = {}


# ---------------------------------------------------------------------------- the night sky
def mw_basis():
    """Galactic -> world rotation for the city's night sky (shared with the city module if present)."""
    if "mwb" not in _C:
        try:
            from ..assets import city
            _C["mwb"] = np.asarray(city.MW_BASIS, float)
        except Exception:
            gc = np.array([0.0, np.sin(np.radians(24)), np.cos(np.radians(24))])
            ngp = np.array([-0.85, 0.45, -0.25])
            _C["mwb"] = sky.galactic_basis(gc, ngp)
    return _C["mwb"]


def gal_dir(l_deg, b_deg):
    l, b = np.radians(l_deg), np.radians(b_deg)
    return np.array([np.cos(b) * np.cos(l), np.sin(b), np.cos(b) * np.sin(l)])


def gold_star_dir():
    """Where the dead civilisation's light arrives from: just beside the Milky Way band, high in the
    southern sky (elevation ~48 deg, a little west of south). Found once by search in galactic coords."""
    if "gsd" in _C:
        return _C["gsd"]
    M = mw_basis()
    best = None
    for l in np.arange(0.0, 360.0, 1.0):
        for b in (7.0, -7.0):
            d = M @ gal_dir(l, b)
            el = np.degrees(np.arcsin(d[1]))
            az = np.degrees(np.arctan2(d[0], d[2]))          # 0 = south (+z), negative = west (-x)
            cost = (el - 48.0) ** 2 + 0.5 * (az + 12.0) ** 2
            if best is None or cost < best[0]:
                best = (cost, d, l, b)
    d = best[1] / np.linalg.norm(best[1])
    _C["gsd"] = d
    _C["gsd_lb"] = (best[2], best[3])
    return d


def draw_night_sky(R, cam, energy=1.0, band=1.0):
    mw = sky.milky_way()
    M = mw_basis()
    R.draw_dirs(cam, mw["g_dirs"] @ M.T, mw["g_rgb"] * (0.010 * band), size_px=1.4, soft=True, energy=energy)
    R.draw_dirs(cam, mw["s_dirs"] @ M.T, mw["s_rgb"], energy=0.8 * energy)
    f = sky.field()
    R.draw_dirs(cam, f["dirs"], f["rgb"], energy=0.9 * energy)


def messenger(R, cam, pos, b, tg, scale=1.0, spikes=True):
    tw = 1.0 + 0.06 * np.sin(tg * 23.0) + 0.04 * np.sin(tg * 37.0)
    p = np.asarray(pos, float)[None]
    k = b * scale
    R.draw(cam, p, GOLD[None] * (780.0 * k * tw), size_px=2.5)
    R.draw(cam, p, GOLD[None] * (240.0 * k), size_px=10, soft=True)
    R.draw(cam, p, GOLD[None] * (900.0 * k), size_px=48, soft=True)
    if spikes:
        x, y, _, _, ok = cam.project(p)
        if ok[0]:
            t = np.linspace(-1, 1, 90)
            L = 70 * R.s * k ** 0.5
            for ang in (np.radians(15), np.radians(105)):
                w = (1 - np.abs(t)) ** 3
                R.draw2d(x[0] + np.cos(ang) * t * L, y[0] + np.sin(ang) * t * L, GOLD[None] * (w * 2.2 * k)[:, None],
                         r_px=0.8)


GRADE_N = dict(wb=(0.92, 0.97, 1.08), sat=0.95, bloom=1.2, vignette=0.25, grain=0.006)


# ---------------------------------------------------------------------------- III8a
def III8a(tl, tg, R, W, H):
    """The photon leaves the star and falls toward her: camera above/behind her follows it down."""
    sd = gold_star_dir()
    G = np.zeros(3)                                        # her eye (local origin)
    C0 = np.array([0.0, 3.2, -4.0])                        # camera: above and behind her
    t_off = E["gold_star_off"]
    u = seg(tg, t_off, 184.5)
    D = 2500.0 * (1.0 - u) ** 3.0 + 2.0
    ph = G + sd * D
    look_star = C0 + sd * 1000.0
    k = ease5(seg(tg, t_off + 0.2, 184.3))
    tgt = look_star * (1 - k) + (ph + np.array([0, -0.4, 0])) * k
    cam = Camera(C0 + handheld(tg, 0.01, seed=81), tgt, focal=lerp(135, 50, k), W=W, H=H)
    draw_night_sky(R, cam)
    on = 1.0 - ease(seg(tg, t_off - 0.08, t_off + 0.10))
    if on > 0:
        messenger(R, cam, C0 + sd * 1e7, 0.10 * on, tg, spikes=False)
    if tg >= t_off:
        messenger(R, cam, ph, 0.05 + 0.25 * u ** 1.5, tg)
        for j in range(1, 16):
            uu = max(u - j * 0.010, 0)
            Dj = 2500.0 * (1.0 - uu) ** 3.0 + 2.0
            R.draw(cam, (G + sd * Dj)[None], GOLD[None] * (3.0 * (1 - j / 16) * (0.3 + u)), size_px=1.3)
        if "motes" not in _C:
            rng = np.random.default_rng(5)
            n = 40000
            dd = rng.uniform(3, 2500, n)
            ax = np.cross(sd, [0, 1, 0])
            ax /= np.linalg.norm(ax)
            ay = np.cross(ax, sd)
            off = rng.normal(0, 1, (n, 2)) * (6 + dd * 0.03)[:, None]
            _C["motes"] = G + np.outer(dd, sd) + np.outer(off[:, 0], ax) + np.outer(off[:, 1], ay)
        Pm = _C["motes"]
        dist = np.linalg.norm(Pm - ph, axis=1)
        glow = np.exp(-(dist / (4 + D * 0.03)) ** 2)
        R.draw(cam, Pm, hex_lin("#FFE3B0")[None] * 0.004 + GOLD[None] * (3.0 * glow)[:, None], size_px=0.8)
    return R.resolve(), Grade(**GRADE_N)


# ---------------------------------------------------------------------------- III8b
def eye_asset():
    if "eye" not in _C:
        _C["eye"] = EYE.build()
    return _C["eye"]


def III8b(tl, tg, R, W, H):
    d = eye_asset()
    hl = EYE.cornea_point(EYE.HIGHLIGHT[:2])
    touch, refl = E["cornea_touch"], E["hand_reflection"]
    # push in slowly, then dive into the highlight once the hand appears in it
    p = ease5(seg(tg, 184.5, touch))
    q = ease_in(seg(tg, refl, 187.5), 2.4)
    base_d = lerp(0.105, 0.078, p)
    pos = np.array([0.0015, 0.0012, 0.0]) * (1 - q) + hl * q
    dist = base_d * (1 - q) + 0.0022 * q
    tgt = np.array([0.0, 0.0006, 0.0]) * (1 - q) + hl * q
    cam = Camera(pos + np.array([0.0, 0.0, dist]) + np.array([0, 0, 0.0]), tgt, focal=100, W=W, H=H,
                 focus=dist, bokeh=30 * (1 - q) + 6)
    # eye points: projected-area energy so the look holds across the push-in
    P, C, kind = d["P"], d["rgb"], d["kind"]
    x, y, z, coc, ok = cam.project(P)
    area_w = np.array([3.1e-10, 3.1e-10, 1.4e-9, 2.0e-10])[kind]
    sp = np.sqrt(area_w) * cam.fpx / np.maximum(z, 1e-6)
    L = np.array([0.30, 0.05, 0.035, 0.03], np.float32)[kind]
    E_ = C * (L * sp ** 2 / R.e)[:, None]
    R.draw(cam, P, E_, size_px=np.clip(sp / R.s * 0.45, 0.4, 6.0))
    # the night sky reflected on the cornea: a faint band of stars
    if "refl" not in _C:
        rng = np.random.default_rng(9)
        n = 5000
        a = rng.uniform(-1, 1, n)
        xy = np.stack([a * 0.006, 0.0022 * a + rng.normal(0, 0.0009, n)], 1)
        xy = xy[np.hypot(xy[:, 0], xy[:, 1]) < 0.0058]
        _C["refl"] = np.array([EYE.cornea_point(v) for v in xy])
    Rf = _C["refl"]
    R.draw(cam, Rf, np.array([[0.02, 0.022, 0.03]]) * (0.25 + hash01(np.arange(len(Rf)), 3)[:, None]), size_px=0.6)
    # the photon descends into view and lands on the highlight
    u = seg(tg, 184.5, touch)
    start = hl + np.array([0.0006, 0.0060, 0.020])
    ph = start + (hl - start) * ease_in(u, 1.6)
    if tg < touch:
        messenger(R, cam, ph, 0.02 + 0.05 * u, tg, scale=0.6)
    flare = np.exp(-max(tg - touch, 0) / 0.18) * (tg >= touch)
    messenger(R, cam, hl, 0.035 + 0.35 * flare, tg, scale=0.5, spikes=False)
    # the reflected palm appears inside the highlight
    if tg >= refl:
        a = ease(seg(tg, refl, refl + 0.4))
        from .act1 import ehand
        from ..assets.figures import hand_pose_dict, elder_points
        s_ = 0.0042
        Rm = np.array([[np.cos(0.6), -np.sin(0.6), 0], [np.sin(0.6), np.cos(0.6), 0], [0, 0, 1]])
        hp = hand_pose_dict(ehand(), Rm, np.zeros(3), 0.04, spread=0.45)
        palm_c = Rm @ ehand().palm_center()
        hp["P"] = ((hp["P"] - palm_c) * s_ + hl + np.array([0, 0, 0.00005])).astype(np.float32)
        hp["area"] = hp["area"] * s_ * s_
        Pd, Ed = elder_points(hp, cam, tg, energy=0.9 * a, spacing_px=2.0, interior=0.14)
        R.draw(cam, Pd, Ed, size_px=0.9)
    return R.resolve(), Grade(**GRADE_N)


TABLE = {"III8a": III8a, "III8b": III8b}


def render(sid, tl, tg, R, W, H):
    return dispatch(TABLE, sid, tl, tg, R, W, H)


# ---------------------------------------------------------------------------- III8c
from ..assets import alien, bodies, rooftop as RT  # noqa: E402
from ..assets.figures import Elder, Girl, elder_ik, elder_points, hand_pose_dict  # noqa: E402
from ..sdf import rot  # noqa: E402
from ..solid import Light, SolidCloud, draw_solid  # noqa: E402

GIRL_HEAD_PITCH = 0.72


def girl_asset():
    if "girl" not in _C:
        _C["girl"] = Girl()
    return _C["girl"]


def elder_asset():
    if "elder" not in _C:
        _C["elder"] = Elder()
    return _C["elder"]


def roof_cloud():
    if "roof" not in _C:
        _C["roof"] = RT.cloud()
    return _C["roof"]


def scene_geometry():
    """Rooftop-local geometry of the climax (computed once)."""
    if "geo" in _C:
        return _C["geo"]
    g = girl_asset()
    gc, gi = g.pose(head_pitch=GIRL_HEAD_PITCH, armL=(0.42, 0.22, 1.55), armR=(0.42, 0.22, 1.55), torch_hand="R")
    seat = RT.GIRL_SEAT
    eye = gi["eyes"][0] + seat
    eye2 = gi["eyes"][1] + seat
    eyec = (eye + eye2) / 2
    sd = gold_star_dir_local()
    palm_c = eyec + sd * 0.32                       # the palm hovers above her face, along her gaze to the star
    # elder stands south of her (in front), facing north (-z), leaning over her
    yaw = np.pi
    Ry = rot("y", yaw)
    base = np.array([eyec[0] + 0.10, RT.TANK_TOP_Y - 0.45, eyec[2] + 1.33])
    lean = 0.45
    wrist_world = palm_c + np.array([0.0, 0.05, 0.10])
    wrist_fig = Ry.T @ (wrist_world - base)
    best = None
    for x0 in ((1.2, 0.15, 0.6), (0.9, 0.3, 1.2), (1.5, 0.0, 1.0), (0.6, 0.2, 1.6)):
        q, err = elder_ik(elder_asset(), wrist_fig, side="R", lean=lean, x0=x0)
        if best is None or err < best[1]:
            best = (q, err)
    q, err = best
    _C["geo"] = dict(gc=gc, gi=gi, seat=seat, eye=eye, eyec=eyec, sd=sd, palm_c=palm_c, base=base, Ry=Ry,
                     lean=lean, armR=q, ik_err=err, wrist_world=wrist_world)
    return _C["geo"]


def gold_star_dir_local():
    """Star direction in rooftop-local coords (the rooftop is yawed in the world)."""
    try:
        from ..assets import city
        yaw = float(city.ROOF_YAW)
    except Exception:
        yaw = 0.0
    return rot("y", -yaw) @ gold_star_dir()


def elder_world(tg, dissolve=0.0):
    geo = scene_geometry()
    e = elder_asset()
    pose, info = e.pose(head_pitch=-0.45, head_yaw=0.0, armR=geo["armR"], armL=(0.12, 0.10, 0.25), curlL=0.7,
                        lean=geo["lean"], skip_hand_R=True, arm_n=16000)
    Ry, base = geo["Ry"], geo["base"]
    # made of light, not of matter: the lower body thins out into the air below the tank's edge
    yf = pose["P"][:, 1]
    fade = np.clip((yf - 0.95) / 0.45, 0, 1) ** 1.5
    pose = dict(pose, col=(pose["col"] * fade[:, None]).astype(np.float32))
    pose = dict(pose, P=(pose["P"] @ Ry.T + base).astype(np.float32), N=(pose["N"] @ Ry.T).astype(np.float32))
    # sculpted hand at the wrist, palm facing her eye
    wrist = info["wristR"] @ Ry.T + base
    to_eye = geo["eyec"] - geo["palm_c"]
    to_eye /= np.linalg.norm(to_eye)
    fing = np.cross(to_eye, np.array([1.0, 0, 0]))
    fing /= np.linalg.norm(fing)
    if fing[1] > 0:
        fing = -fing
    xax = np.cross(-fing, to_eye)
    Rm = np.stack([xax, -fing, to_eye], axis=1)        # local x, y(=-fingers), z(=palm normal) -> world
    hand = hand_pose_dict(ehand(), Rm, wrist, 0.10, spread=0.5)
    palm_c = wrist + Rm @ ehand().palm_center()
    return pose, hand, palm_c, info


def ehand():
    from .act1 import ehand as _eh
    return _eh()


def night_lights(extra=None):
    L = [Light("dir", (0.55, 0.65, 1.0), 0.12, vec=(0.2, 0.8, 0.4)), Light("amb", (0.25, 0.3, 0.5), 0.03)]
    if extra:
        L += extra
    return L


def red_sky(R, cam, W, H, tg, amount):
    """The elder's world rising behind him: the red giant limb and its glow over the southern sky."""
    if amount <= 0:
        return None
    C, Rs = alien.draw_giant(R, cam, tg, elev_deg=-17.0, ang_radius_deg=26.0, energy=2.4 * amount, az_deg=4.0)
    img = R.new_layer()
    # the sky around it glows, strongest toward the giant (screen-space, no hard horizon)
    x, y, _, _, ok = cam.project(C[None])
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    dd = np.sqrt((xx - x[0]) ** 2 + (yy - y[0]) ** 2) / W
    glow = np.exp(-np.maximum(dd - 0.35, 0) / 0.35)[..., None] * hex_lin("#FF3A12") * (0.22 * amount)
    return img + glow.astype(np.float32)


def III8c(tl, tg, R, W, H):
    geo = scene_geometry()
    eyec, palm_c0 = geo["eyec"], geo["palm_c"]
    pose, hand, palm_c, info = elder_world(tg)
    # ---------------- camera choreography (rooftop-local)
    t_pull0 = 187.5
    t_orb0, t_orb1 = E["orbit_start"], E["orbit_end"]
    head = geo["base"] + np.array([0.0, 1.75, -0.55])
    center = eyec * 0.55 + head * 0.45 + np.array([0.0, -0.08, 0.0])
    rad = 2.3

    def orbit_pos(ang):
        return center + np.array([-np.cos(ang) * rad, -0.28 + 0.12 * np.sin(ang), -np.sin(ang) * rad])

    if tg < t_orb0:
        # from her eye's point of view up at the palm, pulling back (over her head) and swinging to the side
        k = ease5(seg(tg, t_pull0, t_orb0))
        start_pos = eyec + (palm_c - eyec) * 0.15
        pos = lerp(start_pos, orbit_pos(0.0), k)
        tgt = lerp(palm_c, center, ease5(seg(tg, t_pull0 + 0.2, t_orb0)))
        focal = lerp(24, 26, k)
        bok = lerp(22, 4, k)
    else:
        k = ease(seg(tg, t_orb0, t_orb1))
        ang = np.pi * k                                 # 180 deg: west -> north (behind her) -> east
        pos = orbit_pos(ang)
        tgt = center + np.array([0, 0.10, 0])
        focal = 26
        bok = 4
    cam = Camera(pos, tgt, focal=focal, W=W, H=H, focus=float(np.linalg.norm(center - pos)), bokeh=bok)
    # ---------------- sky: night, then the red world rises behind the elder
    red = ease(seg(tg, E["sky_red"], E["sky_red"] + 3.0)) * (1.0 - ease(seg(tg, 198.5, 200.5)))
    draw_night_sky(R, cam, energy=1.0 - 0.6 * red)
    sky_img = R.new_layer()
    rs = red_sky(R, cam, W, H, tg, red)
    if rs is not None:
        sky_img = sky_img * (1 - 0.5 * red) + rs
    # ---------------- rooftop + girl (opaque)
    red_light = Light("dir", (1.0, 0.35, 0.12), 1.8 * red, vec=(0.0, 0.25, 1.0))
    arc_u = seg(tg, E["arc_start"], E["arc_end"])
    ph = palm_c + (eyec - palm_c) * ease_in(arc_u, 1.3) + np.array([0, 0.05, 0]) * np.sin(np.pi * arc_u)
    kiss = np.exp(-max(tg - E["arc_end"], 0) / 1.2) * (tg >= E["arc_start"])
    gold_light = Light("point", (1.0, 0.80, 0.45), 0.004 + 0.035 * kiss, vec=ph, radius=0.05)
    girl = geo["gc"].transformed(np.eye(3), geo["seat"])
    env = SolidCloud.concat([roof_cloud(), girl])
    m_env = draw_solid(R, cam, env, night_lights([red_light, gold_light]), rim=((0.55, 0.68, 1.0), 3.0, 0.5),
                       spacing_px=2.1, return_mask=True, flecks=((0.3, 0.25, 0.8), 0.05))
    solid = R.new_layer()
    # girl-only mask for occluding the elder
    R2 = type(R)(W, H)
    m_girl = draw_solid(R2, cam, girl, night_lights(), spacing_px=2.1, return_mask=True)
    # ---------------- the elder (translucent light) + his hand
    dis = ease(seg(tg, E["elder_dissolve"], 200.5))
    Pd, Ed = elder_points(pose, cam, tg, energy=0.9, spacing_px=2.2, dissolve=dis, up=(0, 1, 0))
    R.draw(cam, Pd, Ed, size_px=0.8)
    Ph, Eh = elder_points(hand, cam, tg, energy=0.9, spacing_px=2.0, dissolve=dis)
    R.draw(cam, Ph, Eh, size_px=0.85)
    elder_img = R.new_layer() * (1 - m_girl)[..., None]
    # ---------------- the photon: from the palm into her eye
    if tg < E["arc_start"]:
        messenger(R, cam, palm_c, 0.05, tg, spikes=False)
    elif tg < E["arc_end"] + 0.3:
        messenger(R, cam, ph, 0.14, tg, spikes=False)
        for j in range(1, 30):
            uu = max(arc_u - j * 0.02, 0)
            pj = palm_c + (eyec - palm_c) * ease_in(uu, 1.3) + np.array([0, 0.05, 0]) * np.sin(np.pi * uu)
            R.draw(cam, pj[None], GOLD[None] * (2.5 * (1 - j / 30) ** 1.5), size_px=1.0)
    elif np.dot(cam.pos - eyec, geo["sd"]) > 0:
        # a tiny gold catchlight remains in her eye (only when we can see her face)
        R.draw(cam, eyec[None], GOLD[None] * 1.2, size_px=1.0)
    front = R.resolve()
    img = sky_img * (1 - m_env)[..., None] + solid + elder_img + front
    return img, Grade(wb=(0.95, 0.97, 1.05), sat=1.0, bloom=1.2, vignette=0.28, grain=0.006)


TABLE["III8c"] = III8c
