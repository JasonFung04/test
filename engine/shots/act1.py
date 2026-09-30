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


# ============================================================================ ground scenes
from ..assets import alien  # noqa: E402
from ..assets.figures import Elder, ElderHand, elder_points, hand_pose_dict  # noqa: E402
from ..camera import ease5, ease_in, ease_out, handheld, lerp, seg  # noqa: E402
from ..color import hex_lin  # noqa: E402
from ..noise import fbm, hash01  # noqa: E402

CROWD_C = np.array([0.0, 0.0, 8000.0])
CROWD_C[1] = alien.terrace_h(8000.0)
ELDER_POS = CROWD_C + np.array([-28.2, 0.0, 1.4])     # front-left end of the crowd
MSG_GOLD = hex_lin("#FFD27A")


def city_pts():
    if "city" not in _C:
        _C["city"] = alien.city()
    return _C["city"]


def crowd():
    if "crowd" not in _C:
        _C["crowd"] = alien.Crowd(CROWD_C, rows=7, per_row=40, seed=5)
    return _C["crowd"]


def elder():
    if "elder" not in _C:
        _C["elder"] = Elder()
    return _C["elder"]


GIANT_AZ, GIANT_EL, GIANT_R = -24.0, -14.0, 40.0


def ground_sky(R, cam, W, H, tg, giant_energy=1.5, elev=None):
    """Sky + giant (limb arcing through frame) + horizon mask. Returns (sky_img, above_horizon_mask)."""
    m, hy = alien.horizon_mask(cam, W, H)
    stars_layer(R, cam, field(), energy=0.22)
    stars = R.new_layer()
    C, Rs = alien.draw_giant(R, cam, tg, elev_deg=GIANT_EL, ang_radius_deg=GIANT_R, energy=giant_energy,
                             az_deg=GIANT_AZ)
    gm = bodies.sphere_mask(cam, C, Rs * 1.002, W, H)
    sky = (R.new_layer() + stars * (1 - gm)[..., None]) * m[..., None] + alien.sky_glow(W, H, hy)
    return sky, m


KIND_GAIN = np.array([0.55, 1.35, 1.6, 0.18], np.float32)   # windows, streets, plazas, terrace rings


def draw_city(R, cam, tg, energy=1.0, lift=True, near_cut=0.0):
    c = city_pts()
    P, E_, key, kind = c["P"], c["rgb"], c["key"], c["kind"]
    E_ = E_ * KIND_GAIN[kind][:, None]
    if near_cut:
        far = np.linalg.norm(P - cam.pos, axis=1) > near_cut
        P, E_, key = P[far], E_[far], key[far]
    if lift:
        P, E_, up = alien.lifted(P, E_, key, tg)
    R.draw(cam, P, E_ * energy, size_px=0.6, falloff=1.2, zref=3000.0)


GRADE_G = dict(wb=(1.0, 0.93, 0.86), sat=1.1, bloom=1.3, vignette=0.25, grain=0.006)


def I2(tl, tg, R, W, H):
    p = ease5(seg(tg, 35.0, 38.2))
    g = ease(seg(tg, 35.0, 41.0))
    alt = lerp(3600.0, 2100.0, p)
    z = lerp(-17500.0, -14200.0, g)
    pitch = np.radians(lerp(-30.0, -12.5, ease5(seg(tg, 35.0, 39.0))))
    pos = np.array([150.0, alt, z])
    tgt = pos + np.array([0.0, np.sin(pitch), np.cos(pitch)])
    cam = Camera(pos, tgt, focal=24, W=W, H=H)
    sky, m = ground_sky(R, cam, W, H, tg)
    draw_city(R, cam, tg, energy=1.0, lift=False)
    city = R.new_layer()
    img = sky + city
    # passing through the cloud deck: screen-space layer whose coverage peaks at the crossing
    cover = float(np.clip(np.exp(-((alt - 2950.0) / 260.0) ** 2) * 1.25, 0, 1))
    above = float(np.clip((alt - 2950.0) / 500.0, 0, 1))
    cover = max(cover, 0.55 * above)
    if cover > 0.01:
        import cv2
        hs, ws = H // 4, W // 4
        yy, xx = np.mgrid[0:hs, 0:ws].astype(np.float64)
        drift = (tg - 35.0) * 0.9
        Q = np.stack([xx.ravel() / ws * 3.0, yy.ravel() / hs * 1.3 + drift, np.full(xx.size, tg * 0.05)], 1)
        nz = fbm(Q, octaves=5).reshape(hs, ws)
        nz = cv2.resize(nz.astype(np.float32), (W, H), interpolation=cv2.INTER_CUBIC)
        dens = np.clip((nz + 0.35 * cover - 0.25 + cover * 0.5) * 2.4, 0, 1) * cover
        lit = np.clip(1.0 - np.arange(H, dtype=np.float32) / H, 0, 1)[:, None]
        cc = hex_lin("#FF6A3A") * 0.10 * (0.4 + 0.9 * lit)[..., None] + hex_lin("#2A0C12") * 0.08
        img = img * (1 - dens[..., None] * 0.92) + cc * dens[..., None]
    return img, Grade(**GRADE_G)



def I3(tl, tg, R, W, H):
    p = seg(tg, 41.0, 45.0)
    base = CROWD_C + np.array([3.0 - 3.0 * p, 1.75, -24.0])
    cam = Camera(base, base + np.array([-1.2, 0.35, 20.0]), focal=50, W=W, H=H, focus=26.0, bokeh=14)
    sky, m = ground_sky(R, cam, W, H, tg)
    draw_city(R, cam, tg, energy=0.7, lift=False, near_cut=400.0)
    bg = R.new_layer()
    Pd, Ed = elder_points(crowd().pose(tg), cam, tg, energy=0.55, spacing_px=2.4, interior=0.10)
    R.draw(cam, Pd, Ed, size_px=0.8)
    return sky + bg + R.resolve(), Grade(**GRADE_G)


def _elder_pose(tg):
    look_down = ease5(seg(tg, E_["elder_looks_down"], E_["elder_looks_down"] + 1.4))
    curl = lerp(0.85, 0.05, ease5(seg(tg, E_["palm_open"], E_["palm_open"] + 1.0)))
    # palm faces up (+y), fingers point toward the giant (+z), turned a little inward
    Rpalm = np.array([[np.cos(-0.35), 0, np.sin(-0.35)], [0, 1, 0], [-np.sin(-0.35), 0, np.cos(-0.35)]]) @ \
        np.array([[1, 0, 0], [0, 0, 1], [0, -1, 0]])
    pose, info = elder().pose(head_pitch=lerp(-0.12, 0.55, look_down), head_yaw=lerp(0.0, -0.25, look_down),
                              armR=(lerp(0.25, 1.05, ease5(seg(tg, 46.0, 49.5))), 0.08, lerp(0.2, 1.05, ease5(seg(tg, 46.0, 49.5)))),
                              armL=(0.06, 0.08, 0.12), curlR=curl, curlL=0.7, palmR=Rpalm, spreadR=0.45,
                              hand_n=60000, arm_n=16000)
    return pose, info


E_ = TL.EVENTS


def _to_world_elder(P):
    return P + ELDER_POS


def I4(tl, tg, R, W, H):
    pose, info = _elder_pose(tg)
    head = info["eyes"][0] * 0.5 + info["eyes"][1] * 0.5 + ELDER_POS
    drift = handheld(tg, 0.004, seed=4)
    cam_pos = head + np.array([2.55, -0.25, 2.35]) + drift
    tgt = head + np.array([0.02, -0.10 - 0.10 * seg(tg, 47.8, 49.5), 0.0])
    cam = Camera(cam_pos, tgt, focal=85, W=W, H=H, focus=float(np.linalg.norm(cam_pos - head)), bokeh=40,
                 shift=(0.12 * 1920, 0.0))
    sky, m = ground_sky(R, cam, W, H, tg)
    draw_city(R, cam, tg, energy=0.7, lift=False, near_cut=400.0)
    Pd, Ed = elder_points(crowd().pose(tg, skip=ELDER_POS), cam, tg, energy=0.5, spacing_px=2.4, interior=0.10)
    R.draw(cam, Pd, Ed, size_px=0.8)
    bg = R.new_layer()
    Pw = pose["P"] + ELDER_POS
    pw = dict(pose, P=Pw)
    Pd, Ed = elder_points(pw, cam, tg, energy=0.9, spacing_px=2.2)
    R.draw(cam, Pd, Ed, size_px=0.8)
    for e in info["eyes"]:
        R.draw(cam, (e + ELDER_POS)[None], np.array([[2.5, 2.1, 1.5]]), size_px=1.4)
    return sky + bg + R.resolve(), Grade(**GRADE_G)


def palm_screen_anchor(W, H):
    """Where the messenger light sits on screen in I5 (and again in the climax palm shot)."""
    return 0.60 * W, 0.56 * H


def ehand():
    if "ehand" not in _C:
        _C["ehand"] = ElderHand()
    return _C["ehand"]


def palm_frame(tg):
    """World wrist position + orientation of the right hand (palm up, fingers toward the giant)."""
    pose, info = _elder_pose(tg)
    Rm = np.asarray(info["handR_R"])
    return info["wristR"] + ELDER_POS, Rm


def I5(tl, tg, R, W, H):
    wrist, Rm = palm_frame(tg)
    curl = lerp(0.95, 0.04, ease5(seg(tg, E_["palm_open"], E_["palm_open"] + 1.0)))
    hp = hand_pose_dict(ehand(), Rm, wrist, curl, spread=lerp(0.1, 0.45, ease5(seg(tg, 50.3, 51.5))))
    palm_c = wrist + Rm @ ehand().palm_center()
    nrm = Rm @ np.array([0, 0, 1.0])
    fing = Rm @ np.array([0, -1.0, 0])
    # above the palm, the wrist toward the lower-right corner, fingers toward the upper-left
    side = np.cross(nrm, fing)
    cam_pos = palm_c + nrm * 0.95 - fing * 0.16 + side * 0.14 + handheld(tg, 0.002, seed=5)
    cam = Camera(cam_pos, palm_c, focal=100, W=W, H=H, up=fing * 0.8 - side * 0.6,
                 focus=float(np.linalg.norm(cam_pos - palm_c)), bokeh=48)
    # re-aim so the palm centre lands on the shared anchor (0.60 W, 0.56 H)
    x, y, _, _, _ = cam.project(palm_c[None])
    ax, ay = palm_screen_anchor(W, H)
    cam = Camera(cam_pos, palm_c, focal=100, W=W, H=H, up=fing * 0.8 - side * 0.6,
                 focus=float(np.linalg.norm(cam_pos - palm_c)), bokeh=48,
                 shift=((ax - W / 2) / cam.s, -(ay - H / 2) / cam.s))
    sky, m = ground_sky(R, cam, W, H, tg)
    bg = R.new_layer()
    Pd, Ed = elder_points(hp, cam, tg, energy=0.9, spacing_px=1.9, interior=0.12)
    R.draw(cam, Pd, Ed, size_px=0.9)
    b = ease_out(seg(tg, E_["palm_tone"], E_["palm_tone"] + 1.0), 2.0)
    if b > 0:
        spark = palm_c + nrm * 0.006
        messenger(R, cam, spark, b, tg)
    return sky * 0.6 + bg + R.resolve(), Grade(**GRADE_G)


def messenger(R, cam, pos, b, tg, scale=1.0):
    """The messenger light: a tiny, unmistakable star (same look in the palm, the sky and the climax)."""
    tw = 1.0 + 0.06 * np.sin(tg * 23.0) + 0.04 * np.sin(tg * 37.0)
    p = np.asarray(pos)[None]
    k = b * scale
    R.draw(cam, p, MSG_GOLD[None] * (780.0 * k * tw), size_px=2.5)
    R.draw(cam, p, MSG_GOLD[None] * (240.0 * k), size_px=10, soft=True)
    R.draw(cam, p, MSG_GOLD[None] * (900.0 * k), size_px=48, soft=True)
    # a faint four-point diffraction star
    x, y, _, _, ok = cam.project(p)
    if ok[0]:
        n = 90
        t = np.linspace(-1, 1, n)
        L = 70 * R.s * k ** 0.5
        a = np.radians(15)
        for ang in (a, a + np.pi / 2):
            xs = x[0] + np.cos(ang) * t * L
            ys = y[0] + np.sin(ang) * t * L
            w = (1 - np.abs(t)) ** 3
            R.draw2d(xs, ys, MSG_GOLD[None] * (w * 2.2 * k)[:, None], r_px=0.8)


def I6(tl, tg, R, W, H):
    p = ease(seg(tg, 54.0, 64.0))
    start = ELDER_POS + np.array([-2.4, 1.75, -3.2])
    end = ELDER_POS + np.array([-420.0, 380.0, -2600.0])
    pos = lerp(start, end, ease_in(p, 1.7))
    look = lerp(ELDER_POS + np.array([4.0, 1.9, 8.0]), ELDER_POS + np.array([600.0, 2600.0, 9000.0]), ease5(p))
    cam = Camera(pos, look, focal=35, W=W, H=H, focus=float(np.linalg.norm(pos - ELDER_POS)) if p < 0.3 else None,
                 bokeh=12 * (1 - p))
    sky, m = ground_sky(R, cam, W, H, tg)
    draw_city(R, cam, tg, energy=1.0, lift=True, near_cut=300.0 * (1 - p))
    cp = crowd().pose(tg, skip=ELDER_POS)
    Pd, Ed = elder_points(cp, cam, tg, energy=0.5, spacing_px=2.4, interior=0.10)
    # each body releases its points upward, joining the rising city
    kd = hash01(np.arange(len(Pd)), 55)
    t_rel = E_["lift_first"] + 0.5 + kd * (E_["lift_all"] + 2.5 - E_["lift_first"])
    age = np.maximum(tg - t_rel, 0)
    rel = age > 0
    Pd = Pd.copy()
    Pd[rel, 1] += (3.0 * age[rel] + 30.0 * age[rel] ** 2.2)
    Ed = Ed * np.where(rel, 1.8, 1.0 - 0.8 * seg(tg, 57.0, 62.0))[:, None]
    R.draw(cam, Pd, Ed, size_px=0.8)
    if tg < 60.0:
        pose, info = _elder_pose(53.99)
        pw = dict(pose, P=pose["P"] + ELDER_POS)
        dis = ease(seg(tg, E_["lift_first"] + 1.5, E_["elder_gone"]))
        Pd, Ed = elder_points(pw, cam, tg, energy=0.9, dissolve=dis, spacing_px=2.2)
        R.draw(cam, Pd, Ed, size_px=0.8)
    return sky + R.resolve(), Grade(**GRADE_G)


TABLE.update({"I2": I2, "I3": I3, "I4": I4, "I5": I5, "I6": I6})


# ============================================================================ orbit: I7–I9
from ..assets.shell import Shell  # noqa: E402

PL_C = np.array([0.0, 0.0, 0.0])
PL_R = 6.0
CSTAR = np.array([0.0, PL_R * 1.95, 0.0])          # convergence point above the pole
GIANT_C = np.array([-38.0, 14.0, -210.0])
GIANT_R = 110.0
SHELL_V = 7.2                                       # shell expansion speed (units / s)


def shell_asset():
    if "shell" not in _C:
        _C["shell"] = Shell(seed=7)
    return _C["shell"]


def _orbit_bg(R, cam, W, H, tg, lights_on=True):
    stars_layer(R, cam, field(), energy=0.6)
    st = R.new_layer()
    g = giant()
    Rs = star_radius(tg, GIANT_R)
    gm = bodies.sphere_mask(cam, GIANT_C, Rs * 1.003, W, H)
    P, C = g.surface(tg, GIANT_C, Rs, cam.pos, energy=1.5)
    R.draw(cam, P, C, size_px=1.1)
    P, C = g.corona(tg, GIANT_C, Rs, cam.pos)
    R.draw(cam, P, C, size_px=3.0, soft=True)
    back = R.new_layer() + st * (1 - gm)[..., None]
    pm = bodies.sphere_mask(cam, PL_C, PL_R, W, H)
    back *= (1 - pm)[..., None]
    sd = GIANT_C - PL_C
    sd /= np.linalg.norm(sd)
    pl = planet()
    P, C = pl.rim(PL_C, PL_R, cam.pos, sd, energy=1.6, thick=0.02)
    R.draw(cam, P, C, size_px=0.9, soft=True)
    return back, sd


def _streams(tg, cam_pos, sd, n=14000):
    """Lights leaving the night side and converging on CSTAR along bezier paths, with short trails."""
    pl = planet()
    if "streams" not in _C:
        P, C, vis = pl.lights(PL_C, PL_R, cam_pos, sd, energy=1.0)
        rng = np.random.default_rng(8)
        pick = rng.choice(len(P), min(n, len(P)), replace=False)
        S = P[pick].astype(np.float64)
        nrm = (S - PL_C) / np.linalg.norm(S - PL_C, axis=1, keepdims=True)
        t0 = E_["converge_peak"] - 6.5 + rng.random(len(S)) ** 0.8 * 4.2
        _C["streams"] = (S, nrm, C[pick], t0)
    S, nrm, C, t0 = _C["streams"]
    dur = 2.6
    u = (tg - t0) / dur
    live = (u > 0) & (u < 1)
    arrived = float(np.mean(u >= 1))
    P1 = S + nrm * PL_R * 1.1
    Ps, Es = [], []
    lags = np.linspace(0.0, 0.22, 14)
    for k, lag in enumerate(lags):
        uu = np.clip(u - lag, 0, 1)[:, None]
        p = (1 - uu) ** 2 * S + 2 * uu * (1 - uu) * P1 + uu ** 2 * CSTAR
        Ps.append(p[live])
        Es.append(C[live] * (2.4 * (1 - k / len(lags)) ** 1.5))
    # those still waiting on the ground keep glowing
    wait = u <= 0
    Ps.append(S[wait])
    Es.append(C[wait] * 0.9)
    return np.concatenate(Ps), np.concatenate(Es), arrived


def I7(tl, tg, R, W, H):
    p = ease(seg(tg, 64.0, 71.0))
    cam = Camera(np.array([0.0, 1.0, 44.0 - 2.0 * p]), np.array([0.0, 6.4, 0.0]), focal=50, W=W, H=H)
    back, sd = _orbit_bg(R, cam, W, H, tg)
    P, Ev, arrived = _streams(tg, cam.pos, sd)
    R.draw(cam, P, Ev, size_px=0.7)
    b = arrived ** 1.3
    messenger(R, cam, CSTAR, 0.15 + 2.6 * b, tg)
    return back + R.resolve(), Grade(**GRADE_I)


def I8(tl, tg, R, W, H):
    shake = np.zeros(3)
    if tg > E_["front_pass"] - 0.05:
        shake = np.random.default_rng(int(tg * 24)).normal(0, 0.05, 3) * np.exp(-(tg - E_["front_pass"]) / 0.25)
    cam = Camera(np.array([0.0, 1.2, 30.0]) + shake, np.array([0.0, 6.4, 0.0]) + shake * 0.5, focal=30, W=W, H=H)
    back, sd = _orbit_bg(R, cam, W, H, tg)
    tb = E_["burst"]
    if tg < tb:
        messenger(R, cam, CSTAR, 2.8 + 6.0 * seg(tg, 71.0, tb), tg)
    else:
        r = SHELL_V * (tg - tb)
        shell_asset().draw(R, cam, CSTAR, max(r, 1e-3), t=tg, energy=0.05)
        flash = np.exp(-(tg - tb) / 0.45)
        messenger(R, cam, CSTAR, 6.0 * flash, tg)
    img = back + R.resolve()
    if tg >= tb:
        import cv2
        x, y, _, _, _ = cam.project(CSTAR[None])
        yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
        rr = np.sqrt((xx - x[0]) ** 2 + (yy - y[0]) ** 2) / W
        f = np.exp(-(tg - tb) / 0.16)
        glow = (np.exp(-rr / 0.10) * 5.0 + np.exp(-rr / 0.40) * 0.8) * f
        full = 2.5 * np.exp(-(tg - tb) / 0.045)
        img = img + (glow + full)[..., None].astype(np.float32) * np.array([1.0, 0.88, 0.62], np.float32)
    w = ease_in(seg(tg, E_["front_pass"] - 0.35, E_["front_pass"] + 0.15), 2.0)
    img = img * (1 - w) + np.float32(w * 30.0) * np.array([1.0, 0.95, 0.85], np.float32)
    return img, Grade(**GRADE_I)


def I9(tl, tg, R, W, H):
    """Mirror of I1: same framing; the night side is dark now; the shell passes as a thin gold ring."""
    cam = Camera(pos=(0.0, 0.0, 224.0), target=(0.0, 0.0, 0.0), focal=35, W=W, H=H)
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
    back *= (1.0 - bodies.sphere_mask(cam, PLANET_C, PLANET_R, W, H))[..., None]
    star_dir = (STAR_C - PLANET_C) / np.linalg.norm(STAR_C - PLANET_C)
    P, C = planet().rim(PLANET_C, PLANET_R, cam.pos, star_dir, energy=1.4)
    R.draw(cam, P, C, size_px=0.8, soft=True)
    # the shell: a thin, faint golden ring sweeping outward from above the dark planet
    c = PLANET_C + np.array([0.0, PLANET_R * 1.95, 0.0])
    r = 34.0 + 16.0 * (tg - 76.0)
    shell_asset().draw(R, cam, c, r, t=tg, energy=0.010)
    return back + R.resolve(), Grade(**GRADE_I)


TABLE.update({"I7": I7, "I8": I8, "I9": I9})
