"""OPENING (O1–O6): cursor, '#', the credit panes, the city of text, the galaxy, the title.

All UI text is made of points of light (so it can dissolve into the galaxy)."""
import numpy as np

from .. import timeline as TL
from ..assets import sky
from ..camera import Camera, ease, ease5, ease_in, ease_out, lerp, seg
from ..color import hex_lin
from ..config import ROOT
from ..noise import curl, hash01
from ..post import Grade
from ..text import char_points
from .common import dispatch

K = 0.001                        # world units per pane pixel
TEXT = hex_lin("#E9E4DA")
DIM = hex_lin("#6F6B64")
MID = hex_lin("#B9B2A5")
GOLD = hex_lin("#FFD27A")
E = TL.EVENTS
TYPING = TL.typing_schedule()
G0 = np.array([1.2, -0.2, -5.0])  # galaxy centre (world)
GR = 7.0                          # galaxy radius (world units)

_C = {}


def _rot_y(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


def _rot_x(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])


class Pane:
    """A floating terminal pane. Pane pixel coords: origin top-left, x right, y down."""

    def __init__(self, center, size_px, yaw=0.0, pitch=0.0):
        self.c = np.asarray(center, float)
        self.w, self.h = size_px
        self.R = _rot_y(yaw) @ _rot_x(pitch)
        self.P, self.col, self.ton, self.kind, self.step = [], [], [], [], []

    def to_world(self, xy):
        loc = np.stack([(xy[:, 0] - self.w / 2) * K, -(xy[:, 1] - self.h / 2) * K, np.zeros(len(xy))], 1)
        return loc @ self.R.T + self.c

    def text(self, s, x, y, key, size, col, times=None, step=1.3, seed=0, tracking=0.0, dense_first=None):
        pts, xs, adv = char_points(s, key, size, tracking, step=step, seed=seed)
        if dense_first:
            p0, _, _ = char_points(s[0], key, size, tracking, step=dense_first, seed=seed + 999)
            pts[0] = p0[0]
        for i, p in enumerate(pts):
            if len(p) == 0:
                continue
            xy = p + np.array([x, y])
            self.P.append(self.to_world(xy))
            self.col.append(np.broadcast_to(col, (len(p), 3)))
            self.ton.append(np.full(len(p), -1.0 if times is None else times[i]))
            self.kind.append(np.zeros(len(p)))
            st = step if not (dense_first and i == 0) else dense_first
            self.step.append(np.full(len(p), st))
        return xs, adv

    def frame(self, col=DIM * 0.55, spacing=7.0, header=None):
        n_w, n_h = int(self.w / spacing), int(self.h / spacing)
        xs = np.linspace(0, self.w, n_w)
        ys = np.linspace(0, self.h, n_h)
        edge = np.concatenate([np.stack([xs, np.zeros_like(xs)], 1), np.stack([xs, np.full_like(xs, self.h)], 1),
                               np.stack([np.zeros_like(ys), ys], 1), np.stack([np.full_like(ys, self.w), ys], 1)])
        self.P.append(self.to_world(edge))
        self.col.append(np.broadcast_to(col, (len(edge), 3)))
        self.ton.append(np.full(len(edge), -1.0))
        self.kind.append(np.ones(len(edge)))
        self.step.append(np.full(len(edge), spacing))
        if header:
            self.text(header, 28, 44, "mono_light", 22, DIM * 0.8, step=1.6, seed=5)
            sep = np.stack([np.linspace(0, self.w, int(self.w / 5)), np.full(int(self.w / 5), 62.0)], 1)
            self.P.append(self.to_world(sep))
            self.col.append(np.broadcast_to(col * 0.7, (len(sep), 3)))
            self.ton.append(np.full(len(sep), -1.0))
            self.kind.append(np.ones(len(sep)))
            self.step.append(np.full(len(sep), spacing))

    def arrays(self):
        return (np.concatenate(self.P).astype(np.float32), np.concatenate(self.col).astype(np.float32),
                np.concatenate(self.ton).astype(np.float32), np.concatenate(self.kind).astype(np.int8),
                np.concatenate(self.step).astype(np.float32))


def _times(line):
    return [t for t, ch, l in TYPING if l == line]


def _lines_for_city(n, seed):
    """Dim text for the background panes: the film's own screenplay and source code."""
    rng = np.random.default_rng(seed)
    src = []
    for f in ("film/SCREENPLAY.md", "engine/raster.py", "engine/camera.py", "film/SHOTLIST.md"):
        try:
            for ln in (ROOT / f).read_text().splitlines():
                ln = ln.strip().strip("#>*").strip()
                if 6 < len(ln) < 60 and "http" not in ln:
                    src.append(ln)
        except OSError:
            pass
    return [src[i] for i in rng.integers(0, len(src), n)]


def scene():
    if "scene" in _C:
        return _C["scene"]
    # ---- P0: the prompt pane
    p0 = Pane((0.0, 0.0, 0.0), (1300, 620))
    p0.frame(header="~/light-years")
    t_cn = _times("prompt_cn")
    xs0, _ = p0.text(TL.PROMPT_CN, 60, 185, "mono", 72, TEXT, times=t_cn, step=1.15, seed=1, dense_first=0.30)
    xs1, _ = p0.text(TL.PROMPT_EN, 60, 275, "mono", 44, DIM * 1.25, times=_times("prompt_en"), step=1.3, seed=2)
    xs2, _ = p0.text(TL.STATS, 60, 430, "mono", 40, DIM * 1.25, times=_times("stats"), step=1.3, seed=3)
    xs3, _ = p0.text(TL.SENDING, 60, 525, "mono", 54, GOLD * 1.1, times=_times("sending"), step=1.2, seed=4)
    # ---- credit panes
    creds = []
    pos = [(1.55, 0.06, -0.40), (2.78, -0.12, -1.02), (4.00, 0.05, -1.64), (5.22, -0.10, -2.26)]
    for k, (name, cn, en) in enumerate(TL.CREDIT_LINES):
        pc = Pane(pos[k], (980, 330), yaw=-0.21)
        pc.frame()
        tn = _times(f"credit{k}")
        pc.text(name, 56, 118, "mono", 64, TEXT, times=tn, step=1.2, seed=10 + k)
        t_role = tn[-1] + 0.25
        pc.text(cn, 58, 206, "sans_mono", 40, MID * 1.1, times=[t_role + 0.05 * i for i in range(len(cn))],
                step=1.2, seed=20 + k)
        pc.text(en, 58, 268, "mono_light", 30, DIM * 1.2, times=[t_role + 0.35 + 0.012 * i for i in range(len(en))],
                step=1.3, seed=30 + k)
        creds.append(pc)
    # ---- the city of text
    rng = np.random.default_rng(77)
    bg = []
    lines = _lines_for_city(26 * 7, 5)
    li = 0
    for k in range(26):
        c = np.array([rng.uniform(-3.5, 7.5), rng.uniform(-2.3, 2.3), rng.uniform(-9.5, -2.2)])
        pb = Pane(c, (rng.uniform(800, 1400), rng.uniform(300, 520)), yaw=rng.uniform(-0.45, 0.2),
                  pitch=rng.uniform(-0.08, 0.08))
        pb.frame(col=DIM * 0.35, spacing=10.0)
        for j in range(int(rng.integers(4, 8))):
            pb.text(lines[li % len(lines)][:44], 30, 60 + j * 52, "mono_light", 34, DIM * 0.75, step=2.3, seed=100 + li)
            li += 1
        bg.append(pb)
    parts = [p0] + creds + bg
    arrs = [p.arrays() for p in parts]
    P = np.concatenate([a[0] for a in arrs])
    col = np.concatenate([a[1] for a in arrs])
    ton = np.concatenate([a[2] for a in arrs])
    kind = np.concatenate([a[3] for a in arrs])
    step = np.concatenate([a[4] for a in arrs])
    pid = np.concatenate([np.full(len(a[0]), i) for i, a in enumerate(arrs)])
    # cursor anchor positions (world) for the prompt pane lines
    cur = {
        "cn": [p0.to_world(np.array([[60 + x, 185.0]]))[0] for x in xs0] + [p0.to_world(np.array([[60 + xs0[-1] + 72 * 1.0, 185.0]]))[0]],
        "en_end": p0.to_world(np.array([[60 + xs1[-1] + 44 * 0.62, 275.0]]))[0],
        "stats_end": p0.to_world(np.array([[60 + xs2[-1] + 40 * 0.62, 430.0]]))[0],
        "send_end": p0.to_world(np.array([[60 + xs3[-1] + 54 * 0.62, 525.0]]))[0],
        "stats": [p0.to_world(np.array([[60 + x, 430.0]]))[0] for x in xs2] +
                 [p0.to_world(np.array([[60 + xs2[-1] + 40 * 0.62, 430.0]]))[0]],
        "send": [p0.to_world(np.array([[60 + x, 525.0]]))[0] for x in xs3] +
                [p0.to_world(np.array([[60 + xs3[-1] + 54 * 0.62, 525.0]]))[0]],
    }
    # galaxy targets for the morph
    gal = sky.spiral_galaxy()
    n = len(P)
    grng = np.random.default_rng(3)
    gi = grng.choice(len(gal["pos"]), n, replace=False)
    _C["scene"] = dict(P=P, col=col, ton=ton, kind=kind, pid=pid, p0=p0, creds=creds, cur=cur, gi=gi, gal=gal,
                       step=step)
    _C["gal_basis"] = _galaxy_basis()
    return _C["scene"]


def _galaxy_basis():
    tilt = np.radians(24)
    return _rot_x(tilt) @ _rot_y(0.3)


def galaxy_world(pos, t_rot=0.0):
    Rb = _C.get("gal_basis", _galaxy_basis())
    Rspin = _rot_y(t_rot)
    return (pos @ Rspin.T) * GR @ Rb.T + G0


def cursor_points(anchor, R_pane, size_px, t, on=True):
    """A block cursor of points at `anchor` (world, baseline-left)."""
    if not on:
        return np.zeros((0, 3), np.float32)
    w, h = size_px * 0.07, size_px * 1.05
    n = int(w * h / 0.35)
    rng = np.random.default_rng(9)
    xy = np.stack([rng.random(n) * w, -rng.random(n) * h * 0.95 + size_px * 0.2], 1)
    loc = np.stack([xy[:, 0] * K, -xy[:, 1] * K, np.zeros(n)], 1)
    return (loc @ R_pane.T + anchor).astype(np.float32)


def blink(t, t0, period=0.53):
    return ((t - t0) % period) < period * 0.5


def draw_text(R, cam, S, t, alpha=1.0, energy=1.0, size_scale=0.42, spacing_world=None, exclude_pid=None,
              frames=True):
    """Draw typed text points with projected-area energy (constant apparent brightness)."""
    P, col, ton = S["P"], S["col"], S["ton"]
    vis = (ton <= t)
    if not frames:
        vis &= S["kind"] == 0
    if exclude_pid is not None:
        vis &= ~np.isin(S["pid"], exclude_pid)
    idx = np.nonzero(vis)[0]
    if idx.size == 0:
        return
    age = np.where(ton[idx] < 0, 9.0, t - ton[idx])
    flash = 1.0 + 1.2 * np.exp(-age / 0.06)          # phosphor flash at birth
    fade_in = np.clip(age / 0.05, 0, 1)
    x, y, z, coc, ok = cam.project(P[idx])
    sp = S["step"][idx] * K * cam.fpx / np.maximum(z, 1e-6)       # projected spacing (px)
    e = col[idx] * (sp ** 2 * flash * fade_in * alpha * energy / R.e)[:, None]
    R.draw(cam, P[idx], e, size_px=np.maximum(sp / R.s * size_scale, 0.45))


# ------------------------------------------------------------------------------ shots
def O1(tl, tg, R, W, H):
    return np.zeros((H, W, 3), np.float32), Grade(grain=0.0, vignette=0.0, bloom=0.0)


GRADE_UI = dict(grain=0.004, vignette=0.12, bloom=0.9, sat=1.0)


def O2(tl, tg, R, W, H):
    S = scene()
    cur = S["cur"]
    hash_c = cur["cn"][0] + np.array([0.022, 0.026, 0.0])
    p = ease5(seg(tg, E["hash_key"], 7.0))
    d = 0.95 - 0.55 * p
    drift = np.array([-0.006 + 0.012 * seg(tg, 2.0, 7.0), 0.002 * np.sin(tg * 0.7), 0.0])
    cam = Camera(hash_c + drift + np.array([0, 0, d]), hash_c + drift * 0.6, focal=100, W=W, H=H,
                 focus=d, bokeh=18)
    draw_text(R, cam, S, tg, size_scale=0.36, frames=False)
    # cursor: blinks before the '#', then sits after it
    if tg < E["hash_key"]:
        on = any(b <= tg < b + 0.265 for b in E["cursor_blinks"]) or tg >= 3.9
        anchor = cur["cn"][0]
    else:
        on = blink(tg, E["hash_key"] + 0.1) or tg < E["hash_key"] + 0.1
        anchor = cur["cn"][1]
    C = cursor_points(anchor, S["p0"].R, 72, tg, on)
    if len(C):
        sp = 0.30 * K * cam.fpx / d
        R.draw(cam, C, TEXT * 0.75 * sp ** 2 / R.e, size_px=max(sp * 0.4 / R.s, 0.5))
    return R.resolve(), Grade(**GRADE_UI)


def O3(tl, tg, R, W, H):
    S = scene()
    cur = S["cur"]
    hash_c = cur["cn"][0] + np.array([0.022, 0.026, 0.0])
    line_c = cur["cn"][0] + np.array([0.24, -0.02, 0.0])
    p = ease5(seg(tg, 7.0, 10.5))
    d = lerp(0.62, 1.30, p)
    tgt = lerp(hash_c, line_c + np.array([0.0, -0.035, 0]), ease(seg(tg, 7.2, 10.0)))
    cam = Camera(tgt + np.array([-0.05 + 0.08 * p, 0.03, d]), tgt, focal=50, W=W, H=H, roll=np.radians(-2 + 2 * p),
                 focus=d, bokeh=10)
    draw_text(R, cam, S, tg)
    # cursor follows typing
    tcn = _times("prompt_cn")
    ten = _times("prompt_en")
    if tg < tcn[-1] + 0.2:
        k = sum(1 for tt in tcn if tt <= tg)
        anchor = S["cur"]["cn"][min(k, len(S["cur"]["cn"]) - 1)]
        size, on = 72, True
    else:
        anchor = S["cur"]["en_end"] if tg >= ten[-1] else S["cur"]["en_end"]
        size, on = 44, blink(tg, ten[-1]) or tg < ten[-1]
        if tg < ten[0]:
            anchor, size = S["cur"]["cn"][-1], 72
    C = cursor_points(anchor, S["p0"].R, size, tg, on)
    if len(C):
        sp = 0.59 * K * cam.fpx / d
        R.draw(cam, C, TEXT * 0.75 * sp ** 2 / R.e, size_px=max(sp * 0.45 / R.s, 0.45))
    return R.resolve(), Grade(**GRADE_UI)


def O4(tl, tg, R, W, H):
    S = scene()
    creds = S["creds"]
    # dwell on each pane while it types, glide between them (one continuous move)
    n = len(creds)
    ft = list(TL.CREDIT_T)
    keys_t, keys_i = [10.5], [-0.45]
    for k in range(n):
        keys_t += [ft[k], (ft[k + 1] - 0.5) if k + 1 < n else 17.5]
        keys_i += [float(k), float(k) + 0.12]
    keys_t.append(17.6)
    keys_i.append(n - 1 + 0.2)
    fi = float(np.interp(tg, keys_t, keys_i))
    for k in range(n - 1):
        a_, b_ = ft[k + 1] - 0.5, ft[k + 1]
        if a_ <= tg <= b_:
            i0, i1 = k + 0.12, k + 1.0
            fi = i0 + (i1 - i0) * ease5((tg - a_) / (b_ - a_))
    fi_c = np.clip(fi, 0, n - 1)
    k0 = int(np.floor(fi_c))
    k1 = min(k0 + 1, n - 1)
    f = fi_c - k0
    c = creds[k0].c * (1 - f) + creds[k1].c * f + np.array([0.18 * (fi - fi_c), 0, 0])
    cam_pos = c + np.array([-0.30, 0.03, 1.28])
    tgt = c + np.array([0.02, -0.01, 0.0])
    # focus: rack to the pane whose line is being typed
    focus = float(np.linalg.norm(c - cam_pos))
    cam = Camera(cam_pos, tgt, focal=35, W=W, H=H, focus=focus, bokeh=22)
    draw_text(R, cam, S, tg, energy=0.95)
    return R.resolve(), Grade(**GRADE_UI)


def _o5_cam(tg, W, H):
    p = ease_out(seg(tg, 17.5, 20.5), 3.0)
    pos = lerp([4.1, 0.05, 0.25], [-0.25, 0.28, 2.25], p)
    tgt = lerp([4.6, -0.05, -1.2], [0.55, -0.12, -1.3], p)
    return Camera(pos, tgt, focal=24, W=W, H=H, focus=float(np.linalg.norm(pos - np.array([0.0, -0.1, 0.0]))),
                  bokeh=8)


def O5(tl, tg, R, W, H):
    S = scene()
    cam = _o5_cam(tg, W, H)
    draw_text(R, cam, S, tg, energy=1.0)
    ts, tsend = _times("stats"), _times("sending")
    if tg >= tsend[0]:
        k = sum(1 for tt in tsend if tt <= tg)
        anchor, size = S["cur"]["send"][min(k, len(S["cur"]["send"]) - 1)], 54
        on = blink(tg, tsend[-1] + 0.05) if tg >= tsend[-1] else True
    elif tg >= ts[0]:
        k = sum(1 for tt in ts if tt <= tg)
        anchor, size, on = S["cur"]["stats"][min(k, len(S["cur"]["stats"]) - 1)], 40, True
    else:
        anchor, size, on = S["cur"]["en_end"], 44, blink(tg, 17.5)
    C = cursor_points(anchor, S["p0"].R, size, tg, on)
    if len(C):
        d = np.linalg.norm(anchor - cam.pos)
        sp = 0.59 * K * cam.fpx / d
        R.draw(cam, C, GOLD * 0.9 * sp ** 2 / R.e, size_px=max(sp * 0.45 / R.s, 0.45))
    return R.resolve(), Grade(**GRADE_UI)


def _o6_cam(tg, W, H):
    c0 = _o5_cam(20.5, W, H)
    p = ease5(seg(tg, 20.5, 23.6))
    look = lerp(c0.target, G0 + np.array([0, -0.9, 0]), ease(seg(tg, 20.5, 23.0)))
    back = lerp(c0.pos, G0 + np.array([0.5, 3.4, 24.5]), p)
    drift = np.array([0.25, 0.08, 0.0]) * seg(tg, 23.5, 27.0)
    return Camera(back + drift, look + drift * 0.3, focal=lerp(24, 32, p), W=W, H=H, bokeh=0)


def O6(tl, tg, R, W, H):
    S = scene()
    gal = S["gal"]
    cam = _o6_cam(tg, W, H)
    fade = 1.0 - seg(tg, E["title_out"], E["cut_black"])
    if tg >= E["cut_black"]:
        return np.zeros((H, W, 3), np.float32), Grade(grain=0.0, vignette=0.0, bloom=0.0)
    spin = 0.018 * (tg - 22.0)
    # ---- text points: crumble like sand in a left-to-right wave, then spiral into the galaxy
    P0, col0 = S["P"], S["col"]
    vis = S["ton"] <= tg
    P0, col0, pidv = P0[vis], col0[vis], np.nonzero(vis)[0]
    if "stag" not in _C:
        cend = _o5_cam(20.5, 960, 402)
        xx, _, _, _, _ = cend.project(S["P"])
        xn = np.clip(xx / 960.0, 0, 1)
        _C["stag"] = (0.75 * xn + 0.15 * hash01(np.arange(len(S["P"])), 3)).astype(np.float32)
        rng = np.random.default_rng(4)
        v = rng.normal(0, 1, (len(S["P"]), 3))
        v[:, 1] = np.abs(v[:, 1]) * 1.4 + 0.3
        v[:, 2] -= 0.6
        _C["vdir"] = (v / np.linalg.norm(v, axis=1, keepdims=True)).astype(np.float32)
    stagger = _C["stag"][pidv]
    a = np.clip((tg - E["dissolve_start"] - stagger) / 1.0, 0, 1)
    Pd = P0 + _C["vdir"][pidv] * (0.10 * ease_out(a, 2.0))[:, None] + \
        curl(P0 * 2.3, scale=1.0, t=tg * 0.3) * (0.06 * a ** 2)[:, None]
    tgt = galaxy_world(gal["pos"][S["gi"][pidv]], spin)
    m = ease5(np.clip((tg - 21.25 - stagger * 0.55) / 1.9, 0, 1))
    Rb = _C["gal_basis"]
    axis = Rb @ np.array([0, 1.0, 0])
    P_lin = Pd * (1 - m)[:, None] + tgt * m[:, None]
    rel_p = P_lin - G0
    tdir = np.cross(axis, rel_p)
    tdir /= np.linalg.norm(tdir, axis=1, keepdims=True) + 1e-9
    P = P_lin + tdir * (np.linalg.norm(rel_p, axis=1) * 0.45 * np.sin(np.pi * m))[:, None]
    gcol = gal["rgb"][S["gi"][pidv]]
    col = col0 * (1 - m)[:, None] + gcol * 0.9 * m[:, None]
    z = np.maximum(cam.to_cam(P)[2], 1e-3)
    sp = 1.3 * K * cam.fpx / z
    boost = (1 + 1.1 * np.sin(np.pi * a) * (1 - m))       # sparkle as each grain detaches
    gk_t = gal["kind"][S["gi"][pidv]]
    gfac = np.where(gk_t == 0, 0.28, 1.0) * 0.30
    e = (col0 * (sp ** 2 / R.e * (1 - m) * boost)[:, None] + gcol * (gfac * m)[:, None]) * fade
    R.draw(cam, P, e, size_px=0.5)
    # ---- the rest of the galaxy fades in around them
    gm = seg(tg, 22.0, 24.2)
    if gm > 0:
        n_all = len(gal["pos"])
        keep = np.ones(n_all, bool)
        keep[S["gi"]] = False
        Pg = galaxy_world(gal["pos"][keep], spin)
        gk = gal["kind"][keep]
        gr = gal["rgb"][keep] * np.where(gk == 0, 0.22, 1.0)[:, None]
        R.draw(cam, Pg, gr * (0.30 * ease(gm) * fade), size_px=0.55)
        core = gk == 0
        R.draw(cam, Pg[core][::10], gal["rgb"][keep][core][::10] * (0.006 * ease(gm) * fade), size_px=16, soft=True)
    stars = sky.field()
    R.draw_dirs(cam, stars["dirs"], stars["rgb"], energy=0.35 * seg(tg, 21.5, 24.0) * fade)
    world = R.new_layer()
    # ---- title converges at the centre and lands on the glass tone; the galaxy dims behind it
    mask = title_layer(R, cam, tg, fade)
    if mask is not None:
        world *= (1.0 - 0.93 * mask)[..., None]
    return world + R.resolve(), Grade(grain=0.004, vignette=0.15, bloom=1.1)


def _title_points():
    if "title" in _C:
        return _C["title"]
    pts_a, xs_a, adv_a = char_points("光年", "serif_semibold", 168, 0.62, step=0.62, seed=41)
    pa = np.concatenate([p for p in pts_a if len(p)])
    pa[:, 0] -= (adv_a - 0.62 * 168) / 2
    pa[:, 1] += 8
    pts_b, xs_b, adv_b = char_points("LIGHT-YEARS", "plex_light", 32, 0.62, step=0.5, seed=42)
    pb = np.concatenate([p for p in pts_b if len(p)])
    pb[:, 0] -= (adv_b - 0.62 * 32) / 2
    pb[:, 1] += 92
    xy = np.concatenate([pa, pb])
    col = np.concatenate([np.tile(hex_lin("#F2ECE0"), (len(pa), 1)), np.tile(hex_lin("#D9D1C3"), (len(pb), 1))])
    _C["title"] = (xy.astype(np.float32), col.astype(np.float32))
    return _C["title"]


def title_layer(R, cam, tg, fade):
    t0, t1 = 22.6, E["title_tone"]
    if tg < t0:
        return None
    xy, col = _title_points()
    n = len(xy)
    # screen-space targets (title sits over the galaxy centre)
    cx, cy, _, _, _ = cam.project(G0[None])
    tx = cx[0] + xy[:, 0] * R.s
    ty = cy[0] + xy[:, 1] * R.s - 10 * R.s
    # sources: galaxy-core particles, scattered around the centre
    rng = np.random.default_rng(12)
    ang = rng.random(n) * 2 * np.pi
    rad = (rng.random(n) ** 0.5) * 260 * R.s
    sx = cx[0] + np.cos(ang) * rad * 1.6
    sy = cy[0] + np.sin(ang) * rad * 0.7
    u = ease_out(np.clip((tg - t0 - hash01(np.arange(n), 8) * 0.25) / (t1 - t0 - 0.25), 0, 1), 4.0)
    x = sx + (tx - sx) * u
    y = sy + (ty - sy) * u
    land = np.exp(-max(tg - t1, 0) / 0.35) * (tg >= t1)
    e = col * (0.62 + 0.9 * land) * (0.35 + 0.65 * u)[:, None] * fade
    R.draw2d(x, y, e * 0.72, r_px=0.6)
    # soft dark halo behind the letters (grows as they land)
    import cv2
    H_, W_ = R.H, R.W
    m = np.zeros((H_, W_), np.float32)
    x0, x1 = np.percentile(tx, 1) - 70 * R.s, np.percentile(tx, 99) + 70 * R.s
    y0, y1 = np.percentile(ty, 1) - 45 * R.s, np.percentile(ty, 99) + 40 * R.s
    cv2.ellipse(m, (int((x0 + x1) / 2), int((y0 + y1) / 2)), (int((x1 - x0) / 2), int((y1 - y0) / 2)), 0, 0, 360,
                1.0, -1)
    m = cv2.GaussianBlur(m, (0, 0), 48 * R.s)
    m = np.clip(m * 1.15, 0, 1) * float(np.mean(u)) * fade
    return m


TABLE = {"O1": O1, "O2": O2, "O3": O3, "O4": O4, "O5": O5, "O6": O6}


def render(sid, tl, tg, R, W, H):
    return dispatch(TABLE, sid, tl, tg, R, W, H)
