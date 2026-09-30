"""Typography: bilingual title cards, terminal lines, glyph -> point sampling.

Text is rendered with Pillow at 3x supersampling and composited in display space.
"""
from functools import lru_cache

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .color import hex_srgb

NOTO = "/usr/share/fonts/opentype/noto/"
FONTS = {
    "serif": (NOTO + "NotoSerifCJK-Regular.ttc", 2),
    "serif_light": (NOTO + "NotoSerifCJK-Light.ttc", 2),
    "serif_medium": (NOTO + "NotoSerifCJK-Medium.ttc", 2),
    "serif_semibold": (NOTO + "NotoSerifCJK-SemiBold.ttc", 2),
    "serif_bold": (NOTO + "NotoSerifCJK-Bold.ttc", 2),
    "sans": (NOTO + "NotoSansCJK-Regular.ttc", 2),
    "sans_mono": (NOTO + "NotoSansCJK-Regular.ttc", 7),
    "garamond": ("/usr/share/fonts/opentype/ebgaramond/EBGaramond12-Regular.otf", 0),
    "garamond_it": ("/usr/share/fonts/opentype/ebgaramond/EBGaramond12-Italic.otf", 0),
    "plex": ("/usr/share/fonts/truetype/ibm-plex/IBMPlexSans-Regular.ttf", 0),
    "plex_light": ("/usr/share/fonts/truetype/ibm-plex/IBMPlexSans-Light.ttf", 0),
    "plex_xlight": ("/usr/share/fonts/truetype/ibm-plex/IBMPlexSans-ExtraLight.ttf", 0),
    "mono": ("/usr/share/fonts/truetype/jetbrains-mono/JetBrainsMono-Regular.ttf", 0),
    "mono_light": ("/usr/share/fonts/truetype/jetbrains-mono/JetBrainsMono-Light.ttf", 0),
}
SS = 3  # supersampling


@lru_cache(maxsize=256)
def font(key, size_px):
    path, idx = FONTS[key]
    return ImageFont.truetype(path, max(1, int(round(size_px))), index=idx)


def _is_cjk(ch):
    o = ord(ch)
    return (0x2E80 <= o <= 0x9FFF) or (0x3000 <= o <= 0x303F) or (0xFF00 <= o <= 0xFFEF) or (0xF900 <= o <= 0xFAFF)


# fallbacks for Latin-first fonts when a CJK glyph appears
CJK_FALLBACK = {"mono": "sans_mono", "mono_light": "sans_mono", "garamond": "serif", "garamond_it": "serif",
                "plex": "sans", "plex_light": "sans", "plex_xlight": "sans"}


@lru_cache(maxsize=512)
def line_alpha(text, key, size_px, tracking_em=0.0, scale=1.0):
    """Render one line. Returns (alpha float32 HxW, baseline_y, advance_width) in output px.

    size_px and tracking are authored @1920 wide; `scale` = W/1920."""
    size = size_px * scale * SS
    f_main = font(key, size)
    f_cjk = font(CJK_FALLBACK.get(key, key), size)
    asc, desc = f_main.getmetrics()
    asc2, desc2 = f_cjk.getmetrics()
    asc, desc = max(asc, asc2), max(desc, desc2)
    track = tracking_em * size
    # measure
    xs, w = [], 0.0
    for ch in text:
        f = f_cjk if _is_cjk(ch) else f_main
        xs.append((w, f))
        w += f.getlength(ch) + track
    w = max(w - track, 1.0)
    pad = int(size * 0.3) + 4
    Wc, Hc = int(np.ceil(w)) + 2 * pad, asc + desc + 2 * pad
    img = Image.new("L", (Wc, Hc), 0)
    d = ImageDraw.Draw(img)
    base = pad + asc
    for ch, (x0, f) in zip(text, xs):
        d.text((pad + x0, base), ch, font=f, fill=255, anchor="ls")
    a = np.asarray(img, np.float32) / 255.0
    out_w = max(1, int(round(Wc / SS)))
    out_h = max(1, int(round(Hc / SS)))
    a = cv2.resize(a, (out_w, out_h), interpolation=cv2.INTER_AREA)
    return a, base / SS, w / SS, pad / SS


def place(canvas_alpha, a, x, y):
    """Add alpha image `a` into canvas at top-left (x, y) (floats allowed -> subpixel via warp)."""
    H, W = canvas_alpha.shape
    ix, iy = int(np.floor(x)), int(np.floor(y))
    fx, fy = x - ix, y - iy
    if fx or fy:
        M = np.float32([[1, 0, fx], [0, 1, fy]])
        a = cv2.warpAffine(a, M, (a.shape[1] + 1, a.shape[0] + 1), flags=cv2.INTER_LINEAR)
    h, w = a.shape
    x0, y0 = max(ix, 0), max(iy, 0)
    x1, y1 = min(ix + w, W), min(iy + h, H)
    if x1 <= x0 or y1 <= y0:
        return
    canvas_alpha[y0:y1, x0:x1] = np.maximum(canvas_alpha[y0:y1, x0:x1], a[y0 - iy:y1 - iy, x0 - ix:x1 - ix])


def draw_line(canvas, text, key, size_px, x, baseline, tracking_em=0.0, align="left", scale=1.0):
    """Draw into a float alpha canvas. x/baseline in output px. Returns advance width."""
    a, base, w, pad = line_alpha(text, key, size_px, tracking_em, scale)
    if align == "center":
        x = x - w / 2
    elif align == "right":
        x = x - w
    place(canvas, a, x - pad, baseline - base)
    return w


def envelope(t, t0, t1, fin=0.8, fout=0.8):
    """Card opacity with eased fades."""
    if t < t0 or t > t1:
        return 0.0
    a = min(1.0, (t - t0) / fin) if fin > 0 else 1.0
    b = min(1.0, (t1 - t) / fout) if fout > 0 else 1.0
    a = 1 - (1 - a) ** 2
    b = 1 - (1 - b) ** 2
    return float(max(0.0, min(a, b)))


def glow_composite(y, alpha, color_hex, opacity, glow=0.35, glow_sigma=6.0):
    if opacity <= 0:
        return y
    col = hex_srgb(color_hex)
    s = y.shape[1] / 1920.0
    a = alpha * opacity
    if glow > 0:
        g = cv2.GaussianBlur(alpha, (0, 0), glow_sigma * s) * (glow * opacity)
        y = y + g[..., None] * col * 0.6
    return y * (1 - a[..., None]) + col * a[..., None]


def card_layers(W, H, card, t):
    """Returns list of (alpha, color_hex, opacity) for a card dict at time t (global)."""
    op = envelope(t, card["t0"], card["t1"], card.get("fin", 0.8), card.get("fout", 0.8))
    if op <= 0:
        return []
    s = W / 1920.0
    style = card.get("style", "center")
    life = (t - card["t0"]) / max(card["t1"] - card["t0"], 1e-6)
    drift = -0.01 * H * life if card.get("drift", True) else 0.0
    layers = []
    if style in ("center", "black"):
        yb = (0.70 if style == "center" else 0.50) * H + drift
        if card.get("y") is not None:
            yb = card["y"] * H + drift
        a_cn = np.zeros((H, W), np.float32)
        draw_line(a_cn, card["cn"], "serif", 36, W / 2, yb, 0.08, "center", s)
        layers.append((a_cn, "#EDE7DB", op))
        if card.get("en"):
            a_en = np.zeros((H, W), np.float32)
            draw_line(a_en, card["en"], "garamond_it", 26, W / 2, yb + 46 * s, 0.02, "center", s)
            layers.append((a_en, "#B9B2A5", op))
    elif style in ("slug", "slug_r"):
        right = style == "slug_r"
        x = W - 96 * s if right else 96 * s
        al = "right" if right else "left"
        yb = H - 92 * s + drift
        a_cn = np.zeros((H, W), np.float32)
        draw_line(a_cn, card["cn"], "serif", 25, x, yb - 38 * s, 0.10, al, s)
        layers.append((a_cn, "#DDD5C7", op))
        a_en = np.zeros((H, W), np.float32)
        draw_line(a_en, card["en"], "garamond_it", 22, x, yb, 0.02, al, s)
        layers.append((a_en, "#B3AB9E", op))
    elif style == "museum":
        xr = W - 96 * s
        yb = H - 92 * s
        a = np.zeros((H, W), np.float32)
        w1 = draw_line(a, card["cn"], "sans", 17, xr, yb - 30 * s, 0.06, "right", s)
        w2 = draw_line(a, card["en"], "plex_light", 16, xr, yb, 0.03, "right", s)
        ww = max(w1, w2)
        y_rule = int(yb - 58 * s)
        a[y_rule:y_rule + max(1, int(round(s))), int(xr - ww):int(xr)] = 0.8
        layers.append((a, "#CFC7B8", op))
    return layers


def apply_cards(y, cards, t):
    H, W = y.shape[:2]
    for c in cards:
        for alpha, col, op in card_layers(W, H, c, t):
            y = glow_composite(y, alpha, col, op, glow=c.get("glow", 0.35))
    return y


def text_points(text, key, size_px, tracking_em=0.0, step=1.0, thresh=0.5, scale=1.0, seed=0):
    """Sample a rendered line into points. Returns (N,2) px offsets relative to (left, baseline), weights."""
    a, base, w, pad = line_alpha(text, key, size_px, tracking_em, scale)
    ys, xs = np.nonzero(a > thresh * 0.5)
    rng = np.random.default_rng(seed)
    keep = rng.random(xs.size) < (1.0 / (step * step))
    xs, ys = xs[keep], ys[keep]
    jitter = rng.random((xs.size, 2)) - 0.5
    pts = np.stack([xs + jitter[:, 0] - pad, ys + jitter[:, 1] - base], axis=1).astype(np.float32)
    wts = a[ys, xs].astype(np.float32)
    return pts, wts, w
