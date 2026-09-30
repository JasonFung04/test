"""Colour helpers. Everything inside the renderer is linear-light RGB."""
import numpy as np


def srgb_to_linear(c):
    c = np.asarray(c, dtype=np.float32)
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4).astype(np.float32)


def linear_to_srgb(c):
    c = np.clip(c, 0.0, None)
    return np.where(c <= 0.0031308, c * 12.92, 1.055 * np.power(c, 1 / 2.4) - 0.055)


def hex_lin(h, intensity=1.0):
    """'#FFD27A' -> linear RGB float32[3] scaled by intensity."""
    h = h.lstrip("#")
    rgb = np.array([int(h[i:i + 2], 16) / 255.0 for i in (0, 2, 4)], dtype=np.float32)
    return srgb_to_linear(rgb) * np.float32(intensity)


def hex_srgb(h):
    h = h.lstrip("#")
    return np.array([int(h[i:i + 2], 16) / 255.0 for i in (0, 2, 4)], dtype=np.float32)


def blackbody(T):
    """Approximate linear RGB of a blackbody at temperature T (K), max channel = 1.

    Planck radiance sampled at representative R/G/B wavelengths; good enough for
    tinting stars and plasma."""
    T = np.asarray(T, dtype=np.float64)[..., None]
    lam = np.array([610e-9, 550e-9, 465e-9])
    h, c, k = 6.626e-34, 2.998e8, 1.381e-23
    B = 1.0 / (lam ** 5 * (np.exp(h * c / (lam * k * T)) - 1.0))
    B = B / B.max(axis=-1, keepdims=True)
    return B.astype(np.float32)


def mix(a, b, t):
    return a + (b - a) * t


def desaturate(rgb, amount):
    lum = rgb @ np.array([0.2126, 0.7152, 0.0722], dtype=np.float32)
    return rgb + (lum[..., None] - rgb) * amount


# Palette (see film/BIBLE.md §2). Values are sRGB hex; use hex_lin() to get linear.
PAL = {
    "ui_text": "#E9E4DA", "ui_dim": "#6F6B64", "gold": "#FFD27A",
    "giant_core": "#FF5A1F", "giant_limb": "#8A1406", "giant_hi": "#FFB36B",
    "alien_gold": "#FFC46B", "alien_eye": "#FFF1C9", "alien_sky": "#0B0306",
    "elder_core": "#FFE3A6", "elder_edge": "#FF9B3D",
    "neb_magenta": "#D6336C", "neb_teal": "#2BB3A8", "space": "#03040A",
    "gal_core": "#FFE6B0", "gal_arm": "#9DB7FF", "gal_hii": "#FF6FA3",
    "fire": "#FF8A2A", "ochre": "#B5391E", "cave_shadow": "#3A1A10", "mw_sea": "#CFD8FF",
    "sodium": "#FF9F3A", "led": "#EAF2FF", "tail": "#FF3B2F", "head": "#FFF6E0", "skyglow": "#3A2412",
    "night": "#02040C", "star": "#CFE0FF", "milkyway": "#F3EAD7", "cold": "#0A1630",
    "torch": "#F4F8FF", "hoodie_sodium": "#E8B830", "hoodie_star": "#C9A43A", "iris": "#6B5646",
    "card_cn": "#EDE7DB", "card_en": "#B9B2A5", "slug": "#D8D0C2", "museum": "#CFC7B8", "title": "#F2ECE0",
}
