"""Helpers shared by shot modules."""
import cv2
import numpy as np

from ..post import Grade


def black(W, H):
    return np.zeros((H, W, 3), np.float32)


def placeholder(sid, tl, W, H):
    img = np.zeros((H, W, 3), np.float32)
    cv2.putText(img, f"{sid}  t={tl:5.2f}", (int(W * 0.05), int(H * 0.5)), cv2.FONT_HERSHEY_SIMPLEX,
                1.2 * W / 1920, (0.3, 0.3, 0.3), max(1, int(2 * W / 1920)), cv2.LINE_AA)
    return img, Grade(bloom=0, grain=0, vignette=0)


def dispatch(table, sid, tl, tg, R, W, H):
    fn = table.get(sid)
    if fn is None:
        return placeholder(sid, tl, W, H)
    return fn(tl, tg, R, W, H)


def stars_layer(R, cam, field, energy=1.0, size_px=0.0, min_flux=0.0):
    d, c = field["dirs"], field["rgb"]
    if min_flux:
        k = c.max(axis=1) > min_flux
        d, c = d[k], c[k]
    R.draw_dirs(cam, d, c, size_px=size_px, energy=energy)
