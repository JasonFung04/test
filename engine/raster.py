"""Point splatting with size classes.

Every point carries linear RGB *energy* and a radius in px.  Radii are handled by
splitting energy between log2-spaced size classes; each class is splatted
bilinearly (at reduced resolution for big radii), convolved once with a disk
(bokeh) or gaussian (soft glow) kernel, and upsampled.  Cost is independent of
the radius of individual points.
"""
import cv2
import numpy as np
from numba import njit

# class c (1..7) has radius 0.75 * 2**c : 1.5, 3, 6, 12, 24, 48, 96 px. class 0 = bare bilinear.
N_CLASSES = 8
DOWN = [1, 1, 1, 1, 2, 4, 8, 16]


@njit(cache=True, fastmath=True)
def _splat(buf, x, y, c, w, inv_d):
    H = buf.shape[0]
    Wd = buf.shape[1]
    for i in range(x.shape[0]):
        xi = x[i] * inv_d - 0.5
        yi = y[i] * inv_d - 0.5
        x0 = int(np.floor(xi))
        y0 = int(np.floor(yi))
        if x0 < -1 or y0 < -1 or x0 >= Wd or y0 >= H:
            continue
        fx = xi - x0
        fy = yi - y0
        wi = w[i]
        r = c[i, 0] * wi
        g = c[i, 1] * wi
        b = c[i, 2] * wi
        w00 = (1.0 - fx) * (1.0 - fy)
        w10 = fx * (1.0 - fy)
        w01 = (1.0 - fx) * fy
        w11 = fx * fy
        if y0 >= 0:
            if x0 >= 0:
                buf[y0, x0, 0] += r * w00
                buf[y0, x0, 1] += g * w00
                buf[y0, x0, 2] += b * w00
            if x0 + 1 < Wd:
                buf[y0, x0 + 1, 0] += r * w10
                buf[y0, x0 + 1, 1] += g * w10
                buf[y0, x0 + 1, 2] += b * w10
        if y0 + 1 < H:
            if x0 >= 0:
                buf[y0 + 1, x0, 0] += r * w01
                buf[y0 + 1, x0, 1] += g * w01
                buf[y0 + 1, x0, 2] += b * w01
            if x0 + 1 < Wd:
                buf[y0 + 1, x0 + 1, 0] += r * w11
                buf[y0 + 1, x0 + 1, 1] += g * w11
                buf[y0 + 1, x0 + 1, 2] += b * w11


_KCACHE = {}


def disk_kernel(rad):
    key = ("d", round(rad, 3))
    if key not in _KCACHE:
        R = int(np.ceil(rad + 1.5))
        yy, xx = np.mgrid[-R:R + 1, -R:R + 1].astype(np.float32)
        d = np.sqrt(xx * xx + yy * yy)
        k = np.clip(rad + 0.5 - d, 0.0, 1.0)
        k *= 1.0 + 0.3 * np.clip(d / max(rad, 1e-3), 0, 1) ** 4   # slight bright rim, like real glass
        _KCACHE[key] = (k / k.sum()).astype(np.float32)
    return _KCACHE[key]


def gauss_kernel(rad):
    key = ("g", round(rad, 3))
    if key not in _KCACHE:
        sig = max(rad * 0.5, 0.3)
        R = int(np.ceil(sig * 3.0))
        ax = np.arange(-R, R + 1, dtype=np.float32)
        g = np.exp(-0.5 * (ax / sig) ** 2)
        k = np.outer(g, g)
        _KCACHE[key] = (k / k.sum()).astype(np.float32)
    return _KCACHE[key]


class Accum:
    """Additive HDR accumulator."""

    def __init__(self, W, H):
        self.W, self.H = W, H
        self.base = np.zeros((H, W, 3), np.float32)
        self.cls = {}

    def _buf(self, fam, c):
        key = (fam, c)
        if key not in self.cls:
            d = DOWN[c]
            self.cls[key] = np.zeros(((self.H + d - 1) // d + 1, (self.W + d - 1) // d + 1, 3), np.float32)
        return self.cls[key]

    def splat(self, x, y, r, rgb, soft=False):
        """x,y,r: float32 (N,), rgb: float32 (N,3) energy."""
        n = x.shape[0]
        if n == 0:
            return
        x = np.ascontiguousarray(x, np.float32)
        y = np.ascontiguousarray(y, np.float32)
        rgb = np.ascontiguousarray(rgb, np.float32)
        pos = np.clip(np.log2(np.maximum(r, 1e-6) / 0.75), 0.0, N_CLASSES - 1 - 1e-6)
        i0 = pos.astype(np.int32)
        w1 = (pos - i0).astype(np.float32)
        w0 = (1.0 - w1).astype(np.float32)
        fam = "g" if soft else "d"
        for c in np.unique(i0):
            for sel, wsel, cls in ((i0 == c, w0, c), (i0 == c, w1, c + 1)):
                idx = np.nonzero(sel)[0]
                if idx.size == 0:
                    continue
                ww = wsel[idx]
                keep = ww > 1e-4
                if not keep.all():
                    idx, ww = idx[keep], ww[keep]
                if idx.size == 0:
                    continue
                if cls == 0:
                    _splat(self.base, x[idx], y[idx], rgb[idx], ww, 1.0)
                else:
                    buf = self._buf(fam, cls)
                    _splat(buf, x[idx], y[idx], rgb[idx], ww, 1.0 / DOWN[cls])

    def add_image(self, img):
        self.base += img

    def resolve(self):
        out = self.base.copy()
        for (fam, c), buf in self.cls.items():
            d = DOWN[c]
            rad = 0.75 * 2 ** c / d
            k = disk_kernel(rad) if fam == "d" else gauss_kernel(rad)
            conv = cv2.filter2D(buf, -1, k, borderType=cv2.BORDER_CONSTANT)
            if d > 1:
                # coarse pixel i covers fine pixels [i*d, (i+1)*d) -> centre (i+0.5)*d
                Hc, Wc = conv.shape[:2]
                M = np.float32([[1.0 / d, 0, 0.5 / d - 0.5], [0, 1.0 / d, 0.5 / d - 0.5]])
                up = cv2.warpAffine(conv, M, (self.W, self.H), flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,
                                    borderMode=cv2.BORDER_CONSTANT)
                out += up / float(d * d)
            else:
                out += conv[:self.H, :self.W]
        return out


class Renderer:
    """Convenience wrapper: world-space drawing through a Camera into an Accum."""

    def __init__(self, W, H):
        self.W, self.H = W, H
        self.s = W / 1920.0
        # energies are authored at 1920 wide; a lower-res preview must look like a box-downsample
        self.e = self.s * self.s
        self.acc = Accum(W, H)

    def new_layer(self):
        img = self.acc.resolve()
        self.acc = Accum(self.W, self.H)
        return img

    def draw(self, cam, P, rgb, size=0.0, size_px=0.0, falloff=0.0, zref=1.0, soft=False,
             min_r=0.0, max_r=None, energy=1.0, margin=4.0):
        """P (N,3) world. rgb (N,3) or (3,) energy. size: world radius. size_px: px radius @1920."""
        P = np.asarray(P)
        if P.shape[0] == 0:
            return
        x, y, z, coc, valid = cam.project(P)
        r2 = coc * coc
        if np.any(size):
            rp = np.asarray(size, np.float32) * cam.fpx / z
            r2 = r2 + rp * rp
        if np.any(size_px):
            sp = np.asarray(size_px, np.float32) * self.s
            r2 = r2 + sp * sp
        r = np.sqrt(r2).astype(np.float32)
        if min_r:
            r = np.maximum(r, min_r * self.s)
        if max_r is not None:
            r = np.minimum(r, max_r * self.s)
        m = margin * self.s + r
        keep = valid & (x > -m) & (x < self.W + m) & (y > -m) & (y < self.H + m)
        if not keep.any():
            return
        idx = np.nonzero(keep)[0]
        col = np.broadcast_to(np.asarray(rgb, np.float32), (P.shape[0], 3))[idx]
        if falloff:
            col = col * ((zref / z[idx]) ** falloff)[:, None].astype(np.float32)
        col = col * np.float32(energy * self.e)
        self.acc.splat(x[idx], y[idx], r[idx], col, soft=soft)

    def draw_dirs(self, cam, D, rgb, size_px=0.0, soft=False, energy=1.0):
        """Directions at infinity (stars)."""
        D = np.asarray(D)
        if D.shape[0] == 0:
            return
        x, y, coc, valid = cam.project_dirs(D)
        r = np.sqrt(coc * coc + (np.asarray(size_px, np.float32) * self.s) ** 2).astype(np.float32)
        keep = valid & (x > -4 - r) & (x < self.W + 4 + r) & (y > -4 - r) & (y < self.H + 4 + r)
        if not keep.any():
            return
        idx = np.nonzero(keep)[0]
        col = np.broadcast_to(np.asarray(rgb, np.float32), (D.shape[0], 3))[idx] * np.float32(energy * self.e)
        self.acc.splat(x[idx], y[idx], np.broadcast_to(r, x.shape)[idx], col, soft=soft)

    def draw2d(self, x, y, rgb, r_px=0.0, soft=False, energy=1.0):
        """Screen-space points, coordinates in px @ current resolution, radius px @1920."""
        x = np.asarray(x, np.float32)
        y = np.asarray(y, np.float32)
        if x.size == 0:
            return
        r = np.broadcast_to(np.asarray(r_px, np.float32) * self.s, x.shape).astype(np.float32)
        col = np.broadcast_to(np.asarray(rgb, np.float32), (x.shape[0], 3)) * np.float32(energy * self.e)
        self.acc.splat(x, y, r, np.ascontiguousarray(col, np.float32), soft=soft)

    def resolve(self):
        return self.acc.resolve()
