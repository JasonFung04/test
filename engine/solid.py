"""Opaque point-cloud objects (figures, props): visibility, lighting, constant on-screen dot size.

A SolidCloud is a set of surface samples with normals, per-point albedo/emission and the surface
area each sample represents.  `draw_solid` shades them, removes hidden points with a depth
splat, keeps the pointillist grain size roughly constant on screen, and returns a coverage mask
so the caller can occlude whatever is behind the object.
"""
import cv2
import numpy as np
from numba import njit


def _h(key, seed):
    """Cheap deterministic per-point hash in [0,1) from the point key."""
    x = np.sin(key.astype(np.float64) * (12.9898 + seed * 0.61) + seed * 78.233) * 43758.5453
    return (x - np.floor(x)).astype(np.float32)


def tube(curve, radii, n, seed=0, bumps=None):
    """Sample a smooth tube around a polyline curve (M,3) with per-vertex radii (M,).

    Parallel-transport frames, area-uniform sampling. Returns P, N, area_per_point."""
    C = np.asarray(curve, np.float64)
    r = np.asarray(radii, np.float64)
    seg = np.linalg.norm(np.diff(C, axis=0), axis=1)
    s_ = np.concatenate([[0], np.cumsum(seg)])
    L = s_[-1]
    circ = 2 * np.pi * r
    w = np.concatenate([[0], np.cumsum(0.5 * (circ[1:] + circ[:-1]) * seg)])
    area = w[-1]
    rng = np.random.default_rng(seed)
    q = rng.random(n) * area
    si = np.interp(q, w, s_)
    T = np.gradient(C, axis=0)
    T /= np.linalg.norm(T, axis=1, keepdims=True) + 1e-12
    ref = np.array([1.0, 0, 0]) if abs(T[0][0]) < 0.9 else np.array([0, 1.0, 0])
    Ns = [np.cross(T[0], ref)]
    Ns[0] /= np.linalg.norm(Ns[0])
    for i in range(1, len(C)):
        v = Ns[-1] - T[i] * np.dot(Ns[-1], T[i])
        Ns.append(v / (np.linalg.norm(v) + 1e-12))
    N1 = np.array(Ns)
    B1 = np.cross(T, N1)
    ci = np.interp(si, s_, np.arange(len(C)))
    i0 = np.clip(ci.astype(int), 0, len(C) - 2)
    f = (ci - i0)[:, None]
    pc = C[i0] * (1 - f) + C[i0 + 1] * f
    rc = r[i0] * (1 - f[:, 0]) + r[i0 + 1] * f[:, 0]
    nn = N1[i0] * (1 - f) + N1[i0 + 1] * f
    bb = B1[i0] * (1 - f) + B1[i0 + 1] * f
    th = rng.random(n) * 2 * np.pi
    nrm = nn * np.cos(th)[:, None] + bb * np.sin(th)[:, None]
    nrm /= np.linalg.norm(nrm, axis=1, keepdims=True) + 1e-12
    if bumps is not None:
        rc = rc * (1 + bumps(si / max(L, 1e-9), th))
    P = pc + nrm * rc[:, None]
    return P.astype(np.float32), nrm.astype(np.float32), float(area / n)


def bezier(p0, p1, p2, m=24):
    t = np.linspace(0, 1, m)[:, None]
    return (1 - t) ** 2 * np.asarray(p0) + 2 * (1 - t) * t * np.asarray(p1) + t ** 2 * np.asarray(p2)


class SolidCloud:
    def __init__(self, P, N, albedo, emit=None, area=None, key=None, part=None):
        self.P = np.ascontiguousarray(P, np.float32)
        self.N = np.ascontiguousarray(N, np.float32)
        n = self.P.shape[0]
        self.albedo = np.broadcast_to(np.asarray(albedo, np.float32), (n, 3)).copy()
        self.emit = np.zeros((n, 3), np.float32) if emit is None else np.broadcast_to(
            np.asarray(emit, np.float32), (n, 3)).copy()
        self.area = np.full(n, 1e-6, np.float32) if area is None else np.broadcast_to(
            np.asarray(area, np.float32), (n,)).copy()
        self.key = (np.random.default_rng(n).random(n).astype(np.float32) if key is None
                    else np.asarray(key, np.float32))
        self.part = np.zeros(n, np.int16) if part is None else np.asarray(part, np.int16)

    def __len__(self):
        return self.P.shape[0]

    def subset(self, m):
        return SolidCloud(self.P[m], self.N[m], self.albedo[m], self.emit[m], self.area[m], self.key[m], self.part[m])

    @staticmethod
    def concat(clouds):
        return SolidCloud(np.concatenate([c.P for c in clouds]), np.concatenate([c.N for c in clouds]),
                          np.concatenate([c.albedo for c in clouds]), np.concatenate([c.emit for c in clouds]),
                          np.concatenate([c.area for c in clouds]), np.concatenate([c.key for c in clouds]),
                          np.concatenate([c.part for c in clouds]))

    def transformed(self, Rm, t):
        Rm = np.asarray(Rm, np.float32)
        return SolidCloud(self.P @ Rm.T + np.asarray(t, np.float32), self.N @ Rm.T, self.albedo, self.emit,
                          self.area, self.key, self.part)


@njit(cache=True)
def _zsplat(depth, x, y, z, rad):
    H, W = depth.shape
    for i in range(x.shape[0]):
        xi = int(x[i])
        yi = int(y[i])
        r = rad[i]
        for dy in range(-r, r + 1):
            yy = yi + dy
            if yy < 0 or yy >= H:
                continue
            for dx in range(-r, r + 1):
                xx = xi + dx
                if xx < 0 or xx >= W or dx * dx + dy * dy > r * r + r:
                    continue
                if z[i] < depth[yy, xx]:
                    depth[yy, xx] = z[i]


@njit(cache=True)
def _vis(depth, x, y, z, tol):
    H, W = depth.shape
    out = np.zeros(x.shape[0], np.bool_)
    for i in range(x.shape[0]):
        xi = int(x[i])
        yi = int(y[i])
        if 0 <= xi < W and 0 <= yi < H:
            out[i] = z[i] <= depth[yi, xi] * (1.0 + tol) + 1e-6
    return out


def visibility(x, y, z, W, H, rad_px, tol=0.004, down=1):
    """Depth-splat visibility. rad_px: per-point int radius (px at full res)."""
    d = down
    depth = np.full(((H + d - 1) // d, (W + d - 1) // d), np.inf, np.float32)
    rr = np.maximum((rad_px / d).astype(np.int32), 1)
    _zsplat(depth, (x / d).astype(np.float32), (y / d).astype(np.float32), z.astype(np.float32), rr)
    return _vis(depth, (x / d).astype(np.float32), (y / d).astype(np.float32), z.astype(np.float32), tol)


def coverage_mask(x, y, W, H, rad_px, close=2):
    """Solid silhouette mask from projected points (dilated splats + morphological close)."""
    m = np.zeros((H, W), np.uint8)
    xi = np.clip(x.astype(np.int32), 0, W - 1)
    yi = np.clip(y.astype(np.int32), 0, H - 1)
    inside = (x >= 0) & (x < W) & (y >= 0) & (y < H)
    m[yi[inside], xi[inside]] = 255
    r = int(max(1, np.ceil(rad_px)))
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))
    m = cv2.dilate(m, k)
    if close:
        k2 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * close + 1, 2 * close + 1))
        m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, k2)
    m = cv2.GaussianBlur(m.astype(np.float32) / 255.0, (0, 0), 0.7)
    return m


class Light:
    """kind: 'dir' (direction TO the light), 'point' (position), 'amb' (ambient)."""

    def __init__(self, kind, color, intensity=1.0, vec=None, radius=None, wrap=0.0):
        self.kind, self.color = kind, np.asarray(color, np.float32)
        self.intensity = float(intensity)
        self.vec = None if vec is None else np.asarray(vec, np.float64)
        self.radius = radius
        self.wrap = wrap


def shade(cloud, cam_pos, lights, rim=None, spec=None):
    """Per-point outgoing radiance (linear). rim=(color, power, strength) adds a fresnel edge light."""
    P, N = cloud.P, cloud.N
    V = cam_pos[None, :] - P
    V /= np.linalg.norm(V, axis=1, keepdims=True) + 1e-9
    L_out = np.zeros_like(P)
    for lt in lights:
        if lt.kind == "amb":
            L_out += cloud.albedo * lt.color * lt.intensity
            continue
        if lt.kind == "dir":
            Ld = np.broadcast_to(lt.vec / np.linalg.norm(lt.vec), P.shape)
            att = 1.0
        else:
            d = lt.vec[None, :] - P
            dist = np.linalg.norm(d, axis=1, keepdims=True) + 1e-6
            Ld = d / dist
            att = 1.0 / (dist[:, 0] ** 2 + (lt.radius or 0.05) ** 2)
        ndl = np.sum(N * Ld, axis=1)
        if lt.wrap:
            ndl = (ndl + lt.wrap) / (1 + lt.wrap)
        ndl = np.clip(ndl, 0, None) * att
        L_out += cloud.albedo * (lt.color * lt.intensity)[None, :] * ndl[:, None]
        if spec is not None:
            Hh = Ld + V
            Hh /= np.linalg.norm(Hh, axis=1, keepdims=True) + 1e-9
            s = np.clip(np.sum(N * Hh, axis=1), 0, 1) ** spec[1] * spec[0] * att
            L_out += (lt.color * lt.intensity)[None, :] * s[:, None] * (np.clip(ndl, 0, 1) > 0)[:, None]
    if rim is not None:
        col, pw, st = rim
        f = (1.0 - np.clip(np.abs(np.sum(N * V, axis=1)), 0, 1)) ** pw
        L_out += np.asarray(col, np.float32)[None, :] * (st * f)[:, None]
    return (L_out + cloud.emit).astype(np.float32)


def draw_solid(R, cam, cloud, lights, rim=None, spec=None, spacing_px=2.2, backface=0.12, occlude=True,
               energy=1.0, size_px=None, translucent=False, return_mask=False, mask_close=2, dot_boost=1.0,
               seurat=0.75, l_ref=None, p_min=0.04, jitter=0.10, size_var=0.35, flecks=None, seed=0):
    """Shade + visibility + constant screen dot spacing. Returns coverage mask (H,W) if asked.

    seurat: exponent of the 'density follows light' rule (0 = off): dark regions are drawn with
    fewer, not dimmer, dots; highlights are dense. jitter: per-dot colour variation. flecks:
    optional (rgb, fraction) complementary-colour dots sprinkled into shadows (optical mixing)."""
    W, H = R.W, R.H
    s = R.s
    x, y, z, coc, valid = cam.project(cloud.P)
    V = cam.pos[None, :] - cloud.P
    V /= np.linalg.norm(V, axis=1, keepdims=True) + 1e-9
    ndv = np.sum(cloud.N * V, axis=1)
    keep = valid & (x > -20) & (x < W + 20) & (y > -20) & (y < H + 20)
    if not translucent:
        keep &= ndv > -backface
    # native projected spacing of the samples, then thin to the target on-screen dot spacing
    native = np.sqrt(cloud.area) * cam.fpx / np.maximum(z, 1e-6)
    target = spacing_px * s
    frac = np.clip((native / target) ** 2, 0.0, 1.0)
    keep &= cloud.key < frac
    idx = np.nonzero(keep)[0]
    mask = None
    if idx.size == 0:
        return np.zeros((H, W), np.float32) if return_mask else None
    xs, ys, zs = x[idx], y[idx], z[idx]
    spacing_now = np.maximum(native[idx], target) if True else native[idx]
    if occlude and not translucent:
        rad = np.clip(spacing_now * 0.9, 1, 12).astype(np.int32)
        vis = visibility(xs, ys, zs, W, H, rad)
        idx, xs, ys, zs = idx[vis], xs[vis], ys[vis], zs[vis]
    sub = cloud.subset(idx)
    rad_out = shade(sub, cam.pos, lights, rim, spec)
    # energy = radiance * projected area of the patch this dot stands for
    area_px = np.maximum(native[idx], target) ** 2 * np.clip(np.abs(ndv[idx]), 0.25, 1.0) ** 0.5
    e = rad_out * (area_px / (s * s))[:, None] * dot_boost
    u1 = _h(sub.key, 101 + seed)
    if seurat:
        lum = rad_out @ np.array([0.2126, 0.7152, 0.0722], np.float32)
        ref = l_ref if l_ref is not None else max(float(np.percentile(lum, 92)), 1e-6)
        p = np.clip((lum / ref) ** seurat, p_min, 1.0)
        k2 = u1 < p
        e = e[k2] / p[k2, None]
        xs, ys, idx, sub_key = xs[k2], ys[k2], idx[k2], sub.key[k2]
        rad_out = rad_out[k2]
        u1 = u1[k2]
    else:
        sub_key = sub.key
    if jitter:
        u2 = _h(sub_key, 202 + seed)
        u3 = _h(sub_key, 303 + seed)
        g = 1.0 + jitter * np.stack([u2 - 0.5, (u3 - 0.5) * 0.6, (u2 * u3 - 0.25) * 1.2], 1) * 2
        e = e * g.astype(np.float32)
    if flecks is not None:
        fc, frac_f = flecks
        u4 = _h(sub_key, 404 + seed)
        lum = e @ np.array([0.2126, 0.7152, 0.0722], np.float32)
        fl = u4 < frac_f
        e[fl] = np.asarray(fc, np.float32)[None, :] * lum[fl, None] * 1.4
    r_px = (0.42 * target / s) if size_px is None else size_px
    sv = 1.0 + size_var * (_h(sub_key, 505 + seed) - 0.5) * 2 if size_var else 1.0
    rr = np.sqrt((r_px * s * sv) ** 2 + coc[idx] ** 2).astype(np.float32)
    R.acc.splat(xs, ys, rr, (e * np.float32(energy * R.e)).astype(np.float32))
    if return_mask:
        mask = coverage_mask(xs, ys, W, H, np.median(np.maximum(native[idx], target)) * 0.8 + 0.5, mask_close)
        return mask
    return None
