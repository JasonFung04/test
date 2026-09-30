"""The message shell: the expanding sphere of light the senders become (I8, I9, II1, II2).

A thin, perfect spherical membrane of points. Its surface carries an 'information texture':

  * a crosshatch lattice: two families of loxodromes at the angles of the Blombos '#' (72 and 10 deg),
    hierarchical like graph paper -- finer lattices appear as you get closer; coarse lines are dashed
    like data;
  * latitude rings (every 10 deg, stronger every 30 deg);
  * faint, slowly drifting interference fringes (two-source hyperbolic bands).

Brightest points (lattice nodes, coarse lines, the rim) are messenger gold #FFD27A, the rest warm
white. Edge-on parts of the membrane are denser and brighter, like a soap-bubble rim: that comes for
free from projection (uniform surface points pile up at grazing angles) plus an extra `limb` boost.
The light is beamed outward (`beaming`): seen from outside, the far hemisphere is dimmer.

Points are stable on the sphere at any zoom. They come from a hierarchical cube-map sampling (each
cell keeps its parent's point and adds three), refined by distance to the camera with smooth
per-level fades, so a camera can fly up to -- or through -- the membrane without popping, and the
expanding sphere carries its dots outward smoothly.

    from engine.assets.shell import Shell
    SHELL = Shell(seed=7)
    P, rgb = SHELL.points(center, radius, cam, t=tg, energy=0.02)     # cam: engine Camera
    R.draw(cam, P, rgb, size_px=0.8)
    # or: SHELL.draw(R, cam, center, radius, t=tg, energy=0.02)

`energy` is the surface brightness of the face-on membrane (linear energy per px^2 @1920), so a
shell looks equally bright whatever its size or distance; lines, nodes and the rim are brighter.
"""
import numpy as np
from numba import njit, prange

from ..color import hex_lin

GOLD = hex_lin("#FFD27A")
WARM = hex_lin("#FFF2DC")

L_ROOT = 2          # root level of the cube-map hierarchy (6 * 4**2 cells)
L_MAX = 22          # deepest level (cell ~ 2**-21 rad): enough to fly into a light-year-sized shell
REF_W = 1920.0


# ---------------------------------------------------------------------------- numba kernels
@njit(cache=True, inline="always")
def _mix(x):
    x ^= x >> np.uint64(30)
    x *= np.uint64(0xBF58476D1CE4E5B9)
    x ^= x >> np.uint64(27)
    x *= np.uint64(0x94D049BB133111EB)
    x ^= x >> np.uint64(31)
    return x


@njit(cache=True, inline="always")
def _rnd(key, salt):
    x = _mix(key ^ (np.uint64(salt) * np.uint64(0x9E3779B97F4A7C15)))
    return np.float64(x >> np.uint64(11)) * (1.0 / 9007199254740992.0)


@njit(cache=True, inline="always")
def _ckey(level, face, i, j):
    lf = np.uint64(level) * np.uint64(8) + np.uint64(face)
    return (((lf << np.uint64(23)) | np.uint64(i)) << np.uint64(23)) | np.uint64(j)


@njit(cache=True, inline="always")
def _fdir(face, u, v, out):
    k = face // 2
    s = 1.0 - 2.0 * (face % 2)
    out[k] = s
    out[(k + 1) % 3] = u
    out[(k + 2) % 3] = v
    inv = 1.0 / np.sqrt(1.0 + u * u + v * v)
    out[0] *= inv
    out[1] *= inv
    out[2] *= inv


@njit(cache=True)
def _traverse(C, R, B, cpos, use_cam, cr, cu, cf, fpx, W, H, shx, shy, dl, salt, max_pts, max_level, l_root):
    """Depth-first quadtree walk. Emits each point once, at the level that created it."""
    out_d = np.empty((max_pts, 3))
    out_l = np.empty(max_pts, np.int16)
    out_k = np.empty(max_pts, np.uint64)
    n = 0
    over = False
    S = 8192
    s_l = np.empty(S, np.int32)
    s_f = np.empty(S, np.int32)
    s_i = np.empty(S, np.int64)
    s_j = np.empty(S, np.int64)
    s_u = np.empty(S, np.float64)
    s_v = np.empty(S, np.float64)
    s_n = np.empty(S, np.bool_)
    top = 0
    n0 = 1 << l_root
    cs0 = 2.0 / n0
    for f in range(6):
        for i in range(n0):
            for j in range(n0):
                k = _ckey(l_root, f, i, j)
                s_l[top] = l_root
                s_f[top] = f
                s_i[top] = i
                s_j[top] = j
                s_u[top] = -1.0 + (i + _rnd(k, salt + 1)) * cs0
                s_v[top] = -1.0 + (j + _rnd(k, salt + 2)) * cs0
                s_n[top] = True
                top += 1
    d = np.empty(3)
    wc = np.empty(3)
    while top > 0:
        top -= 1
        lv = s_l[top]
        f = s_f[top]
        i = s_i[top]
        j = s_j[top]
        pu = s_u[top]
        pv = s_v[top]
        isnew = s_n[top]
        cs = 2.0 / (1 << lv)
        _fdir(f, -1.0 + (i + 0.5) * cs, -1.0 + (j + 0.5) * cs, d)
        for a in range(3):
            wc[a] = C[a] + R * (d[0] * B[0, a] + d[1] * B[1, a] + d[2] * B[2, a])
        rad = R * 1.45 * cs
        dx = wc[0] - cpos[0]
        dy = wc[1] - cpos[1]
        dz = wc[2] - cpos[2]
        dist = np.sqrt(dx * dx + dy * dy + dz * dz)
        if use_cam:
            zc = dx * cf[0] + dy * cf[1] + dz * cf[2]
            if abs(zc) > rad:
                if zc <= 0.0:
                    continue
                xc = dx * cr[0] + dy * cr[1] + dz * cr[2]
                yc = dx * cu[0] + dy * cu[1] + dz * cu[2]
                x = W * 0.5 + shx + fpx * xc / zc
                y = H * 0.5 - shy - fpx * yc / zc
                m = rad * fpx / zc
                if x < -m or x > W + m or y < -m or y > H + m:
                    continue
        if isnew:
            # thin by the gnomonic area element -> uniform density on the sphere
            k = _ckey(lv, f, i, j)
            if _rnd(k, salt + 3) < (1.0 + pu * pu + pv * pv) ** -1.5:
                if n >= max_pts:
                    over = True
                    break
                _fdir(f, pu, pv, d)
                out_d[n, 0] = d[0]
                out_d[n, 1] = d[1]
                out_d[n, 2] = d[2]
                out_l[n] = lv
                out_k[n] = k
                n += 1
        if lv >= max_level:
            continue
        rmin = max(dist - rad, 1e-12)
        if rmin >= dl[lv + 1] * 1.4142:
            continue
        if top + 4 > S:
            over = True
            break
        umid = -1.0 + (i + 0.5) * cs
        vmid = -1.0 + (j + 0.5) * cs
        qa = 1 if pu > umid else 0
        qb = 1 if pv > vmid else 0
        cs2 = cs * 0.5
        for a in range(2):
            for b in range(2):
                I2 = 2 * i + a
                J2 = 2 * j + b
                s_l[top] = lv + 1
                s_f[top] = f
                s_i[top] = I2
                s_j[top] = J2
                if a == qa and b == qb:
                    s_u[top] = pu
                    s_v[top] = pv
                    s_n[top] = False
                else:
                    k2 = _ckey(lv + 1, f, I2, J2)
                    s_u[top] = -1.0 + (I2 + _rnd(k2, salt + 1)) * cs2
                    s_v[top] = -1.0 + (J2 + _rnd(k2, salt + 2)) * cs2
                    s_n[top] = True
                top += 1
    return out_d[:n], out_l[:n], out_k[:n], over


@njit(cache=True, inline="always")
def _sm(x):
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    return x * x * (3.0 - 2.0 * x)


@njit(cache=True, inline="always")
def _tex(x, y, z, k_px, t, snap_px, th, dlat, dring, src, fperiod, seed, vis0, vis1, decay):
    """Information texture at unit direction (x,y,z). Returns brightness, goldness, snap offset."""
    yy = min(1.0, max(-1.0, y))
    phi = np.arcsin(yy)
    lam = np.arctan2(z, x)
    cphi = np.cos(phi)
    sphi = np.sin(phi)
    psi = np.arcsinh(np.tan(min(1.45, max(-1.45, phi))))
    ex, ez = -np.sin(lam), np.cos(lam)
    nx, ny, nz = -sphi * np.cos(lam), cphi, -sphi * np.sin(lam)
    b = 0.20
    best = 1e30
    ddx = 0.0
    ddy = 0.0
    ddz = 0.0
    pol = _sm((1.30 - abs(phi)) / 0.25)
    hit0 = 0.0
    hit1 = 0.0
    for fam in range(2):
        ts = np.sin(th[fam])
        tc = np.cos(th[fam])
        c = -lam * ts + psi * tc
        s_al = lam * tc + psi * ts
        vx = -ts * ex + tc * nx
        vy = tc * ny
        vz = -ts * ez + tc * nz
        hit = 0.0
        for m in range(7):
            dd = dlat[fam] / (1 << m)
            sp = dd * cphi * k_px
            vis = _sm((sp - vis0) / vis1) * pol
            if vis < 0.01:
                continue
            q = c / dd
            kk = np.floor(q + 0.5)
            off = (q - kk) * dd
            dpx = abs(off) * cphi * k_px
            if dpx >= snap_px:
                continue
            amp = decay ** m * vis
            if m <= 1:
                dash = np.floor(s_al / (dd * 0.37))
                hk = _mix((np.uint64(np.int64(kk) + 1000000) * np.uint64(1000003)) ^
                          (np.uint64(np.int64(dash) + 1000000) * np.uint64(40503)) ^
                          np.uint64(fam * 77 + m * 13 + seed))
                on = np.float64(hk >> np.uint64(40)) / 16777216.0
                if on < 0.33:
                    amp *= 0.30
            if amp > hit:
                hit = amp
            if dpx < best:
                best = dpx
                s = -off * cphi
                ddx = s * vx
                ddy = s * vy
                ddz = s * vz
        if fam == 0:
            hit0 = hit
        else:
            hit1 = hit
        if hit > 0.0:
            b = max(b, 0.20 + hit)
    for m in range(7):
        dd = dring / (1 << m)
        sp = dd * k_px
        vis = _sm((sp - vis0) / vis1)
        if vis < 0.01:
            continue
        q = phi / dd
        kk = np.floor(q + 0.5)
        off = (q - kk) * dd
        dpx = abs(off) * k_px
        if dpx >= snap_px:
            continue
        amp = 0.85 * decay ** m * vis
        if m == 0 and (np.int64(kk) % 3) == 0:
            amp *= 1.4
        b = max(b, 0.20 + amp)
        if dpx < best:
            best = dpx
            ddx = -off * nx
            ddy = -off * ny
            ddz = -off * nz
    node = min(hit0, hit1)
    b += 1.8 * node
    g = min(1.0, node * 1.8 + min(1.0, max(0.0, b - 0.7)) * 0.9)
    a1 = np.sqrt((x - src[0, 0]) ** 2 + (y - src[0, 1]) ** 2 + (z - src[0, 2]) ** 2)
    a2 = np.sqrt((x - src[1, 0]) ** 2 + (y - src[1, 1]) ** 2 + (z - src[1, 2]) ** 2)
    fvis = _sm((fperiod * k_px - 8.0) / 10.0)
    b *= 1.0 + 0.30 * fvis * np.cos(2.0 * np.pi * (a1 - a2) / fperiod + 0.9 * t)
    return b, g, ddx, ddy, ddz


@njit(cache=True, parallel=True)
def _finish(n, lv, kk, C, R, B, cpos, fpx_ref, dl, lmax, t, energy, thickness, limb, beaming, texture,
            snap_px, twinkle, salt, th, dlat, dring, src, fperiod, seed, warm, gold, P_out, E_out):
    N = n.shape[0]
    for p in prange(N):
        x = n[p, 0]
        y = n[p, 1]
        z = n[p, 2]
        wx = C[0] + R * (x * B[0, 0] + y * B[1, 0] + z * B[2, 0])
        wy = C[1] + R * (x * B[0, 1] + y * B[1, 1] + z * B[2, 1])
        wz = C[2] + R * (x * B[0, 2] + y * B[1, 2] + z * B[2, 2])
        r = np.sqrt((wx - cpos[0]) ** 2 + (wy - cpos[1]) ** 2 + (wz - cpos[2]) ** 2) + 1e-12
        # level-of-detail: effective density (per sr of the unit sphere) and this dot's fade weight
        sig = 4.0 ** (L_ROOT - 1)
        w = 1.0
        for L in range(L_ROOT + 1, lmax + 1):
            wl = _sm(np.log2(dl[L] / r) + 0.5)
            sig += 3.0 * 4.0 ** (L - 2) * wl
            if lv[p] == L:
                w = wl
        k_px = R * fpx_ref / r
        b, g, ddx, ddy, ddz = _tex(x, y, z, k_px, t, snap_px, th, dlat, dring, src, fperiod, seed, 10.0, 12.0, 0.5)
        if texture < 1.0:
            b = 0.35 + texture * (b - 0.35)
            g *= texture
        mx = x + ddx
        my = y + ddy
        mz = z + ddz
        inv = 1.0 / np.sqrt(mx * mx + my * my + mz * mz)
        mx *= inv
        my *= inv
        mz *= inv
        nwx = mx * B[0, 0] + my * B[1, 0] + mz * B[2, 0]
        nwy = mx * B[0, 1] + my * B[1, 1] + mz * B[2, 1]
        nwz = mx * B[0, 2] + my * B[1, 2] + mz * B[2, 2]
        key = kk[p]
        depth = -np.log(1.0 - 0.999 * _rnd(key, salt + 5)) * 0.5
        rr = R * (1.0 - thickness * depth)
        px = C[0] + rr * nwx
        py = C[1] + rr * nwy
        pz = C[2] + rr * nwz
        vx = (cpos[0] - px) / r
        vy = (cpos[1] - py) / r
        vz = (cpos[2] - pz) / r
        facing = nwx * vx + nwy * vy + nwz * vz
        mu = min(1.0, abs(facing))
        rim = 1.0 + limb * (1.0 - mu) ** 3
        beam = (1.0 - beaming) + beaming * _sm((facing + 0.25) / 0.5)
        g = min(1.0, g + 0.55 * (1.0 - mu) ** 8)
        tw = 1.0 + twinkle * np.sin(2.0 * np.pi * (_rnd(key, salt + 6) + t * (0.25 + 0.5 * _rnd(key, salt + 7))))
        e = energy * (R * R / sig) * (fpx_ref / r) ** 2 * w * b * np.exp(-0.9 * depth) * rim * beam * tw
        P_out[p, 0] = px
        P_out[p, 1] = py
        P_out[p, 2] = pz
        for c in range(3):
            E_out[p, c] = (warm[c] * (1.0 - g) + gold[c] * g) * e


def _u01(key, salt):
    x = key ^ np.uint64((salt * 0x9E3779B97F4A7C15) & 0xFFFFFFFFFFFFFFFF)
    x = x ^ (x >> np.uint64(30))
    x = x * np.uint64(0xBF58476D1CE4E5B9)
    x = x ^ (x >> np.uint64(27))
    x = x * np.uint64(0x94D049BB133111EB)
    x = x ^ (x >> np.uint64(31))
    return (x >> np.uint64(11)).astype(np.float64) * (1.0 / 9007199254740992.0)


def _smooth01(x):
    x = np.clip(x, 0.0, 1.0)
    return x * x * (3 - 2 * x)


# ---------------------------------------------------------------------------- the asset
class Shell:
    """One instance can serve every shot; all state is the (seeded) texture design."""

    def __init__(self, seed=7, axis=(0.0, 1.0, 0.0), ref=(1.0, 0.0, 0.0),
                 lattice_deg=(72.0, 10.0), lattice_lines=(24, 4), ring_deg=10.0, fringe_period=0.07):
        self.seed = int(seed)
        a = np.asarray(axis, float)
        a /= np.linalg.norm(a)
        r = np.asarray(ref, float)
        r = r - a * (r @ a)
        if np.linalg.norm(r) < 1e-6:
            r = np.cross(a, [0.0, 0.0, 1.0])
        r /= np.linalg.norm(r)
        # rows: local x (ref), local y (pole axis), local z; world = local @ basis
        self.basis = np.ascontiguousarray(np.stack([r, a, np.cross(r, a)], axis=0))
        th = np.radians(np.asarray(lattice_deg, float))
        self.th = th
        # spacing chosen so each lattice family is periodic in longitude (no seam)
        self.dlat = np.array([2 * np.pi * abs(np.sin(t)) / m for t, m in zip(th, lattice_lines)])
        self.dring = np.radians(ring_deg)
        rng = np.random.default_rng(self.seed)
        s = rng.normal(size=(2, 3))
        s[:, 1] *= 0.3                                            # fringe sources near the equator
        self.src = np.ascontiguousarray(s / np.linalg.norm(s, axis=1, keepdims=True))
        self.fringe_period = fringe_period
        self.salt = self.seed * 7919 + 17

    @staticmethod
    def _lod(R, fpx, spacing_px):
        """dl[l]: distance below which level l fades in (level l-1 spacing exceeds spacing_px)."""
        lv = np.arange(L_MAX + 2)
        return R * 2.0 ** (2 - lv) * fpx / spacing_px

    def _sample(self, C, R, cam, cam_pos, fpx_ref, spacing_px, max_points):
        for _ in range(6):
            dl = self._lod(R, fpx_ref, spacing_px)
            if cam is not None:
                args = (True, cam.right, cam.up, cam.fwd, float(cam.fpx), float(cam.W), float(cam.H),
                        float(cam.shift[0]), float(cam.shift[1]))
            else:
                z3 = np.zeros(3)
                args = (False, z3, z3, z3, 1.0, 1.0, 1.0, 0.0, 0.0)
            d, lv, k, over = _traverse(np.asarray(C, float), float(R), self.basis, np.asarray(cam_pos, float),
                                       *args, dl, self.salt, int(max_points), L_MAX, L_ROOT)
            if not over:
                return d, lv, k, dl, spacing_px
            spacing_px *= 1.5            # over budget: sparser dots (never a partial, uneven sample)
        return d, lv, k, dl, spacing_px

    def points(self, center, radius, cam, t=0.0, energy=0.02, thickness=0.0012, limb=2.0, beaming=0.6,
               texture=1.0, spacing_px=2.6, max_points=1_200_000, twinkle=0.12):
        """Points of the membrane.

        center, radius : the world sphere.
        cam        : engine Camera (preferred: frustum culling and px-accurate level of detail),
                     or just a camera position (then a 35 mm lens @1920 is assumed, no culling).
        t          : seconds; drives the fringe drift and a gentle per-dot twinkle.
        energy     : face-on surface brightness (linear energy per px^2 @1920).
        thickness  : membrane thickness as a fraction of the radius; dots trail *inside* the front
                     with an exponential profile (sharp leading edge, soft wake).
        limb       : extra soap-bubble rim brightening at grazing angles (0 = projection only).
        beaming    : 0..1, how much of the light goes outward (1 = far hemisphere invisible).
        texture    : 0 = plain membrane .. 1 = full information texture.
        spacing_px : on-screen dot spacing (@1920) the level of detail aims for (bigger = sparser).
        Returns P (N,3) float32, rgb (N,3) float32 linear energy (draw with size_px ~0.6-1.0).
        """
        C = np.asarray(center, float)
        R = float(radius)
        empty = (np.zeros((0, 3), np.float32), np.zeros((0, 3), np.float32))
        if R <= 0:
            return empty
        if hasattr(cam, "project"):
            cam_pos, fpx_ref, cam_obj = cam.pos, cam.focal / 36.0 * REF_W, cam
        else:
            cam_pos, fpx_ref, cam_obj = np.asarray(cam, float), 35.0 / 36.0 * REF_W, None
        n, lv, kk, dl, spacing_px = self._sample(C, R, cam_obj, cam_pos, fpx_ref, spacing_px, max_points)
        if n.shape[0] == 0:
            return empty
        P = np.empty((n.shape[0], 3), np.float32)
        E = np.empty((n.shape[0], 3), np.float32)
        _finish(np.ascontiguousarray(n), lv, kk, C, R, self.basis, np.asarray(cam_pos, float), fpx_ref, dl,
                int(lv.max()), float(t), float(energy), float(thickness), float(limb), float(beaming),
                float(texture), 0.55 * spacing_px, float(twinkle), self.salt, self.th, self.dlat, self.dring,
                self.src, self.fringe_period, self.seed, WARM.astype(np.float64), GOLD.astype(np.float64), P, E)
        return P, E

    def draw(self, R, cam, center, radius, t=0.0, energy=0.02, size_px=0.8, **kw):
        """Convenience: points() straight into a Renderer. Returns the number of dots."""
        P, rgb = self.points(center, radius, cam, t=t, energy=energy, **kw)
        R.draw(cam, P, rgb, size_px=size_px)
        return P.shape[0]
