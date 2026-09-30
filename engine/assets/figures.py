"""Characters as sculpted point clouds: THE GIRL and THE ELDER (+ the elder's crowd).

Sculpted with smooth-union SDFs, sampled once (cached), posed by rigid parts.
Figure space: origin at the base (girl: seat on the platform; elder: between the feet),
+y up, +z = the direction the figure faces, +x = the figure's left.
"""
import numpy as np

from ..config import CACHE
from ..noise import fbm, hash01
from ..sdf import (rot, sample_surface, sd_capsule, sd_ellipsoid, sd_round_box, sd_round_cone,
                   sd_sphere, smax, smin)
from ..solid import SolidCloud


def _cache(name, fn):
    path = CACHE / f"fig_{name}.npz"
    if path.exists():
        d = np.load(path)
        return {k: d[k] for k in d.files}
    d = fn()
    np.savez(path, **d)
    return d


def _lin(rgb):
    return np.asarray(rgb, np.float32)


# =============================================================================== THE GIRL
SKIN = _lin([0.54, 0.35, 0.26])
LIP = _lin([0.50, 0.22, 0.19])
HAIR = _lin([0.018, 0.015, 0.013])
EYE = _lin([0.012, 0.009, 0.008])
HOODIE = _lin([0.80, 0.56, 0.09])
PATCH = _lin([0.62, 0.45, 0.16])
PANTS = _lin([0.05, 0.055, 0.075])
SHOE = _lin([0.62, 0.62, 0.64])
TORCH = _lin([0.55, 0.57, 0.60])

NECK_PIVOT = np.array([0.0, 0.436, 0.0])          # figure space
HEAD_PIVOT_LOCAL = np.array([0.0, -0.105, -0.022])  # neck pivot expressed in the head frame
EYE_L = np.array([0.027, 0.004, 0.066])            # head frame (x+ = her left)
EYE_R = np.array([-0.027, 0.004, 0.066])


def girl_head_sdf(p):
    d = sd_ellipsoid(p, (0, 0.018, -0.012), (0.068, 0.083, 0.088))
    d = smin(d, sd_ellipsoid(p, (0, -0.040, 0.030), (0.056, 0.058, 0.057)), 0.03)
    d = smin(d, sd_ellipsoid(p, (0, 0.040, 0.040), (0.056, 0.042, 0.042)), 0.02)
    for sx in (1, -1):
        d = smin(d, sd_ellipsoid(p, (0.034 * sx, -0.030, 0.052), (0.028, 0.027, 0.026)), 0.015)
        d = smin(d, sd_ellipsoid(p, (0.068 * sx, -0.004, -0.006), (0.009, 0.025, 0.016)), 0.004)   # ears
        d = smax(d, -sd_ellipsoid(p, (0.027 * sx, 0.004, 0.0745), (0.0145, 0.0072, 0.0058)), 0.005)  # sockets
        d = smin(d, sd_sphere(p, (0.0095 * sx, -0.031, 0.083), 0.0072), 0.005)                   # alae
    d = smin(d, sd_round_cone(p, (0, -0.006, 0.074), (0, -0.027, 0.0885), 0.0050, 0.0080), 0.007)
    d = smin(d, sd_sphere(p, (0, -0.0262, 0.0880), 0.0090), 0.004)
    d = smin(d, sd_ellipsoid(p, (0, -0.047, 0.0815), (0.019, 0.0062, 0.0095)), 0.004)
    d = smin(d, sd_ellipsoid(p, (0, -0.0565, 0.0790), (0.0165, 0.0065, 0.0095)), 0.004)
    d = smax(d, -sd_ellipsoid(p, (0, -0.0515, 0.0880), (0.016, 0.0014, 0.009)), 0.002)
    d = smin(d, sd_ellipsoid(p, (0, -0.075, 0.062), (0.021, 0.017, 0.018)), 0.012)
    d = smin(d, sd_capsule(p, (0, -0.060, -0.018), (0, -0.140, -0.026), 0.034), 0.02)
    return d


def girl_eyes_sdf(p):
    return np.minimum(sd_sphere(p, EYE_L - (0, 0, 0.0055), 0.0118), sd_sphere(p, EYE_R - (0, 0, 0.0055), 0.0118))


def girl_hair_sdf(p):
    d = sd_ellipsoid(p, (0, 0.028, -0.020), (0.087, 0.093, 0.098))
    face = sd_round_box(p, (0, -0.050, 0.082), (0.058, 0.078, 0.058), 0.02)
    d = smax(d, -face, 0.012)
    d = smax(d, -(p[:, 1] + 0.052), 0.01)                          # bob length
    tuft = sd_ellipsoid(p, (0, 0.072, -0.096), (0.016, 0.022, 0.019))
    d = smin(d, tuft, 0.012)
    d = d + 0.0015 * fbm(p * 180.0, octaves=2)
    return d


def girl_body_sdf(p):
    d = sd_ellipsoid(p, (0, 0.075, -0.010), (0.125, 0.075, 0.100))                       # hips
    torso = sd_round_cone(p, (0, 0.13, -0.005), (0, 0.300, -0.005), 0.150, 0.118)
    torso = smin(torso, sd_ellipsoid(p, (0, 0.225, 0.0), (0.158, 0.150, 0.108)), 0.04)
    torso = smin(torso, sd_ellipsoid(p, (0, 0.330, -0.008), (0.168, 0.058, 0.094)), 0.05)
    torso = smin(torso, sd_ellipsoid(p, (0, 0.388, -0.058), (0.100, 0.042, 0.066)), 0.035)  # hood collar
    torso = torso + 0.0035 * fbm(p * np.array([22.0, 38.0, 22.0]), octaves=3)             # fabric folds
    d = smin(d, torso, 0.04)
    d = smin(d, sd_capsule(p, (0, 0.37, 0.0), (0, 0.445, 0.006), 0.032), 0.012)           # neck
    # crossed legs
    for sx in (1, -1):
        d = smin(d, sd_round_cone(p, (0.075 * sx, 0.075, 0.03), (0.240 * sx, 0.075, 0.215), 0.062, 0.046), 0.03)
    d = smin(d, sd_round_cone(p, (0.240, 0.068, 0.215), (-0.165, 0.046, 0.190), 0.044, 0.032), 0.02)
    d = smin(d, sd_round_cone(p, (-0.240, 0.070, 0.215), (0.170, 0.050, 0.270), 0.044, 0.032), 0.02)
    return d


def girl_feet_sdf(p):
    a = sd_ellipsoid(p, (-0.205, 0.040, 0.170), (0.040, 0.036, 0.085), rot("y", -0.9))
    b = sd_ellipsoid(p, (0.210, 0.043, 0.262), (0.040, 0.036, 0.085), rot("y", 0.9))
    return np.minimum(a, b)


def _segment_cloud(L, r1, r2, n, seed, albedo, cuff=False):
    """Round cone from (0,0,0) to (0,-L,0) (hanging) in a local bone frame."""
    def f(p):
        d = sd_round_cone(p, (0, 0, 0), (0, -L, 0), r1, r2)
        if cuff:
            for k in range(3):
                yy = -L + 0.012 + k * 0.011
                q = p - np.array([0, yy, 0])
                ring = np.sqrt((np.sqrt(q[:, 0] ** 2 + q[:, 2] ** 2) - (r2 + 0.004)) ** 2 + q[:, 1] ** 2) - 0.0062
                d = smin(d, ring, 0.003)
        return d + 0.002 * fbm(p * 40.0, octaves=2)
    R = max(r1, r2) + 0.02
    P, N, a = sample_surface(f, (-R, -L - R, -R), (R, R, R), n, seed=seed)
    return dict(P=P, N=N, area=np.float32(a), alb=np.broadcast_to(albedo, P.shape).astype(np.float32))


def _hand_cloud(n, seed, length=0.075, width=0.028):
    def f(p):
        d = sd_ellipsoid(p, (0, -0.030, 0.004), (width, 0.036, 0.012))
        for k, dx in enumerate((-0.016, -0.005, 0.006, 0.016)):
            fl = length * (0.78, 0.9, 0.95, 0.86)[k]
            d = smin(d, sd_round_cone(p, (dx * 0.9, -0.058, 0.002), (dx * 1.05, -0.058 - fl * 0.55, 0.020),
                                      0.0065, 0.0055), 0.006)
        d = smin(d, sd_round_cone(p, (width * 0.85, -0.020, 0.010), (width * 1.05, -0.050, 0.030), 0.0085,
                                  0.0065), 0.006)
        return d
    P, N, a = sample_surface(f, (-0.05, -0.12, -0.03), (0.05, 0.01, 0.05), n, seed=seed)
    return dict(P=P, N=N, area=np.float32(a), alb=np.broadcast_to(SKIN, P.shape).astype(np.float32))


def _torch_cloud(n, seed):
    def f(p):
        body = sd_capsule(p, (0, 0, 0), (0, 0.118, 0), 0.0145)
        head = sd_capsule(p, (0, 0.105, 0), (0, 0.130, 0), 0.0185)
        return smin(body, head, 0.004)
    P, N, a = sample_surface(f, (-0.03, -0.02, -0.03), (0.03, 0.16, 0.03), n, seed=seed)
    return dict(P=P, N=N, area=np.float32(a), alb=np.broadcast_to(TORCH, P.shape).astype(np.float32))


def _star_patch(P, N):
    """Faded star on the hoodie chest: points whose front projection falls inside a 5-point star."""
    c = np.array([0.045, 0.285])   # (x, y) on the chest, her left side
    q = np.stack([P[:, 0] - c[0], P[:, 1] - c[1]], 1)
    r = np.linalg.norm(q, axis=1)
    th = np.arctan2(q[:, 1], q[:, 0]) - np.pi / 2
    k = (np.cos(5 * th) * 0.5 + 0.5)
    rad = 0.012 + 0.014 * k ** 2.2
    return (r < rad) & (N[:, 2] > 0.35) & (P[:, 2] > 0.05)


class Girl:
    PARTS = {"skin": 0, "hair": 1, "eye": 2, "hoodie": 3, "pants": 4, "shoe": 5, "torch": 6, "lip": 7}

    def __init__(self, density=1.0):
        def make():
            out = {}
            P, N, a = sample_surface(girl_head_sdf, (-0.09, -0.15, -0.11), (0.09, 0.11, 0.115),
                                     int(170_000 * density), seed=1)
            lip = (P[:, 2] > 0.070) & (P[:, 1] < -0.043) & (P[:, 1] > -0.064) & (np.abs(P[:, 0]) < 0.02)
            out["head_P"], out["head_N"], out["head_a"], out["head_lip"] = P, N, np.float32(a), lip
            P, N, a = sample_surface(girl_eyes_sdf, (-0.045, -0.012, 0.045), (0.045, 0.02, 0.08),
                                     int(9_000 * density), seed=2)
            out["eye_P"], out["eye_N"], out["eye_a"] = P, N, np.float32(a)
            P, N, a = sample_surface(girl_hair_sdf, (-0.10, -0.07, -0.13), (0.10, 0.13, 0.10),
                                     int(150_000 * density), seed=3)
            out["hair_P"], out["hair_N"], out["hair_a"] = P, N, np.float32(a)
            P, N, a = sample_surface(girl_body_sdf, (-0.33, -0.01, -0.16), (0.33, 0.50, 0.36),
                                     int(260_000 * density), seed=4)
            out["body_P"], out["body_N"], out["body_a"] = P, N, np.float32(a)
            P, N, a = sample_surface(girl_feet_sdf, (-0.30, -0.01, 0.05), (0.30, 0.10, 0.37),
                                     int(20_000 * density), seed=5)
            out["feet_P"], out["feet_N"], out["feet_a"] = P, N, np.float32(a)
            for nm, L, r1, r2, cuff, sd in (("upper", 0.215, 0.050, 0.043, False, 6),
                                             ("fore", 0.195, 0.043, 0.040, True, 7)):
                d = _segment_cloud(L, r1, r2, int(26_000 * density), sd, HOODIE, cuff)
                out[f"{nm}_P"], out[f"{nm}_N"], out[f"{nm}_a"] = d["P"], d["N"], d["area"]
            d = _hand_cloud(int(12_000 * density), 8)
            out["hand_P"], out["hand_N"], out["hand_a"] = d["P"], d["N"], d["area"]
            d = _torch_cloud(int(9_000 * density), 9)
            out["torch_P"], out["torch_N"], out["torch_a"] = d["P"], d["N"], d["area"]
            return out
        self.d = _cache(f"girl_{density}", make)
        d = self.d
        # hair: keep only the outer shell (drop the faces created by the face/bob cuts), then
        # turn every sample into a tiny 3-dot dash along the strand direction (crown -> tips)
        hp, hn = d["hair_P"], d["hair_N"]
        face = sd_round_box(hp.astype(np.float64), (0, -0.050, 0.082), (0.058, 0.078, 0.058), 0.02)
        outer = (np.abs(face) > 0.0035) & ((hp[:, 1] + 0.052) > 0.0035)
        outer &= sd_ellipsoid(hp.astype(np.float64), (0, 0.018, -0.012), (0.068, 0.083, 0.088)) > 0.002
        hp, hn = hp[outer], hn[outer]
        crown = np.array([0.0, 0.10, -0.03])
        radial = hp - crown
        down = np.tile(np.array([0.0, -1.0, 0.0]), (len(hp), 1))
        tang = 0.35 * radial / (np.linalg.norm(radial, axis=1, keepdims=True) + 1e-9) + 0.65 * down
        tang -= hn * np.sum(tang * hn, axis=1, keepdims=True)
        tang /= np.linalg.norm(tang, axis=1, keepdims=True) + 1e-9
        n = hp.shape[0]
        u = hash01(np.arange(n), 11)
        v = hash01(np.arange(n), 12)
        base = hp + hn * ((u - 0.5) * 0.0012 + (v > 0.99) * v * 0.004)[:, None]
        steps = (-2, -1, 0, 1, 2)
        dash = [base + tang * (k * 0.0017) for k in steps]
        self.hair_P = np.concatenate(dash).astype(np.float32)
        self.hair_N = np.concatenate([hn] * len(steps)).astype(np.float32)
        fade = np.clip(np.abs(face[outer]) / 0.010, 0.35, 1.0)
        self.hair_alb = (HAIR[None, :] * np.concatenate([fade] * len(steps))[:, None]).astype(np.float32)
        self.hair_area = np.float32(d["hair_a"] / len(steps) * 1.2)
        body_hood = _star_patch(d["body_P"], d["body_N"])
        self.body_alb = np.where(body_hood[:, None], PATCH, HOODIE).astype(np.float32)
        below = d["body_P"][:, 1] < 0.105
        legs = below | (d["body_P"][:, 2] > 0.13) & (d["body_P"][:, 1] < 0.12)
        self.body_alb[legs] = PANTS
        neck = (d["body_P"][:, 1] > 0.392) & (np.abs(d["body_P"][:, 0]) < 0.042) & (d["body_P"][:, 2] > -0.025)
        self.body_alb[neck] = SKIN
        self.head_alb = np.where(d["head_lip"][:, None], LIP, SKIN).astype(np.float32)
        # brows: darken a thin arc above each eye
        hp = d["head_P"]
        for ex in (EYE_L, EYE_R):
            q = hp - (ex + np.array([0, 0.018, 0.004]))
            arc = (np.abs(q[:, 1] + 3.0 * q[:, 0] ** 2 * 10) < 0.0022) & (np.abs(q[:, 0]) < 0.014) & (hp[:, 2] > 0.06)
            self.head_alb[arc] = HAIR * 3

    @staticmethod
    def _lashes(n=260, seed=31):
        """Upper-lid lashes for both eyes, head frame. Returns P, N."""
        rng = np.random.default_rng(seed)
        Ps, Ns = [], []
        for ex in (EYE_L, EYE_R):
            u = rng.uniform(-1, 1, n)
            s_ = np.sign(ex[0])
            x = ex[0] + u * 0.0125
            y = ex[1] + 0.0060 * (1 - u ** 2) ** 0.6 + 0.0005
            z = ex[2] + 0.0045 * (1 - u ** 2) ** 0.5 + 0.0012
            L = rng.uniform(0.2, 1.0, n) ** 1.5 * (0.0030 + 0.0035 * (1 - np.abs(u)))
            P = np.stack([x + s_ * 0.0006 * u, y + L * 0.35 + (L / 0.0065) ** 2 * 0.0012, z + L * 0.92], 1)
            N = np.tile(np.array([0.0, 0.5, 0.86]), (n, 1))
            Ps.append(P)
            Ns.append(N)
        return np.concatenate(Ps).astype(np.float32), np.concatenate(Ns).astype(np.float32)

    # ------------------------------------------------------------------ posing
    @staticmethod
    def _arm_frames(side, flex, abd, elbow, twist=0.0):
        sx = 1 if side == "L" else -1
        sh = np.array([0.150 * sx, 0.330, -0.01])
        Rs = rot("z", abd * sx) @ rot("x", -flex)
        Rf = Rs @ rot("x", -elbow) @ rot("y", twist)
        e = sh + Rs @ np.array([0, -0.215, 0])
        w = e + Rf @ np.array([0, -0.195, 0])
        return sh, Rs, e, Rf, w

    def pose(self, head_pitch=0.55, head_yaw=0.0, head_roll=0.0, armL=(0.55, 0.15, 1.2), armR=(0.55, 0.15, 1.2),
             torch_hand="R", torch_aim=None, blink=0.0, twistL=0.0, twistR=0.0):
        d = self.d
        clouds = []
        Rh = rot("y", head_yaw) @ rot("x", -head_pitch) @ rot("z", head_roll)

        def head_tf(P):
            return (P - HEAD_PIVOT_LOCAL) @ Rh.T + NECK_PIVOT

        # head
        clouds.append(SolidCloud(head_tf(d["head_P"]), d["head_N"] @ Rh.T, self.head_alb, area=d["head_a"],
                                 part=np.full(len(d["head_P"]), 0)))
        # eyes (blink: lid colour sweeps down over the eyeball)
        eP = d["eye_P"]
        ey = (eP[:, 1] - 0.004 + 0.0118) / 0.0236
        lid = ey > (1.0 - blink * 1.05)
        ealb = np.where(lid[:, None], SKIN * 0.85, EYE).astype(np.float32)
        clouds.append(SolidCloud(head_tf(eP), d["eye_N"] @ Rh.T, ealb, area=d["eye_a"], part=np.full(len(eP), 2)))
        lP, lN = self._lashes()
        lP = lP.copy()
        # lashes follow the lid when blinking
        lP[:, 1] -= blink * 0.011
        lP[:, 2] -= blink * 0.002
        clouds.append(SolidCloud(head_tf(lP), lN @ Rh.T, HAIR, area=np.float32(2.5e-7),
                                 part=np.full(len(lP), 1)))
        clouds.append(SolidCloud(head_tf(self.hair_P), self.hair_N @ Rh.T, self.hair_alb, area=self.hair_area,
                                 part=np.full(len(self.hair_P), 1)))
        clouds.append(SolidCloud(d["body_P"], d["body_N"], self.body_alb, area=d["body_a"],
                                 part=np.full(len(d["body_P"]), 3)))
        clouds.append(SolidCloud(d["feet_P"], d["feet_N"], SHOE, area=d["feet_a"], part=np.full(len(d["feet_P"]), 5)))
        info = {"eyes": [head_tf(EYE_L[None])[0], head_tf(EYE_R[None])[0]], "head_R": Rh}
        for side, prm, tw in (("L", armL, twistL), ("R", armR, twistR)):
            sh, Rs, e, Rf, w = self._arm_frames(side, *prm, twist=tw)
            clouds.append(SolidCloud(d["upper_P"] @ Rs.T + sh, d["upper_N"] @ Rs.T, HOODIE, area=d["upper_a"],
                                     part=np.full(len(d["upper_P"]), 3)))
            clouds.append(SolidCloud(d["fore_P"] @ Rf.T + e, d["fore_N"] @ Rf.T, HOODIE, area=d["fore_a"],
                                     part=np.full(len(d["fore_P"]), 3)))
            Rhand = Rf @ rot("y", 1.2 if side == "L" else -1.2)
            clouds.append(SolidCloud(d["hand_P"] @ Rhand.T + w, d["hand_N"] @ Rhand.T, SKIN, area=d["hand_a"],
                                     part=np.full(len(d["hand_P"]), 0)))
            info[f"wrist{side}"] = w
            info[f"hand{side}_R"] = Rhand
            if torch_hand == side:
                # flashlight held in the fist, pointing along the forearm or at torch_aim
                if torch_aim is not None:
                    ax = np.asarray(torch_aim, float)
                    ax /= np.linalg.norm(ax)
                else:
                    ax = Rf @ np.array([0, -1.0, 0])
                up = np.array([0, 1.0, 0])
                v = np.cross(up, ax)
                s = np.linalg.norm(v)
                if s < 1e-6:
                    Rt = np.eye(3)
                else:
                    v /= s
                    ang = np.arccos(np.clip(up @ ax, -1, 1))
                    K = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
                    Rt = np.eye(3) + np.sin(ang) * K + (1 - np.cos(ang)) * K @ K
                base = w + Rf @ np.array([0, -0.05, 0.012]) - ax * 0.05
                clouds.append(SolidCloud(d["torch_P"] @ Rt.T + base, d["torch_N"] @ Rt.T, TORCH, area=d["torch_a"],
                                         part=np.full(len(d["torch_P"]), 6)))
                info["torch_tip"] = base + ax * 0.132
                info["torch_dir"] = ax
        return SolidCloud.concat(clouds), info


# =============================================================================== THE ELDER
# Giacometti-like: very tall, very thin, rough surface, fragile. Self-luminous and translucent.
ELDER_CORE = _lin([1.0, 0.80, 0.42])
ELDER_EDGE = _lin([1.0, 0.36, 0.05])
ELDER_EYE = _lin([1.0, 0.90, 0.65])


def elder_body_sdf(p):
    d = sd_round_cone(p, (0, 0.06, 0), (0, 0.95, 0.0), 0.19, 0.085)                 # long robe to waist
    d = smin(d, sd_round_cone(p, (0, 0.95, 0.0), (0, 1.45, 0.005), 0.085, 0.105), 0.05)  # narrow chest
    d = smin(d, sd_ellipsoid(p, (0, 1.47, 0.0), (0.150, 0.045, 0.075)), 0.05)      # thin sloping shoulders
    d = smin(d, sd_round_cone(p, (0, 1.49, 0.0), (0, 1.86, 0.035), 0.030, 0.024), 0.03)  # long thin neck
    d = smin(d, sd_ellipsoid(p, (0, 0.05, 0.05), (0.13, 0.05, 0.19)), 0.06)          # heavy feet/base
    # Giacometti surface: lumps + fine gouges
    d = d + 0.010 * fbm(p * np.array([7.0, 2.5, 7.0]), octaves=2) + 0.005 * fbm(p * 38.0, octaves=2)
    return d


def elder_head_sdf(p):
    Rm = rot("x", np.radians(18))
    d = sd_ellipsoid(p, (0, 1.975, -0.02), (0.050, 0.108, 0.080), Rm)               # narrow, long skull
    d = smin(d, sd_ellipsoid(p, (0, 1.925, 0.030), (0.040, 0.060, 0.045)), 0.025)
    for sx in (1, -1):
        d = smax(d, -sd_ellipsoid(p, (0.020 * sx, 1.950, 0.070), (0.012, 0.008, 0.010)), 0.006)
    d = smin(d, sd_round_cone(p, (0, 1.968, 0.064), (0, 1.915, 0.084), 0.009, 0.012), 0.012)   # brow-nose ridge
    d = smin(d, sd_ellipsoid(p, (0, 1.975, 0.055), (0.036, 0.010, 0.014)), 0.01)              # brow
    d = d + 0.004 * fbm(p * 45.0, octaves=2)
    return d


class Elder:
    """Self-luminous, translucent Giacometti figure. Arms are smooth tubes re-sampled per pose."""

    SH = np.array([0.135, 1.455, 0.0])

    def __init__(self, density=1.0):
        from ..solid import bezier, tube
        self._tube, self._bez = tube, bezier
        self.density = density

        def make():
            out = {}
            P, N, a = sample_surface(elder_body_sdf, (-0.28, 0.0, -0.22), (0.28, 1.92, 0.30),
                                     int(120_000 * density), seed=21)
            out["body_P"], out["body_N"], out["body_a"] = P, N, np.float32(a)
            P, N, a = sample_surface(elder_head_sdf, (-0.10, 1.80, -0.14), (0.10, 2.13, 0.12),
                                     int(30_000 * density), seed=22)
            out["head_P"], out["head_N"], out["head_a"] = P, N, np.float32(a)
            return out
        self.d = _cache(f"elder3_{density}", make)
        rng = np.random.default_rng(7)
        self.key_body = rng.random(self.d["body_P"].shape[0]).astype(np.float32)
        self.key_head = rng.random(self.d["head_P"].shape[0]).astype(np.float32)

    def _arm(self, side, flex, abd, elbow, lean_R, lean_piv, n=9000, seed=0):
        sx = -1 if side == "R" else 1
        sh = self.SH * np.array([sx, 1, 1])
        Rs = lean_R @ rot("z", abd * sx) @ rot("x", -flex)
        Rf = Rs @ rot("x", -elbow)
        shw = (sh - lean_piv) @ lean_R.T + lean_piv
        e = shw + Rs @ np.array([0, -0.46, 0])
        w = e + Rf @ np.array([0, -0.44, 0])
        curve = np.concatenate([self._bez(shw, (shw + e) / 2, e, 12)[:-1], self._bez(e, (e + w) / 2, w, 12)])
        radii = np.concatenate([np.linspace(0.030, 0.021, 11), np.linspace(0.021, 0.015, 12)])
        P, N, a = self._tube(curve, radii, int(n * self.density), seed=seed,
                             bumps=lambda u, th: 0.18 * np.sin(u * 37 + th * 3) * np.sin(u * 13))
        return P, N, a, w, Rf

    def hand(self, Rm, origin, curl=0.0, spread=0.25, n=5200, seed=0, finger_len=1.0):
        """Long-fingered hand from tubes. Local: fingers point -y, palm faces +z."""
        Ps, Ns, As = [], [], []
        # palm: short flat tube bundle
        for k, dx in enumerate((-0.018, -0.006, 0.006, 0.018)):
            c = np.array([[dx * 0.8, 0.0, 0.0], [dx, -0.05, 0.0], [dx * 1.05, -0.095, 0.0]])
            P, N, a = self._tube(c, [0.0105, 0.0115, 0.010], n // 12, seed=seed + k)
            Ps.append(P)
            Ns.append(N)
            As.append(np.full(len(P), a))
        for fi, bx in enumerate((-0.020, -0.007, 0.007, 0.020)):
            ang = (fi - 1.5) * spread * 0.16
            pos = np.array([bx * 1.05, -0.095, 0.0])
            Rf = rot("z", ang)
            pts = [pos.copy()]
            L = (0.080, 0.090, 0.086, 0.070)[fi] * finger_len
            for k in range(3):
                Rf = Rf @ rot("x", curl * (0.5 + 0.25 * k))
                pos = pos + Rf @ np.array([0, -L / 3, 0])
                pts.append(pos.copy())
            c = self._bez(pts[0], pts[1], pts[2], 6)
            c = np.concatenate([c, self._bez(pts[2], (pts[2] + pts[3]) / 2, pts[3], 5)[1:]])
            P, N, a = self._tube(c, np.linspace(0.0085, 0.0055, len(c)), n // 7, seed=seed + 10 + fi)
            Ps.append(P)
            Ns.append(N)
            As.append(np.full(len(P), a))
        # thumb
        Rt = rot("z", 0.9 + spread * 0.4)
        p0 = np.array([0.022, -0.020, 0.004])
        p1 = p0 + Rt @ np.array([0, -0.045, 0.0])
        p2 = p1 + Rt @ rot("x", curl * 0.5) @ np.array([0, -0.040, 0.0])
        P, N, a = self._tube(self._bez(p0, p1, p2, 8), np.linspace(0.010, 0.0065, 8), n // 8, seed=seed + 20)
        Ps.append(P)
        Ns.append(N)
        As.append(np.full(len(P), a))
        P = np.concatenate(Ps) @ np.asarray(Rm).T + origin
        N = np.concatenate(Ns) @ np.asarray(Rm).T
        return P.astype(np.float32), N.astype(np.float32), np.concatenate(As).astype(np.float32)

    def pose(self, head_pitch=0.0, head_yaw=0.0, armR=(0.1, 0.1, 0.1), armL=(0.1, 0.1, 0.1),
             palmR=None, curlR=0.8, curlL=0.9, lean=0.0, spreadR=0.25, arm_n=9000, hand_n=5200,
             skip_hand_R=False):
        d = self.d
        Ps, Ns, As, Ks, parts = [], [], [], [], []
        Rl = rot("x", lean)
        piv = np.array([0, 0.9, 0])

        def lean_tf(P):
            return (P - piv) @ Rl.T + piv

        Ps.append(lean_tf(d["body_P"]))
        Ns.append(d["body_N"] @ Rl.T)
        As.append(np.full(len(d["body_P"]), d["body_a"]))
        Ks.append(self.key_body)
        parts.append(np.zeros(len(d["body_P"]), np.int16))
        hp = np.array([0, 1.86, 0.035])
        Rh = rot("y", head_yaw) @ rot("x", -head_pitch)
        Ph = (d["head_P"] - hp) @ Rh.T + hp
        Ps.append(lean_tf(Ph))
        Ns.append(d["head_N"] @ (Rl @ Rh).T)
        As.append(np.full(len(Ph), d["head_a"]))
        Ks.append(self.key_head)
        parts.append(np.ones(len(Ph), np.int16))
        eyes = [lean_tf(((np.array([sx * 0.020, 1.950, 0.075]) - hp) @ Rh.T + hp)[None])[0] for sx in (1, -1)]
        info = {"eyes": eyes, "head_R": Rl @ Rh}
        for side, prm, curl, sd in (("R", armR, curlR, 100), ("L", armL, curlL, 200)):
            P, N, a, w, Rf = self._arm(side, *prm, Rl, piv, n=arm_n, seed=sd)
            k = hash01(np.arange(len(P)), sd).astype(np.float32)
            Ps.append(P)
            Ns.append(N)
            As.append(np.full(len(P), a))
            Ks.append(k)
            parts.append(np.full(len(P), 2, np.int16))
            sx = -1 if side == "R" else 1
            Rhand = palmR if (palmR is not None and side == "R") else Rf @ rot("y", np.pi / 2 * -sx)
            info[f"wrist{side}"] = w
            info[f"hand{side}_R"] = Rhand
            info[f"palm{side}"] = w + np.asarray(Rhand) @ np.array([0, -0.06, 0.012])
            if side == "R" and skip_hand_R:
                continue
            hP, hN, hA = self.hand(Rhand, w, curl=curl, spread=spreadR if side == "R" else 0.2, seed=sd + 50,
                                   n=hand_n)
            Ps.append(hP)
            Ns.append(hN)
            As.append(hA)
            Ks.append(hash01(np.arange(len(hP)), sd + 1).astype(np.float32))
            parts.append(np.full(len(hP), 3, np.int16))
            info[f"wrist{side}"] = w
            info[f"hand{side}_R"] = Rhand
            info[f"palm{side}"] = w + np.asarray(Rhand) @ np.array([0, -0.06, 0.012])
        P = np.concatenate(Ps).astype(np.float32)
        N = np.concatenate(Ns).astype(np.float32)
        kk = np.concatenate(Ks)
        u = hash01(np.arange(len(P)), 91)[:, None]
        col = (ELDER_CORE * 0.55 + ELDER_EDGE * 0.45) * (0.75 + 0.5 * u)
        col = np.where(kk[:, None] < 0.08, ELDER_EDGE * 0.9, col)          # deep-orange flecks
        col = np.where((kk[:, None] > 0.94), ELDER_CORE * 1.3, col).astype(np.float32)   # pale sparks
        return dict(P=P, N=N, col=col, area=np.concatenate(As).astype(np.float32),
                    key=np.concatenate(Ks).astype(np.float32), part=np.concatenate(parts)), info


def elder_points(pose, cam, t, dissolve=0.0, shed=0.04, energy=1.0, spacing_px=2.0, up=(0, 1, 0),
                 wind=(-0.4, 0.2, 0), interior=0.14, rim_pow=1.6, W_ref=1920):
    """Turn an elder pose into drawable glowing points for R.draw(cam, P, E).

    Translucent light-being look: silhouette edges dense and bright, interior sparse and dim.
    Energy follows projected area so close-ups keep their brightness; points are thinned to a
    constant on-screen spacing (pointillist grain). dissolve in [0,1] releases points upward."""
    P, N, col, key, area = pose["P"], pose["N"], pose["col"], pose["key"], pose["area"]
    V = cam.pos[None, :] - P
    dist = np.linalg.norm(V, axis=1)
    V /= dist[:, None] + 1e-9
    ndv = np.abs(np.sum(N * V, axis=1))
    rim = (1 - ndv) ** rim_pow
    # projected spacing of the samples, thinning to target spacing
    s = cam.W / W_ref
    native = np.sqrt(area) * cam.fpx / np.maximum(dist, 1e-6)
    target = spacing_px * s
    frac = np.clip((native / target) ** 2, 0, 1)
    keep_p = frac * (interior + (1 - interior) * rim)
    keep = hash01(np.arange(len(P)), 77) < keep_p
    k2 = np.nonzero(keep)[0]
    P, col, key, rim, native, dist = P[k2], col[k2], key[k2], rim[k2], native[k2], dist[k2]
    area_px = np.maximum(native, target) ** 2 / (s * s)
    bright = (0.55 + 1.8 * rim) * energy * area_px / np.maximum(frac[k2], 1e-6) * frac[k2]
    e = col * bright[:, None] * 0.35
    # ash: a small fraction of edge points drifts off and fades (a dying body)
    phase = (key * 17.0 + t * 0.35) % 1.0
    is_ash = (key < shed) & (rim > 0.25)
    drift = np.asarray(wind)[None, :] * (phase[:, None] * 0.35) + np.asarray(up)[None, :] * (phase[:, None] * 0.12)
    P2 = P + np.where(is_ash[:, None], drift, 0)
    e = e * np.where(is_ash, (1 - phase) ** 1.5, 1.0)[:, None]
    if dissolve > 0:
        rel = key < dissolve
        age = np.clip((dissolve - key) / 0.25, 0, 1)
        lift = P2 + np.asarray(up)[None, :] * (age[:, None] ** 1.6 * 1.8) + \
            np.asarray(wind)[None, :] * (age[:, None] * 0.4)
        P2 = np.where(rel[:, None], lift, P2)
        e = e * np.where(rel, (1 - age) ** 1.2, 1.0)[:, None]
    return P2.astype(np.float32), e.astype(np.float32)


class ElderHand:
    """Sculpted long-fingered hand for extreme close-ups (I5 and the climax palm).

    Local frame: wrist at the origin, fingers along -y, palm normal +z. Each phalanx is sampled
    in its own joint frame so the fingers can curl/open without re-sampling."""

    KNUCKLES = [(-0.029, -0.098), (-0.0098, -0.104), (0.0098, -0.102), (0.028, -0.094)]
    LENGTHS = [(0.052, 0.040, 0.031), (0.060, 0.046, 0.035), (0.058, 0.044, 0.034), (0.047, 0.036, 0.028)]
    RADII = [(0.0098, 0.0082, 0.0070, 0.0058), (0.0104, 0.0088, 0.0074, 0.0060),
             (0.0102, 0.0086, 0.0072, 0.0059), (0.0090, 0.0076, 0.0065, 0.0054)]

    def __init__(self, density=1.0):
        def make():
            out = {}

            def palm(p):
                d = sd_ellipsoid(p, (0.0, -0.052, 0.0), (0.043, 0.058, 0.0135))
                d = smin(d, sd_ellipsoid(p, (0.024, -0.032, 0.006), (0.020, 0.030, 0.014)), 0.012)   # thenar
                d = smin(d, sd_ellipsoid(p, (-0.02, -0.03, 0.004), (0.02, 0.03, 0.012)), 0.012)      # hypothenar
                d = smin(d, sd_capsule(p, (0, 0.02, 0), (0, -0.01, 0), 0.020), 0.02)                 # wrist
                for kx, ky in ElderHand.KNUCKLES:
                    d = smin(d, sd_sphere(p, (kx, ky + 0.004, 0.0), 0.0105), 0.010)
                return d + 0.0012 * fbm(p * 160.0, octaves=2)
            P, N, a = sample_surface(palm, (-0.07, -0.13, -0.03), (0.07, 0.05, 0.03), int(60_000 * density), seed=61)
            out["palm_P"], out["palm_N"], out["palm_a"] = P, N, np.float32(a)
            for f in range(4):
                for k in range(3):
                    L = ElderHand.LENGTHS[f][k]
                    r1, r2 = ElderHand.RADII[f][k], ElderHand.RADII[f][k + 1]

                    def seg(p, L=L, r1=r1, r2=r2):
                        return sd_round_cone(p, (0, 0, 0), (0, -L, 0), r1, r2) + 0.0008 * fbm(p * 200.0, octaves=2)
                    R_ = r1 + 0.006
                    P, N, a = sample_surface(seg, (-R_, -L - R_, -R_), (R_, R_, R_), int(9_000 * density),
                                             seed=70 + f * 3 + k)
                    out[f"f{f}{k}_P"], out[f"f{f}{k}_N"], out[f"f{f}{k}_a"] = P, N, np.float32(a)
            for k, (L, r1, r2) in enumerate(((0.050, 0.0118, 0.0092), (0.040, 0.0092, 0.0066))):
                def seg(p, L=L, r1=r1, r2=r2):
                    return sd_round_cone(p, (0, 0, 0), (0, -L, 0), r1, r2)
                R_ = r1 + 0.006
                P, N, a = sample_surface(seg, (-R_, -L - R_, -R_), (R_, R_, R_), int(9_000 * density), seed=90 + k)
                out[f"t{k}_P"], out[f"t{k}_N"], out[f"t{k}_a"] = P, N, np.float32(a)
            return out
        self.d = _cache(f"elderhand_{density}", make)
        rng = np.random.default_rng(12)
        self.keys = {k[:-2]: rng.random(self.d[k].shape[0]).astype(np.float32) for k in self.d if k.endswith("_P")}

    def pose(self, curl=0.0, spread=0.3):
        """Hand-local points. curl 0 = open flat, 1 = closed; fingers bend toward +z (the palm side)."""
        d = self.d
        Ps, Ns, As, Ks = [d["palm_P"]], [d["palm_N"]], [np.full(len(d["palm_P"]), d["palm_a"])], [self.keys["palm"]]
        for f, (kx, ky) in enumerate(self.KNUCKLES):
            pos = np.array([kx, ky, 0.0])
            Rf = rot("z", (f - 1.5) * spread * 0.12)
            for k in range(3):
                Rf = Rf @ rot("x", -curl * (0.95 + 0.35 * k))
                Ps.append(d[f"f{f}{k}_P"] @ Rf.T + pos)
                Ns.append(d[f"f{f}{k}_N"] @ Rf.T)
                As.append(np.full(len(d[f"f{f}{k}_P"]), d[f"f{f}{k}_a"]))
                Ks.append(self.keys[f"f{f}{k}"])
                pos = pos + Rf @ np.array([0, -self.LENGTHS[f][k], 0])
        pos = np.array([0.034, -0.026, 0.008])
        Rt = rot("z", 0.95 + spread * 0.35) @ rot("y", -0.35)
        for k in range(2):
            Rt = Rt @ rot("x", -curl * 0.7)
            Ps.append(d[f"t{k}_P"] @ Rt.T + pos)
            Ns.append(d[f"t{k}_N"] @ Rt.T)
            As.append(np.full(len(d[f"t{k}_P"]), d[f"t{k}_a"]))
            Ks.append(self.keys[f"t{k}"])
            pos = pos + Rt @ np.array([0, -(0.050, 0.040)[k], 0])
        return (np.concatenate(Ps).astype(np.float32), np.concatenate(Ns).astype(np.float32),
                np.concatenate(As).astype(np.float32), np.concatenate(Ks).astype(np.float32))

    def palm_center(self):
        return np.array([0.0, -0.055, 0.014])


def hand_pose_dict(hand, Rm, origin, curl, spread=0.3):
    """World-space pose dict for elder_points from an ElderHand."""
    P, N, A, K = hand.pose(curl, spread)
    Rm = np.asarray(Rm)
    Pw = P @ Rm.T + origin
    Nw = N @ Rm.T
    u = hash01(np.arange(len(P)), 93)[:, None]
    col = (ELDER_CORE * 0.55 + ELDER_EDGE * 0.45) * (0.75 + 0.5 * u)
    col = np.where(K[:, None] < 0.08, ELDER_EDGE * 0.9, col)
    col = np.where(K[:, None] > 0.94, ELDER_CORE * 1.3, col).astype(np.float32)
    return dict(P=Pw.astype(np.float32), N=Nw.astype(np.float32), col=col, area=A, key=K,
                part=np.zeros(len(P), np.int16))


def elder_ik(elder, target_wrist, side="R", lean=0.0, x0=(0.9, 0.1, 0.4)):
    """Solve (flex, abd, elbow) so the wrist lands on target_wrist (figure space)."""
    from scipy.optimize import minimize
    Rl = rot("x", lean)
    piv = np.array([0, 0.9, 0])
    sx = -1 if side == "R" else 1
    sh = Elder.SH * np.array([sx, 1, 1])
    shw = (sh - piv) @ Rl.T + piv

    def wrist(q):
        flex, abd, elbow = q
        Rs = Rl @ rot("z", abd * sx) @ rot("x", -flex)
        Rf = Rs @ rot("x", -elbow)
        return shw + Rs @ np.array([0, -0.46, 0]) + Rf @ np.array([0, -0.44, 0])

    def cost(q):
        return float(np.sum((wrist(q) - target_wrist) ** 2) + 1e-4 * (q[1] ** 2 + q[2] ** 2))
    r = minimize(cost, np.asarray(x0, float), method="Nelder-Mead", options=dict(xatol=1e-5, fatol=1e-9, maxiter=4000))
    return tuple(r.x), float(np.sqrt(np.sum((wrist(r.x) - target_wrist) ** 2)))
