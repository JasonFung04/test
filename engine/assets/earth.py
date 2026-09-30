"""Earth at night, seen from above (E2, E3). Units: km. Globe centre at the origin, +y = north pole.

City lights are procedural but anchored on real geography: a land mask (global-land-mask, GLOBE
1 km data) and ~130 real urban areas with approximate populations. The girl's city sits at the
Yangtze delta (31.23 N, 121.47 E); the film never names it.
"""
import numpy as np

from ..color import hex_lin
from ..config import CACHE
from ..noise import fbm

R_E = 6371.0
HOME = (31.23, 121.47)

# (lat, lon, population in millions) -- approximate metropolitan populations
CITIES = [
    (31.23, 121.47, 29), (39.90, 116.40, 22), (22.54, 114.06, 18), (23.13, 113.26, 19), (30.57, 104.07, 17),
    (29.56, 106.55, 17), (39.34, 117.36, 14), (30.59, 114.31, 12), (32.06, 118.80, 9), (30.27, 120.16, 12),
    (34.26, 108.94, 13), (36.07, 120.38, 10), (38.04, 114.51, 11), (41.80, 123.43, 9), (45.75, 126.63, 10),
    (34.75, 113.62, 12), (28.23, 112.94, 10), (26.07, 119.30, 8), (24.48, 118.09, 5), (31.30, 120.58, 12),
    (31.82, 117.23, 9), (36.65, 117.12, 9), (25.04, 102.71, 8), (26.65, 106.63, 6), (37.87, 112.55, 5),
    (43.82, 125.32, 9), (22.82, 108.37, 8), (20.04, 110.34, 3), (28.68, 115.86, 6), (35.69, 139.69, 37),
    (34.69, 135.50, 19), (35.18, 136.91, 9), (33.59, 130.40, 5), (43.06, 141.35, 2), (37.57, 126.98, 25),
    (35.18, 129.08, 3), (25.03, 121.57, 7), (22.63, 120.30, 3), (14.60, 120.98, 24), (10.82, 106.63, 13),
    (21.03, 105.85, 8), (13.76, 100.50, 17), (3.14, 101.69, 8), (1.35, 103.82, 6), (-6.21, 106.85, 34),
    (-7.25, 112.75, 9), (16.87, 96.20, 7), (23.81, 90.41, 22), (22.57, 88.36, 15), (28.61, 77.21, 32),
    (19.08, 72.88, 21), (12.97, 77.59, 13), (13.08, 80.27, 11), (17.39, 78.49, 10), (23.02, 72.57, 8),
    (24.86, 67.01, 16), (31.55, 74.34, 13), (33.69, 73.06, 5), (34.53, 69.17, 5), (35.69, 51.39, 15),
    (33.31, 44.36, 7), (24.71, 46.68, 7), (25.20, 55.27, 3), (41.01, 28.98, 15), (30.04, 31.24, 21),
    (55.76, 37.62, 17), (59.93, 30.34, 5), (50.45, 30.52, 3), (52.23, 21.01, 3), (52.52, 13.40, 4),
    (48.86, 2.35, 11), (51.51, -0.13, 9), (40.42, -3.70, 6), (41.39, 2.17, 5), (45.46, 9.19, 4),
    (41.90, 12.50, 4), (50.85, 4.35, 2), (52.37, 4.90, 2), (48.14, 11.58, 2), (47.50, 19.04, 2),
    (6.52, 3.38, 15), (-1.29, 36.82, 5), (-26.20, 28.05, 10), (9.03, 38.74, 5), (33.57, -7.59, 4),
    (40.71, -74.01, 20), (34.05, -118.24, 13), (41.88, -87.63, 9), (29.76, -95.37, 7), (32.78, -96.80, 7),
    (25.76, -80.19, 6), (33.75, -84.39, 6), (38.91, -77.04, 6), (42.36, -71.06, 5), (37.77, -122.42, 5),
    (47.61, -122.33, 4), (43.65, -79.38, 6), (45.50, -73.57, 4), (19.43, -99.13, 22), (4.71, -74.07, 11),
    (-12.05, -77.04, 11), (-23.55, -46.63, 22), (-22.91, -43.17, 13), (-34.60, -58.38, 15), (-33.45, -70.67, 7),
    (-33.87, 151.21, 5), (-37.81, 144.96, 5), (-27.47, 153.03, 2), (-31.95, 115.86, 2), (-36.85, 174.76, 2),
    (60.17, 24.94, 1), (59.33, 18.07, 2), (55.68, 12.57, 2), (53.35, -6.26, 2), (38.72, -9.14, 3),
    (37.98, 23.73, 3), (44.43, 26.10, 2), (42.70, 23.32, 1), (40.18, 44.51, 1), (41.30, 69.24, 3),
    (43.24, 76.95, 2), (47.92, 106.92, 1.5), (39.03, 125.75, 3), (40.84, 111.75, 3), (36.06, 103.83, 4),
    (43.83, 87.62, 4), (29.65, 91.13, 0.8), (38.47, 106.27, 2), (36.62, 101.78, 2), (35.10, 118.35, 5),
    (34.34, 117.28, 5), (32.40, 119.41, 4), (31.49, 120.31, 7), (30.00, 120.58, 5), (27.99, 120.70, 9),
]


def latlon_to_xyz(lat, lon, r=R_E):
    la, lo = np.radians(lat), np.radians(lon)
    return np.stack([r * np.cos(la) * np.sin(lo), r * np.sin(la), r * np.cos(la) * np.cos(lo)], -1)


def enu(lat, lon):
    """East, North, Up unit vectors at a point (globe frame)."""
    la, lo = np.radians(lat), np.radians(lon)
    up = np.array([np.cos(la) * np.sin(lo), np.sin(la), np.cos(la) * np.cos(lo)])
    east = np.array([np.cos(lo), 0.0, -np.sin(lo)])
    north = np.cross(up, east)
    return east, north, up


def city_to_globe(P_m):
    """City-world metres (+x east, +y up, +z south) at HOME -> globe km."""
    e, n, u = enu(*HOME)
    P = np.asarray(P_m, float) / 1000.0
    base = u * R_E
    return base + P[..., 0:1] * e + P[..., 1:2] * u - P[..., 2:3] * n


def lights(n=1_600_000, seed=12):
    path = CACHE / f"earth_lights2_{n}_{seed}.npz"
    if path.exists():
        d = np.load(path)
        return {k: d[k] for k in d.files}
    from global_land_mask import globe
    rng = np.random.default_rng(seed)
    C = np.array(CITIES, float)
    w = C[:, 2] ** 0.8
    w /= w.sum()
    n_city = int(n * 0.50)
    idx = rng.choice(len(C), n_city, p=w)
    spread = 0.07 + 0.05 * np.sqrt(C[idx, 2])                 # degrees
    r = spread * np.exp(rng.normal(0.0, 0.75, n_city))       # lognormal: dense core, long soft tail
    a = rng.uniform(0, 2 * np.pi, n_city)
    lat = C[idx, 0] + r * np.sin(a)
    lon = C[idx, 1] + r * np.cos(a) / np.cos(np.radians(C[idx, 0]))
    core = np.exp(-(r / spread) ** 2)
    b_city = (0.18 + 0.7 * core) * (0.5 + 0.5 * rng.random(n_city))
    # peri-urban halo: wide, sparse, dim (the lit countryside around big regions)
    n_halo = int(n * 0.18)
    ih = rng.choice(len(C), n_halo, p=w)
    rh = np.abs(rng.normal(0, 1.0, n_halo)) * (0.6 + 0.35 * np.sqrt(C[ih, 2]))
    ah = rng.uniform(0, 2 * np.pi, n_halo)
    lat_h = C[ih, 0] + rh * np.sin(ah)
    lon_h = C[ih, 1] + rh * np.cos(ah) / np.cos(np.radians(C[ih, 0]))
    b_halo = 0.05 + 0.10 * rng.random(n_halo)
    # corridors between near neighbours (meandering roads and rail towns)
    n_road = int(n * 0.10)
    i1 = rng.choice(len(C), n_road, p=w)
    d = np.hypot(C[:, None, 0] - C[None, :, 0], C[:, None, 1] - C[None, :, 1])
    np.fill_diagonal(d, 1e9)
    nb = np.argsort(d, axis=1)[:, :3]
    i2 = nb[i1, rng.integers(0, 3, n_road)]
    ok = d[i1, i2] < 10.0
    t = rng.random(n_road)
    seed_pair = (i1 * 131 + i2) % 997
    mean = 0.35 * np.sin(t * np.pi * 3 + seed_pair) * np.sin(t * np.pi)
    lat_r = C[i1, 0] + (C[i2, 0] - C[i1, 0]) * t + mean + rng.normal(0, 0.04, n_road)
    lon_r = C[i1, 1] + (C[i2, 1] - C[i1, 1]) * t - mean + rng.normal(0, 0.04, n_road)
    lat_r, lon_r = lat_r[ok], lon_r[ok]
    b_road = 0.08 + 0.12 * rng.random(lat_r.size)
    lat = np.concatenate([lat, lat_h])
    lon = np.concatenate([lon, lon_h])
    b_city = np.concatenate([b_city, b_halo])
    n_city = lat.size
    # rural scatter, stronger in populous regions (noise-modulated)
    n_rur = n - n_city - lat_r.size
    lat_s = np.degrees(np.arcsin(rng.uniform(-0.85, 0.95, n_rur * 2)))
    lon_s = rng.uniform(-180, 180, n_rur * 2)
    xyz = latlon_to_xyz(lat_s, lon_s, 1.0)
    dens = np.clip(fbm(xyz * 6.0, octaves=4) * 1.8 + 0.1, 0, 1)
    keep = rng.random(lat_s.size) < dens
    lat_s, lon_s = lat_s[keep][:n_rur], lon_s[keep][:n_rur]
    b_rur = 0.05 + 0.12 * rng.random(lat_s.size)
    lat = np.concatenate([lat, lat_r, lat_s])
    lon = np.concatenate([lon, lon_r, lon_s])
    b = np.concatenate([b_city, b_road, b_rur])
    lon = (lon + 180) % 360 - 180
    lat = np.clip(lat, -89.9, 89.9)
    land = globe.is_land(lat, lon)
    lat, lon, b = lat[land], lon[land], b[land]
    warm = rng.random(lat.size)
    col = np.where(warm[:, None] < 0.78, hex_lin("#FFA548"), hex_lin("#E8EEFF"))
    rgb = (col * b[:, None]).astype(np.float32)
    P = latlon_to_xyz(lat, lon, R_E + 0.05).astype(np.float32)
    d = dict(P=P, rgb=rgb, lat=lat.astype(np.float32), lon=lon.astype(np.float32))
    np.savez(path, **d)
    return d


def land_dots(n=900_000, seed=13):
    """Faint land albedo dots (starlit/airglow-lit continents) + ocean glints: lets the globe read."""
    path = CACHE / f"earth_land_{n}_{seed}.npz"
    if path.exists():
        d = np.load(path)
        return {k: d[k] for k in d.files}
    from global_land_mask import globe
    rng = np.random.default_rng(seed)
    v = rng.normal(size=(n, 3))
    v /= np.linalg.norm(v, axis=1, keepdims=True)
    lat = np.degrees(np.arcsin(v[:, 1]))
    lon = np.degrees(np.arctan2(v[:, 0], v[:, 2]))
    land = globe.is_land(lat, lon)
    rgb = np.where(land[:, None], hex_lin("#2A2A30") * 0.10, hex_lin("#0A1830") * 0.035).astype(np.float32)
    d = dict(P=(v * R_E).astype(np.float32), rgb=rgb, land=land)
    np.savez(path, **d)
    return d


def atmosphere(cam, n=160_000, seed=14, energy=1.0):
    """Limb glow: a thin shell of soft blue points, brightest edge-on."""
    rng = np.random.default_rng(seed)
    v = rng.normal(size=(n, 3))
    v /= np.linalg.norm(v, axis=1, keepdims=True)
    h = np.abs(rng.normal(0, 18.0, n)) + 2.0
    P = v * (R_E + h)[:, None]
    view = cam.pos[None, :] - P
    view /= np.linalg.norm(view, axis=1, keepdims=True)
    grazing = 1.0 - np.abs(np.sum(v * view, axis=1))
    col = hex_lin("#4F7BFF") * (0.05 * energy * grazing ** 6 * np.exp(-h / 25.0))[:, None]
    airglow = (h > 85) & (h < 110)
    col[airglow] += hex_lin("#5CFF9A") * 0.01 * energy * grazing[airglow, None] ** 8
    return P.astype(np.float32), col.astype(np.float32)
