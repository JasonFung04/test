# Engine guide (for anyone writing shots)

Pointillist renderer in numpy/numba/OpenCV. Everything on screen is points of light, splatted
additively into a linear-light HDR buffer, then bloom -> ACES filmic -> grade -> cards -> grain -> 8-bit.

## Run
```
python -m engine.render sheet I1 --n 6 --scale 0.5   # contact sheet -> build/preview/I1.png (look at it with Read)
python -m engine.render frame 31.0                   # one full-res frame at global time t -> build/preview/t0031.00.png
python -m engine.render mp4 I1 --scale 0.5           # quick clip -> build/preview/I1.mp4 (uses all CPUs)
python -m engine.render shot I1                      # final PNG frames -> build/frames/
```
Timing lives ONLY in `engine/timeline.py` (SHOTS, CARDS, EVENTS, typing_schedule, BLACKOUT_OFF, POWER_ON).
Never hard-code a time that exists there; read `TL.EVENTS[...]`. Shot functions receive `tl` (seconds
since shot start) and `tg` (global seconds).

## Shot modules
`engine/shots/<module>.py` exposes `render(sid, tl, tg, R, W, H) -> (hdr, Grade)` via
`dispatch(TABLE, ...)` from `shots/common.py`. Missing shots render a grey placeholder label.
Optional `overlay(sid, tl, tg, y)` draws display-space overlays (0..1 RGB) after tone mapping
(used for UI text); title cards from `TL.CARDS` are applied automatically by the driver.
Module-level caches (`_C = {}`) hold heavy assets per worker process.

## Camera (`engine/camera.py`)
`Camera(pos, target, up=(0,1,0), focal=35, roll=0, focus=None, bokeh=0, W, H)`
- focal in mm on a 36 mm-wide sensor (see SHOTLIST lens table). `focus` defaults to |target-pos|.
- `bokeh` = circle-of-confusion radius in px (@1920) for objects at infinity; coc = bokeh*|1-focus/z|.
  Shallow DOF = the film's signature look: use it on close-ups (e.g. bokeh 20-60).
- helpers: `ease, ease5, ease_in, ease_out, lerp, seg(t,t0,t1), orbit(center,r,yaw,pitch), handheld(t, amp, seed)`.
- Cosmic shots: perfectly smooth eased moves. Human shots (cave, rooftop): add `handheld(t, amp)` drift
  (<=0.3 deg rotation equivalent) to position/target.

## Renderer (`engine/raster.py`)
`R.draw(cam, P, rgb, size=0, size_px=0, falloff=0, zref=1, soft=False, energy=1)`
- P (N,3) world points; rgb (N,3) or (3,) LINEAR energy (use `color.hex_lin('#RRGGBB', k)`).
- `size` = world-space radius, `size_px` = intrinsic px radius (@1920); DOF blur is added on top.
- `soft=True` -> gaussian sprite (glow, gas); default -> disk (crisp dot / bokeh disc).
- `falloff=2, zref=d` makes energy scale with (d/z)^2 (distant things dimmer).
- Energies are authored at 1920 wide; previews at lower scale look like a box-downsample (correct).
- Typical energies: a crisp star 0.05-2; dense surface points 0.5-3 each; soft glow points 0.01-0.3.
`R.draw_dirs(cam, D, rgb, size_px)` draws directions at infinity (star fields).
`R.draw2d(x, y, rgb, r_px)` draws screen-space points.
Layers/occlusion: additive has no occlusion. To hide background behind a silhouette:
`bg = R.new_layer()` (resolves & resets), multiply `bg *= (1-mask)[...,None]`, keep drawing, return `bg + R.resolve()`.
`assets/bodies.sphere_mask(cam, center, radius, W, H)` gives an anti-aliased sphere mask; for other
shapes rasterise a polygon mask with cv2.fillPoly at 4x and INTER_AREA down.

## Post (`engine/post.py`)
`Grade(exposure=0, wb=(r,g,b), sat=1, lift, gamma, gain, contrast=1, vignette=0.18, grain=0.006, bloom=1,
bloom_threshold=0, black=0, fade=1)`. Keep grain <= 0.008 (encoder). `fade` multiplies the final image
(use for fades to black).

## Colour / palette
`engine/color.py`: `hex_lin`, `blackbody(T)`, `PAL` (the BIBLE palette). The messenger gold `#FFD27A`
is reserved (sending / palm light / gold star / photon / shell highlights only).

## Assets
- `assets/sky.py`: `field()` all-sky stars (dirs, rgb); `milky_way()` band in galactic coords
  (g_dirs/g_rgb soft glow + s_dirs/s_rgb stars; rotate with `galactic_basis(center_dir, north_dir)`:
  world_dirs = dirs @ M.T); `spiral_galaxy()` face-on galaxy in XZ plane (pos, rgb, kind).
- `assets/bodies.py`: `fib_sphere`, `sphere_mask`, `RedGiant`, `Planet`.
- `assets/crosshatch.py`: the canonical nine lines (the '#'), shared by the flake and the city;
  the II8->II9 match-cut contract (MATCH_SPAN) is documented there.
- `noise.py`: `fbm(P, octaves, scale, offset)`, `curl(P, scale, t)`, `hash01(i, seed)` (numba, fast).

## Quality bar
Look at your frames (Read the PNG). Pointillism must read as *dots*, never smooth CG. Blacks stay black.
Every shot needs depth (foreground/background separation, parallax or DOF) and a clear focal point.
Render a contact sheet at scale 0.5 for iteration; check a full-res frame before calling a shot done.
Performance target: <= 4 s per full-res frame per process.
