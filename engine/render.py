"""Frame driver.

  python -m engine.render sheet I1 [--n 6] [--scale 0.5]    contact sheet of a shot
  python -m engine.render frame 57.5 [--scale 1]            one frame at a global time
  python -m engine.render mp4 I1 [--scale 0.5] [--jobs 4]   quick preview clip of one shot
  python -m engine.render shot I1 [--jobs 4]                final-quality PNG frames of a shot
  python -m engine.render all [--jobs 4] [--from O1]         every shot
  python -m engine.render encode [--audio build/audio/master.wav]
"""
import argparse
import importlib
import os
import subprocess
import sys
import time
from multiprocessing import get_context

import cv2
import numpy as np

from . import timeline as TL
from .config import BUILD, FPS, H as FULL_H, PREVIEW, W as FULL_W, ffmpeg_exe
from .post import Grade, apply_grade, to_uint8
from .raster import Renderer
from .text import apply_cards

FRAMES = BUILD / "frames"
_MODS = {}


def _mod(name):
    if name not in _MODS:
        _MODS[name] = importlib.import_module(f"engine.shots.{name}")
    return _MODS[name]


# shots rendered with temporal super-sampling (180-degree shutter): (sub-samples)
MOTION_BLUR = {"I2": 3, "I6": 3, "I8": 4, "II1": 3, "E2": 4, "O6": 3, "III8a": 3, "III11": 2, "III3": 2}
SHUTTER = 0.5


def render_time(tg, scale=1.0, frame_idx=None):
    W, H = int(round(FULL_W * scale)), int(round(FULL_H * scale))
    W -= W % 2
    H -= H % 2
    sid, a, b, mod = TL.shot_at(tg)
    n_sub = MOTION_BLUR.get(sid, 1) if scale >= 0.49 else 1
    acc, grade = None, None
    for k in range(n_sub):
        dt = ((k + 0.5) / n_sub - 0.5) * SHUTTER / FPS if n_sub > 1 else 0.0
        t = min(max(tg + dt, a), b - 1e-4)
        R = Renderer(W, H)
        res = _mod(mod).render(sid, t - a, t, R, W, H)
        h, g = res if isinstance(res, tuple) else (res, Grade())
        acc = h if acc is None else acc + h
        grade = grade or g
    hdr = acc / np.float32(n_sub)
    fi = frame_idx if frame_idx is not None else int(round(tg * FPS))
    overlays = []
    post_fn = getattr(_mod(mod), "overlay", None)
    if post_fn is not None:
        overlays.append(lambda y, sid=sid, tl=tg - a: post_fn(sid, tl, tg, y))
    overlays.append(lambda y: apply_cards(y, TL.CARDS, tg))
    y = apply_grade(hdr, grade, fi, overlays=overlays)
    return to_uint8(y, fi)


def _save(img, path):
    cv2.imwrite(str(path), cv2.cvtColor(img, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_PNG_COMPRESSION, 1])


def _work(args):
    f, scale, outdir = args
    path = outdir / f"{f:05d}.png"
    img = render_time(f / FPS, scale, f)
    _save(img, path)
    return f


def _pool(jobs):
    return get_context("fork").Pool(jobs, maxtasksperchild=400)


def render_frames(frames, scale, outdir, jobs):
    outdir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    if jobs <= 1:
        for f in frames:
            _work((f, scale, outdir))
    else:
        with _pool(jobs) as p:
            for i, _ in enumerate(p.imap_unordered(_work, [(f, scale, outdir) for f in frames], chunksize=2)):
                if (i + 1) % 48 == 0:
                    el = time.time() - t0
                    print(f"  {i + 1}/{len(frames)} frames  {el / (i + 1):.2f}s/frame", flush=True)
    print(f"rendered {len(frames)} frames in {time.time() - t0:.1f}s", flush=True)


def contact_sheet(sid, n=6, scale=0.5, times=None):
    _, a, b, _ = TL.shot(sid)
    ts = times or list(np.linspace(a, b - 1.0 / FPS, n))
    imgs = [render_time(t, scale) for t in ts]
    for im, t in zip(imgs, ts):
        cv2.putText(im, f"{sid} {t:6.2f}s", (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (120, 255, 120), 1, cv2.LINE_AA)
    cols = 2
    rows = [np.concatenate(imgs[i:i + cols] + [np.zeros_like(imgs[0])] * (cols - len(imgs[i:i + cols])), axis=1)
            for i in range(0, len(imgs), cols)]
    sheet = np.concatenate(rows, axis=0)
    path = PREVIEW / f"{sid}.png"
    _save(sheet, path)
    return path


def encode(src_dir, out, audio=None, crf=16, fps=FPS, start=0, frames=None, scope=True):
    cmd = [ffmpeg_exe(), "-y", "-loglevel", "error", "-framerate", str(fps), "-start_number", str(start),
           "-i", str(src_dir / "%05d.png")]
    if frames:
        cmd += ["-frames:v", str(frames)]
    if audio:
        cmd += ["-i", str(audio)]
    vf = [] if scope else ["pad=1920:1080:0:138:black"]
    if vf:
        cmd += ["-vf", ",".join(vf)]
    cmd += ["-c:v", "libx264", "-preset", "slow", "-crf", str(crf), "-pix_fmt", "yuv420p",
            "-profile:v", "high", "-tune", "film", "-movflags", "+faststart", "-color_primaries", "bt709",
            "-color_trc", "bt709", "-colorspace", "bt709"]
    if audio:
        cmd += ["-c:a", "aac", "-b:a", "320k", "-shortest"]
    cmd += [str(out)]
    subprocess.run(cmd, check=True)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd")
    ap.add_argument("arg", nargs="?")
    ap.add_argument("--n", type=int, default=6)
    ap.add_argument("--scale", type=float, default=None)
    ap.add_argument("--jobs", type=int, default=os.cpu_count())
    ap.add_argument("--audio", default=None)
    ap.add_argument("--every", type=int, default=1)
    ap.add_argument("--from", dest="from_", default=None)
    ap.add_argument("--to", default=None)
    a = ap.parse_args()

    if a.cmd == "sheet":
        print(contact_sheet(a.arg, a.n, a.scale or 0.5))
    elif a.cmd == "frame":
        t = float(a.arg)
        img = render_time(t, a.scale or 1.0)
        path = PREVIEW / f"t{t:07.2f}.png"
        _save(img, path)
        print(path)
    elif a.cmd == "mp4":
        f0, f1 = TL.frame_range(a.arg)
        scale = a.scale or 0.5
        outdir = BUILD / "tmp" / a.arg
        frames = list(range(f0, f1, a.every))
        render_frames(frames, scale, outdir, a.jobs)
        out = PREVIEW / f"{a.arg}.mp4"
        # renumber via glob-free encode: use start_number
        encode(outdir, out, crf=20, fps=FPS / a.every if a.every > 1 else FPS, start=f0,
               frames=len(frames) if a.every == 1 else None)
        print(out)
    elif a.cmd == "shot":
        f0, f1 = TL.frame_range(a.arg)
        render_frames(list(range(f0, f1)), a.scale or 1.0, FRAMES, a.jobs)
    elif a.cmd == "all":
        ids = [s[0] for s in TL.SHOTS]
        i0 = ids.index(a.from_) if a.from_ else 0
        i1 = ids.index(a.to) + 1 if a.to else len(ids)
        frames = []
        for sid in ids[i0:i1]:
            f0, f1 = TL.frame_range(sid)
            frames += list(range(f0, f1))
        render_frames(frames, a.scale or 1.0, FRAMES, a.jobs)
    elif a.cmd == "encode":
        out = BUILD / "LIGHT-YEARS.mp4"
        encode(FRAMES, out, audio=a.audio, frames=TL.N_FRAMES)
        print(out)
    else:
        print(__doc__)
        sys.exit(1)


if __name__ == "__main__":
    main()
