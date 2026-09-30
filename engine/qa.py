"""Motion QA: render consecutive low-res frames and flag pops/flicker.

  python -m engine.qa I6 [--scale 0.25] [--step 1]
Prints per-frame mean |diff| and flags frames whose change is > 4x the local median.
"""
import argparse
import sys

import numpy as np

from . import timeline as TL
from .config import FPS
from .render import render_time


def scan(sid, scale=0.25, step=1, jobs=1):
    f0, f1 = TL.frame_range(sid)
    frames = list(range(f0, f1, step))
    prev = None
    diffs = []
    for f in frames:
        img = render_time(f / FPS, scale, f).astype(np.float32)
        if prev is not None:
            diffs.append(float(np.mean(np.abs(img - prev))))
        prev = img
    d = np.array(diffs)
    if d.size < 3:
        return d, []
    med = np.array([np.median(d[max(0, i - 6):i + 7]) for i in range(len(d))])
    flags = [(frames[i + 1], d[i], med[i]) for i in range(len(d)) if d[i] > 4 * med[i] + 1.0]
    return d, flags


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("sid", nargs="+")
    ap.add_argument("--scale", type=float, default=0.25)
    ap.add_argument("--step", type=int, default=1)
    a = ap.parse_args()
    for sid in a.sid:
        d, flags = scan(sid, a.scale, a.step)
        print(f"{sid}: mean|diff| {d.mean():.2f}  max {d.max():.2f}  flags {len(flags)}")
        for f, v, m in flags[:12]:
            print(f"   frame {f} (t={f / FPS:.2f})  diff {v:.2f}  local median {m:.2f}")
    sys.stdout.flush()


if __name__ == "__main__":
    main()
