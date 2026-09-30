"""Global constants for the LIGHT-YEARS renderer."""
from pathlib import Path

W, H = 1920, 804          # 2.39:1 scope
FPS = 24
REF_W = 1920              # all pixel quantities in shot code are authored at this width

ROOT = Path(__file__).resolve().parent.parent
BUILD = ROOT / "build"
CACHE = BUILD / "cache"
PREVIEW = BUILD / "preview"
SEGMENTS = BUILD / "segments"

for _d in (BUILD, CACHE, PREVIEW, SEGMENTS):
    _d.mkdir(parents=True, exist_ok=True)


def ffmpeg_exe():
    import shutil
    exe = shutil.which("ffmpeg")
    if exe:
        return exe
    import imageio_ffmpeg
    return imageio_ffmpeg.get_ffmpeg_exe()
