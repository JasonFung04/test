"""THE single source of truth for timing (v1.1). Picture and sound both read this file.

Times are seconds from the first frame. `python -m engine.timeline` writes
build/timeline.json for the audio engine.
"""
import json

import numpy as np

from .config import BUILD, FPS

# (shot id, start, end, module)
SHOTS = [
    ("O1", 0.0, 2.0, "opening"), ("O2", 2.0, 7.0, "opening"), ("O3", 7.0, 10.5, "opening"),
    ("O4", 10.5, 17.5, "opening"), ("O5", 17.5, 20.5, "opening"), ("O6", 20.5, 28.0, "opening"),
    ("I1", 28.0, 35.0, "act1"), ("I2", 35.0, 41.0, "act1"), ("I3", 41.0, 45.0, "act1"),
    ("I4", 45.0, 50.0, "act1"), ("I5", 50.0, 54.0, "act1"), ("I6", 54.0, 64.0, "act1"),
    ("I7", 64.0, 71.0, "act1"), ("I8", 71.0, 76.0, "act1"), ("I9", 76.0, 81.0, "act1"),
    ("II1", 81.0, 87.0, "act2"), ("II2", 87.0, 93.0, "act2"), ("II3", 93.0, 96.5, "act2"),
    ("II4", 96.5, 102.5, "act2"), ("II5", 102.5, 107.5, "act2"), ("II6", 107.5, 110.0, "act2"),
    ("II7", 110.0, 113.5, "act2"), ("II8", 113.5, 118.0, "act2"), ("II9", 118.0, 124.0, "act3"),
    ("III1", 124.0, 131.0, "act3"), ("III2a", 131.0, 138.5, "act3_girl"), ("III2b", 138.5, 145.0, "act3_girl"),
    ("III3", 145.0, 155.0, "act3"), ("III4", 155.0, 158.5, "act3_girl"), ("III5", 158.5, 171.5, "act3"),
    ("III6", 171.5, 176.0, "act3"), ("III7a", 176.0, 177.8, "act3_girl"), ("III7b", 177.8, 180.5, "act3_girl"),
    ("III8a", 180.5, 184.5, "climax"), ("III8b", 184.5, 187.5, "climax"), ("III8c", 187.5, 200.5, "climax"),
    ("III10", 200.5, 205.0, "act3_girl"), ("III11", 205.0, 210.0, "act3"), ("III12", 210.0, 217.5, "act3_girl"),
    ("E1", 217.5, 224.0, "epilogue"), ("E2", 224.0, 237.0, "epilogue"), ("E3", 237.0, 244.0, "epilogue"),
    ("BLACK", 244.0, 246.0, "epilogue"), ("CR", 246.0, 264.0, "epilogue"),
]
DURATION = SHOTS[-1][2]
N_FRAMES = int(round(DURATION * FPS))

CARDS = [
    dict(id="C1", t0=29.5, t1=34.5, style="slug_r",
         cn="七万三千年前 · 银河系的另一侧", en="73,000 years ago · The far side of the Milky Way"),
    dict(id="C2", t0=57.0, t1=62.0, cn="我们不知道你在哪里。", en="We do not know where you are."),
    dict(id="C3", t0=65.5, t1=70.0, cn="所以，我们把自己寄往每一个方向。",
         en="So we are sending ourselves in every direction."),
    dict(id="C5", t0=93.3, t1=96.2, style="black", fin=0.6, fout=0.5,
         cn="它出发的那一夜，在地球上——", en="The night it left, on Earth —"),
    dict(id="C6", t0=114.0, t1=117.8, style="museum", drift=False,
         cn="已知最早的绘画 · 约七万三千年前 · 南非布隆伯斯洞穴",
         en="The oldest known drawing · c. 73,000 years ago · Blombos Cave, South Africa"),
    dict(id="C7", t0=125.3, t1=130.8, cn="人类的全部历史，都发生在这束光的途中。",
         en="All of human history happened while this light was on its way."),
    dict(id="C10a", t0=211.0, t1=217.2, y=0.36, cn="他们不知道我们会存在。", en="They did not know we would exist."),
    dict(id="C10b", t0=214.0, t1=217.2, y=0.53, fin=1.0, cn="他们还是寄出了。", en="They sent it anyway."),
]

# ---------------------------------------------------------------- opening typing
PROMPT_CN = "# 写一束光。"
PROMPT_EN = "# write a light."
CREDIT_LINES = [
    ("claude opus 5.5", "编剧 · 导演 · 渲染 · 配乐", "written · directed · rendered · scored"),
    ("gemini", "科学顾问", "science consultant"),
    ("grok", "剧本顾问", "script consultant"),
]
CREDIT_T = [11.0, 13.4, 15.4]
STATS = "points of light 11,407,288"
SENDING = "sending"


def typing_schedule():
    """[(time, char, line)] for every keystroke in the opening. Deterministic human rhythm."""
    rng = np.random.default_rng(42)
    ev = [(4.0, "#", "prompt_cn")]
    t = 7.25
    for ch in PROMPT_CN[1:]:
        t += 0.07 + rng.random() * 0.05 + (0.15 if ch == "。" else 0.0)
        ev.append((round(t, 3), ch, "prompt_cn"))
    t += 0.35
    for ch in PROMPT_EN:
        t += 0.035 + rng.random() * 0.03
        ev.append((round(t, 3), ch, "prompt_en"))
    for k, (name, cn, en) in enumerate(CREDIT_LINES):
        t = CREDIT_T[k]
        for ch in name:
            t += 0.03 + rng.random() * 0.025
            ev.append((round(t, 3), ch, f"credit{k}"))
    t = 18.0
    for ch in STATS:
        t += 0.022 + rng.random() * 0.018
        ev.append((round(t, 3), ch, "stats"))
    t = max(t, 18.9) + 0.3
    for ch in SENDING:
        t += 0.075 + rng.random() * 0.04
        ev.append((round(t, 3), ch, "sending"))
    return ev


# ---------------------------------------------------------------- blackout schedule
# rings of darkness spreading from the substation (~700 m from the girl's roof)
BLACKOUT_OFF = [146.2, 146.7, 147.3, 147.9, 148.5, 149.1, 149.7, 150.3, 150.9, 151.5, 152.1, 152.7, 153.4, 154.2]
POWER_ON = [205.4, 205.8, 206.2, 206.6, 207.0, 207.4, 207.8, 208.1, 208.4, 208.7, 209.0, 209.2, 209.4, 209.6]

EVENTS = {
    "room_tone_in": 1.0, "cursor_blinks": [2.3, 2.83, 3.36], "hash_key": 4.0,
    "credit_lines": CREDIT_T,
    "dissolve_start": 20.5, "swirl": 22.0, "title_tone": 23.5, "title_out": 26.8, "cut_black": 27.0,
    "act1": 28.0, "star_breath_first": 28.5, "star_breath_period": 4.0, "choir_in": 35.0,
    "elder_looks_down": 47.8, "palm_open": 50.3, "palm_tone": 51.5,
    "lift_first": 54.5, "lift_all": 57.0, "elder_gone": 59.0,
    "converge_peak": 71.0, "burst": 71.8, "front_pass": 75.2, "hard_silence": 76.0,
    "nebula": 81.0, "galaxy": 87.0, "black_card": 93.0, "fire_in": 95.0, "cave": 96.5,
    "strokes_parallel": [102.9, 103.6, 104.3, 105.1, 105.8, 106.6],
    "strokes_cross": [110.4, 111.3, 112.3],
    "match_cut": 118.0,
    "city_tilt": 124.0, "rooftop": 131.0,
    "torch_click_fail": [132.0, 132.8], "torch_tap": [133.5, 133.9], "torch_on": 134.2, "torch_off": 143.2,
    "transformer_flash": 146.0, "transformer_thump": 148.0,
    "district_off": BLACKOUT_OFF, "rooftop_lamp_off": 154.2,
    "car_alarm": [156.0, 156.4, 156.8],
    "stars_begin": 158.5, "theme_return": 162.0,
    "crowd": 171.5, "phone_on": 173.2, "phone_off": 174.4,
    "gold_star_on": 178.3, "gold_star_tone": 178.5, "gold_star_off": 180.7,
    "photon_fall": 180.5, "eye": 184.5, "cornea_touch": 186.5, "hand_reflection": 186.7,
    "pullout": 187.5, "arc_start": 188.5, "arc_end": 190.5, "orbit_start": 190.5, "sky_red": 193.0,
    "orbit_end": 198.0, "elder_dissolve": 198.0,
    "blink": 201.6,
    "power_return": 205.0, "district_on": POWER_ON,
    "rooftop_after": 210.0,
    "torch_raise": 217.8, "torch_click_fail_final": 219.5, "torch_tap_final": 220.6, "torch_on_final": 220.9,
    "beam_rise": 224.0, "beam_fades": 232.5, "space": 234.0, "final_chord": 237.0,
    "cut_black_end": 244.0, "credits": 246.0, "end": DURATION,
}


def shot_at(t):
    for sid, a, b, mod in SHOTS:
        if a <= t < b:
            return sid, a, b, mod
    return SHOTS[-1]


def shot(sid):
    for s in SHOTS:
        if s[0] == sid:
            return s
    raise KeyError(sid)


def frame_range(sid):
    _, a, b, _ = shot(sid)
    return int(round(a * FPS)), int(round(b * FPS))


def export_json(path=None):
    path = path or (BUILD / "timeline.json")
    data = dict(fps=FPS, duration=DURATION,
                shots=[dict(id=s, start=a, end=b) for s, a, b, _ in SHOTS],
                cards=[{k: v for k, v in c.items()} for c in CARDS],
                events=EVENTS,
                typing=[dict(t=t, ch=c, line=l) for t, c, l in typing_schedule()])
    path.write_text(json.dumps(data, ensure_ascii=False, indent=1))
    return path


if __name__ == "__main__":
    print(export_json())
