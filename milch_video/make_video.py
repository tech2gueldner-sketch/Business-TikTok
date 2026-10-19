#!/usr/bin/env python3
"""Vertikales Erklärvideo: "Warum Milch im Supermarkt immer ganz hinten steht".

Ablauf:
  1. Stimme pro Satz erzeugen (edge-tts, Fallback Piper), Länge messen.
  2. Timeline aus den gemessenen Satzlängen bauen, Untertitel daraus ableiten.
  3. Ambient-Pad erzeugen und unter die Stimme mischen (-24 dB).
  4. Frames mit Pillow/numpy zeichnen (Glow = Blur-Layer additiv), an ffmpeg pipen.

Aufruf:
  python make_video.py --stills      # nur Standbilder je Szene (zur Kontrolle)
  python make_video.py               # komplettes Video + SRT
"""
import argparse
import itertools
import json
import math
import subprocess
import wave
from multiprocessing import Pool
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

import ambient
import tts

ROOT = Path(__file__).resolve().parent.parent
BUILD = ROOT / "build"
OUT = ROOT / "output"
NAME = "supermarkt_milch"

W, H, FPS = 1080, 1920, 30
SR = ambient.SR

BG = (10, 12, 16)
TEAL = (60, 200, 170)
ORANGE = (230, 110, 60)
WHITE = (236, 242, 240)
GRAY = (112, 126, 132)
HILITE = (90, 225, 195)

FONT_MONO = "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf"
FONT_MONO_B = "/usr/share/fonts/truetype/dejavu/DejaVuSansMono-Bold.ttf"
FONT_SUB = "/usr/share/fonts/opentype/inter/Inter-ExtraBold.otf"
FONT_SEMI = "/usr/share/fonts/opentype/inter/Inter-SemiBold.otf"

CATEGORY = "KAUFPSYCHOLOGIE"
SCENES = [
    dict(status="ANALYSIERT", lines=[
        "Du willst nur Milch kaufen.",
        "Warum steht sie eigentlich immer ganz hinten?"]),
    dict(status="LAUFWEG ERFASST", lines=[
        "Weil du so fast durch den ganzen Laden laufen musst."]),
    dict(status="SICHTKONTAKT", lines=[
        "Auf dem Weg kommst du an hunderten Produkten vorbei, die du gar nicht kaufen wolltest."]),
    dict(status="IMPULSKAUF", lines=[
        "Und genau dabei landet plötzlich mehr im Wagen als geplant."]),
    dict(status="PRAKTISCH", lines=[
        "Ganz ehrlich: Es gibt auch einen praktischen Grund.",
        "Die Kühlung und die Anlieferung liegen meistens hinten."]),
    dict(status="ABGLEICH", lines=[
        "Aber Supermärkte wissen genau, was dieser Weg mit dir macht."]),
    dict(status="TIPP", lines=[
        "Also: Geh mit Zettel rein.",
        "Und folge nicht dem Weg, den man für dich gebaut hat."]),
]
HIGHLIGHT = {"milch", "hinten", "ganzen", "laden", "hunderten", "wagen", "praktischen",
             "kühlung", "anlieferung", "weg", "zettel", "genau"}

LEAD_IN = 0.45      # Stille vor dem ersten Satz
GAP_SENT = 0.35     # Pause zwischen Sätzen einer Szene
GAP_SCENE = 0.90    # Pause zwischen Szenen
SCENE_PRE = 0.30    # Bild wechselt kurz vor dem ersten Satz der Szene
OUTRO = 3.0
TRANS = 0.7         # Dauer der weichen Übergänge zwischen Szenenzuständen
VOICE_RMS = 0.1     # -20 dBFS je Satz
MUSIC_DB = -24.0

# ---------------------------------------------------------------- Grundriss
# Lokale Koordinaten (800 x 1000), "hinten" = oben.
WALLS = [(200, 1000), (800, 1000), (800, 150), (0, 150), (0, 1000), (60, 1000)]
LAGER = [(0, 80), (0, 110), (800, 110), (800, 0), (0, 0), (0, 30)]
PATH = [(130, 1045), (130, 255), (330, 255), (330, 860), (530, 860), (530, 255),
        (700, 255), (700, 214)]
COOLER = (430, 162, 780, 202)
CHECKOUTS = [(470, 905, 510, 965), (570, 905, 610, 965), (670, 905, 710, 965)]


def _shelf_units():
    units = []
    for i in range(6):
        y0 = 250 + i * 620 / 6
        units.append((12, y0, 52, y0 + 620 / 6))
        units.append((748, y0, 788, y0 + 620 / 6))
    for x0 in (200, 400, 600):
        for face in (0, 1):
            for i in range(5):
                units.append((x0 + 30 * face, 310 + i * 100, x0 + 30 * face + 30, 410 + i * 100))
    for i in range(3):
        units.append((80 + i * 320 / 3, 162, 80 + (i + 1) * 320 / 3, 202))
    return units


def _path_samples(pts, n=600):
    seg = [math.dist(a, b) for a, b in zip(pts, pts[1:])]
    total = sum(seg)
    out = []
    for k in range(n + 1):
        out.append(point_at(pts, k / n)[0])
    return np.array(out), total


def point_at(pts, frac):
    seg = [math.dist(a, b) for a, b in zip(pts, pts[1:])]
    target = max(0.0, min(1.0, frac)) * sum(seg)
    acc = 0.0
    for (a, b), l in zip(zip(pts, pts[1:]), seg):
        if acc + l >= target and l > 0:
            t = (target - acc) / l
            d = ((b[0] - a[0]) / l, (b[1] - a[1]) / l)
            return (a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t), d
        acc += l
    a, b = pts[-2], pts[-1]
    l = math.dist(a, b)
    return b, ((b[0] - a[0]) / l, (b[1] - a[1]) / l)


def subpath(pts, f0, f1):
    """Teilstück einer Polylinie zwischen den Längenanteilen f0..f1."""
    f0, f1 = max(0.0, f0), min(1.0, f1)
    if f1 <= f0:
        return []
    seg = [math.dist(a, b) for a, b in zip(pts, pts[1:])]
    total = sum(seg)
    out = [point_at(pts, f0)[0]]
    acc = 0.0
    for b, l in zip(pts[1:], seg):
        acc += l
        if f0 * total < acc < f1 * total:
            out.append(b)
    out.append(point_at(pts, f1)[0])
    return out


PATH_SAMPLES, PATH_LEN = _path_samples(PATH)
SHELVES = []
for _r in _shelf_units():
    _c = np.array([(_r[0] + _r[2]) / 2, (_r[1] + _r[3]) / 2])
    _d = np.hypot(*(PATH_SAMPLES - _c).T)
    _k = int(np.argmin(_d))
    SHELVES.append(dict(rect=_r, u=_k / (len(PATH_SAMPLES) - 1), near=_d[_k] < 115))

# ---------------------------------------------------------------- Helfer


def clamp(x, a=0.0, b=1.0):
    return max(a, min(b, x))


def ease_out(x):
    x = clamp(x)
    return 1 - (1 - x) ** 3


def ease_io(x):
    x = clamp(x)
    return 4 * x ** 3 if x < 0.5 else 1 - (-2 * x + 2) ** 3 / 2


def smooth(x):
    x = clamp(x)
    return x * x * (3 - 2 * x)


def col(c, a):
    return tuple(int(clamp(v * a, 0, 255)) for v in c)


def rgba(c, a):
    return (c[0], c[1], c[2], int(255 * clamp(a)))


_FONTS = {}


def font(path, size):
    key = (path, int(size))
    if key not in _FONTS:
        _FONTS[key] = ImageFont.truetype(path, int(size))
    return _FONTS[key]


class Plan:
    """Abbildung lokaler Grundriss-Koordinaten auf den Bildschirm."""

    def __init__(self, s, oy):
        self.s, self.ox, self.oy = s, 540 - 400 * s, oy

    def p(self, x, y):
        return (self.ox + x * self.s, self.oy + y * self.s)

    def pts(self, pts):
        return [self.p(*q) for q in pts]

    def r(self, rect):
        x0, y0 = self.p(rect[0], rect[1])
        x1, y1 = self.p(rect[2], rect[3])
        return (x0, y0, x1, y1)

    def w(self, px):
        return max(1, round(px * self.s))


def line(d, pts, fill, width):
    if len(pts) >= 2:
        d.line(pts, fill=fill, width=width, joint="curve")


def arrow(d, a, b, fill, width, head=18):
    line(d, [a, b], fill, width)
    ang = math.atan2(b[1] - a[1], b[0] - a[0])
    pts = [b,
           (b[0] - head * math.cos(ang - 0.45), b[1] - head * math.sin(ang - 0.45)),
           (b[0] - head * math.cos(ang + 0.45), b[1] - head * math.sin(ang + 0.45))]
    d.polygon(pts, fill=fill)


def dashed(d, a, b, fill, width, dash=12, gap=10, phase=0.0):
    l = math.dist(a, b)
    if l <= 0:
        return
    ux, uy = (b[0] - a[0]) / l, (b[1] - a[1]) / l
    t = -(phase % (dash + gap))
    while t < l:
        t0, t1 = max(0, t), min(l, t + dash)
        if t1 > t0:
            d.line([(a[0] + ux * t0, a[1] + uy * t0), (a[0] + ux * t1, a[1] + uy * t1)], fill=fill, width=width)
        t += dash + gap


def circle_pts(cx, cy, r, n=20):
    return [(cx + r * math.cos(2 * math.pi * k / n), cy + r * math.sin(2 * math.pi * k / n)) for k in range(n + 1)]


# ---------------------------------------------------------------- Timeline


def ffprobe_duration(path):
    out = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration",
                          "-of", "csv=p=0", str(path)], capture_output=True, text=True, check=True)
    return float(out.stdout.strip())


def load_mono(path):
    raw = subprocess.run(["ffmpeg", "-v", "error", "-i", str(path), "-f", "f32le", "-ac", "1",
                          "-ar", str(SR), "-"], capture_output=True, check=True).stdout
    return np.frombuffer(raw, np.float32).copy()


def envelope(x, win=0.01):
    n = int(SR * win)
    m = len(x) // n
    return np.sqrt(np.mean(x[:m * n].reshape(m, n) ** 2, axis=1) + 1e-12), win


def trim_silence(x):
    env, win = envelope(x)
    on = np.where(env > env.max() * 10 ** (-40 / 20))[0]
    pad = int(0.03 * SR)
    a = max(0, int(on[0] * win * SR) - pad)
    b = min(len(x), int((on[-1] + 1) * win * SR) + pad)
    return x[a:b]


def inner_pauses(x):
    """Sprechpausen im Satz (z. B. an Kommas), als (start, ende) in Sekunden."""
    env, win = envelope(x)
    quiet = env < env.max() * 10 ** (-32 / 20)
    pauses, k = [], 0
    while k < len(quiet):
        if quiet[k]:
            j = k
            while j < len(quiet) and quiet[j]:
                j += 1
            if k > 5 and j < len(quiet) - 5 and (j - k) >= 9:
                pauses.append((k * win, j * win))
            k = j
        else:
            k += 1
    return pauses


def chunk_words(text, maxlen=29):
    """Teilt einen Satz in gut lesbare Untertitel-Häppchen.

    Erst an Satzzeichen trennen, dann jeden Teilsatz so gleichmäßig wie
    möglich aufteilen (keine einzelnen Restwörter).
    """
    clauses, cur = [], []
    for w in text.split():
        cur.append(w)
        if w[-1] in ",:":
            clauses.append(cur)
            cur = []
    if cur:
        clauses.append(cur)
    merged = []
    for c in clauses:  # sehr kurze Teilsätze ("Also:") an den nächsten hängen
        if merged and len(" ".join(merged[-1])) < 8:
            merged[-1] = merged[-1] + c
        else:
            merged.append(c)

    chunks = []
    for words in merged:
        k = math.ceil(len(" ".join(words)) / maxlen)
        if k <= 1 or len(words) < 2:
            chunks.append(" ".join(words))
            continue
        k = min(k, len(words))
        best = min(itertools.combinations(range(1, len(words)), k - 1),
                   key=lambda cuts: max(len(" ".join(words[a:b]))
                                        for a, b in zip((0,) + cuts, cuts + (len(words),))))
        chunks += [" ".join(words[a:b]) for a, b in zip((0,) + best, best + (len(words),))]
    return chunks


def make_cues(sent):
    chunks = chunk_words(sent["text"])
    weights = [len(c) + (5 if c[-1] in ",:" else 0) for c in chunks]
    dur = sent["end"] - sent["start"]
    bounds, acc = [0.0], 0.0
    for wgt in weights[:-1]:
        acc += wgt
        bounds.append(dur * acc / sum(weights))
    # Grenzen nach Satzzeichen auf echte Sprechpausen einrasten
    for i in range(1, len(bounds)):
        if chunks[i - 1][-1] in ",:" and sent["pauses"]:
            best = min(sent["pauses"], key=lambda p: abs(p[1] - bounds[i]))
            if abs(best[1] - bounds[i]) < 0.6:
                bounds[i] = best[1] - 0.04
    bounds.append(dur + 0.18)
    return [dict(start=sent["start"] + bounds[i], end=sent["start"] + bounds[i + 1], text=chunks[i])
            for i in range(len(chunks))]


def build_timeline(backend):
    tts_dir = BUILD / "tts"
    sentences, scenes, cursor = [], [], LEAD_IN
    for si, sc in enumerate(SCENES):
        if si > 0:
            cursor += GAP_SCENE - GAP_SENT
        scenes.append(dict(start=0.0 if si == 0 else cursor - SCENE_PRE, status=sc["status"]))
        for text in sc["lines"]:
            path = tts.synth(text, tts_dir, backend)
            raw_dur = ffprobe_duration(path)
            x = trim_silence(load_mono(path))
            x *= VOICE_RMS / np.sqrt(np.mean(x ** 2))
            dur = len(x) / SR
            print(f"  [{raw_dur:5.2f}s Datei | {dur:5.2f}s Sprache] {text}")
            sentences.append(dict(text=text, scene=si, start=cursor, end=cursor + dur,
                                  file=str(path), pauses=inner_pauses(x), samples=x))
            cursor += dur + GAP_SENT
    total = cursor - GAP_SENT + OUTRO
    for i, sc in enumerate(scenes):
        sc["end"] = scenes[i + 1]["start"] if i + 1 < len(scenes) else total
    cues = [c for s in sentences for c in make_cues(s)]
    return dict(scenes=scenes, sentences=sentences, cues=cues, total=total, backend=backend)


def srt_time(t):
    ms = int(round(t * 1000))
    return f"{ms // 3600000:02d}:{ms // 60000 % 60:02d}:{ms // 1000 % 60:02d},{ms % 1000:03d}"


def write_srt(cues, path):
    with open(path, "w", encoding="utf-8") as f:
        for i, c in enumerate(cues, 1):
            f.write(f"{i}\n{srt_time(c['start'])} --> {srt_time(c['end'])}\n{c['text']}\n\n")


def write_audio(tl, n_frames, path):
    n = int(round(n_frames / FPS * SR))
    voice = np.zeros(n, np.float32)
    for s in tl["sentences"]:
        i0 = int(round(s["start"] * SR))
        voice[i0:i0 + len(s["samples"])] += s["samples"][:n - i0]
    pad = ambient.make_pad(n / SR)[:n]
    mix = voice[:, None] + pad * (VOICE_RMS * 10 ** (MUSIC_DB / 20))
    mix *= 0.89 / np.abs(mix).max()  # Spitze auf -1 dBFS
    pcm = (np.clip(mix, -1, 1) * 32767).astype(np.int16)
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(2)
        wf.setsampwidth(2)
        wf.setframerate(SR)
        wf.writeframes(pcm.tobytes())


# ---------------------------------------------------------------- Zustände

# Werte je Szene (1..7); dazwischen wird weich überblendet.
TRACKS = dict(
    plan_a=[1, 1, 1, 0, 1, 1, 0],
    plan_s=[1, 1, 1, 0.94, 1, 0.70, 1],
    plan_oy=[270, 270, 270, 270, 270, 250, 270],
    path_i=[1, 1, 0.8, 1, 0.55, 1.8, 0.6],
    hl=[0, 0, 1, 0, 0, 0, 0],
    cart=[0, 0, 0, 1, 0, 0, 0],
    logi=[0, 0, 0, 0, 1, 0, 0],
    lager_keep=[1, 1, 1, 1, 1, 1, 0],
    cards=[0, 0, 0, 0, 0, 1, 0],
    list=[0, 0, 0, 0, 0, 0, 1],
    hint=[1, 0, 0, 0, 0, 0, 0],
)


def blend(tl, T, values):
    v = values[0]
    for k in range(1, len(values)):
        b = tl["scenes"][k]["start"]
        v += (values[k] - values[k - 1]) * smooth((T - (b - TRANS / 2)) / TRANS)
    return v


def sent(tl, scene, idx=0):
    return [s for s in tl["sentences"] if s["scene"] == scene][idx]


# ---------------------------------------------------------------- Zeichnen

_BG = None


def background():
    global _BG
    if _BG is None:
        img = Image.new("RGB", (W, H), BG)
        d = ImageDraw.Draw(img)
        for x in range(0, W + 1, 30):
            d.line([(x, 0), (x, H)], fill=(14, 17, 22) if x % 150 else (19, 23, 30), width=1)
        for y in range(0, H + 1, 30):
            d.line([(0, y), (W, y)], fill=(14, 17, 22) if y % 150 else (19, 23, 30), width=1)
        arr = np.asarray(img, np.float32)
        yy, xx = np.mgrid[0:H, 0:W]
        vig = 1 - 0.55 * (((xx - W / 2) / (W * 0.75)) ** 2 + ((yy - H / 2) / (H * 0.7)) ** 2)
        base = np.array(BG, np.float32)
        _BG = base + (arr - base) * np.clip(vig, 0.25, 1)[..., None]
    return _BG


def draw_header(g, u, tl, T):
    g.rectangle([60, 78, 66, 104], fill=col(TEAL, 1))
    g.text((80, 91), CATEGORY, font=font(FONT_MONO_B, 26), fill=col(TEAL, 0.95), anchor="lm")
    u.text((80, 122), "SUPERMARKT // GRUNDRISS", font=font(FONT_MONO, 18), fill=rgba(GRAY, 1), anchor="lm")
    blink = 0.55 + 0.45 * math.sin(2 * math.pi * T * 1.1)
    for i, sc in enumerate(tl["scenes"]):
        a = blend(tl, T, [1.0 if k == i else 0.0 for k in range(len(SCENES))])
        if a < 0.01:
            continue
        f = font(FONT_MONO_B, 26)
        g.text((1020, 91), sc["status"], font=f, fill=col(ORANGE, a), anchor="rm")
        x = 1020 - f.getlength(sc["status"]) - 22
        g.ellipse([x - 7, 84, x + 7, 98], fill=col(ORANGE, a * blink))
    g.line([(60, 148), (1020, 148)], fill=col(TEAL, 0.28), width=1)
    for x in range(60, 1021, 40):
        g.line([(x, 148), (x, 154 if x % 200 else 160)], fill=col(TEAL, 0.28), width=1)


def draw_plan(g, u, tl, T, pl, a):
    if a < 0.01:
        return
    build = ease_io(T / 1.8)
    walls = subpath(WALLS, 0, build)
    line(g, pl.pts(walls), col(TEAL, 0.9 * a), pl.w(3))
    # Maßstabsticks am rechten Rand
    for k, y in enumerate(range(150, 1001, 50)):
        if k / 17 < build:
            x0, yy = pl.p(812, y)
            g.line([(x0, yy), (x0 + pl.w(8 if y % 250 else 16), yy)], fill=col(TEAL, 0.3 * a), width=1)

    hl = blend(tl, T, TRACKS["hl"])
    s3 = tl["scenes"][2]
    uc = ease_io((T - s3["start"] - 0.1) / (s3["end"] - s3["start"] - 0.5))
    for i, sh in enumerate(SHELVES):
        ai = a * clamp(build * 2.2 - 0.6 - i * 0.012)
        if ai <= 0:
            continue
        x0, y0, x1, y1 = pl.r(sh["rect"])
        inten = 0.0
        if sh["near"] and hl > 0:
            flash = math.exp(-((uc - sh["u"]) / 0.035) ** 2)
            seen = 0.3 * smooth((uc - sh["u"]) / 0.03)
            inten = max(flash, seen) * hl
        if inten > 0.02:
            g.rectangle([x0, y0, x1, y1], fill=col(ORANGE, 0.28 * inten * ai))
            tall = (y1 - y0) > (x1 - x0)
            for k in range(1, 5):
                if tall:
                    yy = y0 + (y1 - y0) * k / 5
                    g.line([(x0 + 4, yy), (x1 - 4, yy)], fill=col(ORANGE, 0.8 * inten * ai), width=1)
                else:
                    xx = x0 + (x1 - x0) * k / 5
                    g.line([(xx, y0 + 4), (xx, y1 - 4)], fill=col(ORANGE, 0.8 * inten * ai), width=1)
        base = col(TEAL, 0.42 * ai)
        edge = tuple(max(b, c) for b, c in zip(base, col(ORANGE, inten * ai)))
        g.rectangle([x0, y0, x1, y1], outline=edge, width=pl.w(2))

    ac = a * smooth((build - 0.6) / 0.4)
    for r in CHECKOUTS:
        g.rectangle(pl.r(r), outline=col(TEAL, 0.42 * ac), width=pl.w(2))
    if ac > 0.01:
        u.text(pl.p(590, 980), "KASSE", font=font(FONT_MONO, 18 * pl.s), fill=rgba(GRAY, ac), anchor="mm")
        u.text(pl.p(88, 1045), "EINGANG", font=font(FONT_MONO, 20 * pl.s), fill=rgba(GRAY, ac), anchor="rm")
        u.text(pl.p(800, 1072), "DRAUFSICHT", font=font(FONT_MONO, 16 * pl.s), fill=rgba(GRAY, 0.7 * ac), anchor="rm")

    # Kühlregal MILCH
    s0 = sent(tl, 0, 0)
    milk = a * ease_out((T - (s0["start"] + 0.45 * (s0["end"] - s0["start"]))) / 0.7)
    if milk > 0.01:
        pulse = 0.78 + 0.22 * math.sin(2 * math.pi * T / 1.6)
        x0, y0, x1, y1 = pl.r(COOLER)
        g.rectangle([x0, y0, x1, y1], fill=col(TEAL, 0.22 * milk * pulse))
        g.rectangle([x0, y0, x1, y1], outline=col(TEAL, milk), width=pl.w(3))
        for k in range(6):  # Milchtüten von oben
            cx = 465 + k * 55
            r = pl.r((cx - 14, 170, cx + 14, 194))
            g.rectangle(r, outline=col(WHITE, 0.75 * milk), width=1)
            g.line([pl.p(cx - 14, 182), pl.p(cx + 14, 182)], fill=col(WHITE, 0.5 * milk), width=1)
        g.text(pl.p(545, 232), "MILCH", font=font(FONT_MONO_B, 30 * pl.s), fill=col(WHITE, milk), anchor="mm")

    # "GANZ HINTEN?"-Hinweis
    s01 = sent(tl, 0, 1)
    hint = a * blend(tl, T, TRACKS["hint"]) * ease_out((T - s01["start"] - 0.9) / 0.5)
    if hint > 0.01:
        cx, cy = pl.p(605, 92)
        g.text((cx, cy), "GANZ HINTEN?", font=font(FONT_MONO_B, 34), fill=col(ORANGE, hint), anchor="mm")
        for k in range(3):
            yy = cy + 32 + k * 9
            g.line([(cx - 10, yy), (cx, yy + 8), (cx + 10, yy)], fill=col(ORANGE, hint * (0.4 + 0.3 * k)), width=2)


def draw_path(g, u, tl, T, pl, a):
    s1 = sent(tl, 1)
    p = ease_io((T - s1["start"] + 0.1) / ((s1["end"] - s1["start"]) * 0.95))
    inten = blend(tl, T, TRACKS["path_i"])
    s0 = sent(tl, 0, 0)
    du = a * ease_out((T - s0["start"]) / 0.4)
    if p > 0 and a > 0.01:
        pts = pl.pts(subpath(PATH, 0, p))
        if inten > 1:
            line(g, pts, col(TEAL, 0.35 * (inten - 1) * a), pl.w(16))
        line(g, pts, col(TEAL, min(1.0, inten) * a), pl.w(5 if inten > 1.3 else 4))
        # Richtungs-Chevrons
        n = int(PATH_LEN / 95)
        for k in range(1, n):
            f = k / n
            if f > p - 0.02:
                break
            (x, y), (dx, dy) = point_at(PATH, f)
            cx, cy = pl.p(x, y)
            s = 9 * pl.s
            g.line([(cx - dx * s - dy * s, cy - dy * s + dx * s), (cx + dx * s, cy + dy * s),
                    (cx - dx * s + dy * s, cy - dy * s - dx * s)], fill=col(BG, 1), width=pl.w(3))
        # Lichtimpulse auf der Linie, wenn sie stark leuchtet
        if inten > 1.05:
            for k in range(3):
                f = (T * 0.22 + k / 3) % 1
                seg = pl.pts(subpath(PATH, f - 0.035, f))
                line(g, seg, col(WHITE, (inten - 1) * a), pl.w(6))
    # Lichtimpuls in Szene 3 (Sichtkontakt entlang des Wegs)
    hl = blend(tl, T, TRACKS["hl"])
    s3 = tl["scenes"][2]
    uc = ease_io((T - s3["start"] - 0.1) / (s3["end"] - s3["start"] - 0.5))
    if hl > 0.01 and 0 < uc < 1:
        seg = pl.pts(subpath(PATH, uc - 0.06, uc))
        line(g, seg, col(WHITE, hl * a), pl.w(6))
        (x, y), (dx, dy) = point_at(PATH, uc)
        cx, cy = pl.p(x, y)
        for sgn in (-1, 1):  # Blickfeld links und rechts
            line(g, [(cx, cy), (cx - dy * sgn * 70 * pl.s + dx * 30 * pl.s, cy + dx * sgn * 70 * pl.s + dy * 30 * pl.s)],
                 col(ORANGE, 0.7 * hl * a), 2)
        g.ellipse([cx - 9, cy - 9, cx + 9, cy + 9], fill=col(WHITE, hl * a))
    # "DU"-Markierung
    if du > 0.01:
        (x, y), _ = point_at(PATH, p)
        cx, cy = pl.p(x, y)
        r = 13 * pl.s
        ring = 1 + 0.6 * ((T * 0.9) % 1)
        g.ellipse([cx - r, cy - r, cx + r, cy + r], fill=col(WHITE, du))
        rr = r * 1.8 * ring
        g.ellipse([cx - rr, cy - rr, cx + rr, cy + rr], outline=col(TEAL, du * (1.6 - ring) * 1.2), width=2)
        g.text((cx + 26 * pl.s, cy), "DU", font=font(FONT_MONO_B, 24 * pl.s), fill=col(WHITE, du), anchor="lm")


def snowflake(g, cx, cy, r, fill, w):
    for k in range(6):
        ang = k * math.pi / 3
        ex, ey = cx + r * math.cos(ang), cy + r * math.sin(ang)
        g.line([(cx, cy), (ex, ey)], fill=fill, width=w)
        for s in (-1, 1):
            mx, my = cx + 0.6 * r * math.cos(ang), cy + 0.6 * r * math.sin(ang)
            g.line([(mx, my), (mx + 0.3 * r * math.cos(ang + s * 0.8), my + 0.3 * r * math.sin(ang + s * 0.8))],
                   fill=fill, width=w)


def draw_lager(g, u, tl, T, pl, a):
    s5 = tl["scenes"][4]
    vis = a * blend(tl, T, TRACKS["lager_keep"]) * ease_io((T - s5["start"]) / 0.9)
    if vis < 0.01:
        return
    line(g, pl.pts(subpath(LAGER, 0, vis)), col(TEAL, 0.9 * vis), pl.w(3))
    u.text(pl.p(0, -24), "LAGER", font=font(FONT_MONO_B, 20 * pl.s), fill=rgba(GRAY, vis), anchor="lm")

    sa, sb = sent(tl, 4, 0), sent(tl, 4, 1)
    kuehl = vis * ease_out((T - sb["start"] - 0.1) / 0.5)
    if kuehl > 0.01:
        cx, cy = pl.p(500, 55)
        snowflake(g, cx, cy, 24 * pl.s, col(TEAL, kuehl), pl.w(2))
        g.text(pl.p(540, 55), "KÜHLUNG", font=font(FONT_MONO_B, 24 * pl.s), fill=col(TEAL, kuehl), anchor="lm")
        for x in (480, 605, 730):
            dashed(g, pl.p(x, 112), pl.p(x, 160), col(TEAL, 0.8 * kuehl), pl.w(2), dash=8 * pl.s, gap=6 * pl.s,
                   phase=-T * 40 * pl.s)
    anl = vis * ease_out((T - (sb["start"] + 0.45 * (sb["end"] - sb["start"]))) / 0.5)
    if anl > 0.01:
        c = col(TEAL, anl)
        g.rectangle([*pl.p(70, 32), *pl.p(150, 78)], outline=c, width=pl.w(2))
        g.rectangle([*pl.p(150, 46), *pl.p(182, 78)], outline=c, width=pl.w(2))
        for wx in (92, 165):
            x, y = pl.p(wx, 84)
            r = 8 * pl.s
            g.ellipse([x - r, y - r, x + r, y + r], outline=c, width=pl.w(2))
        slide = (1 - anl) * 40
        arrow(g, (pl.p(-110, 55)[0] - slide, pl.p(0, 55)[1]), pl.p(55, 55), col(ORANGE, anl), pl.w(3), head=16 * pl.s)
        g.text(pl.p(200, 55), "ANLIEFERUNG", font=font(FONT_MONO_B, 24 * pl.s), fill=col(TEAL, anl), anchor="lm")

    logi = vis * blend(tl, T, TRACKS["logi"]) * ease_out((T - sa["start"] - 0.1) / 0.5)
    if logi > 0.01:
        cx, top = pl.p(400, -80)
        f = font(FONT_MONO_B, 30)
        tw = f.getlength("LOGISTIK")
        drop = (1 - logi) * -20
        box = [cx - tw / 2 - 18, top - 26 + drop, cx + tw / 2 + 18, top + 26 + drop]
        g.rectangle(box, outline=col(ORANGE, logi), width=2)
        g.text((cx, top + drop), "LOGISTIK", font=f, fill=col(ORANGE, logi), anchor="mm")
        grow = ease_out((T - sa["start"] - 0.35) / 0.5)
        if grow > 0:
            y0 = box[3] + 4
            y1 = y0 + (pl.p(0, -6)[1] - y0) * grow
            arrow(g, (cx, y0), (cx, y1), col(ORANGE, logi), 3, head=16)


ICONS = ["chips", "bar", "bottle", "can", "box", "candy"]


def icon_polys(kind):
    if kind == "chips":
        top = [(-0.36 + 0.72 * k / 8, -0.5 + (0.06 if k % 2 else 0)) for k in range(9)]
        return [top + [(0.42, 0.5), (-0.42, 0.5), top[0]], circle_pts(0, 0.05, 0.16, 14)]
    if kind == "bar":
        return [[(-0.5, -0.24), (0.5, -0.24), (0.5, 0.24), (-0.5, 0.24), (-0.5, -0.24)],
                [(-0.17, -0.24), (-0.17, 0.24)], [(0.17, -0.24), (0.17, 0.24)], [(-0.5, 0), (0.5, 0)]]
    if kind == "bottle":
        return [[(-0.11, -0.5), (0.11, -0.5), (0.11, -0.3), (0.24, -0.14), (0.24, 0.5), (-0.24, 0.5),
                 (-0.24, -0.14), (-0.11, -0.3), (-0.11, -0.5)], [(-0.24, 0.05), (0.24, 0.05)],
                [(-0.24, 0.28), (0.24, 0.28)]]
    if kind == "can":
        return [[(-0.26, -0.42), (0.26, -0.42), (0.26, 0.42), (-0.26, 0.42), (-0.26, -0.42)],
                [(-0.26, -0.3), (0.26, -0.3)], [(-0.26, 0.3), (0.26, 0.3)]]
    if kind == "box":
        return [[(-0.38, -0.45), (0.38, -0.45), (0.38, 0.45), (-0.38, 0.45), (-0.38, -0.45)],
                circle_pts(0, -0.05, 0.2, 14), [(-0.25, 0.3), (0.25, 0.3)]]
    return [circle_pts(0, 0, 0.22, 14), [(-0.22, 0), (-0.48, -0.18), (-0.48, 0.18), (-0.22, 0)],
            [(0.22, 0), (0.48, -0.18), (0.48, 0.18), (0.22, 0)]]


def draw_icon(g, kind, cx, cy, size, rot, fill, width):
    c, s = math.cos(rot), math.sin(rot)
    for poly in icon_polys(kind):
        pts = [(cx + size * (x * c - y * s), cy + size * (x * s + y * c)) for x, y in poly]
        line(g, pts, fill, width)


LANDING = [(430, 1090), (520, 1095), (610, 1090), (700, 1095), (480, 1015), (650, 1012)]


def draw_cart(g, u, tl, T):
    a = blend(tl, T, TRACKS["cart"])
    if a < 0.01:
        return
    sc = tl["scenes"][3]
    t, dur = T - sc["start"], sc["end"] - sc["start"]
    oy = (1 - a) * 50

    # Zähler
    n = len(ICONS)
    lands = [0.55 + (dur - 1.5) * (i / (n - 1)) ** 0.85 + 0.5 for i in range(n)]
    val = sum(ease_out((t - L) / 0.3) for L in lands)
    shown = int(val + 0.5)
    pop = max([math.exp(-(t - L) * 9) for L in lands if t >= L] + [0])
    g.rectangle([160, 420 + oy, 920, 720 + oy], outline=col(TEAL, 0.55 * a), width=2)
    for x, y in ((160, 420), (920, 420), (160, 720), (920, 720)):
        g.line([(x - 10, y + oy), (x + 10, y + oy)], fill=col(TEAL, a), width=2)
        g.line([(x, y - 10 + oy), (x, y + 10 + oy)], fill=col(TEAL, a), width=2)
    g.text((190, 460 + oy), "ZUSÄTZLICH IM WAGEN", font=font(FONT_MONO_B, 28), fill=col(TEAL, a), anchor="lm")
    g.text((540, 590 + oy), f"+{shown}", font=font(FONT_SUB, 150 * (1 + 0.12 * pop)), fill=col(WHITE, a),
           anchor="mm")
    g.rectangle([200, 682 + oy, 880, 692 + oy], outline=col(TEAL, 0.4 * a), width=1)
    g.rectangle([200, 682 + oy, 200 + 680 * val / n, 692 + oy], fill=col(ORANGE, a))
    u.text((920, 744 + oy), "SYMBOLDARSTELLUNG", font=font(FONT_MONO, 18), fill=rgba(GRAY, a), anchor="rm")

    # Produkte fallen in den Wagen
    for i, kind in enumerate(ICONS):
        L = lands[i]
        start = L - 0.5
        if t < start:
            continue
        f = clamp((t - start) / 0.5)
        tx, ty = LANDING[i]
        x0 = tx + (-60 if i % 2 else 60)
        y = 790 + (ty - 790) * f * f
        x = x0 + (tx - x0) * ease_out(f)
        bounce = -14 * math.sin(math.pi * clamp((t - L) / 0.22)) if t > L else 0
        rot = (0.5 if i % 2 else -0.5) * (1 - f) + (0.12 if i % 2 else -0.1)
        fade = a * smooth(f / 0.25)
        draw_icon(g, kind, x, y + bounce + oy, 82, rot, col(ORANGE if i % 2 else TEAL, fade), 3)

    # Einkaufswagen (vorne über den Produkten)
    c = col(TEAL, a)
    basket = [(330, 900), (790, 900), (745, 1150), (378, 1150), (330, 900)]
    line(g, [(x, y + oy) for x, y in basket], c, 4)
    for k in range(1, 8):
        xt = 330 + 460 * k / 8
        xb = 378 + 367 * k / 8
        g.line([(xt, 900 + oy), (xb, 1150 + oy)], fill=col(TEAL, 0.35 * a), width=2)
    for k in range(1, 3):
        yy = 900 + 250 * k / 3
        xl = 330 + 48 * k / 3
        xr = 790 - 45 * k / 3
        g.line([(xl, yy + oy), (xr, yy + oy)], fill=col(TEAL, 0.35 * a), width=2)
    line(g, [(330, 900 + oy), (280, 850 + oy), (226, 850 + oy)], c, 4)
    line(g, [(378, 1150 + oy), (400, 1205 + oy), (770, 1205 + oy)], c, 4)
    for wx in (430, 720):
        g.ellipse([wx - 22, 1228 + oy, wx + 22, 1272 + oy], outline=c, width=4)


def draw_cards(g, u, tl, T):
    k = blend(tl, T, TRACKS["cards"])
    if k < 0.01:
        return
    s6 = tl["scenes"][5]
    cards = [
        (70, TEAL, "01 // LOGISTIK", ["Kühlung hinten", "Anlieferung hinten"], 0.05),
        (555, ORANGE, "02 // PSYCHOLOGIE", ["Langer Laufweg", "mehr im Wagen"], 0.45),
    ]
    for x0, c, title, rows, delay in cards:
        a = k * ease_out((T - s6["start"] - delay) / 0.5)
        if a < 0.01:
            continue
        dy = (1 - a) * 30
        y0, x1, y1 = 1065 + dy, x0 + 455, 1330 + dy
        g.rectangle([x0, y0, x1, y1], outline=col(c, 0.75 * a), width=2)
        g.rectangle([x0, y0, x1, y0 + 6], fill=col(c, a))
        g.text((x0 + 26, y0 + 50), title, font=font(FONT_MONO_B, 26), fill=col(c, a), anchor="lm")
        g.line([(x0 + 26, y0 + 82), (x1 - 26, y0 + 82)], fill=col(c, 0.35 * a), width=1)
        for i, r in enumerate(rows):
            yy = y0 + 135 + i * 72
            g.rectangle([x0 + 26, yy - 6, x0 + 38, yy + 6], fill=col(c, a))
            u.text((x0 + 56, yy), r, font=font(FONT_SEMI, 38), fill=rgba(WHITE, a), anchor="lm")


def draw_list(g, u, tl, T):
    k = blend(tl, T, TRACKS["list"])
    if k < 0.01:
        return
    s7 = tl["scenes"][6]
    sa, sb = sent(tl, 6, 0), sent(tl, 6, 1)
    a = k * ease_out((T - s7["start"]) / 0.6)
    dy = (1 - a) * 40
    x0, y0, x1, y1 = 210, 420 + dy, 870, 1230 + dy
    paper = [(x0, y0), (x1 - 40, y0), (x1, y0 + 40), (x1, y1), (x0, y1), (x0, y0)]
    line(g, paper, col(WHITE, 0.85 * a), 3)
    line(g, [(x1 - 40, y0), (x1 - 40, y0 + 40), (x1, y0 + 40)], col(WHITE, 0.6 * a), 2)
    tag = font(FONT_MONO_B, 30)
    g.rectangle([x0, y0 - 74, x0 + tag.getlength("TIPP") + 36, y0 - 22], outline=col(ORANGE, a), width=2)
    g.text((x0 + 18, y0 - 48), "TIPP", font=tag, fill=col(ORANGE, a), anchor="lm")
    g.text((x0 + 40, y0 + 62), "EINKAUFSZETTEL", font=font(FONT_MONO_B, 30), fill=col(TEAL, a), anchor="lm")
    g.line([(x0 + 40, y0 + 100), (x1 - 40, y0 + 100)], fill=col(TEAL, 0.5 * a), width=2)
    for i in range(4):
        yy = y0 + 250 + i * 140
        g.line([(x0 + 40, yy), (x1 - 40, yy)], fill=col(TEAL, 0.12 * a), width=1)

    items = ["Milch", "Brot", "Butter"]
    for i, it in enumerate(items):
        yy = y0 + 200 + i * 140
        ia = a * ease_out((T - s7["start"] - 0.15 - 0.12 * i) / 0.4)
        g.rectangle([x0 + 50, yy - 26, x0 + 102, yy + 26], outline=col(TEAL, ia), width=3)
        u.text((x0 + 135, yy), it, font=font(FONT_SUB, 54), fill=rgba(WHITE, ia), anchor="lm")
        chk = clamp((T - (sa["start"] + 0.35 + 0.3 * i)) / 0.3)
        if chk > 0:
            tick = [(x0 + 58, yy - 2), (x0 + 74, yy + 16), (x0 + 108, yy - 32)]
            line(g, subpath(tick, 0, chk), col(TEAL, ia), 6)
    yy = y0 + 200 + 3 * 140
    sa_ = a * ease_out((T - sb["start"]) / 0.4)
    if sa_ > 0.01:
        g.rectangle([x0 + 50, yy - 26, x0 + 102, yy + 26], outline=col(ORANGE, sa_), width=3)
        u.text((x0 + 135, yy), "Spontankäufe", font=font(FONT_SUB, 54), fill=rgba(GRAY, sa_), anchor="lm")
        st = clamp((T - sb["start"] - 0.4) / 0.5)
        if st > 0:
            tw = font(FONT_SUB, 54).getlength("Spontankäufe")
            g.line([(x0 + 125, yy + 2), (x0 + 125 + (tw + 20) * ease_out(st), yy + 2)], fill=col(ORANGE, sa_), width=5)
            line(g, subpath([(x0 + 60, yy - 16), (x0 + 92, yy + 16)], 0, st), col(ORANGE, sa_), 4)
            line(g, subpath([(x0 + 92, yy - 16), (x0 + 60, yy + 16)], 0, clamp(st * 2 - 1)), col(ORANGE, sa_), 4)


def draw_subtitle(u, tl, T):
    for c in tl["cues"]:
        if not (c["start"] <= T < c["end"]):
            continue
        a = ease_out((T - c["start"]) / 0.12) * clamp((c["end"] - T) / 0.08)
        f = font(FONT_SUB, 66)
        words = c["text"].split()
        lines = [words]
        if f.getlength(c["text"]) > 940:
            best = min(range(1, len(words)),
                       key=lambda i: abs(f.getlength(" ".join(words[:i])) - f.getlength(" ".join(words[i:]))))
            lines = [words[:best], words[best:]]
        y = 1490 - (len(lines) - 1) * 40 + (1 - a) * 14
        sp = f.getlength(" ")
        for ln in lines:
            total = sum(f.getlength(w) for w in ln) + sp * (len(ln) - 1)
            x = 540 - total / 2
            for w in ln:
                key = w.strip(",.:?!").lower()
                fill = HILITE if key in HIGHLIGHT else WHITE
                u.text((x + 3, y + 5), w, font=f, fill=(0, 0, 0, int(150 * a)), anchor="lm")
                u.text((x, y), w, font=f, fill=rgba(fill, a), anchor="lm", stroke_width=4,
                       stroke_fill=(5, 6, 8, int(230 * a)))
                x += f.getlength(w) + sp
            y += 80


def render(tl, T):
    g_img = Image.new("RGB", (W, H), (0, 0, 0))
    u_img = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    g, u = ImageDraw.Draw(g_img), ImageDraw.Draw(u_img)

    # dezente Scanlinie
    sy = (T * 140) % (H + 600) - 300
    g.line([(0, sy), (W, sy)], fill=col(TEAL, 0.07), width=2)

    pl = Plan(blend(tl, T, TRACKS["plan_s"]), blend(tl, T, TRACKS["plan_oy"]))
    pa = blend(tl, T, TRACKS["plan_a"])
    draw_lager(g, u, tl, T, pl, pa)
    draw_plan(g, u, tl, T, pl, pa)
    draw_path(g, u, tl, T, pl, pa)
    draw_cart(g, u, tl, T)
    draw_cards(g, u, tl, T)
    draw_list(g, u, tl, T)
    draw_header(g, u, tl, T)
    draw_subtitle(u, tl, T)

    lines = np.asarray(g_img, np.float32)
    glow1 = np.asarray(g_img.reduce(4).filter(ImageFilter.GaussianBlur(4)).resize((W, H), Image.BILINEAR), np.float32)
    glow2 = np.asarray(g_img.reduce(10).filter(ImageFilter.GaussianBlur(5)).resize((W, H), Image.BILINEAR), np.float32)
    out = background() + lines + 1.25 * glow1 + 0.9 * glow2
    ui = np.asarray(u_img, np.float32)
    al = ui[..., 3:4] / 255
    out = out * (1 - al) + ui[..., :3] * al
    fade = clamp(T / 0.25) * clamp((tl["total"] - T) / 0.8)
    return np.clip(out * fade, 0, 255).astype(np.uint8)


# ---------------------------------------------------------------- Ausgabe

_TL = None


def _init(tl):
    global _TL
    _TL = tl


def _frame(i):
    return render(_TL, i / FPS).tobytes()


def strip_samples(tl):
    return {**tl, "sentences": [{k: v for k, v in s.items() if k != "samples"} for s in tl["sentences"]]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tts", default="auto", choices=["auto", "edge", "piper"])
    ap.add_argument("--stills", action="store_true", help="nur Standbilder je Szene rendern")
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()

    OUT.mkdir(exist_ok=True)
    BUILD.mkdir(exist_ok=True)
    backend = tts.pick_backend(args.tts, BUILD / "tts")
    print(f"Stimme: {backend}")
    tl = build_timeline(backend)
    light = strip_samples(tl)
    (BUILD / "timeline.json").write_text(json.dumps(light, indent=2, ensure_ascii=False))
    n_frames = int(math.ceil(tl["total"] * FPS))
    print(f"Gesamtlänge: {tl['total']:.2f}s, {n_frames} Frames")

    stills = OUT / "stills"
    stills.mkdir(exist_ok=True)
    thumbs = []
    for i, sc in enumerate(tl["scenes"]):
        T = sc["start"] + 0.8 * (sc["end"] - sc["start"])
        img = Image.fromarray(render(light, T))
        img.save(stills / f"szene_{i + 1}.png")
        thumbs.append(img.resize((W // 4, H // 4), Image.LANCZOS))
    sheet = Image.new("RGB", (W // 4 * 4, H // 4 * 2), BG)
    for i, th in enumerate(thumbs):
        sheet.paste(th, ((i % 4) * (W // 4), (i // 4) * (H // 4)))
    sheet.save(stills / "uebersicht.png")
    print(f"Standbilder: {stills}")
    if args.stills:
        return

    write_srt(tl["cues"], OUT / f"{NAME}.srt")
    audio = BUILD / "mix.wav"
    write_audio(tl, n_frames, audio)

    cmd = ["ffmpeg", "-y", "-v", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}",
           "-r", str(FPS), "-i", "-", "-i", str(audio), "-map", "0:v", "-map", "1:a",
           "-c:v", "libx264", "-preset", "medium", "-crf", "18", "-pix_fmt", "yuv420p",
           "-c:a", "aac", "-b:a", "192k", "-movflags", "+faststart", str(OUT / f"{NAME}.mp4")]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    with Pool(args.workers, initializer=_init, initargs=(light,)) as pool:
        for i, buf in enumerate(pool.imap(_frame, range(n_frames), chunksize=4)):
            proc.stdin.write(buf)
            if i % 150 == 0:
                print(f"  Frame {i}/{n_frames}")
    proc.stdin.close()
    if proc.wait() != 0:
        raise SystemExit("ffmpeg fehlgeschlagen")
    print(f"Fertig: {OUT / (NAME + '.mp4')}")


if __name__ == "__main__":
    main()
