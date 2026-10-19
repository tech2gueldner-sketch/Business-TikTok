#!/usr/bin/env python3
"""Prüft das fertige Video: Streams, Pegel und Synchronität Stimme/Untertitel/Bild."""
import json
import re
import subprocess
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
MP4 = ROOT / "output/supermarkt_milch.mp4"
SRT = ROOT / "output/supermarkt_milch.srt"
TL = json.loads((ROOT / "build/timeline.json").read_text())
SR = 48000


def probe():
    out = subprocess.run(["ffprobe", "-v", "error", "-show_entries",
                          "stream=codec_type,codec_name,width,height,r_frame_rate,duration,sample_rate,channels",
                          "-of", "json", str(MP4)], capture_output=True, text=True, check=True)
    return json.loads(out.stdout)["streams"]


def parse_srt():
    cues = []
    for block in SRT.read_text(encoding="utf-8").strip().split("\n\n"):
        lines = block.splitlines()
        a, b = re.findall(r"(\d+):(\d+):(\d+),(\d+)", lines[1])
        to_s = lambda t: int(t[0]) * 3600 + int(t[1]) * 60 + int(t[2]) + int(t[3]) / 1000
        cues.append((to_s(a), to_s(b), " ".join(lines[2:])))
    return cues


def audio_env(win=0.02):
    # Stereo auswerten: ein Mono-Downmix würde den breiten (dekorrelierten) Pad um ~3 dB zu leise messen.
    raw = subprocess.run(["ffmpeg", "-v", "error", "-i", str(MP4), "-f", "f32le", "-ac", "2", "-ar", str(SR), "-"],
                         capture_output=True, check=True).stdout
    x = np.frombuffer(raw, np.float32).reshape(-1, 2)
    n = int(SR * win)
    m = len(x) // n
    return np.sqrt(np.mean(x[:m * n].reshape(m, n * 2) ** 2, axis=1) + 1e-12), win


def main():
    ok = True
    print("== Streams")
    for s in probe():
        print("  ", s)
    v = [s for s in probe() if s["codec_type"] == "video"][0]
    a = [s for s in probe() if s["codec_type"] == "audio"][0]
    if (v["width"], v["height"], v["r_frame_rate"]) != (1080, 1920, "30/1"):
        ok = False
        print("!! falsches Videoformat")
    if abs(float(v["duration"]) - float(a["duration"])) > 0.1:
        ok = False
        print("!! Audio- und Videolänge weichen ab")

    print("== Pegel")
    env, win = audio_env()
    db = 20 * np.log10(env)
    t = np.arange(len(env)) * win
    speech = np.zeros(len(env), bool)
    for s in TL["sentences"]:
        speech |= (t >= s["start"] + 0.1) & (t <= s["end"] - 0.1)
    music_only = np.zeros(len(env), bool)
    for s0, s1 in zip(TL["sentences"], TL["sentences"][1:]):
        music_only |= (t >= s0["end"] + 0.08) & (t <= s1["start"] - 0.08)
    music_only &= (t > 3.0) & (t < TL["total"] - 3.5)  # ohne Fade-in/-out der Musik
    sp = 10 * np.log10(np.mean(env[speech] ** 2))
    mu = 10 * np.log10(np.mean(env[music_only] ** 2))
    print(f"   Stimme (RMS): {sp:6.1f} dBFS   Musik allein (RMS): {mu:6.1f} dBFS   Abstand: {sp - mu:5.1f} dB")

    print("== Stimme vs. Timeline (Sprachbeginn im fertigen MP4)")
    thr = mu + 12  # deutlich über dem Musikbett
    worst = 0.0
    sents = TL["sentences"]
    for i, s in enumerate(sents):
        nxt = sents[i + 1]["start"] - 0.03 if i + 1 < len(sents) else s["end"] + 0.4
        idx = np.where((t >= s["start"] - 0.3) & (t <= s["end"]) & (db > thr))[0]
        onset = t[idx[0]] if len(idx) else float("nan")
        idx2 = np.where((t >= s["start"]) & (t <= nxt) & (db > thr))[0]
        offset = t[idx2[-1]] + win if len(idx2) else float("nan")
        d0, d1 = onset - s["start"], offset - s["end"]
        # Leise Auslaute liegen unter der Schwelle, daher zählt beim Ende nur ein Überhang.
        worst = max(worst, abs(d0), d1)
        print(f"   {s['start']:6.2f}s  Beginn {d0:+.2f}s  Ende {d1:+.2f}s  {s['text'][:45]}")
    if worst > 0.1:
        ok = False
        print(f"!! Abweichung Stimme/Timeline bis {worst:.2f}s")

    print("== Untertitel vs. Stimme")
    cues = parse_srt()
    for c0, c1, text in cues:
        inside = [s for s in TL["sentences"] if s["start"] - 0.05 <= c0 and c1 <= s["end"] + 0.25]
        voiced = np.mean(db[(t >= c0) & (t < c1)] > thr)
        flag = "" if inside and voiced > 0.4 else "  !!"
        if flag:
            ok = False
        print(f"   {c0:6.2f}-{c1:6.2f}s  Stimme aktiv {voiced * 100:3.0f}%  {text}{flag}")
    covered = np.zeros(len(env), bool)
    for c0, c1, _ in cues:
        covered |= (t >= c0) & (t < c1)
    cov = np.mean(covered[speech])
    print(f"   Sprachanteil mit Untertitel: {cov * 100:.1f}%")
    if cov < 0.98:
        ok = False

    print("== Kontrollbilder aus dem MP4")
    shots = [(c0 + c1) / 2 for c0, c1, _ in cues]
    out = ROOT / "output/stills/sync_check.png"
    thumbs = []
    for ts in shots:
        raw = subprocess.run(["ffmpeg", "-v", "error", "-ss", f"{ts:.3f}", "-i", str(MP4), "-frames:v", "1",
                              "-f", "rawvideo", "-pix_fmt", "rgb24", "-"], capture_output=True, check=True).stdout
        thumbs.append(Image.frombytes("RGB", (1080, 1920), raw).resize((216, 384)))
    cols = 7
    rows = (len(thumbs) + cols - 1) // cols
    sheet = Image.new("RGB", (216 * cols, 384 * rows))
    for i, th in enumerate(thumbs):
        sheet.paste(th, ((i % cols) * 216, (i // cols) * 384))
    sheet.save(out)
    print(f"   {out}")
    print("== ERGEBNIS:", "OK" if ok else "PROBLEME GEFUNDEN")


if __name__ == "__main__":
    main()
