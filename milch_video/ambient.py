"""Selbst erzeugter, dunkler Ambient-Pad (keine fremde Musik)."""
import numpy as np

SR = 48000
BPM = 68

# Tiefe, dunkle Akkorde in d-Moll (MIDI-Noten): Dm9 - Bbmaj7 - Gm9 - Asus4
CHORDS = [
    [38, 45, 50, 53, 57, 64],
    [34, 41, 46, 50, 57, 60],
    [31, 43, 50, 55, 58, 62],
    [33, 40, 45, 50, 52, 59],
]
CHORD_LEN = 6.0
XFADE = 2.0


def midi_hz(n):
    return 440.0 * 2 ** ((n - 69) / 12)


def lowpass(x, cutoff, order=4):
    """Butterworth-artiger Tiefpass im Frequenzbereich (ganzer Puffer)."""
    spec = np.fft.rfft(x, axis=0)
    f = np.fft.rfftfreq(x.shape[0], 1 / SR)
    gain = 1 / np.sqrt(1 + (f / cutoff) ** (2 * order))
    return np.fft.irfft(spec * (gain[:, None] if x.ndim == 2 else gain), n=x.shape[0], axis=0)


def _chord_env(t, start, end):
    """Weiches Ein- und Ausblenden (raised cosine) über XFADE Sekunden."""
    a = np.clip((t - (start - XFADE / 2)) / XFADE, 0, 1)
    b = np.clip(((end + XFADE / 2) - t) / XFADE, 0, 1)
    return (0.5 - 0.5 * np.cos(np.pi * a)) * (0.5 - 0.5 * np.cos(np.pi * b))


def make_pad(duration, fade_in=3.0, fade_out=3.5, seed=7):
    """Stereo-Pad (N, 2), auf RMS 1 normiert."""
    rng = np.random.default_rng(seed)
    n = int(duration * SR)
    t = np.arange(n) / SR
    out = np.zeros((n, 2))

    k = 0
    while k * CHORD_LEN < duration + XFADE:
        start, end = k * CHORD_LEN, (k + 1) * CHORD_LEN
        env = _chord_env(t, start, end)
        idx = env > 0
        tt = t[idx]
        for j, note in enumerate(CHORDS[k % len(CHORDS)]):
            f = midi_hz(note)
            amp = 1.0 / (1 + 0.35 * j)  # tiefe Töne tragen mehr
            for ch, det in enumerate((-0.0021, 0.0017)):  # leichte Verstimmung = Breite
                ph = rng.uniform(0, 2 * np.pi)
                ff = f * (1 + det)
                s = np.sin(2 * np.pi * ff * tt + ph)
                s += 0.22 * np.sin(2 * np.pi * 2 * ff * tt + ph * 1.3)
                s += 0.06 * np.sin(2 * np.pi * 3 * ff * tt + ph * 0.7)
                out[idx, ch] += amp * env[idx] * s
        k += 1

    # Langsame "Atmung" + leichter Puls auf dem Beat
    beat = 60.0 / BPM
    phase = (t % beat) / beat
    pulse = np.exp(-phase * 4.0)
    out *= (0.82 + 0.06 * np.sin(2 * np.pi * t / 11.0) + 0.12 * pulse)[:, None]

    # Weicher Sub-Puls auf dem Grundton des jeweiligen Akkords
    roots = np.array([midi_hz(CHORDS[int(x // CHORD_LEN) % len(CHORDS)][0]) for x in t[::SR // 10]])
    root_f = np.repeat(roots, SR // 10)[:n]
    sub_phase = 2 * np.pi * np.cumsum(root_f) / SR
    beat_t = t % beat
    sub = np.sin(sub_phase) * np.exp(-beat_t / 0.28) * (1 - np.exp(-beat_t / 0.02))
    out += 0.55 * sub[:, None]

    # Ein Hauch gefiltertes Rauschen ("Raumluft")
    out += 0.05 * lowpass(rng.standard_normal((n, 2)), 700, order=2)

    out = lowpass(out, 950, order=3)

    fades = np.ones(n)
    fi, fo = int(fade_in * SR), int(fade_out * SR)
    fades[:fi] = np.linspace(0, 1, fi) ** 2
    fades[-fo:] = np.linspace(1, 0, fo) ** 2
    out *= fades[:, None]

    rms = np.sqrt(np.mean(out[int(fade_in * SR):-fo] ** 2))
    return (out / rms).astype(np.float32)
