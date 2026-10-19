# Erklärvideo: Warum Milch im Supermarkt immer ganz hinten steht

Vertikales Video (1080x1920, 30 fps) im Bauplan-Look. Alles wird mit Code gezeichnet:
Pillow und numpy für die Frames, ffmpeg für den Schnitt, numpy für die Musik.

## Ausgabe

- `output/supermarkt_milch.mp4`: das fertige Video
- `output/supermarkt_milch.srt`: die Untertitel
- `output/stills/`: ein Standbild pro Szene (`szene_1..7.png`), eine Übersicht und `sync_check.png`
  mit einem Frame aus dem MP4 für jeden Untertitel

## Ausführen

```bash
pip install edge-tts pillow numpy
cd milch_video
python make_video.py --stills   # nur die Standbilder zur Kontrolle
python make_video.py            # Video + SRT
python verify.py                # prüft Format, Pegel und Synchronität
```

## Stimme

- Primär: edge-tts, Stimme `de-DE-ConradNeural`, Tempo `+5%`, jeder Satz einzeln.
- Fallback: Ist `speech.platform.bing.com` nicht erreichbar, nimmt das Skript eine lokale
  Piper-Stimme (`de-thorsten-low`, Modell unter `build/voices/thorsten/` oder per `PIPER_MODEL`).
  Das Tempo ist an Conrad mit +5 % angeglichen. Der Fallback wird beim Start deutlich geloggt.
- Mit `--tts edge` oder `--tts piper` lässt sich das Backend erzwingen.

Die Länge jeder Audiodatei wird gemessen (ffprobe), Stille am Anfang und Ende wird abgeschnitten.
Aus diesen Längen entsteht die Timeline: Szenenwechsel, Animationen und Untertitel richten sich
danach. Untertitel-Grenzen nach Kommas und Doppelpunkten rasten auf echte Sprechpausen ein.
Die Gesamtlänge hängt deshalb von der Stimme ab.

## Musik

Der Ambient-Pad ist selbst erzeugt (`ambient.py`): tiefe Akkorde in d-Moll (Dm9, Bbmaj7, Gm9, Asus4),
ein weicher Sub-Puls bei 68 BPM, Tiefpass, Fade-in und Fade-out. Er liegt 24 dB unter der Stimme
(RMS, in den Sprechpausen gemessen von `verify.py`). Es wird keine fremde Musik verwendet.

## Zahlen

Das Video blendet keine Statistik ein. Der Zähler „ZUSÄTZLICH IM WAGEN“ zählt nur die animierten
Produkte und ist als „SYMBOLDARSTELLUNG“ gekennzeichnet.
