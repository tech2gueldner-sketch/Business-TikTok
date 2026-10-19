"""Sprachsynthese pro Satz.

Primär: edge-tts mit "de-DE-ConradNeural" (+5 %).
Fallback (nur wenn edge-tts nicht erreichbar ist): lokale Piper-Stimme,
damit die Pipeline auch offline durchläuft. Der Fallback wird laut geloggt.
"""
import asyncio
import hashlib
import os
import ssl
import subprocess
from pathlib import Path

EDGE_VOICE = "de-DE-ConradNeural"
EDGE_RATE = "+5%"
PIPER_LENGTH = 1.4
PIPER_MODEL = Path(os.environ.get(
    "PIPER_MODEL",
    Path(__file__).resolve().parent.parent / "build/voices/thorsten/de-thorsten-low.onnx",
))


def _edge(text, out_path):
    import edge_tts
    import edge_tts.communicate as comm

    # Hinter einem TLS-terminierenden Proxy braucht edge-tts dessen CA-Bundle.
    ca = os.environ.get("SSL_CERT_FILE")
    if ca and os.path.exists(ca):
        comm._SSL_CTX = ssl.create_default_context(cafile=ca)
    proxy = os.environ.get("HTTPS_PROXY") or os.environ.get("https_proxy")

    async def run():
        await edge_tts.Communicate(text, EDGE_VOICE, rate=EDGE_RATE, proxy=proxy).save(str(out_path))

    asyncio.run(run())


def _piper(text, out_path):
    if not PIPER_MODEL.exists():
        raise RuntimeError(f"Piper-Modell fehlt: {PIPER_MODEL}")
    # Die Piper-Stimme spricht von Haus aus deutlich schneller als Conrad;
    # PIPER_LENGTH gleicht das Tempo an Conrad mit +5 % an.
    subprocess.run(
        ["piper", "-m", str(PIPER_MODEL), "-f", str(out_path), "--length-scale", str(PIPER_LENGTH)],
        input=text.encode("utf-8"), check=True, capture_output=True,
    )


BACKENDS = {"edge": (_edge, ".mp3"), "piper": (_piper, ".wav")}


def pick_backend(preferred="auto", probe_dir=None):
    """Liefert den Namen des Backends, das tatsächlich funktioniert."""
    if preferred != "auto":
        return preferred
    probe_dir = Path(probe_dir or ".")
    probe_dir.mkdir(parents=True, exist_ok=True)
    try:
        _edge("Test.", probe_dir / "_probe.mp3")
        return "edge"
    except Exception as exc:  # Netzwerk gesperrt, Dienst down, ...
        print(f"!! edge-tts nicht erreichbar ({type(exc).__name__}: {exc}).")
        print("!! FALLBACK auf lokale Piper-Stimme - für Conrad edge-tts freischalten und neu rendern.")
        return "piper"


def synth(text, cache_dir, backend):
    """Erzeugt (oder holt aus dem Cache) eine Audiodatei für einen Satz."""
    fn, ext = BACKENDS[backend]
    key = hashlib.sha1(f"{backend}|{EDGE_VOICE}|{EDGE_RATE}|{PIPER_LENGTH}|{text}".encode()).hexdigest()[:12]
    out = Path(cache_dir) / f"{backend}_{key}{ext}"
    if not out.exists():
        out.parent.mkdir(parents=True, exist_ok=True)
        fn(text, out)
    return out
