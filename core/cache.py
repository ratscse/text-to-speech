"""Per-line WAV cache: cache/<sha1>.wav. Re-runs only regenerate changed lines."""
from __future__ import annotations

import hashlib
import os
from functools import lru_cache
from pathlib import Path

import numpy as np
import soundfile as sf


@lru_cache(maxsize=256)
def _file_sha1_cached(path: str, mtime_ns: int, size: int) -> str:
    h = hashlib.sha1()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def file_sha1(path: str | os.PathLike) -> str:
    st = os.stat(path)
    return _file_sha1_cached(str(path), st.st_mtime_ns, st.st_size)


def line_key(backend: str, voice_fp: str, speaker: str, emotion_key: str, text: str) -> str:
    """sha1(speaker + emotion + text), plus backend and voice fingerprint so that
    switching models or replacing a reference clip invalidates the entry."""
    h = hashlib.sha1()
    for part in (backend, voice_fp, speaker, emotion_key, text):
        h.update(part.encode("utf-8"))
        h.update(b"\x00")
    return h.hexdigest()


class LineCache:
    def __init__(self, root: str | os.PathLike, enabled: bool = True):
        self.root = Path(root)
        self.enabled = enabled
        if enabled:
            self.root.mkdir(parents=True, exist_ok=True)

    def path(self, key: str) -> Path:
        return self.root / f"{key}.wav"

    def get(self, key: str) -> tuple[np.ndarray, int] | None:
        if not self.enabled:
            return None
        p = self.path(key)
        if not p.is_file():
            return None
        try:
            audio, sr = sf.read(p, dtype="float32", always_2d=False)
        except Exception:
            p.unlink(missing_ok=True)  # corrupt entry: regenerate
            return None
        return audio, sr

    def put(self, key: str, audio: np.ndarray, sr: int) -> None:
        if not self.enabled:
            return
        p = self.path(key)
        tmp = p.with_suffix(".tmp.wav")
        sf.write(tmp, audio, sr, subtype="FLOAT")
        os.replace(tmp, p)
