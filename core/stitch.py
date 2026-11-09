"""Audio assembly: resample, trim, loudness-normalize, insert pauses, export."""
from __future__ import annotations

import shutil
import subprocess
from dataclasses import dataclass
from math import gcd
from pathlib import Path

import numpy as np
import soundfile as sf

TARGET_LUFS = -20.0
PEAK_CEILING = 0.97  # ~ -0.26 dBFS


def to_mono_f32(audio: np.ndarray) -> np.ndarray:
    a = np.asarray(audio, dtype=np.float32)
    if a.ndim == 2:
        # (n, ch) from soundfile or (ch, n) from torch: average the short axis
        a = a.mean(axis=1 if a.shape[0] > a.shape[1] else 0)
    return np.ascontiguousarray(a.reshape(-1), dtype=np.float32)


def resample(audio: np.ndarray, sr_from: int, sr_to: int) -> np.ndarray:
    if sr_from == sr_to or audio.size == 0:
        return audio.astype(np.float32, copy=False)
    from scipy.signal import resample_poly

    g = gcd(sr_from, sr_to)
    return resample_poly(audio, sr_to // g, sr_from // g).astype(np.float32)


def trim_silence(audio: np.ndarray, sr: int, thresh_db: float = -45.0, keep_ms: int = 40) -> np.ndarray:
    """Strip leading/trailing near-silence (models often pad unevenly), keep a margin."""
    if audio.size == 0:
        return audio
    win = max(1, int(sr * 0.01))
    n = audio.size // win
    if n == 0:
        return audio
    frames = audio[: n * win].reshape(n, win)
    rms = np.sqrt(np.mean(frames ** 2, axis=1) + 1e-12)
    peak = float(rms.max())
    if peak <= 0:
        return audio[:0]
    thresh = peak * 10 ** (thresh_db / 20)
    voiced = np.nonzero(rms > thresh)[0]
    if voiced.size == 0:
        return audio[:0]
    keep = int(sr * keep_ms / 1000)
    start = max(0, voiced[0] * win - keep)
    end = min(audio.size, (voiced[-1] + 1) * win + keep)
    return audio[start:end]


def fade(audio: np.ndarray, sr: int, ms: int = 8) -> np.ndarray:
    n = min(int(sr * ms / 1000), audio.size // 2)
    if n <= 0:
        return audio
    out = audio.copy()
    ramp = np.linspace(0.0, 1.0, n, dtype=np.float32)
    out[:n] *= ramp
    out[-n:] *= ramp[::-1]
    return out


def rms_db(audio: np.ndarray) -> float:
    return 20 * np.log10(np.sqrt(np.mean(audio ** 2)) + 1e-12)


def normalize_loudness(audio: np.ndarray, sr: int, target_lufs: float = TARGET_LUFS) -> np.ndarray:
    """Integrated loudness via pyloudnorm (ITU-R BS.1770); for clips too short
    for a 400 ms gating block, fall back to RMS (≈ LUFS for speech)."""
    if audio.size == 0 or not np.any(audio):
        return audio
    gain_db: float
    try:
        import pyloudnorm as pyln

        if audio.size >= int(0.4 * sr) + 1:
            lufs = pyln.Meter(sr).integrated_loudness(audio.astype(np.float64))
            gain_db = target_lufs - lufs if np.isfinite(lufs) else target_lufs - rms_db(audio)
        else:
            gain_db = target_lufs - rms_db(audio)
    except ImportError:
        gain_db = target_lufs - rms_db(audio)
    gain_db = float(np.clip(gain_db, -30.0, 30.0))
    out = audio * (10 ** (gain_db / 20))
    peak = float(np.max(np.abs(out)))
    if peak > PEAK_CEILING:  # simple peak limiting by scaling
        out *= PEAK_CEILING / peak
    return out.astype(np.float32)


def silence(ms: int, sr: int) -> np.ndarray:
    return np.zeros(int(sr * ms / 1000), dtype=np.float32)


@dataclass
class Segment:
    audio: np.ndarray  # mono float32 at the scene sample rate
    pause_before_ms: int


def prepare_clip(audio: np.ndarray, sr_in: int, sr_out: int, loudness: bool = True) -> np.ndarray:
    a = resample(to_mono_f32(audio), sr_in, sr_out)
    a = trim_silence(a, sr_out)
    if loudness:
        a = normalize_loudness(a, sr_out)
    return fade(a, sr_out)


def join_chunks(chunks: list[np.ndarray], sr: int, gap_ms: int = 120) -> np.ndarray:
    parts: list[np.ndarray] = []
    for i, c in enumerate(chunks):
        if i:
            parts.append(silence(gap_ms, sr))
        parts.append(c)
    return np.concatenate(parts) if parts else np.zeros(0, np.float32)


def assemble(segments: list[Segment], sr: int, lead_ms: int = 150, tail_ms: int = 300) -> np.ndarray:
    parts = [silence(lead_ms, sr)]
    for i, seg in enumerate(segments):
        if i:
            parts.append(silence(seg.pause_before_ms, sr))
        parts.append(seg.audio)
    parts.append(silence(tail_ms, sr))
    out = np.concatenate(parts).astype(np.float32)
    peak = float(np.max(np.abs(out))) if out.size else 0.0
    if peak > PEAK_CEILING:
        out *= PEAK_CEILING / peak
    return out


def write_wav(path: str | Path, audio: np.ndarray, sr: int) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(path, audio, sr, subtype="PCM_16")
    return path


def ffmpeg_path() -> str | None:
    return shutil.which("ffmpeg")


def wav_to_mp3(wav: str | Path, mp3: str | Path, bitrate: str = "192k") -> Path:
    exe = ffmpeg_path()
    if exe is None:
        raise RuntimeError("MP3 export needs ffmpeg on PATH (https://ffmpeg.org/download.html). The WAV was still written.")
    mp3 = Path(mp3)
    cmd = [exe, "-y", "-loglevel", "error", "-i", str(wav), "-codec:a", "libmp3lame", "-b:a", bitrate, str(mp3)]
    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode != 0:
        raise RuntimeError(f"ffmpeg failed: {res.stderr.strip()}")
    return mp3
