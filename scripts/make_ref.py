"""Turn any audio/video file into a clean voice reference clip.

  python scripts/make_ref.py input.mp3 voices/rina.wav
  python scripts/make_ref.py talk.mp4 voices/arif.wav --start 12.5 --max 12
  python scripts/make_ref.py in.m4a voices/narr.wav --text "Exact words spoken in the clip."

Steps: decode (ffmpeg, any format) -> mono 24 kHz -> 70 Hz high-pass ->
trim leading/trailing silence -> squeeze long internal pauses -> pick the
most speech-dense window of --min..--max seconds (default 6-12 s) ->
loudness-normalize to -20 LUFS -> 16-bit WAV. `--text` writes voices/x.txt,
the transcript that Qwen3-TTS uses for higher-quality cloning.
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.stitch import fade, normalize_loudness, trim_silence  # noqa: E402

SR = 24000


def decode(path: Path, start: float | None, duration: float | None) -> np.ndarray:
    exe = shutil.which("ffmpeg")
    if exe is None:
        if path.suffix.lower() in (".wav", ".flac", ".ogg"):
            audio, sr = sf.read(path, dtype="float32", always_2d=True)
            from core.stitch import resample

            audio = resample(audio.mean(axis=1), sr, SR)
            s = int((start or 0) * SR)
            return audio[s: s + int(duration * SR)] if duration else audio[s:]
        sys.exit("ffmpeg not found on PATH; it is needed for non-WAV input")
    cmd = [exe, "-v", "error"]
    if start:
        cmd += ["-ss", str(start)]
    cmd += ["-i", str(path)]
    if duration:
        cmd += ["-t", str(duration)]
    cmd += ["-vn", "-ac", "1", "-ar", str(SR), "-af", "highpass=f=70", "-f", "f32le", "-"]
    res = subprocess.run(cmd, capture_output=True)
    if res.returncode != 0:
        sys.exit(f"ffmpeg failed: {res.stderr.decode(errors='replace').strip()}")
    return np.frombuffer(res.stdout, dtype=np.float32).copy()


def frame_rms(audio: np.ndarray, win: int) -> np.ndarray:
    n = audio.size // win
    return np.sqrt(np.mean(audio[: n * win].reshape(n, win) ** 2, axis=1) + 1e-12)


def squeeze_pauses(audio: np.ndarray, max_pause_s: float = 0.6, thresh_db: float = -40) -> np.ndarray:
    """Shorten internal silences longer than max_pause_s (keeps rhythm natural)."""
    win = int(SR * 0.02)
    rms = frame_rms(audio, win)
    if rms.size == 0:
        return audio
    silent = rms < rms.max() * 10 ** (thresh_db / 20)
    keep_frames = int(max_pause_s / 0.02)
    out, run = [], 0
    for i, s in enumerate(silent):
        run = run + 1 if s else 0
        if run <= keep_frames:
            out.append(audio[i * win:(i + 1) * win])
    return np.concatenate(out) if out else audio


def best_window(audio: np.ndarray, min_s: float, max_s: float) -> np.ndarray:
    if audio.size <= int(max_s * SR):
        return audio
    win = int(SR * 0.02)
    rms = frame_rms(audio, win)
    voiced = (rms > rms.max() * 10 ** (-35 / 20)).astype(np.float32)
    span = int(max_s / 0.02)
    score = np.convolve(voiced, np.ones(span), mode="valid")
    i = int(np.argmax(score))
    seg = audio[i * win:(i + span) * win]
    # end on a pause if possible (don't cut mid-word): search the last 2 s
    tail = rms[i + span - int(2 / 0.02): i + span]
    if tail.size:
        cut = int(np.argmin(tail)) + (span - tail.size)
        if cut * 0.02 >= min_s:
            seg = seg[: cut * win]
    return seg


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("input")
    ap.add_argument("output")
    ap.add_argument("--start", type=float, help="seconds to skip at the start of the input")
    ap.add_argument("--duration", type=float, help="only read this many seconds of input")
    ap.add_argument("--min", type=float, default=6.0, help="minimum clip length (s)")
    ap.add_argument("--max", type=float, default=12.0, help="maximum clip length (s), <=15 recommended")
    ap.add_argument("--text", help="transcript of the clip, saved next to it as .txt")
    a = ap.parse_args()

    src, dst = Path(a.input), Path(a.output)
    if not src.is_file():
        sys.exit(f"input not found: {src}")
    audio = decode(src, a.start, a.duration)
    if audio.size == 0:
        sys.exit("no audio decoded")
    audio = trim_silence(audio, SR, thresh_db=-40, keep_ms=80)
    audio = squeeze_pauses(audio)
    audio = best_window(audio, a.min, a.max)
    audio = trim_silence(audio, SR, thresh_db=-40, keep_ms=80)
    dur = audio.size / SR
    if dur < 3:
        sys.exit(f"only {dur:.1f}s of speech found; need at least 3 s (5-15 s ideal)")
    audio = fade(normalize_loudness(audio, SR), SR, ms=15)
    dst.parent.mkdir(parents=True, exist_ok=True)
    sf.write(dst, audio, SR, subtype="PCM_16")
    if a.text:
        dst.with_suffix(".txt").write_text(a.text.strip() + "\n", encoding="utf-8")
    peak = float(np.max(np.abs(audio)))
    print(f"wrote {dst}  ({dur:.1f}s, 24 kHz mono, peak {20 * np.log10(peak + 1e-9):.1f} dBFS)")
    if dur < a.min:
        print(f"warning: clip is shorter than {a.min}s; cloning quality may suffer", file=sys.stderr)
    if a.text is None:
        print("tip: add --text \"<exact transcript>\" for better English (Qwen) cloning", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
