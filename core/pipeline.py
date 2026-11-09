"""Script -> audio. Cache first; only backends with cache misses are loaded,
one at a time (keeps RAM low on CPU machines)."""
from __future__ import annotations

import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np

from core import emotion as emo
from core.cache import LineCache, line_key
from core.script import Line, Script, chunk_text
from core.stitch import Segment, assemble, join_chunks, normalize_loudness, prepare_clip, wav_to_mp3, write_wav


def log(msg: str) -> None:
    print(msg, file=sys.stderr, flush=True)


@dataclass
class RenderResult:
    wav: Path
    mp3: Path | None
    duration_s: float
    generated: int
    cached: int


BackendFactory = Callable[[str], Any]  # backend_id -> TTSBackend


def pause_for(prev: Line | None, line: Line, pauses: dict[str, int]) -> int:
    if prev is None:
        return 0
    if line.pause_before_ms is not None:
        return line.pause_before_ms
    if line.paragraph:
        return pauses["paragraph"]
    return pauses["same_speaker"] if prev.speaker == line.speaker else pauses["turn"]


def render(
    script: Script,
    backend_for_lang: dict[str, str],
    factory: BackendFactory,
    cache: LineCache,
    *,
    max_chars: int = 200,
    loudness: bool = True,
    mp3: bool = False,
    output: Path | None = None,
) -> RenderResult:
    sr = script.sample_rate
    n = len(script.lines)

    # ---- pass 1: resolve keys, collect cache hits -----------------------
    plans: list[tuple[Line, str, str, emo.EmotionSpec]] = []
    audio_by_line: dict[int, np.ndarray] = {}
    misses: dict[str, list[int]] = {}
    for ln in script.lines:
        bid = backend_for_lang[ln.lang]
        spk = script.speakers[ln.voice or ln.speaker]
        e = emo.resolve(ln.emotion, ln.exaggeration, ln.instruct)
        key = line_key(bid, spk.fingerprint(), spk.name, e.key(), ln.text)
        plans.append((ln, bid, key, e))
        hit = cache.get(key)
        if hit is not None:
            a, hit_sr = hit
            audio_by_line[ln.index] = a if hit_sr == sr else prepare_clip(a, hit_sr, sr, loudness=False)
        else:
            misses.setdefault(bid, []).append(ln.index)

    cached = len(audio_by_line)
    if cached:
        log(f"cache: {cached}/{n} line(s) reused")

    # ---- pass 2: per backend, load lazily and generate misses -----------
    generated = 0
    for bid, idxs in misses.items():
        t0 = time.time()
        log(f"loading backend '{bid}' for {len(idxs)} line(s)...")
        backend = factory(bid)
        try:
            backend.ensure_loaded()
            log(f"  '{bid}' ready on {backend.device} ({time.time() - t0:.1f}s)")
            needed: dict[str, dict[str, emo.EmotionSpec]] = {}
            for i in idxs:
                e = plans[i][3]
                needed.setdefault(script.lines[i].voice or script.lines[i].speaker, {})[e.key()] = e
            backend.prepare_voices([script.speakers[s] for s in needed],
                                   {s: list(es.values()) for s, es in needed.items()})
            for i in idxs:
                ln, _, key, e = plans[i]
                spk = script.speakers[ln.voice or ln.speaker]
                t1 = time.time()
                text = backend.normalize_text(ln.text)
                chunks = chunk_text(text, max_chars) or [text]
                pieces = []
                for c in chunks:
                    raw = backend.synth(c, backend.voice(spk), e)
                    pieces.append(prepare_clip(raw, backend.sample_rate, sr, loudness=False))
                audio = join_chunks(pieces, sr)
                cache.put(key, audio, sr)
                audio_by_line[i] = audio
                generated += 1
                dur = audio.size / sr
                log(f"  [{i + 1}/{n}] {ln.voice or ln.speaker} ({ln.lang}, {e.label}) {dur:.1f}s audio "
                    f"in {time.time() - t1:.1f}s{f' [{len(chunks)} chunks]' if len(chunks) > 1 else ''}")
        finally:
            backend.unload()

    # ---- pass 3: stitch ---------------------------------------------------
    segments: list[Segment] = []
    prev: Line | None = None
    for ln in script.lines:
        a = audio_by_line[ln.index]
        if loudness:
            a = normalize_loudness(a, sr)
        segments.append(Segment(a, pause_for(prev, ln, script.pause_ms)))
        prev = ln
    full = assemble(segments, sr)

    out = Path(output or script.output)
    want_mp3 = mp3 or out.suffix.lower() == ".mp3"
    wav_path = out.with_suffix(".wav")
    write_wav(wav_path, full, sr)
    mp3_path = wav_to_mp3(wav_path, out.with_suffix(".mp3")) if want_mp3 else None
    return RenderResult(wav_path, mp3_path, full.size / sr, generated, cached)
