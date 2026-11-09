"""Deterministic fake backend for tests: tones whose pitch depends on the voice
and whose length depends on the text. No ML dependencies."""
from __future__ import annotations

import hashlib

import numpy as np

from backends.base import TTSBackend
from core.emotion import EmotionSpec
from core.script import Speaker


class DummyBackend(TTSBackend):
    name = "dummy"
    langs = ("en", "bn")
    env = ""
    sample_rate = 22050  # deliberately != 24000 to exercise resampling
    supports_design = True

    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        self.load_calls = 0
        self.synth_calls: list[tuple[str, str, str]] = []
        self.voice_calls: list[str] = []

    def load(self) -> None:
        self.load_calls += 1
        self.device = "cpu"

    def make_voice(self, speaker: Speaker) -> float:
        self.voice_calls.append(speaker.name)
        h = int(hashlib.md5(speaker.name.encode()).hexdigest()[:6], 16)
        return 150.0 + (h % 300)  # Hz

    def synth(self, text: str, voice: float, emotion: EmotionSpec) -> np.ndarray:
        self.synth_calls.append((text, str(voice), emotion.label))
        dur = 0.05 * max(1, len(text))
        t = np.arange(int(self.sample_rate * dur)) / self.sample_rate
        amp = 0.1 + 0.4 * emotion.exaggeration
        tone = amp * np.sin(2 * np.pi * voice * t)
        pad = np.zeros(int(self.sample_rate * 0.2))
        return np.concatenate([pad, tone, pad]).astype(np.float32)
