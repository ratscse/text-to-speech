"""Common interface every TTS backend implements."""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import numpy as np

from core.emotion import EmotionSpec
from core.script import Speaker


class TTSBackend(ABC):
    #: registry id, e.g. "qwen"
    name: str = ""
    #: languages this backend may be used for
    langs: tuple[str, ...] = ()
    #: venv group under tts/.venvs/<env> holding this backend's dependencies
    env: str = ""
    #: native output sample rate
    sample_rate: int = 24000
    #: whether the backend supports `design` (invented) voices
    supports_design: bool = False

    def __init__(self, device: str = "auto", options: dict[str, Any] | None = None):
        self.requested_device = device
        self.options = dict(options or {})
        self.device: str | None = None
        self._loaded = False
        self._voices: dict[str, Any] = {}

    # ---- lifecycle --------------------------------------------------------
    def ensure_loaded(self) -> None:
        if not self._loaded:
            self.load()
            self._loaded = True

    @abstractmethod
    def load(self) -> None:
        """Load model weights (from the local cache; no network)."""

    def unload(self) -> None:
        self._loaded = False
        self._voices.clear()

    # ---- voices -----------------------------------------------------------
    def prepare_voices(self, speakers: list[Speaker],
                       emotions: dict[str, list[EmotionSpec]] | None = None) -> None:
        """Build/cache voice handles for every speaker before synthesis.
        `emotions` maps speaker name -> emotions that speaker will use; backends
        that bake emotion into the voice (Qwen designed voices) need it up front
        so they can do all VoiceDesign work before loading the clone model."""
        for sp in speakers:
            self.voice(sp)

    def voice(self, speaker: Speaker) -> Any:
        """Return the cached voice for a speaker (created once per run)."""
        if speaker.name not in self._voices:
            if speaker.design and not self.supports_design:
                raise ValueError(
                    f"speaker {speaker.name!r}: backend {self.name!r} cannot design voices; give a ref_wav"
                )
            self.ensure_loaded()
            self._voices[speaker.name] = self.make_voice(speaker)
        return self._voices[speaker.name]

    @abstractmethod
    def make_voice(self, speaker: Speaker) -> Any:
        """Compute a speaker embedding / conditioning / designed voice."""

    # ---- synthesis --------------------------------------------------------
    def normalize_text(self, text: str) -> str:
        return text

    @abstractmethod
    def synth(self, text: str, voice: Any, emotion: EmotionSpec) -> np.ndarray:
        """Return mono float32 audio at `self.sample_rate`."""

    # ---- setup ------------------------------------------------------------
    @classmethod
    def download(cls, options: dict[str, Any] | None = None) -> None:
        """One-time model download (called by `tts.py --setup`)."""
