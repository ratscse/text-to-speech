"""English fallback: Chatterbox (chatterbox-tts 0.1.7, MIT).

variant "original" (default): `exaggeration` emotion control + cloning.
    Its tokenizer contains [laughter] [giggle] [sigh] [cough] [gasp] [whisper]
    [clear_throat] tokens; Resemble does not document them for this model, so
    tag effects are best-effort.
variant "turbo" (backend_options: {"chatterbox": {"variant": "turbo"}}):
    officially supports paralinguistic tags, but ignores exaggeration/cfg, and
    needs a reference clip longer than 5 s.
"""
from __future__ import annotations

import copy
import sys
from typing import Any

import numpy as np

from backends.base import TTSBackend
from core.emotion import EmotionSpec
from core.runtime import hf_snapshot, pick_device
from core.script import Speaker
from text.en_norm import normalize as en_normalize
from text.tags import map_tags

REPO_ORIGINAL = "ResembleAI/chatterbox"
REPO_TURBO = "ResembleAI/chatterbox-turbo"
FILES_ORIGINAL = ["ve.safetensors", "t3_cfg.safetensors", "s3gen.safetensors", "tokenizer.json", "conds.pt"]

TAGS_ORIGINAL = {
    "laugh": "[laughter]", "laughs": "[laughter]", "laughter": "[laughter]", "chuckle": "[laughter]",
    "giggle": "[giggle]", "sigh": "[sigh]", "cough": "[cough]", "gasp": "[gasp]",
    "whisper": "[whisper]", "clear_throat": "[clear_throat]",
}
TAGS_TURBO = {
    "laugh": "[laugh]", "laughs": "[laugh]", "laughter": "[laugh]", "chuckle": "[chuckle]",
    "giggle": "[chuckle]", "sigh": "[sigh]", "cough": "[cough]", "gasp": "[gasp]",
    "whisper": "[whispering]", "whispering": "[whispering]", "clear_throat": "[clear throat]",
    "cry": "[crying]", "crying": "[crying]", "sob": "[crying]", "groan": "[groan]", "sniff": "[sniff]",
    "shush": "[shush]", "angry": "[angry]", "happy": "[happy]", "fear": "[fear]",
    "surprised": "[surprised]", "sarcastic": "[sarcastic]", "dramatic": "[dramatic]",
}


def log(msg: str) -> None:
    print(f"[chatterbox] {msg}", file=sys.stderr, flush=True)


class ChatterboxEnBackend(TTSBackend):
    name = "chatterbox"
    langs = ("en",)
    env = "chatterbox"
    sample_rate = 24000

    @property
    def variant(self) -> str:
        return str(self.options.get("variant", "original")).lower()

    def load(self) -> None:
        from backends._torch_util import setup_threads

        self.device = pick_device(self.requested_device)
        setup_threads(self.options)
        if self.variant == "turbo":
            from chatterbox.tts_turbo import ChatterboxTurboTTS

            path = hf_snapshot(REPO_TURBO, allow_patterns=["*.safetensors", "*.json", "*.txt", "*.pt", "*.model"])
            self.model = ChatterboxTurboTTS.from_local(path, self.device)
        else:
            from chatterbox.tts import ChatterboxTTS

            path = hf_snapshot(REPO_ORIGINAL, allow_patterns=FILES_ORIGINAL)
            self.model = ChatterboxTTS.from_local(path, self.device)
        self.sample_rate = int(self.model.sr)

    def unload(self) -> None:
        from backends._torch_util import free_memory

        self.model = None
        super().unload()
        free_memory()

    def make_voice(self, speaker: Speaker):
        self.model.prepare_conditionals(speaker.ref_wav, exaggeration=0.5)
        return copy.copy(self.model.conds)  # per-speaker conditioning, reused every line

    def normalize_text(self, text: str) -> str:
        return en_normalize(map_tags(text, TAGS_TURBO if self.variant == "turbo" else TAGS_ORIGINAL))

    def _gen_kwargs(self, emotion: EmotionSpec) -> dict[str, Any]:
        kw: dict[str, Any] = {"temperature": float(self.options.get("temperature", 0.8))}
        if self.variant != "turbo":
            kw["exaggeration"] = emotion.exaggeration
            kw["cfg_weight"] = float(self.options.get("cfg_weight", emotion.cfg_weight))
        return kw

    def synth(self, text: str, voice, emotion: EmotionSpec) -> np.ndarray:
        import torch

        from backends._torch_util import seed_all, to_numpy

        if "seed" in self.options:
            seed_all(int(self.options["seed"]))
        self.model.conds = voice
        with torch.inference_mode():
            wav = self.model.generate(text, **self._gen_kwargs(emotion))
        return to_numpy(wav)

    @classmethod
    def download(cls, options: dict[str, Any] | None = None) -> None:
        options = options or {}
        if str(options.get("variant", "original")).lower() == "turbo":
            hf_snapshot(REPO_TURBO, allow_patterns=["*.safetensors", "*.json", "*.txt", "*.pt", "*.model"])
        else:
            hf_snapshot(REPO_ORIGINAL, allow_patterns=FILES_ORIGINAL)
