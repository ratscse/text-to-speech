"""Bengali fallback: XTTS v2 fine-tuned on Bangla (EMTIAZZ/xtts-bangla-finetuned).

* The HF repo is GATED: accept its terms on huggingface.co and log in
  (`hf auth login` inside .venvs/xtts, or set HF_TOKEN) before `--setup`.
* LICENSE: XTTS-v2 weights are under the Coqui Public Model License
  (non-commercial). A fine-tune is a derivative — treat it as non-commercial.
* Quirk: the checkpoint was trained with the Bangla text routed through the
  English language id, so we pass language="en" for Bengali text
  (language="bn" produces garbage).
* No emotion control in XTTS; `exaggeration` is mapped to sampling
  temperature (more expressive = more varied) and emotion otherwise comes from
  the reference clip.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import numpy as np

from backends.base import TTSBackend
from core.emotion import EmotionSpec
from core.runtime import SetupError, hf_snapshot, pick_device
from core.script import Speaker
from text.bn_norm import normalize as bn_normalize
from text.tags import strip_tags

REPO = "EMTIAZZ/xtts-bangla-finetuned"
FILES = ["config.json", "vocab.json", "model.pth", "*.md"]


class XttsBnBackend(TTSBackend):
    name = "xtts-bn"
    langs = ("bn",)
    env = "xtts"
    sample_rate = 24000

    def load(self) -> None:
        from backends._torch_util import setup_threads

        self.device = pick_device(self.requested_device)
        setup_threads(self.options)
        from TTS.tts.configs.xtts_config import XttsConfig
        from TTS.tts.models.xtts import Xtts

        d = Path(hf_snapshot(str(self.options.get("model") or REPO), allow_patterns=self._files()))
        ckpt = d / str(self.options.get("checkpoint", "model.pth"))
        for f in (d / "config.json", d / "vocab.json", ckpt):
            if not f.is_file():
                raise SetupError(f"XTTS Bangla file missing: {f}. Re-run --setup (repo is gated; log in first).")
        config = XttsConfig()
        config.load_json(str(d / "config.json"))
        model = Xtts.init_from_config(config)
        model.load_checkpoint(config, checkpoint_path=str(ckpt), vocab_path=str(d / "vocab.json"),
                              eval=True, strict=False, use_deepspeed=False)
        model.to(self.device)
        self.model = model
        self.sample_rate = int(getattr(config.audio, "output_sample_rate", 24000))

    def _files(self) -> list[str]:
        ck = str(self.options.get("checkpoint", "model.pth"))
        return sorted(set(FILES + [ck]))

    def unload(self) -> None:
        from backends._torch_util import free_memory

        self.model = None
        super().unload()
        free_memory()

    def make_voice(self, speaker: Speaker):
        gpt_cond_latent, speaker_embedding = self.model.get_conditioning_latents(audio_path=[speaker.ref_wav])
        return gpt_cond_latent, speaker_embedding

    def normalize_text(self, text: str) -> str:
        return bn_normalize(strip_tags(text))

    def synth(self, text: str, voice, emotion: EmotionSpec) -> np.ndarray:
        import torch

        from backends._torch_util import seed_all, to_numpy

        if "seed" in self.options:
            seed_all(int(self.options["seed"]))
        gpt_cond_latent, speaker_embedding = voice
        temp = float(self.options.get("temperature", 0.55 + 0.3 * min(1.0, emotion.exaggeration)))
        with torch.inference_mode():
            out = self.model.inference(
                text, str(self.options.get("language", "en")), gpt_cond_latent, speaker_embedding,
                temperature=temp, repetition_penalty=float(self.options.get("repetition_penalty", 5.0)),
                speed=float(self.options.get("speed", 1.0)), enable_text_splitting=False,
            )
        return to_numpy(out["wav"])

    @classmethod
    def download(cls, options: dict[str, Any] | None = None) -> None:
        options = options or {}
        try:
            hf_snapshot(str(options.get("model") or REPO),
                        allow_patterns=sorted(set(FILES + [str(options.get("checkpoint", "model.pth"))])))
        except Exception as e:
            raise SetupError(
                f"could not download {REPO}: {e}\n  This repo is gated: open https://huggingface.co/{REPO}, "
                f"accept the terms, then run `.venvs\\xtts\\Scripts\\hf auth login` (or set HF_TOKEN) and retry."
            ) from e
        print("[xtts-bn] NOTE: XTTS-v2 derivatives fall under the Coqui Public Model License (non-commercial).",
              file=sys.stderr)
