"""Bengali primary backend: Chatterbox-Bangla (jongy5/chatterbox-bangla, MIT).

The checkpoint is a full fine-tune of Chatterbox with the text vocabulary
extended 704 -> 2530 tokens (new t3_cfg.safetensors + tokenizer.json; s3gen/ve
unchanged). Stock `chatterbox-tts` hard-codes a 704-token T3, so we build the
model ourselves with the vocab size read from the checkpoint — the same fix the
model card applies as a patch. Bengali has no instruction control: emotion is
the reference clip's style + the `exaggeration` slider.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

from backends.en_chatterbox import ChatterboxEnBackend
from core.runtime import hf_snapshot, pick_device
from text.bn_norm import normalize as bn_normalize
from text.tags import strip_tags

REPO_BANGLA = "jongy5/chatterbox-bangla"
FILES = ["ve.safetensors", "t3_cfg.safetensors", "s3gen.safetensors", "tokenizer.json", "conds.pt"]


def load_bangla_chatterbox(ckpt_dir: str | Path, device: str):
    import torch
    from safetensors.torch import load_file
    from chatterbox.models.s3gen import S3Gen
    from chatterbox.models.t3 import T3
    from chatterbox.models.t3.modules.t3_config import T3Config
    from chatterbox.models.tokenizers import EnTokenizer
    from chatterbox.models.voice_encoder import VoiceEncoder
    from chatterbox.tts import ChatterboxTTS, Conditionals

    ckpt_dir = Path(ckpt_dir)
    map_location = torch.device("cpu") if device in ("cpu", "mps") else None

    ve = VoiceEncoder()
    ve.load_state_dict(load_file(ckpt_dir / "ve.safetensors"))
    ve.to(device).eval()

    t3_state = load_file(ckpt_dir / "t3_cfg.safetensors")
    if "model" in t3_state.keys():
        t3_state = t3_state["model"][0]
    vocab = t3_state["text_emb.weight"].shape[0]
    t3 = T3(hp=T3Config(text_tokens_dict_size=vocab))
    t3.load_state_dict(t3_state)
    t3.to(device).eval()

    s3gen = S3Gen()
    s3gen.load_state_dict(load_file(ckpt_dir / "s3gen.safetensors"), strict=False)
    s3gen.to(device).eval()

    tokenizer = EnTokenizer(str(ckpt_dir / "tokenizer.json"))
    if tokenizer.tokenizer.get_vocab_size() > vocab:
        raise RuntimeError(f"tokenizer has {tokenizer.tokenizer.get_vocab_size()} tokens but T3 has {vocab}")

    conds = None
    if (ckpt_dir / "conds.pt").exists():
        conds = Conditionals.load(ckpt_dir / "conds.pt", map_location=map_location).to(device)
    return ChatterboxTTS(t3, s3gen, ve, tokenizer, device, conds=conds)


class ChatterboxBnBackend(ChatterboxEnBackend):
    name = "chatterbox-bn"
    langs = ("bn",)
    env = "chatterbox"

    @property
    def variant(self) -> str:
        return "original"

    def _repo(self) -> str:
        return str(self.options.get("model") or REPO_BANGLA)

    def load(self) -> None:
        from backends._torch_util import setup_threads

        self.device = pick_device(self.requested_device)
        setup_threads(self.options)
        path = hf_snapshot(self._repo(), allow_patterns=FILES)
        self.model = load_bangla_chatterbox(path, self.device)
        self.sample_rate = int(self.model.sr)

    def normalize_text(self, text: str) -> str:
        if "[" in text:
            print("[chatterbox-bn] inline tags are not supported for Bengali; removed", file=sys.stderr)
        return bn_normalize(strip_tags(text))

    def _gen_kwargs(self, emotion) -> dict[str, Any]:
        kw = super()._gen_kwargs(emotion)
        # Slightly more conservative sampling is steadier for the fine-tune.
        kw["temperature"] = float(self.options.get("temperature", 0.7))
        kw["repetition_penalty"] = float(self.options.get("repetition_penalty", 1.3))
        return kw

    @classmethod
    def download(cls, options: dict[str, Any] | None = None) -> None:
        hf_snapshot(str((options or {}).get("model") or REPO_BANGLA), allow_patterns=FILES)
