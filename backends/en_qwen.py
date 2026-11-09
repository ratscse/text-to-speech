"""English primary backend: Qwen3-TTS (qwen-tts, Apache-2.0).

How voices work (verified against qwen-tts 0.1.1 source):
  * The Base (clone) model has NO `instruct` argument; only the VoiceDesign
    model takes natural-language directions.
  * So for `design` speakers we follow Qwen's documented "design then clone"
    workflow: VoiceDesign renders a reference clip per (speaker, emotion) using
    "<design>. <emotion instruction>", then the Base model clones from that clip.
    Same description + fixed seed => consistent timbre; the emotion rides on the
    reference's delivery. Reference clips are cached in cache/voices/.
  * `ref_wav` speakers are cloned directly. Give `ref_text` (or voices/x.txt) for
    best quality; without it we fall back to x-vector-only mode (timbre only).
    Their emotion comes from the reference clip and the text itself.

Memory: VoiceDesign is 1.7B only. On CPU (fp32) it needs ~7 GB RAM, so it is
loaded, used, and freed before the Base model is loaded.
"""
from __future__ import annotations

import hashlib
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf

from backends.base import TTSBackend
from core.emotion import EmotionSpec, resolve
from core.runtime import CACHE_DIR, hf_snapshot, pick_device
from core.script import Speaker
from text.en_norm import normalize as en_normalize
from text.tags import find_tags, strip_tags

DESIGN_MODEL = "Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign"
BASE_MODELS = {"1.7b": "Qwen/Qwen3-TTS-12Hz-1.7B-Base", "0.6b": "Qwen/Qwen3-TTS-12Hz-0.6B-Base"}

# Spoken by every designed voice to create its reference clip (the transcript
# is known exactly, which gives the Base model full in-context cloning).
REF_TEXT = (
    "Oh, hello there! I didn't expect to see you here today. "
    "Well, since you're here, let me tell you a little story about what happened this morning."
)


def log(msg: str) -> None:
    print(f"[qwen] {msg}", file=sys.stderr, flush=True)


@dataclass
class QwenVoice:
    speaker: str
    prompts: dict[str, Any] = field(default_factory=dict)  # EmotionSpec.key() -> clone prompt
    default: Any = None

    def for_emotion(self, emotion: EmotionSpec):
        return self.prompts.get(emotion.key(), self.default)


class QwenBackend(TTSBackend):
    name = "qwen"
    langs = ("en",)
    env = "qwen"
    sample_rate = 24000
    supports_design = True

    # ---- options ------------------------------------------------------------
    def _base_repo(self) -> str:
        opt = str(self.options.get("base_model") or "").strip()
        if opt:
            return BASE_MODELS.get(opt.lower(), opt)
        # 1.7B is better; 0.6B is ~3x faster, the sane default on CPU.
        return BASE_MODELS["1.7b" if self.device == "cuda" else "0.6b"]

    def _design_repo(self) -> str:
        return str(self.options.get("design_model") or DESIGN_MODEL)

    def _seed(self) -> int:
        return int(self.options.get("seed", 1234))

    def _gen_kwargs(self) -> dict[str, Any]:
        kw: dict[str, Any] = {}
        for k in ("temperature", "top_p", "top_k", "max_new_tokens", "repetition_penalty"):
            if k in self.options:
                kw[k] = self.options[k]
        return kw

    # ---- lifecycle ----------------------------------------------------------
    def load(self) -> None:
        import torch

        from backends._torch_util import setup_threads

        self.device = pick_device(self.requested_device)
        setup_threads(self.options)
        self._dtype = torch.bfloat16 if self.device == "cuda" else torch.float32
        self._base = None
        self._design = None
        # Fail fast with a clear message if weights are missing.
        hf_snapshot(self._base_repo())

    def _load_model(self, repo: str):
        from qwen_tts import Qwen3TTSModel

        path = hf_snapshot(repo)
        log(f"loading {repo} on {self.device} ({str(self._dtype).replace('torch.', '')})...")
        kw: dict[str, Any] = {"device_map": "cuda:0" if self.device == "cuda" else "cpu", "dtype": self._dtype}
        if self.device == "cuda":
            try:
                import flash_attn  # noqa: F401

                kw["attn_implementation"] = "flash_attention_2"
            except ImportError:
                kw["attn_implementation"] = "sdpa"
        else:
            kw["attn_implementation"] = "sdpa"
        return Qwen3TTSModel.from_pretrained(path, **kw)

    def _base_model(self):
        if self._base is None:
            self._base = self._load_model(self._base_repo())
        return self._base

    def unload(self) -> None:
        from backends._torch_util import free_memory

        self._base = None
        self._design = None
        super().unload()
        free_memory()

    # ---- voices -------------------------------------------------------------
    def _design_ref_path(self, sp: Speaker, emo: EmotionSpec) -> Path:
        h = hashlib.sha1(json.dumps(
            [self._design_repo(), sp.design, emo.instruct, self._seed(), REF_TEXT]).encode()).hexdigest()
        return CACHE_DIR / "voices" / f"qwen_design_{h}.wav"

    def _design_instruct(self, sp: Speaker, emo: EmotionSpec) -> str:
        return f"{sp.design.strip().rstrip('.')}. {emo.instruct}"

    def prepare_voices(self, speakers: list[Speaker],
                       emotions: dict[str, list[EmotionSpec]] | None = None) -> None:
        from backends._torch_util import free_memory, seed_all

        self.ensure_loaded()
        emotions = emotions or {}
        todo = [s for s in speakers if s.name not in self._voices]

        # Phase 1 — VoiceDesign: render any missing reference clips.
        jobs: list[tuple[Speaker, EmotionSpec, Path]] = []
        for sp in todo:
            if sp.design:
                emos = emotions.get(sp.name) or [resolve("neutral")]
                for e in emos:
                    jobs.append((sp, e, self._design_ref_path(sp, e)))
        missing = [j for j in jobs if not j[2].is_file()]
        if missing:
            design = self._load_model(self._design_repo())
            for sp, e, path in missing:
                ins = self._design_instruct(sp, e)
                log(f"designing voice '{sp.name}' ({e.label}): {ins}")
                seed_all(self._seed())
                wavs, sr = design.generate_voice_design(text=REF_TEXT, instruct=ins, language="English",
                                                        **self._gen_kwargs())
                path.parent.mkdir(parents=True, exist_ok=True)
                sf.write(path, np.asarray(wavs[0], dtype=np.float32), sr)
                path.with_suffix(".txt").write_text(REF_TEXT, encoding="utf-8")
            del design
            free_memory()

        # Phase 2 — Base model: build clone prompts once per speaker/emotion.
        base = self._base_model()
        for sp in todo:
            v = QwenVoice(sp.name)
            if sp.design:
                for e in emotions.get(sp.name) or [resolve("neutral")]:
                    p = base.create_voice_clone_prompt(ref_audio=str(self._design_ref_path(sp, e)),
                                                       ref_text=REF_TEXT)
                    v.prompts[e.key()] = p
                    v.default = v.default or p
            else:
                xvec = not sp.ref_text
                if xvec:
                    log(f"'{sp.name}': no ref_text (or {Path(sp.ref_wav).with_suffix('.txt').name}) — "
                        f"using x-vector-only cloning; add a transcript for better likeness")
                v.default = base.create_voice_clone_prompt(ref_audio=sp.ref_wav, ref_text=sp.ref_text,
                                                           x_vector_only_mode=xvec)
            self._voices[sp.name] = v

    def make_voice(self, speaker: Speaker) -> QwenVoice:
        self.prepare_voices([speaker], {speaker.name: [resolve("neutral")]})
        return self._voices[speaker.name]

    # ---- synthesis ----------------------------------------------------------
    def normalize_text(self, text: str) -> str:
        tags = find_tags(text)
        if tags:
            log(f"inline tags {tags} are not supported by Qwen3-TTS clone mode; removed "
                f"(use an emotion label, or --backend en=chatterbox for tags)")
        return en_normalize(strip_tags(text))

    def synth(self, text: str, voice: QwenVoice, emotion: EmotionSpec) -> np.ndarray:
        from backends._torch_util import seed_all, to_numpy

        prompt = voice.for_emotion(emotion)
        seed_all(self._seed())
        wavs, sr = self._base_model().generate_voice_clone(
            text=text, language="English", voice_clone_prompt=prompt, **self._gen_kwargs())
        self.sample_rate = int(sr)
        return to_numpy(wavs[0])

    # ---- setup --------------------------------------------------------------
    @classmethod
    def download(cls, options: dict[str, Any] | None = None) -> None:
        options = options or {}
        repos = [options.get("design_model") or DESIGN_MODEL]
        base = options.get("base_model")
        if not base:
            import torch

            base = "1.7b" if torch.cuda.is_available() else "0.6b"  # matches _base_repo() default
        repos.append(BASE_MODELS.get(str(base).lower(), base))
        for r in repos:
            log(f"downloading {r} ...")
            hf_snapshot(r)
