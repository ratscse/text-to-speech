"""Proxy that runs a backend inside its own venv via backends.worker."""
from __future__ import annotations

import json
import os
import subprocess
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf

from backends.base import TTSBackend
from core.emotion import EmotionSpec
from core.runtime import ROOT, SetupError
from core.script import Speaker


class RemoteError(RuntimeError):
    pass


def _emo_dict(e: EmotionSpec) -> dict[str, Any]:
    return {"label": e.label, "exaggeration": e.exaggeration,
            "cfg_weight": e.cfg_weight, "instruct": e.instruct}


class RemoteBackend(TTSBackend):
    def __init__(self, backend_id: str, python: Path, device: str = "auto",
                 options: dict[str, Any] | None = None, online: bool = False):
        super().__init__(device, options)
        from backends import SPECS

        self.name = backend_id
        self.langs = SPECS[backend_id][3]
        self.env = SPECS[backend_id][2]
        self.python = Path(python)
        self.online = online
        self.proc: subprocess.Popen | None = None
        self._tmp = Path(tempfile.mkdtemp(prefix="tts_worker_"))
        self._n = 0

    # ---- process plumbing -------------------------------------------------
    def _start(self) -> None:
        env = dict(os.environ)
        env["PYTHONPATH"] = str(ROOT) + os.pathsep + env.get("PYTHONPATH", "")
        env["PYTHONIOENCODING"] = "utf-8"
        env["PYTHONUTF8"] = "1"
        cmd = [str(self.python), "-u", "-m", "backends.worker", self.name,
               "--device", self.requested_device, "--options", json.dumps(self.options)]
        if self.online:
            cmd.append("--online")
        self.proc = subprocess.Popen(cmd, cwd=str(ROOT), env=env, stdin=subprocess.PIPE,
                                     stdout=subprocess.PIPE, stderr=None,
                                     text=True, encoding="utf-8", bufsize=1)

    def _call(self, **msg) -> dict[str, Any]:
        if self.proc is None or self.proc.poll() is not None:
            self._start()
        assert self.proc and self.proc.stdin and self.proc.stdout
        try:
            self.proc.stdin.write(json.dumps(msg, ensure_ascii=False) + "\n")
            self.proc.stdin.flush()
            line = self.proc.stdout.readline()
        except (BrokenPipeError, OSError) as e:
            raise RemoteError(f"{self.name} worker died ({e}); see messages above") from e
        if not line:
            code = self.proc.wait()
            raise RemoteError(f"{self.name} worker exited with code {code}; see messages above")
        res = json.loads(line)
        if not res.get("ok"):
            if os.environ.get("TTS_DEBUG"):
                print(res.get("trace", ""), flush=True)
            exc = SetupError if res.get("kind") == "SetupError" else RemoteError
            raise exc(f"[{self.name}] {res.get('error')}")
        return res

    # ---- TTSBackend -------------------------------------------------------
    def load(self) -> None:
        res = self._call(op="load")
        self.sample_rate = int(res["sample_rate"])
        self.device = res.get("device")

    def prepare_voices(self, speakers: list[Speaker],
                       emotions: dict[str, list[EmotionSpec]] | None = None) -> None:
        self.ensure_loaded()
        emo = {k: [_emo_dict(e) for e in v] for k, v in (emotions or {}).items()}
        self._call(op="prepare_voices", speakers=[s.to_dict() for s in speakers], emotions=emo)
        for s in speakers:
            self._voices[s.name] = s

    def make_voice(self, speaker: Speaker) -> Speaker:
        self._call(op="prepare_voices", speakers=[speaker.to_dict()])
        return speaker

    def voice(self, speaker: Speaker) -> Speaker:
        if speaker.name not in self._voices:
            self.ensure_loaded()
            self._voices[speaker.name] = self.make_voice(speaker)
        return self._voices[speaker.name]

    def normalize_text(self, text: str) -> str:
        return self._call(op="normalize", text=text)["text"]

    def synth(self, text: str, voice: Speaker, emotion: EmotionSpec) -> np.ndarray:
        self._n += 1
        out = self._tmp / f"{self._n}.wav"
        self._call(op="synth", text=text, speaker=voice.to_dict(), out=str(out),
                   emotion=_emo_dict(emotion))
        audio, sr = sf.read(out, dtype="float32")
        out.unlink(missing_ok=True)
        self.sample_rate = sr
        return audio

    def download(self) -> None:  # type: ignore[override]
        self._call(op="download")

    def unload(self) -> None:
        super().unload()
        if self.proc and self.proc.poll() is None:
            try:
                self._call(op="quit")
                self.proc.wait(timeout=30)
            except Exception:
                self.proc.kill()
        self.proc = None
        for f in self._tmp.glob("*"):
            f.unlink(missing_ok=True)
        try:
            self._tmp.rmdir()
        except OSError:
            pass

    def __del__(self):
        try:
            if self.proc and self.proc.poll() is None:
                self.proc.kill()
        except Exception:
            pass
