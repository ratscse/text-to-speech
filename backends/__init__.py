"""Backend registry and factory.

Each backend declares an `env` group. If `tts/.venvs/<env>` exists (created by
`tts.py --setup`), the backend runs in a worker subprocess using that venv's
Python — this lets models with conflicting `transformers` pins coexist.
Otherwise it is imported in-process (single-venv installs).
"""
from __future__ import annotations

import importlib
import os
import sys
from typing import Any

from core.runtime import SetupError, venv_python

# id -> (module, class, env, langs)
SPECS: dict[str, tuple[str, str, str, tuple[str, ...]]] = {
    "qwen":          ("backends.en_qwen",       "QwenBackend",         "qwen",       ("en",)),
    "chatterbox":    ("backends.en_chatterbox", "ChatterboxEnBackend", "chatterbox", ("en",)),
    "chatterbox-bn": ("backends.bn_chatterbox", "ChatterboxBnBackend", "chatterbox", ("bn",)),
    "xtts-bn":       ("backends.bn_xtts",       "XttsBnBackend",       "xtts",       ("bn",)),
    "dummy":         ("backends.dummy",         "DummyBackend",        "",           ("en", "bn")),
}

DEFAULTS = {"en": "qwen", "bn": "chatterbox-bn"}

# what users may type after --backend en=... / bn=...
ALIASES = {
    "en": {"qwen": "qwen", "chatterbox": "chatterbox", "dummy": "dummy"},
    "bn": {"chatterbox": "chatterbox-bn", "chatterbox-bn": "chatterbox-bn",
           "xtts": "xtts-bn", "xtts-bn": "xtts-bn", "dummy": "dummy"},
}


def parse_backend_flags(flags: list[str] | None) -> dict[str, str]:
    """['en=chatterbox', 'bn=xtts'] -> {'en': 'chatterbox', 'bn': 'xtts-bn'}.
    A bare name applies to whichever language(s) it supports."""
    chosen = dict(DEFAULTS)
    for flag in flags or []:
        for item in flag.split(","):
            item = item.strip()
            if not item:
                continue
            if "=" in item:
                lang, name = (s.strip().lower() for s in item.split("=", 1))
                if lang not in ALIASES:
                    raise SetupError(f"--backend {item}: unsupported language {lang!r} (en or bn only)")
                if name not in ALIASES[lang]:
                    raise SetupError(f"--backend {item}: choose one of {', '.join(sorted(ALIASES[lang]))} for {lang}")
                chosen[lang] = ALIASES[lang][name]
            else:
                hits = [lg for lg in ALIASES if item.lower() in ALIASES[lg]]
                if not hits:
                    raise SetupError(f"--backend {item}: unknown backend")
                for lg in hits:
                    chosen[lg] = ALIASES[lg][item.lower()]
    return chosen


def backend_class(backend_id: str):
    if backend_id not in SPECS:
        raise SetupError(f"unknown backend {backend_id!r}")
    mod, cls, _, _ = SPECS[backend_id]
    try:
        return getattr(importlib.import_module(mod), cls)
    except ImportError as e:
        env = SPECS[backend_id][2]
        raise SetupError(
            f"backend {backend_id!r} is not installed ({e}). Run: python tts.py --setup --backend {backend_id}"
            f"  (installs into .venvs/{env})"
        ) from e


def uses_worker(backend_id: str) -> bool:
    env = SPECS[backend_id][2]
    if not env or os.environ.get("TTS_IN_PROCESS") == "1":
        return False
    py = venv_python(env)
    return py.is_file() and os.path.normcase(str(py.resolve())) != os.path.normcase(str(sys.executable))


def create_backend(backend_id: str, device: str = "auto", options: dict[str, Any] | None = None):
    if backend_id not in SPECS:
        raise SetupError(f"unknown backend {backend_id!r}")
    if uses_worker(backend_id):
        from backends.remote import RemoteBackend

        return RemoteBackend(backend_id, venv_python(SPECS[backend_id][2]), device, options)
    return backend_class(backend_id)(device=device, options=options)
