"""Paths, offline mode, device selection."""
from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent  # the tts/ directory
MODELS_DIR = Path(os.environ.get("TTS_MODELS_DIR", ROOT / "models"))
VENVS_DIR = Path(os.environ.get("TTS_VENVS_DIR", ROOT / ".venvs"))
CACHE_DIR = ROOT / "cache"


class SetupError(RuntimeError):
    """User-facing environment/setup problem."""


def configure_hf(offline: bool) -> None:
    """Keep all model weights under tts/models and forbid network at inference."""
    os.environ.setdefault("HF_HOME", str(MODELS_DIR / "hf"))
    os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
    os.environ.setdefault("COQUI_TOS_AGREED", "1")
    os.environ.setdefault("TTS_HOME", str(MODELS_DIR / "coqui"))
    os.environ.setdefault("PYTHONIOENCODING", "utf-8")
    if offline:
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
    else:
        os.environ.pop("HF_HUB_OFFLINE", None)
        os.environ.pop("TRANSFORMERS_OFFLINE", None)


def pick_device(requested: str = "auto") -> str:
    """'auto' -> cuda if available else cpu. 'cuda' without CUDA is an error."""
    import torch

    requested = (requested or "auto").lower()
    has_cuda = torch.cuda.is_available()
    if requested == "cuda":
        if not has_cuda:
            raise SetupError(
                "--device cuda requested but CUDA is not available to PyTorch "
                f"(torch {torch.__version__}, built for CUDA: {torch.version.cuda}). "
                "Install an NVIDIA driver + the CUDA build of torch, or use --device cpu."
            )
        return "cuda"
    if requested == "cpu":
        return "cpu"
    if requested != "auto":
        raise SetupError(f"unknown device {requested!r} (use auto, cpu or cuda)")
    return "cuda" if has_cuda else "cpu"


def venv_python(env: str) -> Path:
    d = VENVS_DIR / env
    return d / ("Scripts/python.exe" if sys.platform == "win32" else "bin/python")


def _local_repo_dir(repo_id: str) -> Path:
    return MODELS_DIR / "repos" / repo_id.replace("/", "--")


def _wanted(files: list[str], allow_patterns: list[str] | str | None) -> list[str]:
    from fnmatch import fnmatch

    if not allow_patterns:
        return files
    pats = [allow_patterns] if isinstance(allow_patterns, str) else list(allow_patterns)
    return [f for f in files if any(fnmatch(f, p) for p in pats)]


def hf_snapshot(repo_id: str, allow_patterns: list[str] | str | None = None) -> str:
    """Return a local directory holding the repo.

    Offline (inference): only local files are used; missing => clear error.
    Online (--setup): reuse a complete HF-cache snapshot if present, otherwise
    download file-by-file into models/repos/<repo> (a plain folder: no
    symlinks, which Windows refuses without Developer Mode / admin).
    Partial cache snapshots are moved there so nothing is re-downloaded.
    """
    import shutil

    from huggingface_hub import snapshot_download

    local = _local_repo_dir(repo_id)
    if (local / ".complete").is_file():
        return str(local)

    try:
        cached: str | None = snapshot_download(repo_id, local_files_only=True, allow_patterns=allow_patterns)
    except Exception:
        cached = None

    if os.environ.get("HF_HUB_OFFLINE") == "1":
        if cached:
            return cached
        raise SetupError(f"model {repo_id!r} is not downloaded yet. Run: python tts.py --setup")

    from huggingface_hub import HfApi, hf_hub_download

    files = _wanted(HfApi().list_repo_files(repo_id), allow_patterns)
    if cached and all((Path(cached) / f).is_file() for f in files):
        return cached
    local.mkdir(parents=True, exist_ok=True)
    if cached:  # salvage a partial snapshot (real files in Windows' no-symlink mode)
        for f in files:
            src, dst = Path(cached) / f, local / f
            if src.is_file() and not src.is_symlink() and not dst.exists():
                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.move(str(src), str(dst))
    for f in files:
        if not (local / f).is_file():
            print(f"  fetching {repo_id}/{f}", file=sys.stderr, flush=True)
            hf_hub_download(repo_id, f, local_dir=str(local))
    (local / ".complete").write_text("\n".join(files), encoding="utf-8")
    return str(local)
