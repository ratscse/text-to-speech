"""Local English + Bengali dialogue TTS.

  python tts.py script.json                     # render (offline)
  python tts.py script.json --mp3               # also write .mp3 (ffmpeg)
  python tts.py script.json --backend en=chatterbox --backend bn=xtts
  python tts.py --setup                         # one-time: venvs + model downloads
  python tts.py --setup --backend bn=xtts       # set up a fallback too
  python tts.py script.json --setup --skip-install --backend en=qwen   # fetch models the script's backend_options name
  python tts.py --check                         # show what is installed
"""
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from core.runtime import CACHE_DIR, MODELS_DIR, ROOT, VENVS_DIR, SetupError, configure_hf, venv_python  # noqa: E402

TORCH_VERSION = "2.6.0"


def log(msg: str) -> None:
    print(msg, file=sys.stderr, flush=True)


# --------------------------------------------------------------------------- #
# setup
# --------------------------------------------------------------------------- #
def _has_nvidia() -> bool:
    return shutil.which("nvidia-smi") is not None


def _run(cmd: list[str]) -> None:
    log("  $ " + " ".join(cmd))
    r = subprocess.run(cmd)
    if r.returncode != 0:
        raise SetupError(f"command failed (exit {r.returncode}): {' '.join(cmd)}")


def _base_python() -> str:
    """Interpreter used to create backend venvs (3.10-3.12 recommended)."""
    exe = os.environ.get("TTS_BASE_PYTHON") or sys.executable
    v = sys.version_info
    if not os.environ.get("TTS_BASE_PYTHON") and not ((3, 10) <= (v.major, v.minor) <= (3, 12)):
        raise SetupError(
            f"Python {v.major}.{v.minor} is too new/old for PyTorch 2.6 wheels. Run setup with "
            f"Python 3.10-3.12 (e.g. `py -3.11 tts.py --setup`) or set TTS_BASE_PYTHON."
        )
    return exe


def setup_env(env: str, cuda: bool) -> Path:
    py = venv_python(env)
    if not py.is_file():
        log(f"creating venv .venvs/{env} ...")
        VENVS_DIR.mkdir(parents=True, exist_ok=True)
        _run([_base_python(), "-m", "venv", str(VENVS_DIR / env)])
    pip = [str(py), "-m", "pip", "install", "--disable-pip-version-check"]
    _run(pip + ["-q", "-U", "pip", "wheel", "setuptools<81"])
    torch_cmd = pip + [f"torch=={TORCH_VERSION}", f"torchaudio=={TORCH_VERSION}"]
    if cuda:
        torch_cmd += ["--index-url", "https://download.pytorch.org/whl/cu124"]
    elif sys.platform.startswith("linux"):
        torch_cmd += ["--index-url", "https://download.pytorch.org/whl/cpu"]
    _run(torch_cmd)
    _run(pip + ["-r", str(ROOT / "requirements" / f"{env}.txt")])
    return py


def backend_options_for(opts: dict, bid: str) -> dict:
    return opts.get(bid) or opts.get({"xtts-bn": "xtts"}.get(bid, bid)) or {}


def cmd_setup(backend_ids: list[str], skip_install: bool, device: str, script_path: str | None = None) -> int:
    from backends import SPECS
    from backends.remote import RemoteBackend

    cuda = _has_nvidia() and device != "cpu"
    log(f"setup: backends={backend_ids}  GPU={'NVIDIA (CUDA torch)' if cuda else 'none (CPU torch)'}")
    envs = sorted({SPECS[b][2] for b in backend_ids if SPECS[b][2]})
    if not skip_install:
        for env in envs:
            setup_env(env, cuda)
    opts: dict = {}
    if script_path:  # honour the script's backend_options (e.g. qwen base_model 1.7b, chatterbox turbo)
        from core.script import load_script

        opts = load_script(script_path).backend_options
    configure_hf(offline=False)
    for b in backend_ids:
        env = SPECS[b][2]
        if not env:
            continue
        log(f"downloading weights for '{b}' into {MODELS_DIR} ...")
        rb = RemoteBackend(b, venv_python(env), device="cpu", options=backend_options_for(opts, b), online=True)
        try:
            rb.download()
        finally:
            rb.unload()
    log("setup complete. Inference is now fully offline.")
    return 0


def setup_targets(flags: list[str] | None) -> list[str]:
    """No flags -> the two primary backends. Otherwise exactly the named ones."""
    from backends import ALIASES, DEFAULTS

    if not flags:
        return sorted(set(DEFAULTS.values()))
    ids: set[str] = set()
    for flag in flags:
        for item in filter(None, (s.strip().lower() for s in flag.split(","))):
            if item == "all":
                ids |= {v for m in ALIASES.values() for v in m.values()}
                continue
            lang, _, name = item.rpartition("=")
            langs = [lang] if lang else [lg for lg in ALIASES if name in ALIASES[lg]]
            for lg in langs:
                if lg not in ALIASES or name not in ALIASES[lg]:
                    raise SetupError(f"--backend {item}: unknown backend")
                ids.add(ALIASES[lg][name])
            if not langs:
                raise SetupError(f"--backend {item}: unknown backend")
    return sorted(ids - {"dummy"})


def cmd_check() -> int:
    from backends import DEFAULTS, SPECS
    from core.stitch import ffmpeg_path

    print(f"root:    {ROOT}\nmodels:  {MODELS_DIR}\nffmpeg:  {ffmpeg_path() or 'NOT FOUND (no MP3 export)'}")
    print(f"nvidia:  {'yes' if _has_nvidia() else 'no (CPU mode)'}")
    for bid, (_, _, env, langs) in SPECS.items():
        if not env:
            continue
        ok = venv_python(env).is_file()
        default = [lg for lg, d in DEFAULTS.items() if d == bid]
        print(f"  {bid:14s} langs={','.join(langs):6s} env=.venvs/{env:11s} "
              f"{'installed' if ok else 'not installed'}{'  (default ' + ','.join(default) + ')' if default else ''}")
    return 0


# --------------------------------------------------------------------------- #
# render
# --------------------------------------------------------------------------- #
def cmd_render(args) -> int:
    from backends import create_backend, parse_backend_flags
    from core.cache import LineCache
    from core.pipeline import render
    from core.script import load_script

    configure_hf(offline=True)
    script = load_script(args.script)
    chosen = parse_backend_flags(args.backend)
    used = {lg: chosen[lg] for lg in sorted(script.langs_used())}
    n_spk = len({ln.speaker for ln in script.lines})
    log(f"{script.path.name}: {len(script.lines)} lines, {n_spk} speakers, backends {used}")
    moods = sorted({ln.voice for ln in script.lines if ln.voice})
    if moods:
        log(f"mood clips: {', '.join(f'{m} <- {Path(script.speakers[m].ref_wav).name}' for m in moods)}")

    for name, sp in script.speakers.items():  # design voices need a design-capable en backend
        if sp.design and used.get("en") not in (None, "qwen", "dummy"):
            raise SetupError(f"speaker {name!r} uses 'design', which needs the Qwen backend "
                             f"(got en={used['en']}). Give {name!r} a ref_wav or use --backend en=qwen.")

    def factory(bid: str):
        return create_backend(bid, device=args.device, options=backend_options_for(script.backend_options, bid))

    t0 = time.time()
    res = render(script, used, factory, LineCache(args.cache_dir, enabled=not args.no_cache),
                 max_chars=args.max_chars, loudness=not args.no_loudnorm, mp3=args.mp3,
                 output=Path(args.output).resolve() if args.output else None)
    log(f"done in {time.time() - t0:.1f}s: {res.generated} generated, {res.cached} cached, "
        f"{res.duration_s:.1f}s of audio")
    print(res.wav)
    if res.mp3:
        print(res.mp3)
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Local English + Bengali dialogue TTS",
                                 formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__)
    ap.add_argument("script", nargs="?", help="script.json")
    ap.add_argument("--setup", action="store_true", help="create backend venvs and download models (needs internet)")
    ap.add_argument("--skip-install", action="store_true", help="with --setup: only download models")
    ap.add_argument("--check", action="store_true", help="show installed backends")
    ap.add_argument("--emotions", action="store_true", help="list emotion modes and their values")
    ap.add_argument("--backend", action="append", metavar="LANG=NAME",
                    help="en=qwen|chatterbox, bn=chatterbox|xtts (repeatable)")
    ap.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    ap.add_argument("-o", "--output", help="override the script's output path")
    ap.add_argument("--mp3", action="store_true", help="also export MP3 via ffmpeg")
    ap.add_argument("--no-cache", action="store_true", help="regenerate every line")
    ap.add_argument("--cache-dir", default=str(CACHE_DIR))
    ap.add_argument("--no-loudnorm", action="store_true", help="skip per-line loudness normalization")
    ap.add_argument("--max-chars", type=int, default=200, help="split lines longer than this (default 200)")
    args = ap.parse_args(argv)

    try:
        if args.setup:
            return cmd_setup(setup_targets(args.backend), args.skip_install, args.device, args.script)
        if args.check:
            return cmd_check()
        if args.emotions:
            from core.emotion import list_emotions

            print(list_emotions())
            return 0
        if not args.script:
            ap.error("give a script.json (or --setup / --check)")
        return cmd_render(args)
    except SetupError as e:
        log(f"error: {e}")
        return 2
    except Exception as e:
        from core.script import ScriptError

        if isinstance(e, ScriptError):
            log(f"script error: {e}")
            return 2
        if os.environ.get("TTS_DEBUG"):
            raise
        log(f"error: {type(e).__name__}: {e}\n(set TTS_DEBUG=1 for a traceback)")
        return 1


if __name__ == "__main__":
    sys.exit(main())
