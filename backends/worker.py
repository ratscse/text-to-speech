"""Backend worker process: `python -m backends.worker <backend_id>`.

Speaks newline-delimited JSON on its *original* stdout; everything the ML
libraries print (including C-level writes) is redirected to stderr so it
cannot corrupt the protocol. Audio is passed back as WAV files.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import traceback


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("backend")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--options", default="{}")
    ap.add_argument("--online", action="store_true")
    args = ap.parse_args()

    # Protect the protocol channel.
    proto_fd = os.dup(1)
    os.dup2(2, 1)
    sys.stdout = sys.stderr
    proto = os.fdopen(proto_fd, "w", encoding="utf-8", buffering=1, newline="\n")
    stdin = open(sys.stdin.fileno(), "r", encoding="utf-8", closefd=False)

    from core.runtime import configure_hf

    configure_hf(offline=not args.online)

    import soundfile as sf

    from backends import backend_class
    from core.emotion import EmotionSpec
    from core.script import Speaker

    def reply(**kw) -> None:
        proto.write(json.dumps(kw, ensure_ascii=False) + "\n")
        proto.flush()

    backend = None
    try:
        cls = backend_class(args.backend)
        backend = cls(device=args.device, options=json.loads(args.options))
    except Exception as e:  # report construction errors on first request
        init_error = e
    else:
        init_error = None

    for raw in stdin:
        raw = raw.strip()
        if not raw:
            continue
        msg = json.loads(raw)
        op = msg.get("op")
        try:
            if op == "quit":
                reply(ok=True)
                return 0
            if init_error is not None:
                raise init_error
            if op == "load":
                backend.ensure_loaded()
                reply(ok=True, sample_rate=backend.sample_rate, device=backend.device)
            elif op == "prepare_voices":
                emos = {k: [EmotionSpec(**e) for e in v] for k, v in (msg.get("emotions") or {}).items()}
                backend.prepare_voices([Speaker.from_dict(d) for d in msg["speakers"]], emos)
                reply(ok=True)
            elif op == "normalize":
                reply(ok=True, text=backend.normalize_text(msg["text"]))
            elif op == "synth":
                sp = Speaker.from_dict(msg["speaker"])
                voice = backend.voice(sp)
                audio = backend.synth(msg["text"], voice, EmotionSpec(**msg["emotion"]))
                sf.write(msg["out"], audio, backend.sample_rate, subtype="FLOAT")
                reply(ok=True, sample_rate=backend.sample_rate)
            elif op == "download":
                cls.download(json.loads(args.options))
                reply(ok=True)
            else:
                raise ValueError(f"unknown op {op!r}")
        except Exception as e:  # keep serving; the parent decides what to do
            reply(ok=False, error=f"{type(e).__name__}: {e}", kind=type(e).__name__,
                  trace=traceback.format_exc())
    return 0


if __name__ == "__main__":
    sys.exit(main())
