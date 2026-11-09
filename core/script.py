"""Load and validate script.json, detect language, and chunk long lines."""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

SUPPORTED_LANGS = ("en", "bn")

DEFAULT_PAUSES = {"same_speaker": 250, "turn": 600, "paragraph": 1200}

# Bengali Unicode block U+0980–U+09FF
_BN_RE = re.compile(r"[ঀ-৿]")
# "Other script" letters that count toward English: Latin letters.
_LATIN_RE = re.compile(r"[A-Za-zÀ-ɏ]")
# Inline tags such as [laugh] [sigh] — not counted for language detection.
_TAG_RE = re.compile(r"\[[a-zA-Z_ -]+\]")


class ScriptError(ValueError):
    """Raised for any problem with the input script. Message is user-facing."""


@dataclass(frozen=True)
class Speaker:
    name: str
    lang: str | None  # None => detect per line
    ref_wav: str | None = None  # absolute path
    ref_text: str | None = None  # transcript of ref_wav (improves Qwen cloning)
    design: str | None = None  # text description of an invented voice (en only)

    def fingerprint(self) -> str:
        """Stable identity of the *voice* (used in the cache key)."""
        from core.cache import file_sha1

        if self.ref_wav:
            return f"ref:{file_sha1(self.ref_wav)}:{self.ref_text or ''}"
        return f"design:{self.design}"

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "lang": self.lang,
            "ref_wav": self.ref_wav,
            "ref_text": self.ref_text,
            "design": self.design,
        }

    @staticmethod
    def from_dict(d: dict[str, Any]) -> "Speaker":
        return Speaker(**d)


@dataclass
class Line:
    index: int
    speaker: str
    text: str
    lang: str
    emotion: str = "neutral"
    paragraph: bool = False  # a paragraph pause precedes this line
    pause_before_ms: int | None = None  # explicit override
    exaggeration: float | None = None  # bn/chatterbox override
    instruct: str | None = None  # en/qwen extra direction
    voice: str | None = None  # key in Script.speakers used for synthesis (mood clip); None => speaker


@dataclass
class Script:
    path: Path
    output: Path
    pause_ms: dict[str, int]
    speakers: dict[str, Speaker]
    lines: list[Line] = field(default_factory=list)
    sample_rate: int = 24000
    backend_options: dict[str, dict[str, Any]] = field(default_factory=dict)

    def langs_used(self) -> set[str]:
        return {ln.lang for ln in self.lines}


# --------------------------------------------------------------------------- #
# Language detection
# --------------------------------------------------------------------------- #
def detect_lang(text: str) -> str:
    """Majority script: Bengali block chars vs Latin letters. Ties/none -> en."""
    clean = _TAG_RE.sub("", text)
    bn = len(_BN_RE.findall(clean))
    lat = len(_LATIN_RE.findall(clean))
    return "bn" if bn > lat else "en"


# --------------------------------------------------------------------------- #
# Chunking
# --------------------------------------------------------------------------- #
# Sentence enders: Bangla dari (।), double dari (॥), ? ! . and their
# full-width/ellipsis forms. Keep the punctuation with the sentence.
_SENT_SPLIT_RE = re.compile(r"(?<=[।॥?!.…])[\"'”’)\]]*\s+")
_CLAUSE_SPLIT_RE = re.compile(r"(?<=[,;:—–])\s+")


def _hard_wrap(text: str, max_chars: int) -> list[str]:
    """Split on whitespace so each piece is <= max_chars (a single word may exceed)."""
    out: list[str] = []
    cur = ""
    for word in text.split():
        cand = f"{cur} {word}".strip()
        if len(cand) <= max_chars or not cur:
            cur = cand
        else:
            out.append(cur)
            cur = word
    if cur:
        out.append(cur)
    return out


def _pack(pieces: list[str], max_chars: int) -> list[str]:
    """Greedily merge adjacent pieces while staying under max_chars."""
    out: list[str] = []
    cur = ""
    for p in pieces:
        cand = f"{cur} {p}".strip() if cur else p
        if len(cand) <= max_chars:
            cur = cand
        else:
            if cur:
                out.append(cur)
            cur = p
    if cur:
        out.append(cur)
    return out


def chunk_text(text: str, max_chars: int = 200) -> list[str]:
    """Split text > max_chars at sentence boundaries (। ? ! .), then clauses,
    then whitespace. Short text is returned unchanged as a single chunk."""
    text = " ".join(text.split())
    if not text:
        return []
    if len(text) <= max_chars:
        return [text]
    pieces: list[str] = []
    for sent in _SENT_SPLIT_RE.split(text):
        sent = sent.strip()
        if not sent:
            continue
        if len(sent) <= max_chars:
            pieces.append(sent)
            continue
        for clause in _CLAUSE_SPLIT_RE.split(sent):
            if len(clause) <= max_chars:
                pieces.append(clause)
            else:
                pieces.extend(_hard_wrap(clause, max_chars))
    return _pack(pieces, max_chars)


# --------------------------------------------------------------------------- #
# Loading / validation
# --------------------------------------------------------------------------- #
def _resolve(base: Path, p: str) -> Path:
    path = Path(p)
    return path if path.is_absolute() else (base / path).resolve()


def _check_lang(lang: Any, where: str) -> str:
    if not isinstance(lang, str) or lang.lower() not in SUPPORTED_LANGS:
        raise ScriptError(
            f"{where}: unsupported language {lang!r}. Only 'en' (English) and "
            f"'bn' (Bengali) are supported."
        )
    return lang.lower()


def _load_speaker(name: str, d: Any, base: Path) -> Speaker:
    where = f"speaker {name!r}"
    if not isinstance(d, dict):
        raise ScriptError(f"{where}: must be an object")
    lang = d.get("lang")
    if lang is not None:
        lang = _check_lang(lang, where)
    ref_wav, design = d.get("ref_wav"), d.get("design")
    if bool(ref_wav) == bool(design):
        raise ScriptError(f"{where}: give exactly one of 'ref_wav' (clone) or 'design' (invented voice)")
    if design and lang == "bn":
        raise ScriptError(
            f"{where}: Bengali speakers support voice cloning only. 'design' is "
            f"English-only (Qwen3-TTS VoiceDesign). Provide a 'ref_wav' clip instead."
        )
    ref_text = d.get("ref_text")
    ref_path: str | None = None
    if ref_wav:
        rp = _resolve(base, ref_wav)
        if not rp.is_file():
            raise ScriptError(
                f"{where}: ref_wav not found: {rp}\n"
                f"  Make one with: python scripts/make_ref.py <any-audio> {rp}"
            )
        ref_path = str(rp)
        if ref_text is None:  # sidecar transcript voices/x.txt next to voices/x.wav
            side = rp.with_suffix(".txt")
            if side.is_file():
                ref_text = side.read_text(encoding="utf-8").strip() or None
    return Speaker(name=name, lang=lang, ref_wav=ref_path, ref_text=ref_text, design=design or None)


def _mood_voice(speakers: dict[str, Speaker], spk: str, emotion: str) -> str | None:
    """voices/arif.wav + emotion 'sad' -> use voices/arif_sad.wav if that file exists,
    else the emotion group's clip (core.emotion.MOOD_FALLBACK, e.g. 'lonely' -> arif_sad.wav).
    Registers a derived speaker 'Arif@<clip label>' and returns its key."""
    from core.emotion import MOOD_FALLBACK, known_label

    sp = speakers[spk]
    if not sp.ref_wav:
        return None
    canon = known_label(emotion) or emotion
    ref = Path(sp.ref_wav)
    tries = [(emotion, canon), (canon, canon)]
    if canon in MOOD_FALLBACK:
        tries.append((MOOD_FALLBACK[canon], MOOD_FALLBACK[canon]))
    for label, key_label in dict.fromkeys(tries):
        cand = ref.with_name(f"{ref.stem}_{label}{ref.suffix}")
        if not cand.is_file():
            continue
        key = f"{spk}@{key_label}"
        if key not in speakers:
            side = cand.with_suffix(".txt")
            ref_text = side.read_text(encoding="utf-8").strip() or None if side.is_file() else None
            speakers[key] = Speaker(name=key, lang=sp.lang, ref_wav=str(cand), ref_text=ref_text)
        return key
    return None


def parse_script(data: dict[str, Any], path: Path) -> Script:
    base = path.parent
    if not isinstance(data, dict):
        raise ScriptError("script root must be a JSON object")
    speakers_raw = data.get("speakers")
    if not isinstance(speakers_raw, dict) or not speakers_raw:
        raise ScriptError("'speakers' must be a non-empty object")
    lines_raw = data.get("lines")
    if not isinstance(lines_raw, list) or not lines_raw:
        raise ScriptError("'lines' must be a non-empty list")

    pauses = dict(DEFAULT_PAUSES)
    for k, v in (data.get("pause_ms") or {}).items():
        if k not in DEFAULT_PAUSES:
            raise ScriptError(f"pause_ms: unknown key {k!r} (use {', '.join(DEFAULT_PAUSES)})")
        if not isinstance(v, (int, float)) or v < 0:
            raise ScriptError(f"pause_ms.{k}: must be a non-negative number")
        pauses[k] = int(v)

    speakers = {n: _load_speaker(n, d, base) for n, d in speakers_raw.items()}

    lines: list[Line] = []
    for i, ln in enumerate(lines_raw):
        where = f"lines[{i}]"
        if not isinstance(ln, dict):
            raise ScriptError(f"{where}: must be an object")
        spk = ln.get("speaker")
        if spk not in speakers:
            raise ScriptError(f"{where}: unknown speaker {spk!r} (defined: {', '.join(speakers)})")
        text = ln.get("text")
        if not isinstance(text, str) or not text.strip():
            raise ScriptError(f"{where}: 'text' must be a non-empty string")
        sp = speakers[spk]
        if ln.get("lang") is not None:
            lang = _check_lang(ln["lang"], where)
        elif sp.lang is not None:
            lang = sp.lang
        else:
            lang = detect_lang(text)
        if lang == "bn" and sp.design:
            raise ScriptError(
                f"{where}: speaker {spk!r} uses a designed voice but this line is Bengali. "
                f"Bengali supports voice cloning only — give {spk!r} a 'ref_wav'."
            )
        exg = ln.get("exaggeration")
        if exg is not None and not (isinstance(exg, (int, float)) and 0 <= exg <= 2):
            raise ScriptError(f"{where}: 'exaggeration' must be a number in [0, 2]")
        pb = ln.get("pause_before_ms")
        if pb is not None and not (isinstance(pb, (int, float)) and pb >= 0):
            raise ScriptError(f"{where}: 'pause_before_ms' must be a non-negative number")
        emotion = str(ln.get("emotion") or "neutral").strip().lower()
        lines.append(
            Line(
                index=i,
                speaker=spk,
                text=text.strip(),
                lang=lang,
                emotion=emotion,
                paragraph=bool(ln.get("paragraph", False)),
                pause_before_ms=int(pb) if pb is not None else None,
                exaggeration=float(exg) if exg is not None else None,
                instruct=ln.get("instruct"),
                voice=_mood_voice(speakers, spk, emotion),
            )
        )

    out = data.get("output") or (path.stem + ".wav")
    out_path = _resolve(base, out)
    if out_path.suffix.lower() not in (".wav", ".mp3"):
        raise ScriptError(f"output: must end in .wav or .mp3, got {out!r}")

    opts = data.get("backend_options") or {}
    if not isinstance(opts, dict):
        raise ScriptError("backend_options must be an object")

    return Script(
        path=path,
        output=out_path,
        pause_ms=pauses,
        speakers=speakers,
        lines=lines,
        sample_rate=int(data.get("sample_rate", 24000)),
        backend_options=opts,
    )


def load_script(path: str | Path) -> Script:
    path = Path(path).resolve()
    if not path.is_file():
        raise ScriptError(f"script not found: {path}")
    try:
        data = json.loads(path.read_text(encoding="utf-8-sig"))
    except json.JSONDecodeError as e:
        raise ScriptError(f"{path.name}: invalid JSON at line {e.lineno} col {e.colno}: {e.msg}") from e
    return parse_script(data, path)
