"""Inline non-verbal tags: [laugh] [sigh] [whisper] [gasp] ..."""
from __future__ import annotations

import re

TAG_RE = re.compile(r"\[\s*([a-zA-Z][a-zA-Z _-]*?)\s*\]")


def find_tags(text: str) -> list[str]:
    return [m.group(1).lower().replace(" ", "_").replace("-", "_") for m in TAG_RE.finditer(text)]


def strip_tags(text: str) -> str:
    return " ".join(TAG_RE.sub(" ", text).split())


def map_tags(text: str, mapping: dict[str, str | None]) -> str:
    """Rewrite tags via mapping (name -> replacement token, or None to drop).
    Tags missing from the mapping are dropped."""

    def sub(m: re.Match) -> str:
        key = m.group(1).lower().replace(" ", "_").replace("-", "_")
        rep = mapping.get(key)
        return f" {rep} " if rep else " "

    return " ".join(TAG_RE.sub(sub, text).split())
