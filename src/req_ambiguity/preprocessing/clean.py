"""Conservative text normalization: formatting only, preserve ambiguity cues."""

from __future__ import annotations

import re


_CTRL_CHARS_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
_WHITESPACE_RE = re.compile(r"\s+")


def normalize_story_text(text: str) -> str:
    """
    Format-only normalization.

    - Strip leading/trailing whitespace
    - Normalize CR/LF to spaces, then collapse runs of whitespace
    - Remove ASCII control characters (except tab, kept as content delimiter)
    - Does NOT remove vague words, placeholders, or rewrite phrasing
    """
    s = text.replace("\r\n", "\n").replace("\r", "\n")
    s = s.replace("\n", " ")
    s = _CTRL_CHARS_RE.sub("", s)
    s = s.strip()
    s = _WHITESPACE_RE.sub(" ", s)
    return s
