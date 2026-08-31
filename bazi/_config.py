from __future__ import annotations

import os

_VALID_LANGS = {"zh", "ko", "en"}
_lang: str = os.environ.get("BAZI_LANG", "zh")

if _lang not in _VALID_LANGS:
    _lang = "zh"


def get_lang() -> str:
    return _lang


def set_lang(lang: str) -> None:
    global _lang
    if lang not in _VALID_LANGS:
        raise ValueError(f"lang must be one of {_VALID_LANGS}")
    _lang = lang
