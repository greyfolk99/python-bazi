from __future__ import annotations

import os

_VALID_LANGS = {"zh", "ko", "en"}


class BaziConfig:
    """전역 설정. bazi.config.lang 으로 접근."""

    _VALID_LANGS = _VALID_LANGS

    def __init__(self) -> None:
        env = os.environ.get("BAZI_LANG", "zh")
        object.__setattr__(self, "lang", env if env in _VALID_LANGS else "zh")

    def __setattr__(self, name: str, value: object) -> None:
        if name == "lang":
            if value not in _VALID_LANGS:
                raise ValueError(f"lang must be one of {_VALID_LANGS}")
        super().__setattr__(name, value)

    def __repr__(self) -> str:
        return f"BaziConfig(lang={self.lang!r})"


config = BaziConfig()
