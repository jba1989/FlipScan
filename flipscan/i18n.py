"""Display-language switch for user-facing backend messages.

Source strings are written in Traditional Chinese and double as catalog keys:
``tr("已匯入 {0} 頁", n)`` looks the template up in ``locales/<lang>.json``
and formats it positionally. A missing entry falls back to the Chinese
template, so an untranslated message still reads correctly — the catalog
coverage test is what keeps that from happening silently.

The language is process-wide (FlipScan is a single-user local app): the GUI
sets it when the viewer picks one, the CLI takes FLIPSCAN_LANG / LANG.
Messages are rendered when emitted, so job log lines already written keep the
language that was active at the time.
"""
from __future__ import annotations

import json
import os
from functools import cache
from pathlib import Path

ZH = "zh-Hant"
EN = "en"
LANGUAGES = (ZH, EN)

_LOCALES = Path(__file__).parent / "locales"
_lang: str | None = None


def normalize(tag: str | None) -> str | None:
    """Map a locale tag ("zh_TW.UTF-8", "zh-Hant", "en-US") to a supported
    language, or None when it names neither."""
    if not tag:
        return None
    t = tag.strip().lower().replace("_", "-")
    if t.startswith("zh"):
        return ZH
    if t.startswith("en"):
        return EN
    return None


def _from_env() -> str:
    for var in ("FLIPSCAN_LANG", "LC_ALL", "LC_MESSAGES", "LANG"):
        lang = normalize(os.environ.get(var))
        if lang:
            return lang
    return ZH


def get_language() -> str:
    global _lang
    if _lang is None:
        _lang = _from_env()
    return _lang


def set_language(tag: str) -> str:
    """Switch the display language; raises ValueError for an unsupported tag."""
    global _lang
    lang = normalize(tag)
    if lang is None:
        raise ValueError(f"unsupported language: {tag!r}")
    _lang = lang
    return lang


@cache
def catalog(lang: str) -> dict[str, str]:
    if lang == ZH:
        return {}
    path = _LOCALES / f"{lang}.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def tr(template: str, *args: object) -> str:
    """Render ``template`` (the Chinese source text) in the current language.

    Placeholders are positional ({0}, {1!r}, {2:.1f}); only the template is
    parsed as a format string, never the argument values."""
    text = catalog(get_language()).get(template, template)
    return text.format(*args) if args else text
