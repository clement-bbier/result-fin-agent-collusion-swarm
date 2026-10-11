"""API keys: the only module that reads ``.env`` (python-dotenv). Keys are never logged.

Other modules ask for a key by environment-variable name and receive the value or
``None``. Use :func:`redact` whenever a string that might contain a key is printed.
"""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

_LOADED = False


def load_env(path: str | Path | None = None, *, force: bool = False) -> None:
    """Load ``.env`` once (process environment variables win over the file)."""
    global _LOADED
    if _LOADED and not force:
        return
    if path is not None:
        load_dotenv(Path(path), override=False)
    else:
        # Search the working directory and its parents (repo root when run from a subdir).
        for parent in (Path.cwd(), *Path.cwd().parents):
            candidate = parent / ".env"
            if candidate.is_file():
                load_dotenv(candidate, override=False)
                break
    _LOADED = True


def get_key(env_name: str | None) -> str | None:
    """Return the key stored in ``env_name`` (after loading ``.env``), or ``None``."""
    if not env_name:
        return None
    load_env()
    value = os.environ.get(env_name, "").strip()
    return value or None


def key_present(env_name: str | None) -> bool:
    return get_key(env_name) is not None


def redact(text: str) -> str:
    """Replace every known key value that appears in ``text`` with ``***``."""
    out = text
    for name in (
        "ANTHROPIC_API_KEY",
        "MISTRAL_API_KEY",
        "GEMINI_API_KEY",
        "OPENROUTER_API_KEY",
    ):
        value = os.environ.get(name, "").strip()
        if len(value) >= 8 and value in out:
            out = out.replace(value, "***")
    return out
