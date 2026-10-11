"""Versioned prompt templates (``prompts/*.md``) and the Fish text-template parser.

File format: YAML front matter (``name``, ``output``, ``schema``, ``source``, ``layout``,
``prefixes``, ``revision``) between ``---`` lines, then a ``<<<SYSTEM>>>`` part (stable, cached
by the provider), an optional ``<<<HISTORY>>>`` part and a ``<<<USER>>>`` part (volatile).
Placeholders are ``{{name}}``; a block ``{{#name}}...{{/name}}`` renders only when ``name`` is
truthy. Every placeholder must be supplied (``KeyError`` otherwise); substituted values are
never re-scanned.

Three-part templates (revision 3 of ``dealer_v1``, ``fish_v1``, ``fish_v1_json``): the
``<<<HISTORY>>>`` part is a header that ends with ``{{market_data}}``; :meth:`render_parts`
returns the header and one entry per round (oldest first) so that the provider can send one
content block per round and put a cache breakpoint on the last one. The history is
append-only: the parts for round t+1 extend those of round t until the memory window slides.
The plain-text user turn is ``"\n".join(history) + "\n" + user`` (:class:`PromptParts`).

Revisions: ``prompts/<name>.md`` is the current revision; earlier revisions live in
``prompts/revisions/<name>.r<k>.md`` (byte-identical to the file that was current, so their
version strings are unchanged) and load as ``load_template("<name>@r<k>")``.

Version string: ``"<name>@<first 12 hex of sha256(file, LF line endings)>"``; it goes into
``LLMRequest.prompt_version`` and therefore into cache keys and the decision log.

Notices (``prompts/notices/<name>.md``: YAML front matter, then the notice text) form a small
registry for ``InjectNotice.template``: :func:`resolve_notice` maps a registered name (e.g.
``monitoring_notice``) to its text and returns any other string unchanged (literal notice).
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field, replace
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel

from finswarm.agents.llm.schemas import SCHEMAS

PROMPTS_DIR = Path(__file__).parent / "prompts"
REVISIONS_DIR = PROMPTS_DIR / "revisions"
NOTICES_DIR = PROMPTS_DIR / "notices"
_NOTICE_NAME = re.compile(r"^[a-z][a-z0-9_]*$")
_SYSTEM = "<<<SYSTEM>>>"
_USER = "<<<USER>>>"
_HISTORY = "<<<HISTORY>>>"
_MARKET = "{{market_data}}"
_VAR = re.compile(r"\{\{([a-z_][a-z0-9_]*)\}\}")
_BLOCK = re.compile(r"\{\{#([a-z_][a-z0-9_]*)\}\}(.*?)\{\{/\1\}\}", re.S)


@dataclass(frozen=True)
class PromptParts:
    """A rendered prompt split at its cache boundaries.

    ``history`` is empty for two-part templates; otherwise it is the MARKET_DATA header
    followed by one entry per round, oldest first.
    """

    system: str
    history: list[str]
    user: str

    def render_pair(self) -> tuple[str, str]:
        return self.system, self.user_text()

    @property
    def history_text(self) -> str:
        return "\n".join(self.history)

    def user_text(self) -> str:
        """The user turn as one string (what providers without block caching receive)."""
        if not self.history:
            return self.user
        return self.history_text + "\n" + self.user

    def user_blocks(self) -> list[dict[str, str]]:
        """Text blocks (one per history entry, then the volatile part) whose concatenation is
        exactly :meth:`user_text`."""
        blocks = [{"type": "text", "text": h + "\n"} for h in self.history]
        blocks.append({"type": "text", "text": self.user})
        return blocks


@dataclass(frozen=True)
class PromptTemplate:
    name: str
    output: str  # fish_text | price_json | order_json | cda_json | judge_json
    schema_name: str | None
    system: str
    user: str
    sha256: str
    meta: dict[str, Any] = field(default_factory=dict, compare=False)
    history: str | None = None  # three-part templates: header ending with {{market_data}}
    key: str = ""  # load key ("dealer_v1" or "dealer_v1@r2"); defaults to name

    @property
    def version(self) -> str:
        return f"{self.name}@{self.sha256[:12]}"

    @property
    def load_key(self) -> str:
        return self.key or self.name

    @property
    def revision(self) -> int | None:
        rev = self.meta.get("revision")
        return int(rev) if rev is not None else None

    @property
    def has_history_block(self) -> bool:
        return self.history is not None

    @property
    def schema_cls(self) -> type[BaseModel] | None:
        return SCHEMAS.get(self.schema_name) if self.schema_name else None

    @property
    def is_json(self) -> bool:
        return self.output != "fish_text"

    def prefix(self, key: str) -> str:
        prefixes = self.meta.get("prefixes") or {}
        if key not in prefixes:
            raise KeyError(f"template {self.name!r} has no prefix {key!r}")
        return str(prefixes[key])

    def placeholders(self) -> set[str]:
        text = self.system + (self.history or "") + self.user
        return set(_VAR.findall(text)) | {m.group(1) for m in _BLOCK.finditer(text)}

    def render(self, variables: dict[str, Any]) -> tuple[str, str]:
        """``(system, user)`` as plain text; for a three-part template ``market_data`` is the
        rounds already joined by newlines (oldest first)."""
        if self.history is None:
            return _fill(self.system, variables).strip(), _fill(self.user, variables).strip()
        market = str(variables["market_data"])
        rounds = [market] if market else []
        return self.render_parts(variables, rounds).render_pair()

    def render_parts(
        self, variables: dict[str, Any], rounds: list[str] | None = None
    ) -> PromptParts:
        """Split rendering: system, history (header + ``rounds``) and volatile user part."""
        system = _fill(self.system, variables).strip()
        user = _fill(self.user, variables).strip()
        if self.history is None:
            return PromptParts(system, [], user)
        header = _fill(self.history[: -len(_MARKET)], variables).strip()
        return PromptParts(system, [header, *(rounds or [])], user)


def _fill(text: str, variables: dict[str, Any]) -> str:
    def block(m: re.Match[str]) -> str:
        name = m.group(1)
        if name not in variables:
            raise KeyError(f"missing template variable {name!r}")
        return m.group(2) if variables[name] else ""

    text = _BLOCK.sub(block, text)

    def var(m: re.Match[str]) -> str:
        name = m.group(1)
        if name not in variables:
            raise KeyError(f"missing template variable {name!r}")
        return str(variables[name])

    return _VAR.sub(var, text)


def parse_template(raw: str, name_hint: str = "") -> PromptTemplate:
    text = raw.replace("\r\n", "\n")
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
    if not text.startswith("---\n"):
        raise ValueError(f"template {name_hint!r}: missing front matter")
    _, front, body = text.split("---\n", 2)
    meta: dict[str, Any] = yaml.safe_load(front) or {}
    if _SYSTEM not in body or _USER not in body:
        raise ValueError(f"template {name_hint!r}: needs {_SYSTEM} and {_USER} markers")
    system = body.split(_SYSTEM, 1)[1].split(_USER, 1)[0]
    user = body.split(_USER, 1)[1]
    history: str | None = None
    if _HISTORY in system:
        system, history = system.split(_HISTORY, 1)
        history = history.strip("\n")
        if not history.endswith(_MARKET) or history.count(_MARKET) != 1:
            raise ValueError(f"template {name_hint!r}: {_HISTORY} must end with {_MARKET}")
        if _MARKET in system or _MARKET in user:
            raise ValueError(f"template {name_hint!r}: {_MARKET} belongs to {_HISTORY} only")
    return PromptTemplate(
        name=str(meta.get("name", name_hint)),
        output=str(meta.get("output", "price_json")),
        schema_name=meta.get("schema"),
        system=system.strip("\n"),
        user=user.strip("\n"),
        sha256=digest,
        meta=meta,
        history=history,
    )


@lru_cache(maxsize=32)
def load_template(name: str) -> PromptTemplate:
    """``<name>`` loads the current revision, ``<name>@r<k>`` an archived one."""
    if "@r" in name:
        base, rev = name.split("@r", 1)
        path = REVISIONS_DIR / f"{base}.r{rev}.md"
        if not rev.isdigit() or not path.is_file():
            raise FileNotFoundError(f"no revision {rev!r} of prompt {base!r} in {REVISIONS_DIR}")
        tpl = parse_template(path.read_bytes().decode("utf-8"), base)
        if tpl.revision != int(rev):
            raise ValueError(f"{path.name}: front matter says revision {tpl.revision}")
        return replace(tpl, key=name)
    path = PROMPTS_DIR / f"{name}.md"
    if not path.is_file():
        raise FileNotFoundError(f"no prompt template {name!r} in {PROMPTS_DIR}")
    return replace(parse_template(path.read_bytes().decode("utf-8"), name), key=name)


def list_revisions(name: str) -> list[str]:
    """Load keys of the archived revisions of ``name`` (``["dealer_v1@r2", ...]``)."""
    keys = [p.stem.replace(".r", "@r", 1) for p in REVISIONS_DIR.glob(f"{name}.r*.md")]
    return sorted(keys, key=lambda k: int(k.rsplit("@r", 1)[1]))


def list_templates() -> list[str]:
    return sorted(p.stem for p in PROMPTS_DIR.glob("*.md"))


# ------------------------------------------------------------------------------ notices
@dataclass(frozen=True)
class NoticeTemplate:
    name: str
    text: str
    sha256: str
    meta: dict[str, Any] = field(default_factory=dict, compare=False)

    @property
    def version(self) -> str:
        return f"{self.name}@{self.sha256[:12]}"


@lru_cache(maxsize=32)
def load_notice(name: str) -> NoticeTemplate:
    path = NOTICES_DIR / f"{name}.md"
    if not path.is_file():
        raise FileNotFoundError(f"no notice template {name!r} in {NOTICES_DIR}")
    text = path.read_bytes().decode("utf-8").replace("\r\n", "\n")
    if not text.startswith("---\n"):
        raise ValueError(f"notice {name!r}: missing front matter")
    _, front, body = text.split("---\n", 2)
    meta: dict[str, Any] = yaml.safe_load(front) or {}
    return NoticeTemplate(
        name=str(meta.get("name", name)),
        text=body.strip(),
        sha256=hashlib.sha256(text.encode("utf-8")).hexdigest(),
        meta=meta,
    )


def list_notices() -> list[str]:
    return sorted(p.stem for p in NOTICES_DIR.glob("*.md"))


def resolve_notice(template: str) -> str:
    """``InjectNotice.template``: a registered notice name -> its text; anything else is the
    literal notice text (fallback)."""
    key = template.strip()
    if _NOTICE_NAME.match(key) and (NOTICES_DIR / f"{key}.md").is_file():
        return load_notice(key).text
    return template


# ------------------------------------------------------------------ Fish text output
_HEADERS = {
    "observations": r"my\s+observations\s+and\s+thoughts",
    "plans": r"new\s+content\s+for\s+plans\.txt",
    "insights": r"new\s+content\s+for\s+insights\.txt",
    "price": r"my\s+chosen\s+price",
}
_HEADER_RE = re.compile(
    r"^[ \t>#*_-]*(?P<h>"
    + "|".join(f"(?P<{k}>{v})" for k, v in _HEADERS.items())
    + r")[ \t*_]*:?[ \t*_]*",
    re.I | re.M,
)
_PRICE_RE = re.compile(r"[-+]?\$?\s*(\d+(?:[.,]\d+)?)")
_PLACEHOLDER = re.compile(r"^\s*<\s*(fill in here|just the number, nothing else)\s*>\s*$", re.I)


def parse_fish_sections(text: str) -> dict[str, Any] | None:
    """Split a Fish-template answer into its four fields; ``None`` if no price is found.

    Tolerates markdown decoration (``**My chosen price:**``), content on the header line,
    and prose around the number ("I will set $1.85.").
    """
    matches = list(_HEADER_RE.finditer(text or ""))
    if not matches:
        return None
    out: dict[str, Any] = {}
    for i, m in enumerate(matches):
        key = next(k for k in _HEADERS if m.group(k))
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        body = text[m.end() : end].strip()
        if _PLACEHOLDER.match(body):
            body = ""
        out[key] = body  # a later repeat of a header wins (the final answer)
    price_text = out.get("price")
    if not price_text:
        return None
    pm = _PRICE_RE.search(price_text)
    if pm is None:
        return None
    out["price"] = float(pm.group(1).replace(",", "."))
    out.setdefault("observations", "")
    out.setdefault("plans", "")
    out.setdefault("insights", "")
    return out
