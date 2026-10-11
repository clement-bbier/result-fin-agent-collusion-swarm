"""Helpers to build and parse judge requests (``judge_v1``, rubric 1-4), and the wiring of the
registered judge configuration (``configs/sprint/tier1/JD-S.yaml``) to a ``metrics.judge``
``score_fn``.

The judge sees reasoning fields only (observations, PLANS.txt, INSIGHTS.txt, messages).
Callers (``finswarm.metrics.judge``) submit the request through the lane scheduler:

    spec = judge_spec_from_config(load_campaign_config("configs/sprint/tier1/JD-S.yaml"))
    judge = select_judge(spec, family="anthropic")          # the judge of Haiku runs
    score_fn = make_score_fn(scheduler, judge, run_id=run_id)
    scores = await metrics.judge.score_items(items, score_fn)

``JudgeSpec`` reads the ``judge: ...`` section of the campaign when the frozen schema carries
one, else the ``tags`` of JD-S (``judge_model``, ``judge_model_sonnet_arm``,
``judge_lane_sonnet_arm``, ``judge_backup_sonnet_arm``, ``judge_rubric``,
``scratchpads_per_window``). One scratchpad = one judge call (``score_fn(text)``); a parse
failure or refusal scores ``None`` (``metrics.judge`` logs it as an error row).
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass
from typing import Any

from finswarm.agents.llm.schemas import JudgeScore, strict_json_schema
from finswarm.agents.llm.templates import load_template
from finswarm.llm.client import LLMRequest
from finswarm.llm.parse import extract_json, validate
from finswarm.llm.pricing import Catalog, load_catalog

#: the Sonnet 5.5 capability arm: its runs are judged by the third family (lane L)
SONNET_ARM_FAMILIES = ("anthropic_sonnet", "sonnet")
JUDGE_TAG_KEYS = {
    "model": "judge_model",
    "model_sonnet_arm": "judge_model_sonnet_arm",
    "lane_sonnet_arm": "judge_lane_sonnet_arm",
    "backup_sonnet_arm": "judge_backup_sonnet_arm",
    "rubric": "judge_rubric",
    "scratchpads_per_window": "scratchpads_per_window",
}


@dataclass(frozen=True)
class Judge:
    """One judge model on one lane (catalogue id, resolved from an alias)."""

    model: str
    lane: str
    provider: str


@dataclass(frozen=True)
class JudgeSpec:
    """The registered judge configuration (JD-S)."""

    model: str  # judge of Haiku (and Mistral) runs, e.g. claude-sonnet-5-5
    model_sonnet_arm: str  # judge of Sonnet-arm runs (third family), e.g. llama-70b
    lane_sonnet_arm: str | None = None  # catalogue lane of ``model_sonnet_arm`` (else looked up)
    backup_sonnet_arm: str | None = None  # fallback judge when lane L is exhausted
    rubric: str = "judge_v1"
    scratchpads_per_window: int = 8
    lane: str | None = None  # catalogue lane of ``model`` (else looked up)


def judge_spec_from_tags(tags: dict[str, str]) -> JudgeSpec:
    """The spec from JD-S ``tags`` (every key optional except ``judge_model``)."""
    t = {k: tags.get(v) for k, v in JUDGE_TAG_KEYS.items()}
    if not t["model"]:
        raise ValueError("judge config: tags.judge_model is required")
    return JudgeSpec(
        model=str(t["model"]),
        model_sonnet_arm=str(t["model_sonnet_arm"] or t["model"]),
        lane_sonnet_arm=t["lane_sonnet_arm"] or None,
        backup_sonnet_arm=t["backup_sonnet_arm"] or None,
        rubric=str(t["rubric"] or "judge_v1"),
        scratchpads_per_window=int(t["scratchpads_per_window"] or 8),
        lane=tags.get("judge_lane") or None,
    )


def judge_spec_from_config(cc: Any) -> JudgeSpec:
    """The spec from a campaign config: its ``judge`` section when the schema has one, else the
    base run's ``tags`` (JD-S as authored against the frozen schema)."""
    section = getattr(cc, "judge", None)
    if section is not None:
        data = section.model_dump() if hasattr(section, "model_dump") else dict(section)
        return JudgeSpec(**{k: v for k, v in data.items() if k in JudgeSpec.__dataclass_fields__})
    base = getattr(cc, "base", cc)
    return judge_spec_from_tags(dict(base.tags))


def select_judge(spec: JudgeSpec, family: str, catalog: Catalog | None = None) -> Judge:
    """The judge of runs whose dealers are ``family`` (``llm.lane`` or slot family): Sonnet-arm
    runs get the third-family judge, every other run the primary one. Aliases are resolved and
    the lane is the one serving the model unless the spec names it."""
    cat = catalog if catalog is not None else load_catalog()
    sonnet_arm = family in SONNET_ARM_FAMILIES
    model = cat.resolve_model(spec.model_sonnet_arm if sonnet_arm else spec.model)
    lane = (spec.lane_sonnet_arm if sonnet_arm else spec.lane) or cat.lane_for_model(model)
    if lane is None:
        raise ValueError(f"judge model {model!r}: no catalogue lane serves it")
    spec_model = cat.models.get(model)
    if spec_model is None:
        raise ValueError(f"judge model {model!r} is not in configs/models.yaml")
    return Judge(model=model, lane=lane, provider=spec_model.provider)


ScoreFn = Callable[[str], Awaitable[float | None]]


def make_score_fn(
    scheduler: Any,
    judge: Judge,
    *,
    run_id: str = "judge",
    rubric: str = "judge_v1",
    seed: int | None = None,
    max_tokens: int = 2048,
) -> ScoreFn:
    """``metrics.judge`` scorer: one ``judge_v1`` request per scratchpad through ``scheduler``
    (``await scheduler.submit(req, lane)``); the rubric score 1-4, or ``None`` when the judge
    refused or returned no parsable score. Anthropic judges keep ``effort: low`` and no
    temperature; other providers get a fixed temperature 0 for reproducible scoring."""
    if rubric != "judge_v1":
        raise ValueError(f"unknown judge rubric {rubric!r}")
    anthropic = judge.provider == "anthropic"

    async def score_fn(text: str) -> float | None:
        req = build_judge_request(
            [text],
            provider=judge.provider,
            model=judge.model,
            max_tokens=max_tokens,
            effort="low" if anthropic else None,
            temperature=None if anthropic else 0.0,
            seed=seed,
            tags={"run_id": run_id, "agent": "judge"},
        )
        resp = await scheduler.submit(req, judge.lane)
        if resp is None or resp.refusal or resp.error:
            return None
        parsed = parse_judge(resp.text)
        return None if parsed is None else float(parsed.score)

    return score_fn


def format_excerpts(excerpts: Sequence[tuple[int, str] | str]) -> str:
    blocks = []
    for i, item in enumerate(excerpts):
        rnd, text = item if isinstance(item, tuple) else (i + 1, item)
        blocks.append(f"Round {rnd}:\n{text.strip()}")
    return "\n\n".join(blocks)


def build_judge_request(
    excerpts: Sequence[tuple[int, str] | str],
    *,
    provider: str,
    model: str,
    max_tokens: int = 2048,
    effort: str | None = "low",
    temperature: float | None = None,
    seed: int | None = None,
    tags: dict[str, str] | None = None,
) -> LLMRequest:
    tpl = load_template("judge_v1")
    system, user = tpl.render({"n_excerpts": len(excerpts), "excerpts": format_excerpts(excerpts)})
    return LLMRequest(
        provider=provider,
        model=model,
        system=system,
        messages=[{"role": "user", "content": user}],
        schema=strict_json_schema(JudgeScore),
        max_tokens=max_tokens,
        temperature=temperature,
        effort=effort,
        seed=seed,
        prompt_version=tpl.version,
        tags={**(tags or {}), "output_format": tpl.output},
    )


def parse_judge(text: str) -> JudgeScore | None:
    data = extract_json(text)
    if data is None:
        return None
    score, _ = validate(data, JudgeScore)
    return score
