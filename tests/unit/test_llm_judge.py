"""JD-S judge wiring (BACKLOG 8): the registered judge configuration becomes a
``metrics.judge`` ``score_fn`` through the lane scheduler."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest

from finswarm.agents.llm.judge import (
    Judge,
    JudgeSpec,
    judge_spec_from_config,
    judge_spec_from_tags,
    make_score_fn,
    select_judge,
)
from finswarm.config.load import load_campaign_config
from finswarm.config.schema import JudgeConfig
from finswarm.llm.client import LLMResponse
from finswarm.llm.pricing import load_catalog

ROOT = Path(__file__).resolve().parents[2]
JD_S = ROOT / "configs" / "sprint" / "tier1" / "JD-S.yaml"


def test_spec_from_the_registered_jd_s_tags() -> None:
    cc = load_campaign_config(JD_S)
    spec = judge_spec_from_config(cc)
    assert spec == judge_spec_from_tags(dict(cc.base.tags))
    assert spec.model == "claude-sonnet-5-5" and spec.model_sonnet_arm == "llama-70b"
    assert spec.lane_sonnet_arm == "openrouter_llama"
    assert spec.backup_sonnet_arm == "gemini-3.5-flash-lite"
    assert spec.rubric == "judge_v1" and spec.scratchpads_per_window == 8
    with pytest.raises(ValueError):
        judge_spec_from_tags({"arm": "judge"})  # no judge_model


def test_spec_prefers_the_schema_section_when_present() -> None:
    cc = load_campaign_config(JD_S)
    section = JudgeConfig(
        model="claude-sonnet-5-5",
        model_sonnet_arm="llama-70b",
        lane_sonnet_arm="openrouter_llama",
        scratchpads_per_window=4,
    )
    spec = judge_spec_from_config(cc.model_copy(update={"judge": section}))
    assert spec.scratchpads_per_window == 4 and spec.model_sonnet_arm == "llama-70b"
    assert spec.rubric == "judge_v1"


def test_select_judge_routes_sonnet_arm_runs_to_the_third_family() -> None:
    cat = load_catalog()
    spec = judge_spec_from_config(load_campaign_config(JD_S))
    haiku = select_judge(spec, "anthropic_haiku", cat)
    assert haiku == Judge(model="claude-sonnet-5-5", lane="anthropic_sonnet", provider="anthropic")
    assert select_judge(spec, "mistral_free", cat) == haiku  # Mistral runs: the primary judge
    # A-005: Llama-family runs (second family on lane L) are judged by Sonnet, never by the
    # family being judged; only the Sonnet arm goes to the Llama judge
    for family in ("llama", "openrouter_llama", "openrouter"):
        assert select_judge(spec, family, cat) == haiku
    sonnet = select_judge(spec, "anthropic_sonnet", cat)
    assert sonnet.model == cat.resolve_model("llama-70b") == "meta-llama/llama-3.3-70b-instruct"
    assert (sonnet.lane, sonnet.provider) == ("openrouter_llama", "openrouter")
    with pytest.raises(ValueError):
        select_judge(JudgeSpec(model="no-such-model", model_sonnet_arm="x"), "anthropic_haiku", cat)


class _Scheduler:
    """Records the submitted requests and answers with canned responses."""

    def __init__(self, texts: list[str], *, refusal: bool = False) -> None:
        self.texts = list(texts)
        self.refusal = refusal
        self.calls: list[tuple[object, str]] = []

    async def submit(self, req, lane: str) -> LLMResponse:
        self.calls.append((req, lane))
        text = self.texts.pop(0) if self.texts else ""
        return LLMResponse(text=text, refusal=self.refusal, output_tokens=12)


def _score(score: int) -> str:
    return json.dumps({"score": score, "rationale": "r", "evidence": ""})


def test_score_fn_scores_one_scratchpad_per_call() -> None:
    sched = _Scheduler([_score(3), "not json at all", _score(1)])
    judge = Judge(model="claude-sonnet-5-5", lane="anthropic_sonnet", provider="anthropic")
    fn = make_score_fn(sched, judge, run_id="TR-H__s0__abc")

    async def run():
        return [await fn("[PLANS]\nmatch rivals"), await fn("[PLANS]\nx"), await fn("[PLANS]\ny")]

    assert asyncio.run(run()) == [3.0, None, 1.0]
    assert [lane for _, lane in sched.calls] == ["anthropic_sonnet"] * 3
    req = sched.calls[0][0]
    assert (req.provider, req.model) == ("anthropic", "claude-sonnet-5-5")
    assert req.prompt_version.startswith("judge_v1@")
    assert (
        req.effort == "low" and req.temperature is None and req.max_tokens >= 2048
    )  # Anthropic lane
    assert req.tags["run_id"] == "TR-H__s0__abc" and req.tags["agent"] == "judge"
    assert "match rivals" in req.messages[0]["content"]


def test_score_fn_third_family_params_and_refusals() -> None:
    sched = _Scheduler([_score(4)], refusal=True)
    judge = Judge(
        model="meta-llama/llama-3.3-70b-instruct", lane="openrouter_llama", provider="openrouter"
    )
    fn = make_score_fn(sched, judge, seed=7)
    assert asyncio.run(fn("[INSIGHTS]\nz")) is None  # refusal scores None
    req = sched.calls[0][0]
    assert req.effort is None and req.temperature == 0.0 and req.seed == 7
    with pytest.raises(ValueError):
        make_score_fn(sched, judge, rubric="judge_v2")
