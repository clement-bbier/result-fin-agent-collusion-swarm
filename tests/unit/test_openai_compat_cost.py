"""OpenAI-compatible adapter prices completed calls from the catalogue (paid OpenRouter lane)."""

from __future__ import annotations

from finswarm.llm.providers.openai_compat import OpenAICompatClient


def test_paid_openrouter_model_is_priced() -> None:
    usd = OpenAICompatClient._cost("meta-llama/llama-3.3-70b-instruct", 1000, 300)
    assert abs(usd - (1000 * 0.22 + 300 * 0.50) / 1e6) < 1e-9


def test_free_and_unknown_models_cost_zero() -> None:
    assert OpenAICompatClient._cost("google/gemma-4-31b-it:free", 1000, 300) == 0.0
    assert OpenAICompatClient._cost("no-such-model", 1000, 300) == 0.0
