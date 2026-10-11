"""Provider adapters. Only this package imports provider SDKs.

``make_client`` builds the adapter named by a provider entry of ``configs/models.yaml``.
Keys come from :mod:`finswarm.llm.keys`; a missing key is reported by ``FatalLLMError``
at the first call (or skipped by the health check), never at import time.
"""

from __future__ import annotations

from typing import Any

from finswarm.llm.client import LLMClient
from finswarm.llm.keys import get_key
from finswarm.llm.pricing import Catalog, load_catalog
from finswarm.llm.timing import Clock


def make_client(
    provider: str,
    catalog: Catalog | None = None,
    *,
    clock: Clock | None = None,
    mock_policy: str = "static_nash",
    timeout_s: float | None = None,
    capture_limits: bool = False,
) -> LLMClient:
    cat = catalog or load_catalog()
    if provider not in cat.providers:
        raise KeyError(f"provider {provider!r} not in configs/models.yaml")
    spec = cat.providers[provider]
    timeout = timeout_s if timeout_s is not None else cat.defaults.timeout_s
    client: Any
    if spec.kind == "mock":
        from finswarm.llm.providers.mock import MockLLM

        client = MockLLM(mock_policy, clock=clock)
    elif spec.kind == "anthropic":
        from finswarm.llm.providers.anthropic_ import AnthropicClient

        client = AnthropicClient(
            api_key=get_key(spec.env_key),
            workspace_id=get_key("ANTHROPIC_WORKSPACE_ID"),
            catalog=cat,
            timeout_s=timeout,
            capture_limits=capture_limits,
        )
    elif spec.kind == "mistral":
        from finswarm.llm.providers.mistral_ import MistralClient

        client = MistralClient(
            api_key=get_key(spec.env_key), json_mode=spec.json_mode, timeout_s=timeout
        )
    elif spec.kind == "openai_compat":
        from finswarm.llm.providers.openai_compat import OpenAICompatClient

        if not spec.base_url:
            raise ValueError(f"provider {provider!r} needs a base_url")
        client = OpenAICompatClient(
            provider,
            spec.base_url,
            api_key=get_key(spec.env_key),
            json_mode=spec.json_mode,
            capabilities=spec.capabilities,
            timeout_s=timeout,
        )
    else:  # pragma: no cover - guarded by the Literal in ProviderSpec
        raise ValueError(spec.kind)
    return client


__all__ = ["make_client"]
