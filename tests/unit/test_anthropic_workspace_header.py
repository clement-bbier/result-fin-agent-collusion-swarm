"""The Anthropic client forwards an optional workspace id as a default header.

Keys that are not scoped to a workspace are rejected by the API (400) unless the request
carries ``anthropic-workspace-id``; workspace-scoped keys ignore the header.
"""

from __future__ import annotations

import pytest

anthropic = pytest.importorskip("anthropic")

from finswarm.llm.providers.anthropic_ import AnthropicClient  # noqa: E402


def test_workspace_header_added_when_configured() -> None:
    client = AnthropicClient(api_key="sk-ant-test", workspace_id="wrkspc_test")
    assert client._client.default_headers.get("anthropic-workspace-id") == "wrkspc_test"


def test_no_workspace_header_by_default() -> None:
    client = AnthropicClient(api_key="sk-ant-test")
    assert "anthropic-workspace-id" not in client._client.default_headers
