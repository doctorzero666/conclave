"""Test provider base and factory"""

import pytest
from conclave.config import ProviderConfig
from conclave.providers import create_provider
from conclave.providers.base import BaseProvider, ProviderCapabilities
from conclave.protocol import ProviderResponse


def test_mock_provider_invoke(mock_provider):
    resp = mock_provider.invoke("test prompt")
    assert isinstance(resp, ProviderResponse)
    assert resp.text == "Mock response"
    assert resp.ok is True
    assert mock_provider.call_count == 1


def test_mock_provider_stream(mock_provider):
    chunks = list(mock_provider.stream("test"))
    assert len(chunks) == 1
    assert chunks[0].delta == "Mock response"
    assert chunks[0].is_done is True


def test_mock_provider_capabilities(mock_provider):
    caps = mock_provider.capabilities()
    assert isinstance(caps, ProviderCapabilities)
    assert caps.supports_streaming is False


def test_create_cli_provider():
    config = ProviderConfig(
        name="test-cli", provider_type="cli", model="test",
        executable="echo", args_template=["{prompt}"],
    )
    provider = create_provider(config)
    assert isinstance(provider, BaseProvider)
    assert provider.name == "test-cli"


def test_create_provider_unknown_type():
    config = ProviderConfig(
        name="bad", provider_type="unknown", model="x",
    )
    with pytest.raises(ValueError, match="未知 provider_type"):
        create_provider(config)
