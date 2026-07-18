"""Test fixtures — shared test infrastructure"""

import os
import sys
import pytest

# Ensure src is importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))


@pytest.fixture
def sample_responses():
    """模拟三个 provider 的回答"""
    from conclave.protocol import ProviderResponse, RoundType
    return [
        ProviderResponse(
            provider_name="test-claude", model="claude-sonnet-4",
            text="Use Redis for caching. It's fast, battle-tested, and has great Python support.",
            elapsed_ms=1500, round_type=RoundType.INITIAL,
        ),
        ProviderResponse(
            provider_name="test-codex", model="gpt-5.6-sol",
            text="Use in-memory cache for single-server. Add Redis only when you need multi-server.",
            elapsed_ms=800, round_type=RoundType.INITIAL,
        ),
        ProviderResponse(
            provider_name="test-deepseek", model="deepseek-v3",
            text="Start with functools.lru_cache. Graduate to Redis when you need persistence.",
            elapsed_ms=1200, round_type=RoundType.INITIAL,
        ),
    ]


@pytest.fixture
def mock_provider():
    """创建一个 Mock provider"""
    from conclave.protocol import ProviderResponse
    from conclave.providers.base import BaseProvider, ProviderCapabilities

    class MockProvider(BaseProvider):
        def __init__(self, name="mock", model="mock-model", text="Mock response"):
            super().__init__(name, model)
            self._text = text
            self.call_count = 0

        def capabilities(self):
            return ProviderCapabilities(supports_streaming=False, models=[self.model])

        def invoke(self, prompt, system_prompt=None, timeout=180):
            self.call_count += 1
            return ProviderResponse(
                provider_name=self.name, model=self.model,
                text=self._text, elapsed_ms=10,
            )

    return MockProvider()
