"""Test synthesizer"""

from conclave.protocol import (
    DeliberationRound, ProviderResponse, RoundType, SynthesisStatus,
)
from conclave.synthesizer import Synthesizer


def test_synthesizer_fallback(mock_provider):
    """当 judge 返回非 JSON 时，应触发 fallback"""
    from conclave.providers.base import BaseProvider, ProviderCapabilities

    class BadJSONProvider(BaseProvider):
        def __init__(self):
            super().__init__("judge", "test-model")
        def capabilities(self):
            return ProviderCapabilities()
        def invoke(self, prompt, system_prompt=None, timeout=180):
            return ProviderResponse(
                provider_name="judge", model="test-model",
                text="Just some random text, not JSON at all",
            )

    judge = BadJSONProvider()
    synth = Synthesizer(judge)

    rounds = [DeliberationRound(
        round_num=1, round_type=RoundType.INITIAL,
        responses=[ProviderResponse(
            provider_name="a", model="m",
            text="Use Redis",
        )]
    )]

    report = synth.synthesize("test prompt", rounds)
    assert report.status == SynthesisStatus.FALLBACK
    assert len(report.executive_summary) > 0


def test_parse_json_with_markdown():
    synth = Synthesizer(None)  # judge=None, won't be called
    text = """```json
{"executive_summary": "test", "consensus": [], "divergences": [], "insights": []}
```"""
    result = synth._parse_json(text)
    assert result is not None
    assert result["executive_summary"] == "test"


def test_parse_json_direct():
    synth = Synthesizer(None)
    text = '{"executive_summary": "hello", "consensus": [], "divergences": [], "insights": []}'
    result = synth._parse_json(text)
    assert result is not None
    assert result["executive_summary"] == "hello"
