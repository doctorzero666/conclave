"""Test engine with mock providers"""

import pytest
from conclave.protocol import RoundType, SynthesisStatus, DeliberationResult
from conclave.engine import DeliberationEngine


def test_engine_one_round(mock_provider):
    engine = DeliberationEngine(providers=[mock_provider])
    result = engine.run("test prompt", rounds=1)

    assert isinstance(result, DeliberationResult)
    assert result.round_count == 1
    assert len(result.rounds) == 1
    assert result.rounds[0].round_type == RoundType.INITIAL
    assert mock_provider.call_count == 1


def test_engine_two_rounds(mock_provider):
    mock_provider.name = "participant-a"
    engine = DeliberationEngine(providers=[mock_provider])
    result = engine.run("test prompt", rounds=2)

    assert result.round_count == 2
    assert len(result.rounds) == 2
    # Round 1 + Round 2 (critique) = 2 calls
    assert mock_provider.call_count == 2


def test_engine_synthesis_disabled(mock_provider):
    engine = DeliberationEngine(providers=[mock_provider])
    result = engine.run("test", rounds=3, synthesis_enabled=False)

    assert result.round_count == 3
    assert result.synthesis is None


def test_engine_preserves_prompt(mock_provider):
    engine = DeliberationEngine(providers=[mock_provider])
    result = engine.run("What is 2+2?", rounds=1)

    assert result.prompt == "What is 2+2?"
