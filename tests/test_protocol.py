"""Test protocol data structures"""

from conclave.protocol import (
    DeliberationRound, ProviderResponse, RoundType,
    SynthesisReport, SynthesisConsensus, SynthesisDivergence,
    SynthesisInsight, SynthesisStatus, DeliberationResult,
)


class TestProviderResponse:
    def test_ok_when_no_error_and_has_text(self):
        resp = ProviderResponse(
            provider_name="test", model="m", text="hello",
        )
        assert resp.ok is True

    def test_not_ok_when_error(self):
        resp = ProviderResponse(
            provider_name="test", model="m", text="", error="timeout",
        )
        assert resp.ok is False

    def test_not_ok_when_empty_text(self):
        resp = ProviderResponse(
            provider_name="test", model="m", text="",
        )
        assert resp.ok is False


class TestDeliberationRound:
    def test_all_ok(self, sample_responses):
        round_ = DeliberationRound(
            round_num=1, round_type=RoundType.INITIAL,
            responses=sample_responses,
        )
        assert round_.all_ok is True
        assert round_.any_ok is True

    def test_not_all_ok_with_error(self, sample_responses):
        sample_responses[1] = ProviderResponse(
            provider_name="bad", model="m", text="", error="fail",
        )
        round_ = DeliberationRound(
            round_num=1, round_type=RoundType.INITIAL,
            responses=sample_responses,
        )
        assert round_.all_ok is False
        assert round_.any_ok is True


class TestSynthesisReport:
    def test_empty_report(self):
        report = SynthesisReport()
        assert report.status == SynthesisStatus.SUCCESS
        assert len(report.consensus) == 0
        assert report.executive_summary == ""

    def test_with_data(self):
        report = SynthesisReport(
            executive_summary="All agree on Redis",
            consensus=[SynthesisConsensus(
                point="Use Redis", agreed_by=["a", "b"], confidence="high",
            )],
            divergences=[SynthesisDivergence(
                point="Which library", positions={"a": "redis-py", "b": "aredis"},
                critical=False,
            )],
        )
        assert len(report.consensus) == 1
        assert len(report.divergences) == 1
        assert report.consensus[0].confidence == "high"


class TestDeliberationResult:
    def test_basic_result(self, sample_responses):
        result = DeliberationResult(
            prompt="test prompt",
            rounds=[DeliberationRound(
                round_num=1, round_type=RoundType.INITIAL,
                responses=sample_responses,
            )],
            provider_count=3,
            round_count=1,
        )
        assert result.round_count == 1
        assert result.provider_count == 3
        assert result.synthesis_enabled is False

    def test_with_synthesis(self, sample_responses):
        report = SynthesisReport(executive_summary="test")
        result = DeliberationResult(
            prompt="test",
            rounds=[DeliberationRound(
                round_num=1, round_type=RoundType.INITIAL,
                responses=sample_responses,
            )],
            synthesis=report,
        )
        assert result.synthesis_enabled is True

    def test_failed_providers_tracking(self, sample_responses):
        result = DeliberationResult(
            prompt="test",
            rounds=[DeliberationRound(
                round_num=1, round_type=RoundType.INITIAL,
                responses=sample_responses,
            )],
            failed_providers=["bad-provider"],
        )
        assert "bad-provider" in result.failed_providers
