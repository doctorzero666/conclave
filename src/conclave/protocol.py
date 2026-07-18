"""审议协议数据结构 — 所有模块共享的 dataclass"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Optional


class RoundType(Enum):
    INITIAL = "initial"       # Round 1: 独立回答
    CRITIQUE = "critique"     # Round 2: 互相批评
    REVISE = "revise"         # Round 3: 修正答案


class ProviderRole(Enum):
    PARTICIPANT = "participant"
    JUDGE = "judge"


class SynthesisStatus(Enum):
    SUCCESS = "success"
    EMPTY = "empty"
    RETRY = "retry"
    FALLBACK = "fallback"


@dataclass
class ProviderResponse:
    """单个 Provider 的一次响应"""
    provider_name: str
    model: str
    text: str
    finish_reason: str = "stop"
    error: Optional[str] = None
    elapsed_ms: int = 0
    round_num: int = 1
    round_type: RoundType = RoundType.INITIAL
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())

    @property
    def ok(self) -> bool:
        return self.error is None and len(self.text) > 0


@dataclass
class StreamChunk:
    """流式输出的 token chunk"""
    provider_name: str
    delta: str
    is_done: bool = False
    finish_reason: Optional[str] = None
    error: Optional[str] = None


@dataclass
class DeliberationRound:
    """一轮审议的完整结果"""
    round_num: int
    round_type: RoundType
    responses: list[ProviderResponse] = field(default_factory=list)
    started_at: Optional[str] = None
    finished_at: Optional[str] = None

    @property
    def all_ok(self) -> bool:
        return all(r.ok for r in self.responses)

    @property
    def any_ok(self) -> bool:
        return any(r.ok for r in self.responses)


@dataclass
class SynthesisConsensus:
    """共识点"""
    point: str
    agreed_by: list[str]  # provider names
    confidence: str = "high"  # high / medium / low


@dataclass
class SynthesisDivergence:
    """分歧点"""
    point: str
    positions: dict[str, str]  # provider_name → position
    critical: bool = False


@dataclass
class SynthesisInsight:
    """独特洞察"""
    point: str
    provider_name: str
    category: str = ""  # 分类标签


@dataclass
class SynthesisReport:
    """合成报告"""
    consensus: list[SynthesisConsensus] = field(default_factory=list)
    divergences: list[SynthesisDivergence] = field(default_factory=list)
    insights: list[SynthesisInsight] = field(default_factory=list)
    executive_summary: str = ""
    judge_model: str = ""
    status: SynthesisStatus = SynthesisStatus.SUCCESS
    error: Optional[str] = None
    total_cost_usd: float = 0.0


@dataclass
class DeliberationResult:
    """一次完整审议的最终结果"""
    prompt: str
    rounds: list[DeliberationRound] = field(default_factory=list)
    synthesis: Optional[SynthesisReport] = None
    total_duration_ms: int = 0
    total_cost_usd: float = 0.0
    provider_count: int = 0
    round_count: int = 1
    failed_providers: list[str] = field(default_factory=list)

    @property
    def synthesis_enabled(self) -> bool:
        return self.synthesis is not None
