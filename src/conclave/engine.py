"""多轮审议协议引擎 V3 — 协调 provider 运行完整审议生命周期。

审议流程：
  rounds=1: 所有 provider 并行独立回答（INITIAL）
  rounds=2: + critique 轮（CRITIQUE：各 model 批评其他模型回答）
  rounds=3: + revise 轮（REVISE：基于批评改进） + 语义合成（synthesis）

设计要点：
- 纯同步，不依赖 asyncio
- 使用 Scheduler 进行并行调用
- 每一轮通过 on_round_callback 通知外部
- 支持 stream_callback 实时输出
- max_cost / timeout 保护
"""

import time
from datetime import datetime, timezone
from typing import Callable, Optional

from .protocol import (
    DeliberationResult,
    DeliberationRound,
    ProviderResponse,
    RoundType,
    SynthesisReport,
)
from .providers.base import BaseProvider
from .scheduler import Scheduler
from .synthesizer import Synthesizer


# ── Round 专用的 system prompt ────────────────────────────────────

INITIAL_SYSTEM_PROMPT = (
    "你是一个AI助手，请对用户的问题给出全面、深入的回答。"
    "不要拘泥于格式，重点放在内容质量上。"
)

CRITIQUE_SYSTEM_PROMPT = (
    "你是一个审议参与者。请仔细阅读其他模型对同一问题的回答，"
    "然后对这些回答进行批判性分析：指出逻辑漏洞、事实错误、"
    "遗漏的重要视角，以及可以改进的地方。保持建设性和客观性。"
)

REVISE_SYSTEM_PROMPT = (
    "你是一个审议参与者。你已经看过了其他模型的回答和批评意见。"
    "现在请基于这些反馈，对你自己最初的观点进行修正和改进。"
    "你可以坚持原来的观点（如果它被证明是正确的），"
    "也可以吸收他人的合理建议来完善你的答案。"
    "请给出你最终的、经过审议打磨的回答。"
)


# ── Engine ─────────────────────────────────────────────────────────

class DeliberationEngine:
    """多轮审议协议引擎。

    使用方式:
        engine = DeliberationEngine(providers=[claude, codex, gpt])
        result = engine.run("什么是量子计算？", rounds=3)
        print(result.synthesis.executive_summary)
    """

    def __init__(
        self,
        providers: list[BaseProvider],
        judge_provider: Optional[BaseProvider] = None,
        config=None,
    ):
        """
        Args:
            providers: 参与审议的 provider 列表（participant 角色）
            judge_provider: 用于合成报告的 judge provider（None 则不合成）
            config: Config 对象（可选，用于读取 max_workers 等设置）
        """
        self.providers = providers
        self.judge_provider = judge_provider
        self.config = config

        max_workers = getattr(config, "max_workers", 8) if config else 8
        self._scheduler = Scheduler(max_workers=max_workers)
        self._synthesizer: Optional[Synthesizer] = None
        if judge_provider:
            self._synthesizer = Synthesizer(judge_provider)

    # ── 主入口 ────────────────────────────────────────────────────

    def run(
        self,
        prompt: str,
        rounds: int = 1,
        stream_callback: Optional[Callable] = None,
        on_round_callback: Optional[Callable] = None,
        max_cost: float = 0,
        timeout: int = 180,
        synthesis_enabled: bool = True,
    ) -> DeliberationResult:
        """运行完整的多轮审议协议。

        Args:
            prompt: 用户问题
            rounds: 审议轮数（1-3）
                - 1: 仅初始回答
                - 2: 初始 + 批评
                - 3: 初始 + 批评 + 修订 + 合成
            stream_callback: 流式回调 callback(provider_name, text_chunk)
            on_round_callback: 每轮完成回调 callback(DeliberationRound)
            max_cost: 最大费用（美元），0=不限制（当前为占位）
            timeout: 每轮超时（秒）

        Returns:
            DeliberationResult 包含所有轮次和可选的合成报告
        """
        t_start = time.monotonic()
        all_rounds: list[DeliberationRound] = []
        failed_providers: set[str] = set()
        total_cost = 0.0

        rounds = max(1, min(rounds, 3))

        # ── Round 1: 独立回答 ──
        r1 = self._run_initial_round(prompt, timeout, stream_callback)
        all_rounds.append(r1)
        self._track_failed(r1, failed_providers)
        if on_round_callback:
            on_round_callback(r1)

        if rounds >= 2:
            # ── Round 2: 互相批评 ──
            r2 = self._run_critique_round(prompt, all_rounds, timeout, stream_callback)
            all_rounds.append(r2)
            self._track_failed(r2, failed_providers)
            if on_round_callback:
                on_round_callback(r2)

        if rounds >= 3:
            # ── Round 3: 修订 ──
            r3 = self._run_revise_round(prompt, all_rounds, timeout, stream_callback)
            all_rounds.append(r3)
            self._track_failed(r3, failed_providers)
            if on_round_callback:
                on_round_callback(r3)

        # ── 合成 ──
        synthesis: Optional[SynthesisReport] = None
        if synthesis_enabled and rounds >= 3 and self._synthesizer is not None:
            synthesis = self._synthesize(prompt, all_rounds)
        elif synthesis_enabled and self._synthesizer is not None:
            # rounds=1 或 2 时也合成（可选择）
            synthesis = self._synthesize(prompt, all_rounds)

        total_ms = int((time.monotonic() - t_start) * 1000)

        return DeliberationResult(
            prompt=prompt,
            rounds=all_rounds,
            synthesis=synthesis,
            total_duration_ms=total_ms,
            total_cost_usd=total_cost,
            provider_count=len(self.providers),
            round_count=len(all_rounds),
            failed_providers=sorted(failed_providers),
        )

    # ── 各轮实现 ──────────────────────────────────────────────────

    def _run_initial_round(
        self,
        prompt: str,
        timeout: int,
        stream_callback: Optional[Callable] = None,
    ) -> DeliberationRound:
        """Round 1: 所有 provider 独立回答同一个问题。"""
        return self._scheduler.run_parallel(
            providers=self.providers,
            prompt=prompt,
            system_prompt=INITIAL_SYSTEM_PROMPT,
            timeout=timeout,
            round_num=1,
            round_type=RoundType.INITIAL,
            stream_callback=stream_callback,
        )

    def _run_critique_round(
        self,
        prompt: str,
        prev_rounds: list[DeliberationRound],
        timeout: int,
        stream_callback: Optional[Callable] = None,
    ) -> DeliberationRound:
        """Round 2: 每个 provider 批评其他模型的 Round 1 回答。

        使用 invoke_with_context 将其他模型的回答作为上下文传入。
        """
        context = prev_rounds[0].responses  # Round 1 的回答

        # 为每个 provider 构建带上下文的调用
        # 用 invoke_with_context 让 provider 看到其他模型的回答
        import threading
        from concurrent.futures import ThreadPoolExecutor, as_completed

        # 这里直接用 scheduler 的简化路径：为每个 provider 准备带上下文的 system prompt
        # 使用 BaseProvider.invoke_with_context
        started_at = datetime.now(timezone.utc).isoformat()
        providers_list = list(self.providers)
        n = len(providers_list)
        responses: list[Optional[ProviderResponse]] = [None] * n
        lock = threading.Lock()

        with ThreadPoolExecutor(max_workers=min(self._scheduler.max_workers, n)) as ex:
            futures = {}
            for i, provider in enumerate(providers_list):
                fut = ex.submit(
                    provider.invoke_with_context,
                    prompt, context, CRITIQUE_SYSTEM_PROMPT, timeout,
                )
                futures[fut] = i

            for future in as_completed(futures):
                idx = futures[future]
                try:
                    resp = future.result(timeout=timeout + 10)
                    resp.round_num = 2
                    resp.round_type = RoundType.CRITIQUE
                    with lock:
                        responses[idx] = resp
                    if stream_callback and resp.ok:
                        try:
                            stream_callback(resp.provider_name, resp.text)
                        except Exception:
                            pass
                except Exception as e:
                    with lock:
                        responses[idx] = ProviderResponse(
                            provider_name=providers_list[idx].name,
                            model=providers_list[idx].model,
                            text="",
                            error=str(e),
                            round_num=2,
                            round_type=RoundType.CRITIQUE,
                        )

        finished_at = datetime.now(timezone.utc).isoformat()
        return DeliberationRound(
            round_num=2,
            round_type=RoundType.CRITIQUE,
            responses=[r for r in responses if r is not None],
            started_at=started_at,
            finished_at=finished_at,
        )

    def _run_revise_round(
        self,
        prompt: str,
        prev_rounds: list[DeliberationRound],
        timeout: int,
        stream_callback: Optional[Callable] = None,
    ) -> DeliberationRound:
        """Round 3: 每个 provider 基于 Round 1 和 Round 2 的结果修订答案。

        上下文包含 Round 1 的全部回答 + Round 2 的全部批评。
        """
        # 合并 Round 1 和 Round 2 的所有回答作为上下文
        all_context = []
        for rnd in prev_rounds:
            all_context.extend(rnd.responses)

        import threading
        from concurrent.futures import ThreadPoolExecutor, as_completed

        started_at = datetime.now(timezone.utc).isoformat()
        providers_list = list(self.providers)
        n = len(providers_list)
        responses: list[Optional[ProviderResponse]] = [None] * n
        lock = threading.Lock()

        with ThreadPoolExecutor(max_workers=min(self._scheduler.max_workers, n)) as ex:
            futures = {}
            for i, provider in enumerate(providers_list):
                fut = ex.submit(
                    provider.invoke_with_context,
                    prompt, all_context, REVISE_SYSTEM_PROMPT, timeout,
                )
                futures[fut] = i

            for future in as_completed(futures):
                idx = futures[future]
                try:
                    resp = future.result(timeout=timeout + 10)
                    resp.round_num = 3
                    resp.round_type = RoundType.REVISE
                    with lock:
                        responses[idx] = resp
                    if stream_callback and resp.ok:
                        try:
                            stream_callback(resp.provider_name, resp.text)
                        except Exception:
                            pass
                except Exception as e:
                    with lock:
                        responses[idx] = ProviderResponse(
                            provider_name=providers_list[idx].name,
                            model=providers_list[idx].model,
                            text="",
                            error=str(e),
                            round_num=3,
                            round_type=RoundType.REVISE,
                        )

        finished_at = datetime.now(timezone.utc).isoformat()
        return DeliberationRound(
            round_num=3,
            round_type=RoundType.REVISE,
            responses=[r for r in responses if r is not None],
            started_at=started_at,
            finished_at=finished_at,
        )

    def _synthesize(
        self,
        prompt: str,
        all_rounds: list[DeliberationRound],
    ) -> SynthesisReport:
        """调用 Synthesizer 生成语义合成报告。"""
        if self._synthesizer is None:
            # 如果没有 judge_provider，返回空的 fallback
            from .protocol import SynthesisStatus
            return SynthesisReport(
                executive_summary="未配置 judge provider，无法合成。",
                status=SynthesisStatus.EMPTY,
                error="缺少 judge_provider",
            )
        return self._synthesizer.synthesize(prompt, all_rounds)

    # ── 辅助 ──────────────────────────────────────────────────────

    @staticmethod
    def _track_failed(
        round_: DeliberationRound,
        failed: set[str],
    ) -> None:
        """记录本轮中失败的 provider。"""
        for r in round_.responses:
            if not r.ok:
                failed.add(r.provider_name)
