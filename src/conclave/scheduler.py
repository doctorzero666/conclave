"""并行调度器 V3 — ThreadPoolExecutor + BaseProvider 接口

设计要点：
- 纯同步，concurrent.futures.ThreadPoolExecutor
- 每个 provider 在独立线程中调用 invoke()
- 收集 ProviderResponse → 组装 DeliberationRound
- 支持 per-provider timeout
- 支持流式并行调用（每轮收集所有 provider 的最新 chunk）
- terminate_all() 终止 CLI 子进程
"""

import threading
import time
from concurrent.futures import Future, ThreadPoolExecutor, as_completed, TimeoutError as FutureTimeoutError
from datetime import datetime, timezone
from typing import Callable, Iterator, Optional

from .providers.base import BaseProvider
from .protocol import DeliberationRound, ProviderResponse, RoundType, StreamChunk


class Scheduler:
    """并行调度器 — 并行调用多个 provider，收集响应为 DeliberationRound。"""

    def __init__(self, max_workers: int = 8, console=None):
        """
        Args:
            max_workers: ThreadPoolExecutor 最大线程数
            console: rich.Console 实例（用于实时展示，可选）
        """
        self.max_workers = max_workers
        self.console = console
        self._executor: Optional[ThreadPoolExecutor] = None
        self._terminated = threading.Event()

    # ── 同步并行调用 ──────────────────────────────────────────────

    def run_parallel(
        self,
        providers: list[BaseProvider],
        prompt: str,
        system_prompt: Optional[str] = None,
        timeout: int = 180,
        round_num: int = 1,
        round_type: RoundType = RoundType.INITIAL,
        stream_callback: Optional[Callable] = None,
    ) -> DeliberationRound:
        """并行调用所有 provider，收集响应为 DeliberationRound。

        Args:
            providers: BaseProvider 实例列表
            prompt: 用户 prompt
            system_prompt: 可选的 system prompt
            timeout: 每个 provider 的超时时间（秒）
            round_num: 轮次编号
            round_type: 轮次类型
            stream_callback: 流式回调，签名为 callback(provider_name, chunk_text)

        Returns:
            DeliberationRound 包含所有 provider 的响应
        """
        n = len(providers)
        if n == 0:
            return DeliberationRound(
                round_num=round_num,
                round_type=round_type,
                started_at=datetime.now(timezone.utc).isoformat(),
                finished_at=datetime.now(timezone.utc).isoformat(),
            )

        started_at = datetime.now(timezone.utc).isoformat()
        responses: list[Optional[ProviderResponse]] = [None] * n  # 保持顺序
        errors: dict[int, str] = {}
        lock = threading.Lock()
        self._terminated.clear()

        with ThreadPoolExecutor(max_workers=min(self.max_workers, n)) as executor:
            futures: dict[Future, int] = {}
            for i, provider in enumerate(providers):
                fut = executor.submit(
                    self._invoke_one,
                    provider, prompt, system_prompt, timeout,
                    round_num, round_type, stream_callback,
                )
                futures[fut] = i

            for future in as_completed(futures):
                idx = futures[future]
                try:
                    resp = future.result(timeout=timeout + 10)
                    with lock:
                        responses[idx] = resp
                except FutureTimeoutError:
                    with lock:
                        responses[idx] = ProviderResponse(
                            provider_name=providers[idx].name,
                            model=providers[idx].model,
                            text="",
                            error=f"超时（{timeout}s）",
                            round_num=round_num,
                            round_type=round_type,
                        )
                except Exception as e:
                    with lock:
                        responses[idx] = ProviderResponse(
                            provider_name=providers[idx].name,
                            model=providers[idx].model,
                            text="",
                            error=str(e),
                            round_num=round_num,
                            round_type=round_type,
                        )

        finished_at = datetime.now(timezone.utc).isoformat()
        # 过滤掉 None（理论上不会发生），并转为非 Optional 列表
        final_responses: list[ProviderResponse] = [r for r in responses if r is not None]
        return DeliberationRound(
            round_num=round_num,
            round_type=round_type,
            responses=final_responses,
            started_at=started_at,
            finished_at=finished_at,
        )

    def _invoke_one(
        self,
        provider: BaseProvider,
        prompt: str,
        system_prompt: Optional[str],
        timeout: int,
        round_num: int,
        round_type: RoundType,
        stream_callback: Optional[Callable],
    ) -> ProviderResponse:
        """在独立线程中调用单个 provider。

        如果支持流式且有 stream_callback，走 stream 路径；
        否则直接 invoke。
        """
        if self._terminated.is_set():
            return ProviderResponse(
                provider_name=provider.name,
                model=provider.model,
                text="",
                error="已终止",
                round_num=round_num,
                round_type=round_type,
            )

        t0 = time.monotonic()

        # 如果 provider 支持流式且传入了 stream_callback，走流式
        caps = provider.capabilities()
        if caps.supports_streaming and stream_callback is not None:
            try:
                full_text = ""
                finish_reason = "stop"
                for chunk in provider.stream(prompt, system_prompt, timeout):
                    if self._terminated.is_set():
                        break
                    if chunk.error:
                        raise RuntimeError(chunk.error)
                    if chunk.delta:
                        full_text += chunk.delta
                        try:
                            stream_callback(provider.name, chunk.delta)
                        except Exception:
                            pass
                    if chunk.is_done:
                        finish_reason = chunk.finish_reason or "stop"
                        break
                elapsed = int((time.monotonic() - t0) * 1000)
                return ProviderResponse(
                    provider_name=provider.name,
                    model=provider.model,
                    text=full_text,
                    finish_reason=finish_reason,
                    elapsed_ms=elapsed,
                    round_num=round_num,
                    round_type=round_type,
                )
            except Exception as e:
                elapsed = int((time.monotonic() - t0) * 1000)
                return ProviderResponse(
                    provider_name=provider.name,
                    model=provider.model,
                    text="",
                    error=str(e),
                    elapsed_ms=elapsed,
                    round_num=round_num,
                    round_type=round_type,
                )

        # 否则直接 invoke
        try:
            resp = provider.invoke(prompt, system_prompt, timeout)
            resp.round_num = round_num
            resp.round_type = round_type
            if stream_callback and resp.ok:
                try:
                    stream_callback(provider.name, resp.text)
                except Exception:
                    pass
            return resp
        except Exception as e:
            elapsed = int((time.monotonic() - t0) * 1000)
            return ProviderResponse(
                provider_name=provider.name,
                model=provider.model,
                text="",
                error=str(e),
                elapsed_ms=elapsed,
                round_num=round_num,
                round_type=round_type,
            )

    # ── 流式并行调用 ──────────────────────────────────────────────

    def run_parallel_streaming(
        self,
        providers: list[BaseProvider],
        prompt: str,
        system_prompt: Optional[str] = None,
        timeout: int = 180,
        round_num: int = 1,
        round_type: RoundType = RoundType.INITIAL,
    ) -> Iterator[list[Optional[StreamChunk]]]:
        """并行流式调用，每次 yield 所有 provider 的最新 chunks。

        每个 provider 在独立线程中 stream()，chunks 通过队列收集。
        主线程每次从队列取出所有可用的 chunk，按 provider 分组后 yield。

        Args:
            providers: BaseProvider 实例列表
            prompt: 用户 prompt
            system_prompt: 可选的 system prompt
            timeout: 每个 provider 的超时
            round_num: 轮次编号
            round_type: 轮次类型

        Yields:
            list[StreamChunk]: 每个 provider 的最新 chunk（未完成的 provider 可能为空）
        """
        n = len(providers)
        if n == 0:
            return

        from queue import Queue

        chunk_queue: Queue = Queue()
        done_count = [0]  # 用 list 做可变计数器
        lock = threading.Lock()
        self._terminated.clear()

        def _stream_one(provider: BaseProvider, idx: int):
            try:
                for chunk in provider.stream(prompt, system_prompt, timeout):
                    if self._terminated.is_set():
                        break
                    chunk_queue.put((idx, chunk))
            except Exception as e:
                chunk_queue.put((idx, StreamChunk(
                    provider_name=provider.name,
                    delta="",
                    is_done=True,
                    error=str(e),
                )))
            finally:
                with lock:
                    done_count[0] += 1

        with ThreadPoolExecutor(max_workers=min(self.max_workers, n)) as executor:
            for i, provider in enumerate(providers):
                executor.submit(_stream_one, provider, i)

            # 持续从队列取 chunk，直到所有 provider 完成
            while done_count[0] < n:
                batch: list[Optional[StreamChunk]] = [None] * n
                has_any = False

                # 非阻塞地取出当前队列中所有 chunk
                while True:
                    try:
                        idx, chunk = chunk_queue.get(timeout=0.05)
                        batch[idx] = chunk
                        has_any = True
                    except Exception:
                        break

                if has_any:
                    yield batch

        # 最后再 drain 一次队列
        final_batch: list[Optional[StreamChunk]] = [None] * n
        while not chunk_queue.empty():
            try:
                idx, chunk = chunk_queue.get_nowait()
                final_batch[idx] = chunk
            except Exception:
                break
        yield final_batch

    # ── 终止 ──────────────────────────────────────────────────────

    def terminate_all(self):
        """终止所有正在运行的子进程（CLI provider 用）。

        设置终止标志，子线程检测到后停止处理。
        注意：这不会强制 kill 子进程，子进程在自己的 invoke/stream 实现中应检查此标志。
        """
        self._terminated.set()

    def is_terminated(self) -> bool:
        """检查是否已设置终止标志。"""
        return self._terminated.is_set()
