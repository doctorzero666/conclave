"""并行调度器：ThreadPoolExecutor + rich.Live 实时展示。"""

import signal
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from queue import Queue
from typing import Optional

from rich.console import Console
from rich.live import Live
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table

from .backends.base import AllBackendsFailedError, Backend, Response


class Scheduler:
    """并行调用多个 Backend，实时展示进度和结果。

    设计要点：
    - ThreadPoolExecutor 并行执行
    - 每个 backend 通过 on_chunk 回调把输出片段推入队列
    - 主线程消费队列，用 rich.Live 实时刷新面板
    - 先返回的先展示
    - SIGINT 传播到所有子进程
    """

    MAX_WORKERS = 4

    def __init__(self, console: Optional[Console] = None):
        self.console = console or Console()

    def run_all(
        self,
        prompt: str,
        backends: list[Backend],
        timeout: float = 180,
    ) -> list[Response]:
        """并行运行所有 backend，返回成功的响应列表。

        Args:
            prompt: 用户 prompt
            backends: backend 实例列表
            timeout: 每个 backend 的独立超时

        Returns:
            成功的 Response 列表

        Raises:
            AllBackendsFailedError: 所有 backend 全部失败
        """
        n = len(backends)
        if n == 0:
            raise AllBackendsFailedError({})

        results: dict[str, Response] = {}    # backend_name → Response
        errors: dict[str, str] = {}          # backend_name → error message
        outputs: dict[str, list[str]] = {}   # backend_name → 实时输出行
        output_lock = threading.Lock()
        result_queue: Queue = Queue()

        # 注册 SIGINT handler —— 注意：Python 信号处理只在主线程执行，
        # 我们在这里设一个标志位，子线程负责 terminate
        interrupted = threading.Event()

        def _sigint_handler(sig, frame):
            interrupted.set()

        old_handler = signal.signal(signal.SIGINT, _sigint_handler)

        try:
            # 构建 Live 渲染函数
            def _render_live() -> Table:
                table = Table.grid(padding=(0, 1))
                table.add_column(style="bold")
                for backend in backends:
                    name = backend.name
                    with output_lock:
                        lines = outputs.get(name, [])
                    if name in results:
                        resp = results[name]
                        if resp.error:
                            status = f"[red]✗ ERROR ({resp.elapsed_ms:.0f}ms)[/red]"
                        else:
                            status = f"[green]✓ DONE ({resp.elapsed_ms:.0f}ms)[/green]"
                    elif name in errors:
                        status = f"[red]✗ FAILED[/red]"
                    else:
                        status = "[yellow]⏳ running...[/yellow]"
                    table.add_row(f"[bold]{name}[/bold] {status}")
                    # 显示最新 3 行输出
                    if lines:
                        preview = "".join(lines[-3:]).strip()[:120]
                        if preview:
                            table.add_row(f"  [dim]{preview}[/dim]")
                return table

            with Live(_render_live(), console=self.console, refresh_per_second=4,
                      transient=False) as live:
                with ThreadPoolExecutor(
                    max_workers=min(self.MAX_WORKERS, n)
                ) as executor:
                    futures = {}
                    for backend in backends:
                        fut = executor.submit(
                            self._invoke_one,
                            backend,
                            prompt,
                            timeout,
                            results,
                            errors,
                            outputs,
                            output_lock,
                            result_queue,
                            interrupted,
                        )
                        futures[fut] = backend.name

                    # 消费结果队列 + 定期刷新
                    for future in as_completed(futures):
                        backend_name = futures[future]
                        try:
                            future.result()
                        except Exception:
                            pass

                        live.update(_render_live())

            # 收集成功的响应
            successful = [r for r in results.values() if r.error is None]
            # 也收集有 error 但至少拿到了 text 的（部分成功）
            partial = [r for r in results.values() if r.error is not None and r.text]

            all_ok = successful + partial

            if not all_ok:
                raise AllBackendsFailedError(errors)

            return all_ok

        finally:
            signal.signal(signal.SIGINT, old_handler)

    def _invoke_one(
        self,
        backend: Backend,
        prompt: str,
        timeout: float,
        results: dict,
        errors: dict,
        outputs: dict,
        lock: threading.Lock,
        queue: Queue,
        interrupted: threading.Event,
    ) -> None:
        """在独立线程中调用一个 backend。"""
        def _on_chunk(line: str):
            if interrupted.is_set():
                return
            queue.put((backend.name, line))
            with lock:
                outputs.setdefault(backend.name, []).append(line)

        try:
            if interrupted.is_set():
                errors[backend.name] = "Interrupted"
                return
            resp = backend.invoke(prompt, timeout=timeout, on_chunk=_on_chunk)
            results[backend.name] = resp
            if resp.error:
                errors[backend.name] = resp.error
        except Exception as e:
            errors[backend.name] = str(e)
