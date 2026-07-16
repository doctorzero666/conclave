"""CLI 入口：conclave run "prompt" --backends claude,codex"""

import sys
from typing import Optional

import click
from rich.console import Console

from . import __version__
from .backends.base import AllBackendsFailedError, Backend
from .backends.cli_backend import PRESETS, CLIBackend
from .cache import Cache
from .formatters import format_diff_text, format_json, format_markdown
from .scheduler import Scheduler


def _get_backends(names: list[str], verbose: bool = False) -> list[Backend]:
    """根据名字列表解析 Backend 实例。

    先在 PRESETS 中查找预设，找不到则尝试作为 CLIBackend 的可执行文件名。
    """
    backends = []
    for name in names:
        name = name.strip()
        if name in PRESETS:
            backend = PRESETS[name]()
        else:
            # 尝试作为 CLIBackend：executable = name, args = ["{prompt}"]
            backend = CLIBackend(
                name=name,
                model=name,
                executable=name,
                args_template=["{prompt}"],
            )
        if verbose:
            click.echo(f"[dim]Backend '{backend.name}' → {backend.model}[/dim]", err=True)
        backends.append(backend)
    return backends


@click.group()
@click.version_option(__version__, prog_name="conclave")
def main():
    """conclave — 多模型并行审议 CLI

    一行命令，同时调用多个 AI 模型，实时展示进度，并排对比输出。
    依赖用户本地已安装并登录的 CLI 工具（Claude Code / Codex CLI）。
    本工具不管理 API 密钥。
    """


@main.command()
@click.argument("prompt")
@click.option(
    "--backends", "-b",
    default="claude,codex",
    envvar="CONCLAVE_BACKENDS",
    help="逗号分隔的 backend 列表 (默认: claude,codex)",
)
@click.option(
    "--format", "-f",
    "fmt",
    type=click.Choice(["markdown", "json"]),
    default="markdown",
    envvar="CONCLAVE_FORMAT",
    help="输出格式 (默认: markdown)",
)
@click.option(
    "--diff/--no-diff",
    default=False,
    envvar="CONCLAVE_DIFF",
    help="并排 diff 视图（仅 2 个 backend 时生效）",
)
@click.option(
    "--no-cache",
    is_flag=True,
    default=False,
    help="跳过缓存，强制重新调用",
)
@click.option(
    "--copy",
    is_flag=True,
    default=False,
    help="将结果复制到剪贴板",
)
@click.option(
    "--timeout", "-t",
    type=int,
    default=180,
    envvar="CONCLAVE_TIMEOUT",
    help="每个 backend 超时秒数 (默认: 180)",
)
@click.option(
    "--verbose", "-v",
    is_flag=True,
    default=False,
    help="打印子进程命令行（调试用）",
)
def run(
    prompt: str,
    backends: str,
    fmt: str,
    diff: bool,
    no_cache: bool,
    copy: bool,
    timeout: int,
    verbose: bool,
):
    """并行调用多个 AI 模型，对比输出。"""
    console = Console()

    # 1. 解析 backend 列表
    backend_names = [b.strip() for b in backends.split(",") if b.strip()]
    if not backend_names:
        click.echo("Error: at least one backend required", err=True)
        sys.exit(1)

    try:
        backend_list = _get_backends(backend_names, verbose=verbose)
    except Exception as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)

    # 2. 初始化 Cache + Scheduler
    cache = Cache()
    scheduler = Scheduler(console=console)

    # 3. 准备 args_templates（用于缓存 key）
    args_templates = []
    for b in backend_list:
        if isinstance(b, CLIBackend):
            args_templates.append(b.args_template_str)
        else:
            args_templates.append("")

    # 4. 检查缓存
    responses = None
    from_cache = False
    if not no_cache:
        cached = cache.get(prompt, backend_names, args_templates)
        if cached is not None:
            responses = cached
            from_cache = True
            for r in responses:
                r.cache_hit = True

    # 5. 调度执行
    if responses is None:
        try:
            responses = scheduler.run_all(prompt, backend_list, timeout=timeout)
        except AllBackendsFailedError as e:
            console.print(f"[red]{e}[/red]")
            sys.exit(1)

    # 6. 写入缓存
    if not no_cache and not from_cache:
        cache.set(prompt, backend_names, args_templates, responses)

    # 7. 格式化输出
    if fmt == "json":
        output = format_json(responses, prompt)
    elif diff and len(responses) == 2:
        output = format_diff_text(responses, prompt)
    else:
        output = format_markdown(responses, prompt, diff_mode=False)

    console.print(output)

    # 8. 剪贴板
    if copy:
        try:
            import pyperclip
            pyperclip.copy(output)
            console.print("[dim](copied to clipboard)[/dim]")
        except ImportError:
            console.print(
                "[yellow]pyperclip not installed. "
                "Run: pip install conclave[clipboard][/yellow]"
            )

    # 9. 退出码
    # 与 Scheduler 语义一致：有 text 的响应算成功（即使有非致命 error）
    # 全成功（无 error）→ 0  /  部分成功（有 text 但有 error）→ 2  /  全失败 → 1
    has_any_text = any(r.text for r in responses)
    all_clean = all(r.error is None for r in responses)
    if all_clean:
        sys.exit(0)
    elif has_any_text:
        sys.exit(2)
    else:
        sys.exit(1)


if __name__ == "__main__":
    main()
