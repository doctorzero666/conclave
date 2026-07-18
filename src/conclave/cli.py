"""CLI 入口 — conclave V3"""

import sys
from typing import Optional

import click
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from . import __version__
from .config import Config, init_config, load_config, validate_config
from .engine import DeliberationEngine
from .protocol import DeliberationResult, ProviderResponse, SynthesisStatus
from .providers import create_provider


console = Console()


# ─── Helper ────────────────────────────────────────────────────────

def _build_providers(config: Config) -> tuple[list, Optional]:
    """从配置构建 participant providers + judge provider"""
    participants = []
    judge = None
    for name in config.default_providers:
        pconf = config.get_provider(name)
        if pconf:
            try:
                provider = create_provider(pconf)
                participants.append(provider)
            except Exception as e:
                console.print(f"[yellow]Warning: 无法创建 provider '{name}': {e}[/yellow]")

    # Judge
    jconf = config.get_provider(config.judge_provider)
    if jconf:
        try:
            judge = create_provider(jconf)
        except Exception:
            pass

    return participants, judge


def _display_result(result: DeliberationResult, fmt: str, diff: bool):
    """格式化输出审议结果"""
    if fmt == "json":
        import json
        console.print_json(json.dumps(_result_to_dict(result), indent=2, ensure_ascii=False))
        return

    # Markdown 输出
    console.print(f"\n[bold cyan]╔══════════════════════════════════════════╗[/bold cyan]")
    console.print(f"[bold cyan]║   conclave v3 — 审议报告                   ║[/bold cyan]")
    console.print(f"[bold cyan]╚══════════════════════════════════════════╝[/bold cyan]")
    console.print(f"[dim]Prompt: {result.prompt[:100]}{'...' if len(result.prompt) > 100 else ''}[/dim]")
    console.print(f"[dim]{result.round_count} 轮审议 · {result.provider_count} 模型 · {result.total_duration_ms/1000:.1f}s[/dim]\n")

    for i, round_ in enumerate(result.rounds):
        console.print(f"[bold]━━━ Round {i+1}: {round_.round_type.value} ━━━[/bold]")
        for resp in round_.responses:
            status = "[green]✓[/green]" if resp.ok else f"[red]✗ {resp.error}[/red]"
            console.print(f"  [{resp.provider_name}] {status} ({resp.elapsed_ms/1000:.1f}s)")
            if resp.ok:
                # 截断显示（太长不适合终端）
                text = resp.text[:500] + ("..." if len(resp.text) > 500 else "")
                console.print(Panel(text, title=f"{resp.provider_name} ({resp.model})",
                                    border_style="dim"))
        console.print()

    # Synthesis
    if result.synthesis and result.synthesis.status == SynthesisStatus.SUCCESS:
        syn = result.synthesis
        console.print(f"[bold green]━━━ 合成报告 (Judge: {syn.judge_model}) ━━━[/bold green]")
        console.print(f"\n[bold]摘要:[/bold] {syn.executive_summary}")

        if syn.consensus:
            console.print("\n[bold]✅ 共识 ({len(syn.consensus)}):[/bold]")
            for c in syn.consensus:
                agreed = ", ".join(c.agreed_by)
                confidence = {"high": "🟢", "medium": "🟡", "low": "🔴"}.get(c.confidence, "")
                console.print(f"  {confidence} {c.point}  [dim]({agreed})[/dim]")

        if syn.divergences:
            console.print(f"\n[bold]🔀 分歧 ({len(syn.divergences)}):[/bold]")
            for d in syn.divergences:
                crit = "🔴" if d.critical else "🟡"
                console.print(f"  {crit} {d.point}")
                for name, pos in d.positions.items():
                    console.print(f"     [{name}]: {pos}")

        if syn.insights:
            console.print(f"\n[bold]💡 独特洞察 ({len(syn.insights)}):[/bold]")
            for ins in syn.insights:
                console.print(f"  [{ins.provider_name}] {ins.category}: {ins.point}")

    if result.failed_providers:
        console.print(f"\n[red]⚠ 失败的 provider: {', '.join(result.failed_providers)}[/red]")


def _result_to_dict(result: DeliberationResult) -> dict:
    """DeliberationResult → JSON 友好 dict"""
    rounds_data = []
    for r in result.rounds:
        rounds_data.append({
            "round_num": r.round_num,
            "round_type": r.round_type.value,
            "responses": [
                {"provider": resp.provider_name, "model": resp.model,
                 "text": resp.text[:2000], "ok": resp.ok, "error": resp.error,
                 "elapsed_ms": resp.elapsed_ms}
                for resp in r.responses
            ]
        })
    return {
        "prompt": result.prompt,
        "rounds": rounds_data,
        "total_duration_ms": result.total_duration_ms,
        "provider_count": result.provider_count,
        "round_count": result.round_count,
        "synthesis": {
            "summary": result.synthesis.executive_summary if result.synthesis else "",
            "consensus_count": len(result.synthesis.consensus) if result.synthesis else 0,
            "divergence_count": len(result.synthesis.divergences) if result.synthesis else 0,
        },
    }


# ─── CLI Commands ──────────────────────────────────────────────────

@click.group()
@click.version_option(__version__, prog_name="conclave")
def main():
    """conclave v3 — 多模型审议协议引擎

    一行命令启动多个 AI 模型的多轮审议，自动合成共识报告。
    """


@main.command()
@click.argument("prompt")
@click.option("--providers", "-p", default=None,
              help="逗号分隔的 provider 列表 (默认: 使用 config.yaml)")
@click.option("--rounds", "-r", type=int, default=None,
              help="审议轮数: 1/2/3 (默认: config.yaml 或 1)")
@click.option("--format", "-f", "fmt", type=click.Choice(["markdown", "json"]),
              default="markdown", help="输出格式")
@click.option("--diff/--no-diff", default=False,
              help="Diff 视图 (2个provider时有效)")
@click.option("--no-cache", is_flag=True, default=False, help="跳过缓存")
@click.option("--copy", is_flag=True, default=False, help="复制到剪贴板")
@click.option("--timeout", "-t", type=int, default=None, help="超时秒数")
@click.option("--max-cost", type=float, default=None, help="最大成本($)")
@click.option("--verbose", "-v", is_flag=True, default=False, help="详细输出")
@click.option("--no-synthesis", is_flag=True, default=False, help="跳过合成")
def run(prompt, providers, rounds, fmt, diff, no_cache, copy, timeout,
        max_cost, verbose, no_synthesis):
    """运行多模型审议"""
    config = load_config()

    # 参数覆盖
    if rounds is None:
        rounds = config.default_rounds
    if timeout is None:
        timeout = config.timeout
    if max_cost is None:
        max_cost = config.max_cost_usd

    # Provider
    if providers:
        provider_names = [p.strip() for p in providers.split(",")]
        participants = []
        for name in provider_names:
            pconf = config.get_provider(name)
            if pconf:
                try:
                    participants.append(create_provider(pconf))
                except Exception as e:
                    console.print(f"[red]Error: 无法创建 provider '{name}': {e}[/red]")
                    sys.exit(1)
            else:
                console.print(f"[red]Error: provider '{name}' 未在配置中定义[/red]")
                sys.exit(1)
    else:
        participants, _ = _build_providers(config)

    if not participants:
        console.print("[red]Error: 没有可用的 provider。运行 'conclave config init'。[/red]")
        sys.exit(1)

    # Judge (for synthesis in rounds>=3)
    if rounds >= 3 and not no_synthesis:
        _, judge = _build_providers(config)
    else:
        judge = None

    # 运行审议
    from time import time as now
    start = now()

    def round_callback(round_):
        pass  # 可以在这里加进度推送

    engine = DeliberationEngine(
        providers=participants,
        judge_provider=judge,
        config=config,
    )

    try:
        result = engine.run(
            prompt=prompt,
            rounds=rounds,
            timeout=timeout,
            max_cost=max_cost,
            on_round_callback=round_callback,
            synthesis_enabled=(rounds >= 3 and not no_synthesis),
        )

        if result.total_duration_ms == 0:
            result.total_duration_ms = int((now() - start) * 1000)

        _display_result(result, fmt, diff)

        # Clipboard
        if copy:
            try:
                import pyperclip
                pyperclip.copy(str(result))
            except ImportError:
                console.print("[yellow]pyperclip not installed[/yellow]")

        # Exit code
        if result.failed_providers:
            has_any_ok = any(r.any_ok for r in result.rounds)
            sys.exit(2 if has_any_ok else 1)
        sys.exit(0)

    except KeyboardInterrupt:
        console.print("\n[yellow]审议已中断[/yellow]")
        sys.exit(130)
    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        if verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


@main.group()
def config():
    """管理配置 (~/.conclave/config.yaml)"""


@config.command("init")
def config_init():
    """初始化配置文件"""
    c = init_config(interactive=True)
    console.print(f"[green]✓[/green] 配置已写入 {c}")
    console.print(f"[dim]Provider 数量: {len(c.providers)}[/dim]")


@config.command("show")
def config_show():
    """显示当前配置"""
    config = load_config()
    console.print(f"[bold]conclave v{__version__} 配置[/bold]\n")
    console.print(f"  默认 provider: {', '.join(config.default_providers)}")
    console.print(f"  Judge: {config.judge_provider}")
    console.print(f"  默认轮数: {config.default_rounds}")
    console.print(f"  缓存: {'启用' if config.cache_enabled else '禁用'} ({config.cache_dir})")
    console.print(f"\n[bold]Providers ({len(config.providers)}):[/bold]")
    for name, pc in config.providers.items():
        console.print(f"  [{name}] {pc.provider_type} · {pc.model}")


if __name__ == "__main__":
    main()
