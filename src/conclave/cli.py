"""CLI 入口 — conclave V3"""

import os
import re
import sys
from pathlib import Path
from typing import Optional

import click
from rich.console import Console
from rich.markup import escape
from rich.panel import Panel

from . import __version__
from .config import Config, init_config, load_config, validate_config
from .engine import DeliberationEngine
from .protocol import DeliberationResult, ProviderResponse, SynthesisStatus
from .providers import create_provider


console = Console()


# ─── ANSI / control sequence stripping ─────────────────────────────
# CSI: ESC [ ...params... final-byte (@-~)
_ANSI_CSI_RE = re.compile(r"\x1B\[[0-?]*[ -/]*[@-~]")
# OSC: ESC ] ...payload... terminator (BEL \x07 or ST \x1B\\)
_ANSI_OSC_RE = re.compile(r"\x1B\][^\x07\x1B]*(?:\x07|\x1B\\)")
# Other control characters, preserving \t (\x09) and \n (\x0A). Range covers
# both C0 (\x00-\x1F, \x7F) and 8-bit C1 (\x80-\x9F, incl. CSI U+009B / OSC
# U+009D) so ANSI-free markdown survives providers emitting 8-bit escapes.
_CTRL_RE = re.compile(r"[\x00-\x08\x0B-\x1F\x7F-\x9F]")


def _strip_ansi(text: str) -> str:
    """Remove ANSI CSI/OSC sequences and stray control chars.

    Preserves \\n and \\t so markdown formatting survives. JSON callers keep
    text lossless — only markdown output routes through this.
    """
    if not text:
        return text
    text = _ANSI_CSI_RE.sub("", text)
    text = _ANSI_OSC_RE.sub("", text)
    text = _CTRL_RE.sub("", text)
    return text


def _terminal_safe(text) -> str:
    """Replace chars that can't UTF-8 encode (e.g. unpaired surrogates like
    \\ud800) with the replacement character. Used only for terminal display —
    never for `--output` serialization, which must remain lossless.
    """
    if text is None:
        return ""
    if not isinstance(text, str):
        text = str(text)
    if not text:
        return text
    return text.encode("utf-8", "replace").decode("utf-8", "replace")


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
                console.print(
                    f"[yellow]Warning: 无法创建 provider '{escape(name)}': {escape(str(e))}[/yellow]"
                )

    # Judge
    jconf = config.get_provider(config.judge_provider)
    if jconf:
        try:
            judge = create_provider(jconf)
        except Exception:
            pass

    return participants, judge


def _display_result(result: DeliberationResult, fmt: str, diff: bool):
    """格式化输出审议结果 (终端)。

    所有动态文本 (prompt/provider/model/response/synthesis) 在 escape/Panel
    前先经 _terminal_safe，避免 provider 返回的 unpaired surrogate 让 Rich
    渲染到 stdout 时 UnicodeEncodeError 崩溃。--output 落盘走独立路径，保持
    lossless。
    """
    ts = _terminal_safe

    if fmt == "json":
        import json
        # 终端展示：先 ensure_ascii=False 得到人类可读 JSON，再过 terminal-safe
        # 把 unpaired surrogate 替换成 U+FFFD。Rich print_json 会 re-parse 再
        # 渲染，若字符串里含 surrogate 会在 stdout UTF-8 编码时崩溃。落盘
        # (--output) 走独立路径，保持 lossless。
        payload = json.dumps(_result_to_dict(result), indent=2, ensure_ascii=False)
        console.print_json(ts(payload))
        return

    # Markdown 输出
    console.print(f"\n[bold cyan]╔══════════════════════════════════════════╗[/bold cyan]")
    console.print(f"[bold cyan]║   conclave v3 — 审议报告                   ║[/bold cyan]")
    console.print(f"[bold cyan]╚══════════════════════════════════════════╝[/bold cyan]")
    prompt_preview = result.prompt[:100] + ("..." if len(result.prompt) > 100 else "")
    console.print(f"[dim]Prompt: {escape(ts(prompt_preview))}[/dim]")
    console.print(
        f"[dim]{result.round_count} 轮审议 · {result.provider_count} 模型 · "
        f"{result.total_duration_ms/1000:.1f}s[/dim]\n"
    )

    for i, round_ in enumerate(result.rounds):
        console.print(f"[bold]━━━ Round {i+1}: {escape(ts(round_.round_type.value))} ━━━[/bold]")
        for resp in round_.responses:
            status = (
                "[green]✓[/green]"
                if resp.ok
                else f"[red]✗ {escape(ts(resp.error or ''))}[/red]"
            )
            console.print(
                f"  \\[{escape(ts(resp.provider_name))}] {status} ({resp.elapsed_ms/1000:.1f}s)"
            )
            if resp.ok:
                # 截断显示（太长不适合终端）
                text = resp.text[:500] + ("..." if len(resp.text) > 500 else "")
                # provider 文本可能含 malformed Rich markup (如 [/bold]) 或
                # unpaired surrogate — 先 terminal-safe 再 escape。
                console.print(Panel(
                    escape(ts(text)),
                    title=escape(ts(f"{resp.provider_name} ({resp.model})")),
                    border_style="dim",
                ))
        console.print()

    # Synthesis
    if result.synthesis and result.synthesis.status == SynthesisStatus.SUCCESS:
        syn = result.synthesis
        console.print(
            f"[bold green]━━━ 合成报告 (Judge: {escape(ts(syn.judge_model))}) ━━━[/bold green]"
        )
        console.print(f"\n[bold]摘要:[/bold] {escape(ts(syn.executive_summary))}")

        if syn.consensus:
            console.print(f"\n[bold]✅ 共识 ({len(syn.consensus)}):[/bold]")
            for c in syn.consensus:
                agreed = ", ".join(c.agreed_by)
                confidence = {"high": "🟢", "medium": "🟡", "low": "🔴"}.get(c.confidence, "")
                console.print(
                    f"  {confidence} {escape(ts(c.point))}  [dim]({escape(ts(agreed))})[/dim]"
                )

        if syn.divergences:
            console.print(f"\n[bold]🔀 分歧 ({len(syn.divergences)}):[/bold]")
            for d in syn.divergences:
                crit = "🔴" if d.critical else "🟡"
                console.print(f"  {crit} {escape(ts(d.point))}")
                for name, pos in d.positions.items():
                    console.print(f"     \\[{escape(ts(name))}]: {escape(ts(pos))}")

        if syn.insights:
            console.print(f"\n[bold]💡 独特洞察 ({len(syn.insights)}):[/bold]")
            for ins in syn.insights:
                console.print(
                    f"  \\[{escape(ts(ins.provider_name))}] {escape(ts(ins.category))}: {escape(ts(ins.point))}"
                )

    if result.failed_providers:
        console.print(
            f"\n[red]⚠ 失败的 provider: {escape(ts(', '.join(result.failed_providers)))}[/red]"
        )


def _result_to_dict(
    result: DeliberationResult,
    truncate: bool = True,
    full: bool = False,
) -> dict:
    """DeliberationResult → JSON 友好 dict

    Args:
        truncate: 若 True，每个 response.text 截断到 2000 字符 (终端场景)；
                  若 False，保留完整文本 (--output 落盘场景)。
        full: 若 True，输出完整字段 (供 --output 落盘，含 metadata/synthesis 全字段);
              若 False，输出精简字段 (终端预览)。
    """
    rounds_data = []
    for r in result.rounds:
        if full:
            responses = [
                {
                    "provider_name": resp.provider_name,
                    "model": resp.model,
                    "text": resp.text if not truncate else resp.text[:2000],
                    "finish_reason": resp.finish_reason,
                    "error": resp.error,
                    "elapsed_ms": resp.elapsed_ms,
                    "round_num": resp.round_num,
                    "round_type": resp.round_type.value,
                    "timestamp": resp.timestamp,
                }
                for resp in r.responses
            ]
            rounds_data.append({
                "round_num": r.round_num,
                "round_type": r.round_type.value,
                "started_at": r.started_at,
                "finished_at": r.finished_at,
                "responses": responses,
            })
        else:
            rounds_data.append({
                "round_num": r.round_num,
                "round_type": r.round_type.value,
                "responses": [
                    {
                        "provider": resp.provider_name,
                        "model": resp.model,
                        "text": (resp.text[:2000] if truncate else resp.text),
                        "ok": resp.ok,
                        "error": resp.error,
                        "elapsed_ms": resp.elapsed_ms,
                    }
                    for resp in r.responses
                ],
            })

    if full:
        if result.synthesis is not None:
            syn = result.synthesis
            synthesis_data = {
                "consensus": [
                    {
                        "point": c.point,
                        "agreed_by": list(c.agreed_by),
                        "confidence": c.confidence,
                    }
                    for c in syn.consensus
                ],
                "divergences": [
                    {
                        "point": d.point,
                        "positions": dict(d.positions),
                        "critical": d.critical,
                    }
                    for d in syn.divergences
                ],
                "insights": [
                    {
                        "point": i.point,
                        "provider_name": i.provider_name,
                        "category": i.category,
                    }
                    for i in syn.insights
                ],
                "executive_summary": syn.executive_summary,
                "judge_model": syn.judge_model,
                "status": syn.status.value,
                "error": syn.error,
                "total_cost_usd": syn.total_cost_usd,
            }
        else:
            synthesis_data = None

        return {
            "prompt": result.prompt,
            "total_duration_ms": result.total_duration_ms,
            "total_cost_usd": result.total_cost_usd,
            "provider_count": result.provider_count,
            "round_count": result.round_count,
            "failed_providers": list(result.failed_providers),
            "rounds": rounds_data,
            "synthesis": synthesis_data,
        }

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


def _result_to_markdown(result: DeliberationResult) -> str:
    """DeliberationResult → 纯文本 markdown (ANSI-free, 供 --output 落盘)。

    与终端渲染不同，此处不截断 response text，也不含 Rich markup。
    所有动态文本 (prompt/response/synthesis/provider 名等) 均经 _strip_ansi
    处理，防止 provider 输出的 ANSI/OSC/控制字符污染落盘 markdown。
    """
    s = _strip_ansi
    lines: list[str] = []
    lines.append(f"# conclave v3 — 审议报告")
    lines.append("")
    lines.append(f"- Prompt: {s(result.prompt)}")
    lines.append(
        f"- {result.round_count} 轮审议 · {result.provider_count} 模型 · "
        f"{result.total_duration_ms/1000:.1f}s"
    )
    lines.append("")

    for i, round_ in enumerate(result.rounds):
        lines.append(f"## Round {i+1}: {s(round_.round_type.value)}")
        lines.append("")
        for resp in round_.responses:
            status = "OK" if resp.ok else f"FAIL: {s(resp.error or '')}"
            lines.append(
                f"### [{s(resp.provider_name)}] {s(resp.model)} — {status} "
                f"({resp.elapsed_ms/1000:.1f}s)"
            )
            lines.append("")
            # 只要 text 非空，即便 ok=False (如 truncated) 也保留已收集内容
            if resp.text:
                lines.append(s(resp.text))
                lines.append("")
        lines.append("")

    if result.synthesis is not None:
        syn = result.synthesis
        # 即便 status != SUCCESS (FALLBACK / EMPTY / RETRY)，判断部分 summary /
        # consensus / insights / error 也要保留 — JSON 早就有，markdown 别静默丢。
        header = f"## 合成报告 (Judge: {s(syn.judge_model)}) — status: {s(syn.status.value)}"
        lines.append(header)
        lines.append("")
        if syn.error:
            lines.append(f"**错误:** {s(syn.error)}")
            lines.append("")
        if syn.executive_summary:
            lines.append(f"**摘要:** {s(syn.executive_summary)}")
            lines.append("")

        if syn.consensus:
            lines.append(f"### ✅ 共识 ({len(syn.consensus)})")
            lines.append("")
            for c in syn.consensus:
                agreed = ", ".join(s(name) for name in c.agreed_by)
                lines.append(f"- [{s(c.confidence)}] {s(c.point)} ({agreed})")
            lines.append("")

        if syn.divergences:
            lines.append(f"### 🔀 分歧 ({len(syn.divergences)})")
            lines.append("")
            for d in syn.divergences:
                crit = "critical" if d.critical else "minor"
                lines.append(f"- [{crit}] {s(d.point)}")
                for name, pos in d.positions.items():
                    lines.append(f"    - [{s(name)}]: {s(pos)}")
            lines.append("")

        if syn.insights:
            lines.append(f"### 💡 独特洞察 ({len(syn.insights)})")
            lines.append("")
            for ins in syn.insights:
                lines.append(
                    f"- [{s(ins.provider_name)}] {s(ins.category)}: {s(ins.point)}"
                )
            lines.append("")

    if result.failed_providers:
        lines.append(f"## ⚠ 失败的 provider")
        lines.append("")
        lines.append(", ".join(s(name) for name in result.failed_providers))
        lines.append("")

    return "\n".join(lines)


def _read_prompt_file(path_str: str) -> str:
    """读取 prompt 文件 (UTF-8)。失败时打印错误并 sys.exit(1)。"""
    p = Path(path_str)
    if not p.exists():
        console.print(f"[red]Error: prompt-file 不存在: {escape(path_str)}[/red]")
        sys.exit(1)
    if p.is_dir():
        console.print(f"[red]Error: prompt-file 是目录，不是文件: {escape(path_str)}[/red]")
        sys.exit(1)
    try:
        content = p.read_text(encoding="utf-8")
    except UnicodeDecodeError as e:
        console.print(f"[red]Error: prompt-file 不是有效 UTF-8: {escape(str(e))}[/red]")
        sys.exit(1)
    except OSError as e:
        console.print(f"[red]Error: 无法读取 prompt-file: {escape(str(e))}[/red]")
        sys.exit(1)
    if not content.strip():
        console.print(f"[red]Error: prompt-file 为空: {escape(path_str)}[/red]")
        sys.exit(1)
    return content


def _same_path(a: str, b: str) -> bool:
    """判断两个路径是否指向同一文件 (对不存在的路径也可用)。

    使用 realpath 解析 symlink；若两边都存在则用 samefile 兜底 (处理 hard link)。
    """
    try:
        if os.path.realpath(a) == os.path.realpath(b):
            return True
    except OSError:
        pass
    try:
        if os.path.exists(a) and os.path.exists(b):
            return os.path.samefile(a, b)
    except OSError:
        return False
    return False


# ─── CLI Commands ──────────────────────────────────────────────────

@click.group()
@click.version_option(__version__, prog_name="conclave")
def main():
    """conclave v3 — 多模型审议协议引擎

    一行命令启动多个 AI 模型的多轮审议，自动合成共识报告。
    """


@main.command()
@click.argument("prompt", required=False)
@click.option("--prompt-file", "prompt_file",
              type=str, default=None,
              help="从 UTF-8 文件读取 prompt (与 positional PROMPT 互斥)")
@click.option("--output", "-o", "output",
              type=str, default=None,
              help="将结果写入文件 (json/markdown)")
@click.option("--providers", "-p", "providers", default=None,
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
def run(prompt, prompt_file, output, providers, rounds, fmt, diff, no_cache,
        copy, timeout, max_cost, verbose, no_synthesis):
    """运行多模型审议"""
    # 1. Prompt 来源解析 — positional 与 --prompt-file 互斥
    if prompt and prompt_file:
        raise click.UsageError("PROMPT 与 --prompt-file 互斥，只能提供其中之一")
    if not prompt and not prompt_file:
        raise click.UsageError("必须提供 PROMPT 或 --prompt-file")

    if prompt_file:
        # 防止 prompt-file 与 output 同路径 (避免落盘覆盖输入)
        if output and _same_path(prompt_file, output):
            console.print(
                f"[red]Error: --output 与 --prompt-file 指向同一路径，"
                f"拒绝覆盖输入文件[/red]"
            )
            sys.exit(1)
        prompt = _read_prompt_file(prompt_file)

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
                    console.print(
                        f"[red]Error: 无法创建 provider '{escape(name)}': {escape(str(e))}[/red]"
                    )
                    sys.exit(1)
            else:
                console.print(
                    f"[red]Error: provider '{escape(name)}' 未在配置中定义[/red]"
                )
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

        # --output 落盘先执行 (写失败仅警告)；确保即便终端渲染因
        # provider 返回的 malformed Rich markup 抛错，归档文件也已落地。
        # 原子写：先写同目录临时文件 → fsync → os.replace。若序列化/写入
        # 途中失败 (unpaired surrogate / disk-full)，目标原文件不动，
        # 避免 write_text 中途截断静默清空旧结果。
        if output:
            target = Path(output)
            tmp_path: Optional[Path] = None
            try:
                parent = target.parent if str(target.parent) else Path(".")
                if fmt == "json":
                    import json
                    payload = json.dumps(
                        _result_to_dict(result, truncate=False, full=True),
                        indent=2,
                        ensure_ascii=False,
                    )
                else:
                    payload = _result_to_markdown(result)

                import tempfile
                fd, tmp_name = tempfile.mkstemp(
                    prefix=f".{target.name}.tmp-",
                    dir=str(parent),
                )
                tmp_path = Path(tmp_name)
                with os.fdopen(fd, "w", encoding="utf-8") as fh:
                    fh.write(payload)
                    fh.flush()
                    os.fsync(fh.fileno())
                os.replace(tmp_path, target)
                tmp_path = None
            except (OSError, UnicodeError) as e:
                if tmp_path is not None:
                    try:
                        tmp_path.unlink()
                    except OSError:
                        pass
                console.print(
                    f"[yellow]Warning: 无法写入 --output {escape(str(output))}: "
                    f"{escape(str(e))}[/yellow]"
                )

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
        console.print(f"[red]Error: {escape(str(e))}[/red]")
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
    console.print(f"[green]✓[/green] 配置已写入 {escape(str(c))}")
    console.print(f"[dim]Provider 数量: {len(c.providers)}[/dim]")


@config.command("show")
def config_show():
    """显示当前配置"""
    config = load_config()
    console.print(f"[bold]conclave v{__version__} 配置[/bold]\n")
    console.print(f"  默认 provider: {escape(', '.join(config.default_providers))}")
    console.print(f"  Judge: {escape(config.judge_provider)}")
    console.print(f"  默认轮数: {config.default_rounds}")
    console.print(
        f"  缓存: {'启用' if config.cache_enabled else '禁用'} "
        f"({escape(str(config.cache_dir))})"
    )
    console.print(f"\n[bold]Providers ({len(config.providers)}):[/bold]")
    for name, pc in config.providers.items():
        console.print(
            f"  \\[{escape(name)}] {escape(pc.provider_type)} · {escape(pc.model)}"
        )


if __name__ == "__main__":
    main()
