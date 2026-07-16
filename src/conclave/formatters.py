"""输出格式化：Markdown 并排 / diff 视图 / JSON。"""

import difflib
import json
from typing import Optional

from .backends.base import Response


def format_markdown(
    responses: list[Response],
    prompt: str,
    diff_mode: bool = False,
) -> str:
    """生成 Markdown 格式的 Panel 输出。

    Args:
        responses: 各 backend 的响应列表
        prompt: 用户原始 prompt
        diff_mode: 是否输出 diff 视图（仅当恰好 2 个 response 时生效）

    Returns:
        Markdown 字符串
    """
    lines = [f"## Panel: {prompt!r}", ""]

    if diff_mode and len(responses) == 2:
        lines.append(_render_diff(responses[0], responses[1]))
    else:
        for i, resp in enumerate(responses):
            lines.append(_render_single(resp, i))

    return "\n".join(lines)


def format_json(responses: list[Response], prompt: str) -> str:
    """生成 JSON 格式输出。"""
    return json.dumps(
        {
            "prompt": prompt,
            "responses": [
                {
                    "backend_name": r.backend_name,
                    "model": r.model,
                    "text": r.text,
                    "elapsed_ms": r.elapsed_ms,
                    "exit_code": r.exit_code,
                    "cache_hit": r.cache_hit,
                    "error": r.error,
                }
                for r in responses
            ],
        },
        ensure_ascii=False,
        indent=2,
    )


def format_diff_text(responses: list[Response], prompt: str) -> str:
    """生成终端友好的 diff 文本（纯文本，不带 rich 渲染）。

    仅当恰好 2 个 response 时生效，否则退化为 Markdown 输出。
    """
    if len(responses) != 2:
        return format_markdown(responses, prompt)

    r1, r2 = responses[0], responses[1]
    lines1 = r1.text.splitlines()
    lines2 = r2.text.splitlines()

    diff_lines = list(
        difflib.unified_diff(
            lines1,
            lines2,
            fromfile=f"{r1.backend_name} ({r1.model})",
            tofile=f"{r2.backend_name} ({r2.model})",
            lineterm="",
        )
    )

    header = f"## Panel: {prompt!r}\n"
    if not diff_lines:
        return header + "\n(两个模型的输出完全相同)"

    return header + "\n".join(diff_lines)


# ---- 内部 ----

def _render_single(resp: Response, index: int) -> str:
    """渲染单个 response 的 Markdown 块。"""
    status = ""
    if resp.cache_hit:
        status += " [dim](cached)[/dim]"

    header = (
        f"### {resp.backend_name} ({resp.model})"
        f" | {resp.elapsed_ms:.0f}ms"
        f"{status}"
    )

    if resp.error:
        lines = [
            header,
            "",
            f"> ⚠️ **Error**: {resp.error}",
            "",
        ]
    else:
        lines = [
            header,
            "",
            resp.text,
            "",
        ]
    return "\n".join(lines)


def _render_diff(r1: Response, r2: Response) -> str:
    """渲染两个 response 的 diff 视图（Markdown 内嵌 diff 格式）。"""
    lines1 = r1.text.splitlines()
    lines2 = r2.text.splitlines()

    diff = list(
        difflib.unified_diff(
            lines1,
            lines2,
            fromfile=f"{r1.backend_name} ({r1.model})",
            tofile=f"{r2.backend_name} ({r2.model})",
            lineterm="",
        )
    )

    if not diff:
        return "\n*(两个模型的输出完全相同)*\n"

    return "\n```diff\n" + "\n".join(diff) + "\n```\n"
