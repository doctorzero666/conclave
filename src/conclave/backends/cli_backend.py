"""CLIBackend — 通过 subprocess 调用外部 CLI 工具（Claude Code / Codex CLI 等）。"""

import shutil
import subprocess
import time
from typing import Callable, Optional

from .base import Backend, BackendNotFoundError, Response

# stdout 最大读取字节数（10MB）
MAX_STDOUT_BYTES = 10 * 1024 * 1024


def _default_parse_line(line: str) -> Optional[str]:
    """默认的行解析器：原样返回。"""
    return line


# ---- 内置预设 ----

def claude_preset() -> "CLIBackend":
    """Claude Code CLI 预设。"""
    return CLIBackend(
        name="claude",
        model="claude-sonnet-4",
        executable="claude",
        args_template=["-p", "{prompt}", "--output-format", "text"],
    )


def codex_preset() -> "CLIBackend":
    """Codex CLI 预设。"""
    return CLIBackend(
        name="codex",
        model="gpt-5.6-sol",
        executable="codex",
        args_template=["exec", "--skip-git-repo-check", "{prompt}"],
    )


# ---- 预设注册表 ----

PRESETS: dict[str, callable] = {
    "claude": claude_preset,
    "codex": codex_preset,
}


# ---- CLIBackend ----

class CLIBackend(Backend):
    """通过 subprocess 调用外部 CLI 工具获取模型响应。

    安全策略：
    - 使用 shutil.which() 查找可执行文件，避免 PATH 注入
    - 使用 list args（禁用 shell=True），避免命令注入
    - 超时三段式清理：terminate → wait(2s) → kill
    - stdout 10MB 上限，防止内存溢出
    - 编码容错：utf-8 + errors="replace"
    """

    def __init__(
        self,
        name: str,
        model: str,
        executable: str,
        args_template: list[str],
        parse_line: Callable[[str], Optional[str]] = _default_parse_line,
    ):
        """初始化 CLIBackend。

        Args:
            name: backend 名字，如 "claude"
            model: 模型标识，如 "claude-sonnet-4"
            executable: 可执行文件名或路径，如 "claude"
            args_template: 参数模板列表，"{prompt}" 会被替换为用户 prompt
            parse_line: 行解析器，输入原始行，输出清洗后的文本或 None（跳过该行）

        Raises:
            BackendNotFoundError: 可执行文件不在 PATH 中
        """
        self.name = name
        self.model = model
        self.args_template = args_template
        self.parse_line = parse_line

        resolved = shutil.which(executable)
        if resolved is None:
            raise BackendNotFoundError(
                f"'{executable}' not found in PATH.\n"
                f"Please install it first:\n"
                f"  Claude Code: npm install -g @anthropic-ai/claude-code\n"
                f"  Codex CLI:   npm install -g @openai/codex"
            )
        self.executable = resolved

    # ---- 公开属性 ----

    @property
    def args_template_str(self) -> str:
        """args_template 的字符串表示，用于缓存 key。"""
        return " ".join(self.args_template)

    # ---- invoke ----

    def invoke(
        self,
        prompt: str,
        timeout: float = 120,
        on_chunk: Optional[Callable[[str], None]] = None,
    ) -> Response:
        """调用外部 CLI 获取模型响应。"""
        start = time.time()

        # 1. 构造参数列表（替换 {prompt} 占位符）
        args = [arg.format(prompt=prompt) for arg in self.args_template]

        # 2. 启动子进程（关键：list args + 禁用 shell=True）
        try:
            proc = subprocess.Popen(
                [self.executable] + args,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1,           # 行缓冲，支持逐行读取
                encoding="utf-8",
                errors="replace",     # 非 UTF-8 字符替换为 �
            )
        except Exception as e:
            elapsed_ms = (time.time() - start) * 1000
            return Response(
                model=self.model,
                backend_name=self.name,
                elapsed_ms=elapsed_ms,
                exit_code=-1,
                error=f"Failed to start process: {e}",
            )

        # 3. 关闭 stdin（CLI 不需要交互输入）
        try:
            proc.stdin.close()
        except Exception:
            pass

        # 4. 逐行读取 stdout（带超时 + 字节上限）
        stdout_lines: list[str] = []
        stderr_lines: list[str] = []
        total_bytes = 0
        truncated = False
        timed_out = False

        try:
            # 读 stdout
            for line in proc.stdout:
                stdout_lines.append(line)
                total_bytes += len(line.encode("utf-8"))

                # 10MB 截断
                if total_bytes > MAX_STDOUT_BYTES:
                    truncated = True
                    break

                # 逐行回调
                parsed = self.parse_line(line)
                if parsed is not None and on_chunk is not None:
                    on_chunk(parsed)

            # 读 stderr（非阻塞收集）
            stderr_text = proc.stderr.read()
            if stderr_text:
                stderr_lines.append(stderr_text)

            # 等待进程结束（剩余超时）
            elapsed = time.time() - start
            remaining = timeout - elapsed
            if remaining > 0:
                proc.wait(timeout=remaining)

        except subprocess.TimeoutExpired:
            timed_out = True
            # 三段式清理
            proc.terminate()
            try:
                proc.wait(timeout=2)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait()

        elapsed_ms = (time.time() - start) * 1000
        text = "".join(stdout_lines).strip()

        # 5. 构造错误信息
        error_parts = []
        if truncated:
            error_parts.append(f"Output truncated at {MAX_STDOUT_BYTES // (1024*1024)}MB")
        if timed_out:
            error_parts.append(f"Timeout after {timeout}s")
        if proc.returncode != 0:
            stderr_str = "".join(stderr_lines).strip()[:500]
            error_parts.append(f"Exit code {proc.returncode}: {stderr_str}")
        error = "; ".join(error_parts) if error_parts else None

        return Response(
            model=self.model,
            backend_name=self.name,
            text=text,
            elapsed_ms=elapsed_ms,
            exit_code=proc.returncode if not timed_out else -1,
            error=error,
        )
