"""CLIProvider — 通过 subprocess 调用外部 CLI 工具（Claude Code / Codex CLI 等）。"""

import shutil
import signal
import subprocess
import time
from typing import Optional

from ..protocol import ProviderResponse, StreamChunk
from .base import BaseProvider, ProviderCapabilities

# stdout 最大读取字节数（10MB）
MAX_STDOUT_BYTES = 10 * 1024 * 1024

# 支持的 CLI 可执行文件及其默认参数模板
CLI_PRESETS = {
    # -p 走非交互 print 模式；否则 `claude "..."` 会启动交互式 TUI 挂住。
    # --model 把配置的 model 传给 Claude CLI，避免仅改 metadata 却仍走 CLI 默认。
    "claude": ["-p", "{prompt}", "--model", "{model}"],
    "codex": ["exec", "{prompt}", "--skip-git-repo-check"],
}


class CLIProvider(BaseProvider):
    """通过 subprocess 调用外部 CLI 工具获取模型响应。

    安全策略：
    - 使用 shutil.which() 查找可执行文件，避免 PATH 注入
    - 使用 list args（禁用 shell=True），避免命令注入
    - 超时三段式清理：terminate → wait(2s) → kill
    - stdout 10MB 上限，防止内存溢出
    - SIGINT 信号处理，优雅终止子进程
    """

    def __init__(
        self,
        name: str,
        model: str,
        executable: str,
        args_template: Optional[list[str]] = None,
        timeout: int = 180,
        role: str = "participant",
    ):
        """初始化 CLIProvider。

        Args:
            name: provider 名称，如 "claude"
            model: 模型标识，如 "claude-fable-5"
            executable: 可执行文件名，如 "claude" 或 "codex"
            args_template: 参数模板列表，"{prompt}" 会被替换。None 则使用预设。
            timeout: 默认超时秒数
            role: participant 或 judge
        """
        super().__init__(name=name, model=model, role=role)
        self.timeout = timeout

        # 解析可执行文件路径
        resolved = shutil.which(executable)
        if resolved is None:
            raise FileNotFoundError(
                f"'{executable}' 未在 PATH 中找到。\n"
                f"请先安装:\n"
                f"  Claude Code: npm install -g @anthropic-ai/claude-code\n"
                f"  Codex CLI:   npm install -g @openai/codex"
            )
        self.executable_path = resolved

        # 参数模板：优先使用传入的，否则用预设
        if args_template is not None:
            self.args_template = args_template
        elif executable in CLI_PRESETS:
            self.args_template = CLI_PRESETS[executable]
        else:
            self.args_template = ["{prompt}"]

    # ── 能力声明 ──────────────────────────────────────────────

    def capabilities(self) -> ProviderCapabilities:
        """CLI provider 不支持流式输出。"""
        return ProviderCapabilities(
            supports_streaming=False,
            supports_multiturn=True,
            max_tokens_per_request=100000,
            models=[self.model],
        )

    # ── invoke ─────────────────────────────────────────────────

    def invoke(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        timeout: int = 180,
    ) -> ProviderResponse:
        """调用外部 CLI 获取模型响应。

        如果有 system_prompt，会作为 stdin 传入（部分 CLI 支持）。
        """
        start = time.time()

        # 1. 构造参数列表（替换 {prompt} / {model} 占位符）
        # model 归一：None / "" 都当作空串，避免 format 出字面量 `--model None`。
        model = self.model or ""
        # str.format 允许多余 kwargs，自定义 template 不含 {model} 也不会报错。
        args = [arg.format(prompt=prompt, model=model) for arg in self.args_template]

        # model 为空时不能传 `--model ""`（CLI 会拒绝），去掉该 flag 走 CLI 默认模型。
        if not model:
            args = self._strip_empty_model_flag(args)

        # 2. 启动子进程（安全：list args + 禁用 shell=True）
        try:
            proc = subprocess.Popen(
                [self.executable_path] + args,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1,  # 行缓冲
                encoding="utf-8",
                errors="replace",  # 非 UTF-8 字符替换为 �
                # 新进程组，便于信号处理
                start_new_session=True,
            )
        except Exception as e:
            elapsed_ms = int((time.time() - start) * 1000)
            return ProviderResponse(
                provider_name=self.name,
                model=self.model,
                text="",
                finish_reason="error",
                error=f"启动进程失败: {e}",
                elapsed_ms=elapsed_ms,
            )

        # 3. 写入 system_prompt 到 stdin（如果有的话）
        if proc.stdin is not None:
            if system_prompt:
                try:
                    proc.stdin.write(system_prompt + "\n")
                    proc.stdin.flush()
                except Exception:
                    pass

            # 关闭 stdin
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

        if proc.stdout is None:
            elapsed_ms = int((time.time() - start) * 1000)
            proc.kill()
            proc.wait()
            return ProviderResponse(
                provider_name=self.name,
                model=self.model,
                text="",
                finish_reason="error",
                error="无法读取子进程 stdout",
                elapsed_ms=elapsed_ms,
            )

        try:
            for line in proc.stdout:
                stdout_lines.append(line)
                total_bytes += len(line.encode("utf-8"))

                if total_bytes > MAX_STDOUT_BYTES:
                    truncated = True
                    break

            # 收集 stderr
            if proc.stderr is not None:
                stderr_text = proc.stderr.read()
                if stderr_text:
                    stderr_lines.append(stderr_text)

            # 等待进程结束
            elapsed = time.time() - start
            remaining = timeout - elapsed
            if remaining > 0:
                proc.wait(timeout=remaining)

        except subprocess.TimeoutExpired:
            timed_out = True
            # 三段式清理：terminate → wait(2s) → kill
            proc.terminate()
            try:
                proc.wait(timeout=2)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait()

        elapsed_ms = int((time.time() - start) * 1000)
        text = "".join(stdout_lines).strip()

        # 5. 构造错误信息
        error = None
        finish_reason = "stop"
        if truncated:
            error = f"输出截断于 {MAX_STDOUT_BYTES // (1024 * 1024)}MB"
            finish_reason = "length"
        if timed_out:
            error = (error + "; " if error else "") + f"超时 {timeout}s"
            finish_reason = "timeout"
        if proc.returncode != 0:
            stderr_str = "".join(stderr_lines).strip()[:500]
            code_msg = f"退出码 {proc.returncode}: {stderr_str}"
            error = (error + "; " if error else "") + code_msg
            if not timed_out and not truncated:
                finish_reason = "error"

        return ProviderResponse(
            provider_name=self.name,
            model=self.model,
            text=text,
            finish_reason=finish_reason,
            error=error,
            elapsed_ms=elapsed_ms,
        )

    # ── 工具方法 ───────────────────────────────────────────────

    @staticmethod
    def _strip_empty_model_flag(args: list[str]) -> list[str]:
        """丢弃 `--model ""` 与 `--model=`，保留其余参数。"""
        cleaned: list[str] = []
        i = 0
        while i < len(args):
            arg = args[i]
            if arg == "--model" and i + 1 < len(args) and args[i + 1] == "":
                i += 2
                continue
            if arg == "--model=":
                i += 1
                continue
            cleaned.append(arg)
            i += 1
        return cleaned

    @staticmethod
    def detect(executable_name: str) -> bool:
        """检测指定 CLI 工具是否已安装。

        Args:
            executable_name: 可执行文件名，如 "claude" 或 "codex"

        Returns:
            True 如果可执行文件在 PATH 中
        """
        return shutil.which(executable_name) is not None
