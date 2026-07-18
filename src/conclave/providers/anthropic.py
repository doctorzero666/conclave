"""AnthropicProvider — 通过 Anthropic Messages API 调用 Claude 模型。"""

import json
import time
from typing import Iterator, Optional

import httpx

from ..protocol import ProviderResponse, StreamChunk
from .base import BaseProvider, ProviderCapabilities

# Anthropic Messages API 端点
DEFAULT_API_BASE = "https://api.anthropic.com/v1"
ANTHROPIC_VERSION = "2023-06-01"


class AnthropicProvider(BaseProvider):
    """Anthropic API Provider — 使用 httpx 同步客户端。

    Anthropic Messages API 原生支持 system 字段，
    与 OpenAI 格式略有不同但整体模式相似。
    """

    def __init__(
        self,
        name: str,
        model: str,
        api_key: str,
        api_base: Optional[str] = None,
        timeout: int = 180,
        role: str = "participant",
    ):
        """初始化 AnthropicProvider。

        Args:
            name: provider 名称，如 "anthropic"
            model: 模型名，如 "claude-sonnet-4-20250514"
            api_key: API 密钥
            api_base: 自定义 API 端点，默认 https://api.anthropic.com/v1
            timeout: 默认超时秒数
            role: participant 或 judge
        """
        super().__init__(name=name, model=model, role=role)
        self.api_key = api_key
        self.api_base = (api_base or DEFAULT_API_BASE).rstrip("/")
        self.timeout = timeout

        self._client = httpx.Client(
            timeout=httpx.Timeout(timeout),
            headers={
                "x-api-key": api_key,
                "anthropic-version": ANTHROPIC_VERSION,
                "Content-Type": "application/json",
            },
        )

    # ── 能力声明 ──────────────────────────────────────────────

    def capabilities(self) -> ProviderCapabilities:
        """Anthropic API 支持流式输出。"""
        return ProviderCapabilities(
            supports_streaming=True,
            supports_multiturn=True,
            max_tokens_per_request=200000,
            models=[self.model],
        )

    # ── invoke ─────────────────────────────────────────────────

    def invoke(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        timeout: int = 180,
    ) -> ProviderResponse:
        """同步调用 Anthropic Messages API，返回完整响应。"""
        start = time.time()

        # 构造请求体
        body = self._build_request(prompt, system_prompt, stream=False)

        try:
            response = self._client.post(
                f"{self.api_base}/messages",
                json=body,
                timeout=httpx.Timeout(timeout),
            )
            response.raise_for_status()
            data = response.json()

            elapsed_ms = int((time.time() - start) * 1000)

            # Anthropic 响应格式: content 是数组
            text = ""
            for block in data.get("content", []):
                if block.get("type") == "text":
                    text += block.get("text", "")
            finish_reason = data.get("stop_reason", "end_turn")

            return ProviderResponse(
                provider_name=self.name,
                model=self.model,
                text=text.strip(),
                finish_reason=finish_reason,
                elapsed_ms=elapsed_ms,
            )

        except httpx.HTTPStatusError as e:
            elapsed_ms = int((time.time() - start) * 1000)
            return ProviderResponse(
                provider_name=self.name,
                model=self.model,
                text="",
                finish_reason="error",
                error=f"HTTP {e.response.status_code}: {self._extract_error(e)}",
                elapsed_ms=elapsed_ms,
            )
        except httpx.TimeoutException:
            elapsed_ms = int((time.time() - start) * 1000)
            return ProviderResponse(
                provider_name=self.name,
                model=self.model,
                text="",
                finish_reason="timeout",
                error=f"请求超时 ({timeout}s)",
                elapsed_ms=elapsed_ms,
            )
        except Exception as e:
            elapsed_ms = int((time.time() - start) * 1000)
            return ProviderResponse(
                provider_name=self.name,
                model=self.model,
                text="",
                finish_reason="error",
                error=f"请求失败: {e}",
                elapsed_ms=elapsed_ms,
            )

    # ── stream ─────────────────────────────────────────────────

    def stream(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        timeout: int = 180,
    ) -> Iterator[StreamChunk]:
        """SSE 流式调用 Anthropic Messages API。"""
        body = self._build_request(prompt, system_prompt, stream=True)

        try:
            with self._client.stream(
                "POST",
                f"{self.api_base}/messages",
                json=body,
                timeout=httpx.Timeout(timeout),
            ) as response:
                response.raise_for_status()

                for line in response.iter_lines():
                    if not line.startswith("data: "):
                        continue
                    data_str = line[6:]

                    try:
                        data = json.loads(data_str)
                    except json.JSONDecodeError:
                        continue

                    event_type = data.get("type", "")

                    if event_type == "message_stop":
                        yield StreamChunk(
                            provider_name=self.name,
                            delta="",
                            is_done=True,
                            finish_reason="end_turn",
                        )
                        return

                    if event_type == "content_block_delta":
                        delta_block = data.get("delta", {})
                        if delta_block.get("type") == "text_delta":
                            yield StreamChunk(
                                provider_name=self.name,
                                delta=delta_block.get("text", ""),
                                is_done=False,
                            )

                    # 错误事件
                    if event_type == "error":
                        error_msg = data.get("error", {}).get("message", str(data))
                        yield StreamChunk(
                            provider_name=self.name,
                            delta="",
                            is_done=True,
                            error=error_msg,
                        )
                        return

                # 如果循环结束但没收到 message_stop
                yield StreamChunk(
                    provider_name=self.name,
                    delta="",
                    is_done=True,
                    finish_reason="end_turn",
                )

        except httpx.HTTPStatusError as e:
            yield StreamChunk(
                provider_name=self.name,
                delta="",
                is_done=True,
                error=f"HTTP {e.response.status_code}: {self._extract_error(e)}",
            )
        except Exception as e:
            yield StreamChunk(
                provider_name=self.name,
                delta="",
                is_done=True,
                error=f"流式请求失败: {e}",
            )

    # ── 内部方法 ───────────────────────────────────────────────

    def _build_request(
        self, prompt: str, system_prompt: Optional[str], stream: bool
    ) -> dict:
        """构造 Anthropic Messages API 请求体。"""
        body = {
            "model": self.model,
            "max_tokens": 4096,
            "messages": [{"role": "user", "content": prompt}],
        }
        if system_prompt:
            body["system"] = system_prompt
        if stream:
            body["stream"] = True
        return body

    @staticmethod
    def _extract_error(e: httpx.HTTPStatusError) -> str:
        """从 HTTP 错误响应中提取错误详情。"""
        try:
            body = e.response.json()
            if "error" in body:
                return body["error"].get("message", str(body))
            return str(body)
        except Exception:
            return e.response.text[:200]

    def close(self):
        """关闭 HTTP 客户端。"""
        self._client.close()

    def __del__(self):
        try:
            self._client.close()
        except Exception:
            pass
