"""OpenAIProvider — 通过 OpenAI Chat Completions API 调用模型。"""

import json
import time
from typing import Iterator, Optional

import httpx

from ..protocol import ProviderResponse, StreamChunk
from .base import BaseProvider, ProviderCapabilities

# 默认 API 端点
DEFAULT_API_BASE = "https://api.openai.com/v1"


class OpenAIProvider(BaseProvider):
    """OpenAI API Provider — 使用 httpx 同步客户端。

    支持：
    - 标准 /v1/chat/completions 端点
    - 自定义 api_base（可用于兼容 API）
    - SSE 流式输出
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
        """初始化 OpenAIProvider。

        Args:
            name: provider 名称，如 "openai"
            model: 模型名，如 "gpt-5.6-sol"
            api_key: API 密钥
            api_base: 自定义 API 端点，默认 https://api.openai.com/v1
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
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
        )

    # ── 能力声明 ──────────────────────────────────────────────

    def capabilities(self) -> ProviderCapabilities:
        """OpenAI API 支持流式输出。"""
        return ProviderCapabilities(
            supports_streaming=True,
            supports_multiturn=True,
            max_tokens_per_request=128000,
            models=[self.model],
        )

    # ── invoke ─────────────────────────────────────────────────

    def invoke(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        timeout: int = 180,
    ) -> ProviderResponse:
        """同步调用 OpenAI Chat API，返回完整响应。"""
        start = time.time()

        # 构造消息列表
        messages = self._build_messages(prompt, system_prompt)

        try:
            response = self._client.post(
                f"{self.api_base}/chat/completions",
                json={
                    "model": self.model,
                    "messages": messages,
                    "stream": False,
                },
                timeout=httpx.Timeout(timeout),
            )
            response.raise_for_status()
            data = response.json()

            elapsed_ms = int((time.time() - start) * 1000)
            choice = data["choices"][0]
            text = choice["message"]["content"]
            finish_reason = choice.get("finish_reason", "stop")

            return ProviderResponse(
                provider_name=self.name,
                model=self.model,
                text=text.strip() if text else "",
                finish_reason=finish_reason,
                elapsed_ms=elapsed_ms,
            )

        except httpx.HTTPStatusError as e:
            elapsed_ms = int((time.time() - start) * 1000)
            error_detail = self._extract_error(e)
            return ProviderResponse(
                provider_name=self.name,
                model=self.model,
                text="",
                finish_reason="error",
                error=f"HTTP {e.response.status_code}: {error_detail}",
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
        """SSE 流式调用 OpenAI Chat API。"""
        messages = self._build_messages(prompt, system_prompt)

        try:
            with self._client.stream(
                "POST",
                f"{self.api_base}/chat/completions",
                json={
                    "model": self.model,
                    "messages": messages,
                    "stream": True,
                    "stream_options": {"include_usage": False},
                },
                timeout=httpx.Timeout(timeout),
            ) as response:
                response.raise_for_status()

                for line in response.iter_lines():
                    if not line.startswith("data: "):
                        continue
                    data_str = line[6:]  # 去掉 "data: " 前缀
                    if data_str == "[DONE]":
                        yield StreamChunk(
                            provider_name=self.name,
                            delta="",
                            is_done=True,
                            finish_reason="stop",
                        )
                        return

                    try:
                        data = json.loads(data_str)
                    except json.JSONDecodeError:
                        continue

                    delta = data.get("choices", [{}])[0].get("delta", {})
                    content = delta.get("content", "")
                    finish_reason = data.get("choices", [{}])[0].get("finish_reason")

                    if finish_reason:
                        yield StreamChunk(
                            provider_name=self.name,
                            delta=content,
                            is_done=True,
                            finish_reason=finish_reason,
                        )
                        return
                    elif content:
                        yield StreamChunk(
                            provider_name=self.name,
                            delta=content,
                            is_done=False,
                        )

        except httpx.HTTPStatusError as e:
            error_detail = self._extract_error(e)
            yield StreamChunk(
                provider_name=self.name,
                delta="",
                is_done=True,
                error=f"HTTP {e.response.status_code}: {error_detail}",
            )
        except Exception as e:
            yield StreamChunk(
                provider_name=self.name,
                delta="",
                is_done=True,
                error=f"流式请求失败: {e}",
            )

    # ── 内部方法 ───────────────────────────────────────────────

    def _build_messages(self, prompt: str, system_prompt: Optional[str]) -> list[dict]:
        """构造 OpenAI 格式的消息列表。"""
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        return messages

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
