"""Provider 基类 — 所有 provider 必须实现此接口"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Iterator, Optional

from conclave.protocol import ProviderResponse, StreamChunk


@dataclass
class ProviderCapabilities:
    """Provider 的能力声明"""
    supports_streaming: bool = False
    supports_multiturn: bool = True
    max_tokens_per_request: int = 100000
    models: list[str] = None

    def __post_init__(self):
        if self.models is None:
            self.models = []


class BaseProvider(ABC):
    """所有 provider 的抽象基类"""

    def __init__(self, name: str, model: str, role: str = "participant"):
        self.name = name
        self.model = model
        self.role = role  # participant or judge

    @abstractmethod
    def capabilities(self) -> ProviderCapabilities:
        """返回 provider 能力"""
        ...

    @abstractmethod
    def invoke(self, prompt: str, system_prompt: Optional[str] = None,
               timeout: int = 180) -> ProviderResponse:
        """同步调用，返回完整响应"""
        ...

    def stream(self, prompt: str, system_prompt: Optional[str] = None,
               timeout: int = 180) -> Iterator[StreamChunk]:
        """流式调用（默认实现：invoke → 返回单个chunk）"""
        response = self.invoke(prompt, system_prompt, timeout)
        if response.ok:
            yield StreamChunk(
                provider_name=self.name,
                delta=response.text,
                is_done=True,
                finish_reason=response.finish_reason,
            )
        else:
            yield StreamChunk(
                provider_name=self.name,
                delta="",
                is_done=True,
                error=response.error,
            )

    def invoke_with_context(self, prompt: str, context: list[ProviderResponse],
                            system_prompt: Optional[str] = None,
                            timeout: int = 180) -> ProviderResponse:
        """
        带上下文的调用 — 用于多轮审议
        默认实现：把其他模型的回答拼接到 system prompt
        """
        ctx_text = "\n\n".join(
            f"--- {r.provider_name} ({r.model})的回答 ---\n{r.text}"
            for r in context if r.ok and r.provider_name != self.name
        )
        full_system = system_prompt or ""
        if ctx_text:
            full_system += f"\n\n## 其他模型的回答（供参考和批评）\n\n{ctx_text}"
        return self.invoke(prompt, full_system, timeout)

    def __repr__(self):
        return f"{self.__class__.__name__}(name={self.name!r}, model={self.model!r})"
