"""Backend 抽象基类 + Response 数据类 + 异常体系。"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Callable, Optional


@dataclass
class Response:
    """单个模型 backend 的一次调用结果。

    Attributes:
        model: 模型标识，如 "claude-sonnet-4"
        backend_name: backend 名字，如 "claude"
        text: 模型输出文本（已截断到 10MB 以内）
        elapsed_ms: 调用耗时（毫秒）
        exit_code: 子进程退出码，0 表示成功
        cache_hit: 是否来自缓存
        error: 错误信息（stdout 截断 + stderr 截断）
    """
    model: str
    backend_name: str
    text: str = ""
    elapsed_ms: float = 0.0
    exit_code: int = 0
    cache_hit: bool = False
    error: Optional[str] = None


class BackendError(Exception):
    """Backend 调用失败（单个 backend 层级）。"""


class BackendNotFoundError(BackendError):
    """可执行文件未找到。应含安装指引。"""


class AllBackendsFailedError(BackendError):
    """所有 backend 全部失败。"""

    def __init__(self, errors: dict[str, str]):
        self.errors = errors  # backend_name → error message
        msg = "\n".join(f"  [{name}] {err}" for name, err in errors.items())
        super().__init__(f"All backends failed:\n{msg}")


class Backend(ABC):
    """所有模型后端的统一接口。

    每个 Backend 封装一种调用模型的方式（CLI subprocess / HTTP API / ...）。
    """

    @abstractmethod
    def invoke(
        self,
        prompt: str,
        timeout: float = 120,
        on_chunk: Optional[Callable[[str], None]] = None,
    ) -> Response:
        """调用模型并返回响应。

        Args:
            prompt: 发送给模型的完整提示词
            timeout: 超时秒数
            on_chunk: 可选回调，每收到一行 stdout 时调用，用于实时进度展示

        Returns:
            Response 对象，含 model/backend_name/text/elapsed_ms/exit_code/error

        Raises:
            BackendError: 调用失败
        """
        ...
