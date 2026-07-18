"""Provider 工厂 — 根据配置创建 provider 实例"""

from typing import Optional

from ..config import ProviderConfig
from .base import BaseProvider
from .cli import CLIProvider
from .openai import OpenAIProvider
from .anthropic import AnthropicProvider
from .deepseek import DeepSeekProvider

__all__ = ["create_provider", "detect_available"]


def create_provider(config: ProviderConfig) -> BaseProvider:
    """根据 ProviderConfig 创建对应的 provider 实例

    Args:
        config: ProviderConfig 对象

    Returns:
        BaseProvider 子类实例

    Raises:
        ValueError: 未知 provider_type
        FileNotFoundError: CLI 可执行文件不存在
    """
    provider_type = config.provider_type.lower()

    if provider_type == "cli":
        if not config.executable:
            raise ValueError(f"CLI provider '{config.name}' 缺少 executable")
        return CLIProvider(
            name=config.name,
            model=config.model,
            executable=config.executable,
            args_template=config.args_template,
            timeout=config.timeout or 180,
        )

    elif provider_type == "openai":
        api_key = config.api_key or ""
        return OpenAIProvider(
            name=config.name,
            model=config.model,
            api_key=api_key,
            api_base=config.api_base,
            timeout=config.timeout or 180,
        )

    elif provider_type == "anthropic":
        api_key = config.api_key or ""
        return AnthropicProvider(
            name=config.name,
            model=config.model,
            api_key=api_key,
            timeout=config.timeout or 180,
        )

    elif provider_type == "deepseek":
        api_key = config.api_key or ""
        return DeepSeekProvider(
            name=config.name,
            model=config.model,
            api_key=api_key,
            api_base=config.api_base,
            timeout=config.timeout or 180,
        )

    else:
        raise ValueError(
            f"未知 provider_type '{provider_type}'。支持: cli, openai, anthropic, deepseek"
        )


def detect_available() -> list[str]:
    """检测当前环境中可用的 provider

    通过检查 CLI 工具是否安装、环境变量是否设置来判断。

    Returns:
        可用的 provider 名称列表
    """
    available = []

    # CLI
    for name in ["claude", "codex"]:
        try:
            CLIProvider(name=name, model=name, executable=name)
            available.append(name)
        except FileNotFoundError:
            pass

    # API — 检查环境变量
    import os
    if os.environ.get("OPENAI_API_KEY"):
        available.append("openai")
    if os.environ.get("ANTHROPIC_API_KEY"):
        available.append("anthropic")
    if os.environ.get("DEEPSEEK_API_KEY"):
        available.append("deepseek")

    return available
