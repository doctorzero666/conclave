"""配置系统 — ~/.conclave/config.yaml 管理"""

import os
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import yaml


CONCLAVE_DIR = Path.home() / ".conclave"
CONFIG_PATH = CONCLAVE_DIR / "config.yaml"
DEFAULT_CACHE_DIR = CONCLAVE_DIR / "cache"


@dataclass
class ProviderConfig:
    """单个 provider 的配置"""
    name: str                              # unique id: "claude", "openai-gpt5"
    provider_type: str = "cli"             # "cli" / "openai" / "anthropic" / "deepseek"
    model: str = ""                        # model name/id
    executable: Optional[str] = None       # for CLI provider
    args_template: Optional[list[str]] = None  # for CLI provider: ["{prompt}"]
    api_key: Optional[str] = None          # for API providers
    api_base: Optional[str] = None         # custom endpoint
    timeout: int = 180
    role: str = "participant"


@dataclass
class Config:
    """conclave 全局配置"""
    # Default backends
    default_providers: list[str] = field(default_factory=lambda: ["claude", "codex"])
    judge_provider: str = "claude"
    default_rounds: int = 1

    # Cache
    cache_enabled: bool = True
    cache_dir: str = str(DEFAULT_CACHE_DIR)
    cache_ttl_seconds: int = 300
    cache_max_entries: int = 1000

    # Limits
    max_cost_usd: float = 0.0  # 0 = unlimited
    max_workers: int = 8
    timeout: int = 180

    # Providers
    providers: dict[str, ProviderConfig] = field(default_factory=dict)

    # Synthesizer
    synthesis_enabled: bool = True

    def get_provider(self, name: str) -> Optional[ProviderConfig]:
        return self.providers.get(name)

    def participant_providers(self) -> list[ProviderConfig]:
        return [p for p in self.providers.values() if p.role == "participant"]


# ─── Load / Save ──────────────────────────────────────────────────────

def _default_providers() -> dict[str, ProviderConfig]:
    """内置默认 provider 配置"""
    providers = {}
    providers["claude"] = ProviderConfig(
        name="claude", provider_type="cli", model="claude-fable-5",
        executable="claude", args_template=["-p", "{prompt}", "--model", "{model}"],
    )
    providers["codex"] = ProviderConfig(
        name="codex", provider_type="cli", model="gpt-5.6-sol",
        executable="codex", args_template=["exec", "{prompt}", "--skip-git-repo-check"],
    )
    # API providers — 环境变量中没有 key 则跳过
    if os.environ.get("OPENAI_API_KEY"):
        providers["openai"] = ProviderConfig(
            name="openai", provider_type="openai", model="gpt-5.6-sol",
            api_key=os.environ["OPENAI_API_KEY"],
        )
    if os.environ.get("ANTHROPIC_API_KEY"):
        providers["anthropic"] = ProviderConfig(
            name="anthropic", provider_type="anthropic", model="claude-sonnet-4",
            api_key=os.environ["ANTHROPIC_API_KEY"],
        )
    if os.environ.get("DEEPSEEK_API_KEY"):
        providers["deepseek"] = ProviderConfig(
            name="deepseek", provider_type="deepseek", model="deepseek-v3",
            api_key=os.environ["DEEPSEEK_API_KEY"],
        )
    return providers


def load_config() -> Config:
    """加载配置：config.yaml → 合并默认值"""
    if not CONFIG_PATH.exists():
        return Config(providers=_default_providers())

    with open(CONFIG_PATH) as f:
        raw = yaml.safe_load(f) or {}

    config = Config()

    # Top-level fields
    for key in ["default_providers", "judge_provider", "default_rounds",
                "cache_enabled", "cache_dir", "cache_ttl_seconds",
                "cache_max_entries", "max_cost_usd", "max_workers",
                "timeout", "synthesis_enabled"]:
        if key in raw:
            setattr(config, key, raw[key])

    # Providers — merge with defaults
    config.providers = _default_providers()
    if "providers" in raw:
        for name, pdata in raw["providers"].items():
            pc = ProviderConfig(name=name, **pdata)
            # 迁移：老 claude args_template 不带 --model，配置的 model 仅是
            # metadata，实际 CLI 走的是默认模型。只迁移真正的 legacy 默认配置
            # (name=claude & executable=claude & args_template ∈ 已知旧默认)。
            # 自定义 executable (wrapper) 或自定义 args_template（尤其已含
            # --model 或其它 flag）都必须原样保留。
            if (
                pc.name == "claude"
                and pc.executable == "claude"
                and pc.args_template in (["{prompt}"], ["-p", "{prompt}"])
            ):
                pc.args_template = ["-p", "{prompt}", "--model", "{model}"]
            config.providers[name] = pc

    return config


def init_config(interactive: bool = False) -> Config:
    """初始化配置文件"""
    CONCLAVE_DIR.mkdir(parents=True, exist_ok=True)
    (CONCLAVE_DIR / "cache").mkdir(exist_ok=True)

    config = Config(providers=_default_providers())

    # Serialize
    out = {
        "default_providers": config.default_providers,
        "judge_provider": config.judge_provider,
        "default_rounds": config.default_rounds,
        "cache_enabled": True,
        "cache_dir": "~/.conclave/cache",
        "cache_ttl_seconds": 300,
        "cache_max_entries": 1000,
        "max_cost_usd": 0,
        "max_workers": 8,
        "timeout": 180,
        "synthesis_enabled": True,
        "providers": {
            name: {
                k: v for k, v in {
                    "provider_type": p.provider_type,
                    "model": p.model,
                    "executable": p.executable,
                    "args_template": p.args_template,
                    "role": p.role,
                }.items() if v is not None
            }
            for name, p in config.providers.items()
        }
    }

    with open(CONFIG_PATH, "w") as f:
        yaml.safe_dump(out, f, default_flow_style=False, allow_unicode=True,
                        sort_keys=False)

    return config


def validate_config(config: Config) -> list[str]:
    """验证配置文件，返回警告列表"""
    warnings = []

    if not config.providers:
        warnings.append("没有配置任何 provider。运行 'conclave config init' 初始化。")

    api_providers = [p for p in config.providers.values()
                     if p.provider_type in ("openai", "anthropic", "deepseek")]
    for p in api_providers:
        if p.api_key and "sk-" not in str(p.api_key) and "sk-" not in str(p.api_key):
            pass  # key might be env var reference
        if p.api_key and len(str(p.api_key)) < 10:
            warnings.append(f"Provider '{p.name}' 的 api_key 似乎无效（长度<10）。")

    if config.judge_provider not in config.providers:
        warnings.append(
            f"Judge provider '{config.judge_provider}' 未在 providers 中定义。"
        )

    return warnings
