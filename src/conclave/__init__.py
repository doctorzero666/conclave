"""conclave — multi-LLM deliberation terminal tool.

Like a conclave of cardinals, but for AI models.
"""

__version__ = "0.1.0"

from dataclasses import dataclass
from typing import Optional


@dataclass
class PanelResult:
    """一次 panel 调用的完整结果。"""
    prompt: str
    responses: list  # list[Response]
    from_cache: bool = False
