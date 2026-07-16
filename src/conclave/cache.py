"""内存缓存，model-aware key，FIFO + TTL 淘汰，线程安全。"""

import hashlib
import threading
import time
from typing import Optional


class Cache:
    """FIFO + TTL 内存缓存。

    缓存 key = SHA256(sorted(backend_names) + sorted(args_templates) + prompt)
    防止相同 prompt 不同 backend 组合 / 不同 args 互相覆盖。
    淘汰策略：写入时淘汰最老的条目（FIFO）+ 读取时淘汰过期条目（TTL）。

    线程安全。
    """

    def __init__(self, ttl: int = 300, max_size: int = 64):
        """
        Args:
            ttl: 缓存有效期（秒），默认 5 分钟
            max_size: 最大缓存条目数
        """
        self._ttl = ttl
        self._max_size = max_size
        self._store: dict[str, tuple[float, list]] = {}  # key → (timestamp, responses)
        self._lock = threading.Lock()

    # ---- 公开 API ----

    @property
    def size(self) -> int:
        """当前缓存条目数。"""
        with self._lock:
            return len(self._store)

    def get(
        self,
        prompt: str,
        backend_names: list[str],
        args_templates: list[str],
    ) -> Optional[list]:
        """获取缓存的响应列表。

        Args:
            prompt: 用户原始 prompt
            backend_names: backend 名字列表（如 ["claude", "codex"]）
            args_templates: 对应的 args_template 字符串列表

        Returns:
            缓存的 Response 列表，未命中或过期返回 None
        """
        key = self._make_key(prompt, backend_names, args_templates)
        with self._lock:
            entry = self._store.get(key)
            if entry is None:
                return None
            ts, responses = entry
            if time.time() - ts > self._ttl:
                del self._store[key]
                return None
            return responses

    def set(
        self,
        prompt: str,
        backend_names: list[str],
        args_templates: list[str],
        responses: list,
    ) -> None:
        """写入缓存。

        Args:
            prompt: 用户原始 prompt
            backend_names: backend 名字列表
            args_templates: 对应的 args_template 字符串列表
            responses: 要缓存的 Response 列表
        """
        key = self._make_key(prompt, backend_names, args_templates)
        with self._lock:
            # LRU 淘汰
            if len(self._store) >= self._max_size:
                oldest = min(self._store, key=lambda k: self._store[k][0])
                del self._store[oldest]
            self._store[key] = (time.time(), responses)

    # ---- 内部 ----

    def _make_key(
        self,
        prompt: str,
        backend_names: list[str],
        args_templates: list[str],
    ) -> str:
        """生成缓存 key。"""
        # 将 backend_names 和 args_templates 打包后一起排序，
        # 确保 --backends claude,codex 和 --backends codex,claude 命中同一缓存
        pairs = sorted(zip(backend_names, args_templates))
        sorted_names = [p[0] for p in pairs]
        sorted_templates = [p[1] for p in pairs]
        raw = (
            "|".join(sorted_names)
            + "|"
            + "|".join(sorted_templates)
            + "|"
            + prompt
        )
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()
