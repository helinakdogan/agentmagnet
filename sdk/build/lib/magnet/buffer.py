"""
Signal Buffer
-------------
Accumulates signals in Redis.
Triggers Reflector when threshold is met.

If no Redis, in-memory fallback works (for development).
"""

from __future__ import annotations
import json
import time
import logging
from typing import Any

logger = logging.getLogger(__name__)

_BUFFER_PREFIX = "vmm:signals:"
_DEFAULT_TTL = 60 * 60 * 24 * 7  # 7 days
_ATOMIC_FLUSH_SCRIPT = """
local vals = redis.call('LRANGE', KEYS[1], 0, -1)
redis.call('DEL', KEYS[1])
return vals
"""


class SignalBuffer:
    def __init__(
        self,
        redis_client: Any | None = None,
        threshold: int = 5,
        ttl: int = _DEFAULT_TTL,
    ):
        self._redis = redis_client
        self.threshold = threshold
        self.ttl = ttl
        self._memory: dict[str, list[dict]] = {}  # fallback

    def push(self, user_id: str, signals: list[dict]) -> int:
        if not signals:
            return self._count(user_id)

        key = _BUFFER_PREFIX + user_id
        enriched = [{"ts": time.time(), **s} for s in signals]

        if self._redis:
            pipe = self._redis.pipeline()
            for sig in enriched:
                pipe.rpush(key, json.dumps(sig))
            pipe.expire(key, self.ttl)
            pipe.execute()
            return self._redis.llen(key)

        self._memory.setdefault(key, []).extend(enriched)
        return len(self._memory[key])

    def should_reflect(self, user_id: str) -> bool:
        return self._count(user_id) >= self.threshold

    def flush(self, user_id: str) -> list[dict]:
        key = _BUFFER_PREFIX + user_id
        if self._redis:
            results = self._redis.eval(_ATOMIC_FLUSH_SCRIPT, 1, key)
            return [json.loads(r) for r in results]

        return self._memory.pop(key, [])

    def peek(self, user_id: str) -> list[dict]:
        key = _BUFFER_PREFIX + user_id
        if self._redis:
            return [json.loads(r) for r in self._redis.lrange(key, 0, -1)]
        return list(self._memory.get(key, []))

    def _count(self, user_id: str) -> int:
        key = _BUFFER_PREFIX + user_id
        if self._redis:
            return self._redis.llen(key)
        return len(self._memory.get(key, []))
