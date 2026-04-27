"""
Profile Store
-------------
Stores user behavioral profiles.
In Redis if available, otherwise in-memory.
"""

from __future__ import annotations
import json
import logging
from typing import Any

logger = logging.getLogger(__name__)

_PROFILE_PREFIX = "vmm:profile:"
_DEFAULT_TTL = 60 * 60 * 24 * 30  # 30 days


class ProfileStore:
    def __init__(self, redis_client: Any | None = None, ttl: int = _DEFAULT_TTL):
        self._redis = redis_client
        self.ttl = ttl
        self._memory: dict[str, dict] = {}

    def save(self, user_id: str, profile: dict) -> None:
        key = _PROFILE_PREFIX + user_id
        data = json.dumps(profile, ensure_ascii=False)
        if self._redis:
            self._redis.setex(key, self.ttl, data)
        else:
            self._memory[key] = profile
        logger.debug(f"Profile saved: {user_id}")

    def load(self, user_id: str) -> dict | None:
        key = _PROFILE_PREFIX + user_id
        if self._redis:
            raw = self._redis.get(key)
            return json.loads(raw) if raw else None
        return self._memory.get(key)

    def delete(self, user_id: str) -> None:
        key = _PROFILE_PREFIX + user_id
        if self._redis:
            self._redis.delete(key)
        else:
            self._memory.pop(key, None)

    def exists(self, user_id: str) -> bool:
        return self.load(user_id) is not None
