"""
Profile Store
-------------
Stores user behavioral profiles.
In Redis if available, otherwise in-memory.
"""

from __future__ import annotations
import json
import logging
import time
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
        history_key = f"vmm:profile_history:{user_id}"
        data = json.dumps(self._stamp_preferences(profile), ensure_ascii=False)
        if self._redis:
            self._redis.setex(key, self.ttl, data)
            # Log history to track profile evolution (max 50 history entries)
            self._redis.lpush(history_key, data)
            self._redis.ltrim(history_key, 0, 49)
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

    def _stamp_preferences(self, profile: dict) -> dict:
        now = time.time()

        def _stamp(pref: dict) -> dict:
            if "valid_from" not in pref:
                return {**pref, "valid_from": now, "decay_rate": pref.get("decay_rate", 0.02)}
            return pref

        gp = profile.get("global_preferences", {})
        cp = profile.get("contextual_profiles", {})
        return {
            **profile,
            "global_preferences": {
                k: _stamp(v) if isinstance(v, dict) and "value" in v else v
                for k, v in gp.items()
            },
            "contextual_profiles": {
                ctx: {
                    k: _stamp(v) if isinstance(v, dict) and "value" in v else v
                    for k, v in prefs.items()
                }
                for ctx, prefs in cp.items()
            },
        }
