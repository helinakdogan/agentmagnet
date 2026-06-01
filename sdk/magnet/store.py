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

BASE_DECAY = {
    "permanent": 0.005,
    "contextual": 0.02,
    "transient": 0.10,
}


def effective_confidence(pref: dict) -> float:
    days = (time.time() - pref.get("valid_from", time.time())) / 86400
    recall_count = pref.get("recall_count", 0)
    pref_type = pref.get("preference_type", "contextual")

    base = BASE_DECAY.get(pref_type, 0.02)
    usage_factor = 1 / (1 + recall_count * 0.1)
    effective_decay = base * usage_factor

    return pref.get("confidence", 0.8) * ((1 - effective_decay) ** days)


class ProfileStore:
    def __init__(self, redis_client: Any | None = None, ttl: int = _DEFAULT_TTL):
        self._redis = redis_client
        self.ttl = ttl
        self._memory: dict[str, dict] = {}

    def save(self, user_id: str, profile: dict) -> None:
        print(f"[SAVE] saving profile with {len(profile.get('preferences', []))} preferences")
        key = _PROFILE_PREFIX + user_id
        history_key = f"vmm:profile_history:{user_id}"
        if "preferences" in profile:
            profile = {**profile, "preferences": self._resolve_conflicts(profile["preferences"])}
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

    def _resolve_conflicts(self, preferences: list) -> list:
        """Remove older entries when a subject has conflicting opposite relations."""
        opposite = {"prefers": "dislikes", "dislikes": "prefers"}
        seen: dict[str, str] = {}  # subject -> relation, last wins
        resolved = []
        for pref in reversed(preferences):
            if not isinstance(pref, dict):
                continue
            subject = pref.get("subject", "").lower().strip()
            relation = pref.get("relation", "")
            existing_relation = seen.get(subject)
            if existing_relation and existing_relation == opposite.get(relation):
                continue  # skip: newer conflicting entry already recorded
            seen[subject] = relation
            resolved.append(pref)
        return list(reversed(resolved))

    def _stamp_preferences(self, profile: dict) -> dict:
        now = time.time()

        def _stamp(pref: dict) -> dict:
            if "valid_from" not in pref:
                return {**pref, "valid_from": now, "decay_rate": pref.get("decay_rate", 0.02)}
            return pref

        gp = profile.get("global_preferences", {})
        cp = profile.get("contextual_profiles", {})
        prefs = profile.get("preferences", [])
        return {
            **profile,
            "global_preferences": {
                k: _stamp(v) if isinstance(v, dict) and "value" in v else v
                for k, v in gp.items()
            },
            "contextual_profiles": {
                ctx: {
                    k: _stamp(v) if isinstance(v, dict) and "value" in v else v
                    for k, v in ctx_prefs.items()
                }
                for ctx, ctx_prefs in cp.items()
            },
            "preferences": [
                _stamp(p) if isinstance(p, dict) else p
                for p in prefs
            ],
        }
