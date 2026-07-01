"""
UsageCounter
------------
Billing groundwork: meters memory writes and retrieval calls.
Does NOT enforce limits — metering only.

Storage:
  - Local (no Redis): ~/.agent-magnet/usage.json
  - Redis: hash at  magnet:usage:{user_id}

TODO — future enforcement point:
  check_usage_limit() currently always returns True (allowed).
  When hosted mode is introduced (MAGNET_API_KEY), plug quota checks here.
  Local mode (no API key) is unlimited — runs on the user's own storage/compute.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

_LOCAL_FILE = Path.home() / ".agent-magnet" / "usage.json"
_REDIS_PREFIX = "magnet:usage:"


class UsageCounter:
    def __init__(self, redis_client: Any = None, user_id: str = "default"):
        self._redis = redis_client
        self._user_id = user_id

    # ── Write ─────────────────────────────────────────────────────────────────

    def record_write(self, project_id: str = "default") -> None:
        self._inc(f"writes:{project_id}")
        self._inc("writes:total")

    def record_retrieval(self, project_id: str = "default") -> None:
        self._inc(f"retrievals:{project_id}")
        self._inc("retrievals:total")

    def _inc(self, metric: str) -> None:
        if self._redis:
            try:
                self._redis.hincrby(f"{_REDIS_PREFIX}{self._user_id}", metric, 1)
            except Exception:
                self._inc_local(metric)
        else:
            self._inc_local(metric)

    def _inc_local(self, metric: str) -> None:
        try:
            data: dict = {}
            if _LOCAL_FILE.exists():
                data = json.loads(_LOCAL_FILE.read_text(encoding="utf-8"))
            user = data.setdefault(self._user_id, {})
            user[metric] = user.get(metric, 0) + 1
            _LOCAL_FILE.parent.mkdir(parents=True, exist_ok=True)
            _LOCAL_FILE.write_text(json.dumps(data, indent=2, ensure_ascii=False))
        except Exception as e:
            logger.debug(f"[usage] local increment failed: {e}")

    # ── Read ──────────────────────────────────────────────────────────────────

    def get_stats(self) -> dict:
        if self._redis:
            try:
                raw = self._redis.hgetall(f"{_REDIS_PREFIX}{self._user_id}")
                if raw:
                    return {k: int(v) for k, v in raw.items()}
            except Exception:
                pass
        try:
            if _LOCAL_FILE.exists():
                data = json.loads(_LOCAL_FILE.read_text(encoding="utf-8"))
                return data.get(self._user_id, {})
        except Exception:
            pass
        return {}

    # ── Enforcement hook (TODO: plug tier limits here) ────────────────────────

    def check_usage_limit(self, metric: str = "writes:total") -> bool:  # noqa: ARG002
        """
        Always returns True (allowed) — enforcement not yet active.

        FUTURE: if MAGNET_API_KEY is set (hosted mode), fetch quota from Magnet API
        and return False when exceeded. Local mode (no API key) is always unlimited.
        """
        return True
