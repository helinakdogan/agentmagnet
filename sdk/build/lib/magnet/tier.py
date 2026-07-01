"""
Tier resolver — Agent Magnet
-----------------------------
FREE  (default, no key): local SQLite, single-user, BYOK, compression
PREMIUM (mg_sk_ key)   : team memory, hosted storage, compression analytics

Key validation is format-only for now.
TODO: validate against remote billing API at https://agentmagnet.app/v1/keys/validate
      POST {"key": key} → {"valid": true, "tier": "premium", "plan": "..."}
"""

from __future__ import annotations

import logging
import os

logger = logging.getLogger(__name__)

_FREE    = "free"
_PREMIUM = "premium"

# Features that require a premium key
PREMIUM_FEATURES: frozenset[str] = frozenset({
    "team_memory",
    "hosted_storage",
    "compression_stats",
    "promote_to_team",
})

_UPSELL_MSG = (
    "This is a premium feature. "
    "Get a key at agentmagnet.app to enable team memory and hosted storage."
)

_PREMIUM_REQUIRED_RESPONSE = {
    "error": "premium_required",
    "message": "Team memory requires a Magnet API key. Get one at agentmagnet.app",
}


def get_tier() -> str:
    """
    Resolve tier from MAGNET_API_KEY env var.

    Returns 'premium' if key starts with 'mg_sk_' and has length > 10.
    Returns 'free' otherwise.

    TODO: replace format-only check with remote validation call.
    """
    key = os.environ.get("MAGNET_API_KEY", "").strip()
    if key.startswith("mg_sk_") and len(key) > 10:
        # TODO: validate remotely — POST /v1/keys/validate
        return _PREMIUM
    return _FREE


def is_premium() -> bool:
    return get_tier() == _PREMIUM


def check_premium_feature(feature_name: str) -> bool:
    """
    Return True if current tier allows feature_name.
    Logs a friendly upsell message on the free tier for premium features.
    """
    if feature_name not in PREMIUM_FEATURES:
        return True
    if is_premium():
        logger.debug(f"[tier] premium feature '{feature_name}' granted")
        return True
    logger.info(f"[tier] '{feature_name}' is premium — {_UPSELL_MSG}")
    return False


def premium_required_response() -> dict:
    """Standard error dict returned when a premium tool is called on free tier."""
    return dict(_PREMIUM_REQUIRED_RESPONSE)


def resolve_storage_mode() -> str:
    """
    Returns a string describing the active storage mode.
    Used for startup logging.

    Priority:
      1. MAGNET_API_KEY + no MAGNET_REDIS_URL → hosted (TODO stub, falls back to local)
      2. MAGNET_REDIS_URL present → redis
      3. MAGNET_LOCAL_MODE=1 → local
      4. Neither → local
    """
    api_key   = os.environ.get("MAGNET_API_KEY", "")
    redis_url = os.environ.get("MAGNET_REDIS_URL", "")
    local     = os.environ.get("MAGNET_LOCAL_MODE", "").lower() in ("1", "true", "yes")

    if api_key.startswith("mg_sk_") and not redis_url:
        # TODO: connect to Magnet hosted Redis endpoint
        logger.info("[storage] API key present but hosted storage not yet available — falling back to local SQLite")
        return "local (hosted coming soon)"
    if redis_url:
        return "redis"
    return "local"
