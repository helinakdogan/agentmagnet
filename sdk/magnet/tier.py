"""
Tier resolver — Agent Magnet
-----------------------------
FREE    (default, no key): local SQLite, single-user, BYOK, compression
PRO     (mg_sk_ key OR team usage during beta): team memory, shared Redis

Team memory requires shared Redis. Two teammates on different machines
cannot share local SQLite. This is enforced at the storage layer.

Billing scaffold: gates are in place but NOT enforced during beta.
All team operations are FREE during beta. Flip TODO_ENFORCE_BILLING to
activate real enforcement when billing is live.

TODO: validate MAGNET_API_KEY against remote billing API at
      https://agentmagnet.app/v1/keys/validate
      POST {"key": key} → {"valid": true, "tier": "pro", "plan": "..."}
"""

from __future__ import annotations

import logging
import os

logger = logging.getLogger(__name__)

# Flip this to True to enforce billing (disable beta free access for teams)
_TODO_ENFORCE_BILLING = False

_FREE    = "free"
_PRO     = "pro"
_BETA    = "pro_beta"  # pro features, free during beta

TEAM_NEEDS_REDIS_MSG = (
    "Team memory needs shared storage. "
    "You and your teammates must all set the same MAGNET_REDIS_URL. "
    "This is a Pro feature — see agentmagnet.app."
)

# Features that require Pro
PRO_FEATURES: frozenset[str] = frozenset({
    "team_memory",
    "hosted_storage",
    "compression_stats",
    "promote_to_team",
})

# Legacy alias
PREMIUM_FEATURES = PRO_FEATURES


def get_tier(team_id: str | None = None) -> str:
    """
    Resolve tier.
    - mg_sk_ key present → pro
    - team_id set + Redis URL → pro_beta (free during beta)
    - otherwise → free
    """
    key = os.environ.get("MAGNET_API_KEY", "").strip()
    if key.startswith("mg_sk_") and len(key) > 10:
        return _PRO
    if team_id or os.environ.get("MAGNET_TEAM_ID", ""):
        # Beta: team usage is pro but free
        return _BETA
    return _FREE


def is_pro(team_id: str | None = None) -> bool:
    return get_tier(team_id) in (_PRO, _BETA)


def is_premium() -> bool:
    """Legacy alias."""
    return is_pro()


def check_team_feature(team_id: str | None = None) -> tuple[bool, str]:
    """
    Returns (allowed, plan_label).

    Redis check is separate (MagnetTeamStore._require_redis handles it).
    This gate is for the Pro tier layer.

    During beta: always allowed (free). Log so we can flip later.
    TODO: when _TODO_ENFORCE_BILLING is True, check real billing status.
    """
    if _TODO_ENFORCE_BILLING:
        # TODO: plug real billing check here
        # e.g. verify MAGNET_API_KEY with billing API
        if not is_pro(team_id):
            return False, ""
    logger.info("[tier] team feature access — Pro beta (free). TODO: enforce when billing is live.")
    return True, "Pro (beta, free)"


def check_premium_feature(feature_name: str) -> bool:
    """Legacy check used by old team tools. Returns True — beta allows all."""
    if feature_name not in PRO_FEATURES:
        return True
    allowed, _ = check_team_feature()
    return allowed


def premium_required_response() -> dict:
    """Legacy error dict for old team tools."""
    return {
        "error": "pro_required",
        "message": "Team memory requires a Pro plan. See agentmagnet.app",
    }


def resolve_storage_mode() -> str:
    """Returns a human-readable description of the active storage mode."""
    api_key   = os.environ.get("MAGNET_API_KEY", "")
    redis_url = os.environ.get("MAGNET_REDIS_URL", "")

    if api_key.startswith("mg_sk_") and not redis_url:
        logger.info("[storage] API key present but hosted storage not yet available — falling back to local SQLite")
        return "local (hosted coming soon)"
    if redis_url:
        return "redis"
    return "local"
