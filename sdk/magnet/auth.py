"""
Auth — hosted HTTP mode
-----------------------
Validates an `mg_sk_...` bearer key against the api_keys table in Postgres.
Never stores or compares raw keys — only their SHA-256 hash.

Used exclusively by http_server.py's auth middleware. stdio mode never
imports this module — local/free tier has no concept of API keys.
"""

from __future__ import annotations

import hashlib
import logging

logger = logging.getLogger(__name__)


def _hash_key(raw_key: str) -> str:
    return hashlib.sha256(raw_key.encode("utf-8")).hexdigest()


def validate_key(raw_key: str) -> dict | None:
    """
    Returns {"user_id": str, "team_id": str, "plan": str, "active": bool}
    on success, or None if the key is missing, malformed, unknown, or
    inactive — callers should treat every None the same way (generic 401).

    team_id is "" (never None) when the key has no team, so downstream
    identity plumbing (mcp_server's contextvars) has one canonical
    "no team" value to compare against.

    TODO: BILLING HOOK seam — validate_key() intentionally does NOT check
    plan/quota. That enforcement point is usage_counter.check_usage_limit(),
    called once per authenticated request by http_server.py's auth
    middleware, right after validate_key() succeeds. Keeping the two
    concerns separate means the future billing check only has to change
    one function.
    """
    if not raw_key or not raw_key.startswith("mg_sk_"):
        return None

    from magnet.postgres_store import get_pool_if_configured

    pool = get_pool_if_configured()
    if pool is None:
        logger.warning("[auth] validate_key called but Postgres is not configured")
        return None

    key_hash = _hash_key(raw_key)
    try:
        with pool.connection() as conn:
            row = conn.execute(
                "SELECT user_id, team_id, plan, active FROM api_keys WHERE key_hash = %s",
                (key_hash,),
            ).fetchone()
    except Exception as e:
        logger.error(f"[auth] key lookup failed: {e}")
        return None

    if row is None:
        return None

    user_id, team_id, plan, active = row
    if not active:
        return None

    return {
        "user_id": user_id,
        "team_id": team_id or "",
        "plan": plan,
        "active": bool(active),
    }
