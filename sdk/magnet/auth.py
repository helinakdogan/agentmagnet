"""
Auth — hosted HTTP mode
-----------------------
Validates an `mg_sk_...` bearer key against the api_keys table in Postgres.
Never stores or compares raw keys — only their SHA-256 hash.

Used exclusively by http_server.py's auth middleware. stdio mode never
imports this module — local/free tier has no concept of API keys.

Also validates Supabase user-session JWTs (verify_supabase_jwt below), for
the dashboard's own API-key-management/memory-browsing endpoints — a
different, website-only auth path from the mg_sk_... key auth above. The
dashboard never holds an mg_sk_... key for itself; it authenticates as the
logged-in Supabase user and the server resolves that to a user_id.
"""

from __future__ import annotations

import hashlib
import logging
import os

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


_jwks_client = None


def _get_jwks_client():
    """Lazily built, process-wide PyJWKClient — it caches fetched signing
    keys internally, so this doesn't hit Supabase's JWKS endpoint on every
    request. Returns None if SUPABASE_PROJECT_URL isn't configured."""
    global _jwks_client
    if _jwks_client is None:
        project_url = os.environ.get("SUPABASE_PROJECT_URL", "").rstrip("/")
        if not project_url:
            return None
        from jwt import PyJWKClient

        _jwks_client = PyJWKClient(f"{project_url}/auth/v1/.well-known/jwks.json")
    return _jwks_client


def verify_supabase_jwt(token: str) -> str | None:
    """
    Returns the Supabase user id (the JWT's `sub` claim) if `token` is a
    valid, unexpired Supabase session access token — verified against
    Supabase's public JWKS (asymmetric RS256/ES256), not a shared secret.
    Returns None on any failure — missing config, malformed token, expired,
    wrong audience, bad signature — callers should treat every None as a
    generic 401, same convention as validate_key() above.

    What this currently validates (unchanged by the logging added below):
      - alg: token must be signed RS256 or ES256 — `algorithms=["RS256", "ES256"]`.
        A Supabase project still on the legacy HS256 shared-secret signing
        mode will fail here even with a perfectly valid token, since there's
        no HS256 key in the JWKS response to resolve.
      - aud: must be exactly the string "authenticated" — `audience="authenticated"`.
      - iss: NOT validated at all (no `issuer=` kwarg passed to jwt.decode).
      - key source: PyJWKClient fetches from
        {SUPABASE_PROJECT_URL}/auth/v1/.well-known/jwks.json, keyed by the
        token's `kid` header.

    Logging below is intentionally verbose (WARNING level, so it shows up
    in Render's default log level) — this is a diagnostic aid, not the
    normal-operation log volume. Ratchet back to DEBUG once the failure
    mode is confirmed.
    """
    if not token:
        logger.warning("[auth] verify_supabase_jwt: empty/missing token")
        return None

    import jwt

    # ── Diagnostics only — peek at the header and claims WITHOUT verifying
    # the signature. Never used to make an auth decision, only to log what
    # Supabase actually sent us vs. what we're about to check it against. ──
    try:
        header = jwt.get_unverified_header(token)
        logger.warning(f"[auth] jwt header: alg={header.get('alg')!r} kid={header.get('kid')!r}")
    except Exception as e:
        logger.warning(f"[auth] could not decode JWT header ({type(e).__name__}): {e}")

    try:
        unverified_claims = jwt.decode(token, options={"verify_signature": False})
        logger.warning(
            "[auth] jwt claims (unverified): "
            f"aud={unverified_claims.get('aud')!r} iss={unverified_claims.get('iss')!r} "
            f"sub={unverified_claims.get('sub')!r} exp={unverified_claims.get('exp')!r} "
            "— we validate aud=='authenticated' only; iss is not checked"
        )
    except Exception as e:
        logger.warning(f"[auth] could not decode JWT claims ({type(e).__name__}): {e}")

    project_url = os.environ.get("SUPABASE_PROJECT_URL", "").rstrip("/")
    if not project_url:
        logger.warning("[auth] verify_supabase_jwt: SUPABASE_PROJECT_URL is not configured — cannot fetch JWKS")
        return None

    jwks_url = f"{project_url}/auth/v1/.well-known/jwks.json"
    client = _get_jwks_client()
    if client is None:
        logger.warning(f"[auth] verify_supabase_jwt: JWKS client unavailable (url would be {jwks_url})")
        return None

    logger.warning(f"[auth] fetching signing key from JWKS url={jwks_url}")
    try:
        signing_key = client.get_signing_key_from_jwt(token)
        logger.warning(f"[auth] JWKS fetch OK — resolved signing key id={getattr(signing_key, 'key_id', None)!r}")
    except Exception as e:
        # Covers: network/DNS failure hitting jwks_url, HTTP error (404 if
        # the project has no JWKS endpoint at all — i.e. HS256-only
        # project), or no key in the JWKS response matching the token's kid.
        logger.warning(f"[auth] JWKS fetch/key resolution FAILED ({type(e).__name__}): {e} — url={jwks_url}")
        return None

    try:
        payload = jwt.decode(
            token,
            signing_key.key,
            algorithms=["RS256", "ES256"],
            audience="authenticated",
        )
    except jwt.ExpiredSignatureError as e:
        logger.warning(f"[auth] jwt verification FAILED: token expired — {e}")
        return None
    except jwt.InvalidAudienceError as e:
        logger.warning(f"[auth] jwt verification FAILED: audience mismatch (we require aud=='authenticated') — {e}")
        return None
    except jwt.InvalidIssuerError as e:
        logger.warning(f"[auth] jwt verification FAILED: issuer mismatch — {e}")
        return None
    except jwt.InvalidSignatureError as e:
        logger.warning(f"[auth] jwt verification FAILED: signature invalid (wrong key, or token wasn't signed with RS256/ES256) — {e}")
        return None
    except jwt.InvalidAlgorithmError as e:
        logger.warning(f"[auth] jwt verification FAILED: token's alg not in ['RS256', 'ES256'] — {e}")
        return None
    except jwt.PyJWTError as e:
        logger.warning(f"[auth] jwt verification FAILED ({type(e).__name__}): {e}")
        return None
    except Exception as e:
        logger.warning(f"[auth] jwt verification FAILED — unexpected error ({type(e).__name__}): {e}")
        return None

    logger.warning(f"[auth] jwt verification OK — sub={payload.get('sub')!r}")
    return payload.get("sub")
