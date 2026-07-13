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
    Returns {"user_id": str, "team_id": str, "plan": str, "active": bool,
    "key_id": str} on success, or None if the key is missing, malformed,
    unknown, or inactive — callers should treat every None the same way
    (generic 401). key_id is used only to tag usage_events rows for
    per-key usage breakdowns, never for identity/authorization.

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
                "SELECT id, user_id, team_id, plan, active FROM api_keys WHERE key_hash = %s",
                (key_hash,),
            ).fetchone()
    except Exception as e:
        logger.error(f"[auth] key lookup failed: {e}")
        return None

    if row is None:
        return None

    key_id, user_id, team_id, plan, active = row
    if not active:
        return None

    return {
        "user_id": user_id,
        "team_id": team_id or "",
        "plan": plan,
        "active": bool(active),
        "key_id": str(key_id),
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


def _verify_hs256(token: str, expected_audience: str = "authenticated") -> dict | None:
    """HS256 path — Supabase's legacy shared-secret signing mode. Confirmed
    (via the diagnostic logging previously added here) to be what this
    project's Supabase instance actually uses.

    expected_audience defaults to "authenticated" (every regular Supabase
    session token) but is overridden by verify_mcp_oauth_token() to require
    the resource-bound aud our Custom Access Token Hook sets on OAuth
    tokens — that's what makes an OAuth token audience-bound to this
    server specifically, not just any Supabase-issued token."""
    import jwt

    secret = os.environ.get("SUPABASE_JWT_SECRET", "")
    if not secret:
        logger.warning(
            "[auth] jwt verification FAILED: token alg is HS256 but SUPABASE_JWT_SECRET is not set "
            "(set it to the Supabase project's legacy JWT secret — Project Settings > API > JWT Secret)"
        )
        return None

    try:
        return jwt.decode(token, secret, algorithms=["HS256"], audience=expected_audience)
    except jwt.ExpiredSignatureError as e:
        logger.warning(f"[auth] jwt verification FAILED (HS256): token expired — {e}")
    except jwt.InvalidAudienceError as e:
        logger.warning(f"[auth] jwt verification FAILED (HS256): audience mismatch (we require aud=={expected_audience!r}) — {e}")
    except jwt.InvalidSignatureError as e:
        logger.warning(f"[auth] jwt verification FAILED (HS256): signature invalid — is SUPABASE_JWT_SECRET correct? — {e}")
    except jwt.PyJWTError as e:
        logger.warning(f"[auth] jwt verification FAILED (HS256, {type(e).__name__}): {e}")
    except Exception as e:
        logger.warning(f"[auth] jwt verification FAILED (HS256, unexpected {type(e).__name__}): {e}")
    return None


def _verify_jwks(token: str, expected_audience: str = "authenticated") -> dict | None:
    """RS256/ES256 path — Supabase's newer asymmetric signing-key mode.
    Kept as a fallback for projects that have migrated off the legacy
    shared secret. See _verify_hs256's docstring re: expected_audience."""
    import jwt

    project_url = os.environ.get("SUPABASE_PROJECT_URL", "").rstrip("/")
    if not project_url:
        logger.warning(
            "[auth] jwt verification FAILED: token alg is RS256/ES256 but SUPABASE_PROJECT_URL is not set "
            "(needed to fetch the JWKS)"
        )
        return None

    jwks_url = f"{project_url}/auth/v1/.well-known/jwks.json"
    client = _get_jwks_client()
    if client is None:
        logger.warning(f"[auth] jwt verification FAILED: JWKS client unavailable (url would be {jwks_url})")
        return None

    logger.debug(f"[auth] fetching signing key from JWKS url={jwks_url}")
    try:
        signing_key = client.get_signing_key_from_jwt(token)
    except Exception as e:
        # Covers: network/DNS failure hitting jwks_url, HTTP error (404 if
        # the project has no JWKS endpoint at all — i.e. HS256-only
        # project), or no key in the JWKS response matching the token's kid.
        logger.warning(f"[auth] JWKS fetch/key resolution FAILED ({type(e).__name__}): {e} — url={jwks_url}")
        return None

    try:
        return jwt.decode(token, signing_key.key, algorithms=["RS256", "ES256"], audience=expected_audience)
    except jwt.ExpiredSignatureError as e:
        logger.warning(f"[auth] jwt verification FAILED (JWKS): token expired — {e}")
    except jwt.InvalidAudienceError as e:
        logger.warning(f"[auth] jwt verification FAILED (JWKS): audience mismatch (we require aud=={expected_audience!r}) — {e}")
    except jwt.InvalidSignatureError as e:
        logger.warning(f"[auth] jwt verification FAILED (JWKS): signature invalid — {e}")
    except jwt.InvalidAlgorithmError as e:
        logger.warning(f"[auth] jwt verification FAILED (JWKS): token's alg not in ['RS256', 'ES256'] — {e}")
    except jwt.PyJWTError as e:
        logger.warning(f"[auth] jwt verification FAILED (JWKS, {type(e).__name__}): {e}")
    except Exception as e:
        logger.warning(f"[auth] jwt verification FAILED (JWKS, unexpected {type(e).__name__}): {e}")
    return None


def _check_issuer_soft(iss: str | None) -> None:
    """Warn-only issuer check — never fails the request. Guards against a
    trailing-slash or scheme mismatch in SUPABASE_PROJECT_URL breaking auth
    outright; iss just isn't load-bearing for the trust decision here (the
    signature + audience already prove the token is Supabase's)."""
    project_url = os.environ.get("SUPABASE_PROJECT_URL", "").rstrip("/")
    if not project_url:
        return
    expected = f"{project_url}/auth/v1"
    if (iss or "").rstrip("/") != expected.rstrip("/"):
        logger.warning(f"[auth] jwt issuer mismatch (non-fatal): expected {expected!r}, got {iss!r}")


def verify_supabase_jwt(token: str) -> str | None:
    """
    Returns the Supabase user id (the JWT's `sub` claim) if `token` is a
    valid, unexpired Supabase session access token. Returns None on any
    failure — missing config, malformed token, expired, wrong audience,
    bad signature — callers should treat every None as a generic 401, same
    convention as validate_key() above.

    Verification path is chosen from the token's own `alg` header:
      - HS256        -> verified against the shared secret SUPABASE_JWT_SECRET
                        (Supabase's legacy signing mode — confirmed to be
                        what this project uses).
      - RS256/ES256  -> verified via Supabase's JWKS
                        ({SUPABASE_PROJECT_URL}/auth/v1/.well-known/jwks.json),
                        for projects on the newer asymmetric signing keys.
    aud must be "authenticated" in both paths. iss is checked (against
    f"{SUPABASE_PROJECT_URL}/auth/v1") as a soft, warning-only match — never
    a hard failure.
    """
    if not token:
        logger.warning("[auth] verify_supabase_jwt: empty/missing token")
        return None

    import jwt

    try:
        header = jwt.get_unverified_header(token)
    except Exception as e:
        logger.warning(f"[auth] could not decode JWT header ({type(e).__name__}): {e}")
        return None

    alg = header.get("alg")
    logger.debug(f"[auth] jwt header: alg={alg!r} kid={header.get('kid')!r}")

    if alg == "HS256":
        payload = _verify_hs256(token)
    elif alg in ("RS256", "ES256"):
        payload = _verify_jwks(token)
    else:
        logger.warning(f"[auth] jwt verification FAILED: unsupported alg {alg!r} (only HS256/RS256/ES256 are handled)")
        return None

    if payload is None:
        return None

    _check_issuer_soft(payload.get("iss"))

    logger.debug(f"[auth] jwt verification OK — sub={payload.get('sub')!r}")
    return payload.get("sub")


def verify_mcp_oauth_token(token: str) -> dict | None:
    """
    Validates an OAuth 2.1 access token issued by Supabase's OAuth Server
    (Authentication > OAuth Server) FOR THIS MCP RESOURCE SPECIFICALLY —
    the /mcp path's second accepted credential type, alongside the
    mg_sk_... static key validated by validate_key() above.

    Same HS256/JWKS verification as verify_supabase_jwt, but with one hard
    difference: aud must equal the full resource identifier
    ({MAGNET_MCP_RESOURCE}/mcp) — the EXACT SAME value advertised as the
    "resource" field in GET /.well-known/oauth-protected-resource — not the
    generic "authenticated" every regular Supabase session token carries.
    (These two MUST stay byte-identical: a spec-compliant client requests a
    token scoped to the resource value it discovered from that metadata
    endpoint, then may itself verify the returned token's aud matches what
    it asked for — if our hook's aud and our metadata's resource ever
    drift apart, the client rejects the token before it even reaches us.)
    A Custom Access Token Hook installed in the Supabase project (see
    supabase_oauth_hook.sql at the repo root) sets that custom aud only on
    tokens carrying a client_id claim — i.e. only on OAuth-issued tokens,
    never on regular login sessions. That's what makes this genuinely
    audience-bound to this server: a session token, or any token not
    issued through this project's OAuth Server, cannot pass.

    Returns {"user_id": str, "client_id": str} on success, None on any
    failure (missing config, bad token, wrong audience, no client_id claim)
    — same fail-closed convention as validate_key()/verify_supabase_jwt().
    """
    resource = os.environ.get("MAGNET_MCP_RESOURCE", "")
    if not resource:
        logger.warning("[auth] verify_mcp_oauth_token: MAGNET_MCP_RESOURCE is not configured")
        return None
    if not token:
        return None

    expected_audience = f"{resource.rstrip('/')}/mcp"

    import jwt

    try:
        header = jwt.get_unverified_header(token)
    except Exception as e:
        logger.warning(f"[auth] could not decode OAuth token header ({type(e).__name__}): {e}")
        return None

    alg = header.get("alg")
    logger.debug(f"[auth] oauth token header: alg={alg!r} kid={header.get('kid')!r}")

    if alg == "HS256":
        payload = _verify_hs256(token, expected_audience=expected_audience)
    elif alg in ("RS256", "ES256"):
        payload = _verify_jwks(token, expected_audience=expected_audience)
    else:
        logger.warning(f"[auth] oauth token verification FAILED: unsupported alg {alg!r}")
        return None

    if payload is None:
        return None

    client_id = payload.get("client_id")
    if not client_id:
        # The access-token hook only ever sets our custom aud on
        # client_id-bearing tokens, so this should be unreachable in
        # practice — checked anyway as defense in depth, not trust-on-faith.
        logger.warning("[auth] oauth token verification FAILED: aud matched but no client_id claim present")
        return None

    logger.debug(f"[auth] oauth token verification OK — sub={payload.get('sub')!r} client_id={client_id!r}")
    return {"user_id": payload.get("sub"), "client_id": client_id}
