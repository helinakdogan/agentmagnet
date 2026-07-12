"""
team_permissions — the paid moat, Postgres-only, server-side
--------------------------------------------------------------
Owns team COORDINATION: who owns a team, who's a member, is it active, what
storage_mode/redis_url it uses. This is deliberately the ONE place that
decides "is this user_id allowed to touch this team_id" — and it can only
ever produce an answer by reaching OUR Postgres (via
postgres_store.get_pool_if_configured()). Credentials for that Postgres are
never distributed to users, so this module is unusable outside our hosted
deployment even though its source ships in the open `agent-magnet` package.

DATA (shared project/item contents) is a separate concern, handled by
MagnetTeamStore (team_store.py) against whichever backend the team's
storage_mode picks — this module never touches Redis/kv directly except to
encrypt/decrypt a BYO team's connection string.

Every mcp_server.py team tool handler calls check_team_permission() FIRST,
before touching MagnetTeamStore. If it returns None, the handler must return
TEAM_KEY_REQUIRED_MSG and go no further — that includes stdio/local mode,
which never has MAGNET_DATABASE_URL pointed at our Postgres and therefore
can NEVER pass this check, regardless of what MAGNET_REDIS_URL a local user
sets. That's what makes team memory "hosted-server-only" in practice.
"""

from __future__ import annotations

import logging
import os
import random
import string

logger = logging.getLogger(__name__)

TEAM_KEY_REQUIRED_MSG = (
    "Team memory requires an Agent Magnet key. Get one at agentmagnet.app."
)

_TEAM_ID_CHARS = string.ascii_lowercase + string.digits


def _gen_team_id() -> str:
    return "team-" + "".join(random.choices(_TEAM_ID_CHARS, k=6))


# ── Encryption (BYO redis_url at rest) ──────────────────────────────────────

_fernet = None


def _get_fernet():
    """Lazily built, process-wide Fernet instance from MAGNET_ENCRYPTION_KEY.
    Returns None if the env var isn't set — callers must fail closed, never
    store a redis_url in plaintext."""
    global _fernet
    if _fernet is None:
        key = os.environ.get("MAGNET_ENCRYPTION_KEY", "")
        if not key:
            return None
        from cryptography.fernet import Fernet

        _fernet = Fernet(key.encode("utf-8") if isinstance(key, str) else key)
    return _fernet


def encrypt_redis_url(redis_url: str) -> str | None:
    f = _get_fernet()
    if f is None:
        return None
    return f.encrypt(redis_url.encode("utf-8")).decode("utf-8")


def decrypt_redis_url(enc: str) -> str | None:
    f = _get_fernet()
    if f is None:
        return None
    try:
        return f.decrypt(enc.encode("utf-8")).decode("utf-8")
    except Exception as e:
        logger.error(f"[team_permissions] failed to decrypt team redis_url: {type(e).__name__}: {e}")
        return None


# ── Row shaping ──────────────────────────────────────────────────────────────

def _team_row_to_dict(row, decrypt: bool = True) -> dict:
    (team_id, name, owner_user_id, storage_mode, redis_url_enc,
     plan, sync_limit, active, created_at) = row
    redis_url = None
    if decrypt and storage_mode == "byo" and redis_url_enc:
        redis_url = decrypt_redis_url(redis_url_enc)
    return {
        "id": team_id,
        "name": name,
        "owner_user_id": owner_user_id,
        "storage_mode": storage_mode,
        "redis_url": redis_url,
        "has_redis_url": bool(redis_url_enc),
        "plan": plan,
        "sync_limit": sync_limit,
        "active": bool(active),
        "created_at": created_at.isoformat() if created_at else None,
    }


_TEAM_COLUMNS = (
    "id, name, owner_user_id, storage_mode, redis_url_enc, "
    "plan, sync_limit, active, created_at"
)


# ── Coordination ─────────────────────────────────────────────────────────────

def create_team(owner_user_id: str, name: str) -> dict | None:
    """Creates a new team room, owner as first member. Returns the team dict,
    or None if Postgres isn't configured (hosted-only feature)."""
    from magnet.postgres_store import get_pool_if_configured

    pool = get_pool_if_configured()
    if pool is None:
        logger.warning("[team_permissions] create_team called without hosted Postgres")
        return None

    team_id = _gen_team_id()
    with pool.connection() as conn:
        conn.execute(
            "INSERT INTO teams (id, name, owner_user_id) VALUES (%s, %s, %s)",
            (team_id, name, owner_user_id),
        )
        conn.execute(
            "INSERT INTO team_members (team_id, user_id, role) VALUES (%s, %s, 'owner')",
            (team_id, owner_user_id),
        )
    logger.info(f"[team_permissions] created team '{name}' ({team_id}) — owner: {owner_user_id}")
    return get_team(team_id)


def get_team(team_id: str) -> dict | None:
    from magnet.postgres_store import get_pool_if_configured

    pool = get_pool_if_configured()
    if pool is None:
        return None
    with pool.connection() as conn:
        row = conn.execute(
            f"SELECT {_TEAM_COLUMNS} FROM teams WHERE id = %s", (team_id,)
        ).fetchone()
    return _team_row_to_dict(row) if row else None


def join_team(user_id: str, team_id: str) -> dict | None:
    """Adds user_id as a member if the team exists and is active. Returns the
    team dict on success (idempotent — already-a-member is still success),
    None if the team doesn't exist/isn't active, or Postgres is unreachable."""
    from magnet.postgres_store import get_pool_if_configured

    pool = get_pool_if_configured()
    if pool is None:
        return None

    with pool.connection() as conn:
        exists = conn.execute(
            "SELECT 1 FROM teams WHERE id = %s AND active = true", (team_id,)
        ).fetchone()
        if not exists:
            return None
        conn.execute(
            """
            INSERT INTO team_members (team_id, user_id, role)
            VALUES (%s, %s, 'member')
            ON CONFLICT (team_id, user_id) DO NOTHING
            """,
            (team_id, user_id),
        )
    logger.info(f"[team_permissions] {user_id} joined team {team_id}")
    return get_team(team_id)


def add_member(actor_user_id: str, team_id: str, new_user_id: str) -> tuple[bool, str]:
    """Owner-only. Mirrors the same role-check message the open package used
    to enforce client-side — now enforced here instead."""
    from magnet.postgres_store import get_pool_if_configured

    pool = get_pool_if_configured()
    if pool is None:
        return False, TEAM_KEY_REQUIRED_MSG

    with pool.connection() as conn:
        role_row = conn.execute(
            "SELECT role FROM team_members WHERE team_id = %s AND user_id = %s",
            (team_id, actor_user_id),
        ).fetchone()
        if not role_row or role_row[0] != "owner":
            return False, "Only the team owner can add members directly. Share your team_id instead."

        already = conn.execute(
            "SELECT 1 FROM team_members WHERE team_id = %s AND user_id = %s",
            (team_id, new_user_id),
        ).fetchone()
        if already:
            return True, f"'{new_user_id}' is already a member."

        conn.execute(
            "INSERT INTO team_members (team_id, user_id, role) VALUES (%s, %s, 'member')",
            (team_id, new_user_id),
        )
    return True, f"'{new_user_id}' added to the team."


def list_members(team_id: str) -> list[dict]:
    from magnet.postgres_store import get_pool_if_configured

    pool = get_pool_if_configured()
    if pool is None:
        return []
    with pool.connection() as conn:
        rows = conn.execute(
            "SELECT user_id, role, joined_at FROM team_members WHERE team_id = %s ORDER BY joined_at",
            (team_id,),
        ).fetchall()
    return [{"user_id": uid, "role": role, "joined_at": joined_at.isoformat()} for uid, role, joined_at in rows]


def get_teams_for_user(user_id: str) -> list[dict]:
    """Every team this user belongs to, each enriched with their role."""
    from magnet.postgres_store import get_pool_if_configured

    pool = get_pool_if_configured()
    if pool is None:
        return []
    with pool.connection() as conn:
        rows = conn.execute(
            f"""
            SELECT {", ".join("t." + c.strip() for c in _TEAM_COLUMNS.split(","))}, tm.role
            FROM teams t
            JOIN team_members tm ON tm.team_id = t.id
            WHERE tm.user_id = %s
            ORDER BY t.created_at DESC
            """,
            (user_id,),
        ).fetchall()
    result = []
    for row in rows:
        team = _team_row_to_dict(row[:-1], decrypt=False)
        team["role"] = row[-1]
        result.append(team)
    return result


def set_storage_mode(actor_user_id: str, team_id: str, mode: str, redis_url: str | None = None) -> dict | None:
    """Owner-only. Switches a team between 'managed' and 'byo' storage.
    For 'byo', pings the given Redis URL before saving (fail fast on a bad
    URL) and stores it encrypted — never in plaintext, never in a config
    file. Returns the updated team dict, or None on failure (caller decides
    the right HTTP status: not-owner vs bad-url vs no-encryption-key)."""
    if mode not in ("managed", "byo"):
        raise ValueError(f"invalid storage_mode: {mode!r}")

    from magnet.postgres_store import get_pool_if_configured

    pool = get_pool_if_configured()
    if pool is None:
        return None

    with pool.connection() as conn:
        role_row = conn.execute(
            "SELECT role FROM team_members WHERE team_id = %s AND user_id = %s",
            (team_id, actor_user_id),
        ).fetchone()
        if not role_row or role_row[0] != "owner":
            logger.warning(f"[team_permissions] set_storage_mode denied: {actor_user_id} is not owner of {team_id}")
            return None

        if mode == "managed":
            conn.execute(
                "UPDATE teams SET storage_mode = 'managed', redis_url_enc = NULL WHERE id = %s",
                (team_id,),
            )
        else:
            if not redis_url:
                logger.warning("[team_permissions] set_storage_mode(byo) called without a redis_url")
                return None
            try:
                import redis as redis_lib
                redis_lib.Redis.from_url(redis_url, socket_connect_timeout=5).ping()
            except Exception as e:
                logger.warning(f"[team_permissions] BYO redis ping failed for team {team_id}: {type(e).__name__}: {e}")
                return None

            enc = encrypt_redis_url(redis_url)
            if enc is None:
                logger.error("[team_permissions] MAGNET_ENCRYPTION_KEY is not configured — refusing to store a BYO redis_url")
                return None

            conn.execute(
                "UPDATE teams SET storage_mode = 'byo', redis_url_enc = %s WHERE id = %s",
                (enc, team_id),
            )

    return get_team(team_id)


# ── Permission gate — the actual moat ────────────────────────────────────────

def check_team_permission(user_id: str, team_id: str, action: str = "read") -> dict | None:
    """
    THE gate every team-touching MCP tool call must pass before it's allowed
    to read/write anything. Returns the team dict (with `redis_url`
    decrypted if storage_mode == "byo") on success, or None on ANY denial —
    no Postgres reachable (closes stdio/local mode unconditionally), team
    doesn't exist, team isn't active, or user_id isn't a member.

    On success, records exactly one usage_events row (the billable "sync
    request") — this is the single place that happens, so "one validated
    read/write = one usage_event row" is exact, not approximate.
    """
    from magnet.postgres_store import get_pool_if_configured

    pool = get_pool_if_configured()
    if pool is None:
        # No hosted Postgres reachable — this is what makes team memory
        # impossible to run standalone from the open package. Every stdio/
        # local invocation lands here, always, by construction.
        return None

    with pool.connection() as conn:
        team_row = conn.execute(
            f"SELECT {_TEAM_COLUMNS} FROM teams WHERE id = %s AND active = true",
            (team_id,),
        ).fetchone()
        if team_row is None:
            return None

        member_row = conn.execute(
            "SELECT 1 FROM team_members WHERE team_id = %s AND user_id = %s",
            (team_id, user_id),
        ).fetchone()
        if member_row is None:
            return None

    # TODO: BILLING HOOK — this is the ONE place future enforcement plugs
    # in. Once billing is live: call get_team_sync_usage(team_id), compare
    # against team["sync_limit"], and return None (denied) if exceeded.
    # During beta, every membership-valid call is allowed.

    from magnet.usage_counter import record_usage_event
    record_usage_event(user_id, team_id, action)

    return _team_row_to_dict(team_row)


def get_team_sync_usage(team_id: str) -> int:
    """Sync requests (validated team read/write calls) this calendar month —
    same date_trunc('month', now()) bucketing as usage_counter's personal
    usage summary."""
    from magnet.postgres_store import get_pool_if_configured

    pool = get_pool_if_configured()
    if pool is None:
        return 0
    with pool.connection() as conn:
        row = conn.execute(
            """
            SELECT COUNT(*) FROM usage_events
            WHERE team_id = %s AND created_at >= date_trunc('month', now())
            """,
            (team_id,),
        ).fetchone()
    return row[0] if row else 0
