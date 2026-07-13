"""
HTTP (Streamable HTTP) transport — Agent Magnet, hosted mode
-------------------------------------------------------------
Standalone service exposing the SAME MCP tool set as the stdio server
(mcp_server.app / list_tools / call_tool — reused as-is, never forked) over
the MCP Streamable HTTP transport, at /mcp. Plus a /health endpoint.

This is the ONLY entry point that requires MAGNET_DATABASE_URL (Postgres) —
identity, auth, and metering for hosted mode all live there. Local stdio
mode (agent-magnet-mcp) never imports this module and is unaffected.

New console script: agent-magnet-http = "magnet.http_server:main"
"""

from __future__ import annotations

import logging
import os
import secrets
import sys
import time
from datetime import datetime, timezone

from starlette.applications import Starlette
from starlette.middleware import Middleware
from starlette.middleware.cors import CORSMiddleware
from starlette.responses import JSONResponse
from starlette.routing import Route
from starlette.types import Receive, Scope, Send

from mcp.server.streamable_http_manager import StreamableHTTPSessionManager

from magnet.mcp_server import app as mcp_app
from magnet.mcp_server import _set_current_identity, _reset_current_identity
from magnet.mcp_server import _get_memory_store, _get_team_store
from magnet.auth import validate_key, verify_supabase_jwt, _hash_key
from magnet.usage_counter import check_usage_limit

logger = logging.getLogger(__name__)

session_manager = StreamableHTTPSessionManager(
    app=mcp_app,
    json_response=True,   # simpler JSON responses over SSE-for-everything —
                           # easier behind hosted infra (proxies/LBs) and for curl testing
    stateless=True,        # no server-side session affinity — safe behind a
                           # horizontally-scaled Render web service
)


# ── Rate limiter — Part 1 of the hard usage limits ──────────────────────────
#
# In-memory, per-process, fixed 60-second-bucket counter — same pattern as
# proxy/main.py's existing _check_rate_limit, reused deliberately for
# consistency rather than inventing a second scheme. Cheap on purpose: one
# dict lookup + increment per request, no DB round trip on the hot path.
#
# Known limitation, accepted for v1: state is per-process, not shared across
# horizontally-scaled instances, so a key could get up to
# (N_instances * MAGNET_RATE_LIMIT_PER_MIN) requests/min if Render scales
# out. Moving this to Postgres/Redis is a natural upgrade if that matters in
# practice; not worth the extra hot-path round trip until it does.
#
# Applies to every authenticated entry point: the /mcp transport (keyed by
# the API key's hash — "per API key" per spec), and every /api/* route
# (keyed by whatever identity that route authenticates with: API key hash
# for /api/team/*, Supabase user_id for the Supabase-session dashboard
# routes via _require_user).

_RATE_LIMIT_PER_MIN = int(os.environ.get("MAGNET_RATE_LIMIT_PER_MIN", "60"))
_rate_limit_store: dict[tuple[str, int], int] = {}


def _check_rate_limit(identifier: str, max_per_min: int = _RATE_LIMIT_PER_MIN) -> bool:
    """Returns True if this request is allowed, False if identifier has
    exceeded max_per_min requests in the current 60s bucket. Always counts
    the current request (even if it turns out to be the one that exceeds),
    matching proxy/main.py's existing semantics."""
    minute = int(time.time() // 60)
    key = (identifier, minute)
    _rate_limit_store[key] = _rate_limit_store.get(key, 0) + 1
    stale = [k for k in list(_rate_limit_store) if k[1] < minute - 1]
    for k in stale:
        del _rate_limit_store[k]
    return _rate_limit_store[key] <= max_per_min


_RATE_LIMIT_RESPONSE = JSONResponse(
    {"error": "rate_limited", "message": "Rate limit exceeded, slow down."},
    status_code=429,
)


class MagnetAuthASGIMiddleware:
    """
    Raw ASGI middleware (not Starlette's BaseHTTPMiddleware — that buffers
    the whole response body, which breaks Streamable HTTP's streaming
    responses). Wraps session_manager.handle_request directly.

    Identity over HTTP comes ONLY from here — the validated API key. A
    user_id/team_id in the request body is never read or trusted anywhere
    in mcp_server.py's tool dispatch.
    """

    def __init__(self, app) -> None:
        self._app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self._app(scope, receive, send)
            return

        headers = dict(scope.get("headers") or [])
        auth_header = headers.get(b"authorization", b"").decode("latin-1")

        if not auth_header.startswith("Bearer "):
            await JSONResponse(
                {"error": "unauthorized", "message": "Missing 'Authorization: Bearer mg_sk_...' header."},
                status_code=401,
            )(scope, receive, send)
            return

        raw_key = auth_header[len("Bearer "):].strip()
        identity = validate_key(raw_key)
        if identity is None or not identity.get("active"):
            await JSONResponse(
                {"error": "unauthorized", "message": "Invalid, unknown, or inactive API key."},
                status_code=401,
            )(scope, receive, send)
            return

        if not _check_rate_limit(_hash_key(raw_key)):
            await _RATE_LIMIT_RESPONSE(scope, receive, send)
            return

        # Never trust anything from the request body — only the resolved
        # identity from the validated key.
        tokens = _set_current_identity(identity["user_id"], identity["team_id"], identity.get("key_id"))
        try:
            allowed = check_usage_limit(identity["user_id"], identity["team_id"])
            if not allowed:
                await JSONResponse(
                    {"error": "quota_exceeded", "message": "Usage limit reached for this plan."},
                    status_code=402,
                )(scope, receive, send)
                return
            await self._app(scope, receive, send)
        finally:
            _reset_current_identity(tokens)


async def health(request) -> JSONResponse:
    # Deliberately does NOT touch Postgres — must stay up even during a
    # transient DB blip, so a health-check flap doesn't take the whole
    # service down.
    #
    # package_version + git_commit let anyone confirm what's ACTUALLY
    # running in one curl, instead of assuming source and deployment match.
    # Two real incidents already happened from that exact assumption being
    # wrong: the PyPI package (agent-magnet) shipped without the team
    # plan-gate fix for several versions, and a stale mcp.agentmagnet.app
    # deploy was mistaken for a live bug in the plan-gate logic itself when
    # it was actually already fixed in source. git_commit is None unless
    # the platform sets it (Render sets RENDER_GIT_COMMIT automatically for
    # git-connected deploys).
    try:
        from importlib.metadata import version as _pkg_version
        package_version = _pkg_version("agent-magnet")
    except Exception:
        package_version = None
    return JSONResponse({
        "status": "ok",
        "package_version": package_version,
        "git_commit": os.environ.get("RENDER_GIT_COMMIT"),
    })


# ── Dashboard API — Supabase-session-authenticated, distinct from the
# mg_sk_... key auth above. The website calls these directly (not through
# the /mcp transport) to manage keys and browse/delete stored memory. ──────

async def _require_user(request):
    """Returns the Supabase user id (str) on success, or a ready-to-return
    401 JSONResponse on failure — callers do:
        user_id = await _require_user(request)
        if isinstance(user_id, JSONResponse):
            return user_id
    """
    authz = request.headers.get("authorization", "")
    if not authz.startswith("Bearer "):
        return JSONResponse(
            {"error": "unauthorized", "message": "Missing 'Authorization: Bearer <supabase_access_token>' header."},
            status_code=401,
        )
    user_id = verify_supabase_jwt(authz[len("Bearer "):].strip())
    if not user_id:
        return JSONResponse(
            {"error": "unauthorized", "message": "Invalid or expired session token."},
            status_code=401,
        )
    if not _check_rate_limit(user_id):
        return _RATE_LIMIT_RESPONSE
    return user_id


async def list_keys(request) -> JSONResponse:
    user_id = await _require_user(request)
    if isinstance(user_id, JSONResponse):
        return user_id

    from magnet.postgres_store import get_pool_if_configured

    pool = get_pool_if_configured()
    if pool is None:
        return JSONResponse({"error": "unavailable", "message": "Postgres is not configured."}, status_code=503)

    with pool.connection() as conn:
        rows = conn.execute(
            "SELECT id, name, masked_key, created_at, active, plan FROM api_keys WHERE user_id = %s ORDER BY created_at DESC",
            (user_id,),
        ).fetchall()

    from magnet.usage_counter import get_usage_by_key

    usage_by_key = get_usage_by_key(user_id)

    return JSONResponse({
        "keys": [
            {
                "id": str(kid), "name": name, "masked": masked, "created_at": created_at.isoformat(),
                "active": active, "plan": plan, "sync_used": usage_by_key.get(str(kid), 0),
            }
            for kid, name, masked, created_at, active, plan in rows
        ]
    })


async def create_key(request) -> JSONResponse:
    user_id = await _require_user(request)
    if isinstance(user_id, JSONResponse):
        return user_id

    try:
        body = await request.json()
    except Exception:
        body = {}
    name = (body.get("name") or "").strip() or "Untitled key"
    key_type = (body.get("type") or "individual").strip().lower()
    if key_type not in ("individual", "team"):
        return JSONResponse(
            {"error": "bad_request", "message": "type must be 'individual' or 'team'."},
            status_code=400,
        )

    from magnet.postgres_store import get_pool_if_configured

    pool = get_pool_if_configured()
    if pool is None:
        return JSONResponse({"error": "unavailable", "message": "Postgres is not configured."}, status_code=503)

    if key_type == "team":
        # Gate: a "Team" key requires already belonging to a team — avoids
        # a free-plan user minting a team-plan key out of nowhere, without
        # a circular bootstrap problem (teams themselves are still free to
        # create during beta, via POST /api/teams).
        # TODO: BILLING HOOK — once billing is live, this is the one place
        # a real subscription/payment check for the 'team' plan belongs,
        # in addition to (not instead of) the team-membership check below.
        from magnet.team_permissions import get_teams_for_user

        if not get_teams_for_user(user_id):
            return JSONResponse(
                {
                    "error": "team_required",
                    "message": "Create or join a team first, then come back to make a Team key.",
                },
                status_code=400,
            )
        plan = "team"
    else:
        # "Individual" ties the new key to whatever individual (non-team)
        # plan this user already holds — pro if they have any active pro
        # key, otherwise free. Mirrors team_permissions.get_active_plan's
        # "highest active plan wins" logic, scoped to non-team keys only.
        with pool.connection() as conn:
            rows = conn.execute(
                "SELECT plan FROM api_keys WHERE user_id = %s AND active = true AND plan != 'team'",
                (user_id,),
            ).fetchall()
        plan = "pro" if any(r[0] == "pro" for r in rows) else "free"

    raw_key = "mg_sk_live_" + secrets.token_hex(32)
    key_hash = _hash_key(raw_key)
    masked = f"mg_sk_live_{raw_key[11:15]}...{raw_key[-4:]}"

    with pool.connection() as conn:
        row = conn.execute(
            """
            INSERT INTO api_keys (key_hash, user_id, name, masked_key, plan, active)
            VALUES (%s, %s, %s, %s, %s, true)
            RETURNING id, created_at
            """,
            (key_hash, user_id, name, masked, plan),
        ).fetchone()

    key_id, created_at = row
    return JSONResponse({
        "rawKey": raw_key,
        "key": {"id": str(key_id), "name": name, "masked": masked, "plan": plan, "created_at": created_at.isoformat()},
    })


async def revoke_key(request) -> JSONResponse:
    user_id = await _require_user(request)
    if isinstance(user_id, JSONResponse):
        return user_id

    key_id = request.path_params["key_id"]

    from magnet.postgres_store import get_pool_if_configured

    pool = get_pool_if_configured()
    if pool is None:
        return JSONResponse({"error": "unavailable", "message": "Postgres is not configured."}, status_code=503)

    with pool.connection() as conn:
        result = conn.execute(
            "UPDATE api_keys SET active = false WHERE id = %s AND user_id = %s",
            (key_id, user_id),
        )
        found = result.rowcount > 0

    if not found:
        return JSONResponse({"error": "not_found", "message": "Key not found."}, status_code=404)
    return JSONResponse({"ok": True})


# ── Team stdio-relay routes — mg_sk_ key auth (same convention as /mcp) ─────
#
# stdio/local Agent Magnet processes never hold Postgres credentials (see
# team_permissions.py's module docstring), so they cannot call
# team_permissions.py directly the way this HTTP server can. These two
# routes are the ONLY way a stdio process can verify team-plan permission
# and perform team-mutating operations — see magnet.hosted_client, the
# stdio-side counterpart, and mcp_server.py's stdio branches of
# _handle_create_team/_handle_join_team/_handle_add_team_member.
#
# Deliberately mg_sk_-key-authenticated (identical convention to /mcp), NOT
# Supabase-session-authenticated like the dashboard routes above/below —
# the caller here is a local process acting on behalf of its configured
# MAGNET_API_KEY, not a logged-in dashboard user in a browser.

async def team_check(request) -> JSONResponse:
    """General entitlement check: does this key currently carry a paid
    ('team' or 'pro') plan? Used by stdio as the FIRST gate before
    attempting any team operation at all."""
    try:
        body = await request.json()
    except Exception:
        body = {}
    raw_key = (body.get("key") or "").strip()

    identity = validate_key(raw_key)
    if identity is None or not identity.get("active"):
        return JSONResponse(
            {"allowed": False, "plan": None, "message": "Invalid, unknown, or inactive API key."},
            status_code=401,
        )

    if not _check_rate_limit(_hash_key(raw_key)):
        return _RATE_LIMIT_RESPONSE

    from magnet.team_permissions import PAID_PLANS

    plan = identity.get("plan")
    return JSONResponse({
        "allowed": plan in PAID_PLANS,
        "plan": plan,
        "user_id": identity["user_id"],
    })


async def team_op(request) -> JSONResponse:
    """
    Relay for the team-mutating MCP tools (create_team/join_team/
    add_team_member) when invoked from stdio. Re-validates the key AND
    re-checks the plan server-side — never trusts that a prior call to
    team_check() already covered this; each request stands alone.
    """
    try:
        body = await request.json()
    except Exception:
        body = {}
    raw_key = (body.get("key") or "").strip()

    identity = validate_key(raw_key)
    if identity is None or not identity.get("active"):
        return JSONResponse(
            {"error": "unauthorized", "message": "Invalid, unknown, or inactive API key."},
            status_code=401,
        )

    if not _check_rate_limit(_hash_key(raw_key)):
        return _RATE_LIMIT_RESPONSE

    from magnet.team_permissions import (
        add_member, create_team, has_paid_plan, join_team, team_permission_denied_message,
    )

    user_id = identity["user_id"]
    if not has_paid_plan(user_id):
        return JSONResponse(
            {"error": "plan_required", "message": team_permission_denied_message(user_id)},
            status_code=402,
        )

    op = body.get("op")

    if op == "create_team":
        name = (body.get("name") or "").strip() or "My Team"
        team = create_team(user_id, name)
        if team is None:
            return JSONResponse({"error": "unavailable", "message": "Postgres is not configured."}, status_code=503)
        return JSONResponse({"team": team})

    if op == "join_team":
        team_id = body.get("team_id")
        team = join_team(user_id, team_id)
        if team is None:
            return JSONResponse({"error": "not_found", "message": "Team not found or inactive."}, status_code=404)
        return JSONResponse({"team": team})

    if op == "add_member":
        team_id = body.get("team_id")
        new_user_id = body.get("new_user_id")
        ok, msg = add_member(user_id, team_id, new_user_id)
        return JSONResponse({"ok": ok, "message": msg}, status_code=200 if ok else 403)

    return JSONResponse({"error": "bad_request", "message": f"Unknown op {op!r}"}, status_code=400)


def _sync_cap_for_plan(plan: str | None) -> int:
    """Personal (non-team) monthly sync-request cap, mirroring
    mcp_server._memory_cap_for_plan's paid/free split and env-var
    convention. Display-only for now — see get_usage's TODO below for the
    single enforcement seam this and memories_cap will plug into."""
    from magnet.team_permissions import PAID_PLANS
    if plan in PAID_PLANS:
        return int(os.environ.get("MAGNET_PAID_SYNC_CAP", "50000"))
    return int(os.environ.get("MAGNET_FREE_SYNC_CAP", "1000"))


async def get_usage(request) -> JSONResponse:
    user_id = await _require_user(request)
    if isinstance(user_id, JSONResponse):
        return user_id

    from magnet.postgres_store import get_pool_if_configured
    from magnet.usage_counter import get_hosted_usage_summary
    from magnet.mcp_server import _get_backend, _memory_count_key, _memory_cap_for_plan

    plan = "free"
    pool = get_pool_if_configured()
    if pool is not None:
        with pool.connection() as conn:
            row = conn.execute(
                "SELECT plan FROM api_keys WHERE user_id = %s AND active = true ORDER BY created_at DESC LIMIT 1",
                (user_id,),
            ).fetchone()
        if row:
            plan = row[0]

    # "sync request" = every validated tool call recorded this period,
    # personal or team — same event log team_permissions.check_team_permission
    # writes to, summed rather than filtered, since a personal cap doesn't
    # need the read/write split the old writes_this_period/reads_this_period
    # fields did.
    summary = get_hosted_usage_summary(user_id) or {}
    sync_used = sum(summary.values())
    sync_cap = _sync_cap_for_plan(plan)

    # Same maintained counter mcp_server._memory_cap_check reads/enforces on
    # every write — using it here (not a live MemoryStore scan) means
    # memories_used always matches what the cap is actually checked against.
    backend = _get_backend()
    memories_used = backend.hincrby(_memory_count_key(user_id), "total", 0)
    memories_cap = _memory_cap_for_plan(plan)

    period_start = datetime.now(timezone.utc).replace(day=1, hour=0, minute=0, second=0, microsecond=0)

    # TODO: BILLING HOOK — memories_cap/sync_cap are enforced today
    # (mcp_server._memory_cap_check for writes; team sync cap via
    # team_permissions.check_team_permission + MAGNET_ENFORCE_SYNC_CAP for
    # team ops). A personal (non-team) sync_cap has no enforcement point
    # yet — this response is display-only for that field until one exists.
    return JSONResponse({
        "plan": plan,
        "memories_used": memories_used,
        "memories_cap": memories_cap,
        "sync_used": sync_used,
        "sync_cap": sync_cap,
        "rate_limit_per_min": _RATE_LIMIT_PER_MIN,
        "period_start": period_start.isoformat(),
    })


def _item_out(item: dict) -> dict:
    """Shared item -> API shape, for both personal (stored_at) and
    team-shared (shared_at) items."""
    ts = item.get("stored_at", item.get("shared_at"))
    created_at = datetime.fromtimestamp(ts, tz=timezone.utc).isoformat() if ts else None
    return {
        "id": item.get("id"),
        "category": item.get("category"),
        "text": item.get("text"),
        "confidence": item.get("confidence"),
        "status": item.get("status", "active"),
        "created_at": created_at,
        "shared_by": item.get("shared_by"),
    }


async def get_memory(request) -> JSONResponse:
    user_id = await _require_user(request)
    if isinstance(user_id, JSONResponse):
        return user_id

    store = _get_memory_store()
    profiles_out = []
    for profile, _count in store.list_profiles(user_id):
        projects_out = []
        for project in store.list_projects(user_id, profile):
            items = store.load(user_id, profile, project)
            projects_out.append({"project": project, "items": [_item_out(i) for i in items]})
        profiles_out.append({"profile": profile, "projects": projects_out})

    teams_out = []
    team_store = _get_team_store()
    for team_id in team_store.get_teams_for_user(user_id):
        # Broad catch, matching mcp_server._load_team_items_if_shared's
        # convention: team memory needing real Redis is an expected
        # condition, not an error — skip that team rather than 500ing.
        try:
            meta = team_store.get_team_meta(team_id) or {}
            role = next(
                (m["role"] for m in team_store.list_members(team_id) if m["user_id"] == user_id),
                "member",
            )
            projects_out = []
            for shared in team_store.list_shared_projects(team_id):
                items = team_store.load_team_items(team_id, shared["project"])
                projects_out.append({"project": shared["project"], "items": [_item_out(i) for i in items]})
            teams_out.append({
                "team_id": team_id,
                "name": meta.get("name", team_id),
                "role": role,
                "projects": projects_out,
            })
        except Exception as e:
            logger.debug(f"[api/memory] skipping team {team_id}: {e}")
            continue

    return JSONResponse({"profiles": profiles_out, "teams": teams_out})


async def delete_memory_item(request) -> JSONResponse:
    user_id = await _require_user(request)
    if isinstance(user_id, JSONResponse):
        return user_id

    item_id = request.path_params["item_id"]
    store = _get_memory_store()
    for profile, _count in store.list_profiles(user_id):
        for project in store.list_projects(user_id, profile):
            if store.delete_entry(user_id, profile, project, item_id) is not None:
                return JSONResponse({"ok": True})

    return JSONResponse({"error": "not_found", "message": "Memory item not found."}, status_code=404)


# ── Team dashboard routes ───────────────────────────────────────────────────
#
# Thin wrappers over team_permissions.py — the actual coordination/moat logic
# lives there (Postgres-only), not here. Same Supabase-session auth as the
# routes above.

async def create_team_route(request) -> JSONResponse:
    user_id = await _require_user(request)
    if isinstance(user_id, JSONResponse):
        return user_id

    try:
        body = await request.json()
    except Exception:
        body = {}
    name = (body.get("name") or "").strip() or "My Team"

    from magnet.team_permissions import create_team

    team = create_team(user_id, name)
    if team is None:
        return JSONResponse({"error": "unavailable", "message": "Postgres is not configured."}, status_code=503)

    # Key creation lives ONLY in the API Keys tab (type="team") — this route
    # no longer mints one. Create the room, then go create a "Team" key.
    return JSONResponse({"team": team})


async def list_teams_route(request) -> JSONResponse:
    user_id = await _require_user(request)
    if isinstance(user_id, JSONResponse):
        return user_id

    from magnet.team_permissions import get_teams_for_user, get_team_sync_usage

    teams = get_teams_for_user(user_id)
    for team in teams:
        team["sync_requests_this_period"] = get_team_sync_usage(team["id"])
        team.pop("redis_url", None)  # never sent to the client, even if populated
    return JSONResponse({"teams": teams})


async def update_team_storage_route(request) -> JSONResponse:
    user_id = await _require_user(request)
    if isinstance(user_id, JSONResponse):
        return user_id

    team_id = request.path_params["team_id"]
    try:
        body = await request.json()
    except Exception:
        body = {}
    mode = body.get("storage_mode")
    redis_url = body.get("redis_url")

    if mode not in ("managed", "byo"):
        return JSONResponse(
            {"error": "bad_request", "message": "storage_mode must be 'managed' or 'byo'."},
            status_code=400,
        )

    from magnet.team_permissions import set_storage_mode

    team = set_storage_mode(user_id, team_id, mode, redis_url)
    if team is None:
        return JSONResponse(
            {
                "error": "denied",
                "message": (
                    "Not the team owner, team not found, the Redis URL couldn't be "
                    "reached, or server-side encryption isn't configured."
                ),
            },
            status_code=400,
        )
    team.pop("redis_url", None)
    return JSONResponse({"team": team})


# ── Team member management (dashboard) ──────────────────────────────────────
#
# Owner + paid-plan enforced server-side in team_permissions._require_owner,
# reused by all four routes below — lets the owner manage members without
# the *team chat commands.

async def list_team_members_route(request) -> JSONResponse:
    user_id = await _require_user(request)
    if isinstance(user_id, JSONResponse):
        return user_id

    team_id = request.path_params["team_id"]
    from magnet.team_permissions import _require_owner, list_members

    _, err = _require_owner(user_id, team_id)
    if err:
        return JSONResponse({"error": "denied", "message": err}, status_code=403)

    return JSONResponse({"members": list_members(team_id)})


async def add_team_member_route(request) -> JSONResponse:
    user_id = await _require_user(request)
    if isinstance(user_id, JSONResponse):
        return user_id

    team_id = request.path_params["team_id"]
    try:
        body = await request.json()
    except Exception:
        body = {}
    # Accepts either field name — see add_team_member_route's known
    # limitation noted in the dashboard's Team tab: this must be the
    # teammate's Magnet user id today, not a real email lookup (no
    # Supabase admin/email-resolution wired up server-side yet).
    new_user_id = (body.get("user_id") or body.get("email") or "").strip()
    if not new_user_id:
        return JSONResponse(
            {"error": "bad_request", "message": "user_id (or email) is required."},
            status_code=400,
        )

    from magnet.team_permissions import add_member

    ok, msg = add_member(user_id, team_id, new_user_id)
    return JSONResponse({"ok": ok, "message": msg}, status_code=200 if ok else 403)


async def update_team_member_role_route(request) -> JSONResponse:
    user_id = await _require_user(request)
    if isinstance(user_id, JSONResponse):
        return user_id

    team_id = request.path_params["team_id"]
    target_user_id = request.path_params["user_id"]
    try:
        body = await request.json()
    except Exception:
        body = {}
    new_role = (body.get("role") or "").strip()

    from magnet.team_permissions import update_member_role

    ok, msg = update_member_role(user_id, team_id, target_user_id, new_role)
    return JSONResponse({"ok": ok, "message": msg}, status_code=200 if ok else 403)


async def remove_team_member_route(request) -> JSONResponse:
    user_id = await _require_user(request)
    if isinstance(user_id, JSONResponse):
        return user_id

    team_id = request.path_params["team_id"]
    target_user_id = request.path_params["user_id"]

    from magnet.team_permissions import remove_member

    ok, msg = remove_member(user_id, team_id, target_user_id)
    return JSONResponse({"ok": ok, "message": msg}, status_code=200 if ok else 403)


def _lifespan(app: Starlette):
    return session_manager.run()


def build_app() -> Starlette:
    # Route (not Mount) matches "/mcp" exactly, with no trailing-slash
    # rewriting: Streamable HTTP uses one single endpoint path for every
    # operation (POST for messages, GET for the SSE stream, DELETE for
    # session termination), so there's no sub-path to mount for. A plain
    # class-instance endpoint (not a function) is passed straight through
    # by Starlette as a raw ASGI app — see Route.__init__ — so this needs
    # no request/response wrapping. This avoids Mount's default behavior of
    # 307-redirecting bare "/mcp" to "/mcp/", which not every HTTP client
    # follows on POST, and which would matter for a URL pasted directly
    # into a connector.
    mcp_endpoint = MagnetAuthASGIMiddleware(session_manager.handle_request)

    origins = os.environ.get(
        "MAGNET_CORS_ORIGINS",
        "https://agentmagnet.app,https://www.agentmagnet.app,http://localhost:5173,http://localhost:5001",
    ).split(",")

    return Starlette(
        routes=[
            Route("/health", health, methods=["GET"]),
            Route("/mcp", endpoint=mcp_endpoint, methods=["GET", "POST", "DELETE"]),
            Route("/api/keys", list_keys, methods=["GET"]),
            Route("/api/keys", create_key, methods=["POST"]),
            Route("/api/keys/{key_id}", revoke_key, methods=["DELETE"]),
            Route("/api/usage", get_usage, methods=["GET"]),
            Route("/api/memory", get_memory, methods=["GET"]),
            Route("/api/memory/{item_id}", delete_memory_item, methods=["DELETE"]),
            Route("/api/teams", list_teams_route, methods=["GET"]),
            Route("/api/teams", create_team_route, methods=["POST"]),
            Route("/api/teams/{team_id}/storage", update_team_storage_route, methods=["PATCH"]),
            Route("/api/teams/{team_id}/members", list_team_members_route, methods=["GET"]),
            Route("/api/teams/{team_id}/members", add_team_member_route, methods=["POST"]),
            Route("/api/teams/{team_id}/members/{user_id}", update_team_member_role_route, methods=["PATCH"]),
            Route("/api/teams/{team_id}/members/{user_id}", remove_team_member_route, methods=["DELETE"]),
            Route("/api/team/check", team_check, methods=["POST"]),
            Route("/api/team/op", team_op, methods=["POST"]),
        ],
        middleware=[
            Middleware(
                CORSMiddleware,
                allow_origins=origins,
                allow_methods=["GET", "POST", "DELETE", "PATCH"],
                allow_headers=["authorization", "content-type"],
            ),
        ],
        lifespan=_lifespan,
    )


def main() -> None:
    logging.basicConfig(stream=sys.stderr, level=logging.INFO)
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    if not os.environ.get("MAGNET_DATABASE_URL"):
        print(
            "agent-magnet-http requires MAGNET_DATABASE_URL — hosted mode needs "
            "Postgres for api_keys/usage_events. For a single local user, run "
            "`agent-magnet-mcp` (stdio) instead.",
            file=sys.stderr,
        )
        raise SystemExit(1)

    from magnet.postgres_store import get_pool, run_migrations
    pool = get_pool()
    run_migrations(pool)

    import uvicorn
    port = int(os.environ.get("PORT", "8000"))
    uvicorn.run(build_app(), host="0.0.0.0", port=port, log_level="info")


if __name__ == "__main__":
    main()
