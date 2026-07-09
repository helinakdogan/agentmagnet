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

        # Never trust anything from the request body — only the resolved
        # identity from the validated key.
        tokens = _set_current_identity(identity["user_id"], identity["team_id"])
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
    return JSONResponse({"status": "ok"})


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
            "SELECT id, name, masked_key, created_at, active FROM api_keys WHERE user_id = %s ORDER BY created_at DESC",
            (user_id,),
        ).fetchall()

    return JSONResponse({
        "keys": [
            {"id": str(kid), "name": name, "masked": masked, "created_at": created_at.isoformat(), "active": active}
            for kid, name, masked, created_at, active in rows
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

    from magnet.postgres_store import get_pool_if_configured

    pool = get_pool_if_configured()
    if pool is None:
        return JSONResponse({"error": "unavailable", "message": "Postgres is not configured."}, status_code=503)

    raw_key = "mg_sk_live_" + secrets.token_hex(32)
    key_hash = _hash_key(raw_key)
    masked = f"mg_sk_live_{raw_key[11:15]}...{raw_key[-4:]}"

    with pool.connection() as conn:
        row = conn.execute(
            """
            INSERT INTO api_keys (key_hash, user_id, name, masked_key, plan, active)
            VALUES (%s, %s, %s, %s, 'free', true)
            RETURNING id, created_at
            """,
            (key_hash, user_id, name, masked),
        ).fetchone()

    key_id, created_at = row
    return JSONResponse({
        "rawKey": raw_key,
        "key": {"id": str(key_id), "name": name, "masked": masked, "created_at": created_at.isoformat()},
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


# MCP tool names that mutate memory/team state — everything else recorded in
# usage_events counts as a read. Kept here (not in usage_counter.py) since
# it's dashboard-display logic, not metering logic.
_WRITE_EVENTS = {
    "remember", "forget_memory", "mark_done", "checkpoint", "save_now",
    "create_team", "join_team", "add_team_member", "share_project_to_team",
    "share_item_to_team", "create_profile", "create_project",
    "set_active_context", "save_session", "add_team_signal",
}


async def get_usage(request) -> JSONResponse:
    user_id = await _require_user(request)
    if isinstance(user_id, JSONResponse):
        return user_id

    from magnet.postgres_store import get_pool_if_configured
    from magnet.usage_counter import get_hosted_usage_summary

    summary = get_hosted_usage_summary(user_id) or {}
    writes = sum(count for event, count in summary.items() if event in _WRITE_EVENTS)
    reads = sum(count for event, count in summary.items() if event not in _WRITE_EVENTS)

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

    store = _get_memory_store()
    memories_stored = 0
    for profile, _count in store.list_profiles(user_id):
        for project in store.list_projects(user_id, profile):
            memories_stored += len(store.load(user_id, profile, project))

    period_start = datetime.now(timezone.utc).replace(day=1, hour=0, minute=0, second=0, microsecond=0)

    return JSONResponse({
        "plan": plan,
        "memories_stored": memories_stored,
        "writes_this_period": writes,
        "reads_this_period": reads,
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
        ],
        middleware=[
            Middleware(
                CORSMiddleware,
                allow_origins=origins,
                allow_methods=["GET", "POST", "DELETE"],
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
