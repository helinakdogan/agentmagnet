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
import sys

from starlette.applications import Starlette
from starlette.responses import JSONResponse
from starlette.routing import Route
from starlette.types import Receive, Scope, Send

from mcp.server.streamable_http_manager import StreamableHTTPSessionManager

from magnet.mcp_server import app as mcp_app
from magnet.mcp_server import _set_current_identity, _reset_current_identity
from magnet.auth import validate_key
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
    return Starlette(
        routes=[
            Route("/health", health, methods=["GET"]),
            Route("/mcp", endpoint=mcp_endpoint, methods=["GET", "POST", "DELETE"]),
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
