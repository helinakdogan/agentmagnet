"""
hosted_client — stdio-mode relay to the hosted Agent Magnet server
--------------------------------------------------------------------
Local/stdio processes never hold Postgres credentials (see
team_permissions.py's module docstring — that module is deliberately
Postgres-only and unusable outside our hosted deployment). This module is
the ONLY way a stdio process can perform team operations: over HTTPS,
authenticated by the user's own MAGNET_API_KEY, against the hosted server's
Postgres-backed team registry (magnet.team_permissions, reached via the
new /api/team/check and /api/team/op endpoints in http_server.py).

FAIL CLOSED, always: any network error, timeout, non-2xx-with-unparsable-
body, or malformed response returns None (or (False, msg) where noted) —
never an implicit "allowed". A local process being unable to reach the
hosted server must never be treated as "team memory is available locally
instead" — there is no local fallback for permission, by design (see
mcp_server._is_hosted_mode / the stdio branches of _handle_create_team
etc.). MAGNET_REDIS_URL, if set, only ever affects where TEAM DATA is
stored once permission has already been granted here — it has zero
bearing on any decision made in this file.

Uses stdlib urllib only — no new dependency, since this must work in the
plain `pip install agent-magnet` (no [postgres] extra) case that's the
normal stdio/local install.
"""

from __future__ import annotations

import json
import logging
import os
import urllib.error
import urllib.request

logger = logging.getLogger(__name__)

_DEFAULT_HOSTED_URL = "https://mcp.agentmagnet.app"
_TIMEOUT = 8.0


def _hosted_base_url() -> str:
    return os.environ.get("MAGNET_HOSTED_URL", _DEFAULT_HOSTED_URL).rstrip("/")


def _post(path: str, payload: dict) -> dict | None:
    """POST JSON to the hosted server. Returns the parsed JSON body on ANY
    response we can parse (2xx or not — our endpoints return a JSON body
    even on 401/402/404/503, and callers need that body to pick the right
    deny message), or None if the server is unreachable / times out / sends
    something we can't parse at all."""
    url = f"{_hosted_base_url()}{path}"
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url, data=data, method="POST",
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=_TIMEOUT) as resp:
            return json.loads(resp.read())
    except urllib.error.HTTPError as e:
        try:
            return json.loads(e.read())
        except Exception:
            logger.warning(f"[hosted_client] {path} failed: HTTP {e.code}")
            return None
    except Exception as e:
        logger.warning(f"[hosted_client] {path} unreachable: {type(e).__name__}: {e}")
        return None


def check_team_key(api_key: str) -> dict | None:
    """
    {"allowed": bool, "plan": str|None, "user_id": str} on any response the
    hosted server actually gave us, or None if it couldn't be reached at
    all. Callers MUST treat None as denied, not as "skip the check" — see
    module docstring.
    """
    if not api_key:
        return None
    return _post("/api/team/check", {"key": api_key})


def remote_create_team(api_key: str, name: str) -> dict | None:
    """Team dict on success, None on any denial or failure. The hosted
    endpoint re-validates the key AND re-checks the plan server-side —
    this is not just a permission pre-check, it performs the actual
    creation, since team_permissions.py's registry only exists there."""
    result = _post("/api/team/op", {"key": api_key, "op": "create_team", "name": name})
    if not result or "team" not in result:
        return None
    return result["team"]


def remote_join_team(api_key: str, team_id: str) -> dict | None:
    result = _post("/api/team/op", {"key": api_key, "op": "join_team", "team_id": team_id})
    if not result or "team" not in result:
        return None
    return result["team"]


def remote_add_member(api_key: str, team_id: str, new_user_id: str) -> tuple[bool, str]:
    result = _post(
        "/api/team/op",
        {"key": api_key, "op": "add_member", "team_id": team_id, "new_user_id": new_user_id},
    )
    if not result:
        return False, "Could not reach the hosted server to add this member. Try again shortly."
    return bool(result.get("ok")), result.get("message", "")
