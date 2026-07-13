"""
MCP Server — Agent Magnet
-------------------------
Memory tools for AI assistants.

Mental model:
  helin (user)
  ├── personal (profile)   →  active profile
  │   ├── general (project)
  │   └── kuika   (project)  →  active project
  └── hobby (profile)

Active context (which profile + project to read/write) is stored in
~/.agent-magnet/active.json and set via list_profiles / list_projects menus.

Primary tools:
  recall               — load active project memory at session start
  remember             — save a decision/preference/etc to the active project
  show_project_memory  — display organized memory for the active project (with item IDs)
  forget_memory        — delete a memory item by id or text query (*forget trigger)
  mark_done            — mark a goal as completed instead of deleting it
  recap                — synthesized natural-language catch-up (*recap trigger)
  show_all_memory      — full dump of active project or bird's-eye across all (*memory trigger)
  list_profiles        — TV menu: pick a profile (*profiles trigger)
  list_projects        — TV menu: pick a project (*projects trigger)
  set_active_context   — set the active profile + project
  get_active_context   — show which profile + project is currently active
  create_profile       — create a new profile
  create_project       — create a new project in a profile

Alias tools (backward compat — same behavior as primary):
  inject_memory        → recall
  add_signal           → remember
  get_project_memory   → show_project_memory
"""

from __future__ import annotations

import asyncio
import contextvars
import json
import logging
import os
import time
from pathlib import Path
from typing import Any

import mcp.types as types
from mcp.server import Server
from mcp.server.stdio import stdio_server

logger = logging.getLogger(__name__)

_DEFAULT_USER_ID = os.environ.get("MAGNET_USER_ID", "user")
_DEFAULT_TEAM_ID = os.environ.get("MAGNET_TEAM_ID", "")
_ACTIVE_FILE = Path.home() / ".agent-magnet" / "active.json"

# ── Per-request identity (HTTP/hosted mode only) ─────────────────────────────
#
# stdio mode is one user per process — _DEFAULT_USER_ID/_DEFAULT_TEAM_ID
# (env vars, fixed at import time) are all it ever needs.
#
# HTTP mode serves many users concurrently in one process. http_server.py's
# auth middleware resolves identity from the validated API key (never from
# the request body) and sets it here, scoped to that request's asyncio task
# via contextvars — so concurrent requests can never see each other's
# identity. _UNSET (not None/"") is the sentinel so a real "no team" ("")
# is distinguishable from "contextvar was never set" (stdio mode).

_UNSET = object()
_user_id_ctx: contextvars.ContextVar[Any] = contextvars.ContextVar("magnet_user_id", default=_UNSET)
_team_id_ctx: contextvars.ContextVar[Any] = contextvars.ContextVar("magnet_team_id", default=_UNSET)


def _current_user_id() -> str:
    v = _user_id_ctx.get()
    return _DEFAULT_USER_ID if v is _UNSET else v


def _current_team_id() -> str:
    v = _team_id_ctx.get()
    return _DEFAULT_TEAM_ID if v is _UNSET else v


def _set_current_identity(user_id: str, team_id: str) -> tuple:
    """Called only by http_server.py's auth middleware, once per request."""
    return _user_id_ctx.set(user_id), _team_id_ctx.set(team_id)


def _reset_current_identity(tokens: tuple) -> None:
    """Called only by http_server.py's auth middleware, in a finally block."""
    _user_id_ctx.reset(tokens[0])
    _team_id_ctx.reset(tokens[1])


def _in_hosted_request() -> bool:
    return _user_id_ctx.get() is not _UNSET


# ── Active context ────────────────────────────────────────────────────────────

def _read_active_context() -> dict:
    if _in_hosted_request():
        raw = _get_backend().get(f"vmm:{_current_user_id()}:__active__")
        try:
            return json.loads(raw) if raw else {}
        except Exception:
            return {}
    try:
        if _ACTIVE_FILE.exists():
            return json.loads(_ACTIVE_FILE.read_text(encoding="utf-8"))
    except Exception:
        pass
    return {}


def _write_active_context(profile: str, project: str) -> None:
    payload = json.dumps({"profile": profile, "project": project}, ensure_ascii=False)
    if _in_hosted_request():
        _get_backend().set(f"vmm:{_current_user_id()}:__active__", payload)
        return
    _ACTIVE_FILE.parent.mkdir(parents=True, exist_ok=True)
    _ACTIVE_FILE.write_text(
        json.dumps({"profile": profile, "project": project}, indent=2),
        encoding="utf-8",
    )


def _resolve_context(profile: str | None = None, project: str | None = None) -> tuple[str, str, str]:
    """Return (user, profile, project) — fills gaps from active context."""
    active = _read_active_context()
    resolved_profile = profile or active.get("profile") or "personal"
    resolved_project = project or active.get("project") or "general"
    return _current_user_id(), resolved_profile, resolved_project


def _ctx_tag(profile: str, project: str) -> str:
    return f"({profile} / {project})"


_SAVE_EVERY = int(os.environ.get("MAGNET_SAVE_EVERY", "8"))
_RHYTHM_FILE = Path.home() / ".agent-magnet" / "rhythm.json"


def _read_rhythm(profile: str, project: str) -> dict:
    key = f"{profile}/{project}"
    if _in_hosted_request():
        # Same fixed-file leak class as active.json — must be per-user in
        # hosted mode, or concurrent users' checkpoint rhythms collide.
        raw = _get_backend().get(f"vmm:{_current_user_id()}:__rhythm__:{key}")
        try:
            return json.loads(raw) if raw else {}
        except Exception:
            return {}
    try:
        if _RHYTHM_FILE.exists():
            return json.loads(_RHYTHM_FILE.read_text(encoding="utf-8")).get(key, {})
    except Exception:
        pass
    return {}


def _write_rhythm(profile: str, project: str, **updates: Any) -> None:
    key = f"{profile}/{project}"
    if _in_hosted_request():
        backend = _get_backend()
        rkey = f"vmm:{_current_user_id()}:__rhythm__:{key}"
        try:
            data = _read_rhythm(profile, project)
            data.update(updates)
            backend.set(rkey, json.dumps(data, ensure_ascii=False))
        except Exception as e:
            logger.debug(f"[rhythm] hosted write failed: {e}")
        return
    try:
        data: dict = {}
        if _RHYTHM_FILE.exists():
            data = json.loads(_RHYTHM_FILE.read_text(encoding="utf-8"))
        data.setdefault(key, {}).update(updates)
        _RHYTHM_FILE.parent.mkdir(parents=True, exist_ok=True)
        _RHYTHM_FILE.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    except Exception as e:
        logger.debug(f"[rhythm] write failed: {e}")


# Keywords that indicate a user message has a real preference worth saving
_PREFERENCE_TRIGGERS = frozenset({
    "prefer", "like", "don't like", "dislike", "hate", "love", "want",
    "always", "never", "use ", "using ", "tercih", "kullan", "sev",
    "istemiyorum", "kullanıyoruz", "yapıyoruz", "kullanmak",
})


async def _extract_from_messages(messages: list[dict], user: str, profile: str, project: str) -> int:
    """Extract project-relevant insights from a message window and save to MemoryStore."""
    from magnet.local_extractor import detect_category
    store = _get_memory_store()
    saved = 0
    for msg in messages:
        if msg.get("role") != "user":
            continue
        text = msg.get("content", "").strip()
        if len(text) < 20:
            continue
        text = text[:400]
        category = detect_category(text)
        if category == "preference":
            text_lower = text.lower()
            if not any(kw in text_lower for kw in _PREFERENCE_TRIGGERS):
                continue
        added = await asyncio.to_thread(store.add_entry, user, profile, project, category, text)
        if added:
            saved += 1
    return saved


# ── Singleton backends ────────────────────────────────────────────────────────

_backend: Any = None
_memory: Any = None
_memory_store: Any = None
_compressor: Any = None
_team_store: Any = None


def _get_backend() -> Any:
    """Shared Redis, Postgres, or SQLite backend — initialized once.

    Resolution order: Redis (MAGNET_REDIS_URL) > Postgres (MAGNET_DATABASE_URL,
    hosted HTTP mode) > SQLite (default, stdio/free tier — unchanged)."""
    global _backend
    if _backend is not None:
        return _backend

    redis_url = os.environ.get("MAGNET_REDIS_URL")
    database_url = os.environ.get("MAGNET_DATABASE_URL")
    client: Any = None
    if redis_url:
        try:
            import redis as redis_lib
            client = redis_lib.from_url(redis_url, decode_responses=True)
            client.ping()
            logger.info("[magnet] Redis connected")
        except Exception as e:
            logger.warning(f"[magnet] Redis unavailable ({e}); falling back to SQLite")

    if client is None and database_url:
        try:
            from magnet.postgres_store import PostgresBackend
            client = PostgresBackend(database_url)
            logger.info("[magnet] Postgres connected (hosted mode)")
        except Exception as e:
            logger.warning(f"[magnet] Postgres unavailable ({e}); falling back to SQLite")

    if client is None:
        from magnet.local_store import SQLiteBackend
        client = SQLiteBackend()
        logger.info("[magnet] Using SQLite (~/.agent-magnet/memory.db)")

    _backend = client
    return _backend


def _get_memory() -> Any:
    """BehavioralMemory — used only by save_session / end_session."""
    global _memory
    if _memory is not None:
        return _memory
    openai_key = os.environ.get("MAGNET_OPENAI_KEY") or os.environ.get("OPENAI_API_KEY")
    qdrant_url = os.environ.get("MAGNET_QDRANT_URL") or os.environ.get("QDRANT_URL")
    qdrant_api_key = os.environ.get("MAGNET_QDRANT_API_KEY") or os.environ.get("QDRANT_API_KEY")
    from magnet.client import BehavioralMemory
    _memory = BehavioralMemory(
        openai_api_key=openai_key,
        redis_client=_get_backend(),
        qdrant_url=qdrant_url,
        qdrant_api_key=qdrant_api_key,
        enable_aggregate=bool(os.environ.get("MAGNET_REDIS_URL") or os.environ.get("MAGNET_DATABASE_URL")),
    )
    return _memory


def _get_memory_store() -> Any:
    """MemoryStore — reads/writes vmm:{user}:{profile}:{project}."""
    global _memory_store
    if _memory_store is None:
        from magnet.project_store import MemoryStore
        _memory_store = MemoryStore(redis_client=_get_backend())
    return _memory_store


def _get_usage_counter() -> Any:
    """UsageCounter is a thin, stateless wrapper — constructed fresh per call
    (not cached) so it's always bound to the CURRENT request's user_id. It
    used to be a process-wide singleton baked with _DEFAULT_USER_ID at first
    call, which was correct for stdio (one user per process) but would leak
    identity across concurrent HTTP requests from different users."""
    from magnet.usage_counter import UsageCounter
    return UsageCounter(redis_client=_get_backend(), user_id=_current_user_id())


def _get_compressor() -> Any:
    global _compressor
    if _compressor is None:
        from magnet.compress import Compressor
        _compressor = Compressor()
    return _compressor


def _get_team_store() -> Any:
    """MagnetTeamStore — category-based team memory, "managed" storage_mode.
    Requires Redis/Postgres-backed shared backend."""
    global _team_store
    if _team_store is None:
        from magnet.team_store import MagnetTeamStore
        _team_store = MagnetTeamStore(redis_client=_get_backend())
    return _team_store


def _team_store_for(team: dict) -> Any:
    """Pick the right MagnetTeamStore backend for a team's storage_mode.
    'managed' reuses the shared process-wide backend singleton (same one
    personal memory uses); 'byo' opens a connection to the team's own
    (already-decrypted) redis_url. `team` must be a dict already returned by
    team_permissions.check_team_permission()/join_team()/create_team() — this
    function never itself decides whether the caller is allowed to be here.
    """
    if team.get("storage_mode") == "byo" and team.get("redis_url"):
        from magnet.team_store import MagnetTeamStore
        import redis as redis_lib
        return MagnetTeamStore(redis_client=redis_lib.from_url(team["redis_url"], decode_responses=True))
    return _get_team_store()


async def _load_team_items_if_shared(project: str, team_id: str) -> list[dict]:
    """Load team items for project if it's shared; returns [] if not shared,
    permission denied, or no hosted Postgres reachable. Runs on every recall
    when a team_id is active, so a denial here must never surface as an
    error — only ever silently fall back to personal-only memory."""
    if not team_id:
        return []
    try:
        from magnet.team_permissions import check_team_permission
        team = await asyncio.to_thread(check_team_permission, _current_user_id(), team_id, "recall")
        if team is None:
            return []
        ts = _team_store_for(team)
        if await asyncio.to_thread(ts.is_project_shared, team_id, project):
            return await asyncio.to_thread(ts.load_team_items, team_id, project)
    except Exception as e:
        logger.debug(f"[team] load_team_items failed: {e}")
    return []


# ── Signal type → storage category ───────────────────────────────────────────

_SIGNAL_TO_CATEGORY = {
    "decision":          "decision",
    "watch_out":         "watch_out",
    "tried_failed":      "tried_failed",
    "convention":        "convention",
    "goal":              "goal",
    "preference":        "preference",
    "preference_like":   "preference",
    "preference_dislike":"preference",
    "correction":        "preference",
    "rejection":         "preference",
    "tone_preference":   "preference",
}

# ── MCP server ────────────────────────────────────────────────────────────────

app = Server("agent-magnet")


@app.list_tools()
async def list_tools() -> list[types.Tool]:
    return [
        # ── PRIMARY: recall ───────────────────────────────────────────────────
        types.Tool(
            name="recall",
            description=(
                "ALWAYS call at the start of every conversation, before your first response. "
                "Loads everything remembered for the ACTIVE project and returns it as adopted memory. "
                "The text returned IS your memory — adopt it silently and apply it immediately. "
                "Also call recall when the user asks 'what do you know?', 'show my memory', "
                "'what's saved?', or any similar question about what you remember."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "profile": {"type": "string", "description": "Profile name (defaults to active profile)"},
                    "project": {"type": "string", "description": "Project name (defaults to active project)"},
                },
                "required": [],
            },
        ),
        # ── PRIMARY: remember ─────────────────────────────────────────────────
        types.Tool(
            name="remember",
            description=(
                "Call AUTOMATICALLY, in the background, the moment the user states:\n"
                "  decision      — 'I decided to X', 'we're going with Y', 'let's use Z'\n"
                "  watch_out     — 'be careful about X', 'don't forget Y', 'this breaks if...'\n"
                "  tried_failed  — 'we tried X and it broke / didn't work'\n"
                "  convention    — 'we always use X', 'components go in /ui/'\n"
                "  goal          — 'we're building X', 'the aim is Y'\n"
                "  preference    — 'I prefer X', 'I like X', 'I don't like Y'\n"
                "Pass the extracted insight as 'text' (one clear sentence). "
                "Never announce that you are calling this. Saves to the ACTIVE project. "
                "Every response confirms with (profile / project)."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "text": {
                        "type": "string",
                        "description": "The insight to save — one clear sentence (preferred over 'messages')",
                    },
                    "messages": {
                        "type": "array",
                        "description": "Conversation messages — only if 'text' not provided",
                        "items": {
                            "type": "object",
                            "properties": {
                                "role": {"type": "string"},
                                "content": {"type": "string"},
                            },
                            "required": ["role", "content"],
                        },
                    },
                    "signal_type": {
                        "type": "string",
                        "enum": [
                            "decision", "watch_out", "tried_failed", "convention", "goal",
                            "preference", "preference_like", "preference_dislike",
                            "correction", "rejection", "tone_preference",
                        ],
                        "description": "Category that best fits what was said",
                    },
                    "profile": {"type": "string", "description": "Defaults to active profile"},
                    "project": {"type": "string", "description": "Defaults to active project"},
                },
                "required": ["signal_type"],
            },
        ),
        # ── PRIMARY: show_project_memory ──────────────────────────────────────
        types.Tool(
            name="show_project_memory",
            description=(
                "Show an organized view of the active project's memory, grouped by: "
                "goals, decisions, watch-outs, tried & failed, conventions, preferences. "
                "Call when the user wants to review what has been saved."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "profile": {"type": "string", "description": "Defaults to active profile"},
                    "project": {"type": "string", "description": "Defaults to active project"},
                },
                "required": [],
            },
        ),
        # ── PRIMARY: forget_memory (*forget) ─────────────────────────────────
        types.Tool(
            name="forget_memory",
            description=(
                "Remove a memory item. Triggered when the user types '*forget <something>' "
                "or says 'delete', 'remove', 'forget' about a specific memory.\n"
                "Two modes:\n"
                "  1. item_id provided → delete immediately (id shown in brackets in show_project_memory).\n"
                "  2. query provided, no item_id → find best match and return a preview; "
                "     show it to the user and ask for confirmation, then call again with item_id to delete.\n"
                "Return a clear confirmation: \"Forgot: '<text>' from <category>.\""
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "item_id": {
                        "type": "string",
                        "description": "6-char id shown in brackets, e.g. 'a1b2c3' — deletes directly",
                    },
                    "query": {
                        "type": "string",
                        "description": "Text to search for — returns best match preview; call again with item_id to confirm",
                    },
                    "profile": {"type": "string", "description": "Defaults to active profile"},
                    "project": {"type": "string", "description": "Defaults to active project"},
                },
                "required": [],
            },
        ),
        # ── PRIMARY: mark_done ────────────────────────────────────────────────
        types.Tool(
            name="mark_done",
            description=(
                "Mark a goal as completed (status → done) instead of deleting it. "
                "Call when the user says a goal is finished/done/completed.\n"
                "Two modes:\n"
                "  1. item_id provided → mark done immediately.\n"
                "  2. query provided → find best matching goal and return preview; "
                "     confirm with user, then call again with item_id.\n"
                "Done goals are hidden from recall/inject but still visible in show_project_memory."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "item_id": {
                        "type": "string",
                        "description": "Goal item id to mark done",
                    },
                    "query": {
                        "type": "string",
                        "description": "Text to find the goal — returns match preview, call again with item_id to confirm",
                    },
                    "profile": {"type": "string", "description": "Defaults to active profile"},
                    "project": {"type": "string", "description": "Defaults to active project"},
                },
                "required": [],
            },
        ),
        # ── TEAM: create_team ─────────────────────────────────────────────────
        types.Tool(
            name="create_team",
            description=(
                "Create a new team and become its owner. "
                "Triggered by '*team new <name>'. "
                "REQUIRES a paid Agent Magnet key (MAGNET_API_KEY, plan team/pro) — "
                "get one at agentmagnet.app. Setting MAGNET_REDIS_URL alone does "
                "nothing; it only decides where shared data lives after a paid key "
                "has already been verified. "
                "Returns a team_id (e.g. 'team-a1b2c3') to share with teammates. "
                "Teammates join with join_team(team_id)."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "team_name": {"type": "string", "description": "Name for the team (e.g. 'backend-crew')"},
                },
                "required": ["team_name"],
            },
        ),
        # ── TEAM: join_team ───────────────────────────────────────────────────
        types.Tool(
            name="join_team",
            description=(
                "Join an existing team by id. "
                "Triggered by '*team join <team_id>'. "
                "REQUIRES a paid Agent Magnet key (MAGNET_API_KEY, plan team/pro) — "
                "get one at agentmagnet.app. The team_id is given by the team owner."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "team_id": {"type": "string", "description": "Team id to join (e.g. 'team-a1b2c3')"},
                },
                "required": ["team_id"],
            },
        ),
        # ── TEAM: add_team_member ─────────────────────────────────────────────
        types.Tool(
            name="add_team_member",
            description=(
                "Owner adds a user directly to the team (owner-only). "
                "The added user must also set MAGNET_TEAM_ID in their MCP config. "
                "Alternative: share your team_id so they can run join_team themselves."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "team_id": {"type": "string", "description": "Team id"},
                    "user_id": {"type": "string", "description": "User id of the person to add"},
                },
                "required": ["team_id", "user_id"],
            },
        ),
        # ── TEAM: list_team_members ───────────────────────────────────────────
        types.Tool(
            name="list_team_members",
            description=(
                "Show all members of a team. "
                "Triggered by '*team members'."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "team_id": {"type": "string", "description": "Defaults to MAGNET_TEAM_ID env var"},
                },
                "required": [],
            },
        ),
        # ── TEAM: list_team_projects ──────────────────────────────────────────
        types.Tool(
            name="list_team_projects",
            description=(
                "List projects that have been shared with the team, so a member can "
                "discover and pick one even if their own local active project is "
                "something else entirely. Triggered by '*team projects'."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "team_id": {"type": "string", "description": "Defaults to MAGNET_TEAM_ID env var"},
                },
                "required": [],
            },
        ),
        # ── TEAM: share_project_to_team ───────────────────────────────────────
        types.Tool(
            name="share_project_to_team",
            description=(
                "Copy the active project's memory into the team's shared space. "
                "After this, all team members who recall or work in this project "
                "will see the shared items labeled [team]. "
                "Triggered by '*team share'."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "team_id": {"type": "string", "description": "Team id (defaults to MAGNET_TEAM_ID)"},
                    "profile": {"type": "string", "description": "Defaults to active profile"},
                    "project": {"type": "string", "description": "Defaults to active project"},
                },
                "required": [],
            },
        ),
        # ── TEAM: share_item_to_team ──────────────────────────────────────────
        types.Tool(
            name="share_item_to_team",
            description=(
                "Share one specific memory item to the team by its item id. "
                "Triggered by '*share <item_id>'. "
                "Use show_project_memory to see item ids."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "item_id": {"type": "string", "description": "The 6-char item id to share"},
                    "team_id": {"type": "string", "description": "Team id (defaults to MAGNET_TEAM_ID)"},
                    "profile": {"type": "string", "description": "Defaults to active profile"},
                    "project": {"type": "string", "description": "Defaults to active project"},
                },
                "required": ["item_id"],
            },
        ),
        # ── TEAM: get_team_memory ─────────────────────────────────────────────
        types.Tool(
            name="get_team_memory",
            description=(
                "Show the team's shared memory for a project (items shared by all members). "
                "Shows who shared each item and which were auto-promoted. "
                "If no project is given and the caller's local active project has no team data, "
                "this auto-selects the team's one shared project, or lists them to choose from "
                "if there are several — it never silently returns empty just because the local "
                "active project points somewhere unrelated."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "team_id": {"type": "string", "description": "Team id (defaults to MAGNET_TEAM_ID)"},
                    "project": {"type": "string", "description": "Defaults to active project"},
                },
                "required": [],
            },
        ),
        # ── PRIMARY: recap (*recap) ───────────────────────────────────────────
        types.Tool(
            name="recap",
            description=(
                "SYNTHESIZED CATCH-UP — call when the user asks 'where were we', "
                "'what were we doing', 'catch me up', 'remind me where we left off', "
                "or types '*recap'. "
                "Pulls all memory for the active project and returns a natural prose summary — "
                "like a helpful teammate catching you up: what we were building, key decisions, "
                "things to watch out for, and what's still open. "
                "NEVER return a raw category list — deliver this as a human narrative."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "profile": {"type": "string", "description": "Defaults to active profile"},
                    "project": {"type": "string", "description": "Defaults to active project"},
                },
                "required": [],
            },
        ),
        # ── PRIMARY: show_all_memory (*memory) ───────────────────────────────
        types.Tool(
            name="show_all_memory",
            description=(
                "FULL MEMORY DUMP — call when the user types '*memory' or says "
                "'what's saved', 'show everything', 'what do you have stored in agent magnet'.\n"
                "Two modes:\n"
                "  DEFAULT (*memory): full dump of the ACTIVE project — every category, "
                "every item with its id, clean readable text (not JSON).\n"
                "  ALL (*memory all): bird's-eye view across ALL profiles and projects — "
                "shows item counts per category so the user sees the whole memory landscape.\n"
                "Pass show_all=true for the all-projects overview."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "show_all": {
                        "type": "boolean",
                        "description": "True for cross-project overview, False (default) for active project full dump",
                    },
                    "profile": {"type": "string", "description": "Defaults to active profile"},
                    "project": {"type": "string", "description": "Defaults to active project"},
                },
                "required": [],
            },
        ),
        # ── PRIMARY: list_profiles (TV menu) ──────────────────────────────────
        types.Tool(
            name="list_profiles",
            description=(
                "TV MENU — profiles. "
                "Trigger: user types '*profiles' OR says 'show profiles', 'switch profile', "
                "'change profile', 'list profiles', 'my profiles'. "
                "Returns a numbered menu. ALWAYS present it verbatim to the user and wait for "
                "their choice. "
                "When they pick a number or name → call set_active_context(profile=<chosen>). "
                "When they say 'new <name>' → call create_profile(name=<name>) instead."
            ),
            inputSchema={"type": "object", "properties": {}, "required": []},
        ),
        # ── PRIMARY: list_projects (TV menu) ──────────────────────────────────
        types.Tool(
            name="list_projects",
            description=(
                "TV MENU — projects. "
                "Trigger: user types '*projects' OR says 'show projects', 'switch project', "
                "'change project', 'list projects', 'my projects'. "
                "Returns a numbered list of projects in the ACTIVE profile. "
                "ALWAYS present it verbatim and wait for their choice. "
                "When they pick → call set_active_context(profile=<active>, project=<chosen>). "
                "When they say 'new <name>' → call create_project(profile=<active>, name=<name>)."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "profile": {"type": "string", "description": "Profile to list (defaults to active profile)"},
                },
                "required": [],
            },
        ),
        # ── PRIMARY: set_active_context ───────────────────────────────────────
        types.Tool(
            name="set_active_context",
            description=(
                "Set the active profile and/or project. Call after the user picks from a menu. "
                "Returns a confirmation string shown to the user."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "profile": {"type": "string", "description": "Profile name to activate"},
                    "project": {"type": "string", "description": "Project name to activate (optional — set only profile if omitted)"},
                },
                "required": ["profile"],
            },
        ),
        # ── PRIMARY: get_active_context ───────────────────────────────────────
        types.Tool(
            name="get_active_context",
            description="Return which profile and project are currently active.",
            inputSchema={"type": "object", "properties": {}, "required": []},
        ),
        # ── PRIMARY: create_profile ───────────────────────────────────────────
        types.Tool(
            name="create_profile",
            description=(
                "Create a new profile and make it active. "
                "Call when the user says 'new <name>' during list_profiles menu, "
                "or explicitly asks to create a profile."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "name": {"type": "string", "description": "Profile name (e.g. 'mavimavi', 'hobby', 'company')"},
                },
                "required": ["name"],
            },
        ),
        # ── PRIMARY: create_project ───────────────────────────────────────────
        types.Tool(
            name="create_project",
            description=(
                "Create a new project under a profile and make it active. "
                "Call when the user says 'new <name>' during list_projects menu, "
                "or explicitly asks to create a project."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "profile": {"type": "string", "description": "Profile name (defaults to active profile)"},
                    "name": {"type": "string", "description": "Project name (e.g. 'kuika', 'side-thing')"},
                },
                "required": ["name"],
            },
        ),
        # ── PRIMARY: checkpoint (rhythm-based auto-save) ─────────────────────
        types.Tool(
            name="checkpoint",
            description=(
                f"RHYTHM-BASED AUTO-SAVE — call silently every ~{_SAVE_EVERY} user turns. "
                "Count user messages in the conversation. When the count reaches "
                f"~{_SAVE_EVERY}, call this tool (pass the recent messages), then reset your count. "
                "Extracts decisions, preferences, and watch-outs from the window and saves to "
                "the active project. "
                "Do NOT announce this to the user — fold it silently into your response. "
                "A one-line note is acceptable if it flows naturally."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "messages": {
                        "type": "array",
                        "description": f"Recent conversation messages (last ~{_SAVE_EVERY}–15 messages)",
                        "items": {
                            "type": "object",
                            "properties": {"role": {"type": "string"}, "content": {"type": "string"}},
                            "required": ["role", "content"],
                        },
                    },
                    "profile": {"type": "string", "description": "Defaults to active profile"},
                    "project": {"type": "string", "description": "Defaults to active project"},
                },
                "required": ["messages"],
            },
        ),
        # ── PRIMARY: save_now (*save manual seal) ────────────────────────────
        types.Tool(
            name="save_now",
            description=(
                "MANUAL CUMULATIVE SAVE — triggered when user types '*save'. "
                "Pass ALL conversation messages accumulated so far (full history, not just recent). "
                "Saves everything to the active project and resets the rhythm counter. "
                "Confirm to the user: 'Saved for (profile / project). N items captured.'"
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "messages": {
                        "type": "array",
                        "description": "ALL conversation messages so far (cumulative)",
                        "items": {
                            "type": "object",
                            "properties": {"role": {"type": "string"}, "content": {"type": "string"}},
                            "required": ["role", "content"],
                        },
                    },
                    "profile": {"type": "string", "description": "Defaults to active profile"},
                    "project": {"type": "string", "description": "Defaults to active project"},
                },
                "required": ["messages"],
            },
        ),
        # ── PRIMARY: get_status (*status) ────────────────────────────────────
        types.Tool(
            name="get_status",
            description=(
                "MEMORY STATUS — triggered when user types '*status' or asks about memory, "
                "storage, or usage. Returns current active context, storage backend, "
                "checkpoint history, usage counts, and plan info."
            ),
            inputSchema={"type": "object", "properties": {}, "required": []},
        ),
        # ── ALIAS: inject_memory → recall ─────────────────────────────────────
        types.Tool(
            name="inject_memory",
            description="Alias for recall. Use recall instead.",
            inputSchema={
                "type": "object",
                "properties": {
                    "user_id": {"type": "string"},
                    "project_id": {"type": "string"},
                    "profile_id": {"type": "string"},
                    "current_message": {"type": "string"},
                },
                "required": [],
            },
        ),
        # ── ALIAS: add_signal → remember ─────────────────────────────────────
        types.Tool(
            name="add_signal",
            description="Alias for remember. Use remember instead.",
            inputSchema={
                "type": "object",
                "properties": {
                    "user_id": {"type": "string"},
                    "project_id": {"type": "string"},
                    "profile_id": {"type": "string"},
                    "messages": {
                        "type": "array",
                        "items": {"type": "object", "properties": {"role": {"type": "string"}, "content": {"type": "string"}}, "required": ["role", "content"]},
                    },
                    "signal_type": {
                        "type": "string",
                        "enum": ["decision", "watch_out", "tried_failed", "convention", "goal",
                                 "correction", "rejection", "preference_like", "preference_dislike", "tone_preference"],
                    },
                },
                "required": ["messages", "signal_type"],
            },
        ),
        # ── ALIAS: get_project_memory → show_project_memory ──────────────────
        types.Tool(
            name="get_project_memory",
            description="Alias for show_project_memory. Use show_project_memory instead.",
            inputSchema={
                "type": "object",
                "properties": {
                    "project_id": {"type": "string"},
                    "user_id": {"type": "string"},
                    "team_id": {"type": "string"},
                },
                "required": [],
            },
        ),
        # ── Session tools ─────────────────────────────────────────────────────
        types.Tool(
            name="save_session",
            description=(
                "Call ONCE at the END of a substantial session to summarize and persist what was learned. "
                "Do NOT use for individual decisions — use remember for those."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "user_id": {"type": "string"},
                    "project_id": {"type": "string"},
                    "profile": {"type": "string"},
                    "project": {"type": "string"},
                    "messages": {
                        "type": "array",
                        "items": {"type": "object", "properties": {"role": {"type": "string"}, "content": {"type": "string"}}, "required": ["role", "content"]},
                    },
                },
                "required": ["messages"],
            },
        ),
        types.Tool(
            name="end_session",
            description="Alias for save_session.",
            inputSchema={
                "type": "object",
                "properties": {
                    "user_id": {"type": "string"},
                    "project_id": {"type": "string"},
                    "messages": {
                        "type": "array",
                        "items": {"type": "object", "properties": {"role": {"type": "string"}, "content": {"type": "string"}}, "required": ["role", "content"]},
                    },
                },
                "required": ["user_id", "messages"],
            },
        ),
        # ── Usage ─────────────────────────────────────────────────────────────
        types.Tool(
            name="usage_stats",
            description="Show memory write and retrieval counts for this user.",
            inputSchema={"type": "object", "properties": {}, "required": []},
        ),
        # ── Team tools ────────────────────────────────────────────────────────
        types.Tool(
            name="get_team_profile",
            description="Get the shared memory profile for a team.",
            inputSchema={
                "type": "object",
                "properties": {
                    "team_id": {"type": "string"},
                    "project_id": {"type": "string"},
                },
                "required": ["team_id"],
            },
        ),
        types.Tool(
            name="add_team_signal",
            description=(
                "Write a signal to TEAM-SCOPED memory. Use only for team-wide conventions or watch-outs. "
                "For personal preferences, use remember instead."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "team_id": {"type": "string"},
                    "project_id": {"type": "string"},
                    "messages": {
                        "type": "array",
                        "items": {"type": "object", "properties": {"role": {"type": "string"}, "content": {"type": "string"}}, "required": ["role", "content"]},
                    },
                    "signal_type": {
                        "type": "string",
                        "enum": ["correction", "rejection", "preference_like", "preference_dislike", "tone_preference", "watch_out"],
                    },
                },
                "required": ["team_id", "messages", "signal_type"],
            },
        ),
        # ── Compression tools ─────────────────────────────────────────────────
        types.Tool(
            name="compress_context",
            description=(
                "Compress a large block of text to reduce token usage. "
                "Original is cached locally for full retrieval. "
                "Returns compressed text + cache_key + token savings."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "text": {"type": "string"},
                    "content_type": {
                        "type": "string",
                        "enum": ["json_array", "log", "long_text", "whitespace"],
                    },
                },
                "required": ["text"],
            },
        ),
        types.Tool(
            name="retrieve_original",
            description="Retrieve the original (uncompressed) text by cache key from compress_context.",
            inputSchema={
                "type": "object",
                "properties": {"cache_key": {"type": "string"}},
                "required": ["cache_key"],
            },
        ),
    ]


# ── Tool dispatch ─────────────────────────────────────────────────────────────

# Team tools record their own usage_events row — but ONLY on a successful
# team_permissions.check_team_permission() (see that module) — never here,
# unconditionally, before the handler even runs. A denied team call must not
# count as a billable "sync request"; that's the whole point of gating it at
# the permission layer instead of the dispatch layer.
_TEAM_TOOL_NAMES = frozenset({
    "create_team", "join_team", "add_team_member", "list_team_members",
    "list_team_projects", "share_project_to_team", "share_item_to_team",
    "get_team_memory", "get_team_profile", "add_team_signal",
})


def _record_usage_event(tool_name: str) -> None:
    """Fires on every non-team tool call, both transports. No-op outside
    hosted/Postgres mode (record_usage_event itself checks). Wrapped so
    metering can never break a tool response."""
    try:
        from magnet.usage_counter import record_usage_event
        record_usage_event(_current_user_id(), _current_team_id(), tool_name)
    except Exception as e:
        logger.debug(f"[usage] _record_usage_event failed: {e}")


@app.call_tool()
async def call_tool(name: str, arguments: dict) -> list[types.TextContent]:
    if name not in _TEAM_TOOL_NAMES:
        _record_usage_event(name)
    try:
        if name == "recall" or name == "inject_memory":
            result = await _handle_recall(
                profile=arguments.get("profile"),
                project=arguments.get("project"),
            )
        elif name == "remember" or name == "add_signal":
            result = await _handle_remember(
                text=arguments.get("text"),
                messages=arguments.get("messages"),
                signal_type=arguments.get("signal_type", "preference"),
                profile=arguments.get("profile"),
                project=arguments.get("project"),
            )
        elif name == "show_project_memory" or name == "get_project_memory":
            result = await _handle_show_project_memory(
                profile=arguments.get("profile"),
                project=arguments.get("project"),
            )
        elif name == "forget_memory":
            result = await _handle_forget_memory(
                item_id=arguments.get("item_id"),
                query=arguments.get("query"),
                profile=arguments.get("profile"),
                project=arguments.get("project"),
            )
        elif name == "mark_done":
            result = await _handle_mark_done(
                item_id=arguments.get("item_id"),
                query=arguments.get("query"),
                profile=arguments.get("profile"),
                project=arguments.get("project"),
            )
        elif name == "create_team":
            result = await _handle_create_team(team_name=arguments["team_name"])
        elif name == "join_team":
            result = await _handle_join_team(team_id=arguments["team_id"])
        elif name == "add_team_member":
            result = await _handle_add_team_member(
                team_id=arguments["team_id"],
                user_id=arguments["user_id"],
            )
        elif name == "list_team_members":
            result = await _handle_list_team_members(team_id=arguments.get("team_id"))
        elif name == "list_team_projects":
            result = await _handle_list_team_projects(team_id=arguments.get("team_id"))
        elif name == "share_project_to_team":
            result = await _handle_share_project_to_team(
                team_id=arguments.get("team_id"),
                profile=arguments.get("profile"),
                project=arguments.get("project"),
            )
        elif name == "share_item_to_team":
            result = await _handle_share_item_to_team(
                item_id=arguments["item_id"],
                team_id=arguments.get("team_id"),
                profile=arguments.get("profile"),
                project=arguments.get("project"),
            )
        elif name == "get_team_memory":
            result = await _handle_get_team_memory(
                team_id=arguments.get("team_id"),
                project=arguments.get("project"),
            )
        elif name == "recap":
            result = await _handle_recap(
                profile=arguments.get("profile"),
                project=arguments.get("project"),
            )
        elif name == "show_all_memory":
            result = await _handle_show_all_memory(
                show_all=bool(arguments.get("show_all", False)),
                profile=arguments.get("profile"),
                project=arguments.get("project"),
            )
        elif name == "list_profiles":
            result = await _handle_list_profiles()
        elif name == "list_projects":
            result = await _handle_list_projects(profile=arguments.get("profile"))
        elif name == "set_active_context":
            result = await _handle_set_active_context(
                profile=arguments["profile"],
                project=arguments.get("project"),
            )
        elif name == "get_active_context":
            result = await _handle_get_active_context()
        elif name == "create_profile":
            result = await _handle_create_profile(name=arguments["name"])
        elif name == "create_project":
            result = await _handle_create_project(
                profile=arguments.get("profile"),
                name=arguments["name"],
            )
        elif name == "checkpoint":
            result = await _handle_checkpoint(
                messages=arguments["messages"],
                profile=arguments.get("profile"),
                project=arguments.get("project"),
            )
        elif name == "save_now":
            result = await _handle_save_now(
                messages=arguments["messages"],
                profile=arguments.get("profile"),
                project=arguments.get("project"),
            )
        elif name == "get_status":
            result = await _handle_get_status()
        elif name in ("save_session", "end_session"):
            result = await _handle_save_session(
                messages=arguments["messages"],
                profile=arguments.get("profile"),
                project=arguments.get("project"),
            )
        elif name == "usage_stats":
            result = await _handle_usage_stats()
        elif name == "get_team_profile":
            result = await _handle_get_team_profile(
                team_id=arguments["team_id"],
                project_id=arguments.get("project_id", "default"),
            )
        elif name == "add_team_signal":
            result = await _handle_add_team_signal(
                team_id=arguments["team_id"],
                project_id=arguments.get("project_id", "default"),
                messages=arguments["messages"],
                signal_type=arguments["signal_type"],
            )
        elif name == "compress_context":
            result = await _handle_compress_context(
                text=arguments["text"],
                content_type=arguments.get("content_type"),
            )
        elif name == "retrieve_original":
            result = await _handle_retrieve_original(arguments["cache_key"])
        else:
            result = {"error": f"Unknown tool: {name}"}
    except Exception as e:
        logger.error(f"Tool {name} error: {e}", exc_info=True)
        result = {"error": str(e)}

    if isinstance(result, str):
        return [types.TextContent(type="text", text=result)]
    return [types.TextContent(type="text", text=json.dumps(result, ensure_ascii=False))]


# ── Primary handlers ──────────────────────────────────────────────────────────

async def _handle_recall(profile: str | None = None, project: str | None = None) -> str:
    user, profile, project = _resolve_context(profile, project)
    store = _get_memory_store()
    usage = _get_usage_counter()

    usage.record_retrieval(project)

    team_id = _current_team_id()
    team_items = await _load_team_items_if_shared(project, team_id)

    if team_items:
        usage.record_team_recall(team_id, project)
        body = await asyncio.to_thread(
            store.format_merged_for_injection, user, profile, project, team_items
        )
    else:
        body = await asyncio.to_thread(store.format_for_injection, user, profile, project)

    ctx = _ctx_tag(profile, project)

    if not body:
        team_note = f" (shared with team {team_id})" if team_id else ""
        return (
            f"Fresh start — no memory yet for {profile} / {project}{team_note}. "
            f"I'll remember things as we work together. {ctx}"
        )

    team_note = f"\n[Team context from {team_id} is included — items marked [team].]" if team_items else ""
    lines = [
        f"You're working on {project} in {profile}. Here's what I know:",
        "",
        body,
        "",
        f"Apply this naturally. The user can override anything.{team_note} {ctx}",
    ]
    return "\n".join(lines)


async def _handle_remember(
    signal_type: str,
    text: str | None = None,
    messages: list[dict] | None = None,
    profile: str | None = None,
    project: str | None = None,
) -> str:
    user, profile, project = _resolve_context(profile, project)
    store = _get_memory_store()
    usage = _get_usage_counter()

    # Resolve text: direct > last user message
    if text:
        extracted = text.strip()[:500]
    elif messages:
        user_msgs = [m for m in messages if m.get("role") == "user"]
        extracted = user_msgs[-1].get("content", "").strip()[:300] if user_msgs else ""
    else:
        extracted = ""

    if not extracted:
        return f"Nothing to save. {_ctx_tag(profile, project)}"

    category = _SIGNAL_TO_CATEGORY.get(signal_type, "preference")
    saved = await asyncio.to_thread(store.add_entry, user, profile, project, category, extracted)
    usage.record_write(project)

    ctx = _ctx_tag(profile, project)
    preview = extracted[:80] + ("…" if len(extracted) > 80 else "")

    auto_promoted = False
    if saved and _current_team_id():
        # Check if this new item agrees with an existing team item → auto-promote
        try:
            ts = _get_team_store()
            if await asyncio.to_thread(ts.is_project_shared, _current_team_id(), project):
                items = await asyncio.to_thread(store.load, user, profile, project)
                new_item = items[-1] if items else None
                if new_item:
                    auto_promoted = await asyncio.to_thread(
                        ts.auto_promote_if_agreed, _current_team_id(), project, new_item, user
                    )
                    if auto_promoted:
                        usage.record_team_write(_current_team_id(), project)
        except Exception as e:
            logger.debug(f"[team] auto-promote check failed: {e}")

    if saved:
        team_note = " — also auto-promoted to team memory ✓" if auto_promoted else ""
        return f"Saved [{category}]: \"{preview}\"{team_note} {ctx}"
    return f"Already known (skipped duplicate): \"{preview[:60]}\" {ctx}"


async def _handle_show_project_memory(profile: str | None = None, project: str | None = None) -> str:
    user, profile, project = _resolve_context(profile, project)
    store = _get_memory_store()
    team_items = await _load_team_items_if_shared(project, _current_team_id())
    if team_items:
        return await asyncio.to_thread(store.format_merged_for_display, user, profile, project, team_items)
    return await asyncio.to_thread(store.format_for_display, user, profile, project)


async def _handle_forget_memory(
    item_id: str | None = None,
    query: str | None = None,
    profile: str | None = None,
    project: str | None = None,
) -> str:
    user, profile, project = _resolve_context(profile, project)
    store = _get_memory_store()
    ctx = _ctx_tag(profile, project)

    if item_id:
        removed = await asyncio.to_thread(store.delete_entry, user, profile, project, item_id)
        if removed:
            return f"Forgot: '{removed['text'][:80]}' from {removed['category']}. {ctx}"
        return f"No item with id '{item_id}' found in {profile} / {project}."

    if query:
        items = await asyncio.to_thread(store.load, user, profile, project)
        if not items:
            return f"No memories in {profile} / {project} to search. {ctx}"
        from magnet.local_embeddings import rank_by_similarity
        matches = await asyncio.to_thread(rank_by_similarity, query, items, "text", 3)
        if not matches:
            return f"No matching memory found for '{query}'. {ctx}"
        best = matches[0]
        best_id = best.get("id", "?")
        lines = [
            f"Best match: [{best_id}] ({best['category']}) \"{best['text'][:100]}\"",
            "",
            f"Call forget_memory(item_id='{best_id}') to delete it, or say 'cancel'.",
        ]
        if len(matches) > 1:
            lines += ["", "Other close matches:"]
            for m in matches[1:]:
                lines.append(f"  [{m.get('id', '?')}] ({m['category']}) \"{m['text'][:80]}\"")
        return "\n".join(lines)

    return f"Provide item_id or query. Use show_project_memory to see item ids. {ctx}"


async def _handle_mark_done(
    item_id: str | None = None,
    query: str | None = None,
    profile: str | None = None,
    project: str | None = None,
) -> str:
    user, profile, project = _resolve_context(profile, project)
    store = _get_memory_store()
    ctx = _ctx_tag(profile, project)

    if item_id:
        updated = await asyncio.to_thread(store.mark_goal_done, user, profile, project, item_id)
        if updated:
            return f"Goal marked done: '{updated['text'][:80]}'. {ctx}"
        return f"No goal with id '{item_id}' found (or it's not a goal). {ctx}"

    if query:
        items = await asyncio.to_thread(store.load, user, profile, project)
        goals = [i for i in items if i.get("category") == "goal"]
        if not goals:
            return f"No goals in {profile} / {project}. {ctx}"
        from magnet.local_embeddings import rank_by_similarity
        matches = await asyncio.to_thread(rank_by_similarity, query, goals, "text", 1)
        if not matches:
            return f"No matching goal found for '{query}'. {ctx}"
        best = matches[0]
        best_id = best.get("id", "?")
        status = best.get("status", "active")
        return (
            f"Best match: [{best_id}] \"{best['text'][:100]}\"  ({status})\n\n"
            f"Call mark_done(item_id='{best_id}') to mark it done."
        )

    return f"Provide item_id or query. Use show_project_memory to see goal ids. {ctx}"


# ── Team handlers ─────────────────────────────────────────────────────────────
#
# Every handler below calls into team_permissions.py FIRST — the Postgres-
# only, server-side module that owns team coordination/permission. None of
# these handlers can succeed without a hosted key reaching our Postgres;
# that's true in both stdio and HTTP transports, since this dispatch code is
# shared between them. See team_permissions.py's module docstring.

def _is_hosted_mode() -> bool:
    """True only when running as agent-magnet-http, which holds direct
    Postgres credentials. False in stdio — which must NEVER treat local
    state (MAGNET_REDIS_URL, MAGNET_TEAM_ID, or anything else local) as
    permission. stdio verifies team-plan access exclusively by calling the
    hosted server over HTTPS — see hosted_client.py — using MAGNET_API_KEY.
    MAGNET_REDIS_URL is completely irrelevant to this decision; it only
    ever affects WHERE team data is stored once permission is granted."""
    return bool(os.environ.get("MAGNET_DATABASE_URL"))


_STDIO_NO_KEY_MSG = "Team memory requires a paid Agent Magnet key. Get one at agentmagnet.app."
_STDIO_UNREACHABLE_MSG = "Could not verify your plan with the hosted server. Try again shortly."


async def _stdio_team_permission_check() -> tuple[str, None] | tuple[None, str]:
    """
    stdio-mode gate: returns (api_key, None) if a hosted, paid-plan key is
    confirmed by the hosted server, or (None, deny_message) otherwise.

    Every branch here fails closed:
      - no MAGNET_API_KEY at all         -> deny, no network call made
      - hosted server unreachable/error  -> deny (never treated as "allow")
      - key invalid/inactive/free plan   -> deny with the hosted server's message
    There is no code path in here that can return "allowed" without a
    successful, plan-confirming round trip to the hosted server.
    """
    api_key = os.environ.get("MAGNET_API_KEY", "").strip()
    if not api_key:
        return None, _STDIO_NO_KEY_MSG

    from magnet.hosted_client import check_team_key

    check = await asyncio.to_thread(check_team_key, api_key)
    if check is None:
        return None, _STDIO_UNREACHABLE_MSG
    if not check.get("allowed"):
        from magnet.team_permissions import PLAN_REQUIRED_MSG
        return None, check.get("message") or PLAN_REQUIRED_MSG
    return api_key, None


async def _handle_create_team(team_name: str) -> str:
    user = _current_user_id()

    if _is_hosted_mode():
        from magnet.team_permissions import (
            create_team, has_paid_plan, team_permission_denied_message, TEAM_KEY_REQUIRED_MSG,
        )
        if not await asyncio.to_thread(has_paid_plan, user):
            return await asyncio.to_thread(team_permission_denied_message, user)
        team = await asyncio.to_thread(create_team, user, team_name)
        if team is None:
            return TEAM_KEY_REQUIRED_MSG
    else:
        api_key, deny_msg = await _stdio_team_permission_check()
        if deny_msg:
            return deny_msg
        from magnet.hosted_client import remote_create_team
        team = await asyncio.to_thread(remote_create_team, api_key, team_name)
        if team is None:
            return "Could not create the team on the hosted server. Try again shortly."

    team_id = team["id"]
    _get_usage_counter().record_team_write(team_id)
    return (
        f"Team '{team_name}' created! Your team id: {team_id}\n\n"
        f"Share this id with your teammates — they run:\n"
        f"  *team join {team_id}\n\n"
        f"Then they add MAGNET_TEAM_ID={team_id} to their MCP config and restart. "
        f"[{team['plan']}]"
    )


async def _handle_join_team(team_id: str) -> str:
    user = _current_user_id()

    if _is_hosted_mode():
        from magnet.team_permissions import (
            join_team, has_paid_plan, team_permission_denied_message, TEAM_KEY_REQUIRED_MSG,
        )
        if not await asyncio.to_thread(has_paid_plan, user):
            return await asyncio.to_thread(team_permission_denied_message, user)
        team = await asyncio.to_thread(join_team, user, team_id)
        if team is None:
            return TEAM_KEY_REQUIRED_MSG
    else:
        api_key, deny_msg = await _stdio_team_permission_check()
        if deny_msg:
            return deny_msg
        from magnet.hosted_client import remote_join_team
        team = await asyncio.to_thread(remote_join_team, api_key, team_id)
        if team is None:
            return "Could not join the team on the hosted server (it may not exist). Try again shortly."

    _get_usage_counter().record_team_write(team_id)
    return (
        f"Joined team '{team['name']}' ({team_id}).\n\n"
        f"Add MAGNET_TEAM_ID={team_id} to your MCP config env and restart Claude. "
        f"Then your recalls and recaps will include shared team memory."
    )


async def _handle_add_team_member(team_id: str, user_id: str) -> str:
    actor = _current_user_id()

    if _is_hosted_mode():
        from magnet.team_permissions import add_member, has_paid_plan, team_permission_denied_message
        if not await asyncio.to_thread(has_paid_plan, actor):
            return await asyncio.to_thread(team_permission_denied_message, actor)
        try:
            ok, msg = await asyncio.to_thread(add_member, actor, team_id, user_id)
        except Exception as e:
            return f"Could not add member: {e}"
    else:
        api_key, deny_msg = await _stdio_team_permission_check()
        if deny_msg:
            return deny_msg
        from magnet.hosted_client import remote_add_member
        ok, msg = await asyncio.to_thread(remote_add_member, api_key, team_id, user_id)

    if not ok:
        return msg
    _get_usage_counter().record_team_write(team_id)
    return f"{msg} They still need to add MAGNET_TEAM_ID={team_id} to their MCP config."


async def _handle_list_team_members(team_id: str | None = None) -> str:
    tid = team_id or _current_team_id()
    if not tid:
        return "No team set. Use *team new <name> to create one, or set MAGNET_TEAM_ID."
    from magnet.team_permissions import check_team_permission, list_members, TEAM_KEY_REQUIRED_MSG, team_permission_denied_message
    team = await asyncio.to_thread(check_team_permission, _current_user_id(), tid, "list_team_members")
    if team is None:
        return await asyncio.to_thread(team_permission_denied_message, _current_user_id())
    try:
        members = await asyncio.to_thread(list_members, tid)
    except Exception as e:
        return f"Could not list members: {e}"
    lines = [f"Team: {team.get('name', tid)} ({tid})", ""]
    for m in members:
        role_tag = " (owner)" if m["role"] == "owner" else ""
        lines.append(f"  · {m['user_id']}{role_tag}")
    lines += ["", f"Total: {len(members)} member{'s' if len(members) != 1 else ''}"]
    return "\n".join(lines)


def _format_shared_projects_menu(team_id: str, shared_projects: list[dict]) -> str:
    lines = [f"Projects shared in team {team_id}:"]
    for i, p in enumerate(shared_projects, 1):
        who = ", ".join(p["shared_by"]) if p["shared_by"] else "?"
        count = p["item_count"]
        lines.append(f"  {i}. {p['project']}  ({count} item{'s' if count != 1 else ''}, shared by {who})")
    lines.append("")
    lines.append("Which one? (number or name)")
    return "\n".join(lines)


async def _handle_list_team_projects(team_id: str | None = None) -> str:
    tid = team_id or _current_team_id()
    if not tid:
        return "No team set. Use *team new <name> to create one, or set MAGNET_TEAM_ID."
    from magnet.team_permissions import check_team_permission, TEAM_KEY_REQUIRED_MSG, team_permission_denied_message
    team = await asyncio.to_thread(check_team_permission, _current_user_id(), tid, "list_team_projects")
    if team is None:
        return await asyncio.to_thread(team_permission_denied_message, _current_user_id())
    ts = _team_store_for(team)
    try:
        shared_projects = await asyncio.to_thread(ts.list_shared_projects, tid)
    except Exception as e:
        return f"Could not load team projects: {e}"
    if not shared_projects:
        return f"No projects shared yet in team {tid}. Use *team share to share the active project."
    return _format_shared_projects_menu(tid, shared_projects)


async def _handle_share_project_to_team(
    team_id: str | None = None,
    profile: str | None = None,
    project: str | None = None,
) -> str:
    tid = team_id or _current_team_id()
    if not tid:
        return "No team set. Use *team new <name> to create one first."
    from magnet.team_permissions import check_team_permission, TEAM_KEY_REQUIRED_MSG, team_permission_denied_message
    team = await asyncio.to_thread(check_team_permission, _current_user_id(), tid, "share_project_to_team")
    if team is None:
        return await asyncio.to_thread(team_permission_denied_message, _current_user_id())
    user, profile, project = _resolve_context(profile, project)
    store = _get_memory_store()
    items = await asyncio.to_thread(store.load, user, profile, project)
    if not items:
        return f"No memory in {profile} / {project} to share yet."
    ts = _team_store_for(team)
    try:
        result = await asyncio.to_thread(ts.share_project, user, project, tid, items)
    except Exception as e:
        return f"Could not share project: {e}"
    _get_usage_counter().record_team_write(tid, project)
    return (
        f"Shared {result['shared']} item{'s' if result['shared'] != 1 else ''} "
        f"from {profile} / {project} → team {tid}.\n\n"
        f"Team members who recall '{project}' will now see these items labeled [team]."
    )


async def _handle_share_item_to_team(
    item_id: str,
    team_id: str | None = None,
    profile: str | None = None,
    project: str | None = None,
) -> str:
    tid = team_id or _current_team_id()
    if not tid:
        return "No team set. Use *team new <name> to create one first."
    from magnet.team_permissions import check_team_permission, TEAM_KEY_REQUIRED_MSG, team_permission_denied_message
    team = await asyncio.to_thread(check_team_permission, _current_user_id(), tid, "share_item_to_team")
    if team is None:
        return await asyncio.to_thread(team_permission_denied_message, _current_user_id())
    user, profile, project = _resolve_context(profile, project)
    store = _get_memory_store()
    items = await asyncio.to_thread(store.load, user, profile, project)
    ts = _team_store_for(team)
    try:
        result = await asyncio.to_thread(ts.share_item, tid, project, item_id, user, items)
    except Exception as e:
        return f"Could not share item: {e}"
    if "error" in result:
        return result["error"]
    if result.get("already_shared"):
        return f"Already shared: '{result['text']}'"
    _get_usage_counter().record_team_write(tid, project)
    return f"Shared [{result['category']}]: '{result['item']}' → team {tid} / {project}."


async def _handle_get_team_memory(
    team_id: str | None = None,
    project: str | None = None,
) -> str:
    tid = team_id or _current_team_id()
    if not tid:
        return "No team set. Use *team new <name> to create one first."
    from magnet.team_permissions import check_team_permission, TEAM_KEY_REQUIRED_MSG, team_permission_denied_message
    team = await asyncio.to_thread(check_team_permission, _current_user_id(), tid, "get_team_memory")
    if team is None:
        return await asyncio.to_thread(team_permission_denied_message, _current_user_id())

    ts = _team_store_for(team)
    explicit_project = project is not None
    _, profile, resolved_project = _resolve_context(None, project)

    if not explicit_project:
        # Local active project may point somewhere the team never shared —
        # don't silently return empty, find the right shared project instead.
        try:
            is_shared = await asyncio.to_thread(ts.is_project_shared, tid, resolved_project)
        except Exception as e:
            return f"Could not check shared projects: {e}"

        if not is_shared:
            try:
                shared_projects = await asyncio.to_thread(ts.list_shared_projects, tid)
            except Exception as e:
                return f"Could not load team projects: {e}"

            if len(shared_projects) == 1:
                resolved_project = shared_projects[0]["project"]
                _write_active_context(profile, resolved_project)
            elif len(shared_projects) > 1:
                return _format_shared_projects_menu(tid, shared_projects)
            else:
                return f"No projects shared yet in team {tid}. Use *team share to share the active project."

    try:
        return await asyncio.to_thread(ts.format_team_display, tid, resolved_project)
    except Exception as e:
        return f"Could not load team memory: {e}"


def _recap_template(project: str, profile: str, by_cat: dict) -> str:
    """Template-based recap when no LLM key is available."""
    active_goals = [t for t, s in by_cat.get("goal", []) if s == "active"]
    done_goals   = [t for t, s in by_cat.get("goal", []) if s == "done"]
    decisions    = [t for t, _ in by_cat.get("decision", [])]
    watch_outs   = [t for t, _ in by_cat.get("watch_out", [])]
    tried        = [t for t, _ in by_cat.get("tried_failed", [])]

    parts: list[str] = []

    if active_goals:
        parts.append(f"Last time on {project}: working toward — {active_goals[-1]}.")
    elif decisions:
        parts.append(f"Last time on {project}: making progress on the build.")
    else:
        parts.append(f"Last time on {project}: getting started.")

    if decisions:
        if len(decisions) == 1:
            parts.append(f"Decided: {decisions[0]}.")
        else:
            parts.append(f"Key decisions: {'; '.join(decisions[-3:])}.")

    if watch_outs:
        parts.append(f"Heads up — {watch_outs[0]}.")
    if tried:
        parts.append(f"Already tried (skip it): {tried[0]}.")

    if active_goals:
        parts.append(f"Still open: {active_goals[0]}. Want to continue there?")
    elif done_goals:
        parts.append("All tracked goals are done. What's next?")

    return " ".join(parts)


async def _recap_with_llm(project: str, profile: str, by_cat: dict, openai_key: str) -> str:
    """LLM-synthesized recap — natural prose, like a teammate catching you up."""
    import litellm

    active_goals = [t for t, s in by_cat.get("goal", []) if s == "active"]
    done_goals   = [t for t, s in by_cat.get("goal", []) if s == "done"]
    decisions    = [t for t, _ in by_cat.get("decision", [])][-6:]
    watch_outs   = [t for t, _ in by_cat.get("watch_out", [])]
    tried        = [t for t, _ in by_cat.get("tried_failed", [])]
    conventions  = [t for t, _ in by_cat.get("convention", [])][-3:]
    preferences  = [t for t, _ in by_cat.get("preference", [])][-3:]

    sections: list[str] = []
    if active_goals:  sections.append("Open goals: " + "; ".join(active_goals))
    if done_goals:    sections.append("Completed goals: " + "; ".join(done_goals))
    if decisions:     sections.append("Decisions made: " + "; ".join(decisions))
    if watch_outs:    sections.append("Watch out for: " + "; ".join(watch_outs))
    if tried:         sections.append("Tried & failed: " + "; ".join(tried))
    if conventions:   sections.append("Conventions: " + "; ".join(conventions))
    if preferences:   sections.append("Preferences: " + "; ".join(preferences))

    memory_text = "\n".join(f"- {s}" for s in sections)
    prompt = (
        f"You are catching up a developer on their '{project}' project. "
        "Write a brief 2-4 sentence recap, like a helpful teammate. "
        "Lead with what they were building, mention the key decisions made, "
        "flag any watch-outs or failed approaches, and end with the open goal or next step. "
        "Sound natural and conversational — NOT like a bullet list or database report.\n\n"
        f"Memory:\n{memory_text}\n\nRecap:"
    )

    try:
        response = await asyncio.to_thread(
            litellm.completion,
            model="openai/gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            api_key=openai_key,
            max_tokens=220,
        )
        text = (response.choices[0].message.content or "").strip()
        if text:
            return text
    except Exception as e:
        logger.warning(f"[recap] LLM failed ({e}), falling back to template")

    return _recap_template(project, profile, by_cat)


async def _handle_recap(profile: str | None = None, project: str | None = None) -> str:
    user, profile, project = _resolve_context(profile, project)
    store = _get_memory_store()
    items = await asyncio.to_thread(store.load, user, profile, project)

    # Merge team items (labeled differently in recap)
    team_items = await _load_team_items_if_shared(project, _current_team_id())
    personal_texts = {i.get("text", "").lower() for i in items}
    for ti in team_items:
        if ti.get("text", "").lower() not in personal_texts:
            items.append({**ti, "_team": True})

    if not items:
        return (
            f"No memory yet for {profile} / {project} — fresh start. "
            "What are we working on?"
        )

    from magnet.project_store import CATEGORIES
    by_cat: dict[str, list[tuple[str, str]]] = {c: [] for c in CATEGORIES}
    for item in items:
        c = item.get("category", "preference")
        if c in by_cat:
            text = item["text"]
            if item.get("_team"):
                text = f"[team] {text}"
            by_cat[c].append((text, item.get("status", "active")))

    openai_key = os.environ.get("MAGNET_OPENAI_KEY") or os.environ.get("OPENAI_API_KEY")
    if openai_key:
        return await _recap_with_llm(project, profile, by_cat, openai_key)
    return _recap_template(project, profile, by_cat)


async def _handle_show_all_memory(
    show_all: bool = False,
    profile: str | None = None,
    project: str | None = None,
) -> str:
    user = _current_user_id()
    store = _get_memory_store()

    if show_all:
        profiles = await asyncio.to_thread(store.list_profiles, user)
        if not profiles:
            return "No memory yet. Say *profiles to create your first profile."

        _cat_labels = {
            "decision": "decision", "goal": "goal", "watch_out": "watch-out",
            "tried_failed": "tried & failed", "convention": "convention",
            "preference": "preference",
        }
        lines = ["Your memory — all projects:\n"]
        for prof_name, _ in profiles:
            projects = await asyncio.to_thread(store.list_projects, user, prof_name)
            if not projects:
                continue
            lines.append(f"  {prof_name}:")
            for proj_name in projects:
                proj_items = await asyncio.to_thread(store.load, user, prof_name, proj_name)
                if not proj_items:
                    lines.append(f"    {proj_name} — (empty)")
                    continue
                counts: dict[str, int] = {}
                for it in proj_items:
                    c = it.get("category", "preference")
                    counts[c] = counts.get(c, 0) + 1
                parts = []
                for cat in ["decision", "goal", "watch_out", "tried_failed", "convention", "preference"]:
                    n = counts.get(cat, 0)
                    if n:
                        lbl = _cat_labels[cat]
                        parts.append(f"{n} {lbl}{'s' if n != 1 else ''}")
                lines.append(f"    {proj_name} — {', '.join(parts) if parts else 'empty'}")
            lines.append("")

        lines.append("Say *memory to see any project in full, or *projects to switch.")
        return "\n".join(lines)

    # Default: full dump of active project
    user, profile, project = _resolve_context(profile, project)
    return await asyncio.to_thread(store.format_for_display, user, profile, project)


async def _handle_list_profiles() -> str:
    user = _current_user_id()
    store = _get_memory_store()
    profiles = await asyncio.to_thread(store.list_profiles, user)

    if not profiles:
        return (
            "No profiles yet.\n"
            "Say 'new <name>' to create your first profile (e.g. 'new personal')."
        )

    lines = ["Your profiles:"]
    for i, (name, count) in enumerate(profiles, 1):
        suffix = f"({count} project{'s' if count != 1 else ''})"
        lines.append(f"  {i}. {name}   {suffix}")
    lines.append("")
    lines.append("Which one? (type a number or name) — or say 'new <name>' to create one.")
    return "\n".join(lines)


async def _handle_list_projects(profile: str | None = None) -> str:
    user, profile, _ = _resolve_context(profile, None)
    store = _get_memory_store()
    projects = await asyncio.to_thread(store.list_projects, user, profile)

    lines = [f"Projects in {profile}:"]
    if projects:
        for i, name in enumerate(projects, 1):
            lines.append(f"  {i}. {name}")
    else:
        lines.append("  (none yet)")
    lines.append("  + new project")
    lines.append("")
    lines.append("Which one? (number or name) — or say 'new <name>' to create one.")
    return "\n".join(lines)


async def _handle_set_active_context(profile: str, project: str | None = None) -> str:
    user = _current_user_id()
    store = _get_memory_store()

    # Ensure profile exists
    await asyncio.to_thread(store.create_profile, user, profile)

    active = _read_active_context()
    current_project = project or active.get("project") or "general"

    if project:
        await asyncio.to_thread(store.create_project, user, profile, project)
        _write_active_context(profile, project)
        return (
            f"Active: {profile} / {project}. "
            "I'll remember everything here now."
        )
    else:
        _write_active_context(profile, current_project)
        return (
            f"Active profile: {profile}. "
            "Say *projects to pick a project."
        )


async def _handle_get_active_context() -> str:
    _, profile, project = _resolve_context()
    return f"Active: {profile} / {project}"


async def _handle_create_profile(name: str) -> str:
    user = _current_user_id()
    store = _get_memory_store()
    created = await asyncio.to_thread(store.create_profile, user, name)
    active = _read_active_context()
    _write_active_context(name, active.get("project") or "general")
    if created:
        return (
            f"Profile '{name}' created and set as active. "
            "Say *projects to add a project."
        )
    return (
        f"Profile '{name}' already exists — switching to it. "
        "Say *projects to pick a project."
    )


async def _handle_create_project(name: str, profile: str | None = None) -> str:
    user, profile, _ = _resolve_context(profile, None)
    store = _get_memory_store()
    created = await asyncio.to_thread(store.create_project, user, profile, name)
    _write_active_context(profile, name)
    if created:
        return f"Project '{name}' created in {profile}. Active: {profile} / {name}."
    return f"Project '{name}' already exists in {profile}. Switched to {profile} / {name}."


# ── Rhythm / checkpoint handlers ──────────────────────────────────────────────

async def _handle_checkpoint(
    messages: list[dict],
    profile: str | None = None,
    project: str | None = None,
) -> str:
    user, profile, project = _resolve_context(profile, project)
    saved = await _extract_from_messages(messages, user, profile, project)
    _get_usage_counter().record_write(project)
    _write_rhythm(
        profile, project,
        last_checkpoint_at=time.time(),
        last_messages_in_window=len([m for m in messages if m.get("role") == "user"]),
        total_checkpoints=(_read_rhythm(profile, project).get("total_checkpoints", 0) + 1),
        last_items_saved=saved,
    )
    ctx = _ctx_tag(profile, project)
    if saved:
        return f"Checkpoint — {saved} item{'s' if saved != 1 else ''} saved. {ctx}"
    return f"Checkpoint — nothing new to save. {ctx}"


async def _handle_save_now(
    messages: list[dict],
    profile: str | None = None,
    project: str | None = None,
) -> str:
    user, profile, project = _resolve_context(profile, project)
    saved = await _extract_from_messages(messages, user, profile, project)
    _get_usage_counter().record_write(project)
    _write_rhythm(
        profile, project,
        last_checkpoint_at=time.time(),
        last_messages_in_window=len([m for m in messages if m.get("role") == "user"]),
        total_checkpoints=(_read_rhythm(profile, project).get("total_checkpoints", 0) + 1),
        last_items_saved=saved,
    )
    ctx = _ctx_tag(profile, project)
    store = _get_memory_store()
    total = len(await asyncio.to_thread(store.load, user, profile, project))
    return (
        f"Saved everything up to here for {ctx}. "
        f"{saved} new item{'s' if saved != 1 else ''} captured. "
        f"{total} total memories in this project."
    )


async def _handle_get_status() -> str:
    user, profile, project = _resolve_context()
    store = _get_memory_store()
    usage = _get_usage_counter()
    backend = _get_backend()

    # Backend type
    backend_type = type(backend).__name__
    if backend_type == "SQLiteBackend":
        db_path = Path.home() / ".agent-magnet" / "memory.db"
        storage_line = f"local (this machine) — {db_path}"
    elif backend_type == "PostgresBackend":
        storage_line = "hosted (Postgres)"
    else:
        storage_line = "cloud (Redis)"

    # Plan
    if backend_type == "PostgresBackend":
        plan_line = "Hosted Magnet — metered"
    elif os.environ.get("MAGNET_API_KEY"):
        plan_line = "Hosted Magnet — metered"
    elif os.environ.get("MAGNET_REDIS_URL"):
        plan_line = "Self-hosted Redis — unlimited"
    else:
        plan_line = "Free — local storage, unlimited"

    # Memory counts
    items = await asyncio.to_thread(store.load, user, profile, project)
    total_memories = len(items)

    # Usage stats
    stats = usage.get_stats()
    total_writes = stats.get("writes:total", 0)
    total_retrievals = stats.get("retrievals:total", 0)

    # Rhythm info
    rhythm = _read_rhythm(profile, project)
    last_cp = rhythm.get("last_checkpoint_at")
    total_cps = rhythm.get("total_checkpoints", 0)
    last_items = rhythm.get("last_items_saved", 0)

    if last_cp:
        mins_ago = int((time.time() - last_cp) / 60)
        if mins_ago < 1:
            cp_line = f"just now ({last_items} items saved)"
        elif mins_ago < 60:
            cp_line = f"{mins_ago} min ago ({last_items} items saved)"
        else:
            cp_line = f"{mins_ago // 60}h ago ({last_items} items saved)"
    else:
        cp_line = "never (no checkpoint yet this session)"

    # Team info
    team_line = "none (solo mode)"
    if _current_team_id():
        try:
            from magnet.team_permissions import check_team_permission, list_members, team_permission_denied_message
            team = await asyncio.to_thread(check_team_permission, user, _current_team_id(), "get_status")
            if team is None:
                deny_msg = await asyncio.to_thread(team_permission_denied_message, user)
                team_line = f"{_current_team_id()} ({deny_msg})"
            else:
                ts = _team_store_for(team)
                members = await asyncio.to_thread(list_members, _current_team_id())
                shared = await asyncio.to_thread(ts.is_project_shared, _current_team_id(), project)
                shared_tag = " · project shared ✓" if shared else " · project not yet shared"
                team_line = f"{team['name']} ({len(members)} member{'s' if len(members) != 1 else ''}) · {team['plan']}{shared_tag}"
        except Exception as e:
            team_line = f"{_current_team_id()} (error checking team status: {e})"

    lines = [
        f"Active:          {profile} / {project}",
        f"Team:            {team_line}",
        f"Storage:         {storage_line}",
        f"Save rhythm:     every ~{_SAVE_EVERY} user messages",
        f"Last checkpoint: {cp_line}",
        f"Total checkpoints: {total_cps}",
        f"Memories in project: {total_memories}",
        f"All-time writes: {total_writes} | recalls: {total_retrievals}",
        f"Plan:            {plan_line}",
    ]

    if backend_type == "PostgresBackend":
        from magnet.usage_counter import get_hosted_usage_summary
        summary = get_hosted_usage_summary(user, _current_team_id())
        if summary:
            period_line = ", ".join(f"{k}: {v}" for k, v in sorted(summary.items()))
            lines.append(f"This period:     {period_line}")

    return "\n".join(lines)


# ── Session handler ───────────────────────────────────────────────────────────

async def _handle_save_session(
    messages: list[dict],
    profile: str | None = None,
    project: str | None = None,
) -> dict:
    user, profile, project = _resolve_context(profile, project)
    memory = _get_memory()
    store = _get_memory_store()
    usage = _get_usage_counter()

    # Use active profile/project as the "project_id" for legacy session_end
    result = await asyncio.to_thread(
        memory.session_end, user, project, messages, 20, profile
    )

    # Promote concrete decisions/watch-outs into the new MemoryStore
    summary = result.get("summary", "")
    if summary:
        await _promote_summary_to_memory(summary, user, profile, project, store)

    usage.record_write(project)
    ctx = _ctx_tag(profile, project)
    return {**result, "active_context": ctx}


async def _promote_summary_to_memory(
    summary: str, user: str, profile: str, project: str, store: Any
) -> None:
    from magnet.local_extractor import detect_category

    project_categories = frozenset({"decision", "watch_out", "tried_failed", "convention", "goal"})
    for line in summary.splitlines():
        text = line.strip().lstrip("-•*").strip()
        if len(text) < 10:
            continue
        cat = detect_category(text)
        if cat in project_categories:
            try:
                await asyncio.to_thread(store.add_entry, user, profile, project, cat, text)
            except Exception as e:
                logger.debug(f"_promote_summary: {e}")


# ── Usage handler ─────────────────────────────────────────────────────────────

async def _handle_usage_stats() -> dict:
    _, profile, project = _resolve_context()
    stats = _get_usage_counter().get_stats()
    return {
        "user": _current_user_id(),
        "active_context": _ctx_tag(profile, project),
        "stats": stats,
        "note": "Metering active. Local mode is unlimited.",
    }


# ── Team handlers ─────────────────────────────────────────────────────────────

async def _handle_get_team_profile(team_id: str, project_id: str) -> dict:
    from magnet.team_permissions import check_team_permission, TEAM_KEY_REQUIRED_MSG, team_permission_denied_message
    team = await asyncio.to_thread(check_team_permission, _current_user_id(), team_id, "get_team_profile")
    if team is None:
        return {"error": await asyncio.to_thread(team_permission_denied_message, _current_user_id())}
    # NOTE: this legacy preference-profile subsystem (TeamStore/memory._team_store,
    # distinct from MagnetTeamStore above) always uses the shared managed
    # backend — storage_mode/BYO selection is not wired into it. Permission
    # is enforced the same way as everywhere else; data location is not.
    from magnet.team_store import TeamMemoryRequiresRedis
    memory = _get_memory()
    try:
        profile = await asyncio.to_thread(
            memory._team_store.load_team_profile, team_id, project_id
        )
    except TeamMemoryRequiresRedis as e:
        return {"error": str(e)}
    if not profile:
        return {"team_id": team_id, "project_id": project_id, "prefers": [], "watch_out": []}
    buckets: dict[str, list] = {"prefers": [], "dislikes": [], "expects": [], "watch_out": []}
    for pref in profile.get("preferences", []):
        if not isinstance(pref, dict):
            continue
        rel = pref.get("relation", "")
        if rel in buckets:
            buckets[rel].append({
                "text": pref.get("natural_text", pref.get("subject", "")),
                "confidence": round(pref.get("confidence", 0.5), 3),
            })
    return {"team_id": team_id, "project_id": project_id, **buckets}


async def _handle_add_team_signal(
    team_id: str, project_id: str, messages: list[dict], signal_type: str
) -> dict:
    from magnet.team_permissions import check_team_permission, TEAM_KEY_REQUIRED_MSG, team_permission_denied_message
    team = await asyncio.to_thread(check_team_permission, _current_user_id(), team_id, "add_team_signal")
    if team is None:
        return {"error": await asyncio.to_thread(team_permission_denied_message, _current_user_id())}
    from magnet.team_store import TeamMemoryRequiresRedis
    memory = _get_memory()
    try:
        memory._team_store._require_redis()
    except TeamMemoryRequiresRedis as e:
        return {"error": str(e)}

    user_msgs = [m for m in messages if m.get("role") == "user"]
    if not user_msgs:
        return {"status": "ok", "signal_type": signal_type, "saved": False}

    last_msg = user_msgs[-1].get("content", "")
    tenant_id = f"{project_id}:{team_id}"

    instant_types = {"preference_like", "preference_dislike", "tone_preference", "watch_out"}
    if signal_type in instant_types:
        existing = await asyncio.to_thread(memory._team_store.load_team_profile, team_id, project_id) or {}
        updated = await asyncio.to_thread(
            memory._reflector.instant_learn, tenant_id, signal_type, last_msg[:200], 0.75, existing
        )
        await asyncio.to_thread(memory._team_store.save_team_profile, team_id, project_id, updated)
        return {"status": "ok", "signal_type": signal_type, "team_id": team_id, "instant_learned": True}

    signal = {"type": signal_type, "message": last_msg[:200], "confidence": 0.6}
    count = await asyncio.to_thread(memory._buffer.push, tenant_id, [signal])
    return {"status": "ok", "signal_type": signal_type, "team_id": team_id, "buffer_count": count}


# ── Compression handlers ──────────────────────────────────────────────────────

async def _handle_compress_context(text: str, content_type: str | None) -> dict:
    comp = _get_compressor()
    compressed, meta = await asyncio.to_thread(comp.compress, text, content_type)
    return {
        "compressed_text": compressed,
        "cache_key": meta.get("cache_key"),
        "strategy": meta.get("strategy"),
        "original_tokens": meta.get("original_tokens"),
        "compressed_tokens": meta.get("compressed_tokens"),
        "saved_tokens": meta.get("saved_tokens", 0),
        "is_compressed": meta.get("strategy") != "none",
    }


async def _handle_retrieve_original(cache_key: str) -> dict:
    comp = _get_compressor()
    original = await asyncio.to_thread(comp.retrieve_by_key, cache_key)
    if original is None:
        return {"error": f"No cached original for key '{cache_key}'"}
    return {"original_text": original, "cache_key": cache_key}


# ── Prompt (MCP prompts API) ──────────────────────────────────────────────────

@app.list_prompts()
async def list_prompts() -> list[types.Prompt]:
    return [
        types.Prompt(
            name="load-memory",
            description="Load your memory for the active project into this conversation",
            arguments=[
                types.PromptArgument(name="profile", description="Profile name", required=False),
                types.PromptArgument(name="project", description="Project name", required=False),
            ],
        )
    ]


@app.get_prompt()
async def get_prompt(name: str, arguments: dict | None) -> types.GetPromptResult:
    if name != "load-memory":
        raise ValueError(f"Unknown prompt: {name}")
    arguments = arguments or {}
    injection = await _handle_recall(
        profile=arguments.get("profile"),
        project=arguments.get("project"),
    )
    return types.GetPromptResult(
        description="Active project memory",
        messages=[
            types.PromptMessage(
                role="user",
                content=types.TextContent(type="text", text=injection),
            )
        ],
    )


# ── Entry point ───────────────────────────────────────────────────────────────

async def _serve() -> None:
    async with stdio_server() as (read_stream, write_stream):
        await app.run(read_stream, write_stream, app.create_initialization_options())


def main() -> None:
    import sys as _sys
    logging.basicConfig(stream=_sys.stderr, level=logging.WARNING)
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    asyncio.run(_serve())


if __name__ == "__main__":
    main()
