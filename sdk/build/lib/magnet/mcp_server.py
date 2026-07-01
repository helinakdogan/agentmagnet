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
  show_project_memory  — display organized memory for the active project
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
_ACTIVE_FILE = Path.home() / ".agent-magnet" / "active.json"

# ── Active context ────────────────────────────────────────────────────────────

def _read_active_context() -> dict:
    try:
        if _ACTIVE_FILE.exists():
            return json.loads(_ACTIVE_FILE.read_text(encoding="utf-8"))
    except Exception:
        pass
    return {}


def _write_active_context(profile: str, project: str) -> None:
    _ACTIVE_FILE.parent.mkdir(parents=True, exist_ok=True)
    _ACTIVE_FILE.write_text(
        json.dumps({"profile": profile, "project": project}, indent=2),
        encoding="utf-8",
    )


def _resolve_context(profile: str | None = None, project: str | None = None) -> tuple[str, str, str]:
    """Return (user, profile, project) — fills gaps from active.json."""
    active = _read_active_context()
    resolved_profile = profile or active.get("profile") or "personal"
    resolved_project = project or active.get("project") or "general"
    return _DEFAULT_USER_ID, resolved_profile, resolved_project


def _ctx_tag(profile: str, project: str) -> str:
    return f"({profile} / {project})"


_SAVE_EVERY = int(os.environ.get("MAGNET_SAVE_EVERY", "8"))
_RHYTHM_FILE = Path.home() / ".agent-magnet" / "rhythm.json"


def _read_rhythm(profile: str, project: str) -> dict:
    key = f"{profile}/{project}"
    try:
        if _RHYTHM_FILE.exists():
            return json.loads(_RHYTHM_FILE.read_text(encoding="utf-8")).get(key, {})
    except Exception:
        pass
    return {}


def _write_rhythm(profile: str, project: str, **updates: Any) -> None:
    key = f"{profile}/{project}"
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
_usage_counter: Any = None
_compressor: Any = None


def _get_backend() -> Any:
    """Shared Redis or SQLite backend — initialized once."""
    global _backend
    if _backend is not None:
        return _backend

    redis_url = os.environ.get("MAGNET_REDIS_URL")
    client: Any = None
    if redis_url:
        try:
            import redis as redis_lib
            client = redis_lib.from_url(redis_url, decode_responses=True)
            client.ping()
            logger.info("[magnet] Redis connected")
        except Exception as e:
            logger.warning(f"[magnet] Redis unavailable ({e}); falling back to SQLite")

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
        enable_aggregate=bool(os.environ.get("MAGNET_REDIS_URL")),
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
    global _usage_counter
    if _usage_counter is None:
        from magnet.usage_counter import UsageCounter
        _usage_counter = UsageCounter(redis_client=_get_backend(), user_id=_DEFAULT_USER_ID)
    return _usage_counter


def _get_compressor() -> Any:
    global _compressor
    if _compressor is None:
        from magnet.compress import Compressor
        _compressor = Compressor()
    return _compressor


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

@app.call_tool()
async def call_tool(name: str, arguments: dict) -> list[types.TextContent]:
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

    body = await asyncio.to_thread(store.format_for_injection, user, profile, project)
    ctx = _ctx_tag(profile, project)

    if not body:
        return (
            f"Fresh start — no memory yet for {profile} / {project}. "
            f"I'll remember things as we work together. {ctx}"
        )

    lines = [
        f"You're working on {project} in {profile}. Here's what I know:",
        "",
        body,
        "",
        f"Apply this naturally. The user can override anything. {ctx}",
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
    if saved:
        return f"Saved [{category}]: \"{preview}\" {ctx}"
    return f"Already known (skipped duplicate): \"{preview[:60]}\" {ctx}"


async def _handle_show_project_memory(profile: str | None = None, project: str | None = None) -> str:
    user, profile, project = _resolve_context(profile, project)
    store = _get_memory_store()
    return await asyncio.to_thread(store.format_for_display, user, profile, project)


async def _handle_list_profiles() -> str:
    user = _DEFAULT_USER_ID
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
    user = _DEFAULT_USER_ID
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
    user = _DEFAULT_USER_ID
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
    else:
        storage_line = "cloud (Redis)"

    # Plan
    if os.environ.get("MAGNET_API_KEY"):
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

    lines = [
        f"Active:          {profile} / {project}",
        f"Storage:         {storage_line}",
        f"Save rhythm:     every ~{_SAVE_EVERY} user messages",
        f"Last checkpoint: {cp_line}",
        f"Total checkpoints: {total_cps}",
        f"Memories in project: {total_memories}",
        f"All-time writes: {total_writes} | recalls: {total_retrievals}",
        f"Plan:            {plan_line}",
    ]
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
        "user": _DEFAULT_USER_ID,
        "active_context": _ctx_tag(profile, project),
        "stats": stats,
        "note": "Metering active. Local mode is unlimited.",
    }


# ── Team handlers ─────────────────────────────────────────────────────────────

async def _handle_get_team_profile(team_id: str, project_id: str) -> dict:
    from magnet.tier import check_premium_feature, premium_required_response
    if not check_premium_feature("team_memory"):
        return premium_required_response()
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
    from magnet.tier import check_premium_feature, premium_required_response
    if not check_premium_feature("team_memory"):
        return premium_required_response()
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
