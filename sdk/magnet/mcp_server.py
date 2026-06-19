"""
MCP Server — Agent Magnet
-------------------------
Exposes Magnet's memory capabilities as MCP tools over stdio.

Requires Python 3.10+ and the `mcp` package:
    pip install mcp

Tools:
  get_profile    — Retrieve a user's learned preference profile
  inject_memory  — Build a system-prompt injection string
  add_signal     — Record a behavioral signal from a user interaction
  get_cold_start — Aggregate onboarding context for new users

Usage:
    python -m magnet.mcp_server
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import time
from typing import Any

import mcp.types as types
from mcp.server import Server
from mcp.server.stdio import stdio_server

logger = logging.getLogger(__name__)

_DEFAULT_PROJECT_ID = os.environ.get("MAGNET_PROJECT_ID", "default")

# ── Lazy-init singleton ───────────────────────────────────────────────────────

_memory: Any = None  # BehavioralMemory


def _get_memory() -> Any:
    global _memory
    if _memory is not None:
        return _memory

    redis_url = os.environ.get("MAGNET_REDIS_URL")
    local_mode = os.environ.get("MAGNET_LOCAL_MODE", "").lower() in ("1", "true", "yes")
    openai_key = os.environ.get("MAGNET_OPENAI_KEY") or os.environ.get("OPENAI_API_KEY")
    qdrant_url = os.environ.get("MAGNET_QDRANT_URL") or os.environ.get("QDRANT_URL")
    qdrant_api_key = os.environ.get("MAGNET_QDRANT_API_KEY") or os.environ.get("QDRANT_API_KEY")
    # Neo4j is read directly from NEO4J_URL / NEO4J_AUTH inside BehavioralMemory

    redis_client = None
    if redis_url:
        try:
            import redis as redis_lib
            redis_client = redis_lib.from_url(redis_url, decode_responses=True)
            redis_client.ping()
        except Exception as e:
            logger.warning(f"Redis unavailable ({e}); running in-memory only")
            redis_client = None
    elif local_mode:
        from magnet.local_store import SQLiteBackend
        redis_client = SQLiteBackend()
        logger.info("Local mode: using SQLite storage at ~/.agent-magnet/memory.db")

    from magnet.client import BehavioralMemory

    _memory = BehavioralMemory(
        openai_api_key=openai_key,
        redis_client=redis_client,
        qdrant_url=qdrant_url,
        qdrant_api_key=qdrant_api_key,
        enable_aggregate=not local_mode,
    )
    return _memory


# ── Server definition ─────────────────────────────────────────────────────────

app = Server("agent-magnet")


@app.list_prompts()
async def list_prompts() -> list[types.Prompt]:
    return [
        types.Prompt(
            name="load-memory",
            description="Load your memory profile into this conversation",
            arguments=[
                types.PromptArgument(
                    name="user_id",
                    description="Your user ID (leave blank to use MAGNET_USER_ID env var)",
                    required=False,
                ),
                types.PromptArgument(
                    name="project_id",
                    description="Project ID (default: 'default')",
                    required=False,
                ),
            ],
        )
    ]


@app.get_prompt()
async def get_prompt(name: str, arguments: dict | None) -> types.GetPromptResult:
    if name != "load-memory":
        raise ValueError(f"Unknown prompt: {name}")

    arguments = arguments or {}
    user_id = arguments.get("user_id") or os.environ.get("MAGNET_USER_ID", "default_user")
    project_id = arguments.get("project_id") or _DEFAULT_PROJECT_ID

    result = await _handle_inject_memory(user_id, project_id, None)
    injection = result.get("injection", "")

    if injection:
        content = (
            f"Memory profile for {user_id}:\n\n" + injection
        )
    else:
        content = f"No memory profile found yet for {user_id}."

    return types.GetPromptResult(
        description="Behavioral memory profile",
        messages=[
            types.PromptMessage(
                role="user",
                content=types.TextContent(type="text", text=content),
            ),
        ],
    )


@app.list_tools()
async def list_tools() -> list[types.Tool]:
    return [
        types.Tool(
            name="get_profile",
            description="Get the learned memory profile for a user",
            inputSchema={
                "type": "object",
                "properties": {
                    "user_id": {"type": "string", "description": "User identifier"},
                    "project_id": {"type": "string", "description": "Project identifier (defaults to MAGNET_PROJECT_ID env var)"},
                },
                "required": ["user_id"],
            },
        ),
        types.Tool(
            name="inject_memory",
            description="Get memory injection string for system prompt",
            inputSchema={
                "type": "object",
                "properties": {
                    "user_id": {"type": "string"},
                    "project_id": {"type": "string", "description": "Defaults to MAGNET_PROJECT_ID env var"},
                    "current_message": {
                        "type": "string",
                        "description": "The user's current message (used for episodic retrieval)",
                    },
                },
                "required": ["user_id"],
            },
        ),
        types.Tool(
            name="add_signal",
            description="Record a behavioral signal from user interaction",
            inputSchema={
                "type": "object",
                "properties": {
                    "user_id": {"type": "string"},
                    "project_id": {"type": "string", "description": "Defaults to MAGNET_PROJECT_ID env var"},
                    "messages": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "role": {"type": "string"},
                                "content": {"type": "string"},
                            },
                            "required": ["role", "content"],
                        },
                        "description": "Conversation messages",
                    },
                    "signal_type": {
                        "type": "string",
                        "enum": ["correction", "rejection", "preference_like", "preference_dislike", "tone_preference"],
                        "description": "The type of behavioral signal detected",
                    },
                },
                "required": ["user_id", "messages", "signal_type"],
            },
        ),
        types.Tool(
            name="get_cold_start",
            description="Get cold start profile for a new user",
            inputSchema={
                "type": "object",
                "properties": {
                    "project_id": {"type": "string", "description": "Defaults to MAGNET_PROJECT_ID env var"},
                    "context": {
                        "type": "string",
                        "description": "Current query context (e.g. 'coding', 'writing', 'general_chat')",
                    },
                },
                "required": [],
            },
        ),
        types.Tool(
            name="end_session",
            description="Summarize and save current session to memory",
            inputSchema={
                "type": "object",
                "properties": {
                    "user_id": {"type": "string", "description": "User identifier"},
                    "project_id": {"type": "string", "description": "Defaults to MAGNET_PROJECT_ID env var"},
                    "messages": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "role": {"type": "string"},
                                "content": {"type": "string"},
                            },
                            "required": ["role", "content"],
                        },
                        "description": "Conversation messages to summarize",
                    },
                },
                "required": ["user_id", "messages"],
            },
        ),
        types.Tool(
            name="save_session",
            description="Manually save current session to memory (use when the Stop hook is unavailable)",
            inputSchema={
                "type": "object",
                "properties": {
                    "user_id": {"type": "string", "description": "User identifier"},
                    "project_id": {"type": "string", "description": "Defaults to MAGNET_PROJECT_ID env var"},
                    "messages": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "role": {"type": "string"},
                                "content": {"type": "string"},
                            },
                            "required": ["role", "content"],
                        },
                        "description": "Conversation messages to summarize and save",
                    },
                },
                "required": ["user_id", "messages"],
            },
        ),
        # ── Team memory tools ─────────────────────────────────────────
        types.Tool(
            name="get_team_profile",
            description="Get the shared memory profile for a team (requires Redis)",
            inputSchema={
                "type": "object",
                "properties": {
                    "team_id": {"type": "string", "description": "Team identifier"},
                    "project_id": {"type": "string", "description": "Project identifier"},
                },
                "required": ["team_id"],
            },
        ),
        types.Tool(
            name="add_team_signal",
            description="Record a behavioral signal directly to team-scoped memory (requires Redis)",
            inputSchema={
                "type": "object",
                "properties": {
                    "team_id": {"type": "string"},
                    "project_id": {"type": "string", "description": "Defaults to MAGNET_PROJECT_ID env var"},
                    "messages": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {"role": {"type": "string"}, "content": {"type": "string"}},
                            "required": ["role", "content"],
                        },
                        "description": "Conversation messages",
                    },
                    "signal_type": {
                        "type": "string",
                        "enum": ["correction", "rejection", "preference_like", "preference_dislike",
                                 "tone_preference", "watch_out"],
                    },
                },
                "required": ["team_id", "messages", "signal_type"],
            },
        ),
        types.Tool(
            name="get_merged_injection",
            description="Get merged memory injection combining user + team + org memory (user scope wins)",
            inputSchema={
                "type": "object",
                "properties": {
                    "user_id": {"type": "string"},
                    "team_id": {"type": "string", "description": "Optional team identifier"},
                    "org_id": {"type": "string", "description": "Optional org identifier"},
                    "project_id": {"type": "string", "description": "Defaults to MAGNET_PROJECT_ID env var"},
                    "current_message": {"type": "string", "description": "Current user message for episodic retrieval"},
                },
                "required": ["user_id"],
            },
        ),
        types.Tool(
            name="get_project_memory",
            description="Get a per-user breakdown of what was learned in a project (requires Redis)",
            inputSchema={
                "type": "object",
                "properties": {
                    "project_id": {"type": "string", "description": "Project identifier"},
                    "team_id": {"type": "string", "description": "Optional team identifier to include team-shared memory"},
                },
                "required": ["project_id"],
            },
        ),
        types.Tool(
            name="share_to_team",
            description="Explicitly share one preference from a user's personal memory to their team (requires Redis)",
            inputSchema={
                "type": "object",
                "properties": {
                    "user_id": {"type": "string", "description": "User who owns the preference"},
                    "project_id": {"type": "string", "description": "Defaults to MAGNET_PROJECT_ID env var"},
                    "fact_or_subject": {"type": "string", "description": "Subject or text of the preference to share"},
                    "team_id": {"type": "string", "description": "Target team identifier"},
                },
                "required": ["user_id", "fact_or_subject", "team_id"],
            },
        ),
        types.Tool(
            name="forget_team",
            description="Remove a specific preference from team memory by subject match (requires Redis)",
            inputSchema={
                "type": "object",
                "properties": {
                    "team_id": {"type": "string"},
                    "project_id": {"type": "string", "description": "Defaults to MAGNET_PROJECT_ID env var"},
                    "fact_or_subject": {"type": "string", "description": "Subject or text to remove from team memory"},
                },
                "required": ["team_id", "fact_or_subject"],
            },
        ),
    ]


@app.call_tool()
async def call_tool(name: str, arguments: dict) -> list[types.TextContent]:
    try:
        project_id = arguments.get("project_id") or _DEFAULT_PROJECT_ID
        if name == "get_profile":
            result = await _handle_get_profile(
                arguments["user_id"],
                project_id,
            )
        elif name == "inject_memory":
            result = await _handle_inject_memory(
                arguments["user_id"],
                project_id,
                arguments.get("current_message"),
            )
        elif name == "add_signal":
            result = await _handle_add_signal(
                arguments["user_id"],
                project_id,
                arguments["messages"],
                arguments["signal_type"],
            )
        elif name == "get_cold_start":
            result = await _handle_get_cold_start(
                project_id,
                arguments.get("context"),
            )
        elif name == "end_session":
            result = await _handle_end_session(
                arguments["user_id"],
                project_id,
                arguments["messages"],
            )
        elif name == "save_session":
            result = await _handle_end_session(
                arguments["user_id"],
                project_id,
                arguments["messages"],
            )
        elif name == "get_team_profile":
            result = await _handle_get_team_profile(
                arguments["team_id"],
                project_id,
            )
        elif name == "add_team_signal":
            result = await _handle_add_team_signal(
                arguments["team_id"],
                project_id,
                arguments["messages"],
                arguments["signal_type"],
            )
        elif name == "get_merged_injection":
            result = await _handle_get_merged_injection(
                arguments["user_id"],
                arguments.get("team_id"),
                arguments.get("org_id"),
                project_id,
                arguments.get("current_message"),
            )
        elif name == "get_project_memory":
            result = await _handle_get_project_memory(
                project_id,
                arguments.get("team_id"),
            )
        elif name == "share_to_team":
            result = await _handle_share_to_team(
                arguments["user_id"],
                project_id,
                arguments["fact_or_subject"],
                arguments["team_id"],
            )
        elif name == "forget_team":
            result = await _handle_forget_team(
                arguments["team_id"],
                project_id,
                arguments["fact_or_subject"],
            )
        else:
            result = {"error": f"Unknown tool: {name}"}
    except Exception as e:
        logger.error(f"Tool {name} error: {e}", exc_info=True)
        result = {"error": str(e)}

    return [types.TextContent(type="text", text=json.dumps(result, ensure_ascii=False))]


# ── Tool handlers ─────────────────────────────────────────────────────────────

async def _handle_get_profile(user_id: str, project_id: str) -> dict:
    memory = _get_memory()
    profile = await asyncio.to_thread(memory.get_profile, user_id, project_id)
    if not profile:
        return {"user_id": user_id, "project_id": project_id, "prefers": [], "dislikes": [], "expects": []}

    from magnet.store import effective_confidence

    prefers, dislikes, expects, watch_out = [], [], [], []
    for pref in profile.get("preferences", []):
        if not isinstance(pref, dict):
            continue
        conf = round(effective_confidence(pref), 3)
        if conf < 0.1:
            continue
        entry = {"text": pref.get("natural_text", pref.get("subject", "")), "confidence": conf}
        relation = pref.get("relation", "")
        if relation == "prefers":
            prefers.append(entry)
        elif relation == "dislikes":
            dislikes.append(entry)
        elif relation == "expects":
            expects.append(entry)
        elif relation == "watch_out":
            watch_out.append(entry)

    reflected_at = profile.get("reflected_at")
    age_days = round((time.time() - reflected_at) / 86400, 1) if reflected_at else None

    return {
        "user_id": user_id,
        "project_id": project_id,
        "prefers": sorted(prefers, key=lambda x: -x["confidence"]),
        "dislikes": sorted(dislikes, key=lambda x: -x["confidence"]),
        "expects": sorted(expects, key=lambda x: -x["confidence"]),
        "watch_out": sorted(watch_out, key=lambda x: -x["confidence"]),
        "profile_age_days": age_days,
        "signal_count": profile.get("signal_count", 0),
    }


async def _handle_inject_memory(user_id: str, project_id: str, current_message: str | None) -> dict:
    memory = _get_memory()
    current_messages = (
        [{"role": "user", "content": current_message}] if current_message else None
    )
    injection = await asyncio.to_thread(
        memory.get_injection, user_id, project_id, current_messages
    )
    return {"injection": injection, "has_profile": bool(injection)}


async def _handle_add_signal(
    user_id: str, project_id: str, messages: list[dict], signal_type: str
) -> dict:
    memory = _get_memory()
    tenant_id = f"{project_id}:{user_id}"

    user_msgs = [m for m in messages if m.get("role") == "user"]
    if not user_msgs:
        return {"status": "ok", "signal_type": signal_type, "buffered": False}

    last_msg = user_msgs[-1].get("content", "")

    signal: dict = {
        "type": signal_type,
        "message": last_msg[:200],
        "confidence": 0.75,
    }

    # Instant-learn path for strong signals (mirrors client.py logic)
    instant_types = {"preference_like", "preference_dislike", "tone_preference"}
    if signal_type in instant_types:
        existing = await asyncio.to_thread(memory._store.load, tenant_id)
        updated = await asyncio.to_thread(
            memory._reflector.instant_learn,
            tenant_id,
            signal_type,
            last_msg[:200],
            0.75,
            existing,
        )
        await asyncio.to_thread(memory._store.save, tenant_id, updated)
        memory._profile_cache.pop(tenant_id, None)
        return {"status": "ok", "signal_type": signal_type, "instant_learned": True}

    # Buffer soft signals
    count = await asyncio.to_thread(memory._buffer.push, tenant_id, [signal])
    reflected = False
    if await asyncio.to_thread(memory._buffer.should_reflect, tenant_id):
        signals_to_reflect = await asyncio.to_thread(memory._buffer.flush, tenant_id)
        if signals_to_reflect:
            existing = await asyncio.to_thread(memory._store.load, tenant_id)
            profile = await asyncio.to_thread(
                memory._reflector.reflect, tenant_id, signals_to_reflect, existing
            )
            await asyncio.to_thread(memory._store.save, tenant_id, profile)
            memory._profile_cache.pop(tenant_id, None)
            reflected = True

    return {
        "status": "ok",
        "signal_type": signal_type,
        "buffer_count": count,
        "reflected": reflected,
    }


async def _handle_end_session(user_id: str, project_id: str, messages: list[dict]) -> dict:
    memory = _get_memory()
    return await asyncio.to_thread(memory.session_end, user_id, project_id, messages)


async def _handle_get_team_profile(team_id: str, project_id: str) -> dict:
    from magnet.team_store import TeamMemoryRequiresRedis
    memory = _get_memory()
    try:
        profile = await asyncio.to_thread(
            memory._team_store.load_team_profile, team_id, project_id
        )
    except TeamMemoryRequiresRedis as e:
        return {"error": str(e)}
    if not profile:
        return {"team_id": team_id, "project_id": project_id, "prefers": [], "dislikes": [], "expects": [], "watch_out": []}

    buckets: dict[str, list] = {"prefers": [], "dislikes": [], "expects": [], "watch_out": []}
    for pref in profile.get("preferences", []):
        if not isinstance(pref, dict):
            continue
        relation = pref.get("relation", "")
        if relation not in buckets:
            continue
        buckets[relation].append({
            "text": pref.get("natural_text", pref.get("subject", "")),
            "confidence": round(pref.get("confidence", 0.5), 3),
            "shared_by": pref.get("shared_by"),
        })
    return {"team_id": team_id, "project_id": project_id, **buckets}


async def _handle_add_team_signal(
    team_id: str, project_id: str, messages: list[dict], signal_type: str
) -> dict:
    from magnet.team_store import TeamMemoryRequiresRedis
    memory = _get_memory()
    try:
        memory._team_store._require_redis()
    except TeamMemoryRequiresRedis as e:
        return {"error": str(e)}

    user_msgs = [m for m in messages if m.get("role") == "user"]
    if not user_msgs:
        return {"status": "ok", "signal_type": signal_type, "buffered": False}

    last_msg = user_msgs[-1].get("content", "")
    tenant_id = f"{project_id}:{team_id}"

    instant_types = {"preference_like", "preference_dislike", "tone_preference", "watch_out"}
    if signal_type in instant_types:
        existing = await asyncio.to_thread(memory._team_store.load_team_profile, team_id, project_id) or {}
        updated = await asyncio.to_thread(
            memory._reflector.instant_learn,
            tenant_id,
            signal_type,
            last_msg[:200],
            0.75,
            existing,
        )
        await asyncio.to_thread(memory._team_store.save_team_profile, team_id, project_id, updated)
        return {"status": "ok", "signal_type": signal_type, "team_id": team_id, "instant_learned": True}

    signal = {"type": signal_type, "message": last_msg[:200], "confidence": 0.6}
    count = await asyncio.to_thread(memory._buffer.push, tenant_id, [signal])
    reflected = False
    if await asyncio.to_thread(memory._buffer.should_reflect, tenant_id):
        signals_to_reflect = await asyncio.to_thread(memory._buffer.flush, tenant_id)
        if signals_to_reflect:
            existing = await asyncio.to_thread(memory._team_store.load_team_profile, team_id, project_id) or {}
            profile = await asyncio.to_thread(
                memory._reflector.reflect, tenant_id, signals_to_reflect, existing
            )
            await asyncio.to_thread(memory._team_store.save_team_profile, team_id, project_id, profile)
            reflected = True

    return {"status": "ok", "signal_type": signal_type, "team_id": team_id,
            "buffer_count": count, "reflected": reflected}


async def _handle_get_merged_injection(
    user_id: str,
    team_id: str | None,
    org_id: str | None,
    project_id: str,
    current_message: str | None,
) -> dict:
    memory = _get_memory()
    current_msgs = [{"role": "user", "content": current_message}] if current_message else None
    injection = await asyncio.to_thread(
        memory.get_injection_with_team, user_id, project_id, team_id, org_id, current_msgs
    )
    return {"injection": injection, "has_profile": bool(injection), "team_id": team_id, "org_id": org_id}


async def _handle_get_project_memory(project_id: str, team_id: str | None) -> dict:
    from magnet.team_store import TeamMemoryRequiresRedis
    memory = _get_memory()
    try:
        result = await asyncio.to_thread(memory.get_project_memory, project_id, team_id)
    except TeamMemoryRequiresRedis as e:
        return {"error": str(e)}
    return result


async def _handle_share_to_team(
    user_id: str, project_id: str, fact_or_subject: str, team_id: str
) -> dict:
    from magnet.team_store import TeamMemoryRequiresRedis
    memory = _get_memory()
    try:
        result = await asyncio.to_thread(
            memory.share_to_team, user_id, project_id, fact_or_subject, team_id
        )
    except TeamMemoryRequiresRedis as e:
        return {"error": str(e)}
    return result


async def _handle_forget_team(team_id: str, project_id: str, fact_or_subject: str) -> dict:
    from magnet.team_store import TeamMemoryRequiresRedis
    memory = _get_memory()
    try:
        result = await asyncio.to_thread(
            memory._team_store.forget_team, team_id, project_id, fact_or_subject
        )
    except TeamMemoryRequiresRedis as e:
        return {"error": str(e)}
    return result


async def _handle_get_cold_start(project_id: str, context: str | None) -> dict:
    memory = _get_memory()
    category = context or "general_chat"

    if memory._aggregate:
        injection = await asyncio.to_thread(
            memory._aggregate.get_cold_start_injection, category
        )
        if injection:
            return {
                "project_id": project_id,
                "context": category,
                "injection": injection,
                "has_data": True,
            }

    return {
        "project_id": project_id,
        "context": category,
        "injection": "",
        "has_data": False,
    }


# ── Entry point ───────────────────────────────────────────────────────────────

async def _serve() -> None:
    async with stdio_server() as (read_stream, write_stream):
        await app.run(
            read_stream,
            write_stream,
            app.create_initialization_options(),
        )


def main() -> None:
    logging.basicConfig(level=logging.WARNING)
    asyncio.run(_serve())


if __name__ == "__main__":
    main()
