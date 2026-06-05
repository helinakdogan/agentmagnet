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

# ── Lazy-init singleton ───────────────────────────────────────────────────────

_memory: Any = None  # BehavioralMemory


def _get_memory() -> Any:
    global _memory
    if _memory is not None:
        return _memory

    redis_url = os.environ.get("MAGNET_REDIS_URL")
    openai_key = os.environ.get("MAGNET_OPENAI_KEY")

    redis_client = None
    if redis_url:
        try:
            import redis as redis_lib
            redis_client = redis_lib.from_url(redis_url, decode_responses=True)
            redis_client.ping()
        except Exception as e:
            logger.warning(f"Redis unavailable ({e}); running in-memory only")
            redis_client = None

    from magnet.client import BehavioralMemory

    _memory = BehavioralMemory(
        openai_api_key=openai_key,
        redis_client=redis_client,
    )
    return _memory


# ── Server definition ─────────────────────────────────────────────────────────

app = Server("agent-magnet")


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
                    "project_id": {"type": "string", "description": "Project identifier"},
                },
                "required": ["user_id", "project_id"],
            },
        ),
        types.Tool(
            name="inject_memory",
            description="Get memory injection string for system prompt",
            inputSchema={
                "type": "object",
                "properties": {
                    "user_id": {"type": "string"},
                    "project_id": {"type": "string"},
                    "current_message": {
                        "type": "string",
                        "description": "The user's current message (used for episodic retrieval)",
                    },
                },
                "required": ["user_id", "project_id"],
            },
        ),
        types.Tool(
            name="add_signal",
            description="Record a behavioral signal from user interaction",
            inputSchema={
                "type": "object",
                "properties": {
                    "user_id": {"type": "string"},
                    "project_id": {"type": "string"},
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
                "required": ["user_id", "project_id", "messages", "signal_type"],
            },
        ),
        types.Tool(
            name="get_cold_start",
            description="Get cold start profile for a new user",
            inputSchema={
                "type": "object",
                "properties": {
                    "project_id": {"type": "string"},
                    "context": {
                        "type": "string",
                        "description": "Current query context (e.g. 'coding', 'writing', 'general_chat')",
                    },
                },
                "required": ["project_id"],
            },
        ),
    ]


@app.call_tool()
async def call_tool(name: str, arguments: dict) -> list[types.TextContent]:
    try:
        if name == "get_profile":
            result = await _handle_get_profile(
                arguments["user_id"],
                arguments["project_id"],
            )
        elif name == "inject_memory":
            result = await _handle_inject_memory(
                arguments["user_id"],
                arguments["project_id"],
                arguments.get("current_message"),
            )
        elif name == "add_signal":
            result = await _handle_add_signal(
                arguments["user_id"],
                arguments["project_id"],
                arguments["messages"],
                arguments["signal_type"],
            )
        elif name == "get_cold_start":
            result = await _handle_get_cold_start(
                arguments["project_id"],
                arguments.get("context"),
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

    prefers, dislikes, expects = [], [], []
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

    reflected_at = profile.get("reflected_at")
    age_days = round((time.time() - reflected_at) / 86400, 1) if reflected_at else None

    return {
        "user_id": user_id,
        "project_id": project_id,
        "prefers": sorted(prefers, key=lambda x: -x["confidence"]),
        "dislikes": sorted(dislikes, key=lambda x: -x["confidence"]),
        "expects": sorted(expects, key=lambda x: -x["confidence"]),
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
