#!/usr/bin/env python3
"""
Claude Code Stop hook — saves session to Agent Magnet memory.

Called automatically when a Claude Code session ends.
Receives conversation data as JSON via stdin.

Required env vars:
  MAGNET_USER_ID    — user identifier
  MAGNET_REDIS_URL  — Redis connection URL (optional but recommended)
  MAGNET_OPENAI_KEY — OpenAI API key for LLM summarization
"""

from __future__ import annotations

import json
import os
import sys

MAX_MESSAGES = 30


def _extract_messages(data: object) -> list[dict]:
    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        for key in ("transcript", "messages", "conversation"):
            candidate = data.get(key)
            if isinstance(candidate, list):
                return candidate
    return []


def main() -> None:
    try:
        raw = sys.stdin.read()
        if not raw.strip():
            return

        data = json.loads(raw)
        messages = _extract_messages(data)
        if not messages:
            return

        normalized: list[dict] = []
        for m in messages[-MAX_MESSAGES:]:
            if isinstance(m, dict) and m.get("role") and m.get("content"):
                normalized.append({"role": m["role"], "content": str(m["content"])})

        if not normalized:
            return

        user_id = os.environ.get("MAGNET_USER_ID")
        if not user_id:
            return

        redis_url = os.environ.get("MAGNET_REDIS_URL")
        openai_key = os.environ.get("MAGNET_OPENAI_KEY") or os.environ.get("OPENAI_API_KEY")

        redis_client = None
        if redis_url:
            import redis as redis_lib
            redis_client = redis_lib.from_url(redis_url, decode_responses=True)

        from magnet.client import BehavioralMemory

        memory = BehavioralMemory(
            openai_api_key=openai_key,
            redis_client=redis_client,
        )
        memory.session_end(user_id=user_id, messages=normalized)

        print("Magnet: session saved")

    except Exception:
        pass  # Never crash Claude Code


if __name__ == "__main__":
    main()
