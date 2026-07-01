"""
MemoryStore
-----------
Stores project memory under a clean three-level hierarchy:

  helin (user)
  ├── personal (profile)
  │   ├── general (project)  →  key: vmm:helin:personal:general
  │   └── kuika   (project)  →  key: vmm:helin:personal:kuika
  └── hobby (profile)
      └── side-thing         →  key: vmm:helin:hobby:side-thing

Key format:   vmm:{user}:{profile}:{project}
Value:        JSON {"items": [{category, text, confidence, stored_at}]}

Index key:    vmm:{user}:__index__
Value:        JSON {"personal": ["general", "kuika"], "hobby": ["side-thing"]}

ProjectStore is an alias for backward compat with existing imports.
"""

from __future__ import annotations

import json
import logging
import time
from typing import Any

logger = logging.getLogger(__name__)

_MAX_ENTRIES = 200
_TTL = 60 * 60 * 24 * 90  # 90 days

CATEGORIES = frozenset({"decision", "watch_out", "tried_failed", "convention", "goal", "preference"})

_LABELS: dict[str, str] = {
    "goal":         "Working on",
    "decision":     "Decisions made",
    "watch_out":    "Watch out for",
    "tried_failed": "Tried & failed",
    "convention":   "Conventions",
    "preference":   "Preferences",
}
_ORDER = ["goal", "decision", "watch_out", "tried_failed", "convention", "preference"]


def _n(s: str) -> str:
    """Normalize a name: lowercase + strip."""
    return s.strip().lower()


class MemoryStore:
    def __init__(self, redis_client: Any = None):
        self._redis = redis_client
        self._mem: dict[str, Any] = {}

    # ── Key helpers ──────────────────────────────────────────────────────────────

    def _key(self, user: str, profile: str, project: str) -> str:
        return f"vmm:{_n(user)}:{_n(profile)}:{_n(project)}"

    def _index_key(self, user: str) -> str:
        return f"vmm:{_n(user)}:__index__"

    # ── Profile / project index ───────────────────────────────────────────────────

    def _load_index(self, user: str) -> dict[str, list[str]]:
        key = self._index_key(user)
        if self._redis:
            raw = self._redis.get(key)
            return json.loads(raw) if raw else {}
        return dict(self._mem.get(key, {}))

    def _save_index(self, user: str, index: dict[str, list[str]]) -> None:
        key = self._index_key(user)
        data = json.dumps(index, ensure_ascii=False)
        if self._redis:
            self._redis.set(key, data)
        else:
            self._mem[key] = dict(index)

    def create_profile(self, user: str, name: str) -> bool:
        """Add a profile. Returns True if newly created."""
        p = _n(name)
        index = self._load_index(user)
        if p in index:
            return False
        index[p] = []
        self._save_index(user, index)
        logger.info(f"[memory_store] created profile '{p}' for {user}")
        return True

    def create_project(self, user: str, profile: str, name: str) -> bool:
        """Add a project under a profile. Creates profile if absent. Returns True if newly created."""
        p, pr = _n(profile), _n(name)
        index = self._load_index(user)
        projects = index.setdefault(p, [])
        if pr in projects:
            return False
        projects.append(pr)
        self._save_index(user, index)
        logger.info(f"[memory_store] created project '{p}/{pr}' for {user}")
        return True

    def list_profiles(self, user: str) -> list[tuple[str, int]]:
        """Return [(profile_name, project_count)] sorted alphabetically."""
        index = self._load_index(user)
        return sorted((p, len(pjs)) for p, pjs in index.items())

    def list_projects(self, user: str, profile: str) -> list[str]:
        """Return project names in a profile, in insertion order."""
        index = self._load_index(user)
        return list(index.get(_n(profile), []))

    # ── Memory entries ───────────────────────────────────────────────────────────

    def load(self, user: str, profile: str, project: str) -> list[dict]:
        key = self._key(user, profile, project)
        if self._redis:
            raw = self._redis.get(key)
            if not raw:
                return []
            data = json.loads(raw)
            return data.get("items", []) if isinstance(data, dict) else data
        stored = self._mem.get(key, {})
        return list(stored.get("items", []))

    def _save(self, user: str, profile: str, project: str, items: list[dict]) -> None:
        key = self._key(user, profile, project)
        trimmed = items[-_MAX_ENTRIES:]
        payload = json.dumps({"items": trimmed}, ensure_ascii=False)
        if self._redis:
            self._redis.setex(key, _TTL, payload)
        else:
            self._mem[key] = {"items": trimmed}

    def add_entry(
        self,
        user: str,
        profile: str,
        project: str,
        category: str,
        text: str,
        confidence: float = 0.8,
        dedup: bool = True,
    ) -> bool:
        """Add a memory item. Returns True if saved (False if empty or duplicate)."""
        if not text or not text.strip():
            return False
        if category not in CATEGORIES:
            category = "preference"

        # Ensure profile+project exist in the index
        self.create_project(user, profile, project)

        items = self.load(user, profile, project)

        if dedup:
            try:
                from magnet.local_embeddings import is_semantic_duplicate
                existing_texts = [e["text"] for e in items if e.get("category") == category]
                if is_semantic_duplicate(text, existing_texts):
                    logger.debug(f"[memory_store] dedup skip: {text[:60]!r}")
                    return False
            except Exception:
                pass

        items.append({
            "category": category,
            "text": text.strip(),
            "confidence": confidence,
            "stored_at": time.time(),
        })
        self._save(user, profile, project, items)
        logger.info(f"[memory_store] [{category}] saved for {user}/{profile}/{project}")
        return True

    # ── Format helpers ───────────────────────────────────────────────────────────

    def _group_by_category(self, items: list[dict]) -> dict[str, list[str]]:
        by_cat: dict[str, list[str]] = {c: [] for c in CATEGORIES}
        for e in items:
            c = e.get("category", "preference")
            if c in by_cat:
                by_cat[c].append(e["text"])
        return by_cat

    def format_for_injection(self, user: str, profile: str, project: str) -> str:
        """Compact string for system-prompt injection."""
        items = self.load(user, profile, project)
        if not items:
            return ""
        by_cat = self._group_by_category(items)
        lines: list[str] = []
        for cat in _ORDER:
            xs = by_cat.get(cat, [])
            if xs:
                lines.append(f"{_LABELS[cat]}:")
                for x in xs[-10:]:
                    lines.append(f"  - {x}")
        return "\n".join(lines)

    def format_for_display(self, user: str, profile: str, project: str) -> str:
        """Human-readable display of project memory."""
        items = self.load(user, profile, project)
        if not items:
            return f"No memory yet in {profile} / {project}."
        by_cat = self._group_by_category(items)
        lines = [f"Memory — {profile} / {project}:"]
        for cat in _ORDER:
            xs = by_cat.get(cat, [])
            if xs:
                lines.append(f"\n  {_LABELS[cat]}:")
                for x in xs[-15:]:
                    lines.append(f"    - {x}")
        return "\n".join(lines)


# Backward-compat alias
ProjectStore = MemoryStore
