"""
ConsolidationEngine
-------------------
Scheduled batch process for cross-user pattern learning.
Scans episodic memory across multiple users, finds common
behavioral patterns, and writes abstractions to aggregate_store.
"""

from __future__ import annotations

import asyncio
import difflib
import json
import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .episodic_store import EpisodicStore
    from .store import ProfileStore
    from .aggregate_store import AggregateSignalStore

logger = logging.getLogger(__name__)

_MIN_SUPPORT = 3
_SIMILARITY_THRESHOLD = 0.75
_EPISODIC_KEY_PREFIX = "magnet:episodic:"
_QDRANT_COLLECTION = "magnet_episodes"

_TOPIC_KEYWORDS: dict[str, list[str]] = {
    "food": [
        "food", "meal", "eat", "drink", "restaurant", "cook", "recipe",
        "diet", "mushroom", "pizza", "pasta", "meat", "vegetarian", "vegan",
        "spicy", "breakfast", "lunch", "dinner", "snack", "cuisine",
    ],
    "format": [
        "bullet", "list", "paragraph", "format", "structure", "markdown",
        "table", "json", "summary", "brief", "concise", "detailed", "header",
    ],
    "communication": [
        "tone", "formal", "casual", "professional", "friendly",
        "technical", "simple", "explain", "style", "language",
    ],
}


class ConsolidationEngine:
    """
    Scheduled batch process.
    Scans episodic memory across multiple users,
    finds common behavioral patterns,
    writes abstractions to aggregate_store.

    Cross-user learning — what one user
    teaches the system benefits similar users.
    """

    def __init__(
        self,
        episodic_store: "EpisodicStore",
        profile_store: "ProfileStore",
        aggregate_store: "AggregateSignalStore | None",
    ) -> None:
        self._episodic = episodic_store
        self._profiles = profile_store
        self._aggregate = aggregate_store

    async def run(self, tenant_prefix: str) -> dict:
        """
        Main consolidation cycle:

        1. Load all episodes for this tenant_prefix
           (e.g. all stores under one project)

        2. Group episodes by topic/context:
           - meal/food related
           - format preferences
           - communication style
           - domain-specific (detect from content)

        3. Find patterns that appear in 3+ users:
           - Same preference type
           - Same relation (dislikes/prefers)
           - Similar subject (use difflib ratio > 0.75)

        4. For each validated pattern, write to aggregate store.

        5. Return summary dict.
        """
        summary = {
            "episodes_scanned": 0,
            "patterns_found": 0,
            "patterns_written": 0,
            "skipped_low_support": 0,
        }

        episodes_by_user = await asyncio.to_thread(
            self._load_all_episodes, tenant_prefix
        )
        if not episodes_by_user:
            logger.info(
                f"ConsolidationEngine: no episodes for prefix={tenant_prefix!r}, exiting."
            )
            return summary

        summary["episodes_scanned"] = sum(
            len(eps) for eps in episodes_by_user.values()
        )

        grouped = self._group_by_topic(episodes_by_user)

        user_prefs = await asyncio.to_thread(
            self._load_user_preferences, list(episodes_by_user.keys())
        )

        patterns = self._find_patterns(user_prefs, grouped)
        summary["patterns_found"] = len(patterns)

        written = 0
        skipped = 0
        for pattern in patterns:
            if pattern["support"] < _MIN_SUPPORT:
                skipped += 1
                continue
            if self._aggregate is not None:
                wrote = await asyncio.to_thread(
                    self._aggregate.store_consolidated_pattern, pattern
                )
                if wrote:
                    written += 1

        summary["patterns_written"] = written
        summary["skipped_low_support"] = skipped

        logger.info(
            f"ConsolidationEngine: prefix={tenant_prefix!r} | "
            f"episodes={summary['episodes_scanned']} | "
            f"patterns_found={summary['patterns_found']} | "
            f"written={summary['patterns_written']} | "
            f"skipped={summary['skipped_low_support']}"
        )
        return summary

    async def schedule(self, tenant_prefix: str, interval_hours: int = 24) -> None:
        """
        Run consolidation every interval_hours using asyncio.
        Designed to be called once at startup.
        Log each run result.
        """
        while True:
            try:
                result = await self.run(tenant_prefix)
                logger.info(f"ConsolidationEngine scheduled run complete: {result}")
            except Exception as e:
                logger.error(
                    f"ConsolidationEngine scheduled run error: {e}", exc_info=True
                )
            await asyncio.sleep(interval_hours * 3600)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _load_all_episodes(self, tenant_prefix: str) -> dict[str, list[dict]]:
        """Returns {tenant_id: [episodes]} for all users under tenant_prefix."""
        result: dict[str, list[dict]] = {}

        if self._episodic._qdrant_available:
            try:
                offset = None
                while True:
                    points, offset = self._episodic._qdrant.scroll(
                        collection_name=_QDRANT_COLLECTION,
                        limit=100,
                        offset=offset,
                        with_payload=True,
                        with_vectors=False,
                    )
                    for point in points:
                        tid = point.payload.get("tenant_id", "")
                        if tid.startswith(f"{tenant_prefix}:"):
                            result.setdefault(tid, []).append(point.payload)
                    if offset is None:
                        break
            except Exception as e:
                logger.error(f"ConsolidationEngine: Qdrant scroll error: {e}")
            return result

        if self._episodic._redis:
            pattern = f"{_EPISODIC_KEY_PREFIX}{tenant_prefix}:*"
            try:
                for raw_key in self._episodic._redis.scan_iter(pattern):
                    key = raw_key if isinstance(raw_key, str) else raw_key.decode()
                    tenant_id = key[len(_EPISODIC_KEY_PREFIX):]
                    items = self._episodic._redis.zrevrange(key, 0, -1)
                    episodes = [json.loads(i) for i in items]
                    if episodes:
                        result[tenant_id] = episodes
            except Exception as e:
                logger.error(f"ConsolidationEngine: Redis scan error: {e}")
            return result

        # In-memory fallback
        prefix = f"{tenant_prefix}:"
        for tenant_id, episodes in self._episodic._memory.items():
            if tenant_id.startswith(prefix):
                result[tenant_id] = list(episodes)
        return result

    def _group_by_topic(
        self, episodes_by_user: dict[str, list[dict]]
    ) -> dict[str, dict[str, list[dict]]]:
        """Returns {topic: {tenant_id: [episodes]}}."""
        grouped: dict[str, dict[str, list[dict]]] = {}
        for tenant_id, episodes in episodes_by_user.items():
            for episode in episodes:
                topic = self._detect_topic(episode)
                grouped.setdefault(topic, {}).setdefault(tenant_id, []).append(episode)
        return grouped

    def _detect_topic(self, episode: dict) -> str:
        summary = episode.get("summary", "")
        msgs = episode.get("messages", [])
        text = (
            summary + " "
            + " ".join(m.get("content", "") for m in msgs if m.get("role") == "user")
        ).lower()
        for topic, keywords in _TOPIC_KEYWORDS.items():
            if any(kw in text for kw in keywords):
                return topic
        return "general"

    def _load_user_preferences(
        self, tenant_ids: list[str]
    ) -> dict[str, list[dict]]:
        """Returns {tenant_id: [preferences]} for users who have profiles."""
        result: dict[str, list[dict]] = {}
        for tenant_id in tenant_ids:
            profile = self._profiles.load(tenant_id)
            if not profile:
                continue
            prefs = [p for p in profile.get("preferences", []) if isinstance(p, dict)]
            if prefs:
                result[tenant_id] = prefs
        return result

    def _find_patterns(
        self,
        user_prefs: dict[str, list[dict]],
        grouped: dict[str, dict[str, list[dict]]],
    ) -> list[dict]:
        """
        Cluster preferences by (relation, ~subject) across users.
        Only subjects with difflib ratio > _SIMILARITY_THRESHOLD are grouped.
        Returns all clusters regardless of support — callers filter by _MIN_SUPPORT.
        """
        entries: list[tuple[str, dict, str]] = [
            (tid, pref, self._infer_context(tid, pref, grouped))
            for tid, prefs in user_prefs.items()
            for pref in prefs
        ]

        used = [False] * len(entries)
        patterns: list[dict] = []

        for i, (tid_i, pref_i, ctx_i) in enumerate(entries):
            if used[i]:
                continue
            relation = pref_i.get("relation", "")
            subj_i = pref_i.get("subject", "").lower()
            if not subj_i or not relation:
                continue

            cluster_users: set[str] = {tid_i}
            cluster_subjects = [pref_i.get("subject", "")]
            cluster_confidences = [pref_i.get("confidence", 0.5)]

            for j, (tid_j, pref_j, _) in enumerate(entries):
                if i == j or used[j] or tid_j in cluster_users:
                    continue
                if pref_j.get("relation", "") != relation:
                    continue
                subj_j = pref_j.get("subject", "").lower()
                if not subj_j:
                    continue
                ratio = difflib.SequenceMatcher(None, subj_i, subj_j).ratio()
                if ratio >= _SIMILARITY_THRESHOLD:
                    cluster_users.add(tid_j)
                    cluster_subjects.append(pref_j.get("subject", ""))
                    cluster_confidences.append(pref_j.get("confidence", 0.5))
                    used[j] = True

            used[i] = True
            canonical = min(cluster_subjects, key=len)
            avg_confidence = sum(cluster_confidences) / len(cluster_confidences)
            patterns.append({
                "subject": canonical,
                "relation": relation,
                "confidence": round(avg_confidence, 3),
                "support": len(cluster_users),
                "context": ctx_i,
                "source": "consolidation",
            })

        return patterns

    def _infer_context(
        self,
        tenant_id: str,
        pref: dict,
        grouped: dict[str, dict[str, list[dict]]],
    ) -> str:
        """Infers context from preference content, falling back to episode topics."""
        text = (
            pref.get("subject", "") + " " + pref.get("natural_text", "")
        ).lower()
        for topic, keywords in _TOPIC_KEYWORDS.items():
            if any(kw in text for kw in keywords):
                return topic
        for topic, users in grouped.items():
            if tenant_id in users:
                return topic
        return "general"
