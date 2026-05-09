"""
MemoryOrchestrator
------------------
Memory Orchestrator — Inter-layer coordination

Rule-based orchestrator that decides which memory layer should be
activated per request and merges context from selected layers.
"""

from __future__ import annotations

import logging
import re
from typing import TYPE_CHECKING

from .episodic_store import EpisodicStore
from .knowledge_store import KnowledgeStore
from .store import ProfileStore

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


class MemoryOrchestrator:
    """
    Determines which memory layer operates using rule-based heuristics.

    - Behavioral: Always active on every request.
    - Episodic:   Activated if the query references past interactions.
    - Knowledge:  Graph-based long-term memory.

    Args:
        behavioral_store: Instance of ProfileStore.
        episodic_store:   Instance of EpisodicStore.
        knowledge_store:  Instance of KnowledgeStore.
    """

    # Regex patterns triggering the episodic layer
    EPISODIC_TRIGGERS: list[str] = [
        r"\bgeçen\b",
        r"\bdaha önce\b",
        r"\bhatırlıyor musun\b",
        r"\beverything\b",
        r"\bpreviously\b",
        r"\blast time\b",
        r"\brecently\b",
        r"\bwe discussed\b",
        r"\byou mentioned\b",
        r"\bgeçen sefer\b",
        r"\bönce konuşmuştuk\b",
        r"\bdo you remember\b",
        r"\bsöylemiştik\b",
        r"\bkonuşmuştuk\b",
    ]

    def __init__(
        self,
        behavioral_store: ProfileStore,
        episodic_store: EpisodicStore,
        knowledge_store: KnowledgeStore,
    ) -> None:
        self._behavioral = behavioral_store
        self._episodic = episodic_store
        self._knowledge = knowledge_store
        self._episodic_re = re.compile(
            "|".join(self.EPISODIC_TRIGGERS),
            re.IGNORECASE,
        )

    # ------------------------------------------------------------------
    # Core Decision Logic
    # ------------------------------------------------------------------

    def decide(self, query: str, tenant_id: str) -> dict[str, bool]:
        """
        Decides which layers to activate based on the incoming query.

        Args:
            query:     The user's latest message.
            tenant_id: Tenant ID in ``project_id:user_id`` format.

        Returns:
            Dict: ``{"behavioral": True, "episodic": bool, "knowledge": bool}``
        """
        use_episodic = bool(self._episodic_re.search(query))

        use_knowledge = False

        decision = {
            "behavioral": True,
            "episodic": use_episodic,
            "knowledge": use_knowledge,
        }

        logger.debug(f"MemoryOrchestrator.decide: {decision} (tenant={tenant_id})")
        return decision

    def build_context(
        self,
        query: str,
        tenant_id: str,
        current_messages: list[dict] | None = None,
    ) -> str:
        """
        Gathers and merges context from all activated memory layers.

        Behavioral injection is always included.
        Episodic context is added only when a trigger pattern matches.

        Args:
            query:            The user's latest message (used for trigger detection).
            tenant_id:        Tenant ID in ``project_id:user_id`` format.
            current_messages: Current conversation history (for context detection).

        Returns:
            The combined context string (may be empty).
        """
        decision = self.decide(query, tenant_id)
        context_parts: list[str] = []

        # ── Layer 1: Behavioral (always active) ───────────────────────
        if decision["behavioral"]:
            profile = self._behavioral.load(tenant_id)
            if profile:
                from .reflector import Reflector
                from .classifier import ContextClassifier

                current_context = "general_chat"
                if current_messages:
                    user_msgs = [
                        m["content"]
                        for m in current_messages
                        if m.get("role") == "user" and m.get("content")
                    ]
                    if user_msgs:
                        dynamic_contexts = list(profile.get("contextual_profiles", {}).keys())
                        current_context = ContextClassifier.detect(user_msgs[-1], dynamic_contexts)

                reflector = Reflector()
                behavioral_ctx = reflector.build_injection(profile, current_context)
                if behavioral_ctx:
                    context_parts.append(behavioral_ctx)

        # ── Layer 2: Episodic (conditional) ───────────────────────────
        if decision["episodic"]:
            episodes = self._episodic.recall(tenant_id, query, top_k=2)
            if episodes:
                ep_lines = ["[Past Conversations]"]
                for ep in episodes:
                    summary = ep.get("summary", "").strip()
                    if summary:
                        ep_lines.append(f"- {summary}")
                if len(ep_lines) > 1:
                    context_parts.append("\n".join(ep_lines))
                    logger.debug(
                        f"MemoryOrchestrator: {len(episodes)} episodic memory added "
                        f"({tenant_id})"
                    )

        # ── Layer 3: Knowledge ────────────────────────────────────────

        return "\n\n".join(context_parts)

    # ------------------------------------------------------------------
    # Episodic Storage Decision
    # ------------------------------------------------------------------

    def should_store_episode(
        self,
        messages: list[dict],
        signals: list[dict],
    ) -> float:
        """
        Determines whether this conversation should be stored in episodic
        memory by calculating an importance score.

        High importance criteria:
          - Presence of a correction signal (learning took place).
          - Long conversation (8+ messages implies deep interaction).
          - Multiple corrections (strong preferences formed).

        Args:
            messages: Conversation message list.
            signals:  Detected signal list.

        Returns:
            float: Importance score between 0.0-1.0.
                   Scores 0.7+ are persisted to episodic store.
        """
        importance = 0.3  # baseline

        if len(messages) >= 8:
            importance += 0.2

        correction_count = sum(
            1 for s in signals if s.get("type") in ("correction", "rejection")
        )
        importance += min(correction_count * 0.15, 0.4)

        preference_count = sum(
            1 for s in signals
            if s.get("type") in ("preference", "formatting_preference",
                                  "tone_preference", "detail_preference")
        )
        if preference_count > 0:
            importance += min(preference_count * 0.05, 0.1)

        return min(importance, 1.0)
