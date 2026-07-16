"""
MemoryOrchestrator
------------------
Memory Orchestrator — Inter-layer coordination

Rule-based orchestrator that decides which memory layer should be
activated per request and merges context from selected layers.
"""

from __future__ import annotations

import logging
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
    - Episodic:   Always active — relevance is decided by semantic
                  similarity in EpisodicStore.recall() (local embedding
                  model, or Qdrant when configured), not by whether the
                  query happens to contain a hardcoded phrase. A query
                  that never says "remember when" can still be about a
                  past conversation; a query that does say it isn't
                  guaranteed to be. Gating on phrase-matching either
                  hides real matches or fires on empty ones — semantic
                  similarity against the actual episode content is the
                  only signal that means anything here.
    - Knowledge:  Graph-based long-term memory.

    Args:
        behavioral_store: Instance of ProfileStore.
        episodic_store:   Instance of EpisodicStore.
        knowledge_store:  Instance of KnowledgeStore.
    """

    def __init__(
        self,
        behavioral_store: ProfileStore,
        episodic_store: EpisodicStore,
        knowledge_store: KnowledgeStore,
    ) -> None:
        self._behavioral = behavioral_store
        self._episodic = episodic_store
        self._knowledge = knowledge_store

    # ------------------------------------------------------------------
    # Core Decision Logic
    # ------------------------------------------------------------------

    def decide(self, query: str, tenant_id: str) -> dict[str, bool]:
        """
        Decides which layers to activate. Behavioral and episodic are both
        always active — episodic relevance is filtered by semantic
        similarity inside EpisodicStore.recall() itself, not by whether
        this query happens to match a hardcoded phrase.

        Args:
            query:     The user's latest message.
            tenant_id: Tenant ID in ``project_id:user_id`` format.

        Returns:
            Dict: ``{"behavioral": True, "episodic": True, "knowledge": bool}``
        """
        use_knowledge = False

        decision = {
            "behavioral": True,
            "episodic": True,
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

        # ── Layer 1: Behavioral (always active) ────────────────────────
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

        # ── Layer 2: Episodic (always active — recall() itself filters
        #    for relevance by semantic similarity) ────────────────────
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

        # ── Layer 3: Knowledge (always active if entities exist) ─────────
        knowledge_ctx = self._knowledge.build_knowledge_injection(tenant_id)
        if knowledge_ctx:
            context_parts.append(knowledge_ctx)
            logger.debug(f"MemoryOrchestrator: knowledge layer context added ({tenant_id})")

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
        importance = 0.4  # baseline (raised from 0.3 — capture more conversations)

        if len(messages) >= 6:  # Lowered from 8
            importance += 0.2

        correction_count = sum(
            1 for s in signals if s.get("type") in ("correction", "rejection")
        )
        importance += min(correction_count * 0.15, 0.4)

        preference_count = sum(
            1 for s in signals
            if s.get("type") in ("preference", "formatting_preference",
                                  "tone_preference", "detail_preference",
                                  "preference_like", "preference_dislike", "personality")
        )
        if preference_count > 0:
            importance += min(preference_count * 0.1, 0.2)

        return min(importance, 1.0)
