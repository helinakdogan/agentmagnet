"""
BehavioralMemory
----------------
A behavioral memory client using a 3-Layered Hybrid Memory Architecture.

Layers:
  Layer 1 — Behavioral : Redis + ProfileStore + SignalBuffer + Reflector
             Runs on every request.
  Layer 2 — Episodic   : EpisodicStore (Qdrant or Redis fallback)
             Long-term memory for important conversations.
  Layer 3 — Knowledge  : KnowledgeStore
             Graph-based entity memory.

Supports BYOK (Bring Your Own Key). No third-party memory provider dependencies.
"""

from __future__ import annotations
import asyncio
import logging
import os
import time
from typing import Any

from .signals import SignalDetector
from .buffer import SignalBuffer
from .reflector import Reflector
from .store import ProfileStore
from .classifier import IntelligentClassifier
from .classifier import ContextClassifier
from .router import ModelRouter, RouterDecision
from .aggregate_store import AggregateSignalStore
from .episodic_store import EpisodicStore
from .knowledge_store import KnowledgeStore
from .memory_orchestrator import MemoryOrchestrator
from .consolidation import ConsolidationEngine
from .team_store import TeamStore, TeamMemoryRequiresRedis, REDIS_REQUIRED_MSG

logger = logging.getLogger(__name__)


class BehavioralMemory:
    """
    A memory client that learns from user behavior using a 3-tiered hybrid architecture.

    The Behavioral layer runs on every request. The Episodic layer saves important conversations
    into long-term memory and activates automatically for queries referencing the past.
    The Knowledge layer provides graph-based memory capabilities.

    Args:
        openai_api_key (str, optional): BYOK — OpenAI API key.
            Falls back to OPENAI_API_KEY environment variable if not provided.
        anthropic_api_key (str, optional): BYOK — Anthropic API key.
            Falls back to ANTHROPIC_API_KEY environment variable if not provided.
        redis_client (Any, optional): An initialized Redis client for persistence.
        signal_threshold (int): Number of signals to buffer before triggering reflection.
        reflector_model (str): The LLM to use for the reflection process.
        classifier_model (str): The LLM to use for the classification fallback.
        router (ModelRouter, optional): An instance of ModelRouter for dynamic model selection.
        qdrant_url (str, optional): Qdrant vector DB URL for episodic layer (optional).
        qdrant_api_key (str, optional): Qdrant API key.
        enable_aggregate (bool): Enable aggregate signal tracking.
    """

    def __init__(
        self,
        openai_api_key: str | None = None,
        anthropic_api_key: str | None = None,
        redis_client: Any | None = None,
        signal_threshold: int = 5,
        reflector_model: str = "openai/gpt-4o-mini",
        classifier_model: str = "openai/gpt-4o-mini",
        inject_profile: bool = True,
        router: ModelRouter | None = None,
        qdrant_url: str | None = None,
        qdrant_api_key: str | None = None,
        enable_aggregate: bool = True,
        profile_id: str | None = None,
        # Legacy parameters — silently ignored for backwards compatibility
        api_key: str | None = None,
        use_mem0: bool = False,
        openai_client: Any = None,
        anthropic_client: Any = None,
        **kwargs,
    ):
        self._inject_profile = inject_profile
        self.router = router
        self._profile_cache: dict[str, tuple[dict | None, float]] = {}
        self._profile_cache_ttl = 60.0
        self._background_tasks: set[asyncio.Task] = set()

        # BYOK — read from parameter or fall back to environment variable
        self._byok_openai_key = openai_api_key or os.environ.get("OPENAI_API_KEY")
        self._byok_anthropic_key = anthropic_api_key or os.environ.get("ANTHROPIC_API_KEY")

        # Cross-tool identity: MAGNET_USER_ID lets Claude Code, Cursor, Codex, etc.
        # share the same user profile without the caller passing user_id explicitly.
        self._default_user_id = os.environ.get("MAGNET_USER_ID")

        # Profile/persona: MAGNET_PROFILE lets users maintain separate memory contexts
        # (e.g. "company" vs "hobby"). Defaults to "default" for full backward compat.
        self._default_profile_id = profile_id or os.environ.get("MAGNET_PROFILE", "default")
        if self._default_profile_id != "default":
            logger.info(f"[magnet] active profile: {self._default_profile_id}")

        self._redis = redis_client

        if openai_client is not None:
            logger.warning("BehavioralMemory: openai_client deprecated, ignoring.")
        if anthropic_client is not None:
            logger.warning("BehavioralMemory: anthropic_client deprecated, ignoring.")
        if kwargs:
            logger.warning(f"BehavioralMemory: unknown parameters ignored: {list(kwargs.keys())}")

        # ── Layer 1: Behavioral ───────────────────────────────────────
        self._detector = SignalDetector(param_change_threshold=3, redis_client=redis_client)
        self._buffer = SignalBuffer(redis_client=redis_client, threshold=signal_threshold)
        self._reflector = Reflector(
            model=reflector_model,
            openai_api_key=self._byok_openai_key,
            anthropic_api_key=self._byok_anthropic_key,
        )
        self.classifier = IntelligentClassifier(
            model=classifier_model,
            llm_client=None,
            fallback_rules=True,
        )
        self._store = ProfileStore(redis_client=redis_client)

        if enable_aggregate and redis_client:
            self._aggregate = AggregateSignalStore(redis_client)
        else:
            self._aggregate = None

        # ── Layer 2: Episodic ─────────────────────────────────────────
        _qdrant_url = qdrant_url or os.environ.get("QDRANT_URL")
        _qdrant_api_key = qdrant_api_key or os.environ.get("QDRANT_API_KEY")
        self._episodic = EpisodicStore(
            redis_client=redis_client,
            qdrant_url=_qdrant_url,
            qdrant_api_key=_qdrant_api_key,
            openai_api_key=self._byok_openai_key,
        )

        # ── Layer 3: Knowledge ────────────────────────────────────────
        _neo4j_url = os.environ.get("NEO4J_URL")
        _neo4j_auth_str = os.environ.get("NEO4J_AUTH")
        _neo4j_auth = None
        if _neo4j_auth_str:
            # Robust parser: regex-first (handles tuple format), slash fallback
            import re as _re
            _m = _re.findall(r'["\'](.*?)["\']', _neo4j_auth_str)
            if len(_m) >= 2:
                _neo4j_auth = (_m[0], _m[1])
            else:
                _c = _neo4j_auth_str.strip(" ()\"'")
                if "/" in _c:
                    _u, _p = _c.split("/", 1)
                    _neo4j_auth = (_u.strip(), _p.strip())
                elif _c:
                    _neo4j_auth = (_c, "")

        self._knowledge = KnowledgeStore(
            neo4j_url=_neo4j_url,
            neo4j_auth=_neo4j_auth,
            redis_client=redis_client,
        )

        # Threshold values for dynamic signal learning
        # Strong semantic signals (like/dislike) are instant — no buffering needed
        self._instant_signal_types = {
            "preference_dislike", "preference_like", "tone_preference",
        }
        # Soft signals still buffer until threshold
        self._soft_signal_threshold = signal_threshold
        # Minimum classifier confidence to accept a signal; below this is noise
        self._min_signal_confidence = 0.30

        # ── Memory Orchestrator ───────────────────────────────────────
        self._orchestrator = MemoryOrchestrator(
            behavioral_store=self._store,
            episodic_store=self._episodic,
            knowledge_store=self._knowledge,
        )

        # ── Consolidation ─────────────────────────────────────────────
        self._consolidation = ConsolidationEngine(
            episodic_store=self._episodic,
            profile_store=self._store,
            aggregate_store=self._aggregate,
        )

        # ── Team memory (Redis required for shared scopes) ────────────
        self._team_store = TeamStore(redis_client=redis_client)

    def _resolve_user_id(self, user_id: str | None) -> str:
        resolved = user_id or self._default_user_id
        if not resolved:
            raise ValueError("user_id is required (or set MAGNET_USER_ID env variable)")
        return resolved

    def _resolve_profile_id(self, profile_id: str | None) -> str:
        return profile_id or self._default_profile_id

    @staticmethod
    def _build_tenant_id(project_id: str, user_id: str, profile_id: str) -> str:
        """
        Build the storage key for a user+project+profile combination.

        "default" profile uses the legacy format for full backward compatibility.
        Non-default profiles use an explicit :profile_id:user: segment so they
        never collide with the default namespace.
        """
        if profile_id == "default":
            return f"{project_id}:{user_id}"
        return f"{project_id}:{profile_id}:user:{user_id}"

    @staticmethod
    def _parse_tenant_id(tenant_id: str) -> tuple[str, str, str]:
        """Reverse of _build_tenant_id → (project_id, profile_id, user_id)."""
        if ":user:" in tenant_id:
            proj_profile, user_id = tenant_id.split(":user:", 1)
            project_id, profile_id = proj_profile.rsplit(":", 1)
            return project_id, profile_id, user_id
        project_id, user_id = tenant_id.split(":", 1)
        return project_id, "default", user_id

    def _register_profile_save(
        self, user_id: str, profile_id: str, project_id: str, pref_count: int
    ) -> None:
        try:
            self._store.register_profile(user_id, profile_id, project_id, pref_count)
        except Exception as e:
            logger.debug(f"_register_profile_save: {e}")

    def add(
        self,
        messages: list[dict],
        user_id: str | None = None,
        project_id: str = "default",
        profile_id: str | None = None,
        session_id: str | None = None,
        metadata: dict | None = None,
        **kwargs,
    ) -> dict:
        """
        Adds a conversation to behavioral memory and processes it for signals.

        Args:
            messages (list[dict]): The list of messages in the conversation.
            project_id (str): The ID of the project.
            user_id (str): The ID of the user.
            profile_id (str, optional): Persona/profile context (default: MAGNET_PROFILE env var).
            session_id (str, optional): The ID of the current session.
            metadata (dict, optional): Additional metadata about the interaction.

        Returns:
            dict: Result containing routing information if a router is configured.
        """
        user_id = self._resolve_user_id(user_id)
        profile_id = self._resolve_profile_id(profile_id)
        tenant_id = self._build_tenant_id(project_id, user_id, profile_id)
        result = {}

        try:
            sid = session_id or tenant_id
            signals = []

            if self.classifier:
                user_msgs = [m for m in messages if m.get("role") == "user"]
                if user_msgs:
                    last_msg = user_msgs[-1].get("content", "")
                    context_msgs = messages[:-1] if len(messages) > 1 else []
                    cls_res = self.classifier.classify(context_msgs, last_msg)
                    if cls_res.signal_type not in ("neutral",) and cls_res.confidence >= self._min_signal_confidence:
                        signal_entry = {
                            "type": cls_res.signal_type,
                            "message": last_msg[:200],
                            "confidence": cls_res.confidence,
                            "dimension": cls_res.dimension,
                            "extracted_preference": cls_res.extracted_preference,
                        }
                        signals.append(signal_entry)

                        # ── INSTANT LEARNING: strong signals bypass buffer ──
                        if cls_res.signal_type in self._instant_signal_types and cls_res.extracted_preference:
                            existing_profile = self._store.load(tenant_id)
                            updated_profile = self._reflector.instant_learn(
                                user_id=tenant_id,
                                signal_type=cls_res.signal_type,
                                extracted_preference=cls_res.extracted_preference,
                                confidence=cls_res.confidence,
                                existing_profile=existing_profile,
                            )
                            self._store.save(tenant_id, updated_profile)
                            self._profile_cache.pop(tenant_id, None)
                            self._register_profile_save(
                                user_id, profile_id, project_id,
                                len(updated_profile.get("preferences", [])),
                            )

                            # Also store in Knowledge Layer (Layer 3)
                            entity_type = (
                                "dislike" if cls_res.signal_type == "preference_dislike"
                                else "like" if cls_res.signal_type == "preference_like"
                                else "personality"
                            )
                            self._knowledge.store_entity(tenant_id, {
                                "type": entity_type,
                                "content": cls_res.extracted_preference,
                                "dimension": cls_res.dimension,
                                "confidence": cls_res.confidence,
                            })
                            logger.info(
                                f"add(): instant_learn [{cls_res.signal_type}] for {tenant_id}: "
                                f"{cls_res.extracted_preference!r} (conf={cls_res.confidence:.2f})"
                            )

                if metadata:
                    param_sig = self._detector._check_param_change(sid, metadata)
                    if param_sig:
                        signals.append(param_sig)
            else:
                signals = self._detector.detect(
                    messages=messages, session_id=sid, metadata=metadata
                )

            if signals:
                if self._aggregate:
                    user_msgs = [m for m in messages if m.get("role") == "user"]
                    last_user_msg = user_msgs[-1].get("content", "") if user_msgs else ""
                    for signal in signals:
                        if signal.get("type") in (
                            "correction", "rejection", "preference",
                            "clarification", "positive",
                            "preference_dislike", "preference_like",
                        ):
                            category = self._classify_category_local(last_user_msg)
                            self._aggregate.record(
                                signal_type=signal["type"],
                                query_category=category,
                                dimension=signal.get("dimension", "unknown"),
                                dimension_value=self._extract_dimension_value(signal),
                            )
                count = self._buffer.push(tenant_id, signals)
                logger.debug(f"add(): {len(signals)} signals added, total={count}")
                if self._buffer.should_reflect(tenant_id):
                    try:
                        loop = asyncio.get_running_loop()
                        loop.create_task(self._reflect_async(tenant_id))
                    except RuntimeError:
                        signals_to_reflect = self._buffer.flush(tenant_id)
                        if signals_to_reflect:
                            import threading

                            t = threading.Thread(
                                target=lambda: asyncio.run(
                                    self._do_reflect(tenant_id, signals_to_reflect)
                                ),
                                daemon=True,
                            )
                            t.start()

            # ── Episodic storage decision ─────────────────────────────────
            importance = self._orchestrator.should_store_episode(messages, signals)
            if importance >= 0.5:  # Lowered from 0.7 — more conversations captured
                self._episodic.store_episode(
                    tenant_id=tenant_id,
                    messages=messages,
                    importance=importance,
                )

            if self.router:
                profile = self.get_profile(user_id, project_id, profile_id)
                routing_decision = self.router.route(messages, profile)
                result["routing"] = {
                    "selected_model": routing_decision.selected_model,
                    "reason": routing_decision.reason,
                    "confidence": routing_decision.confidence,
                    "cost_tier": routing_decision.cost_tier,
                }
        except Exception as e:
            logger.error(f"Behavioral add error: {e}", exc_info=True)

        return result

    async def async_add(
        self,
        messages: list[dict],
        user_id: str | None = None,
        project_id: str = "default",
        profile_id: str | None = None,
        session_id: str | None = None,
        metadata: dict | None = None,
        **kwargs,
    ) -> dict:
        """Asynchronous version of the `add` method."""
        user_id = self._resolve_user_id(user_id)
        profile_id = self._resolve_profile_id(profile_id)
        tenant_id = self._build_tenant_id(project_id, user_id, profile_id)
        result = {}

        try:
            sid = session_id or tenant_id
            signals = []

            if self.classifier:
                user_msgs = [m for m in messages if m.get("role") == "user"]
                if user_msgs:
                    last_msg = user_msgs[-1].get("content", "")
                    context_msgs = messages[:-1] if len(messages) > 1 else []
                    cls_res = await asyncio.to_thread(
                        self.classifier.classify, context_msgs, last_msg
                    )
                    if cls_res.signal_type not in ("neutral",) and cls_res.confidence >= self._min_signal_confidence:
                        signal_entry = {
                            "type": cls_res.signal_type,
                            "message": last_msg[:200],
                            "confidence": cls_res.confidence,
                            "dimension": cls_res.dimension,
                            "extracted_preference": cls_res.extracted_preference,
                        }
                        signals.append(signal_entry)

                        # ── INSTANT LEARNING: strong signals bypass buffer ──
                        if cls_res.signal_type in self._instant_signal_types and cls_res.extracted_preference:
                            existing_profile = await asyncio.to_thread(self._store.load, tenant_id)
                            updated_profile = await asyncio.to_thread(
                                self._reflector.instant_learn,
                                tenant_id,
                                cls_res.signal_type,
                                cls_res.extracted_preference,
                                cls_res.confidence,
                                existing_profile,
                            )
                            await asyncio.to_thread(self._store.save, tenant_id, updated_profile)
                            self._profile_cache.pop(tenant_id, None)
                            self._register_profile_save(
                                user_id, profile_id, project_id,
                                len(updated_profile.get("preferences", [])),
                            )

                            # Also store in Knowledge Layer (Layer 3)
                            entity_type = (
                                "dislike" if cls_res.signal_type == "preference_dislike"
                                else "like" if cls_res.signal_type == "preference_like"
                                else "personality"
                            )
                            await asyncio.to_thread(
                                self._knowledge.store_entity,
                                tenant_id,
                                {
                                    "type": entity_type,
                                    "content": cls_res.extracted_preference,
                                    "dimension": cls_res.dimension,
                                    "confidence": cls_res.confidence,
                                },
                            )
                            logger.info(
                                f"async_add(): instant_learn [{cls_res.signal_type}] for {tenant_id}: "
                                f"{cls_res.extracted_preference!r} (conf={cls_res.confidence:.2f})"
                            )

                if metadata:
                    param_sig = self._detector._check_param_change(sid, metadata)
                    if param_sig:
                        signals.append(param_sig)
            else:
                signals = self._detector.detect(
                    messages=messages, session_id=sid, metadata=metadata
                )

            if signals:
                if self._aggregate:
                    user_msgs = [m for m in messages if m.get("role") == "user"]
                    last_user_msg = user_msgs[-1].get("content", "") if user_msgs else ""
                    for signal in signals:
                        if signal.get("type") in (
                            "correction", "rejection", "preference",
                            "clarification", "positive",
                            "preference_dislike", "preference_like",
                        ):
                            category = self._classify_category_local(last_user_msg)
                            self._aggregate.record(
                                signal_type=signal["type"],
                                query_category=category,
                                dimension=signal.get("dimension", "unknown"),
                                dimension_value=self._extract_dimension_value(signal),
                            )
                count = self._buffer.push(tenant_id, signals)
                logger.debug(f"async_add(): {len(signals)} signals added, total={count}")
                if self._buffer.should_reflect(tenant_id):
                    await self._reflect_async(tenant_id)

            # ── Episodic storage decision ─────────────────────────────────
            importance = self._orchestrator.should_store_episode(messages, signals)
            if importance >= 0.5:  # Lowered from 0.7
                await asyncio.to_thread(
                    self._episodic.store_episode,
                    tenant_id,
                    messages,
                    None,
                    importance,
                )

            if self.router:
                profile = self.get_profile(user_id, project_id, profile_id)
                routing_decision = self.router.route(messages, profile)
                result["routing"] = {
                    "selected_model": routing_decision.selected_model,
                    "reason": routing_decision.reason,
                    "confidence": routing_decision.confidence,
                    "cost_tier": routing_decision.cost_tier,
                }
        except Exception as e:
            logger.error(f"Behavioral async_add error: {e}", exc_info=True)

        return result

    def search(
        self,
        query: str,
        user_id: str | None = None,
        project_id: str = "default",
        profile_id: str | None = None,
        limit: int = 10,
        **kwargs,
    ) -> dict:
        """
        Returns behavioral context for a user based on their learned profile.

        Args:
            query (str): The search query (used for context-aware injection).
            user_id (str): The ID of the user.
            project_id (str): The ID of the project.
            profile_id (str, optional): Persona/profile context.
            limit (int): Unused — kept for API compatibility.

        Returns:
            dict: Contains 'behavioral_context' (injection string) and 'behavioral_profile'.
        """
        user_id = self._resolve_user_id(user_id)
        profile_id = self._resolve_profile_id(profile_id)
        tenant_id = self._build_tenant_id(project_id, user_id, profile_id)
        profile = self._store.load(tenant_id)
        if not profile:
            return {}
        injection = self._reflector.build_injection(profile) if self._inject_profile else ""
        return {"behavioral_context": injection, "behavioral_profile": profile}

    def get_all(
        self,
        user_id: str | None = None,
        project_id: str = "default",
        profile_id: str | None = None,
        **kwargs,
    ) -> dict:
        """Retrieves the full behavioral profile for a user."""
        user_id = self._resolve_user_id(user_id)
        profile_id = self._resolve_profile_id(profile_id)
        tenant_id = self._build_tenant_id(project_id, user_id, profile_id)
        profile = self._store.load(tenant_id)
        return {"behavioral_profile": profile} if profile else {}

    def delete_all(
        self,
        user_id: str | None = None,
        project_id: str = "default",
        profile_id: str | None = None,
        **kwargs,
    ) -> dict:
        """Deletes all behavioral memory for a user."""
        user_id = self._resolve_user_id(user_id)
        profile_id = self._resolve_profile_id(profile_id)
        tenant_id = self._build_tenant_id(project_id, user_id, profile_id)
        self._store.delete(tenant_id)
        self._detector.clear_session(tenant_id)
        self._profile_cache.pop(tenant_id, None)
        return {"deleted": True}

    def get_profile(
        self,
        user_id: str | None = None,
        project_id: str = "default",
        profile_id: str | None = None,
    ) -> dict | None:
        """Retrieves a user's behavioral profile, using a time-based cache."""
        user_id = self._resolve_user_id(user_id)
        profile_id = self._resolve_profile_id(profile_id)
        tenant_id = self._build_tenant_id(project_id, user_id, profile_id)
        now = time.time()
        cached = self._profile_cache.get(tenant_id)
        if cached:
            profile, ts = cached
            if now - ts < self._profile_cache_ttl:
                return profile
        profile = self._store.load(tenant_id)
        self._profile_cache[tenant_id] = (profile, now)
        return profile

    def get_injection(
        self,
        user_id: str | None = None,
        project_id: str = "default",
        current_messages: list[dict] | None = None,
        profile_id: str | None = None,
    ) -> str:
        """
        Generates a system prompt injection based on the user's profile and current conversation.

        Operates via the Orchestrator:
          - Behavioral context is always included.
          - Episodic context is added if the query references past interactions.
          - Knowledge context is currently bypassed.

        Args:
            user_id:          User ID.
            project_id:       Project ID (default: "default").
            current_messages: Current conversation messages.
            profile_id:       Persona/profile context (default: MAGNET_PROFILE env var).

        Returns:
            Combined context string. Returns aggregate cold-start if no profile exists.
        """
        user_id = self._resolve_user_id(user_id)
        profile_id = self._resolve_profile_id(profile_id)
        tenant_id = self._build_tenant_id(project_id, user_id, profile_id)

        # Use the last user message as the query
        query = ""
        if current_messages:
            user_msgs = [
                m["content"]
                for m in current_messages
                if m.get("role") == "user" and m.get("content")
            ]
            query = user_msgs[-1] if user_msgs else ""

        # Cold-start fallback if no profile exists
        profile = self._store.load(tenant_id)
        if not profile:
            if self._aggregate:
                current_context = self._classify_category_local(query) if query else "general_chat"
                return self._aggregate.get_cold_start_injection(current_context)
            return ""

        return self._orchestrator.build_context(
            query=query,
            tenant_id=tenant_id,
            current_messages=current_messages,
        )

    def session_end(
        self,
        user_id: str | None = None,
        project_id: str = "default",
        messages: list[dict] | None = None,
        max_messages: int = 20,
        profile_id: str | None = None,
    ) -> dict:
        """
        Summarizes the session and saves it as a high-importance episode.

        Calls the LLM to extract key decisions, preferences, and context from
        the last `max_messages` turns, then stores the result in EpisodicStore
        (importance=0.9) and in Redis under a 30-day TTL key for quick recall.

        Returns:
            {"status": "ok", "summary": str}
        """
        import litellm  # already a project dependency

        user_id = self._resolve_user_id(user_id)
        profile_id = self._resolve_profile_id(profile_id)
        tenant_id = self._build_tenant_id(project_id, user_id, profile_id)
        tail = (messages or [])[-max_messages:]

        if not tail:
            return {"status": "ok", "summary": ""}

        convo = "\n".join(
            f"{m['role'].upper()}: {m.get('content', '')}"
            for m in tail
            if m.get("content")
        )
        prompt = (
            "Extract key decisions, learned preferences, and important context "
            "from this conversation. Be specific. Format as bullet points.\n\n"
            + convo
        )

        try:
            kwargs: dict = {
                "model": getattr(self._reflector, "model", "openai/gpt-4o-mini"),
                "messages": [{"role": "user", "content": prompt}],
            }
            if self._byok_openai_key:
                kwargs["api_key"] = self._byok_openai_key
            response = litellm.completion(**kwargs)
            summary = response.choices[0].message.content or ""
        except Exception as e:
            logger.error(f"session_end(): LLM summarization failed: {e}")
            summary = self._episodic._auto_summarize(tail)

        # Layer 2 — high-importance episode
        self._episodic.store_episode(
            tenant_id=tenant_id,
            messages=tail,
            summary=summary,
            importance=0.9,
        )

        # Redis fast-lookup with 30-day TTL
        if self._redis:
            self._redis.set(
                f"magnet:session_summary:{tenant_id}",
                summary,
                ex=60 * 60 * 24 * 30,
            )

        logger.info(f"session_end(): session summarized for {tenant_id}")
        return {"status": "ok", "summary": summary}

    def force_reflect(
        self,
        user_id: str | None = None,
        project_id: str = "default",
        profile_id: str | None = None,
    ) -> dict:
        """
        Triggers the reflection process immediately, bypassing the signal threshold.
        Useful for debugging and testing purposes.
        """
        user_id = self._resolve_user_id(user_id)
        profile_id = self._resolve_profile_id(profile_id)
        tenant_id = self._build_tenant_id(project_id, user_id, profile_id)
        signals = self._buffer.flush(tenant_id)
        if not signals:
            return {}
        existing_profile = self.get_profile(user_id, project_id, profile_id)
        profile = self._reflector.reflect(tenant_id, signals, existing_profile)
        self._store.save(tenant_id, profile)
        self._profile_cache.pop(tenant_id, None)
        self._register_profile_save(user_id, profile_id, project_id, len(profile.get("preferences", [])))
        return profile

    async def start_consolidation(
        self,
        tenant_prefix: str,
        interval_hours: int = 24,
    ) -> None:
        """Start the scheduled consolidation process."""
        await self._consolidation.schedule(tenant_prefix, interval_hours)

    def get_pending_signals(
        self,
        user_id: str | None = None,
        project_id: str = "default",
        profile_id: str | None = None,
    ) -> list[dict]:
        """Returns the list of signals currently in the buffer for a user."""
        user_id = self._resolve_user_id(user_id)
        profile_id = self._resolve_profile_id(profile_id)
        tenant_id = self._build_tenant_id(project_id, user_id, profile_id)
        return self._buffer.peek(tenant_id)

    def get_recommended_model(
        self,
        user_id: str | None = None,
        messages: list[dict] = [],
        project_id: str = "default",
        profile_id: str | None = None,
    ) -> RouterDecision | None:
        """Recommends the optimal model for a given request using the router."""
        if not self.router:
            return None
        user_id = self._resolve_user_id(user_id)
        profile = self.get_profile(user_id, project_id, profile_id)
        return self.router.route(messages, profile)

    async def _reflect_async(self, tenant_id: str) -> None:
        """Schedules the reflection process to run as a background asyncio task."""
        signals = self._buffer.flush(tenant_id)
        if not signals:
            return
        task = asyncio.create_task(self._do_reflect(tenant_id, signals))
        self._background_tasks.add(task)
        task.add_done_callback(self._background_tasks.discard)

    async def _do_reflect(self, tenant_id: str, signals: list[dict]) -> None:
        try:
            existing_profile = await asyncio.to_thread(self._store.load, tenant_id)
            old_pref_keys = {
                (p.get("subject", ""), p.get("relation", ""))
                for p in (existing_profile or {}).get("preferences", [])
                if isinstance(p, dict)
            }

            reflected_profile = await asyncio.to_thread(
                self._reflector.reflect, tenant_id, signals, existing_profile
            )
            await asyncio.to_thread(self._store.save, tenant_id, reflected_profile)
            self._profile_cache.pop(tenant_id, None)

            # Register the profile save in the index
            project_id, profile_id, user_id = self._parse_tenant_id(tenant_id)
            self._register_profile_save(
                user_id, profile_id, project_id,
                len(reflected_profile.get("preferences", [])),
            )

            # Sync newly learned preferences to the knowledge layer
            for pref in reflected_profile.get("preferences", []):
                if not isinstance(pref, dict):
                    continue
                if (pref.get("subject", ""), pref.get("relation", "")) in old_pref_keys:
                    continue
                relation = pref.get("relation", "")
                entity_type = (
                    "dislike" if relation == "dislikes"
                    else "like" if relation == "prefers"
                    else "personality"
                )
                self._knowledge.store_entity(tenant_id, {
                    "type": entity_type,
                    "content": pref.get("natural_text", pref.get("subject", "")),
                    "dimension": pref.get("subject_type", "general"),
                    "confidence": pref.get("confidence", 0.6),
                })

            logger.info(f"Reflect completed: {tenant_id}")
        except Exception as e:
            logger.error(
                f"Reflect error ({tenant_id}): {e}. {len(signals)} signals may be lost.",
                exc_info=True,
            )

    # ── Team memory public API ────────────────────────────────────────

    def get_injection_with_team(
        self,
        user_id: str | None = None,
        project_id: str = "default",
        team_id: str | None = None,
        org_id: str | None = None,
        current_messages: list[dict] | None = None,
        profile_id: str | None = None,
    ) -> str:
        """
        Build a merged memory injection: user preferences take precedence
        over team, team over org.

        Falls back to personal-only injection if team/org Redis is unavailable.
        """
        user_id = self._resolve_user_id(user_id)
        profile_id = self._resolve_profile_id(profile_id)
        tenant_id = self._build_tenant_id(project_id, user_id, profile_id)
        user_profile = self._store.load(tenant_id)

        if not team_id and not org_id:
            return self.get_injection(user_id, project_id, current_messages, profile_id)

        try:
            return self._team_store.build_merged_injection(
                user_profile=user_profile,
                team_id=team_id,
                org_id=org_id,
                project_id=project_id,
                reflector=self._reflector,
            )
        except TeamMemoryRequiresRedis as e:
            logger.warning(f"get_injection_with_team: {e}")
            return self.get_injection(user_id, project_id, current_messages)

    def get_project_memory(
        self,
        project_id: str,
        team_id: str | None = None,
    ) -> dict:
        """
        Return a per-user breakdown of preferences learned within project_id.
        Requires Redis.
        """
        try:
            return self._team_store.get_project_memory(project_id, team_id)
        except TeamMemoryRequiresRedis:
            return {
                "error": REDIS_REQUIRED_MSG,
                "project_id": project_id,
                "contributors": {},
            }

    def share_to_team(
        self,
        user_id: str | None = None,
        project_id: str = "default",
        fact_or_subject: str = "",
        team_id: str = "",
        profile_id: str | None = None,
    ) -> dict:
        """
        Explicitly copy one preference from the user's personal profile to team memory.
        """
        user_id = self._resolve_user_id(user_id)
        profile_id = self._resolve_profile_id(profile_id)
        tenant_id = self._build_tenant_id(project_id, user_id, profile_id)
        personal_profile = self._store.load(tenant_id)
        try:
            return self._team_store.share_to_team(
                user_id=user_id,
                project_id=project_id,
                fact_or_subject=fact_or_subject,
                team_id=team_id,
                personal_profile=personal_profile,
            )
        except TeamMemoryRequiresRedis:
            return {"error": REDIS_REQUIRED_MSG}

    def promote_to_team(
        self,
        team_id: str,
        project_id: str = "default",
        min_users: int = 2,
    ) -> dict:
        """
        Scan all personal profiles in project_id, find preferences shared by
        min_users+ members, and write them to the team's shared profile.
        """
        try:
            return self._team_store.promote_to_team(team_id, project_id, min_users)
        except TeamMemoryRequiresRedis:
            return {"error": REDIS_REQUIRED_MSG}

    # ── Profile / persona management ──────────────────────────────────────

    def list_profiles(self, user_id: str | None = None) -> list[dict]:
        """
        Return all profiles created for a user, with a summary of each.

        Requires Redis. Returns an empty list in local/in-memory mode.

        Each entry: {"profile_id", "projects", "preference_count"}
        """
        user_id = self._resolve_user_id(user_id)
        return self._store.list_user_profiles(user_id)

    def get_profile_summary(
        self,
        user_id: str | None = None,
        profile_id: str | None = None,
        project_id: str = "default",
    ) -> dict:
        """
        Return a summary of a specific profile — useful for switch_profile MCP tool.

        Returns: {"profile_id", "project_id", "preference_count", "preferences_preview"}
        """
        user_id = self._resolve_user_id(user_id)
        profile_id = self._resolve_profile_id(profile_id)
        profile = self.get_profile(user_id, project_id, profile_id)
        if not profile:
            return {
                "profile_id": profile_id,
                "project_id": project_id,
                "preference_count": 0,
                "preferences_preview": [],
                "exists": False,
            }
        prefs = profile.get("preferences", [])
        preview = [
            p.get("natural_text", p.get("subject", ""))
            for p in prefs[:5]
            if isinstance(p, dict)
        ]
        return {
            "profile_id": profile_id,
            "project_id": project_id,
            "preference_count": len(prefs),
            "preferences_preview": preview,
            "exists": True,
        }

    def _classify_category_local(self, text: str) -> str:
        """
        Simple keyword matching — no LLM involved, zero cost.
        Privacy: The actual text content is never sent to the aggregate store, only the category.
        """
        text_lower = text.lower()
        if any(kw in text_lower for kw in ["kod", "code", "python", "javascript", "sql", "bug", "hata"]):
            return "coding"
        if any(kw in text_lower for kw in ["analiz", "rapor", "veri", "analysis", "data"]):
            return "analysis"
        if any(kw in text_lower for kw in ["yaz", "makale", "blog", "write", "essay"]):
            return "writing"
        if any(kw in text_lower for kw in ["öğren", "anlat", "explain", "nedir", "what is"]):
            return "learning"
        return "general_chat"

    def _extract_dimension_value(self, signal: dict) -> str:
        """Extracts the dimension value based on the signal type."""
        if signal.get("type") == "correction":
            msg = signal.get("message", "").lower()
            if any(kw in msg for kw in ["kısa", "short", "özet", "brief"]):
                return "short"
            if any(kw in msg for kw in ["uzun", "long", "detaylı", "detailed"]):
                return "long"
        return "unknown"
