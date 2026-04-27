"""
BehavioralMemory
----------------
Mem0 wrapper + behavioral memory layer.

This is the main client-facing class. It provides a drop-in replacement for
mem0.MemoryClient, augmenting it with behavioral learning capabilities such
as signal detection, reflection, and intelligent routing.
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

logger = logging.getLogger(__name__)


class BehavioralMemory:
    """
    A memory client that learns from user behavior.

    This class wraps a standard memory client (like Mem0) and adds a
    behavioral layer that observes user interactions, extracts signals,
    and builds a dynamic user profile to personalize future responses.

    Args:
        api_key (str, optional): API key for the underlying memory provider (e.g., Mem0).
        redis_client (Any, optional): An initialized Redis client for persistence.
        signal_threshold (int): Number of signals to buffer before triggering reflection.
        reflector_model (str): The LLM to use for the reflection process.
        classifier_model (str): The LLM to use for the classification fallback.
        router (ModelRouter, optional): An instance of ModelRouter for dynamic model selection.
    """
    def __init__(
        self,
        api_key: str | None = None,
        redis_client: Any | None = None,
        signal_threshold: int = 5,
        reflector_model: str = "openai/gpt-4o-mini",
        classifier_model: str = "openai/gpt-4o-mini",
        inject_profile: bool = True,
        use_mem0: bool = True,
        router: ModelRouter | None = None,
        openai_client: Any = None,
        anthropic_client: Any = None,
        **kwargs,
    ):
        self._inject_profile = inject_profile
        self.router = router
        self._profile_cache: dict[str, tuple[dict | None, float]] = {}
        self._profile_cache_ttl = 60.0
        self._background_tasks: set[asyncio.Task] = set()

        if openai_client is not None:
            logger.warning("BehavioralMemory: openai_client deprecated, ignoring.")
        if anthropic_client is not None:
            logger.warning("BehavioralMemory: anthropic_client deprecated, ignoring.")
        if kwargs:
            logger.warning(f"BehavioralMemory: unknown parameters ignored: {list(kwargs.keys())}")

        self._mem0 = None
        if use_mem0:
            try:
                from mem0 import MemoryClient

                key = api_key or os.environ.get("MEM0_API_KEY", "")
                if key:
                    self._mem0 = MemoryClient(api_key=key)
                else:
                    logger.warning("MEM0_API_KEY not found. Mem0 is disabled.")
            except ImportError:
                logger.warning("mem0ai package not installed: pip install mem0ai")

        self._detector = SignalDetector(param_change_threshold=3, redis_client=redis_client)
        self._buffer = SignalBuffer(redis_client=redis_client, threshold=signal_threshold)
        self._reflector = Reflector(
            model=reflector_model,
            openai_client=openai_client,
            anthropic_client=anthropic_client,
        )
        self.classifier = IntelligentClassifier(
            model=classifier_model,
            llm_client=None,
            fallback_rules=True,
        )
        self._store = ProfileStore(redis_client=redis_client)

    def add(
        self,
        messages: list[dict],
        project_id: str,
        user_id: str,
        session_id: str | None = None,
        metadata: dict | None = None,
        **kwargs,
    ) -> dict:
        """
        Adds a conversation to memory and processes it for behavioral signals.

        This method first calls the underlying memory provider's `add` method,
        then asynchronously processes the messages to detect signals, update
        the signal buffer, and trigger reflection if the threshold is met.

        Args:
            messages (list[dict]): The list of messages in the conversation.
                project_id (str): The ID of the project.
            user_id (str): The ID of the user.
            session_id (str, optional): The ID of the current session.
            metadata (dict, optional): Additional metadata about the interaction.
        """
        tenant_id = f"{project_id}:{user_id}"
        result = {}
        if self._mem0:
            result = self._mem0.add(messages=messages, user_id=tenant_id, metadata=metadata or {}, **kwargs)

        try:
            sid = session_id or tenant_id
            signals = []
            classifier_complexity = "medium"

            if self.classifier:
                user_msgs = [m for m in messages if m.get("role") == "user"]
                if user_msgs:
                    last_msg = user_msgs[-1].get("content", "")
                    context_msgs = messages[:-1] if len(messages) > 1 else []
                    cls_res = self.classifier.classify(context_msgs, last_msg)
                    if cls_res.signal_type in ("correction", "rejection", "preference", "formatting_preference", "tone_preference", "detail_preference"):
                     if cls_res.signal_type in ("correction", "rejection", "preference", "formatting_preference", "tone_preference", "detail_preference"):
                        signals.append(
                            {
                                "type": cls_res.signal_type,
                                "message": last_msg[:200],
                                "confidence": cls_res.confidence,
                                "dimension": cls_res.dimension,
                            }
                        )
                if metadata:
                    param_sig = self._detector._check_param_change(sid, metadata)
                    if param_sig:
                        signals.append(param_sig)
            else:
                signals = self._detector.detect(messages=messages, session_id=sid, metadata=metadata)

            if signals:
                count = self._buffer.push(tenant_id, signals)
                logger.debug(f"add(): {len(signals)} signals added, total={count}")
                if self._buffer.should_reflect(tenant_id):
                    # Trigger reflection in the background if threshold is met.
                    try:
                        loop = asyncio.get_running_loop()
                        loop.create_task(self._reflect_async(tenant_id))
                    except RuntimeError:
                        signals_to_reflect = self._buffer.flush(tenant_id) # Fallback for sync contexts
                        if signals_to_reflect:
                            asyncio.run(self._do_reflect(tenant_id, signals_to_reflect))

            if self.router:
                profile = self.get_profile(project_id, user_id)
                routing_decision = self.router.route(messages, profile)
                result["routing"] = {
                    "selected_model": routing_decision.selected_model,
                    "reason": routing_decision.reason,
                    "confidence": routing_decision.confidence,
                    "cost_tier": routing_decision.cost_tier
                }
        except Exception as e:
            logger.error(f"Behavioral add error: {e}")

        return result

    async def async_add(
        self,
        messages: list[dict],
        project_id: str,
        user_id: str,
        session_id: str | None = None,
        metadata: dict | None = None,
        **kwargs,
    ) -> dict:
        """Asynchronous version of the `add` method."""
        tenant_id = f"{project_id}:{user_id}"
        result = {}
        if self._mem0:
            result = await asyncio.to_thread(
                self._mem0.add,
                messages=messages,
                user_id=tenant_id,
                metadata=metadata or {},
                **kwargs,
            )

        try:
            sid = session_id or tenant_id
            signals = []
            classifier_complexity = "medium"

            if self.classifier:
                user_msgs = [m for m in messages if m.get("role") == "user"]
                if user_msgs:
                    last_msg = user_msgs[-1].get("content", "")
                    context_msgs = messages[:-1] if len(messages) > 1 else []
                    cls_res = await asyncio.to_thread(self.classifier.classify, context_msgs, last_msg)
                    classifier_complexity = cls_res.query_complexity
                    if cls_res.signal_type in ("correction", "rejection", "positive", "clarification"):
                        signals.append(
                            {
                                "type": cls_res.signal_type,
                                "message": last_msg[:200],
                                "confidence": cls_res.confidence,
                                "dimension": cls_res.dimension,
                            }
                        )
                if metadata: # Also check for parameter changes (zero-cost)
                    param_sig = self._detector._check_param_change(sid, metadata)
                    if param_sig:
                        signals.append(param_sig)
            else:
                signals = self._detector.detect(messages=messages, session_id=sid, metadata=metadata)

            if signals:
                count = self._buffer.push(tenant_id, signals)
                logger.debug(f"async_add(): {len(signals)} signals added, total={count}")
                if self._buffer.should_reflect(tenant_id):
                    await self._reflect_async(tenant_id)

            if self.router:
                profile = self.get_profile(project_id, user_id)
                routing_decision = self.router.route(messages, profile)
                result["routing"] = {
                    "selected_model": routing_decision.selected_model,
                    "reason": routing_decision.reason,
                    "confidence": routing_decision.confidence,
                    "cost_tier": routing_decision.cost_tier
                }
        except Exception as e:
            logger.error(f"Behavioral async_add error: {e}")

        return result

    def search(self, query: str, project_id: str, user_id: str, limit: int = 10, **kwargs) -> dict:
        """
        Searches the memory and injects behavioral context into the results.

        Wraps the underlying provider's `search` method and, if enabled,
        appends the user's behavioral profile to the search results.
        """
        tenant_id = f"{project_id}:{user_id}"
        result = {}
        if self._mem0:
            result = self._mem0.search(query=query, user_id=tenant_id, limit=limit, **kwargs)

        if self._inject_profile:
            profile = self._store.load(tenant_id)
            if profile:
                injection = self._reflector.build_injection(profile)
                if injection:
                    result["behavioral_context"] = injection
                    result["behavioral_profile"] = profile
        return result

    def get_all(self, project_id: str, user_id: str, **kwargs) -> dict:
        """Retrieves all memories for a user, including the behavioral profile."""
        tenant_id = f"{project_id}:{user_id}"
        result = self._mem0.get_all(user_id=tenant_id, **kwargs) if self._mem0 else {"memories": []}
        profile = self._store.load(tenant_id)
        if profile:
            result["behavioral_profile"] = profile
        return result

    def delete_all(self, project_id: str, user_id: str, **kwargs) -> dict:
        """Deletes all memories for a user, including the behavioral profile."""
        tenant_id = f"{project_id}:{user_id}"
        result = self._mem0.delete_all(user_id=tenant_id, **kwargs) if self._mem0 else {}
        self._store.delete(tenant_id)
        self._detector.clear_session(tenant_id)
        self._profile_cache.pop(tenant_id, None)
        return result

    def get_profile(self, project_id: str, user_id: str) -> dict | None:
        """Retrieves a user's behavioral profile, using a time-based cache."""
        tenant_id = f"{project_id}:{user_id}"
        now = time.time()
        cached = self._profile_cache.get(tenant_id)
        if cached:
            profile, ts = cached
            if now - ts < self._profile_cache_ttl:
                return profile
        profile = self._store.load(tenant_id)
        self._profile_cache[tenant_id] = (profile, now)
        return profile

    def get_injection(self, project_id: str, user_id: str, current_messages: list[dict] | None = None) -> str:
        """
        Generates a system prompt injection string based on the user's profile.

        This method analyzes the context of the current messages to select the
        most relevant parts of the user's profile for injection.
        """
        tenant_id = f"{project_id}:{user_id}"
        profile = self._store.load(tenant_id)
        if not profile:
            return ""
            
        current_context = "general_chat"
        if current_messages:
            user_msgs = [m["content"] for m in current_messages if m.get("role") == "user"]
            if user_msgs:
                current_context = ContextClassifier.detect(user_msgs[-1])
                
        return self._reflector.build_injection(profile, current_context)

    def force_reflect(self, project_id: str, user_id: str) -> dict:
        """
        Triggers the reflection process immediately, bypassing the signal threshold.

        Useful for debugging and testing purposes.
        """
        tenant_id = f"{project_id}:{user_id}"
        signals = self._buffer.flush(tenant_id)
        if not signals:
            return {}
        existing_profile = self.get_profile(project_id, user_id)
        profile = self._reflector.reflect(tenant_id, signals, existing_profile)
        self._store.save(tenant_id, profile)
        self._profile_cache.pop(tenant_id, None)
        return profile

    def get_pending_signals(self, project_id: str, user_id: str) -> list[dict]:
        """Returns the list of signals currently in the buffer for a user."""
        tenant_id = f"{project_id}:{user_id}"
        return self._buffer.peek(tenant_id)

    def get_recommended_model(self, project_id: str, user_id: str, messages: list[dict]) -> RouterDecision | None:
        """Recommends the optimal model for a given request using the router."""
        if not self.router:
            return None
        profile = self.get_profile(project_id, user_id)
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
            profile = await asyncio.to_thread(self._reflector.reflect, tenant_id, signals, existing_profile)
            await asyncio.to_thread(self._store.save, tenant_id, profile)
            self._profile_cache.pop(tenant_id, None)
            logger.info(f"Reflect completed: {tenant_id}")
        except Exception as e:
            logger.error(f"Reflect error ({tenant_id}): {e}")
