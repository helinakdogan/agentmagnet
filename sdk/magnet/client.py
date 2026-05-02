"""
BehavioralMemory
----------------
Behavioral memory layer with BYOK (Bring Your Own Key) support.

This is the main client-facing class. It provides intelligent behavioral
learning capabilities including signal detection, reflection, and model routing.
No third-party memory provider dependency required.
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

    This class provides a behavioral layer that observes user interactions,
    extracts signals, and builds a dynamic user profile to personalize
    future responses. All processing happens within your own infrastructure.

    Args:
        openai_api_key (str, optional): BYOK — OpenAI API key for the Reflector LLM.
            Falls back to OPENAI_API_KEY environment variable if not provided.
        anthropic_api_key (str, optional): BYOK — Anthropic API key (alternative).
            Falls back to ANTHROPIC_API_KEY environment variable if not provided.
        redis_client (Any, optional): An initialized Redis client for persistence.
        signal_threshold (int): Number of signals to buffer before triggering reflection.
        reflector_model (str): The LLM to use for the reflection process.
        classifier_model (str): The LLM to use for the classification fallback.
        router (ModelRouter, optional): An instance of ModelRouter for dynamic model selection.
        qdrant_url (str, optional): Qdrant vector DB URL for long-term storage (optional).
        qdrant_api_key (str, optional): Qdrant API key.
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

        if openai_client is not None:
            logger.warning("BehavioralMemory: openai_client deprecated, ignoring.")
        if anthropic_client is not None:
            logger.warning("BehavioralMemory: anthropic_client deprecated, ignoring.")
        if kwargs:
            logger.warning(f"BehavioralMemory: unknown parameters ignored: {list(kwargs.keys())}")

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

    def add(
        self,
        messages: list[dict],
        user_id: str,
        project_id: str = "default",
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
            session_id (str, optional): The ID of the current session.
            metadata (dict, optional): Additional metadata about the interaction.

        Returns:
            dict: Result containing routing information if a router is configured.
        """
        tenant_id = f"{project_id}:{user_id}"
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
                    try:
                        loop = asyncio.get_running_loop()
                        loop.create_task(self._reflect_async(tenant_id))
                    except RuntimeError:
                        signals_to_reflect = self._buffer.flush(tenant_id)
                        if signals_to_reflect:
                            import threading
                            t = threading.Thread(
                                target=lambda: asyncio.run(self._do_reflect(tenant_id, signals_to_reflect)),
                                daemon=True,
                            )
                            t.start()

            if self.router:
                profile = self.get_profile(user_id, project_id)
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
        user_id: str,
        project_id: str = "default",
        session_id: str | None = None,
        metadata: dict | None = None,
        **kwargs,
    ) -> dict:
        """Asynchronous version of the `add` method."""
        tenant_id = f"{project_id}:{user_id}"
        result = {}

        try:
            sid = session_id or tenant_id
            signals = []

            if self.classifier:
                user_msgs = [m for m in messages if m.get("role") == "user"]
                if user_msgs:
                    last_msg = user_msgs[-1].get("content", "")
                    context_msgs = messages[:-1] if len(messages) > 1 else []
                    cls_res = await asyncio.to_thread(self.classifier.classify, context_msgs, last_msg)
                    if cls_res.signal_type in ("correction", "rejection", "positive", "clarification"):
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
                logger.debug(f"async_add(): {len(signals)} signals added, total={count}")
                if self._buffer.should_reflect(tenant_id):
                    await self._reflect_async(tenant_id)

            if self.router:
                profile = self.get_profile(user_id, project_id)
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

    def search(self, query: str, user_id: str, project_id: str = "default", limit: int = 10, **kwargs) -> dict:
        """
        Returns behavioral context for a user based on their learned profile.

        Args:
            query (str): The search query (used for context-aware injection).
            user_id (str): The ID of the user.
            project_id (str): The ID of the project.
            limit (int): Unused — kept for API compatibility.

        Returns:
            dict: Contains 'behavioral_context' (injection string) and 'behavioral_profile'.
        """
        tenant_id = f"{project_id}:{user_id}"
        profile = self._store.load(tenant_id)
        if not profile:
            return {}
        injection = self._reflector.build_injection(profile) if self._inject_profile else ""
        return {"behavioral_context": injection, "behavioral_profile": profile}

    def get_all(self, user_id: str, project_id: str = "default", **kwargs) -> dict:
        """Retrieves the full behavioral profile for a user."""
        tenant_id = f"{project_id}:{user_id}"
        profile = self._store.load(tenant_id)
        return {"behavioral_profile": profile} if profile else {}

    def delete_all(self, user_id: str, project_id: str = "default", **kwargs) -> dict:
        """Deletes all behavioral memory for a user."""
        tenant_id = f"{project_id}:{user_id}"
        self._store.delete(tenant_id)
        self._detector.clear_session(tenant_id)
        self._profile_cache.pop(tenant_id, None)
        return {"deleted": True}

    def get_profile(self, user_id: str, project_id: str = "default") -> dict | None:
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

    def get_injection(self, user_id: str, project_id: str = "default", current_messages: list[dict] | None = None) -> str:
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

    def force_reflect(self, user_id: str, project_id: str = "default") -> dict:
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

    def get_pending_signals(self, user_id: str, project_id: str = "default") -> list[dict]:
        """Returns the list of signals currently in the buffer for a user."""
        tenant_id = f"{project_id}:{user_id}"
        return self._buffer.peek(tenant_id)

    def get_recommended_model(self, user_id: str, messages: list[dict], project_id: str = "default") -> RouterDecision | None:
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
