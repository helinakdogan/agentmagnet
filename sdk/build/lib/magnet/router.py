"""
Model Router
------------
Selects the optimal LLM for a given request based on heuristics.

This module provides a deterministic, LLM-free routing mechanism to balance
cost and performance by directing simple queries to cheaper models and
complex queries to more powerful ones.
"""
from __future__ import annotations
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class RouterDecision:
    """Represents the output of the routing logic."""
    selected_model: str
    reason: str
    confidence: float
    estimated_complexity: str
    cost_tier: str


class ModelRouter:
    """
    A rule-based, LLM-free router for model selection.

    The router analyzes various features of the user's request, such as prompt
    length, conversation history, and the presence of complexity-indicating
    keywords, to make a routing decision.

    Attributes:
        cheap_model (str): Identifier for the cost-effective model.
        expensive_model (str): Identifier for the high-performance model.
    """
    def __init__(
        self, 
        cheap_model: str = "openai/gpt-4o-mini", 
        expensive_model: str = "openai/gpt-4o"
    ):
        self.cheap_model = cheap_model
        self.expensive_model = expensive_model
        
        self.complex_keywords = [
            "detailed analysis", "architecture", "comparison", "debug", 
            "production-ready", "optimization", "comprehensive", "design",
            "what is the difference", "why", "explain", "evaluate", "refactor"
        ]

    def route(self, messages: list[dict], user_profile: dict | None = None) -> RouterDecision:
        """
        Determines the best model for the given request.

        Args:
            messages (list[dict]): The full conversation history, including
                the latest user message.
            user_profile (dict | None): The user's behavioral profile (currently
                unused in this version but available for future enhancements).

        Returns:
            RouterDecision: An object containing the routing decision and rationale.
        """
        try:
            user_msgs = [m["content"] for m in messages if m.get("role") == "user"]
            if not user_msgs:
                return self._default_expensive("No user messages found")

            last_msg = user_msgs[-1].lower()
            history_len = len(messages)
            char_count = len(last_msg)
            
            score = 0
            reasons = []

            # Heuristic 1: Prompt Length
            if char_count > 600:
                score += 3
                reasons.append(f"Long prompt ({char_count} chars)")
            elif char_count < 50:
                score -= 1

            # Heuristic 2: Conversation Depth
            if history_len > 6:
                score += 2
                reasons.append(f"Long history ({history_len} msgs)")

            # 3. Keyword / Task Complexity
            matched_keywords = [kw for kw in self.complex_keywords if kw in last_msg]
            if matched_keywords:
                score += len(matched_keywords) * 3
                reasons.append(f"Complex keywords: {matched_keywords}")

            # Heuristic 4: Code Presence
            if "```" in last_msg or "def " in last_msg or "class " in last_msg:
                score += 2
                reasons.append("Code presence detected")

            # Final Decision Logic
            if score >= 4:
                confidence = min(0.5 + (score * 0.1), 1.0)
                return RouterDecision(self.expensive_model, " + ".join(reasons), confidence, "complex", "expensive")
            elif score <= 0:
                confidence = 0.85
                return RouterDecision(self.cheap_model, "Short/Simple query", confidence, "simple", "cheap")
            else:
                # Ambiguous case: default to the more powerful model for safety.
                return RouterDecision(self.expensive_model, f"Low confidence routing (score: {score}), defaulting to expensive", 0.40, "medium", "expensive")
                
        except Exception as e:
            logger.error(f"Router error: {e}")
            return self._default_expensive(f"Fallback due to error: {str(e)}")
            
    def _default_expensive(self, reason: str) -> RouterDecision:
        """Returns a default decision for the expensive model in case of errors."""
        return RouterDecision(self.expensive_model, reason, 0.0, "unknown", "expensive")
