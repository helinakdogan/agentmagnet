"""
Intelligent Classifier
----------------------
Detects behavioral signals and query characteristics from user messages.

This module implements a cascade classification pipeline that uses fast,
zero-cost heuristics first, and only falls back to a more expensive LLM
for ambiguous cases, optimizing for both speed and accuracy.
"""
from __future__ import annotations
import json
import logging
from dataclasses import dataclass
from typing import Any
import litellm
from .signals import _CORRECTION_RE, _REJECTION_RE

logger = logging.getLogger(__name__)


@dataclass
class ClassificationResult:
    """Represents the structured output of a classification task."""
    signal_type: str
    dimension: str
    query_complexity: str
    confidence: float
    reasoning: str


_CLASSIFIER_PROMPT = """You are a behavioral and intent analyst AI. 
Analyze the user's latest message and context.

TASK:
1. Signal Type: Choose from "correction", "rejection", "preference", "formatting_preference", "tone_preference", "detail_preference", "neutral".
2. Dimension: Use "llm_extracted".
3. Query Complexity: Choose from "simple", "medium", "complex".

RULES:
- ONLY return a valid JSON, do not use explanations or markdown backticks (```).
- If unsure, use signal_type="neutral", dimension="llm_extracted", query_complexity="medium".

OUTPUT FORMAT:
{
  "signal_type": "...",
  "dimension": "...",
  "query_complexity": "...",
  "confidence": 0.95,
  "reasoning": "Why did you make this decision? (brief)"
}
"""


class IntelligentClassifier:
    """
    A hybrid classifier for detecting user intent and query complexity.

    This class uses a multi-stage "cascade" approach:
    1.  A fast, rule-based heuristic classifier runs first.
    2.  If its confidence is high, the result is returned immediately.
    3.  If confidence is low, it falls back to a more powerful LLM-based
        classifier for a more nuanced analysis.
    """

    def __init__(
        self,
        model: str = "openai/gpt-4o-mini",
        fallback_rules: bool = True,
        llm_client: Any = None,
    ):
        if llm_client is not None:
            logger.warning(
                "IntelligentClassifier: llm_client parameter is deprecated, ignoring. "
                "Use litellm model string for model selection."
            )
        self.model = model
        self.confidence_threshold = 0.75

    def _heuristic_classify(self, text: str) -> ClassificationResult:
        """
        Performs a fast, zero-cost classification using keyword matching.

        Args:
            text (str): The user's message text.

        Returns:
            ClassificationResult: The result of the heuristic analysis.
        """
        text = text.lower()
        rules = {
            "correction": ["no", "not", "wrong", "not like that", "fix", "didn't mean", "incorrect"],
            "rejection": ["don't want", "no way", "no need", "skip", "pass", "nevermind"],
            "formatting_preference": ["bullet points", "list", "json", "markdown", "table", "bold"],
            "tone_preference": ["formal", "casual", "friendly", "serious", "polite", "funny", "professional"],
            "detail_preference": ["short", "long", "detailed", "summary", "briefly", "step by step", "simply"],
            "preference": ["prefer", "do it like this", "let it be like", "use this", "always"],
        }
        
        best_signal = "neutral"
        best_score = 0
        
        for signal, keywords in rules.items():
            score = sum(1 for kw in keywords if kw in text)
            if score > best_score:
                best_score = score
                best_signal = signal
                
        if best_score == 0:
            return ClassificationResult("neutral", "heuristic", "medium", 0.0, "No heuristic match")
            
        # A simple confidence score based on the number of matched keywords.
        confidence = min(0.60 + (best_score * 0.15), 0.95)
        
        return ClassificationResult(
            signal_type=best_signal,
            dimension="heuristic",
            query_complexity="simple",
            confidence=confidence,
            reasoning=f"Heuristic match: {best_score} keywords"
        )

    def classify(self, messages: list[dict], new_message: str) -> ClassificationResult:
        """
        Classifies a user message using the cascade pipeline.

        Args:
            messages (list[dict]): The preceding conversation history for context.
            new_message (str): The latest user message to classify.

        Returns:
            ClassificationResult: The final classification result.
        """
        # Stage 1: Attempt classification with zero-cost heuristics.
        heuristic_res = self._heuristic_classify(new_message)
        
        # Stage 2: If confidence is high, short-circuit and return immediately.
        if heuristic_res.confidence >= self.confidence_threshold:
            return heuristic_res
            
        # Stage 3: Fallback to the LLM for a more detailed analysis.
        return self._llm_classify(messages, new_message)
        
    def _llm_classify(self, messages: list[dict], new_message: str) -> ClassificationResult:
        """Performs classification using a Large Language Model."""
        try:
            context = messages[-3:] if len(messages) > 3 else messages
            context_str = json.dumps(context, ensure_ascii=False)
            user_prompt = f"Context:\n{context_str}\n\nNew Message: {new_message}"
            
            response = litellm.completion(
                model=self.model,
                messages=[
                    {"role": "system", "content": _CLASSIFIER_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.0,
                response_format={"type": "json_object"},
            )

            raw_content = response.choices[0].message.content.strip()
            data = json.loads(raw_content)

            return ClassificationResult(
                signal_type=data.get("signal_type", "neutral"),
                dimension=data.get("dimension", "llm_extracted"),
                query_complexity=data.get("query_complexity", "medium"),
                confidence=float(data.get("confidence", 0.5)),
                reasoning=data.get("reasoning", "LLM Extraction"),
            )
        except Exception as e:
            logger.error(f"LLM Classification error, returning neutral. Detail: {e}")
            return ClassificationResult("neutral", "fallback", "medium", 0.0, f"Error: {str(e)}")


class ContextClassifier:
    """
    A fast, keyword-based classifier to determine the high-level context of a query.
    Now supports dynamic contexts generated by the Reflector.
    """

    @classmethod
    def detect(cls, text: str, dynamic_contexts: list[str] = None) -> str:
        """Detects the most likely context from the user's dynamic profiles."""
        if not text or not dynamic_contexts:
            return "general_chat"
            
        text = text.lower()
        scores = {ctx: 0 for ctx in dynamic_contexts}
        
        for ctx in dynamic_contexts:
            # e.g., 'react_native_development' -> ['react', 'native', 'development']
            keywords = ctx.replace("_", " ").split()
            for kw in keywords:
                if len(kw) > 2 and kw in text:  # Ignore very short words
                    scores[ctx] += 1
        
        best_context = max(scores, key=scores.get)
        if scores[best_context] > 0:
            return best_context
        return "general_chat"
