"""
Intelligent Classifier
----------------------
Detects behavioral signals and query characteristics from user messages.

This module implements a cascade classification pipeline that uses fast,
zero-cost heuristics first, and only falls back to a more expensive LLM
for ambiguous cases, optimizing for both speed and accuracy.

Signal types:
  - preference_like     : "color looks great", "keep it up", "I love this"
  - preference_dislike  : "I don't like red", "don't do this", "I hate it"
  - correction          : "no that's wrong", "that's not right", "fix it"
  - rejection           : "cancel", "never mind", "not needed"
  - tone_preference     : "be more casual", "don't be so formal"
  - formatting_preference : "use a list", "use markdown"
  - detail_preference   : "keep it short", "explain in detail"
  - neutral             : no behavioral signal detected
"""
from __future__ import annotations
import json
import logging
from dataclasses import dataclass
from typing import Any
import litellm  # type: ignore
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
    extracted_preference: str = ""  # Free-text: what exactly was learned


_CLASSIFIER_PROMPT = """You are a behavioral and intent analyst AI.
Analyze the user's latest message and the conversation context.

TASK:
1. **signal_type**: Choose from:
   - "preference_dislike" : User expresses dislike for something (color, style, tone, topic, person, etc.)
   - "preference_like"    : User expresses liking/approval for something
   - "correction"         : User says the AI did something wrong and wants it fixed
   - "rejection"          : User cancels, dismisses, or says it's not needed
   - "tone_preference"    : User wants a different communication style
   - "formatting_preference" : User wants different output format
   - "detail_preference"  : User wants more or less detail
   - "neutral"            : No behavioral signal detected

2. **dimension**: A short slug describing WHAT the preference is about.
   Examples: "color_preference", "tone", "response_length", "topic_avoidance", "language", "formality"

3. **extracted_preference**: A short human-readable sentence describing EXACTLY what was learned.
   This will be stored in long-term memory. Be specific.
   Write the extracted preference in the SAME LANGUAGE as the user's message.
   Examples:
   - "User dislikes the color red"
   - "User prefers bullet point lists over paragraphs"
   - "User wants a friendly and witty tone"
   - "User dislikes overly long explanations"
   Leave empty string "" if signal_type is "neutral".

4. **query_complexity**: "simple", "medium", or "complex"

5. **confidence**: Float 0.0-1.0
   - Strong explicit statement ("don't like", "hate", "I love") → 0.85-0.95
   - Implicit or soft signal ("it was a bit boring", "hmm not really") → 0.50-0.70
   - Repeated preference (mentioned multiple times in context) → boost to 0.95+

RULES:
- ONLY return valid JSON, no markdown, no backticks.
- Detect preferences even if phrased indirectly. E.g. "I really don't like red, like we talked about" → preference_dislike.
- Consider the full conversation context, not just the last message.
- Turkish and English are both valid. Detect signals in both languages.
- If the user references a past conversation, boost confidence.

OUTPUT FORMAT:
{
  "signal_type": "...",
  "dimension": "...",
  "extracted_preference": "...",
  "query_complexity": "...",
  "confidence": 0.90,
  "reasoning": "Brief explanation"
}
"""


# Heuristic keyword sets — zero cost, runs first
_LIKE_KEYWORDS_TR = [
    "seviyorum", "severim", "çok güzel", "harika", "mükemmel", "süper",
    "beğendim", "beğeniyorum", "böyle devam et", "tam istediğim gibi",
    "bu güzeldi", "bu çok iyi", "devam et böyle", "aynen böyle", "bravo",
    "memnunum", "çok iyi oldu", "çok beğendim",
]
_LIKE_KEYWORDS_EN = [
    "love", "like", "great", "perfect", "excellent", "awesome", "amazing",
    "keep it", "keep doing", "well done", "exactly", "this is good",
    "i prefer", "please always", "please keep",
]
_DISLIKE_KEYWORDS_TR = [
    "sevmiyorum", "sevmem", "istemiyorum", "istemem", "nefret", "hiç beğenmedim",
    "beğenmiyorum", "böyle olmasın", "bunu istemiyorum", "sıkıcı", "berbat",
    "kötü", "olmaz", "hoşlanmıyorum", "hoşlanmam", "rahatsız ediyor",
    "rahatsız oluyorum", "sinir bozucu", "can sıkıcı", "iğrenç",
    "hiç sevmiyorum", "hiç istemiyorum", "asla istemem",
]
_DISLIKE_KEYWORDS_EN = [
    "don't like", "dislike", "hate", "not a fan", "prefer not", "avoid",
    "never", "please don't", "stop doing", "annoying", "terrible",
    "awful", "horrible", "bad", "worst", "i don't want",
]
_TONE_KEYWORDS = [
    "samimi", "daha samimi", "daha dostane", "resmi olma", "sert konuşma",
    "sert olma", "kibar ol", "daha eğlenceli", "espri yap", "formal",
    "informal", "casual", "friendly", "be more", "tone",
]
_FORMAT_KEYWORDS = [
    "bullet", "liste", "madde", "markdown", "tablo", "table",
    "json formatında", "numara ver", "numbered", "bold", "başlık",
]
_DETAIL_KEYWORDS_SHORT = [
    "kısa", "özet", "brief", "short", "briefly", "özetle", "sadece özet",
    "uzatma", "uzun uzun anlatma", "çok kısa",
]
_DETAIL_KEYWORDS_LONG = [
    "uzun", "detaylı", "ayrıntılı", "detailed", "in detail", "step by step",
    "adım adım", "açıkla", "derinlemesine",
]


class IntelligentClassifier:
    """
    A hybrid classifier for detecting user intent and behavioral preferences.

    Uses a multi-stage "cascade" approach:
    1. Fast, zero-cost heuristic classification (keyword matching).
    2. If heuristic confidence is >= threshold, return immediately.
    3. Otherwise, fall back to LLM for nuanced semantic analysis.

    Key improvement over previous version:
    - Detects semantic like/dislike signals, not just correction patterns.
    - Extracts `extracted_preference` field for direct long-term memory storage.
    - Dynamic confidence based on signal strength and repetition in context.
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
        Fast, zero-cost classification using keyword matching.
        Now covers like/dislike semantics in addition to corrections.
        """
        text_lower = text.lower()

        # Check dislike first (higher priority — more actionable)
        dislike_score = sum(1 for kw in _DISLIKE_KEYWORDS_TR if kw in text_lower)
        dislike_score += sum(1 for kw in _DISLIKE_KEYWORDS_EN if kw in text_lower)
        if dislike_score > 0:
            confidence = min(0.65 + (dislike_score * 0.15), 0.92)
            return ClassificationResult(
                signal_type="preference_dislike",
                dimension="general_dislike",
                query_complexity="simple",
                confidence=confidence,
                reasoning=f"Dislike keywords matched: {dislike_score}",
                extracted_preference="",  # LLM will fill this in detail
            )

        # Check like/approval
        like_score = sum(1 for kw in _LIKE_KEYWORDS_TR if kw in text_lower)
        like_score += sum(1 for kw in _LIKE_KEYWORDS_EN if kw in text_lower)
        if like_score > 0:
            confidence = min(0.60 + (like_score * 0.12), 0.88)
            return ClassificationResult(
                signal_type="preference_like",
                dimension="general_like",
                query_complexity="simple",
                confidence=confidence,
                reasoning=f"Like keywords matched: {like_score}",
                extracted_preference="",
            )

        # Tone preference
        tone_score = sum(1 for kw in _TONE_KEYWORDS if kw in text_lower)
        if tone_score > 0:
            return ClassificationResult(
                signal_type="tone_preference",
                dimension="tone",
                query_complexity="simple",
                confidence=min(0.70 + (tone_score * 0.10), 0.90),
                reasoning=f"Tone keywords matched: {tone_score}",
                extracted_preference="",
            )

        # Formatting preference
        format_score = sum(1 for kw in _FORMAT_KEYWORDS if kw in text_lower)
        if format_score > 0:
            return ClassificationResult(
                signal_type="formatting_preference",
                dimension="formatting",
                query_complexity="simple",
                confidence=min(0.65 + (format_score * 0.10), 0.88),
                reasoning=f"Format keywords matched: {format_score}",
                extracted_preference="",
            )

        # Detail preference
        short_score = sum(1 for kw in _DETAIL_KEYWORDS_SHORT if kw in text_lower)
        long_score = sum(1 for kw in _DETAIL_KEYWORDS_LONG if kw in text_lower)
        if short_score > 0:
            return ClassificationResult(
                signal_type="detail_preference",
                dimension="response_length",
                query_complexity="simple",
                confidence=min(0.65 + (short_score * 0.12), 0.88),
                reasoning=f"Short-detail keywords: {short_score}",
                extracted_preference="",
            )
        if long_score > 0:
            return ClassificationResult(
                signal_type="detail_preference",
                dimension="response_length",
                query_complexity="simple",
                confidence=min(0.65 + (long_score * 0.12), 0.88),
                reasoning=f"Long-detail keywords: {long_score}",
                extracted_preference="",
            )

        # Correction / rejection (original patterns)
        if _CORRECTION_RE.search(text):
            return ClassificationResult(
                signal_type="correction",
                dimension="correction",
                query_complexity="simple",
                confidence=0.80,
                reasoning="Correction pattern matched",
                extracted_preference="",
            )
        if _REJECTION_RE.search(text):
            return ClassificationResult(
                signal_type="rejection",
                dimension="rejection",
                query_complexity="simple",
                confidence=0.80,
                reasoning="Rejection pattern matched",
                extracted_preference="",
            )

        return ClassificationResult("neutral", "heuristic", "medium", 0.0, "No heuristic match")

    def classify(self, messages: list[dict], new_message: str) -> ClassificationResult:
        """
        Classifies a user message using the cascade pipeline.

        Args:
            messages (list[dict]): Preceding conversation history for context.
            new_message (str): The latest user message to classify.

        Returns:
            ClassificationResult: Final classification result with extracted_preference.
        """
        # Stage 1: Zero-cost heuristic
        heuristic_res = self._heuristic_classify(new_message)

        # Stage 2: High-confidence heuristic → return immediately (but still get extracted_preference via LLM)
        if heuristic_res.confidence >= self.confidence_threshold:
            # Heuristic is confident about TYPE; use LLM to extract the *what* (extracted_preference)
            # only if the dimension is generic (we need specifics like "red color dislike")
            if not heuristic_res.extracted_preference:
                llm_res = self._llm_classify(messages, new_message)
                heuristic_res.extracted_preference = llm_res.extracted_preference
                heuristic_res.dimension = llm_res.dimension or heuristic_res.dimension
            return heuristic_res

        # Stage 3: Fallback to LLM for ambiguous cases
        return self._llm_classify(messages, new_message)

    def _llm_classify(self, messages: list[dict], new_message: str) -> ClassificationResult:
        """Performs classification using a Large Language Model."""
        try:
            context = messages[-5:] if len(messages) > 5 else messages
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
                extracted_preference=data.get("extracted_preference", ""),
            )
        except Exception as e:
            logger.error(f"LLM Classification error, returning neutral. Detail: {e}")
            return ClassificationResult("neutral", "fallback", "medium", 0.0, f"Error: {str(e)}")


class ContextClassifier:
    """
    A fast, keyword-based classifier to determine the high-level context of a query.
    Supports dynamic contexts generated by the Reflector.
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
                if len(kw) > 2 and kw in text:
                    scores[ctx] += 1

        best_context = max(scores, key=scores.get)
        if scores[best_context] > 0:
            return best_context
        return "general_chat"
