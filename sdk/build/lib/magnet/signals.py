"""
Signal Detector
---------------
Extracts signals from user behavior.
Does not use LLM — purely rule-based, zero cost.
"""

from __future__ import annotations
import re
import json
import hashlib
import logging
from typing import Any

CORRECTION_PATTERNS = [
    r"\bno\b", r"\bnot\b", r"\bwrong\b", r"\bagain\b",
    r"\bnot like that\b", r"\bdidn't mean\b", r"\bnot (that|what)\b",
    r"\bthat'?s? (not|wrong)\b", r"\bno,?\s+i\b", r"\bi (didn'?t|don'?t) mean\b",
    r"\bhay[ıi]r\b", r"\byanl[ıi][sş]\b", r"\b[oö]yle de[gğ]il\b", r"\bd[uü]zelt\b",
    r"\bkastetmedim\b", r"\bb[oö]yle istemedim\b", r"\bhatal[ıi]\b",
    r"\bupdate\b", r"\bchange\b", r"\binstead\b", r"\bmodify\b"
]

REJECTION_PATTERNS = [
    r"\bdon't want\b", r"\bno way\b", r"\bno need\b", r"\bskip\b",
    r"\bskip\b", r"\bno thanks?\b", r"\bnot (now|this|that)\b",
    r"\bi('ll)? pass\b", r"\bnevermind\b",
    r"\bistemiyorum\b", r"\bgerek yok\b", r"\bge[cç]\b", r"\bbo[sş]ver\b",
    r"\bvazge[cç]tim\b", r"\biptal\b", r"\bistemem\b",
    r"\breject\b", r"\brefuse\b"
]

_CORRECTION_RE = re.compile("|".join(CORRECTION_PATTERNS), re.IGNORECASE)
_REJECTION_RE = re.compile("|".join(REJECTION_PATTERNS), re.IGNORECASE)
logger = logging.getLogger(__name__)
_PARAM_HISTORY_TTL = 60 * 60 * 24 * 7


class SignalDetector:
    def __init__(self, param_change_threshold: int = 3, redis_client: Any | None = None):
        self.param_change_threshold = param_change_threshold
        self._redis = redis_client
        self._param_history: dict[str, dict[str, list[str]]] = {}

    def detect(
        self,
        messages: list[dict],
        session_id: str,
        metadata: dict | None = None,
    ) -> list[dict]:
        signals: list[dict] = []

        for msg in messages:
            role = msg.get("role", "")
            content = msg.get("content", "")
            if not isinstance(content, str):
                content = str(content)

            if role == "user":
                if self._is_correction(content):
                    signals.append({"type": "correction", "message": content[:200], "confidence": 0.85})
                if self._is_rejection(content):
                    signals.append({"type": "rejection", "message": content[:200], "confidence": 0.80})

        if metadata:
            param_signal = self._check_param_change(session_id, metadata)
            if param_signal:
                signals.append(param_signal)

        return signals

    def clear_session(self, session_id: str) -> None:
        self._param_history.pop(session_id, None)
        if self._redis:
            try:
                self._redis.delete(self._history_key(session_id))
            except Exception as e:
                logger.warning(f"Redis session clear error ({session_id}): {e}")

    def _is_correction(self, text: str) -> bool:
        return bool(_CORRECTION_RE.search(text))

    def _is_rejection(self, text: str) -> bool:
        return bool(_REJECTION_RE.search(text))

    def _check_param_change(self, session_id: str, metadata: dict) -> dict | None:
        history = self._load_history(session_id)
        changed_params = []

        for key, value in metadata.items():
            if key in ("user_id", "session_id", "timestamp"):
                continue
            value_hash = hashlib.md5(str(value).encode()).hexdigest()[:8]
            if key not in history:
                history[key] = [value_hash]
            else:
                if value_hash != history[key][-1]:
                    history[key].append(value_hash)
                unique_values = len(set(history[key]))
                if unique_values >= self.param_change_threshold:
                    changed_params.append({"param": key, "change_count": unique_values, "latest": value})

        if changed_params:
            self._save_history(session_id, history)
            return {"type": "parameter_change", "params": changed_params, "confidence": 0.90}

        self._save_history(session_id, history)
        return None

    def _history_key(self, session_id: str) -> str:
        return f"vmm:param_history:{session_id}"

    def _load_history(self, session_id: str) -> dict[str, list[str]]:
        if self._redis:
            try:
                raw_map = self._redis.hgetall(self._history_key(session_id))
                if raw_map:
                    parsed: dict[str, list[str]] = {}
                    for key, raw_value in raw_map.items():
                        values = json.loads(raw_value)
                        if isinstance(values, list):
                            parsed[key] = [str(v) for v in values]
                    return parsed
            except Exception as e:
                logger.warning(f"Could not read Redis param history ({session_id}), memory fallback: {e}")
        return self._param_history.setdefault(session_id, {})

    def _save_history(self, session_id: str, history: dict[str, list[str]]) -> None:
        self._param_history[session_id] = history
        if self._redis:
            try:
                key = self._history_key(session_id)
                pipe = self._redis.pipeline()
                pipe.delete(key)
                for field, values in history.items():
                    pipe.hset(key, field, json.dumps(values, ensure_ascii=False))
                pipe.expire(key, _PARAM_HISTORY_TTL)
                pipe.execute()
            except Exception as e:
                logger.warning(f"Could not write Redis param history ({session_id}), memory fallback: {e}")
