"""
Reflector
---------
Extracts a behavioral profile from accumulated signals.

This module uses an LLM to analyze a buffer of behavioral signals and
synthesizes them into structured preference facts. Each preference is a
typed object with subject, relation, confidence, and decay metadata.
"""

from __future__ import annotations
import json
import logging
import time
import litellm  # type: ignore

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT = """You are a behavioral analyst AI.
Analyze behavioral signals from a user's interaction with an AI agent.
Extract structured preference facts.

OUTPUT FORMAT (only JSON array, write nothing else):
[
  {
    "subject": "mushroom",
    "subject_type": "ingredient",
    "relation": "dislikes",
    "value": "avoid",
    "natural_text": "User dislikes mushrooms and does not want any dishes that include them",
    "preference_type": "permanent",
    "confidence": 0.9
  }
]

RULES:
- subject_type: "ingredient" | "format" | "tone" | "creative" | "behavior"
- relation: "dislikes" | "prefers" | "expects"
- preference_type:
    ingredient → permanent
    format, behavior → contextual
    creative, emotional, one-time requests → transient
    "never"/"always"/"hate" keywords → permanent
    session-specific requests → transient
- confidence: 1 signal=0.40, 2 signals=0.60, 3 signals=0.75, 5+ signals=0.90+
- Never hallucinate. Only extract what is actually in the signals.
- Always output a valid JSON array with no extra text.
- If a signal has an `extracted_preference` field, use it verbatim in natural_text.
"""

_USER_TEMPLATE = """User ID: {user_id}
Existing preferences: {existing_preferences}
Signal count: {signal_count}

Signals:
{signals_json}

Extract or update preference facts from these signals.
Return only NEW or CHANGED preferences as a JSON array."""

_FALLBACK_SYSTEM_PROMPT = """You are a behavioral analyst AI.
You will analyze behavioral signals from a user's interaction with an AI agent.
Your goal is to extract a structured user preference profile from these signals.

OUTPUT FORMAT (only JSON, write nothing else):
{
  "global_preferences": {
    "language": {"value": "turkish", "confidence": 0.95},
    "formatting": {"value": "markdown", "confidence": 0.80}
  },
  "contextual_profiles": {},
  "likes": [
    "Markdown-formatted responses",
    "Short and concise explanations"
  ],
  "dislikes": [
    "Long, repetitive explanations",
    "Formal language"
  ],
  "personality_expectations": [
    "Friendly and witty tone",
    "Direct answers without unnecessary preamble"
  ]
}

RULES:
- likes: Free-text list of things the user explicitly likes or approves of. Be specific.
- dislikes: Free-text list of things the user dislikes, wants to avoid, or complained about.
- personality_expectations: How the user wants the AI to behave or communicate.
- Confidence: 1 signal=0.40, 2 signals=0.60, 3 signals=0.75, 5+ signals=0.90+
- Never hallucinate. Only extract what's actually in the signals.
- Always output valid JSON with no extra text.
- If a signal has `extracted_preference` field, use it verbatim in the appropriate list.
"""

_FALLBACK_USER_TEMPLATE = """User ID: {user_id}
Signal count: {signal_count}

Signals:
{signals_json}

Extract the behavioral profile from these signals."""


class Reflector:
    """
    Analyzes behavioral signals to generate and update a user's profile.

    Profile contains:
      - preferences: list of hybrid fact objects (subject, relation, confidence, decay)
      - global_preferences: formatting, language, response style (legacy support)
      - contextual_profiles: topic-specific adaptations (legacy support)
    """
    def __init__(
        self,
        model: str = "openai/gpt-4o-mini",
        openai_api_key: str | None = None,
        anthropic_api_key: str | None = None,
        profile_ttl: int = 60 * 60 * 24 * 30,
        **kwargs,
    ):
        self.model = model
        self._openai_api_key = openai_api_key
        self._anthropic_api_key = anthropic_api_key
        self.profile_ttl = profile_ttl

        if "openai_client" in kwargs and kwargs["openai_client"] is not None:
            logger.warning("Reflector: openai_client parameter is deprecated, ignoring.")
        if "anthropic_client" in kwargs and kwargs["anthropic_client"] is not None:
            logger.warning("Reflector: anthropic_client parameter is deprecated, ignoring.")

    def reflect(self, user_id: str, signals: list[dict], existing_profile: dict | None = None) -> dict:
        """
        Performs the reflection process to update a user profile.

        Pass 1: request new structured JSON array format. If the model returns
        a non-empty list with valid subject keys, use it directly.
        Pass 2: if Pass 1 yields nothing (model returned old dict format,
        empty array, or failed), re-call with the legacy prompt and convert
        the result to the new preference object schema.
        """
        if not signals:
            return existing_profile or self._empty_profile()

        current_state = self._migrate_profile(existing_profile) if existing_profile else self._empty_profile()
        existing_preferences = current_state.get("preferences", [])

        try:
            pass1_result = self._reflect_pass1(user_id, signals, existing_preferences)
            print(f"[PASS1] result: {pass1_result[:2] if pass1_result else 'EMPTY'}")

            if not pass1_result:
                logger.info(f"Reflector: Pass 1 empty for {user_id}, falling back to legacy format")
                pass2_result = self._reflect_pass2(user_id, signals)
                print(f"[PASS2] result: {pass2_result[:2] if pass2_result else 'EMPTY'}")
                new_prefs = pass2_result
            else:
                new_prefs = pass1_result

            is_correction = any(s.get("type") in ("correction", "rejection") for s in signals)
            profile = self._merge_state(current_state, new_prefs, is_correction)
            profile["reflected_at"] = time.time()
            profile["signal_count"] = len(signals)
            logger.info(f"Reflector: profile updated for {user_id} — {len(signals)} signals, {len(new_prefs)} new prefs")
            return profile
        except Exception as e:
            logger.error(f"Reflector error ({user_id}): {e}")
            return current_state

    def _reflect_pass1(self, user_id: str, signals: list[dict], existing_preferences: list) -> list:
        """Request new structured JSON array; return validated preference objects."""
        prompt = _USER_TEMPLATE.format(
            user_id=user_id,
            existing_preferences=json.dumps(existing_preferences, ensure_ascii=False),
            signal_count=len(signals),
            signals_json=json.dumps(signals, ensure_ascii=False, indent=2),
        )
        try:
            raw = self._call_llm(prompt)
            parsed = self._parse(raw)
            if isinstance(parsed, list):
                valid = [p for p in parsed if isinstance(p, dict) and "subject" in p]
                return valid
            # Model wrapped the array in a dict
            if isinstance(parsed, dict):
                wrapped = parsed.get("preferences", [])
                if isinstance(wrapped, list):
                    valid = [p for p in wrapped if isinstance(p, dict) and "subject" in p]
                    return valid
        except Exception as e:
            logger.warning(f"Reflector Pass 1 failed for {user_id}: {type(e).__name__}: {e}")
        return []

    def _reflect_pass2(self, user_id: str, signals: list[dict]) -> list:
        """Fallback: use legacy prompt, then convert old dict to new preference objects."""
        prompt = _FALLBACK_USER_TEMPLATE.format(
            user_id=user_id,
            signal_count=len(signals),
            signals_json=json.dumps(signals, ensure_ascii=False, indent=2),
        )
        try:
            raw = self._call_llm(prompt, system_prompt=_FALLBACK_SYSTEM_PROMPT)
            parsed = self._parse(raw)
            if isinstance(parsed, dict):
                return self._convert_old_format(parsed)
            # Model returned a list even with fallback prompt — normalize and use it
            if isinstance(parsed, list):
                valid = [p for p in parsed if isinstance(p, dict) and "subject" in p]
                return valid
        except Exception as e:
            logger.warning(f"Reflector Pass 2 failed for {user_id}: {type(e).__name__}: {e}")
        return []

    @staticmethod
    def _convert_old_format(old_dict: dict) -> list:
        """Convert a legacy likes/dislikes/personality dict to new preference objects."""
        now = time.time()
        result = []
        for item in old_dict.get("likes", []):
            if isinstance(item, str) and item:
                result.append({
                    "subject": item, "subject_type": "general", "relation": "prefers",
                    "value": item, "natural_text": item, "preference_type": "contextual",
                    "confidence": 0.7, "valid_from": now, "decay_rate": 0.02, "recall_count": 0,
                })
        for item in old_dict.get("dislikes", []):
            if isinstance(item, str) and item:
                result.append({
                    "subject": item, "subject_type": "general", "relation": "dislikes",
                    "value": item, "natural_text": item, "preference_type": "contextual",
                    "confidence": 0.7, "valid_from": now, "decay_rate": 0.02, "recall_count": 0,
                })
        for item in old_dict.get("personality_expectations", []):
            if isinstance(item, str) and item:
                result.append({
                    "subject": item, "subject_type": "tone", "relation": "expects",
                    "value": item, "natural_text": item, "preference_type": "contextual",
                    "confidence": 0.7, "valid_from": now, "decay_rate": 0.02, "recall_count": 0,
                })
        return result

    def instant_learn(
        self,
        user_id: str,
        signal_type: str,
        extracted_preference: str,
        confidence: float,
        existing_profile: dict | None = None,
    ) -> dict:
        """
        Instantly learns a single strong preference without waiting for threshold.
        Used for high-confidence signals (dislike/like) on the first occurrence.
        """
        profile = existing_profile or self._empty_profile()
        profile = self._migrate_profile(profile)

        if not extracted_preference:
            return profile

        relation_map = {
            "preference_dislike": "dislikes",
            "preference_like": "prefers",
            "tone_preference": "expects",
            "formatting_preference": "expects",
            "detail_preference": "expects",
        }
        relation = relation_map.get(signal_type)
        if not relation:
            return profile

        subject_type_map = {
            "preference_dislike": "behavior",
            "preference_like": "behavior",
            "tone_preference": "tone",
            "formatting_preference": "format",
            "detail_preference": "format",
        }
        subject_type = subject_type_map.get(signal_type, "behavior")

        pref_type_map = {"format": "contextual", "tone": "contextual", "behavior": "contextual"}
        preference_type = pref_type_map.get(subject_type, "contextual")

        keywords = ("never", "always", "hate", "love", "despise")
        if any(k in extracted_preference.lower() for k in keywords):
            preference_type = "permanent"

        pref_obj = {
            "subject": extracted_preference,
            "subject_type": subject_type,
            "relation": relation,
            "value": "avoid" if relation == "dislikes" else "prefer",
            "natural_text": extracted_preference,
            "preference_type": preference_type,
            "confidence": confidence,
            "valid_from": time.time(),
            "decay_rate": 0.02,
            "recall_count": 0,
        }
        self._upsert_preference(profile.setdefault("preferences", []), pref_obj)
        logger.info(f"instant_learn: preference upserted for {user_id}: {extracted_preference!r}")

        profile["reflected_at"] = time.time()
        return profile

    def build_injection(self, profile: dict, current_context: str = "general_chat") -> str:
        """
        Constructs a rich system prompt injection string from the user profile.
        Increments recall_count for each injected preference.
        """
        profile = self._migrate_profile(profile)
        preferences = profile.get("preferences", [])
        global_prefs = profile.get("global_preferences", {})
        ctx_profile = profile.get("contextual_profiles", {}).get(current_context, {})

        _BASE_DECAY = {"permanent": 0.005, "contextual": 0.02, "transient": 0.10}

        active_prefs = []
        for pref in preferences:
            if not isinstance(pref, dict):
                continue
            days = (time.time() - pref.get("valid_from", time.time())) / 86400
            recall_count = pref.get("recall_count", 0)
            pref_type = pref.get("preference_type", "contextual")
            base = _BASE_DECAY.get(pref_type, 0.02)
            usage_factor = 1 / (1 + recall_count * 0.1)
            effective_decay = base * usage_factor
            eff_conf = pref.get("confidence", 0.8) * ((1 - effective_decay) ** days)
            if eff_conf < 0.1:
                continue
            pref["recall_count"] = recall_count + 1
            active_prefs.append((pref, eff_conf))

        has_content = bool(active_prefs or global_prefs or ctx_profile)
        if not has_content:
            return ""

        lines = ["[Behavioral Profile — Learned from this user's behavior]"]

        # Legacy structured preferences
        all_prefs = {}
        for k, v in global_prefs.items():
            if k == "patterns":
                continue
            if isinstance(v, dict) and "value" in v and v["value"] not in (None, "unknown", ""):
                all_prefs[k] = v
        for k, v in ctx_profile.items():
            if isinstance(v, dict) and "value" in v and v["value"] not in (None, "unknown", ""):
                all_prefs[k] = v

        if all_prefs:
            lines.append("\nLearned preferences:")
            for k, v in all_prefs.items():
                conf = v.get("confidence", 0.5)
                valid_from = v.get("valid_from")
                decay_rate = v.get("decay_rate", 0.02)
                if valid_from:
                    days_elapsed = (time.time() - valid_from) / 86400
                    conf *= (1 - decay_rate) ** days_elapsed
                if conf < 0.1:
                    continue
                pct = int(conf * 100)
                lines.append(f"  - {k}: {v['value']} (confidence: {pct}%)")

        likes = [(p, c) for p, c in active_prefs if p.get("relation") == "prefers"]
        dislikes = [(p, c) for p, c in active_prefs if p.get("relation") == "dislikes"]
        expects = [(p, c) for p, c in active_prefs if p.get("relation") == "expects"]

        if likes:
            lines.append("\nThis user likes / approves of:")
            for pref, _ in likes[:10]:
                lines.append(f"  ✓ {pref['natural_text']}")

        if dislikes:
            lines.append("\nThis user DISLIKES — always avoid these:")
            for pref, _ in dislikes[:10]:
                lines.append(f"  ✗ {pref['natural_text']}")

        if expects:
            lines.append("\nPersonality & communication expectations:")
            for pref, _ in expects[:5]:
                lines.append(f"  → {pref['natural_text']}")

        patterns = global_prefs.get("patterns", [])
        if isinstance(patterns, list) and patterns:
            lines.append("\nRecent patterns:")
            for p in patterns[:3]:
                lines.append(f"  - {p}")

        lines.append("\nNote: These preferences were learned from user behavior.")
        lines.append("Respect them strictly. Allow user to override at any time.")

        return "\n".join(lines)

    def _upsert_preference(self, existing: list, item: dict) -> None:
        subject = item.get("subject", "").lower().strip()
        relation = item.get("relation", "")
        opposite = {"prefers": "dislikes", "dislikes": "prefers"}
        opposite_relation = opposite.get(relation)

        for i, pref in enumerate(existing):
            if not isinstance(pref, dict):
                continue
            pref_subject = pref.get("subject", "").lower().strip()
            if pref_subject != subject:
                continue
            if pref.get("relation") == relation:
                existing[i] = {**pref, "natural_text": item["natural_text"], "confidence": item["confidence"]}
                logger.debug(f"_upsert_preference: updated {subject!r} {relation!r}")
                return
            if pref.get("relation") == opposite_relation:
                existing.pop(i)
                logger.debug(f"_upsert_preference: conflict removed {subject!r} {opposite_relation!r}")
                break

        existing.append(item)

    def _call_llm(self, prompt: str, system_prompt: str = _SYSTEM_PROMPT) -> str:
        api_key = (
            self._openai_api_key if "openai" in self.model
            else self._anthropic_api_key
        ) or None

        response = litellm.completion(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt},
            ],
            temperature=0,
            max_tokens=800,
            api_key=api_key,
        )
        return response.choices[0].message.content

    def _parse(self, raw: str) -> list | dict:
        raw = raw.strip()
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
        parsed = json.loads(raw.strip())
        if isinstance(parsed, list):
            return [self._normalize_pref_keys(p) if isinstance(p, dict) else p for p in parsed]
        return parsed

    @staticmethod
    def _normalize_pref_keys(pref: dict) -> dict:
        """Accept alternative field names the LLM may use."""
        normalized = dict(pref)
        if "subject" not in normalized and "entity" in normalized:
            normalized["subject"] = normalized.pop("entity")
        if "relation" not in normalized and "type" in normalized:
            normalized["relation"] = normalized.pop("type")
        return normalized

    def _merge_state(self, current: dict, new_prefs: list, is_correction: bool = False) -> dict:
        existing_list = current.setdefault("preferences", [])
        print(f"[MERGE] received {len(new_prefs)} preferences")
        for pref in new_prefs:
            if not isinstance(pref, dict) or "subject" not in pref:
                continue
            if "valid_from" not in pref:
                pref["valid_from"] = time.time()
            pref.setdefault("decay_rate", 0.02)
            pref.setdefault("recall_count", 0)
            if is_correction:
                pref["confidence"] = max(0.0, pref.get("confidence", 0.5) - 0.2)
            self._upsert_preference(existing_list, pref)
            print(f"[MERGE] profile now has {len(current.get('preferences', []))} items")
        return current

    @staticmethod
    def _empty_profile() -> dict:
        return {
            "global_preferences": {},
            "contextual_profiles": {},
            "preferences": [],
            "reflected_at": None,
            "signal_count": 0,
        }

    def _migrate_profile(self, profile: dict) -> dict:
        """Migrates legacy profiles to the hybrid fact schema."""
        # Handle very old format where preferences was a dict
        old_prefs = profile.get("preferences")
        if isinstance(old_prefs, dict):
            profile.pop("preferences")
            if "contextual_profiles" not in profile:
                profile["contextual_profiles"] = {
                    "general_chat": {
                        "response_length": old_prefs.get("response_length", "unknown"),
                        "explanation_depth": old_prefs.get("detail_level", "unknown"),
                        "tone": "unknown",
                    }
                }
            profile.setdefault("global_preferences", {"patterns": profile.get("patterns", [])})

        profile.setdefault("preferences", [])
        profile.setdefault("global_preferences", {})
        profile.setdefault("contextual_profiles", {})
        profile.setdefault("reflected_at", None)
        profile.setdefault("signal_count", 0)

        # Migrate legacy free-text lists to hybrid fact objects
        for item in profile.pop("likes", []):
            if isinstance(item, str) and item:
                profile["preferences"].append({
                    "subject": item, "subject_type": "behavior", "relation": "prefers",
                    "value": "prefer", "natural_text": item, "preference_type": "contextual",
                    "confidence": 0.6, "valid_from": time.time(), "decay_rate": 0.02, "recall_count": 0,
                })
        for item in profile.pop("dislikes", []):
            if isinstance(item, str) and item:
                profile["preferences"].append({
                    "subject": item, "subject_type": "behavior", "relation": "dislikes",
                    "value": "avoid", "natural_text": item, "preference_type": "contextual",
                    "confidence": 0.6, "valid_from": time.time(), "decay_rate": 0.02, "recall_count": 0,
                })
        for item in profile.pop("personality_expectations", []):
            if isinstance(item, str) and item:
                profile["preferences"].append({
                    "subject": item, "subject_type": "tone", "relation": "expects",
                    "value": "expect", "natural_text": item, "preference_type": "contextual",
                    "confidence": 0.6, "valid_from": time.time(), "decay_rate": 0.02, "recall_count": 0,
                })

        profile.pop("confidence_scores", None)
        return profile
