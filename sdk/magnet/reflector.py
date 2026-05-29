"""
Reflector
---------
Extracts a behavioral profile from accumulated signals.

This module uses an LLM to analyze a buffer of behavioral signals and
synthesizes them into a structured, stateful user profile. The profile
now includes semantic likes, dislikes, and personality expectations —
not just formatting preferences.
"""

from __future__ import annotations
import json
import logging
import time
import litellm  # type: ignore

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT = """You are a behavioral analyst AI.
You will analyze behavioral signals from a user's interaction with an AI agent.
Your goal is to extract a structured user preference profile from these signals.

OUTPUT FORMAT (only JSON, write nothing else):
{
  "global_preferences": {
    "language": {"value": "turkish", "confidence": 0.95},
    "formatting": {"value": "markdown", "confidence": 0.80}
  },
  "contextual_profiles": {
    "react_development": {
      "response_length": {"value": "short", "confidence": 0.9}
    }
  },
  "likes": [
    "Markdown-formatted responses",
    "Short and concise explanations"
  ],
  "dislikes": [
    "Red color",
    "Long, repetitive explanations",
    "Formal language"
  ],
  "personality_expectations": [
    "Friendly and witty tone",
    "Direct answers without unnecessary preamble"
  ]
}

RULES:
- `global_preferences`: For universal preferences (language, formatting, response style).
- `contextual_profiles`: Group related topics into broad domains. Reuse existing contexts if possible.
- `likes`: Free-text list of things the user explicitly likes or approves of. Be specific.
- `dislikes`: Free-text list of things the user dislikes, wants to avoid, or complained about. Be specific.
  Include colors, topics, styles, tones, behaviors — anything the user expressed dislike for.
- `personality_expectations`: How the user wants the AI to behave or communicate.
- Confidence: 1 signal=0.40, 2 signals=0.60, 3 signals=0.75, 5+ signals=0.90+
- Never hallucinate. Only extract what's actually in the signals.
- Always output valid JSON with no extra text.
- If a signal has `extracted_preference` field, use it verbatim in the appropriate list.
"""

_USER_TEMPLATE = """User ID: {user_id}
Existing Contexts: {existing_contexts}
Existing Likes: {existing_likes}
Existing Dislikes: {existing_dislikes}
Existing Personality Expectations: {existing_personality}
Signal count: {signal_count}

Signals:
{signals_json}

Extract or update the behavioral profile from these signals.
Preserve existing likes/dislikes/personality_expectations and ADD new ones (do not overwrite unless contradicted)."""


class Reflector:
    """
    Analyzes behavioral signals to generate and update a user's profile.

    Profile now contains:
      - global_preferences: formatting, language, response style
      - contextual_profiles: topic-specific adaptations
      - likes: explicit positive preferences (free-text list)
      - dislikes: explicit negative preferences (free-text list)
      - personality_expectations: desired AI behavior/tone (free-text list)
      - confidence_scores: per-preference confidence tracking
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
        Now handles likes, dislikes, and personality_expectations.
        """
        if not signals:
            return existing_profile or self._empty_profile()

        current_state = self._migrate_profile(existing_profile) if existing_profile else self._empty_profile()

        # Time-based confidence decay
        now = time.time()
        last_reflected = current_state.get("reflected_at")
        if last_reflected:
            days_elapsed = (now - last_reflected) / 86400
            decay_factor = max(0.5, 1.0 - (days_elapsed * 0.02))
            for key in current_state.get("confidence_scores", {}):
                current_state["confidence_scores"][key] *= decay_factor

        existing_contexts_list = list(current_state.get("contextual_profiles", {}).keys())
        existing_contexts_str = ", ".join(existing_contexts_list) if existing_contexts_list else "None"

        prompt = _USER_TEMPLATE.format(
            user_id=user_id,
            existing_contexts=existing_contexts_str,
            existing_likes=json.dumps(current_state.get("likes", []), ensure_ascii=False),
            existing_dislikes=json.dumps(current_state.get("dislikes", []), ensure_ascii=False),
            existing_personality=json.dumps(current_state.get("personality_expectations", []), ensure_ascii=False),
            signal_count=len(signals),
            signals_json=json.dumps(signals, ensure_ascii=False, indent=2),
        )

        try:
            raw = self._call_llm(prompt)
            updates = self._parse(raw)

            is_correction = any(s.get("type") in ("correction", "rejection") for s in signals)
            profile = self._merge_state(current_state, updates, is_correction)

            profile["reflected_at"] = time.time()
            profile["signal_count"] = len(signals)
            logger.info(f"Reflector: profile updated for {user_id} — {len(signals)} signals")
            return profile
        except Exception as e:
            logger.error(f"Reflector error ({user_id}): {e}")
            return current_state

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

        Args:
            user_id: User identifier.
            signal_type: "preference_dislike", "preference_like", "tone_preference", etc.
            extracted_preference: Free-text preference string to store.
            confidence: Signal confidence score.
            existing_profile: Existing profile to update.

        Returns:
            Updated profile dict.
        """
        profile = existing_profile or self._empty_profile()
        profile = self._migrate_profile(profile)

        if not extracted_preference:
            return profile

        if signal_type == "preference_dislike":
            self._upsert_preference(profile.setdefault("dislikes", []), extracted_preference)
            logger.info(f"instant_learn: dislike upserted for {user_id}: {extracted_preference!r}")
        elif signal_type == "preference_like":
            self._upsert_preference(profile.setdefault("likes", []), extracted_preference)
            logger.info(f"instant_learn: like upserted for {user_id}: {extracted_preference!r}")
        elif signal_type in ("tone_preference", "formatting_preference", "detail_preference"):
            self._upsert_preference(profile.setdefault("personality_expectations", []), extracted_preference)
            logger.info(f"instant_learn: personality expectation upserted for {user_id}: {extracted_preference!r}")

        profile["reflected_at"] = time.time()
        return profile

    def build_injection(self, profile: dict, current_context: str = "general_chat") -> str:
        """
        Constructs a rich system prompt injection string from the user profile.
        Now includes likes, dislikes, and personality expectations.
        """
        profile = self._migrate_profile(profile)
        global_prefs = profile.get("global_preferences", {})
        ctx_profile = profile.get("contextual_profiles", {}).get(current_context, {})
        confidence_scores = profile.get("confidence_scores", {})
        likes = profile.get("likes", [])
        dislikes = profile.get("dislikes", [])
        personality = profile.get("personality_expectations", [])

        has_content = (
            global_prefs or ctx_profile or likes or dislikes or personality
        )
        if not has_content:
            return ""

        lines = ["[Behavioral Profile — Learned from this user's behavior]"]

        # Structured preferences
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
                conf = confidence_scores.get(k, v.get("confidence", 0))
                valid_from = v.get("valid_from")
                decay_rate = v.get("decay_rate", 0.02)
                if valid_from:
                    days_elapsed = (time.time() - valid_from) / 86400
                    conf *= (1 - decay_rate) ** days_elapsed
                if conf < 0.1:
                    continue
                pct = int(conf * 100)
                lines.append(f"  - {k}: {v['value']} (confidence: {pct}%)")

        # Likes
        if likes:
            lines.append("\nThis user likes / approves of:")
            for item in likes[:10]:
                lines.append(f"  ✓ {item}")

        # Dislikes — critical for compliance
        if dislikes:
            lines.append("\nThis user DISLIKES — always avoid these:")
            for item in dislikes[:10]:
                lines.append(f"  ✗ {item}")

        # Personality expectations
        if personality:
            lines.append("\nPersonality & communication expectations:")
            for item in personality[:5]:
                lines.append(f"  → {item}")

        # Patterns (legacy)
        patterns = global_prefs.get("patterns", [])
        if isinstance(patterns, list) and patterns:
            lines.append("\nRecent patterns:")
            for p in patterns[:3]:
                lines.append(f"  - {p}")

        lines.append("\nNote: These preferences were learned from user behavior.")
        lines.append("Respect them strictly. Allow user to override at any time.")

        return "\n".join(lines)

    def _upsert_preference(self, pref_list: list, new_item: str) -> None:
        if not pref_list:
            pref_list.append(new_item)
            return
        try:
            logger.debug(f"_upsert_preference: embedding {len(pref_list) + 1} texts for dedup")
            embeddings = self._embed_batch(pref_list + [new_item])
            new_emb = embeddings[-1]
            best_idx, best_sim = -1, 0.0
            for i, emb in enumerate(embeddings[:-1]):
                sim = self._cosine_sim(new_emb, emb)
                if sim > best_sim:
                    best_sim, best_idx = sim, i
            if best_sim > 0.85:
                logger.debug(
                    f"_upsert_preference: dedup hit (sim={best_sim:.3f}), "
                    f"replacing {pref_list[best_idx]!r} with {new_item!r}"
                )
                pref_list[best_idx] = new_item
            else:
                logger.debug(
                    f"_upsert_preference: no dedup match (best_sim={best_sim:.3f}), appending {new_item!r}"
                )
                pref_list.append(new_item)
        except Exception as e:
            logger.warning(
                f"_upsert_preference: embedding call failed, falling back to exact match: {e}",
                exc_info=True,
            )
            if new_item not in pref_list:
                pref_list.append(new_item)

    def _embed_batch(self, texts: list) -> list:
        kwargs: dict = {
            "model": "openai/text-embedding-3-small",
            "input": texts,
        }
        if self._openai_api_key:
            kwargs["api_key"] = self._openai_api_key
        response = litellm.embedding(**kwargs)
        return [
            d["embedding"] if isinstance(d, dict) else d.embedding
            for d in response.data
        ]

    @staticmethod
    def _cosine_sim(a: list, b: list) -> float:
        dot = sum(x * y for x, y in zip(a, b))
        norm_a = sum(x * x for x in a) ** 0.5
        norm_b = sum(x * x for x in b) ** 0.5
        return dot / (norm_a * norm_b) if norm_a and norm_b else 0.0

    def _call_llm(self, prompt: str) -> str:
        api_key = (
            self._openai_api_key if "openai" in self.model
            else self._anthropic_api_key
        ) or None

        response = litellm.completion(
            model=self.model,
            messages=[
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            temperature=0,
            max_tokens=800,
            api_key=api_key,
        )
        return response.choices[0].message.content

    def _parse(self, raw: str) -> dict:
        raw = raw.strip()
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
        return json.loads(raw.strip())

    def _merge_state(self, current: dict, updates: dict, is_correction: bool = False) -> dict:
        """
        Merges new signal-derived updates into the existing profile.
        Lists (likes, dislikes, personality_expectations) are merged additively.
        Structured preferences use confidence-based merging.
        """
        confidence_scores = current.setdefault("confidence_scores", {})

        # Merge lists with semantic dedup
        for list_key in ("likes", "dislikes", "personality_expectations"):
            existing_list = current.setdefault(list_key, [])
            new_items = updates.get(list_key, [])
            for item in new_items:
                if isinstance(item, str) and item:
                    self._upsert_preference(existing_list, item)

        # Merge structured preferences
        for scope in ["global_preferences", "contextual_profiles"]:
            curr_scope = current.setdefault(scope, {})
            upd_scope = updates.get(scope, {})
            items_to_iterate = upd_scope.items() if scope == "contextual_profiles" else [("general", upd_scope)]

            for ctx_key, ctx_updates in items_to_iterate:
                if scope == "contextual_profiles":
                    target_dict = curr_scope.setdefault(ctx_key, {})
                else:
                    target_dict = curr_scope
                    ctx_updates = upd_scope

                for pref_key, new_data in ctx_updates.items():
                    if not isinstance(new_data, dict) or "value" not in new_data:
                        continue

                    new_val = new_data["value"]
                    new_conf = float(new_data.get("confidence", 0.5))
                    old_data = target_dict.get(pref_key)

                    if not old_data:
                        target_dict[pref_key] = {"value": new_val}
                        confidence_scores[pref_key] = new_conf
                    else:
                        current_conf = confidence_scores.get(pref_key, 0.5)
                        if is_correction:
                            confidence_scores[pref_key] = max(0.0, current_conf - 0.2)
                            target_dict[pref_key] = {"value": new_val}
                        else:
                            if new_val == old_data.get("value"):
                                confidence_scores[pref_key] = max(current_conf, new_conf)
                            else:
                                if new_conf > current_conf:
                                    target_dict[pref_key] = {"value": new_val}
                                    confidence_scores[pref_key] = new_conf
                                else:
                                    confidence_scores[pref_key] = max(0.1, current_conf - 0.2)

        return current

    @staticmethod
    def _empty_profile() -> dict:
        return {
            "global_preferences": {},
            "contextual_profiles": {},
            "likes": [],
            "dislikes": [],
            "personality_expectations": [],
            "confidence_scores": {},
            "reflected_at": None,
            "signal_count": 0,
        }

    def _migrate_profile(self, profile: dict) -> dict:
        """Migrates legacy profiles to the new schema with likes/dislikes."""
        # Ensure new fields exist
        profile.setdefault("likes", [])
        profile.setdefault("dislikes", [])
        profile.setdefault("personality_expectations", [])
        profile.setdefault("confidence_scores", {})

        if "contextual_profiles" not in profile:
            prefs = profile.get("preferences", {})
            profile["contextual_profiles"] = {
                "general_chat": {
                    "response_length": prefs.get("response_length", "unknown"),
                    "explanation_depth": prefs.get("detail_level", "unknown"),
                    "tone": "unknown",
                }
            }
            profile.setdefault("global_preferences", {"patterns": profile.get("patterns", [])})
            profile.setdefault("reflected_at", time.time())
            profile.setdefault("signal_count", 0)

        return profile
