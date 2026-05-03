"""
Reflector
---------
Extracts a behavioral profile from accumulated signals.

This module uses an LLM to analyze a buffer of behavioral signals (e.g.,
corrections, rejections) and synthesizes them into a structured, stateful
user profile that can be used to guide future AI interactions.
"""

from __future__ import annotations
import json
import logging
import time
import litellm

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT = """You are a behavioral analyst AI.
You will analyze behavioral signals from a user's interaction with an AI agent.
Your goal is to extract ONLY the new preferences indicated by the signals in JSON format.

OUTPUT FORMAT (only JSON, write nothing else):
{
  "global_preferences": {
    "language": {"value": "english", "confidence": 0.95},
    "formatting": {"value": "markdown", "confidence": 0.80}
  },
  "contextual_profiles": {
    "react_development": {
      "response_length": {"value": "short", "confidence": 0.9}
    }
  }
}

RULES:
- Provide a value (string) and confidence (float between 0.0 - 1.0) for each preference.
- Her preference için 0.0-1.0 arası confidence score ver. 1 sinyal = 0.3, 3 sinyal = 0.6, 5+ sinyal = 0.85+
- Only extract preferences explicitly stated in the signal. Do not add contexts and preferences not present in the signal.
- For `contextual_profiles`, group related topics into cohesive, broad technical/business domains (e.g., use 'react_development' instead of fragmenting into 'react_hooks' or 'react_components').
- REUSE "Existing Contexts" provided below if the new signals fit into them. Only create a new key if the topic is entirely different.
- Never guess or hallucinate.
"""

_USER_TEMPLATE = """User ID: {user_id}
Existing Contexts: {existing_contexts}
Signal count: {signal_count}
Signals:
{signals_json}

Extract the behavioral profile from these signals."""


class Reflector:
    """
    Analyzes behavioral signals to generate and update a user's profile.
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
        """
        if not signals:
            return existing_profile or self._empty_profile()

        current_state = self._migrate_profile(existing_profile) if existing_profile else self._empty_profile()

        now = time.time()
        last_reflected = current_state.get("reflected_at")
        if last_reflected:
            days_elapsed = (now - last_reflected) / 86400
            decay_factor = max(0.5, 1.0 - (days_elapsed * 0.02))  # günde %2 düşüş
            for key in current_state.get("confidence_scores", {}):
                current_state["confidence_scores"][key] *= decay_factor

        existing_contexts_list = list(current_state.get("contextual_profiles", {}).keys())
        existing_contexts_str = ", ".join(existing_contexts_list) if existing_contexts_list else "None"

        prompt = _USER_TEMPLATE.format(
            user_id=user_id,
            existing_contexts=existing_contexts_str,
            signal_count=len(signals),
            signals_json=json.dumps(signals, ensure_ascii=False, indent=2),
        )

        try:
            raw = self._call_llm(prompt)
            updates = self._parse(raw)
            
            # Count corrections from the new signals to penalize confidence if needed
            is_correction = any(s.get("type") in ("correction", "rejection") for s in signals)
            
            profile = self._merge_state(current_state, updates, is_correction)
            
            profile["reflected_at"] = time.time()
            profile["signal_count"] = len(signals)
            logger.info(f"Reflector: profile created for {user_id} — {len(signals)} signals")
            return profile
        except Exception as e:
            logger.error(f"Reflector error ({user_id}): {e}")
            return current_state

    def build_injection(self, profile: dict, current_context: str = "general_chat") -> str:
        """
        Constructs a rich system prompt injection string from the user profile.
        """
        profile = self._migrate_profile(profile)
        global_prefs = profile.get("global_preferences", {})
        ctx_profile = profile.get("contextual_profiles", {}).get(current_context, {})
        confidence_scores = profile.get("confidence_scores", {})

        if not global_prefs and not ctx_profile:
            return ""

        lines = ["[Behavioral Profile]"]

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
                pct = int(conf * 100)
                lines.append(f"  - {k}: {v['value']} (confidence: {pct}%)")

        patterns = global_prefs.get("patterns", [])
        if isinstance(patterns, list) and patterns:
            lines.append("\nRecent patterns:")
            for p in patterns[:3]:
                lines.append(f"  - {p}")

        lines.append("\nNote: These preferences were learned from user behavior, not explicit instructions.")
        lines.append("Respect them but allow the user to override at any time.")

        return "\n".join(lines)

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
            max_tokens=500,
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
        Eğer existing_profile varsa:
        Yeni sinyaller existing_profile'ı tamamen ezmez.
        Bunun yerine merge et:
        - Aynı preference key varsa: confidence = max(mevcut, yeni)
        - Yeni key ise: ekle
        - Düzeltme sinyali ise: ilgili key'in confidence'ını düşür (-0.2)
        """
        confidence_scores = current.setdefault("confidence_scores", {})
        
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
                        mevcut_conf = confidence_scores.get(pref_key, 0.5)
                        
                        if is_correction:
                            confidence_scores[pref_key] = max(0.0, mevcut_conf - 0.2)
                            # Overwrite value based on new correction
                            target_dict[pref_key] = {"value": new_val}
                        else:
                            if new_val == old_data.get("value"):
                                confidence_scores[pref_key] = max(mevcut_conf, new_conf)
                            else:
                                if new_conf > mevcut_conf:
                                    target_dict[pref_key] = {"value": new_val}
                                    confidence_scores[pref_key] = new_conf
                                else:
                                    confidence_scores[pref_key] = max(0.1, mevcut_conf - 0.2)
                                    
        return current

    @staticmethod
    def _empty_profile() -> dict:
        return {
            "global_preferences": {},
            "contextual_profiles": {},
            "confidence_scores": {},
            "reflected_at": None,
            "signal_count": 0,
        }

    def _migrate_profile(self, profile: dict) -> dict:
        if "contextual_profiles" in profile:
            if "confidence_scores" not in profile:
                profile["confidence_scores"] = {}
            return profile
            
        prefs = profile.get("preferences", {})
        return {
            "global_preferences": {
                "patterns": profile.get("patterns", [])
            },
            "contextual_profiles": {
                "general_chat": {
                    "response_length": prefs.get("response_length", "unknown"),
                    "explanation_depth": prefs.get("detail_level", "unknown"),
                    "tone": "unknown"
                }
            },
            "confidence_scores": {},
            "reflected_at": profile.get("reflected_at", time.time()),
            "signal_count": profile.get("signal_count", 0)
        }
