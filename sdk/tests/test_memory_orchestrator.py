"""
Tests for MemoryOrchestrator.

Scenarios:
  - decide(): Correctly detects episodic trigger patterns (TR + EN)
  - decide(): Returns episodic=False if there is no trigger
  - decide(): Behavioral is always True, knowledge is always False
  - build_context(): Returns only behavioral context when there is no trigger
  - build_context(): Appends episodic context when a trigger is matched
  - build_context(): Returns an empty string if no profile exists
  - should_store_episode(): Long conversations increase importance
  - should_store_episode(): Correction signals increase importance
  - should_store_episode(): Short conversations with no signals remain below 0.7
  - should_store_episode(): Importance score does not exceed 1.0
"""

from __future__ import annotations

import unittest
from unittest.mock import MagicMock, patch

from magnet.memory_orchestrator import MemoryOrchestrator
from magnet.episodic_store import EpisodicStore
from magnet.knowledge_store import KnowledgeStore
from magnet.store import ProfileStore


def _make_orchestrator(
    profile: dict | None = None,
    episodes: list[dict] | None = None,
) -> MemoryOrchestrator:
    """Helper: creates a MemoryOrchestrator with mocks."""
    behavioral = MagicMock(spec=ProfileStore)
    behavioral.load.return_value = profile

    episodic = MagicMock(spec=EpisodicStore)
    episodic.recall.return_value = episodes or []

    knowledge = MagicMock(spec=KnowledgeStore)

    return MemoryOrchestrator(
        behavioral_store=behavioral,
        episodic_store=episodic,
        knowledge_store=knowledge,
    )


class TestDecide(unittest.TestCase):
    """Tests for the decide() method."""

    def test_behavioral_always_true(self):
        orc = _make_orchestrator()
        result = orc.decide("Merhaba", "t:u")
        self.assertTrue(result["behavioral"])

    def test_knowledge_always_false(self):
        orc = _make_orchestrator()
        result = orc.decide("herhangi bir şey", "t:u")
        self.assertFalse(result["knowledge"])

    def test_no_trigger_episodic_false(self):
        orc = _make_orchestrator()
        result = orc.decide("Bugün hava nasıl?", "t:u")
        self.assertFalse(result["episodic"])

    # ── Turkish triggers ─────────────────────────────────────────────
    def test_trigger_gecen_tr(self):
        orc = _make_orchestrator()
        self.assertTrue(orc.decide("geçen konuşmamızda ne dedik?", "t:u")["episodic"])

    def test_trigger_daha_once_tr(self):
        orc = _make_orchestrator()
        self.assertTrue(orc.decide("daha önce söylediğin neydi?", "t:u")["episodic"])

    def test_trigger_hatirliyor_musun_tr(self):
        orc = _make_orchestrator()
        self.assertTrue(orc.decide("hatırlıyor musun bunu?", "t:u")["episodic"])

    def test_trigger_gecen_sefer_tr(self):
        orc = _make_orchestrator()
        self.assertTrue(orc.decide("geçen sefer farklı yaptın", "t:u")["episodic"])

    def test_trigger_konusmuştuk_tr(self):
        orc = _make_orchestrator()
        self.assertTrue(orc.decide("önce konuşmuştuk bu konuyu", "t:u")["episodic"])

    # ── English triggers ─────────────────────────────────────────────
    def test_trigger_previously_en(self):
        orc = _make_orchestrator()
        self.assertTrue(orc.decide("previously you said...", "t:u")["episodic"])

    def test_trigger_last_time_en(self):
        orc = _make_orchestrator()
        self.assertTrue(orc.decide("last time we spoke about this", "t:u")["episodic"])

    def test_trigger_you_mentioned_en(self):
        orc = _make_orchestrator()
        self.assertTrue(orc.decide("you mentioned React hooks before", "t:u")["episodic"])

    def test_trigger_we_discussed_en(self):
        orc = _make_orchestrator()
        self.assertTrue(orc.decide("we discussed this topic last week", "t:u")["episodic"])

    def test_trigger_recently_en(self):
        orc = _make_orchestrator()
        self.assertTrue(orc.decide("recently you helped me with SQL", "t:u")["episodic"])

    def test_trigger_do_you_remember_en(self):
        orc = _make_orchestrator()
        self.assertTrue(orc.decide("do you remember what we decided?", "t:u")["episodic"])

    def test_case_insensitive(self):
        orc = _make_orchestrator()
        self.assertTrue(orc.decide("PREVIOUSLY I ASKED YOU...", "t:u")["episodic"])


class TestBuildContext(unittest.TestCase):
    """Tests for the build_context() method."""

    def test_no_profile_returns_empty(self):
        orc = _make_orchestrator(profile=None)
        ctx = orc.build_context("test query", "t:u")
        self.assertEqual(ctx, "")

    def test_behavioral_context_included_when_profile_exists(self):
        profile = {
            "global_preferences": {
                "language": {"value": "turkish", "confidence": 0.9}
            },
            "contextual_profiles": {},
            "confidence_scores": {"language": 0.9},
            "reflected_at": None,
            "signal_count": 3,
        }
        orc = _make_orchestrator(profile=profile)
        ctx = orc.build_context("normal soru", "t:u")
        self.assertIn("Behavioral Profile", ctx)

    def test_episodic_context_not_included_without_trigger(self):
        profile = {
            "global_preferences": {"tone": {"value": "formal", "confidence": 0.8}},
            "contextual_profiles": {},
            "confidence_scores": {},
            "reflected_at": None,
            "signal_count": 1,
        }
        episodes = [{"summary": "Past episode summary", "importance": 0.9}]
        orc = _make_orchestrator(profile=profile, episodes=episodes)
        ctx = orc.build_context("Bugün ne yapmalıyım?", "t:u")
        self.assertNotIn("Past Conversations", ctx)
        self.assertNotIn("Past episode summary", ctx)

    def test_episodic_context_included_with_trigger(self):
        profile = {
            "global_preferences": {"tone": {"value": "formal", "confidence": 0.8}},
            "contextual_profiles": {},
            "confidence_scores": {},
            "reflected_at": None,
            "signal_count": 1,
        }
        episodes = [{"summary": "We talked about Python", "importance": 0.9}]
        orc = _make_orchestrator(profile=profile, episodes=episodes)
        ctx = orc.build_context("geçen konuştuğumuzu hatırlıyor musun?", "t:u")
        self.assertIn("Past Conversations", ctx)
        self.assertIn("We talked about Python", ctx)

    def test_episodic_not_added_if_no_episodes_returned(self):
        profile = {
            "global_preferences": {"language": {"value": "en", "confidence": 0.7}},
            "contextual_profiles": {},
            "confidence_scores": {},
            "reflected_at": None,
            "signal_count": 1,
        }
        orc = _make_orchestrator(profile=profile, episodes=[])
        ctx = orc.build_context("previously you told me...", "t:u")
        self.assertNotIn("Past Conversations", ctx)


class TestShouldStoreEpisode(unittest.TestCase):
    """Tests for the should_store_episode() method."""

    def _orc(self) -> MemoryOrchestrator:
        return _make_orchestrator()

    def test_baseline_importance(self):
        orc = self._orc()
        score = orc.should_store_episode([], [])
        self.assertAlmostEqual(score, 0.3)

    def test_long_conversation_bonus(self):
        orc = self._orc()
        msgs = [{"role": "user", "content": str(i)} for i in range(8)]
        score = orc.should_store_episode(msgs, [])
        self.assertGreaterEqual(score, 0.5)  # 0.3 + 0.2

    def test_correction_signal_bonus(self):
        orc = self._orc()
        signals = [{"type": "correction"}]
        score = orc.should_store_episode([], signals)
        self.assertAlmostEqual(score, 0.45)  # 0.3 + 0.15

    def test_multiple_corrections_capped(self):
        orc = self._orc()
        signals = [{"type": "correction"}] * 5
        score = orc.should_store_episode([], signals)
        # max correction bonus = 0.4 → 0.3 + 0.4 = 0.7
        self.assertAlmostEqual(score, 0.7)

    def test_rejection_counts_as_correction(self):
        orc = self._orc()
        signals = [{"type": "rejection"}]
        score = orc.should_store_episode([], signals)
        self.assertAlmostEqual(score, 0.45)

    def test_preference_signal_bonus(self):
        orc = self._orc()
        signals = [{"type": "preference"}]
        score = orc.should_store_episode([], signals)
        self.assertAlmostEqual(score, 0.35)  # 0.3 + 0.05

    def test_long_conversation_with_corrections(self):
        orc = self._orc()
        msgs = [{"role": "user", "content": str(i)} for i in range(10)]
        signals = [{"type": "correction"}, {"type": "correction"}]
        score = orc.should_store_episode(msgs, signals)
        # 0.3 + 0.2 + 0.3 = 0.8
        self.assertAlmostEqual(score, 0.8)

    def test_score_never_exceeds_1(self):
        orc = self._orc()
        msgs = [{"role": "user", "content": str(i)} for i in range(20)]
        signals = [{"type": "correction"}] * 10
        score = orc.should_store_episode(msgs, signals)
        self.assertLessEqual(score, 1.0)

    def test_short_conversation_no_signals_below_threshold(self):
        orc = self._orc()
        msgs = [{"role": "user", "content": "Hi"}, {"role": "assistant", "content": "Hello"}]
        score = orc.should_store_episode(msgs, [])
        self.assertLess(score, 0.7)


if __name__ == "__main__":
    unittest.main()
