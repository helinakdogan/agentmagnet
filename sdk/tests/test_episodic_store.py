"""
Tests for EpisodicStore (Layer 2 — Episodic Memory).

Scenarios:
  - Importance < 0.7 → not stored
  - Importance >= 0.7 → stored in Redis fallback
  - recall() returns episodes with the highest importance
  - _auto_summarize concatenates the latest user messages
  - In-memory fallback is used when Redis and Qdrant are absent
"""

from __future__ import annotations

import json
import time
import unittest
from unittest.mock import MagicMock, patch

from magnet.episodic_store import EpisodicStore, _EPISODIC_KEY_PREFIX, _EPISODE_TTL, _MAX_EPISODES


def _make_messages(n: int = 4) -> list[dict]:
    msgs = []
    for i in range(n):
        msgs.append({"role": "user", "content": f"User message {i}"})
        msgs.append({"role": "assistant", "content": f"Assistant reply {i}"})
    return msgs


class TestEpisodicStoreImportanceFiler(unittest.TestCase):
    """Episodes with low importance should not be stored."""

    def test_low_importance_not_stored(self):
        redis_mock = MagicMock()
        store = EpisodicStore(redis_client=redis_mock)
        store.store_episode("tenant:user", _make_messages(), importance=0.5)
        redis_mock.zadd.assert_not_called()

    def test_boundary_importance_not_stored(self):
        redis_mock = MagicMock()
        store = EpisodicStore(redis_client=redis_mock)
        store.store_episode("tenant:user", _make_messages(), importance=0.69)
        redis_mock.zadd.assert_not_called()


class TestEpisodicStoreRedisBackend(unittest.TestCase):
    """Tests for the Redis fallback backend."""

    def _make_store(self):
        redis_mock = MagicMock()
        store = EpisodicStore(redis_client=redis_mock)
        return store, redis_mock

    def test_high_importance_stored_to_redis(self):
        store, redis_mock = self._make_store()
        store.store_episode("proj:user1", _make_messages(), importance=0.8)
        redis_mock.zadd.assert_called_once()
        call_args = redis_mock.zadd.call_args
        key = call_args[0][0]
        self.assertEqual(key, f"{_EPISODIC_KEY_PREFIX}proj:user1")

    def test_redis_expire_set(self):
        store, redis_mock = self._make_store()
        store.store_episode("proj:user1", _make_messages(), importance=0.9)
        redis_mock.expire.assert_called_once_with(
            f"{_EPISODIC_KEY_PREFIX}proj:user1", _EPISODE_TTL
        )

    def test_redis_trimmed_to_max_episodes(self):
        store, redis_mock = self._make_store()
        store.store_episode("proj:user1", _make_messages(), importance=0.8)
        redis_mock.zremrangebyrank.assert_called_once_with(
            f"{_EPISODIC_KEY_PREFIX}proj:user1", 0, -(_MAX_EPISODES + 1)
        )

    def test_recall_redis_returns_parsed_episodes(self):
        store, redis_mock = self._make_store()
        episode = {
            "tenant_id": "proj:user1",
            "summary": "test summary",
            "importance": 0.8,
            "stored_at": time.time(),
            "messages": [],
        }
        redis_mock.zrevrange.return_value = [json.dumps(episode).encode()]
        result = store.recall("proj:user1", "test query", top_k=3)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["summary"], "test summary")

    def test_recall_redis_respects_top_k(self):
        store, redis_mock = self._make_store()
        redis_mock.zrevrange.return_value = []
        store.recall("proj:user1", "query", top_k=2)
        redis_mock.zrevrange.assert_called_once_with(
            f"{_EPISODIC_KEY_PREFIX}proj:user1", 0, 1
        )

    def test_recall_returns_empty_when_no_redis(self):
        store = EpisodicStore()  # No Redis or Qdrant
        result = store.recall("proj:user1", "query")
        self.assertEqual(result, [])


class TestEpisodicStoreMemoryBackend(unittest.TestCase):
    """Tests for the in-memory fallback backend (no Redis or Qdrant)."""

    def test_store_and_recall_memory_backend(self):
        store = EpisodicStore()  # Neither Redis nor Qdrant
        msgs = _make_messages(5)
        store.store_episode("proj:u1", msgs, importance=0.9)
        results = store.recall("proj:u1", "test", top_k=3)
        self.assertEqual(len(results), 1)
        self.assertAlmostEqual(results[0]["importance"], 0.9)

    def test_memory_sorted_by_importance(self):
        store = EpisodicStore()
        store.store_episode("p:u", _make_messages(), importance=0.7, summary="low")
        store.store_episode("p:u", _make_messages(), importance=0.95, summary="high")
        results = store.recall("p:u", "q", top_k=1)
        self.assertEqual(results[0]["summary"], "high")

    def test_low_importance_not_stored_memory(self):
        store = EpisodicStore()
        store.store_episode("p:u", _make_messages(), importance=0.3)
        results = store.recall("p:u", "q", top_k=5)
        self.assertEqual(results, [])


class TestAutoSummarize(unittest.TestCase):
    """Tests for the _auto_summarize method."""

    def test_summarize_last_two_user_messages(self):
        store = EpisodicStore()
        msgs = [
            {"role": "user", "content": "First question"},
            {"role": "assistant", "content": "Answer"},
            {"role": "user", "content": "Second question"},
            {"role": "assistant", "content": "Answer 2"},
            {"role": "user", "content": "Third question"},
        ]
        summary = store._auto_summarize(msgs)
        self.assertIn("Second question", summary)
        self.assertIn("Third question", summary)
        self.assertNotIn("First question", summary)

    def test_summarize_truncates_at_300(self):
        store = EpisodicStore()
        long_msg = "A" * 400
        msgs = [{"role": "user", "content": long_msg}]
        summary = store._auto_summarize(msgs)
        self.assertLessEqual(len(summary), 300)

    def test_summarize_empty_messages(self):
        store = EpisodicStore()
        summary = store._auto_summarize([])
        self.assertEqual(summary, "")

    def test_summarize_only_assistant_messages(self):
        store = EpisodicStore()
        msgs = [{"role": "assistant", "content": "Only assistant"}]
        summary = store._auto_summarize(msgs)
        self.assertEqual(summary, "")


class TestEpisodicStoreCustomSummary(unittest.TestCase):
    """Tests that a provided summary overrides _auto_summarize."""

    def test_custom_summary_used_when_provided(self):
        store = EpisodicStore()
        msgs = _make_messages(4)
        store.store_episode(
            "p:u", msgs, summary="Custom summary", importance=0.8
        )
        results = store.recall("p:u", "q", top_k=1)
        self.assertEqual(results[0]["summary"], "Custom summary")


if __name__ == "__main__":
    unittest.main()
