import unittest
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from magnet.buffer import SignalBuffer


class TestSignalBuffer(unittest.TestCase):
    def setUp(self):
        self.buffer = SignalBuffer(redis_client=None, threshold=3)

    def test_push_and_count(self):
        count = self.buffer.push("user1", [{"type": "correction"}])
        self.assertEqual(count, 1)

    def test_should_reflect_false_below_threshold(self):
        self.buffer.push("user2", [{"type": "correction"}])
        self.assertFalse(self.buffer.should_reflect("user2"))

    def test_should_reflect_true_at_threshold(self):
        signals = [{"type": "correction"}] * 3
        self.buffer.push("user3", signals)
        self.assertTrue(self.buffer.should_reflect("user3"))

    def test_flush_clears_buffer(self):
        self.buffer.push("user4", [{"type": "rejection"}, {"type": "correction"}])
        flushed = self.buffer.flush("user4")
        self.assertEqual(len(flushed), 2)
        self.assertFalse(self.buffer.should_reflect("user4"))

    def test_peek_does_not_clear(self):
        self.buffer.push("user5", [{"type": "correction"}])
        self.buffer.peek("user5")
        remaining = self.buffer.peek("user5")
        self.assertEqual(len(remaining), 1)

    def test_empty_signals_ignored(self):
        count = self.buffer.push("user6", [])
        self.assertEqual(count, 0)


if __name__ == "__main__":
    unittest.main()
