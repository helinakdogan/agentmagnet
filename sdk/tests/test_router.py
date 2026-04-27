import unittest
from magnet.router import ModelRouter


class TestModelRouter(unittest.TestCase):
    def setUp(self):
        self.config = {
            "simple": {"openai": "openai/gpt-4o-mini", "anthropic": "anthropic/claude-haiku-4-5"},
            "medium": {"openai": "openai/gpt-4o", "anthropic": "anthropic/claude-3-5-sonnet"},
            "complex": {"openai": "openai/gpt-4o", "anthropic": "anthropic/claude-3-opus"},
            "preferred_provider": "openai",
        }
        self.router = ModelRouter(self.config)

    def test_base_routing(self):
        decision = self.router.route("medium", None)
        self.assertEqual(decision.provider, "openai")
        self.assertEqual(decision.model, "openai/gpt-4o")
        self.assertEqual(decision.estimated_cost_tier, "medium")

    def test_provider_override(self):
        profile = {"preferences": {"preferred_provider": "anthropic"}}
        decision = self.router.route("medium", profile)
        self.assertEqual(decision.provider, "anthropic")
        self.assertEqual(decision.model, "anthropic/claude-3-5-sonnet")


if __name__ == "__main__":
    unittest.main()
