import unittest
from magnet.router import ModelRouter


class TestModelRouter(unittest.TestCase):
    def setUp(self):
        self.router = ModelRouter(
            cheap_model="openai/gpt-4o-mini",
            expensive_model="openai/gpt-4o",
        )

    def test_short_simple_message_routes_to_cheap_model(self):
        messages = [{"role": "user", "content": "hi"}]
        decision = self.router.route(messages, None)
        self.assertEqual(decision.selected_model, "openai/gpt-4o-mini")
        self.assertEqual(decision.estimated_complexity, "simple")
        self.assertEqual(decision.cost_tier, "cheap")

    def test_long_complex_message_routes_to_expensive_model(self):
        content = (
            "detailed analysis, architecture comparison, debug, "
            "production-ready optimization, comprehensive design review: " + "x" * 600
        )
        messages = [{"role": "user", "content": content}]
        decision = self.router.route(messages, None)
        self.assertEqual(decision.selected_model, "openai/gpt-4o")
        self.assertEqual(decision.estimated_complexity, "complex")
        self.assertEqual(decision.cost_tier, "expensive")


if __name__ == "__main__":
    unittest.main()
