import unittest
from magnet import classifier as c_module
from magnet.classifier import IntelligentClassifier


class MockLLMResponse:
    def __init__(self, content):
        self.choices = [type("obj", (object,), {"message": type("obj", (object,), {"content": content})})]


class TestIntelligentClassifier(unittest.TestCase):
    def setUp(self):
        self._original_completion = c_module.litellm.completion

    def tearDown(self):
        c_module.litellm.completion = self._original_completion

    def test_llm_classification(self):
        mock_resp = '{"signal_type": "clarification", "dimension": "detail_level", "query_complexity": "complex", "confidence": 0.9, "reasoning": "Test"}'

        def fake_completion(**kwargs):
            return MockLLMResponse(mock_resp)

        c_module.litellm.completion = fake_completion
        classifier = IntelligentClassifier()
        res = classifier.classify([], "Could you explain this in more detail?")
        self.assertEqual(res.signal_type, "clarification")
        self.assertEqual(res.dimension, "detail_level")
        self.assertEqual(res.query_complexity, "complex")

    def test_regex_fallback_correction(self):
        def failing_completion(**kwargs):
            raise Exception("API Timeout")

        c_module.litellm.completion = failing_completion
        classifier = IntelligentClassifier(fallback_rules=True)
        res = classifier.classify([], "No, that's not what I meant.")
        self.assertEqual(res.signal_type, "correction")


if __name__ == "__main__":
    unittest.main()
