import unittest

from rlkb.decider.backends import MockKeyPressBackend
from rlkb.decider.engine import DeciderEngine
from rlkb.decider.layouts import DEFAULT_KEY_LAYOUT
from rlkb.decider.strategy import build_default_registry
from rlkb.scripts.run_decider import parse_verification_plan


class DeciderEngineTest(unittest.TestCase):
    def setUp(self):
        self.strategy = build_default_registry().get("sequence")

    def test_sequence_code_completes(self):
        backend = MockKeyPressBackend()
        engine = DeciderEngine(
            code="1230",
            strategy=self.strategy,
            backend=backend,
            layout=DEFAULT_KEY_LAYOUT,
            max_retries=1,
        )

        result = engine.run()

        self.assertTrue(result.success)
        self.assertEqual(result.final_state, "completed")
        self.assertEqual([step.target.key_id for step in result.completed_steps], ["1", "2", "3", "0"])

    def test_failed_verification_respects_retry_budget(self):
        backend = MockKeyPressBackend(verification_plan={"2": [False, False]})
        engine = DeciderEngine(
            code="12",
            strategy=self.strategy,
            backend=backend,
            layout=DEFAULT_KEY_LAYOUT,
            max_retries=1,
        )

        result = engine.run()

        self.assertFalse(result.success)
        self.assertEqual(result.final_state, "failed")
        self.assertEqual(result.failed_step.target.key_id, "2")
        self.assertEqual(result.failed_step.attempts, 2)

    def test_parse_verification_plan(self):
        parsed = parse_verification_plan("2:false,true;5:false")
        self.assertEqual(parsed, {"2": [False, True], "5": [False]})

    def test_non_digit_code_is_rejected(self):
        backend = MockKeyPressBackend()
        engine = DeciderEngine(
            code="12A",
            strategy=self.strategy,
            backend=backend,
            layout=DEFAULT_KEY_LAYOUT,
            max_retries=1,
        )

        with self.assertRaisesRegex(ValueError, "only digits 0-9"):
            engine.run()


if __name__ == "__main__":
    unittest.main()
