"""Completion-policy fixtures are not actual model experiments."""
import copy
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from fire_detection.training import completion_summary, validate_completion


class TrainingCompletionTests(unittest.TestCase):
    settings = {"epochs": 60, "patience": 15}

    def trainer(self, count=45, best=30, patience=15, stop=True):
        return SimpleNamespace(epoch=count - 1, stop=stop,
                               stopper=SimpleNamespace(best_epoch=best, best_fitness=.4, patience=patience))

    def test_historical_full_budget_e0_and_e1_remain_valid(self):
        for budget in (50, 100):
            validate_completion({"status": "completed", "epochs_completed": budget},
                                {"epochs": budget, "patience": 0})

    def test_confirmed_plateau_is_a_successful_early_stop(self):
        outcome = completion_summary(self.trainer(), self.settings)
        self.assertEqual(outcome["reason"], "early_stopping")
        self.assertEqual(outcome["epochs_completed"], 45)

    def test_epoch_limit_is_not_labelled_early_stopping(self):
        self.assertEqual(completion_summary(self.trainer(count=60, best=55), self.settings)["reason"], "epoch_limit")

    def test_short_run_without_evidence_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "evidence mismatch"):
            validate_completion({"status": "completed", "epochs_completed": 45}, self.settings)

    def test_arbitrary_stop_before_patience_is_rejected(self):
        for trainer in (self.trainer(best=35), self.trainer(stop=False), self.trainer(patience=10)):
            with self.subTest(trainer=trainer), self.assertRaisesRegex(ValueError, "early stopping evidence"):
                completion_summary(trainer, self.settings)

    def test_interruption_and_oom_are_never_successful_completion(self):
        for status in ("running", "interrupted", "failed"):
            with self.subTest(status=status), self.assertRaisesRegex(ValueError, "successfully"):
                validate_completion({"status": status, "epochs_completed": 60}, self.settings)

    def test_tampered_completion_and_epoch_counts_are_rejected(self):
        outcome = completion_summary(self.trainer(), self.settings)
        for key, value in (("requested_epochs", 100), ("patience", 10), ("epochs_completed", 44),
                           ("best_fitness", float("nan")), ("reason", "time_limit"),
                           ("epochs_without_improvement", 100)):
            bad = copy.deepcopy(outcome)
            bad[key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                validate_completion({"status": "completed", "epochs_completed": 45, "completion": bad}, self.settings)
        for count in (0, 61, None):
            with self.subTest(count=count), self.assertRaises(ValueError):
                validate_completion({"status": "completed", "epochs_completed": count}, self.settings)

    def test_disabled_patience_cannot_accept_early_stopping(self):
        with self.assertRaises(ValueError):
            completion_summary(self.trainer(patience=float("inf")), {"epochs": 60, "patience": 0})


if __name__ == "__main__":
    unittest.main()
