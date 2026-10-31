"""Protocol and failure-mode checks; mocks contain no real model results."""
import copy
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from fire_detection import common, evaluation
from fire_detection.analysis import compare_runs
from fire_detection.common import ROOT, file_sha256, read_json, write_json
from fire_detection.config import experiment_config
from fire_detection.evaluation import evaluate_test, freeze_selection, match_boxes
from fire_detection.model import verify_model


class ProtocolTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(dir=ROOT / "work")
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)

    def test_epoch_lr_and_augmentation_comparisons_are_independent(self):
        path = ROOT / "configs/experiments.json"
        baseline = experiment_config(path, "E0")["training"]
        reference = experiment_config(path, "E1")["training"]
        self.assertEqual([k for k in baseline if baseline[k] != reference[k]], ["epochs"])
        for experiment_id, expected in (("E2-A", {"lr0"}), ("E2-C", {"lr0"}),
                ("E3-HSV", {"hsv_s", "hsv_v"}), ("E3-Mosaic", {"mosaic"}),
                ("E4-800", {"imgsz"}), ("E5-8", {"batch"}), ("E5-32", {"batch"})):
            changed = experiment_config(path, experiment_id)["training"]
            self.assertEqual({k for k in changed if changed[k] != reference[k]}, expected)

    def test_combined_experiment_needs_validation_evidence(self):
        with self.assertRaisesRegex(ValueError, "disabled"):
            experiment_config(ROOT / "configs/experiments.json", "E6")

    def test_auto_optimizer_and_weight_overrides_are_rejected(self):
        source = read_json(ROOT / "configs/experiments.json")
        for override in ({"optimizer": "auto"}, {"model": "best.pt"}, {"batch": -1}, {"patience": 5}, {"amp": True}):
            suite = copy.deepcopy(source)
            suite["experiments"][0]["overrides"] = override
            path = self.root / "suite.json"
            write_json(path, suite)
            with self.subTest(override=override), self.assertRaises(ValueError):
                experiment_config(path, "E0")

    def test_model_checksum_mismatch_fails(self):
        model = self.root / "model.pt"
        model.write_bytes(b"test fixture, not weights")
        with self.assertRaisesRegex(ValueError, "checksum/size mismatch"):
            verify_model(model, {"size_bytes": model.stat().st_size, "sha256": "0" * 64})

    def test_one_prediction_cannot_match_two_targets(self):
        predictions = [[1, 0, 0, 10, 10]]
        targets = [[1, 0, 0, 10, 10], [1, 0, 0, 9, 9]]
        matched_p, matched_t = match_boxes(predictions, targets, .5)
        self.assertEqual(matched_p, {0})
        self.assertEqual(matched_t, {0})

    def test_wrong_class_does_not_match(self):
        self.assertEqual(match_boxes([[0, 0, 0, 10, 10]], [[1, 0, 0, 10, 10]], .5), (set(), set()))

    def test_duplicate_predictions_are_false_positives(self):
        matched_p, matched_t = match_boxes([[0, 0, 0, 10, 10], [0, 0, 0, 10, 10]], [[0, 0, 0, 10, 10]], .5)
        self.assertEqual(len(matched_p), 1)
        self.assertEqual(len(matched_t), 1)

    def validation_fixture(self):
        model = self.root / "fake.pt"
        model.write_bytes(b"fixture, not actual model weights")
        folder = self.root / "validation"
        folder.mkdir()
        record = {"status": "completed", "split": "val", "model_role": "trained", "config": {},
                  "checkpoint": str(model), "checkpoint_sha256": file_sha256(model),
                  "dataset_fingerprint": "fixture", "training_run": "fixture"}
        write_json(folder / "evaluation.json", record)
        return folder

    def test_freeze_and_test_reject_mutated_checkpoint(self):
        folder = self.validation_fixture()
        with patch.object(evaluation, "ROOT", self.root):
            selection = freeze_selection(folder, "Unit-test fixture")
        (self.root / "fake.pt").write_bytes(b"changed")
        with self.assertRaisesRegex(ValueError, "checkpoint checksum mismatch"):
            evaluate_test(selection)

    def test_frozen_threshold_mutation_is_rejected(self):
        folder = self.validation_fixture()
        with patch.object(evaluation, "ROOT", self.root):
            path = freeze_selection(folder, "Unit-test fixture")
        selection = read_json(path)
        selection["config"] = {"evaluation": {"operating_conf": .01}}
        write_json(path, selection)
        with self.assertRaisesRegex(ValueError, "selection checksum mismatch"):
            evaluate_test(path)

    def test_test_cannot_be_used_as_selection_evidence(self):
        folder = self.validation_fixture()
        record = read_json(folder / "evaluation.json")
        record["split"] = "test"
        write_json(folder / "evaluation.json", record)
        with self.assertRaisesRegex(ValueError, "completed validation"):
            freeze_selection(folder, "Fixture")

    def test_test_api_requires_frozen_selection(self):
        with self.assertRaisesRegex(ValueError, "frozen final selection"):
            evaluation.evaluate_checkpoint("unused.pt", {}, "unused", split="test")

    def test_compare_does_not_invent_results(self):
        with self.assertRaisesRegex(ValueError, "No completed"):
            compare_runs([self.root])

    def test_invalidated_evaluation_cannot_be_frozen_or_compared(self):
        folder = self.validation_fixture()
        write_json(folder / "invalidated.json", {"reason": "Unit-test input mutation fixture"})
        with self.assertRaisesRegex(ValueError, "invalidated"):
            freeze_selection(folder, "Fixture")
        with self.assertRaisesRegex(ValueError, "No completed"):
            compare_runs([folder])


if __name__ == "__main__":
    unittest.main()
