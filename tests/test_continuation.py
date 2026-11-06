"""Archived E6 migration fixtures never launch a real model experiment."""
import copy
import sys
import tempfile
import unittest
from contextlib import ExitStack, contextmanager
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from fire_detection.common import ROOT, file_sha256, object_sha256, read_json, write_json
if sys.platform == "linux":
    from fire_detection import continuation, queue


@unittest.skipUnless(sys.platform == "linux", "Queue continuation requires Linux flock")
class ContinuationTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(dir=ROOT / "work")
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.archive = self.root / "work/archive"
        self.archive.mkdir(parents=True)
        source = self.archive / "sources/configs/frozen.json"
        source.parent.mkdir(parents=True)
        source.write_bytes(b"Old immutable source fixture")
        configs, tasks, anchors = {}, [], []
        for key in queue.CORE_IDS:
            training = {"epochs": 100 if key == "E1" else 60,
                        "patience": 0 if key == "E1" else 15,
                        "imgsz": 800 if key == "E4-800" else 640}
            configs[key] = {"training": training}
            train, val = self.root / "runs" / key, self.root / "runs" / (key + "-val")
            write_json(train / "training.json", {"fixture": key})
            write_json(val / "evaluation.json", {"fixture": key, "split": "val"})
            tasks.append({"id": key, "status": "completed", "training_run": str(train),
                          "validation_run": str(val), "workers": [], "last_completed_epoch": training["epochs"]})
            anchors.append({"id": key, "training_run": str(train), "validation_run": str(val),
                            "training_record_sha256": file_sha256(train / "training.json"),
                            "validation_record_sha256": file_sha256(val / "evaluation.json"),
                            "best_sha256": "fixture-model-" + key})
        self.core_plan = {"ids": list(queue.CORE_IDS), "configs": configs, "dataset_fingerprint": "fixture",
                          "pinned_files": {"configs/frozen.json": file_sha256(source)}}
        self.core_plan["plan_sha256"] = object_sha256(self.core_plan)
        self.old_state = {"status": "awaiting_analysis", "manager": None, "tasks": tasks,
                          "plan_sha256": self.core_plan["plan_sha256"], "finished_at_utc": "fixture"}
        write_json(self.archive / "state.json", self.old_state)
        write_json(self.archive / "plan.json", self.core_plan)
        self.archive_record = {"kind": "completed_core_queue_archive", "completed": anchors,
                               "previous_plan_sha256": self.core_plan["plan_sha256"],
                               "previous_state_sha256": file_sha256(self.archive / "state.json"),
                               "previous_plan_file_sha256": file_sha256(self.archive / "plan.json"),
                               "files": {str(p.relative_to(self.archive)): file_sha256(p)
                                         for p in self.archive.rglob("*") if p.is_file()}}
        write_json(self.archive / "archive.json", self.archive_record)
        self.digest = file_sha256(self.archive / "archive.json")
        self.state_path = self.root / "work/server-queue.json"
        write_json(self.state_path, self.old_state)
        evidence = [{"experiment_id": a["id"], "validation_run": a["validation_run"],
                     "validation_record_sha256": a["validation_record_sha256"],
                     "training_record_sha256": a["training_record_sha256"],
                     "checkpoint_sha256": a["best_sha256"]}
                    for a in anchors if a["id"] in ("E1", "E4-800")]
        self.selected = {"experiment": {"id": "E6", "selection_rationale": "Fixture selection",
                                        "selection_evidence": evidence},
                         "training": {"epochs": 100, "patience": 0, "imgsz": 800}}

    def extended_plan(self):
        plan = copy.deepcopy(self.core_plan)
        plan.update(kind="authorized_core_queue_with_selected_e6", ids=[*queue.CORE_IDS, "E6"],
                    after_queue="awaiting_final_selection")
        plan["configs"]["E6"] = self.selected
        plan["extension"] = {"archive": str(self.archive), "archive_manifest_sha256": self.digest,
                             "previous_plan_sha256": self.core_plan["plan_sha256"],
                             "previous_state_sha256": self.archive_record["previous_state_sha256"],
                             "authorized_at_utc": "fixture"}
        return plan

    @contextmanager
    def runtime(self):
        anchors = {a["validation_run"]: a for a in self.archive_record["completed"]}
        with ExitStack() as stack:
            for module in (queue, continuation):
                stack.enter_context(patch.object(module, "ROOT", self.root))
            stack.enter_context(patch.object(queue, "assert_no_active_training"))
            stack.enter_context(patch.object(queue, "existing_runs", return_value=[]))
            stack.enter_context(patch.object(queue, "make_plan", side_effect=lambda: copy.deepcopy(self.core_plan)))
            stack.enter_context(patch.object(continuation, "experiment_config", return_value=self.selected))
            stack.enter_context(patch.object(queue, "validate_training", return_value={}))
            stack.enter_context(patch.object(queue, "validate_evaluation",
                side_effect=lambda val, *_: {"checkpoint_sha256": anchors[str(val)]["best_sha256"]}))
            stack.enter_context(patch.object(queue, "verify_plan", side_effect=continuation.verify_e6_extension))
            yield

    def test_migration_preserves_completed_tasks_and_adds_only_one_pending_e6(self):
        old_bytes = (self.archive / "state.json").read_bytes()
        with self.runtime():
            state = continuation.prepare_e6_extension(self.state_path, self.archive, self.digest)
        self.assertEqual(state["tasks"][:-1], self.old_state["tasks"])
        self.assertEqual(state["tasks"][-1]["id"], "E6")
        self.assertEqual(state["tasks"][-1]["status"], "pending")
        self.assertEqual(state["status"], "prepared")
        self.assertEqual((self.archive / "state.json").read_bytes(), old_bytes)
        self.assertTrue(Path(state["plan_path"]).exists())
        with self.runtime(), self.assertRaisesRegex(ValueError, "changed since"):
            continuation.prepare_e6_extension(self.state_path, self.archive, self.digest)

    def test_active_manager_blocks_migration(self):
        write_json(self.state_path, {**self.old_state, "manager": queue.identity()})
        with self.runtime(), self.assertRaisesRegex(RuntimeError, "still active"):
            continuation.prepare_e6_extension(self.state_path, self.archive, self.digest)

    def test_live_worker_blocks_migration_even_without_a_manager(self):
        state = copy.deepcopy(self.old_state)
        state["tasks"][0]["workers"] = [queue.identity()]
        write_json(self.state_path, state)
        with self.runtime(), self.assertRaisesRegex(RuntimeError, "still active"):
            continuation.prepare_e6_extension(self.state_path, self.archive, self.digest)

    def test_existing_e6_run_is_never_retrained(self):
        with self.runtime(), patch.object(queue, "existing_runs", return_value=[self.root / "E6"]), \
             self.assertRaisesRegex(RuntimeError, "Existing E6"):
            continuation.prepare_e6_extension(self.state_path, self.archive, self.digest)
        self.assertEqual(read_json(self.state_path), self.old_state)

    def test_archive_manifest_and_archived_source_tampering_fail_visibly(self):
        with self.runtime(), self.assertRaisesRegex(ValueError, "manifest checksum"):
            continuation.read_completed_archive(self.archive, "0" * 64)
        (self.archive / "sources/configs/frozen.json").write_bytes(b"tampered")
        with self.runtime(), self.assertRaisesRegex(ValueError, "checksum mismatch"):
            continuation.read_completed_archive(self.archive, self.digest)

    def test_live_core_record_tampering_invalidates_e6_selection(self):
        Path(self.archive_record["completed"][0]["training_run"], "training.json").write_text("changed")
        with self.runtime(), self.assertRaisesRegex(ValueError, "evidence checksum"):
            continuation.verify_e6_extension(self.extended_plan())

    def test_selection_must_pin_the_exact_validation_model(self):
        self.selected["experiment"]["selection_evidence"][0]["checkpoint_sha256"] = "wrong-model"
        with self.runtime(), self.assertRaisesRegex(ValueError, "selection evidence checksum"):
            continuation.verify_e6_extension(self.extended_plan())

    def test_duplicate_e1_settings_are_reused_not_retrained_as_e6(self):
        self.selected["training"] = copy.deepcopy(self.core_plan["configs"]["E1"]["training"])
        with self.runtime(), self.assertRaisesRegex(ValueError, "duplicates a completed"):
            continuation.verify_e6_extension(self.extended_plan())

    def test_extended_manager_runs_only_e6_and_stops_before_test(self):
        plan = self.extended_plan()
        plan_path = self.root / "work/extended-plan.json"
        write_json(plan_path, plan)
        state = copy.deepcopy(self.old_state)
        state.update(status="prepared", plan_path=str(plan_path))
        state["tasks"].append({"id": "E6", "status": "pending", "workers": []})
        write_json(self.state_path, state)
        calls = []
        def stage(path, value, task, name, fd):
            calls.append((task["id"], name))
            task["training_run" if name == "train" else "validation_run"] = str(self.root / ("E6-" + name))
        with self.runtime(), patch.object(queue, "verify_plan"), patch.object(queue, "validate_evaluation"), \
             patch.object(queue, "execute_stage", side_effect=stage), \
             patch.object(queue, "report_completed", side_effect=lambda _, task: calls.append((task["id"], "report"))):
            queue.run_queue(self.state_path)
        self.assertEqual(calls, [("E6", "train"), ("E6", "val"), ("E6", "report")])
        final = read_json(self.state_path)
        self.assertEqual(final["status"], "awaiting_final_selection")
        self.assertTrue(all(t["status"] == "completed" for t in final["tasks"]))
        final["manager"] = None  # Simulate the finished manager process exiting.
        write_json(self.state_path, final)
        with self.runtime(), self.assertRaisesRegex(RuntimeError, "already completed"):
            queue.run_queue(self.state_path, resume=True)

    def test_e6_oom_preserves_completed_core_and_never_runs_validation_or_test(self):
        plan = self.extended_plan()
        plan_path = self.root / "work/extended-plan.json"
        write_json(plan_path, plan)
        state = copy.deepcopy(self.old_state)
        state.update(status="prepared", plan_path=str(plan_path))
        state["tasks"].append({"id": "E6", "status": "pending", "workers": []})
        write_json(self.state_path, state)
        with self.runtime(), patch.object(queue, "verify_plan"), patch.object(queue, "validate_evaluation"), \
             patch.object(queue, "execute_stage", side_effect=RuntimeError("CUDA out of memory")) as stage:
            with self.assertRaisesRegex(RuntimeError, "out of memory"):
                queue.run_queue(self.state_path)
        self.assertEqual(stage.call_count, 1)
        final = read_json(self.state_path)
        self.assertEqual(final["failure_kind"], "oom")
        self.assertEqual(final["tasks"][:-1], self.old_state["tasks"])
        self.assertEqual(final["tasks"][-1]["status"], "failed")


if __name__ == "__main__":
    unittest.main()
