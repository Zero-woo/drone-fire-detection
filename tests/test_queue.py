"""Queue safety and detachment tests use fixtures, never actual model experiments."""
import os
import signal
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from fire_detection import queue
from fire_detection.common import ROOT, file_sha256, read_json, write_json


class QueueTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(dir=ROOT / "work")
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)

    def test_exclusive_lock_rejects_second_manager(self):
        path = self.root / "queue.lock"
        with queue.exclusive(path):
            with self.assertRaisesRegex(RuntimeError, "holds lock"):
                with queue.exclusive(path):
                    self.fail("Duplicate manager acquired the lock")
        with queue.exclusive(path):
            pass

    def test_identity_checks_pid_creation_time(self):
        current = queue.identity()
        self.assertTrue(queue.alive(current))
        self.assertFalse(queue.alive({**current, "process_created_at": current["process_created_at"] - 10}))

    def test_inherited_manager_lock_lives_until_worker_exit(self):
        path = self.root / "inherited.lock"
        with queue.exclusive(path) as handle:
            child = subprocess.Popen([str(queue.PYTHON), "-c", "import time; time.sleep(.5)"],
                                     pass_fds=(handle.fileno(),), stdin=subprocess.DEVNULL)
        try:
            with self.assertRaisesRegex(RuntimeError, "holds lock"):
                with queue.exclusive(path):
                    self.fail("An orphan worker lost the manager lock")
        finally:
            child.wait(timeout=5)
        with queue.exclusive(path):
            pass

    def test_detached_probe_survives_launcher_exit_and_sighup(self):
        receipt, log = self.root / "probe.json", self.root / "probe.log"
        launcher = ("import sys; from pathlib import Path; sys.path.insert(0, 'src'); "
                    "from fire_detection.queue import detached, PYTHON, SCRIPT; "
                    "p = detached([str(PYTHON), str(SCRIPT), 'probe', '--output', sys.argv[1], "
                    "'--delay', '1.5'], Path(sys.argv[2])); print(p.pid)")
        started = subprocess.run([str(queue.PYTHON), "-c", launcher, str(receipt), str(log)],
                                 cwd=ROOT, capture_output=True, text=True, check=True, timeout=15)
        pid = int(started.stdout.strip())
        deadline = time.monotonic() + 10
        while time.monotonic() < deadline and not receipt.exists():
            time.sleep(.05)
        self.assertTrue(receipt.exists(), log.read_text())
        os.kill(pid, signal.SIGHUP)
        while time.monotonic() < deadline:
            result = read_json(receipt)
            if result["status"] == "completed":
                break
            time.sleep(.05)
        self.assertEqual(result["status"], "completed")
        self.assertEqual(result["sid"], pid)
        self.assertEqual(result["ppid"], 1)
        self.assertEqual(result["stdin"], "/dev/null")
        self.assertEqual(result["stdout"], str(log))
        self.assertTrue(result["sighup_ignored"])

    def training_fixture(self):
        run = self.root / "train-run"
        weight_dir = run / "train/weights"
        weight_dir.mkdir(parents=True)
        entries = {}
        for name in ("last", "best"):
            path = weight_dir / f"{name}.pt"
            path.write_bytes(b"Synthetic checkpoint, not model results")
            entries[name] = {"path": str(path), "sha256": file_sha256(path), "epoch": 100}
        record = {"status": "completed", "config": {"training": {"epochs": 100}}, "dataset_fingerprint": "fixture",
                  "initial_model": {"sha256": queue.EXPECTED_COCO},
                  "attempts": [{"checkpoint_sha256": queue.EXPECTED_COCO}],
                  "epochs_completed": 100, "best_path": entries["best"]["path"], "best_sha256": entries["best"]["sha256"]}
        write_json(run / "training.json", record)
        write_json(run / "checkpoints.json", entries)
        return run, record

    def test_completed_training_requires_registered_checkpoint_hashes(self):
        run, record = self.training_fixture()
        with patch.object(queue, "runs_root", return_value=self.root):
            queue.validate_training(run, record["config"], "fixture")
            (run / "train/weights/last.pt").write_bytes(b"corrupt")
            with self.assertRaisesRegex(ValueError, "checksum mismatch"):
                queue.validate_training(run, record["config"], "fixture")

    def test_training_configuration_drift_is_rejected(self):
        run, record = self.training_fixture()
        with patch.object(queue, "runs_root", return_value=self.root):
            with self.assertRaisesRegex(ValueError, "integrity mismatch"):
                queue.validate_training(run, {"training": {"epochs": 50}}, "fixture")

    def test_validation_must_belong_to_exact_training_run(self):
        run, record = self.training_fixture()
        config = record["config"]
        config["evaluation"] = {"operating_conf": .25}
        folder = self.root / "val"
        value = {"status": "completed", "split": "val", "model_role": "trained", "training_run": str(run),
                 "config": config, "dataset_fingerprint": "fixture", "checkpoint_sha256": record["best_sha256"],
                 "checkpoint": record["best_path"], "operating": {"settings": config["evaluation"]}}
        write_json(folder / "evaluation.json", value)
        with patch.object(queue, "runs_root", return_value=self.root):
            queue.validate_evaluation(folder, run, record)
            value["training_run"] = str(self.root / "another-run")
            write_json(folder / "evaluation.json", value)
            with self.assertRaisesRegex(ValueError, "integrity mismatch"):
                queue.validate_evaluation(folder, run, record)

    def test_unrecorded_interrupted_training_is_not_guessed(self):
        with self.assertRaisesRegex(RuntimeError, "no recorded training run"):
            queue.recover_task({"id": "E1"}, {})

    def test_resume_preserves_exact_run_and_rejects_oom(self):
        run, record = self.training_fixture()
        record.update(status="interrupted")
        write_json(run / "training.json", record)
        plan = {"configs": {"E1": record["config"]}, "dataset_fingerprint": "fixture"}
        task = {"id": "E1", "training_run": str(run)}
        with patch.object(queue, "runs_root", return_value=self.root):
            self.assertEqual(queue.recover_task(task, plan), "resume")
            record["status"] = "failed"
            record["attempts"][-1]["error"] = "CUDA out of memory"
            write_json(run / "training.json", record)
            with self.assertRaisesRegex(RuntimeError, "Non-transient"):
                queue.recover_task(task, plan)

    def state_fixture(self):
        plan = {"plan_sha256": "fixture", "configs": {key: {} for key in queue.CORE_IDS},
                "dataset_fingerprint": "fixture"}
        plan_path = self.root / "plan.json"
        write_json(plan_path, plan)
        state_path = self.root / "state.json"
        state = {"status": "prepared", "manager": None, "plan_path": str(plan_path),
                 "plan_sha256": "fixture", "tasks": [{"id": key, "status": "pending"} for key in queue.CORE_IDS]}
        write_json(state_path, state)
        return state_path, state

    def test_sequential_validation_gate_and_analysis_stop(self):
        state_path, _ = self.state_fixture()
        calls = []

        def stage(path, state, task, name, fd):
            calls.append((task["id"], name))
            task["training_run" if name == "train" else "validation_run"] = str(self.root / (task["id"] + name))

        def report(state, task):
            calls.append((task["id"], "report"))

        with patch.object(queue, "ROOT", self.root), patch.object(queue, "assert_no_active_training"), \
             patch.object(queue, "verify_plan"), patch.object(queue, "existing_runs", return_value=[]), \
             patch.object(queue, "execute_stage", side_effect=stage), patch.object(queue, "validate_training", return_value={}), \
             patch.object(queue, "validate_evaluation"), patch.object(queue, "report_completed", side_effect=report):
            queue.run_queue(state_path)
        self.assertEqual(calls, [(key, stage) for key in queue.CORE_IDS for stage in ("train", "val", "report")])
        state = read_json(state_path)
        self.assertEqual(state["status"], "awaiting_analysis")
        self.assertTrue(all(task["status"] == "completed" for task in state["tasks"]))
        self.assertNotIn("E6", [key for key, stage in calls])

    def test_worker_failure_stops_queue_without_retry_or_settings_change(self):
        for message, kind in (("CUDA out of memory", "oom"), ("checkpoint checksum mismatch", "integrity")):
            state_path, initial = self.state_fixture()
            with self.subTest(kind=kind), patch.object(queue, "ROOT", self.root), \
                 patch.object(queue, "assert_no_active_training"), patch.object(queue, "verify_plan"), \
                 patch.object(queue, "existing_runs", return_value=[]), \
                 patch.object(queue, "execute_stage", side_effect=RuntimeError(message)) as execution:
                with self.assertRaisesRegex(RuntimeError, message):
                    queue.run_queue(state_path)
                self.assertEqual(execution.call_count, 1)
            state = read_json(state_path)
            self.assertEqual(state["status"], "failed")
            self.assertEqual(state["failure_kind"], kind)
            self.assertEqual(state["tasks"][1:], initial["tasks"][1:])

    def test_live_worker_blocks_manager_restart(self):
        state_path, state = self.state_fixture()
        state["tasks"][0]["workers"] = [queue.identity()]
        write_json(state_path, state)
        with patch.object(queue, "ROOT", self.root), self.assertRaisesRegex(RuntimeError, "worker is still active"):
            queue.run_queue(state_path, resume=True)

    def test_plan_mutation_is_rejected_before_loading_inputs(self):
        plan = {"ids": list(queue.CORE_IDS), "plan_sha256": "not-a-valid-digest"}
        with self.assertRaisesRegex(ValueError, "plan checksum"):
            queue.verify_plan(plan)


if __name__ == "__main__":
    unittest.main()
