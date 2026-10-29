"""Synthetic media + a fake detector verify plumbing, NOT fire accuracy."""
import hashlib
import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path

TEST_WORK = Path(__file__).resolve().parents[1] / "work" / "tests"
TEST_WORK.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fire_detection.common import DEFAULT_CONFIG, load_config, read_json
from fire_detection.experiment import run_experiment
from fire_detection.media import frames, read_image, resolve_sources, write_image

HAS_MEDIA = all(importlib.util.find_spec(m) for m in ("cv2", "numpy"))


class FakeDetector:
    def __init__(self, checkpoint, config, classes, root):
        pass

    def predict(self, frame):
        if float(frame.mean()) < 100:
            return []
        return [{
            "class_id": 1, "class_name": "fire", "confidence": 0.9,
            "bbox_xyxy": [2.0, 2.0, 20.0, 20.0],
            "center_normalized_xy": [11 / frame.shape[1], 11 / frame.shape[0]],
        }]


@unittest.skipUnless(HAS_MEDIA, "OpenCV and NumPy required for synthetic media tests")
class ExperimentTests(unittest.TestCase):
    def setUp(self):
        import cv2
        import numpy as np
        self.cv2, self.np = cv2, np
        self.temp = tempfile.TemporaryDirectory(dir=TEST_WORK)
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name) / "한글 실험"
        self.root.mkdir()
        self.config = load_config(DEFAULT_CONFIG)
        self.config["experiment_id"] = "test_synthetic"
        payload = b"synthetic test model, not YOLO"
        checkpoint = self.root / "models/test.pt"
        checkpoint.parent.mkdir()
        checkpoint.write_bytes(payload)
        self.spec = {
            "id": "synthetic", "local_path": "models/test.pt",
            "size_bytes": len(payload), "sha256": hashlib.sha256(payload).hexdigest(),
            "classes": {"0": "smoke", "1": "fire"},
        }
        self.images = self.root / "입력 사진"
        self.images.mkdir()
        for index, value in enumerate((0, 80, 180, 255)):
            write_image(self.images / f"사진{index}.png",
                        np.full((48, 64, 3), value, dtype=np.uint8))

    def run_fake(self, source=None, **kwargs):
        return run_experiment(source or self.images, self.config, self.spec,
                              root=self.root, detector_factory=FakeDetector, **kwargs)

    def make_video(self):
        path = self.root / "입력 영상.avi"
        writer = self.cv2.VideoWriter(
            str(path), self.cv2.VideoWriter_fourcc(*"MJPG"), 10.0, (64, 48)
        )
        if not writer.isOpened():
            self.skipTest("MJPG encoder unavailable")
        for value in range(0, 240, 30):
            writer.write(self.np.full((48, 64, 3), value, dtype=self.np.uint8))
        writer.release()
        return path

    def test_unicode_paths_and_image_previews(self):
        result = self.run_fake(save_preview=True)
        summary = read_json(result / "summary.json")
        self.assertEqual(summary["status"], "completed")
        self.assertEqual(summary["processed_frames"], 4)
        self.assertEqual(summary["frames_with_detections"], 2)
        self.assertEqual(summary["detections_by_class"], {"fire": 2})
        self.assertIsNone(summary["accuracy_metrics"])
        self.assertEqual(len(list((result / "preview").glob("*.jpg"))), 4)
        metadata = read_json(result / "run.json")
        self.assertEqual(len(metadata["inputs"]), 4)
        self.assertEqual(len(metadata["inputs"][0]["sha256"]), 64)
        self.assertEqual(metadata["model"]["sha256"], self.spec["sha256"])
        self.assertEqual(metadata["options"]["save_preview"], True)
        self.assertIn("packages", metadata["environment"])
        self.assertEqual(read_image(result / "preview/000001.jpg").shape[:2], (48, 64))

    def test_no_detections_is_valid_success(self):
        result = self.run_fake(self.images / "사진0.png")
        summary = read_json(result / "summary.json")
        self.assertEqual(summary["status"], "completed")
        self.assertEqual(summary["detections_by_class"], {})
        self.assertEqual(summary["processed_frames"], 1)

    def test_runtime_metadata_uses_observed_values(self):
        class RuntimeDetector(FakeDetector):
            def runtime_settings(self):
                return {"torch_num_threads": 7, "test_double": True}

        result = run_experiment(self.images, self.config, self.spec, root=self.root,
                                detector_factory=RuntimeDetector)
        metadata = read_json(result / "run.json")
        self.assertEqual(metadata["config"]["threads"], 4)
        self.assertEqual(metadata["runtime"]["torch_num_threads"], 7)
        self.assertTrue(metadata["runtime"]["test_double"])

    def test_repeated_runs_never_overwrite(self):
        first = self.run_fake(max_frames=1)
        second = self.run_fake(max_frames=1)
        self.assertNotEqual(first, second)
        self.assertTrue((first / "summary.json").exists())

    def test_video_stride_limit_and_preview(self):
        self.config["frame_stride"] = 2
        result = self.run_fake(self.make_video(), max_frames=3, save_preview=True)
        records = [json.loads(line) for line in
                   (result / "detections.jsonl").read_text(encoding="utf-8").splitlines()]
        self.assertEqual([r["frame_index"] for r in records], [0, 2, 4])
        self.assertEqual([r["video_time_s"] for r in records], [0.0, 0.2, 0.4])
        summary = read_json(result / "summary.json")
        self.assertEqual(summary["stop_reason"], "max_frames")
        capture = self.cv2.VideoCapture(str(result / "preview/annotated.avi"))
        self.assertTrue(capture.isOpened())
        self.assertAlmostEqual(capture.get(self.cv2.CAP_PROP_FPS), 5.0, places=1)
        self.assertEqual(int(capture.get(self.cv2.CAP_PROP_FRAME_COUNT)), 3)
        capture.release()

    def test_video_end_of_input(self):
        result = self.run_fake(self.make_video())
        summary = read_json(result / "summary.json")
        self.assertEqual(summary["processed_frames"], 8)
        self.assertEqual(summary["stop_reason"], "end_of_input")

    def test_missing_model_records_failure(self):
        (self.root / self.spec["local_path"]).unlink()
        with self.assertRaises(FileNotFoundError):
            self.run_fake()
        result = next((self.root / "experiments/runs").iterdir())
        self.assertEqual(read_json(result / "run.json")["status"], "failed")
        summary = read_json(result / "summary.json")
        self.assertEqual(summary["processed_frames"], 0)
        self.assertIsNone(summary["pipeline_fps"])

    def test_corrupt_image_records_failure(self):
        corrupt = self.root / "bad.png"
        corrupt.write_bytes(b"not an image")
        with self.assertRaises(ValueError):
            self.run_fake(corrupt)
        result = next((self.root / "experiments/runs").iterdir())
        self.assertEqual(read_json(result / "run.json")["status"], "failed")

    def test_interruption_is_not_success(self):
        class InterruptedDetector(FakeDetector):
            def predict(self, frame):
                raise KeyboardInterrupt()
        with self.assertRaises(KeyboardInterrupt):
            run_experiment(self.images, self.config, self.spec, root=self.root,
                           detector_factory=InterruptedDetector)
        result = next((self.root / "experiments/runs").iterdir())
        self.assertEqual(read_json(result / "run.json")["status"], "interrupted")
        self.assertEqual(read_json(result / "summary.json")["stop_reason"], "keyboard_interrupt")

    def test_invalid_input_creates_no_run(self):
        with self.assertRaises(FileNotFoundError):
            self.run_fake(self.root / "missing.mp4")
        self.assertFalse((self.root / "experiments/runs").exists())

    def test_nonpositive_limit_rejected(self):
        with self.assertRaises(ValueError):
            self.run_fake(max_frames=0)

    def test_image_directory_does_not_silently_take_videos(self):
        video = self.make_video()
        image = video.parent / "standalone.png"
        image.write_bytes((self.images / "사진0.png").read_bytes())
        files, kind = resolve_sources(video.parent)
        self.assertEqual(kind, "images")
        self.assertEqual(files, [image])


if __name__ == "__main__":
    unittest.main()
