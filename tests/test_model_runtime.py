"""Exercise real Ultralytics device selection without downloading/loading weights."""
import importlib.util
import os
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from fire_detection.common import DEFAULT_CONFIG, load_config
from fire_detection.model import YoloDetector

HAS_ML = all(importlib.util.find_spec(name) for name in ("torch", "ultralytics"))


@unittest.skipUnless(HAS_ML, "Project ML dependencies required")
class RuntimeTests(unittest.TestCase):
    def setUp(self):
        settings_dir = ROOT / "work/tests/ultralytics"
        settings_dir.mkdir(parents=True, exist_ok=True)
        environment = patch.dict(os.environ, {
            "YOLO_CONFIG_DIR": str(settings_dir), "YOLO_AUTOINSTALL": "false",
        })
        environment.start()
        self.addCleanup(environment.stop)
        import torch
        self.torch = torch
        original_threads = torch.get_num_threads()
        self.addCleanup(torch.set_num_threads, original_threads)
        self.config = load_config(DEFAULT_CONFIG)

    def detector(self, *, override_threads=False):
        from ultralytics.utils.torch_utils import select_device
        torch = self.torch

        class ModelStub:
            names = {0: "smoke", 1: "fire"}
            predictor = None

            def __init__(self, *args, **kwargs):
                pass

            def predict(self, **kwargs):
                device = select_device(kwargs["device"], verbose=False)
                if override_threads:
                    torch.set_num_threads(1)
                self.predictor = SimpleNamespace(device=device, imgsz=[640, 640])
                return [SimpleNamespace(boxes=None)]

        with patch("ultralytics.YOLO", ModelStub):
            return YoloDetector(ROOT / "work/not-a-real-checkpoint.pt",
                                self.config, {"0": "smoke", "1": "fire"})

    def test_first_and_later_predictions_preserve_requested_threads(self):
        detector = self.detector()
        for _ in range(2):
            self.assertEqual(detector.predict(None), [])
            self.assertEqual(self.torch.get_num_threads(), self.config["threads"])
        runtime = detector.runtime_settings()
        self.assertEqual(runtime["device"], "cpu")
        self.assertEqual(runtime["torch_num_threads"], 4)
        self.assertEqual(runtime["inference_shape"], [640, 640])

    def test_unexpected_thread_override_fails_visibly(self):
        detector = self.detector(override_threads=True)
        with self.assertRaisesRegex(RuntimeError, "CPU threads changed"):
            detector.predict(None)


if __name__ == "__main__":
    unittest.main()
