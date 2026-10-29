import hashlib
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path

TEST_WORK = Path(__file__).resolve().parents[1] / "work" / "tests"
TEST_WORK.mkdir(parents=True, exist_ok=True)
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fire_detection.common import DEFAULT_CONFIG, load_config, read_json, timing_summary
from fire_detection.model import fetch_model, model_path, verify_model


class ConfigTests(unittest.TestCase):
    def test_default_is_cpu(self):
        config = load_config(DEFAULT_CONFIG)
        self.assertEqual(config["device"], "cpu")
        self.assertEqual(config["imgsz"], 640)

    def test_invalid_settings_fail_before_inference(self):
        cases = [
            ("confidence", 0), ("confidence", float("nan")), ("iou", 2),
            ("frame_stride", 0), ("threads", True), ("warmup_frames", -1),
            ("imgsz", 641), ("device", "cuda"), ("experiment_id", "../escape"),
        ]
        with tempfile.TemporaryDirectory(dir=TEST_WORK) as directory:
            path = Path(directory) / "config.json"
            for field, value in cases:
                with self.subTest(field=field, value=value):
                    config = read_json(DEFAULT_CONFIG)
                    config[field] = value
                    path.write_text(json.dumps(config), encoding="utf-8")
                    with self.assertRaises(ValueError):
                        load_config(path)

    def test_unknown_config_key_rejected(self):
        with tempfile.TemporaryDirectory(dir=TEST_WORK) as directory:
            path = Path(directory) / "config.json"
            config = read_json(DEFAULT_CONFIG)
            config["confidnce"] = 0.8
            path.write_text(json.dumps(config), encoding="utf-8")
            with self.assertRaises(ValueError):
                load_config(path)


class TimingTests(unittest.TestCase):
    def test_percentile_is_nearest_rank(self):
        values = timing_summary(list(range(1, 21)))
        self.assertEqual(values["p95_ms"], 19)
        self.assertEqual(values["median_ms"], 10.5)
        self.assertEqual(values["mean_ms"], 10.5)

    def test_empty_is_not_zero_performance(self):
        values = timing_summary([])
        self.assertIsNone(values["mean_ms"])
        self.assertIsNone(values["model_call_fps"])


class ModelTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(dir=TEST_WORK)
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.payload = b"test checkpoint placeholder; never executed"
        self.spec = {
            "id": "synthetic", "local_path": "models/test/best.pt",
            "url": "https://huggingface.co/test/test/resolve/pinned/best.pt",
            "sha256": hashlib.sha256(self.payload).hexdigest(),
            "size_bytes": len(self.payload),
        }

    def test_existing_model_verified_without_network(self):
        path = model_path(self.spec, self.root)
        path.parent.mkdir(parents=True)
        path.write_bytes(self.payload)
        with patch("urllib.request.urlopen") as network:
            self.assertEqual(fetch_model(self.spec, self.root), path)
            network.assert_not_called()

    def test_corrupt_file_not_overwritten(self):
        path = model_path(self.spec, self.root)
        path.parent.mkdir(parents=True)
        path.write_bytes(b"wrong")
        with patch("urllib.request.urlopen") as network:
            with self.assertRaises(ValueError):
                fetch_model(self.spec, self.root)
            self.assertEqual(path.read_bytes(), b"wrong")
            network.assert_not_called()

    def test_missing_file_rejected(self):
        with self.assertRaises(FileNotFoundError):
            verify_model(self.root / "missing.pt", self.spec)

    def test_model_path_cannot_escape(self):
        spec = dict(self.spec, local_path="../outside.pt")
        with self.assertRaises(ValueError):
            model_path(spec, self.root)

    def test_download_verified_and_metadata_written(self):
        with patch("urllib.request.urlopen", return_value=io.BytesIO(self.payload)):
            path = fetch_model(self.spec, self.root)
        self.assertEqual(path.read_bytes(), self.payload)
        self.assertTrue(path.with_suffix(".metadata.json").is_file())
        self.assertEqual(list(path.parent.glob("*.part")), [])

    def test_bad_download_leaves_no_checkpoint(self):
        with patch("urllib.request.urlopen", return_value=io.BytesIO(b"wrong")):
            with self.assertRaises(ValueError):
                fetch_model(self.spec, self.root)
        path = model_path(self.spec, self.root)
        self.assertFalse(path.exists())
        self.assertEqual(list(path.parent.glob("*.part")), [])

    def test_oversized_download_rejected(self):
        with patch("urllib.request.urlopen", return_value=io.BytesIO(self.payload * 2)):
            with self.assertRaises(ValueError):
                fetch_model(self.spec, self.root)
        self.assertFalse(model_path(self.spec, self.root).exists())

    def test_download_disconnect_cleans_partial_file(self):
        class BrokenResponse(io.BytesIO):
            def read(self, *args):
                raise OSError("simulated disconnect")
        with patch("urllib.request.urlopen", return_value=BrokenResponse(self.payload)):
            with self.assertRaises(OSError):
                fetch_model(self.spec, self.root)
        path = model_path(self.spec, self.root)
        self.assertFalse(path.exists())
        self.assertEqual(list(path.parent.glob("*.part")), [])


if __name__ == "__main__":
    unittest.main()
