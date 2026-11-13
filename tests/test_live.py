"""Live-pipeline fixtures exercise buffering and provenance, never model accuracy."""
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from fire_detection import live
from fire_detection.cli import parser
from fire_detection.common import ROOT, file_sha256, object_sha256, read_json, write_json
from fire_detection.live_model import frozen_model, verified_deployment


class LiveTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(dir=ROOT / "work")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.config = read_json(ROOT / "configs/live.json")

    def revision_fixture(self):
        checkpoint = self.root / "fixture.pt"
        checkpoint.write_bytes(b"Checksum fixture, no trained model")
        revision = {"weights": checkpoint.name, "checkpoint_sha256": file_sha256(checkpoint),
                    "classes": {"0": "smoke", "1": "fire"}, "inference": {"imgsz": 800}}
        revision["revision_sha256"] = object_sha256(revision)
        path = self.root / "model_revision.json"
        write_json(path, revision)
        return {"revision": str(path), "revision_sha256": revision["revision_sha256"]}, revision, checkpoint

    def test_frozen_identity_rejects_changed_model_and_self_rehashed_revision(self):
        config, revision, checkpoint = self.revision_fixture()
        self.assertEqual(frozen_model(config)[1], checkpoint)
        checkpoint.write_bytes(b"changed")
        with self.assertRaisesRegex(ValueError, "checkpoint checksum"):
            frozen_model(config)
        revision["checkpoint_sha256"] = file_sha256(checkpoint)
        revision["revision_sha256"] = object_sha256({k: v for k, v in revision.items() if k != "revision_sha256"})
        write_json(Path(config["revision"]), revision)
        with self.assertRaisesRegex(ValueError, "revision checksum"):
            frozen_model(config)

    def test_deployment_rejects_artifact_mutation_and_escaping_paths(self):
        config, revision, _ = self.revision_fixture()
        xml = self.root / "model.xml"
        xml.write_bytes(b"Export fixture")
        binary = xml.with_suffix(".bin")
        binary.write_bytes(b"Export binary fixture")
        deployment = {"status": "completed", "source_revision_sha256": config["revision_sha256"],
                      "source_checkpoint_sha256": revision["checkpoint_sha256"],
                      "inference": revision["inference"], "xml": xml.name,
                      "files": {xml.name: file_sha256(xml), binary.name: file_sha256(binary)}}
        path = self.root / "deployment.json"
        deployment["deployment_sha256"] = object_sha256(deployment)
        write_json(path, deployment)
        self.assertEqual(verified_deployment(path, revision)[1], xml)
        xml.write_bytes(b"Changed export")
        with self.assertRaisesRegex(ValueError, "artifact checksum"):
            verified_deployment(path, revision)
        deployment["files"] = {"../outside.xml": "0" * 64}
        deployment["deployment_sha256"] = object_sha256({k: v for k, v in deployment.items() if k != "deployment_sha256"})
        write_json(path, deployment)
        with self.assertRaisesRegex(ValueError, "Expected a path below"):
            verified_deployment(path, revision)

    def test_live_settings_reject_nonfinite_and_nonpositive_values(self):
        live.validate_config(self.config)
        for key, value in (("target_fps", float("nan")), ("target_fps", 0),
                           ("threads", True), ("event_min_hits", 1.5), ("event_match_iou", 2)):
            config = {**self.config, key: value}
            with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                live.validate_config(config)

    def test_droidcam_root_url_selects_video_and_redacts_credentials(self):
        source, _, info = live.select_source(" http://user:secret@192.168.1.3:4747/?token=private ", "unused")
        self.assertEqual(source, "http://user:secret@192.168.1.3:4747/video?token=private")
        self.assertEqual(info["url"], "http://192.168.1.3:4747/video")
        self.assertEqual(info["kind"], "stream")

    def test_latest_capture_replaces_pending_frames_and_releases_on_eof(self):
        class Capture:
            def __init__(self):
                self.frames = iter(["one", "two", "three"])
                self.released = False

            def read(self):
                image = next(self.frames, None)
                return image is not None, image

            def release(self):
                self.released = True

        capture = Capture()
        worker = live.LatestCapture(capture, file_fps=1000)
        worker.start()
        worker.thread.join(timeout=2)
        self.assertFalse(worker.thread.is_alive())
        self.assertEqual(worker.take().image, "three")
        self.assertEqual((worker.received, worker.dropped), (3, 2))
        self.assertTrue(capture.released)
        self.assertIsNone(worker.take())

    def test_camera_disconnect_propagates_and_releases(self):
        class Capture:
            released = False

            def read(self):
                raise OSError("Disconnected")

            def release(self):
                self.released = True

        capture = Capture()
        worker = live.LatestCapture(capture)
        worker.start()
        worker.thread.join(timeout=2)
        self.assertEqual(worker.error, "Disconnected")
        self.assertTrue(worker.stop())
        self.assertTrue(capture.released)

    def test_events_need_repeated_same_class_region_and_expire(self):
        tracker = live.EventTracker(.5, 3, .3)
        fire = {"class_id": 1, "bbox": [0, 0, 10, 10], "confidence": .8}
        smoke = {**fire, "class_id": 0}
        self.assertEqual(tracker.update([fire], 0), ([], []))
        self.assertEqual(tracker.update([fire, smoke], .2), ([], []))
        active, created = tracker.update([fire], .4)
        self.assertEqual(len(active), 1)
        self.assertEqual(created, [1])
        self.assertEqual(tracker.update([fire], .45)[1], [])
        self.assertEqual(tracker.update([], 1), ([], []))

    def test_multiple_boxes_in_one_frame_cannot_supply_repeated_hits(self):
        tracker = live.EventTracker(.5, 3, .3)
        detection = {"class_id": 1, "bbox": [0, 0, 10, 10], "confidence": .8}
        self.assertEqual(tracker.update([detection] * 3, 0), ([], []))

    def test_cli_keeps_offline_predict_and_adds_explicit_live_options(self):
        offline = parser().parse_args(["predict", "--source", "video.mp4"])
        self.assertEqual(offline.device, "0")
        args = parser().parse_args(["live", "--source", "2", "--no-display", "--duration", "1"])
        self.assertEqual((args.source, args.no_display, args.duration), ("2", True, 1))

    def test_headless_saved_video_uses_normal_cli_and_writes_complete_session(self):
        import cv2
        import numpy as np
        from types import SimpleNamespace
        from fire_detection.cli import main
        video = self.root / "fixture.avi"
        writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"MJPG"), 10, (64, 48))
        self.assertTrue(writer.isOpened())
        for _ in range(3):
            writer.write(np.zeros((48, 64, 3), dtype=np.uint8))
        writer.release()
        config, revision, checkpoint = self.revision_fixture()
        path = self.root / "config.json"
        write_json(path, {**self.config, **config})
        run = self.root / "run"
        run.mkdir()

        class Detector:
            runtime = {"backend": "test_fixture"}
            names = {0: "smoke", 1: "fire"}

            def predict(self, image):
                data = SimpleNamespace(cpu=lambda: SimpleNamespace(tolist=lambda: []))
                return SimpleNamespace(boxes=SimpleNamespace(data=data), names=self.names,
                                       plot=lambda: image.copy())

        with patch.object(live, "CpuDetector", return_value=Detector()), patch.object(live, "new_run", return_value=run):
            code = main(["live", "--config", str(path), "--source", str(video),
                         "--no-display", "--max-frames", "2"])
        self.assertEqual(code, 0)
        record = read_json(run / "session.json")
        self.assertEqual((record["status"], record["processed_frames"], record["capture_shutdown"]),
                         ("completed", 2, "joined"))
        import json
        rows = [json.loads(line) for line in (run / "detections.jsonl").read_text().splitlines()]
        self.assertEqual(len(rows), 2)
        self.assertTrue(all(row["frame_shape"] == [48, 64, 3] for row in rows))

        closed_run = self.root / "closed-run"
        closed_run.mkdir()
        with patch.object(live, "CpuDetector", return_value=Detector()), \
                patch.object(live, "new_run", return_value=closed_run), \
                patch.object(cv2, "namedWindow"), patch.object(cv2, "pollKey", return_value=-1), \
                patch.object(cv2, "getWindowProperty", side_effect=cv2.error("Window already closed")), \
                patch.object(cv2, "destroyAllWindows"):
            code = main(["live", "--config", str(path), "--source", str(video)])
        self.assertEqual(code, 0)
        closed = read_json(closed_run / "session.json")
        self.assertEqual((closed["status"], closed["stopped_by"], closed["capture_shutdown"]),
                         ("completed", "window_closed", "joined"))


if __name__ == "__main__":
    unittest.main()
