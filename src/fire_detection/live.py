"""Latest-frame camera inference with bounded buffering and CPU-only detection."""
from collections import deque
from dataclasses import dataclass
import hashlib
import json
import math
import sys
import threading
import time
from urllib.parse import urlsplit

from .common import ROOT, environment, file_sha256, new_run, read_json, resolve, utc_now, write_json
from .evaluation import iou_xyxy
from .live_model import CpuDetector, frozen_model


def validate_config(config):
    if not isinstance(config["camera_name"], str) or not config["camera_name"].strip():
        raise ValueError("camera_name must be nonempty")
    for name in ("width", "height", "capture_fps", "threads", "event_min_hits"):
        if type(config[name]) is not int or config[name] <= 0:
            raise ValueError(f"{name} must be a positive integer")
    for name in ("target_fps", "source_timeout_seconds", "event_window_seconds", "event_match_iou"):
        value = config[name]
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
            raise ValueError(f"{name} must be finite and positive")
    if config["event_match_iou"] > 1:
        raise ValueError("event_match_iou must be at most 1")


def cameras():
    import cv2
    from cv2_enumerate_cameras import enumerate_cameras
    backend = cv2.CAP_DSHOW if sys.platform == "win32" else cv2.CAP_ANY
    return [{"index": info.index, "name": info.name, "backend": info.backend}
            for info in enumerate_cameras(backend)]


def select_source(source, camera_name):
    import cv2
    backend = cv2.CAP_DSHOW if sys.platform == "win32" else cv2.CAP_ANY
    if source is not None:
        source = str(source).strip()
        address = urlsplit(source)
        if address.scheme in {"http", "https"}:
            if not address.hostname:
                raise ValueError("Stream URL must include a host")
            if address.path in {"", "/"}:
                address = address._replace(path="/video")
            address = address._replace(fragment="")
            selected = address.geturl()
            redacted = address._replace(netloc=address.netloc.rsplit("@", 1)[-1], query="").geturl()
            return selected, cv2.CAP_FFMPEG, {"kind": "stream", "url": redacted,
                                            "address_sha256": hashlib.sha256(selected.encode()).hexdigest()}
        if str(source).isdecimal():
            return int(source), backend, {"kind": "camera", "index": int(source)}
        path = resolve(source)
        if not path.is_file():
            raise ValueError("--source must be a camera index, HTTP video URL, or existing saved video")
        return str(path), cv2.CAP_ANY, {"kind": "file", "path": str(path), "sha256": file_sha256(path)}
    matches = [entry for entry in cameras() if camera_name.casefold() in entry["name"].casefold()]
    if len(matches) != 1:
        raise ValueError(f"Found {len(matches)} cameras matching {camera_name!r}. "
                         "Run firelab.py cameras and select --source INDEX or --camera-name NAME.")
    selected = matches[0]
    return selected["index"], selected["backend"], {"kind": "camera", **selected}


@dataclass
class Frame:
    number: int
    image: object
    read_at: float
    read_at_utc: str


class LatestCapture:
    """One producer owns the capture handle. Unconsumed frames are replaced."""
    def __init__(self, capture, file_fps=0):
        self.capture = capture
        self.file_fps = file_fps
        self.condition = threading.Condition()
        self.stop_event = threading.Event()
        self.pending = None
        self.finished = False
        self.error = None
        self.received = 0
        self.dropped = 0
        self.last_read_at = time.perf_counter()
        self.thread = threading.Thread(target=self._read, name="live-capture", daemon=True)

    def start(self):
        self.thread.start()

    def _read(self):
        try:
            next_frame_at = time.perf_counter()
            while not self.stop_event.is_set():
                if self.file_fps and self.stop_event.wait(max(0, next_frame_at - time.perf_counter())):
                    break
                ok, image = self.capture.read()
                read_at = time.perf_counter()
                if not ok:
                    if not self.file_fps:
                        raise RuntimeError("Video source stopped delivering frames. Check DroidCam connection.")
                    break
                with self.condition:
                    self.received += 1
                    self.dropped += int(self.pending is not None)
                    self.last_read_at = read_at
                    self.pending = Frame(self.received, image, read_at, utc_now())
                    self.condition.notify_all()
                next_frame_at = max(next_frame_at + 1 / self.file_fps, read_at) if self.file_fps else read_at
        except BaseException as exc:
            self.error = str(exc)
        finally:
            self.capture.release()
            with self.condition:
                self.finished = True
                self.condition.notify_all()

    def take(self, timeout=.02):
        with self.condition:
            self.condition.wait_for(lambda: self.pending is not None or self.finished, timeout)
            frame, self.pending = self.pending, None
            return frame

    def stop(self):
        self.stop_event.set()
        self.thread.join(timeout=2)
        return not self.thread.is_alive()


class EventTracker:
    """Associate same-class boxes and confirm repeated hits within a time window."""
    def __init__(self, window, min_hits, match_iou):
        self.window = window
        self.min_hits = min_hits
        self.match_iou = match_iou
        self.tracks = []
        self.next_id = 1

    def update(self, detections, now):
        self.tracks = [track for track in self.tracks if now - track["last_seen"] <= self.window]
        used, newly_confirmed = set(), []
        for detection in detections:
            candidates = [(iou_xyxy(track["bbox"], detection["bbox"]), track)
                          for track in self.tracks
                          if track["id"] not in used and track["class_id"] == detection["class_id"]]
            overlap, track = max(candidates, key=lambda item: item[0], default=(0, None))
            if track is None or overlap < self.match_iou:
                track = {"id": self.next_id, "class_id": detection["class_id"],
                         "hits": deque(), "confirmed": False}
                self.next_id += 1
                self.tracks.append(track)
            used.add(track["id"])
            track.update(bbox=detection["bbox"], last_seen=now, confidence=detection["confidence"])
            track["hits"].append(now)
            while track["hits"][0] < now - self.window:
                track["hits"].popleft()
            if not track["confirmed"] and len(track["hits"]) >= self.min_hits:
                track["confirmed"] = True
                newly_confirmed.append(track["id"])
        active = [{k: track[k] for k in ("id", "class_id", "bbox", "confidence")}
                  for track in self.tracks if track["confirmed"]]
        return active, newly_confirmed


def detections_from_result(result):
    return [{"class_id": int(box[5]), "class_name": result.names[int(box[5])],
             "confidence": float(box[4]), "bbox": [float(value) for value in box[:4]]}
            for box in result.boxes.data.cpu().tolist()]


def live(config_path="configs/live.json", source=None, camera_name=None, target_fps=None,
         deployment=None, display=True, duration=None, max_frames=None):
    config = read_json(resolve(config_path))
    config["camera_name"] = camera_name or config["camera_name"]
    if target_fps is not None:
        config["target_fps"] = target_fps
    validate_config(config)
    if duration is not None and (not math.isfinite(duration) or duration <= 0):
        raise ValueError("duration must be finite and positive")
    if max_frames is not None and max_frames <= 0:
        raise ValueError("max_frames must be positive")
    revision, checkpoint = frozen_model(config)
    selected_source, backend, source_info = select_source(source, config["camera_name"])
    run = new_run("live", ROOT / "outputs")
    record = {"status": "starting", "created_at_utc": utc_now(), "config": config,
              "model_revision_sha256": revision["revision_sha256"],
              "checkpoint_sha256": revision["checkpoint_sha256"], "inference": revision["inference"],
              "source": source_info, "environment": environment(), "display": display,
              "timing_note": "PC VideoCapture.read return to result/log/overlay preparation; phone/network/driver buffering precedes this clock.",
              "input_hash_note": "SHA256 of contiguous decoded BGR uint8 frame bytes, with shape recorded per frame.",
              "bbox_coordinates": "pixel xyxy in the original decoded frame"}
    write_json(run / "session.json", record)
    capture = worker = None
    window = "E6 CPU - press Q or Esc to stop"
    processed, samples = 0, deque(maxlen=3000)
    start = None
    stopped_by = None
    try:
        import cv2
        import numpy as np
        detector = CpuDetector(revision, checkpoint, config["threads"], deployment)
        record["runtime"] = detector.runtime
        if source_info["kind"] == "stream":
            timeout_ms = int(config["source_timeout_seconds"] * 1000)
            capture = cv2.VideoCapture(selected_source, backend, [
                cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, timeout_ms,
                cv2.CAP_PROP_READ_TIMEOUT_MSEC, timeout_ms,
            ])
        else:
            capture = cv2.VideoCapture(selected_source, backend)
        if not capture.isOpened():
            raise RuntimeError("Cannot open selected video source. Start DroidCam and check its video output.")
        file_fps = 0
        if source_info["kind"] == "file":
            file_fps = capture.get(cv2.CAP_PROP_FPS)
            if not math.isfinite(file_fps) or file_fps <= 0:
                raise ValueError("Saved video must declare a finite positive frame rate")
        elif source_info["kind"] == "camera":
            capture.set(cv2.CAP_PROP_FRAME_WIDTH, config["width"])
            capture.set(cv2.CAP_PROP_FRAME_HEIGHT, config["height"])
            capture.set(cv2.CAP_PROP_FPS, config["capture_fps"])
        record["source_effective"] = {"width": capture.get(cv2.CAP_PROP_FRAME_WIDTH),
                                      "height": capture.get(cv2.CAP_PROP_FRAME_HEIGHT),
                                      "reported_fps": capture.get(cv2.CAP_PROP_FPS),
                                      "backend": capture.getBackendName()}
        worker = LatestCapture(capture, file_fps)
        tracker = EventTracker(config["event_window_seconds"], config["event_min_hits"], config["event_match_iou"])
        if display:
            cv2.namedWindow(window, cv2.WINDOW_NORMAL)
        start = next_inference = last_print = time.perf_counter()
        worker.start()
        record.update(status="running", capture_started_at_utc=utc_now())
        write_json(run / "session.json", record)
        print(f"E6 CPU running. Results: {run}", flush=True)
        with (run / "detections.jsonl").open("x", encoding="utf-8", buffering=1) as log, \
                (run / "events.jsonl").open("x", encoding="utf-8", buffering=1) as event_log:
            while True:
                if display:
                    key = cv2.pollKey() & 0xFF
                    try:
                        visible = cv2.getWindowProperty(window, cv2.WND_PROP_VISIBLE) >= 1
                    except cv2.error:
                        visible = False  # A window closed by its title-bar button can already be destroyed.
                    if key in (27, ord("q")) or not visible:
                        stopped_by = "window_closed"
                        break
                now = time.perf_counter()
                if duration is not None and now - start >= duration:
                    stopped_by = "duration"
                    break
                if max_frames is not None and processed >= max_frames:
                    stopped_by = "max_frames"
                    break
                if worker.error:
                    raise RuntimeError(worker.error)
                if now - worker.last_read_at > config["source_timeout_seconds"]:
                    raise RuntimeError("Video source timed out waiting for a new frame")
                if now < next_inference:
                    time.sleep(min(.01, next_inference - now))
                    continue
                frame = worker.take()
                if frame is None:
                    if worker.finished:
                        stopped_by = "end_of_file"
                        break
                    continue
                infer_start = time.perf_counter()
                result = detector.predict(frame.image)
                infer_ms = (time.perf_counter() - infer_start) * 1000
                detections = detections_from_result(result)
                active, new_events = tracker.update(detections, frame.read_at)
                processed += 1
                annotated = result.plot()
                annotated = cv2.copyMakeBorder(annotated, 72, 0, 0, 0,
                                               cv2.BORDER_CONSTANT, value=(20, 20, 20))
                elapsed = time.perf_counter() - start
                actual_fps = processed / elapsed
                age_ms = (time.perf_counter() - frame.read_at) * 1000
                text = f"CPU {actual_fps:.1f}/{config['target_fps']:g} FPS | predict {infer_ms:.0f} ms | PC age {age_ms:.0f} ms"
                cv2.putText(annotated, text, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, .55, (255, 255, 255), 1, cv2.LINE_AA)
                state = "CONFIRMED " + ", ".join(sorted({detector.names[t['class_id']] for t in active})) if active else "Monitoring"
                cv2.putText(annotated, f"{state} | replaced frames {worker.dropped}", (10, 53),
                            cv2.FONT_HERSHEY_SIMPLEX, .55, (0, 200, 255), 1, cv2.LINE_AA)
                frame_record = {"frame": frame.number, "read_at_utc": frame.read_at_utc,
                                "frame_shape": list(frame.image.shape),
                                "input_bgr_sha256": hashlib.sha256(np.ascontiguousarray(frame.image).tobytes()).hexdigest(),
                                "detections": detections, "confirmed_tracks": active,
                                "predict_ms": infer_ms, "pc_read_to_overlay_ms": age_ms,
                                "processing_fps": actual_fps, "replaced_frames": worker.dropped}
                for track_id in new_events:
                    image_path = run / f"event-{track_id:04d}-frame-{frame.number}.jpg"
                    if not cv2.imwrite(str(image_path), annotated):
                        raise OSError("Failed to save event image")
                    event_log.write(json.dumps({"track_id": track_id, "frame": frame.number,
                                                "read_at_utc": frame.read_at_utc,
                                                "detection": next(t for t in active if t["id"] == track_id),
                                                "image": image_path.name, "image_sha256": file_sha256(image_path)}, allow_nan=False) + "\n")
                log.write(json.dumps(frame_record, allow_nan=False) + "\n")
                if display:
                    cv2.imshow(window, annotated)
                samples.append({"predict_ms": infer_ms, "pc_read_to_overlay_ms": age_ms})
                next_inference = max(infer_start + 1 / config["target_fps"], time.perf_counter())
                if time.perf_counter() - last_print >= 1:
                    print(text, flush=True)
                    last_print = time.perf_counter()
        record.update(status="completed", stopped_by=stopped_by)
    except KeyboardInterrupt:
        record.update(status="stopped", stopped_by="keyboard_interrupt")
    except BaseException as exc:
        record.update(status="failed", error=str(exc))
        raise
    finally:
        capture_elapsed = time.perf_counter() - start if start is not None else 0
        if worker is not None and worker.thread.ident is not None:
            record["capture_shutdown"] = "joined" if worker.stop() else "driver_read_timeout"
        elif capture is not None:
            capture.release()
        if display and "cv2" in locals():
            cv2.destroyAllWindows()
        record.update(finished_at_utc=utc_now(), processed_frames=processed,
                      capture_elapsed_seconds=capture_elapsed,
                      captured_frames=worker.received if worker else 0,
                      replaced_frames=worker.dropped if worker else 0)
        if samples:
            record["summary"] = {key: {"mean": float(np.mean([s[key] for s in samples])),
                                      "p95": float(np.percentile([s[key] for s in samples], 95))}
                                 for key in samples[0]}
            record["summary"]["latency_sample_frames"] = len(samples)
            record["summary"]["processing_fps"] = processed / capture_elapsed
        write_json(run / "session.json", record)
    return run
