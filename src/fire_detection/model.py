"""Pinned checkpoint download, validation and lazy inference adapter."""
from __future__ import annotations

import os
import urllib.request
from pathlib import Path
from uuid import uuid4

from .common import ROOT, file_sha256, utc_now, write_json


def model_path(spec, root=ROOT):
    path = (Path(root) / spec["local_path"]).resolve()
    allowed = (Path(root) / "models").resolve()
    if not path.is_relative_to(allowed):
        raise ValueError("Model path must stay inside models/.")
    return path


def verify_model(path, spec):
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError("Model missing. Run: python firelab.py fetch-model")
    if path.stat().st_size != spec["size_bytes"] or file_sha256(path) != spec["sha256"]:
        raise ValueError("Model checksum/size mismatch; refusing to load or overwrite it.")
    return path


def fetch_model(spec, root=ROOT):
    path = model_path(spec, root)
    if path.exists():
        return verify_model(path, spec)
    if not spec["url"].startswith("https://huggingface.co/"):
        raise ValueError("Expected a Hugging Face HTTPS checkpoint URL.")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + "." + uuid4().hex + ".part")
    try:
        request = urllib.request.Request(
            spec["url"], headers={"User-Agent": "drone-fire-detection/0.1"}
        )
        count = 0
        with urllib.request.urlopen(request, timeout=60) as response, temporary.open("xb") as out:
            while chunk := response.read(1024 * 1024):
                count += len(chunk)
                if count > spec["size_bytes"]:
                    raise ValueError("Download exceeds the expected checkpoint size.")
                out.write(chunk)
        verify_model(temporary, spec)
        # Another process may have finished while this one was downloading.
        if path.exists():
            verify_model(path, spec)
        else:
            temporary.replace(path)
        write_json(path.with_suffix(".metadata.json"), {
            **spec, "downloaded_at_utc": utc_now(),
        })
        return path
    finally:
        if temporary.exists():
            temporary.unlink()


class YoloDetector:
    def __init__(self, checkpoint, config, classes, root=ROOT):
        local_settings = Path(root) / "work/ultralytics"
        local_settings.mkdir(parents=True, exist_ok=True)
        os.environ["YOLO_CONFIG_DIR"] = str(local_settings)
        os.environ["YOLO_AUTOINSTALL"] = "false"
        try:
            import torch
            from ultralytics import YOLO, settings
        except ImportError as exc:
            raise RuntimeError("ML packages missing. Run scripts/setup.ps1 first.") from exc
        settings.update({"sync": False})
        torch.set_num_threads(config["threads"])
        self.model = YOLO(str(checkpoint), task="detect")
        actual = {int(key): value for key, value in self.model.names.items()}
        expected = {int(key): value for key, value in classes.items()}
        if actual != expected:
            raise ValueError("Checkpoint class names do not match the registered fire/smoke model.")
        self.config = config

    def predict(self, frame):
        result = self.model.predict(
            source=frame, imgsz=self.config["imgsz"], conf=self.config["confidence"],
            iou=self.config["iou"], device="cpu", batch=1, rect=False,
            max_det=self.config["max_detections"], verbose=False, save=False,
        )[0]
        output = []
        if result.boxes is None:
            return output
        height, width = frame.shape[:2]
        for box, score, class_id in zip(
            result.boxes.xyxy.cpu().tolist(), result.boxes.conf.cpu().tolist(),
            result.boxes.cls.cpu().tolist(),
        ):
            class_id = int(class_id)
            output.append({
                "class_id": class_id,
                "class_name": self.model.names[class_id],
                "confidence": float(score),
                "bbox_xyxy": [float(v) for v in box],
                "center_normalized_xy": [
                    (box[0] + box[2]) / (2 * width), (box[1] + box[3]) / (2 * height),
                ],
            })
        return output
