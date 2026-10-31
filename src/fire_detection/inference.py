"""Offline image/video inference with input and checkpoint provenance."""
from pathlib import Path

from .common import environment, file_sha256, new_run, read_json, resolve, write_json, yolo_class, ROOT
from .dataset import IMAGE_SUFFIXES
from .model import spec_for, verify_model


def predict(source, training_run=None, confidence=.25, iou=.7, imgsz=640, device="0"):
    source = resolve(source)
    if not source.is_file() or source.suffix.lower() not in IMAGE_SUFFIXES | {".mp4", ".avi", ".mov", ".mkv"}:
        raise ValueError("Supply one saved image or video file")
    if not 0 < confidence <= 1 or not 0 < iou <= 1:
        raise ValueError("confidence and iou must be in (0,1]")
    if training_run:
        training = read_json(resolve(training_run) / "training.json")
        if training["status"] != "completed":
            raise ValueError("Training has not completed")
        checkpoint = Path(training["best_path"])
        if file_sha256(checkpoint) != training["best_sha256"]:
            raise ValueError("Checkpoint checksum mismatch")
    else:
        spec = spec_for("published")
        checkpoint = verify_model(resolve(spec["local_path"]), spec)
    run = new_run("inference", ROOT / "outputs")
    config = {"confidence": confidence, "iou": iou, "imgsz": imgsz, "device": device}
    record = {"input": str(source), "input_sha256": file_sha256(source), "model": str(checkpoint),
              "model_sha256": file_sha256(checkpoint), "settings": config, "environment": environment(cuda_probe=device != "cpu"),
              "status": "running"}
    write_json(run / "inference.json", record)
    try:
        model = yolo_class()(str(checkpoint), task="detect")
        if model.names != {0: "smoke", 1: "fire"}:
            raise ValueError("Expected smoke/fire classes")
        import json
        count = 0
        with (run / "detections.jsonl").open("w", encoding="utf-8") as handle:
            for count, result in enumerate(model.predict(source=str(source), stream=True, conf=confidence, iou=iou,
                    imgsz=imgsz, device=device, half=False, rect=False, save=True, verbose=False,
                    project=str(run), name="media", exist_ok=False), 1):
                handle.write(json.dumps({"frame": count, "detections": result.summary(normalize=True)}, allow_nan=False) + "\n")
        record.update(status="completed", frames=count)
    except BaseException as exc:
        record.update(status="failed", error=str(exc))
        raise
    finally:
        write_json(run / "inference.json", record)
    return run
