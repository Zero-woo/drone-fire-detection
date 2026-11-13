"""Verify frozen model identity and run batch-one CPU deployment backends."""
import os
from pathlib import Path
import shutil

from .common import (ROOT, environment, file_sha256, inside, new_run,
                     object_sha256, read_json, resolve, utc_now, write_json, yolo_class)


def frozen_model(config):
    revision_path = resolve(config["revision"])
    revision = read_json(revision_path)
    digest = object_sha256({k: v for k, v in revision.items() if k != "revision_sha256"})
    if digest != revision["revision_sha256"] or digest != config["revision_sha256"]:
        raise ValueError("Frozen model revision checksum mismatch")
    checkpoint = inside(revision_path.parent / revision["weights"], revision_path.parent)
    if file_sha256(checkpoint) != revision["checkpoint_sha256"]:
        raise ValueError("Frozen checkpoint checksum mismatch")
    if revision["classes"] != {"0": "smoke", "1": "fire"}:
        raise ValueError("Expected frozen smoke/fire classes")
    return revision, checkpoint


def export_openvino(config):
    revision, checkpoint = frozen_model(config)
    run = new_run("live-model", ROOT / "outputs")
    record = {"status": "exporting", "created_at_utc": utc_now(),
              "source_revision_sha256": revision["revision_sha256"],
              "source_checkpoint_sha256": revision["checkpoint_sha256"],
              "inference": revision["inference"], "environment": environment()}
    write_json(run / "deployment.json", record)
    try:
        copied = run / "best.pt"
        shutil.copyfile(checkpoint, copied)
        if file_sha256(copied) != revision["checkpoint_sha256"]:
            raise ValueError("Export input checksum mismatch")
        model = yolo_class()(str(copied), task="detect")
        exported = Path(model.export(format="openvino", imgsz=revision["inference"]["imgsz"],
                                    batch=1, dynamic=False, half=False, int8=False,
                                    nms=False, device="cpu"))
        exported = inside(exported, run)
        files = {p.relative_to(run).as_posix(): file_sha256(p)
                 for p in sorted(exported.rglob("*")) if p.is_file()}
        xml_files = list(exported.glob("*.xml"))
        if len(xml_files) != 1 or not xml_files[0].with_suffix(".bin").is_file():
            raise ValueError("Expected one OpenVINO XML/BIN model pair")
        record.update(status="completed", xml=xml_files[0].relative_to(run).as_posix(),
                      files=files, finished_at_utc=utc_now())
        record["deployment_sha256"] = object_sha256(record)
    except BaseException as exc:
        record.update(status="failed", error=str(exc))
        raise
    finally:
        write_json(run / "deployment.json", record)
    return run / "deployment.json"


def verified_deployment(path, revision):
    path = resolve(path)
    deployment = read_json(path)
    digest = object_sha256({k: v for k, v in deployment.items() if k != "deployment_sha256"})
    if deployment.get("status") != "completed" or digest != deployment.get("deployment_sha256"):
        raise ValueError("OpenVINO deployment checksum/status mismatch")
    if (deployment["source_revision_sha256"] != revision["revision_sha256"]
            or deployment["source_checkpoint_sha256"] != revision["checkpoint_sha256"]
            or deployment["inference"] != revision["inference"]):
        raise ValueError("OpenVINO deployment differs from frozen model/settings")
    for name, expected in deployment["files"].items():
        if file_sha256(inside(path.parent / name, path.parent)) != expected:
            raise ValueError(f"OpenVINO artifact checksum mismatch: {name}")
    required = {deployment["xml"], Path(deployment["xml"]).with_suffix(".bin").as_posix()}
    if not required.issubset(deployment["files"]):
        raise ValueError("Unregistered OpenVINO XML/BIN pair")
    return deployment, inside(path.parent / deployment["xml"], path.parent)


class CpuDetector:
    def __init__(self, revision, checkpoint, threads, deployment_path=None):
        os.environ["OMP_NUM_THREADS"] = str(threads)
        import torch
        self.settings = revision["inference"]
        self.names = {int(k): v for k, v in revision["classes"].items()}
        self.model = None
        self.compiled = None
        model_class = yolo_class()
        self.runtime = {"backend": "pytorch", "device": "cpu", "threads": threads,
                        "precision": "FP32"}
        if deployment_path:
            deployment, xml = verified_deployment(deployment_path, revision)
            import openvino as ov
            core = ov.Core()
            self.compiled = core.compile_model(core.read_model(xml), "CPU", {
                "PERFORMANCE_HINT": "LATENCY", "INFERENCE_PRECISION_HINT": "f32",
                "INFERENCE_NUM_THREADS": threads,
            })
            self.runtime.update(backend="openvino", deployment_sha256=deployment["deployment_sha256"],
                                execution_devices=self.compiled.get_property("EXECUTION_DEVICES"))
        else:
            self.model = model_class(str(checkpoint), task="detect")
            if self.model.names != self.names:
                raise ValueError("Unexpected model classes")
        import numpy as np
        blank = np.zeros((self.settings["imgsz"], self.settings["imgsz"], 3), dtype=np.uint8)
        self.predict(blank)
        # Ultralytics initializes the PyTorch thread count during its first predict call.
        torch.set_num_threads(threads)
        for _ in range(2):
            self.predict(blank)

    def predict(self, frame):
        settings = self.settings
        if self.model is not None:
            return self.model.predict(frame, device="cpu", imgsz=settings["imgsz"],
                                      conf=settings["confidence"], iou=settings["nms_iou"],
                                      max_det=settings["max_det"], half=False, rect=False,
                                      batch=1, verbose=False, save=False)[0]
        import numpy as np
        import torch
        from ultralytics.data.augment import LetterBox
        from ultralytics.engine.results import Results
        from ultralytics.utils import ops
        from ultralytics.utils.nms import non_max_suppression
        image = LetterBox((settings["imgsz"], settings["imgsz"]), auto=False)(image=frame)
        tensor = np.ascontiguousarray(image[..., ::-1].transpose(2, 0, 1))[None].astype(np.float32) / 255
        output = self.compiled(tensor)[self.compiled.output(0)]
        boxes = non_max_suppression(torch.from_numpy(output), conf_thres=settings["confidence"],
                                   iou_thres=settings["nms_iou"], nc=2, max_det=settings["max_det"])[0]
        boxes[:, :4] = ops.scale_boxes(tensor.shape[2:], boxes[:, :4], frame.shape)
        return Results(frame, path="live-frame", names=self.names, boxes=boxes)
