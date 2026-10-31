"""Validation metrics, fixed-threshold failure cases, and frozen final-test evaluation."""
from __future__ import annotations

import copy
import statistics
import time
from collections import Counter
from pathlib import Path

from .common import (ROOT, environment, file_sha256, new_run, object_sha256, read_json,
                     resolve, utc_now, write_json, yolo_class)
from .config import experiment_config
from .dataset import verify_dataset
from .model import spec_for, verify_model


def iou_xyxy(a, b):
    intersection = max(0., min(a[2], b[2]) - max(a[0], b[0])) * max(0., min(a[3], b[3]) - max(a[1], b[1]))
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - intersection
    return intersection / union if union > 0 else 0.


def match_boxes(predictions, targets, threshold):
    """Class-aware greedy one-to-one matching, descending IoU (independent operating metrics)."""
    candidates = sorted(((iou_xyxy(p[1:], t[1:]), pi, ti)
                         for pi, p in enumerate(predictions) for ti, t in enumerate(targets)
                         if int(p[0]) == int(t[0])), reverse=True)
    used_p, used_t = set(), set()
    for overlap, pi, ti in candidates:
        if overlap >= threshold and pi not in used_p and ti not in used_t:
            used_p.add(pi)
            used_t.add(ti)
    return used_p, used_t


def targets_xyxy(boxes, width, height):
    return [[int(c), (x - w / 2) * width, (y - h / 2) * height,
             (x + w / 2) * width, (y + h / 2) * height] for c, x, y, w, h in boxes]


def operating_analysis(model, dataset_path, manifest, split, config, settings, output):
    import cv2
    records = {Path(row["prepared_image"]).name: row for row in manifest["records"] if row["split"] == split}
    first = list(records.values())[:settings["warmup_images"]]
    kwargs = {"imgsz": config["training"]["imgsz"], "device": config["training"]["device"],
              "conf": settings["operating_conf"], "iou": settings["nms_iou"],
              "max_det": settings["max_det"], "batch": 1, "rect": False,
              "half": False, "verbose": False, "save": False}
    for row in first:
        model.predict(source=str(dataset_path / row["prepared_image"]), **kwargs)
    counts, seen, cases, samples = Counter(), set(), [], []
    start = time.perf_counter()
    for result in model.predict(source=str(dataset_path / f"{split}.txt"), stream=True, **kwargs):
        name = Path(result.path).name
        if name not in records or name in seen:
            raise ValueError(f"Unexpected/repeated inference image: {result.path}")
        seen.add(name)
        row = records[name]
        boxes = result.boxes
        predictions = [[int(c), *map(float, b)] for c, b in zip(boxes.cls.cpu().tolist(), boxes.xyxy.cpu().tolist())]
        targets = targets_xyxy(row["boxes"], row["width"], row["height"])
        matched_p, matched_t = match_boxes(predictions, targets, settings["match_iou"])
        for pi, p in enumerate(predictions):
            counts[f"{'tp' if pi in matched_p else 'fp'}_{int(p[0])}"] += 1
        for ti, t in enumerate(targets):
            if ti not in matched_t:
                counts[f"fn_{int(t[0])}"] += 1
        if not targets:
            counts["negative_images"] += 1
            counts["negative_fp_detections"] += len(predictions)
            counts["negative_images_with_fp"] += bool(predictions)
        false_positive = len(matched_p) < len(predictions)
        false_negative = len(matched_t) < len(targets)
        if false_positive or false_negative:
            case = {"image": name, "fp_classes": [int(p[0]) for i, p in enumerate(predictions) if i not in matched_p],
                    "fn_classes": [int(t[0]) for i, t in enumerate(targets) if i not in matched_t]}
            cases.append(case)
            categories = (["fp"] if false_positive else []) + (["fn"] if false_negative else [])
            for category in categories:
                key = f"saved_{category}"
                if counts[key] < settings["max_case_images"]:
                    canvas = result.plot()  # predictions plus green ground truth below
                    for cls, left, top, right, bottom in targets:
                        cv2.rectangle(canvas, (int(left), int(top)), (int(right), int(bottom)), (0, 255, 0), 2)
                        cv2.putText(canvas, f"GT {model.names[cls]}", (int(left), max(15, int(top))),
                                    cv2.FONT_HERSHEY_SIMPLEX, .5, (0, 255, 0), 1)
                    target_dir = output / "cases" / category
                    target_dir.mkdir(parents=True, exist_ok=True)
                    if not cv2.imwrite(str(target_dir / (Path(name).stem + ".jpg")), canvas):
                        raise OSError("Could not write annotated case image")
                    counts[key] += 1
        samples.append({k: float(v) for k, v in result.speed.items()})
    if seen != set(records):
        raise ValueError("Inference did not cover the complete held-out split")
    duration = time.perf_counter() - start
    classes = {}
    for c, name in enumerate(("smoke", "fire")):
        tp, fp, fn = (counts[f"{key}_{c}"] for key in ("tp", "fp", "fn"))
        precision = tp / (tp + fp) if tp + fp else 0.
        recall = tp / (tp + fn) if tp + fn else 0.
        classes[name] = {"tp": tp, "fp": fp, "fn": fn, "precision": precision, "recall": recall,
                         "f1": 2 * precision * recall / (precision + recall) if precision + recall else 0.}
    inference_ms = statistics.mean(s["inference"] for s in samples)
    pipeline_ms = statistics.mean(s["preprocess"] + s["inference"] + s["postprocess"] for s in samples)
    write_json(output / "failure-cases.json", cases)
    return {"settings": settings, "matching": "class-aware greedy descending IoU, one-to-one",
            "classes": classes, "images": len(seen), "negative_images": counts["negative_images"],
            "negative_fp_detections": counts["negative_fp_detections"],
            "negative_images_with_fp": counts["negative_images_with_fp"],
            "inference_ms_per_image": inference_ms, "inference_fps": 1000 / inference_ms if inference_ms > 0 else None,
            "pipeline_ms_per_image": pipeline_ms, "pipeline_fps": 1000 / pipeline_ms if pipeline_ms > 0 else None,
            "analysis_wall_seconds_including_io_and_case_export": duration,
            "timing_note": "Batch=1, FP32; warmed before measurement. Inference FPS excludes I/O, preprocessing and NMS."}


def evaluate_checkpoint(checkpoint, config, dataset_fingerprint, split="val", selection=None, model_role="trained", training_run=None):
    if split == "test" and selection is None:
        raise ValueError("Test evaluation requires a frozen final selection")
    if split not in ("val", "test"):
        raise ValueError("Unsupported evaluation split")
    import torch
    dataset_path = resolve(config["dataset_config"]["prepared_root"])
    manifest = verify_dataset(dataset_path)
    if manifest["fingerprint"] != dataset_fingerprint:
        raise ValueError("Evaluation dataset fingerprint differs from training/selection")
    env = environment(cuda_probe=True)
    if env.get("cuda_compute_probe") != "passed":
        raise RuntimeError("CUDA compute is unavailable")
    run = new_run(f"{split}-{config['experiment']['id']}")
    settings = config["evaluation"]
    checkpoint = Path(checkpoint).resolve()
    record = {"kind": "evaluation", "split": split, "status": "running", "created_at_utc": utc_now(),
              "experiment_id": config["experiment"]["id"], "model_role": model_role,
              "training_run": str(training_run) if training_run else None, "config": config,
              "checkpoint": str(checkpoint), "checkpoint_sha256": file_sha256(checkpoint),
              "dataset_fingerprint": manifest["fingerprint"], "dataset_summary": manifest["summary"],
              "environment": env, "selection": selection,
              "published_split_identity_verified": config["dataset_config"]["published_split_identity_verified"]}
    write_json(run / "evaluation.json", record)
    torch.cuda.reset_peak_memory_stats(0)
    start = time.perf_counter()
    try:
        model = yolo_class()(str(checkpoint), task="detect")
        if {int(k): v for k, v in model.names.items()} != {0: "smoke", 1: "fire"}:
            raise ValueError("Checkpoint classes must be 0=smoke, 1=fire")
        captured = {}

        def capture_validator(validator):
            captured["validator"] = validator

        model.add_callback("on_val_end", capture_validator)
        metrics = model.val(data=str(dataset_path / ("test.yaml" if split == "test" else "data.yaml")),
                            split=split, imgsz=config["training"]["imgsz"], batch=settings["batch"],
                            device=config["training"]["device"], workers=0, conf=settings["ap_conf"],
                            iou=settings["nms_iou"], max_det=settings["max_det"], half=False,
                            rect=False, augment=False, plots=True, save_json=False,
                            seed=config["training"]["seed"], deterministic=True,
                            project=str(run), name="ap", exist_ok=False)
        box = metrics.box
        classes = {name: {"precision": None, "recall": None, "ap50": None, "ap50_95": None, "f1": None}
                   for name in ("smoke", "fire")}
        for i, c in enumerate(box.ap_class_index):
            classes[model.names[int(c)]] = {"precision": float(box.p[i]), "recall": float(box.r[i]),
                "ap50": float(box.ap50[i]), "ap50_95": float(box.ap[i]), "f1": float(box.f1[i])}
        record["metrics"] = {"precision": float(box.mp), "recall": float(box.mr),
            "map50": float(box.map50), "map50_95": float(box.map),
            "f1_macro": statistics.mean(float(v) for v in box.f1) if len(box.f1) else 0.,
            "classes": classes, "point_note": "Ultralytics P/R/F1 use the best mean-F1 confidence on this split; AP integrates confidence.",
            "speed_ms_per_image": {k: float(v) for k, v in metrics.speed.items()}}
        matrix = captured["validator"].confusion_matrix.matrix
        record["confusion_matrix"] = {"matrix": matrix.tolist(), "axes": "rows=predicted, columns=true; smoke,fire,background",
                                       "note": "Ultralytics AP-validation confusion matrix; fixed operating metrics are separate."}
        record["operating"] = operating_analysis(model, dataset_path, manifest, split, config, settings, run)
        record["status"] = "completed"
    except BaseException as exc:
        record["status"] = "failed"
        record["error"] = str(exc)
        raise
    finally:
        torch.cuda.synchronize()
        record["wall_seconds"] = time.perf_counter() - start
        record["peak_allocated_bytes"] = torch.cuda.max_memory_allocated(0)
        record["peak_reserved_bytes"] = torch.cuda.max_memory_reserved(0)
        write_json(run / "evaluation.json", record)
        print(f"Evaluation record: {run / 'evaluation.json'}", flush=True)
    return run


def evaluate_validation(training_run=None, published=False, suite_path=None, operating_conf=None):
    if published:
        config = experiment_config(suite_path or ROOT / "configs/experiments.json", "E0")
        spec = spec_for("published")
        checkpoint = verify_model(resolve(spec["local_path"]), spec)
        config["experiment"] = {"id": "published", "stage": "reference", "reference": None}
        manifest = verify_dataset(resolve(config["dataset_config"]["prepared_root"]))
        dataset_fingerprint = manifest["fingerprint"]
    else:
        training_run = resolve(training_run)
        record = read_json(training_run / "training.json")
        if record["status"] != "completed":
            raise ValueError("Validation requires a completed training run")
        config = copy.deepcopy(record["config"])
        checkpoint = Path(record["best_path"])
        if file_sha256(checkpoint) != record["best_sha256"]:
            raise ValueError("Trained checkpoint checksum mismatch")
        dataset_fingerprint = record["dataset_fingerprint"]
    if operating_conf is not None:
        if not 0 < operating_conf <= 1:
            raise ValueError("operating_conf must be in (0,1]")
        config["evaluation"]["operating_conf"] = operating_conf
    return evaluate_checkpoint(checkpoint, config, dataset_fingerprint, model_role="published" if published else "trained",
                               training_run=training_run)


def freeze_selection(evaluation_dir, rationale):
    evaluation_path = resolve(evaluation_dir) / "evaluation.json"
    record = read_json(evaluation_path)
    if record["status"] != "completed" or record["split"] != "val" or record["model_role"] != "trained":
        raise ValueError("Final selection requires completed validation of a trained model")
    if not rationale.strip():
        raise ValueError("Record the validation selection rationale")
    selection = {"created_at_utc": utc_now(), "config": record["config"],
                 "checkpoint": record["checkpoint"], "checkpoint_sha256": record["checkpoint_sha256"],
                 "dataset_fingerprint": record["dataset_fingerprint"], "validation_record": str(evaluation_path),
                 "validation_record_sha256": file_sha256(evaluation_path), "rationale": rationale,
                 "model_role": record["model_role"], "training_run": record["training_run"]}
    if file_sha256(Path(selection["checkpoint"])) != selection["checkpoint_sha256"]:
        raise ValueError("Checkpoint changed after validation")
    selection["selection_sha256"] = object_sha256(selection)
    output = new_run("final-selection", ROOT / "outputs") / "selection.json"
    write_json(output, selection)
    return output


def evaluate_test(selection_path):
    selection = read_json(selection_path)
    digest = selection.pop("selection_sha256")
    if object_sha256(selection) != digest:
        raise ValueError("Final selection checksum mismatch")
    selection["selection_sha256"] = digest
    if file_sha256(selection["validation_record"]) != selection["validation_record_sha256"]:
        raise ValueError("Frozen validation record changed")
    if file_sha256(selection["checkpoint"]) != selection["checkpoint_sha256"]:
        raise ValueError("Final checkpoint checksum mismatch")
    return evaluate_checkpoint(selection["checkpoint"], selection["config"], selection["dataset_fingerprint"],
                               split="test", selection=selection, training_run=selection["training_run"])
