"""One saved-input run with provenance and explicit failure status."""
from __future__ import annotations

import csv
import json
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

from .common import ROOT, environment, file_sha256, git_state, timing_summary, utc_now, write_json
from .media import PreviewWriter, frames, resolve_sources
from .model import YoloDetector, model_path, verify_model


def summarize(run_dir, status, samples, counts, detected_frames, elapsed, reason):
    summary = {
        "status": status, "processed_frames": len(samples),
        "frames_with_detections": detected_frames,
        "detections_by_class": dict(counts),
        "model_call_timing": timing_summary(samples),
        "processing_wall_seconds": elapsed,
        "pipeline_fps": len(samples) / elapsed if elapsed and samples else None,
        "stop_reason": reason,
        "accuracy_metrics": None,
        "accuracy_note": "No ground-truth labels: detection counts are not accuracy.",
        "timing_note": "Offline processing; not camera-to-display or flight-control latency.",
        "video_time_note": "Nominal timestamp from frame_index / FPS; variable frame rate may differ.",
    }
    write_json(run_dir / "summary.json", summary)
    stats = summary["model_call_timing"]
    def number(value):
        return "not measured" if value is None else f"{value:.3f}"
    text = (
        "# Experiment run\n\n"
        f"- Status: {status}\n- Processed frames: {len(samples)}\n"
        f"- Detection counts: {dict(counts)}\n"
        f"- Model call mean / median / p95 (ms): {number(stats['mean_ms'])} / "
        f"{number(stats['median_ms'])} / {number(stats['p95_ms'])}\n"
        f"- Processing wall time (s): {number(elapsed)}\n"
        f"- Offline pipeline FPS: {number(summary['pipeline_fps'])}\n"
        f"- Stop reason: {reason}\n\n"
        "No ground-truth labels; no accuracy, precision, recall or mAP was measured.\n"
        "Warmup/loading excluded from timings. Offline FPS is not live latency.\n"
    )
    (run_dir / "summary.md").write_text(text, encoding="utf-8")
    return summary


def run_experiment(source, config, spec, *, root=ROOT, max_frames=None,
                   save_preview=False, detector_factory=YoloDetector):
    root = Path(root)
    if max_frames is not None and (type(max_frames) is not int or max_frames < 1):
        raise ValueError("max_frames must be positive.")
    files, kind = resolve_sources(source)
    run_id = config["experiment_id"] + "_" + datetime.now(timezone.utc).strftime(
        "%Y%m%dT%H%M%SZ"
    ) + "_" + uuid4().hex[:8]
    run_dir = root / "experiments/runs" / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    system = environment()
    metadata = {
        "run_id": run_id, "started_at_utc": utc_now(), "status": "running",
        "config": config, "options": {"max_frames": max_frames, "save_preview": save_preview},
        "model": spec, "source_kind": kind, "inputs": [],
        "git": git_state(root), "environment": system,
        "warmup_excluded": True,
    }
    write_json(run_dir / "run.json", metadata)
    (run_dir / "packages.txt").write_text(
        "\n".join(f"{key}=={value}" for key, value in system["packages"].items()) + "\n",
        encoding="utf-8",
    )
    samples, counts, detected_frames = [], Counter(), 0
    preview = None
    detector = None
    iterator = None
    started = None
    reason = "not_started"
    try:
        for path in files:
            metadata["inputs"].append({
                "name": path.relative_to(root).as_posix() if path.is_relative_to(root) else path.name,
                "size_bytes": path.stat().st_size, "sha256": file_sha256(path),
            })
        checkpoint = verify_model(model_path(spec, root), spec)
        write_json(run_dir / "run.json", metadata)
        detector = detector_factory(checkpoint, config, spec["classes"], root=root)
        import numpy as np
        warmup = np.zeros((config["imgsz"], config["imgsz"], 3), dtype=np.uint8)
        for _ in range(config["warmup_frames"]):
            detector.predict(warmup)
        runtime_settings = getattr(detector, "runtime_settings", None)
        if callable(runtime_settings):
            metadata["runtime"] = runtime_settings()
            write_json(run_dir / "run.json", metadata)
        if save_preview:
            preview = PreviewWriter(run_dir / "preview", kind, config["frame_stride"])
        iterator = frames(files, kind, config["frame_stride"])
        started = time.perf_counter()
        reason = "end_of_input"
        with (run_dir / "detections.jsonl").open("w", encoding="utf-8") as detections_file, (
            run_dir / "timings.csv"
        ).open("w", encoding="utf-8", newline="") as timing_file:
            writer = csv.DictWriter(
                timing_file, fieldnames=["source", "frame_index", "video_time_s", "model_call_ms"]
            )
            writer.writeheader()
            for info in iterator:
                before = time.perf_counter()
                detections = detector.predict(info["image"])
                duration_ms = (time.perf_counter() - before) * 1000
                record = {
                    "source": info["source"], "frame_index": info["frame_index"],
                    "video_time_s": info["video_time_s"],
                    "image_width": int(info["image"].shape[1]),
                    "image_height": int(info["image"].shape[0]),
                    "detections": detections,
                }
                detections_file.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
                writer.writerow({**{k: info[k] for k in ("source", "frame_index", "video_time_s")},
                                 "model_call_ms": duration_ms})
                samples.append(duration_ms)
                counts.update(d["class_name"] for d in detections)
                detected_frames += bool(detections)
                if preview:
                    preview.add(info, detections, len(samples))
                if max_frames is not None and len(samples) >= max_frames:
                    reason = "max_frames"
                    break
        if not samples:
            raise ValueError("No frames processed.")
        metadata["status"] = "completed"
    except KeyboardInterrupt:
        metadata["status"] = "interrupted"
        reason = "keyboard_interrupt"
        raise
    except Exception as exc:
        metadata["status"] = "failed"
        metadata["error"] = {"type": type(exc).__name__, "message": str(exc)}
        reason = "error"
        raise
    finally:
        runtime_settings = getattr(detector, "runtime_settings", None)
        if callable(runtime_settings):
            metadata["runtime"] = runtime_settings()
        if iterator is not None:
            iterator.close()
        if preview is not None:
            preview.close()
        elapsed = time.perf_counter() - started if started is not None else None
        metadata["finished_at_utc"] = utc_now()
        metadata["stop_reason"] = reason
        write_json(run_dir / "run.json", metadata)
        summarize(run_dir, metadata["status"], samples, counts, detected_frames, elapsed, reason)
        print("Run results:", run_dir)
    return run_dir
