"""Standard-library utilities; usable before ML dependencies are installed."""
from __future__ import annotations

import hashlib
import importlib.metadata
import json
import math
import os
import platform
import re
import statistics
import subprocess
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = ROOT / "configs/001_pretrained_cpu.json"
REGISTRY = ROOT / "configs/model_registry.json"


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def write_json(path, value):
    Path(path).write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_config(path):
    value = read_json(path)
    expected = {
        "experiment_id", "imgsz", "confidence", "iou", "device", "threads",
        "warmup_frames", "frame_stride", "max_detections",
    }
    if not isinstance(value, dict) or set(value) != expected:
        raise ValueError("Config must contain exactly: " + ", ".join(sorted(expected)))
    if not isinstance(value["experiment_id"], str) or not re.fullmatch(
        r"[a-zA-Z0-9_-]{1,64}", value["experiment_id"]
    ):
        raise ValueError("experiment_id must use letters, numbers, underscore or hyphen.")
    for key in ("imgsz", "threads", "warmup_frames", "frame_stride", "max_detections"):
        if type(value[key]) is not int:
            raise ValueError(key + " must be an integer.")
        minimum = 0 if key == "warmup_frames" else 1
        if value[key] < minimum:
            raise ValueError(key + " is too small.")
    if value["imgsz"] % 32:
        raise ValueError("imgsz must be a multiple of 32.")
    for key in ("confidence", "iou"):
        number = value[key]
        if type(number) not in (int, float) or not math.isfinite(number) or not 0 < number <= 1:
            raise ValueError(key + " must be in (0, 1].")
    if value["device"] != "cpu":
        raise ValueError("EXP-001 supports only device=cpu.")
    return value


def git_state(root):
    def call(*args):
        result = subprocess.run(
            ["git", "-C", str(root), *args], capture_output=True,
            text=True, encoding="utf-8", errors="replace", timeout=10,
        )
        return result.stdout.strip() if result.returncode == 0 else None
    try:
        head = call("rev-parse", "HEAD")
        status = call("status", "--porcelain", "--untracked-files=normal")
        return {"commit": head, "dirty": bool(status) if status is not None else None}
    except (OSError, subprocess.SubprocessError):
        return {"commit": None, "dirty": None}


def package_versions():
    versions = {}
    for distribution in importlib.metadata.distributions():
        name = distribution.metadata.get("Name")
        if name:
            versions[name.lower().replace("_", "-")] = distribution.version
    return dict(sorted(versions.items()))


def environment():
    packages = package_versions()
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "processor": os.environ.get("PROCESSOR_IDENTIFIER", platform.processor()),
        "logical_cpus": os.cpu_count(),
        "packages": packages,
    }


def timing_summary(samples_ms):
    if not samples_ms:
        return {"count": 0, "mean_ms": None, "median_ms": None, "p95_ms": None,
                "model_call_fps": None}
    values = sorted(samples_ms)
    mean = statistics.mean(values)
    return {
        "count": len(values),
        "mean_ms": mean,
        "median_ms": statistics.median(values),
        "p95_ms": values[math.ceil(0.95 * len(values)) - 1],
        "model_call_fps": 1000 / mean if mean > 0 else None,
    }
