"""Provenance and filesystem utilities shared by every command."""
from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import platform
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

ROOT = Path(__file__).resolve().parents[2]


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + "." + uuid4().hex + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)
                         + "\n", encoding="utf-8")
    temporary.replace(path)


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def object_sha256(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=True,
                                    allow_nan=False).encode()).hexdigest()


def resolve(path):
    return (ROOT / path).resolve()


def inside(path, parent):
    path, parent = Path(path).resolve(), Path(parent).resolve()
    if path == parent or not path.is_relative_to(parent):
        raise ValueError(f"Expected a path below {parent}: {path}")
    return path


def runs_root():
    """Allow explicit persistent run storage, e.g. a mounted Colab Drive directory."""
    configured = os.environ.get("FIRELAB_RUNS_DIR")
    return resolve(configured) if configured else ROOT / "experiments/runs"


def new_run(prefix, parent=None):
    parent = Path(parent) if parent else runs_root()
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    path = parent / f"{prefix}_{stamp}_{uuid4().hex[:8]}"
    path.mkdir(parents=True, exist_ok=False)
    return path


def git_state():
    def call(*args):
        result = subprocess.run(["git", "-C", str(ROOT), *args], capture_output=True,
                                text=True, encoding="utf-8", errors="replace", timeout=15)
        return result.stdout.strip() if result.returncode == 0 else None
    try:
        status = call("status", "--porcelain")
        diff = subprocess.run(["git", "-C", str(ROOT), "diff", "HEAD", "--binary"],
                              capture_output=True, timeout=15).stdout
        return {"commit": call("rev-parse", "HEAD"), "status": status,
                "dirty": bool(status), "tracked_diff_sha256": hashlib.sha256(diff).hexdigest()}
    except (OSError, subprocess.SubprocessError):
        return {"commit": None, "dirty": None}


def environment(cuda_probe=False):
    versions = {d.metadata["Name"]: d.version for d in importlib.metadata.distributions()
                if d.metadata.get("Name")}
    report = {"created_at_utc": utc_now(), "python": platform.python_version(),
              "platform": platform.platform(), "packages": dict(sorted(versions.items())),
              "git": git_state()}
    if cuda_probe:
        import torch
        report.update(torch=torch.__version__, cuda_runtime=torch.version.cuda,
                      cudnn=torch.backends.cudnn.version(), cuda_available=torch.cuda.is_available())
        if torch.cuda.is_available():
            report["gpus"] = [{"index": i, "name": torch.cuda.get_device_name(i),
                               "capability": list(torch.cuda.get_device_capability(i)),
                               "memory_bytes": torch.cuda.get_device_properties(i).total_memory}
                              for i in range(torch.cuda.device_count())]
            try:
                # Exercise a CUDA convolution and backward, not only device discovery.
                layer = torch.nn.Conv2d(3, 8, 3).cuda()
                layer(torch.ones(2, 3, 32, 32, device="cuda")).sum().backward()
                torch.cuda.synchronize()
                report["cuda_compute_probe"] = "passed"
            except Exception as exc:
                report["cuda_compute_probe"] = "failed"
                report["cuda_error"] = str(exc)
        try:
            result = subprocess.run(["nvidia-smi", "--query-gpu=name,driver_version,memory.total",
                                     "--format=csv,noheader"], capture_output=True, text=True, timeout=15)
            report["nvidia_smi"] = result.stdout.strip()
        except (OSError, subprocess.SubprocessError) as exc:
            report["nvidia_smi"] = str(exc)
    return report


def yolo_class():
    settings_dir = ROOT / "work/ultralytics"
    settings_dir.mkdir(parents=True, exist_ok=True)
    os.environ["YOLO_CONFIG_DIR"] = str(settings_dir)
    os.environ["YOLO_AUTOINSTALL"] = "false"
    os.environ["YOLO_OFFLINE"] = "true"
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    from ultralytics import YOLO, settings
    settings.update({"sync": False, "wandb": False, "mlflow": False,
                     "comet": False, "clearml": False, "neptune": False,
                     "tensorboard": False, "dvc": False, "hub": False})
    return YOLO
