"""Fresh COCO initialization, observable effective settings, and same-run resume."""
from __future__ import annotations

import json
import os
import time
import traceback
from pathlib import Path

from .common import (ROOT, environment, file_sha256, inside, new_run, read_json,
                     resolve, runs_root, utc_now, write_json, yolo_class)
from .config import experiment_config
from .dataset import verify_dataset, verify_unchanged_dataset
from .model import spec_for, verify_model


def train(suite_path=None, experiment_id=None, resume_run=None):
    YOLO = yolo_class()
    import torch
    from ultralytics.optim import MuSGD  # Fail visibly if the pinned optimizer is unavailable.

    if resume_run:
        run = inside(resolve(resume_run), runs_root())
        record = read_json(run / "training.json")
        if record["status"] not in ("failed", "interrupted", "running"):
            raise ValueError("Only an unfinished experiment can be resumed.")
        if record["status"] == "running":
            import psutil
            previous = record["attempts"][-1]
            if "pid" not in previous:
                raise RuntimeError("Cannot establish whether the previous process is active; inspect the run first.")
            try:
                process = psutil.Process(previous["pid"])
                if abs(process.create_time() - previous["process_created_at"]) < .01:
                    raise RuntimeError("This training process is still active; refusing concurrent resume.")
            except psutil.NoSuchProcess:
                pass
        config = record["config"]
        checkpoint = run / "train/weights/last.pt"
        registered = read_json(run / "checkpoints.json")
        if file_sha256(checkpoint) != registered["last"]["sha256"]:
            raise ValueError("Resume checkpoint checksum mismatch")
    else:
        config = experiment_config(suite_path, experiment_id)
        spec = spec_for("coco")
        checkpoint = verify_model(resolve(spec["local_path"]), spec)
        run = None
        record = {"kind": "training", "experiment_id": experiment_id, "config": config,
                  "initial_model": spec, "attempts": [], "created_at_utc": utc_now()}
    dataset_path = resolve(config["dataset_config"]["prepared_root"])
    dataset = verify_dataset(dataset_path)
    if dataset["config"] != config["dataset_config"]:
        raise ValueError("Prepared dataset does not match configured dataset policy")
    if resume_run and record["dataset_fingerprint"] != dataset["fingerprint"]:
        raise ValueError("Resume dataset fingerprint mismatch")
    env = environment(cuda_probe=True)
    if env.get("cuda_compute_probe") != "passed":
        raise RuntimeError("CUDA convolution/backward probe failed; GPU training was not started.")
    if run is None:
        run = new_run(experiment_id)
    record["dataset_fingerprint"] = dataset["fingerprint"]
    attempt_id = len(record["attempts"]) + 1
    import psutil
    attempt = {"number": attempt_id, "started_at_utc": utc_now(), "environment": env,
               "pid": os.getpid(), "process_created_at": psutil.Process().create_time(),
               "resume": bool(resume_run), "checkpoint": str(checkpoint),
               "checkpoint_sha256": file_sha256(checkpoint)}
    record["attempts"].append(attempt)
    record["status"] = "running"
    write_json(run / "training.json", record)
    effective_path = run / f"attempt-{attempt_id}-effective.json"
    learning_rates = run / f"attempt-{attempt_id}-learning-rates.jsonl"
    torch.cuda.reset_peak_memory_stats(0)
    start = time.perf_counter()

    def effective(trainer):
        # Ultralytics scans images while building loaders; detect any automatic rewrite now.
        verify_unchanged_dataset(dataset_path, record["dataset_fingerprint"])
        if trainer.batch_size != config["training"]["batch"]:
            raise RuntimeError("Effective batch changed; define a separate experiment instead.")
        groups = [{"lr": float(group["lr"]), "initial_lr": float(group.get("initial_lr", group["lr"])),
                   "weight_decay": float(group.get("weight_decay", 0)),
                   "parameter_count": sum(p.numel() for p in group["params"]),
                   "use_muon": bool(group.get("use_muon", False))} for group in trainer.optimizer.param_groups]
        write_json(effective_path, {"args": {k: str(v) if isinstance(v, Path) else v for k, v in vars(trainer.args).items()},
                   "optimizer_class": type(trainer.optimizer).__name__, "parameter_groups": groups,
                   "accumulate": int(trainer.accumulate), "scheduler": type(trainer.scheduler).__name__,
                   "warmup_note": "First batch is logged separately; warmup changes actual group learning rates."})

    batch_in_epoch = [0]

    def epoch_start(trainer):
        batch_in_epoch[0] = 0

    def batch_end(trainer):
        if trainer.batch_size != config["training"]["batch"]:
            raise RuntimeError("Effective batch changed")
        if batch_in_epoch[0] == 0:
            row = {"epoch": int(trainer.epoch + 1), "sample": "first_batch_after_warmup_update",
                   "lr": [float(g["lr"]) for g in trainer.optimizer.param_groups],
                   "accumulate": int(trainer.accumulate)}
            with learning_rates.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(row, allow_nan=False) + "\n")
        batch_in_epoch[0] += 1

    def saved(trainer):
        entries = {}
        for name in ("last", "best"):
            path = Path(getattr(trainer, name))
            if path.is_file():
                entries[name] = {"path": str(path), "sha256": file_sha256(path), "epoch": int(trainer.epoch + 1)}
        write_json(run / "checkpoints.json", entries)

    try:
        model = YOLO(str(checkpoint), task="detect")
        model.add_callback("on_pretrain_routine_end", effective)
        model.add_callback("on_train_epoch_start", epoch_start)
        model.add_callback("on_train_batch_end", batch_end)
        model.add_callback("on_model_save", saved)
        if resume_run:
            model.train(resume=str(checkpoint), device="0")
        else:
            model.train(**config["training"], data=str(dataset_path / "data.yaml"),
                        project=str(run), name="train", exist_ok=False)
        # Ultralytics strips optimizer at completion, changing checkpoint hashes.
        verify_unchanged_dataset(dataset_path, record["dataset_fingerprint"])
        saved(model.trainer)
        record["best_path"] = str(model.trainer.best)
        record["best_sha256"] = file_sha256(model.trainer.best)
        record["status"] = "completed"
        record["epochs_completed"] = int(model.trainer.epoch + 1)
    except BaseException as exc:
        record["status"] = "interrupted" if isinstance(exc, KeyboardInterrupt) else "failed"
        attempt["error"] = str(exc)
        attempt["traceback"] = traceback.format_exc()
        raise
    finally:
        torch.cuda.synchronize()
        attempt["wall_seconds"] = time.perf_counter() - start
        attempt["finished_at_utc"] = utc_now()
        attempt["peak_allocated_bytes"] = torch.cuda.max_memory_allocated(0)
        attempt["peak_reserved_bytes"] = torch.cuda.max_memory_reserved(0)
        record["training_wall_seconds"] = sum(a["wall_seconds"] for a in record["attempts"])
        write_json(run / "training.json", record)
        print(f"Training record: {run / 'training.json'}", flush=True)
    return run
