"""Expand staged experiment settings without carrying weights between experiments."""
from __future__ import annotations

import math
import re

from .common import ROOT, object_sha256, read_json, resolve


def experiment_config(path, experiment_id):
    suite = read_json(path)
    entries = suite["experiments"]
    ids = [entry["id"] for entry in entries]
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate experiment IDs")
    entry = next((x for x in entries if x["id"] == experiment_id), None)
    if entry is None:
        raise ValueError(f"Unknown experiment: {experiment_id}")
    if not re.fullmatch(r"[A-Za-z0-9_-]{1,64}", experiment_id):
        raise ValueError("Unsafe experiment ID")
    if not entry.get("enabled", True):
        raise ValueError(f"{experiment_id} is disabled: record validation selection evidence before enabling it.")
    training = {**suite["training"], **entry["overrides"]}
    if training["optimizer"] not in ("MuSGD", "SGD", "AdamW", "Adam"):
        raise ValueError("Use an explicit supported optimizer; optimizer=auto is forbidden.")
    forbidden = {"model", "pretrained", "data", "resume", "fraction", "split", "project", "name", "exist_ok", "save_dir"}
    if forbidden & training.keys():
        raise ValueError("Model/data/output/resume settings are controlled by firelab, not experiment overrides.")
    for key in ("epochs", "batch", "imgsz", "nbs"):
        if type(training[key]) is not int or training[key] <= 0:
            raise ValueError(f"{key} must be a positive integer")
    if training["imgsz"] % 32:
        raise ValueError("imgsz must be a multiple of 32")
    for key in ("workers", "seed", "patience"):
        if type(training[key]) is not int or training[key] < 0:
            raise ValueError(f"Invalid {key}")
    if training["patience"] and training["patience"] <= training["epochs"]:
        raise ValueError("Disable early stopping or set patience above the epoch budget.")
    if not training["val"] or not training["plots"]:
        raise ValueError("Validation and plots are required")
    if training["device"] != "0":
        raise ValueError("Training requires explicit GPU 0; no silent CPU fallback or multi-GPU.")
    if training["amp"]:
        raise ValueError("This pinned protocol uses FP32; AMP requires a separately pinned check model/protocol.")
    for key in ("lr0", "lrf"):
        if not math.isfinite(training[key]) or training[key] <= 0:
            raise ValueError(f"Invalid {key}")
    evaluation = suite["evaluation"]
    for key in ("ap_conf", "operating_conf", "nms_iou", "match_iou"):
        if not math.isfinite(evaluation[key]) or not 0 < evaluation[key] <= 1:
            raise ValueError(f"Invalid evaluation {key}")
    for key in ("batch", "max_det"):
        if type(evaluation[key]) is not int or evaluation[key] <= 0:
            raise ValueError(f"Invalid evaluation {key}")
    for key in ("warmup_images", "max_case_images"):
        if type(evaluation[key]) is not int or evaluation[key] < 0:
            raise ValueError(f"Invalid evaluation {key}")
    if entry["stage"] == "combined" and not entry.get("selection_evidence"):
        raise ValueError("Combined experiment needs validation selection_evidence")
    dataset = read_json(resolve(suite["dataset_config"]))
    config = {"experiment": entry, "training": training, "evaluation": evaluation,
              "dataset_config": dataset, "suite_path": str(resolve(path).relative_to(ROOT))}
    config["config_sha256"] = object_sha256(config)
    return config
