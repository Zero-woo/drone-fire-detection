"""Command-line entry points with actionable errors."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .common import DEFAULT_CONFIG, REGISTRY, ROOT, environment, git_state, load_config, read_json, write_json
from .model import fetch_model, model_path, verify_model


def doctor():
    system = environment()
    required = ("torch", "torchvision", "ultralytics", "opencv-python", "numpy")
    missing = [p for p in required if p not in system["packages"]]
    spec = read_json(REGISTRY)
    try:
        verify_model(model_path(spec), spec)
        checkpoint = "verified"
    except (OSError, ValueError) as exc:
        checkpoint = str(exc)
    return {
        "runtime_ready": not missing and checkpoint == "verified",
        "missing_packages": missing, "checkpoint": checkpoint,
        "git": git_state(ROOT), "environment": system,
        "note": "Presence/checksum check only; run a real input to verify inference compatibility.",
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description="CPU fire-detection experiment runner")
    commands = parser.add_subparsers(dest="command", required=True)
    check = commands.add_parser("doctor", help="Inspect packages/model/Git without modifying them")
    check.add_argument("--output", type=Path)
    commands.add_parser("fetch-model", help="Download and verify the pinned checkpoint")
    run = commands.add_parser("run", help="Analyze local images or a saved video")
    run.add_argument("--source", required=True, help="Local image, image folder, or saved video")
    run.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    run.add_argument("--max-frames", type=int)
    run.add_argument("--save-preview", action="store_true")
    args = parser.parse_args(argv)
    try:
        if args.command == "doctor":
            result = doctor()
            # Keep terminal output compact; full package list is retained in the optional report.
            visible = {k: v for k, v in result.items() if k != "environment"}
            visible["python"] = result["environment"]["python"]
            visible["packages"] = {
                p: result["environment"]["packages"].get(p)
                for p in ("torch", "torchvision", "ultralytics", "opencv-python", "numpy")
            }
            print(json.dumps(visible, ensure_ascii=False, indent=2))
            if args.output:
                args.output.parent.mkdir(parents=True, exist_ok=True)
                write_json(args.output, result)
            return 0 if result["runtime_ready"] else 1
        if args.command == "fetch-model":
            print("Verified model:", fetch_model(read_json(REGISTRY)))
            return 0
        from .experiment import run_experiment
        run_experiment(
            args.source, load_config(args.config), read_json(REGISTRY),
            max_frames=args.max_frames, save_preview=args.save_preview,
        )
        return 0
    except KeyboardInterrupt:
        print("Interrupted. Partial run, if created, is marked interrupted.", file=sys.stderr)
        return 130
    except Exception as exc:
        print(f"ERROR ({type(exc).__name__}): {exc}", file=sys.stderr)
        return 1
