"""Independent staged commands; a batch starts only explicitly named experiments."""
import argparse
import json
import sys
from pathlib import Path

from .common import ROOT, environment, read_json, resolve, write_json


def parser():
    root = argparse.ArgumentParser(description="Reproducible D-Fire YOLOv8n GPU experiments")
    sub = root.add_subparsers(dest="command", required=True)
    doctor = sub.add_parser("doctor", help="Probe CUDA convolution/backward and record environment")
    doctor.add_argument("--output", default="outputs/environment.json")
    doctor.add_argument("--require-gpu", action="store_true")
    validate = sub.add_parser("validate-data", help="Audit raw data and create validated copies")
    validate.add_argument("--config", default="configs/dataset.json")
    verify = sub.add_parser("verify-data", help="Rehash all prepared images, labels and split artifacts")
    verify.add_argument("--path", help="Defaults to prepared_root in configs/dataset.json")
    sub.add_parser("fetch-models", help="Download pinned COCO and published checkpoints")
    export = sub.add_parser("export-colab", help="Bundle exact prepared inputs for Colab")
    export.add_argument("--config", default="configs/dataset.json")
    export.add_argument("--output")
    importing = sub.add_parser("import-colab", help="Verify and relocate a Colab input bundle")
    importing.add_argument("--archive", required=True)
    importing.add_argument("--sha256", required=True)
    importing.add_argument("--config", default="configs/dataset.json")
    for name in ("plan", "train", "batch"):
        command = sub.add_parser(name)
        command.add_argument("--suite", default="configs/experiments.json")
        command.add_argument("--ids" if name == "batch" else "--id", required=True, nargs="+" if name == "batch" else None)
    resume = sub.add_parser("resume")
    resume.add_argument("--run", required=True)
    evaluate = sub.add_parser("evaluate", help="Validation only; final test has a separate command")
    group = evaluate.add_mutually_exclusive_group(required=True)
    group.add_argument("--run")
    group.add_argument("--published", action="store_true")
    evaluate.add_argument("--suite", default="configs/experiments.json")
    evaluate.add_argument("--operating-conf", type=float)
    freeze = sub.add_parser("freeze", help="Freeze a completed validation result, model and thresholds")
    freeze.add_argument("--evaluation", required=True)
    freeze.add_argument("--rationale", required=True)
    test = sub.add_parser("test", help="Evaluate test using a checksummed final selection")
    test.add_argument("--selection", required=True)
    compare = sub.add_parser("compare")
    compare.add_argument("--runs", nargs="+", required=True)
    predict = sub.add_parser("predict")
    predict.add_argument("--source", required=True)
    predict.add_argument("--run")
    predict.add_argument("--confidence", type=float, default=.25)
    predict.add_argument("--iou", type=float, default=.7)
    predict.add_argument("--imgsz", type=int, default=640)
    predict.add_argument("--device", default="0")
    return root


def main(argv=None):
    args = parser().parse_args(argv)
    try:
        if args.command == "doctor":
            report = environment(cuda_probe=True)
            write_json(resolve(args.output), report)
            print(json.dumps({k: v for k, v in report.items() if k not in ("packages", "git")}, ensure_ascii=False, indent=2))
            if args.require_gpu and report.get("cuda_compute_probe") != "passed":
                raise RuntimeError("CUDA compute probe did not pass")
        elif args.command == "validate-data":
            from .dataset import prepare_dataset
            print(prepare_dataset(resolve(args.config)))
        elif args.command == "verify-data":
            from .dataset import verify_dataset
            dataset_path = args.path or read_json(ROOT / "configs/dataset.json")["prepared_root"]
            report = verify_dataset(resolve(dataset_path))
            print(json.dumps({"fingerprint": report["fingerprint"], "summary": report["summary"]}, indent=2))
        elif args.command == "fetch-models":
            from .model import fetch_model
            for name in ("coco", "published"):
                print(fetch_model(name))
        elif args.command == "export-colab":
            from .colab import export_data
            print(json.dumps(export_data(resolve(args.config), resolve(args.output) if args.output else None), indent=2))
        elif args.command == "import-colab":
            from .colab import import_data
            print(import_data(resolve(args.archive), args.sha256, resolve(args.config)))
        elif args.command == "plan":
            from .config import experiment_config
            print(json.dumps(experiment_config(resolve(args.suite), args.id), ensure_ascii=False, indent=2))
        elif args.command in ("train", "batch", "resume"):
            from .training import train
            if args.command == "resume":
                print(train(resume_run=args.run))
            else:
                ids = args.ids if args.command == "batch" else [args.id]
                # Validate the entire requested batch before starting the first experiment.
                from .config import experiment_config
                if len(set(ids)) != len(ids):
                    raise ValueError("Duplicate batch IDs")
                for experiment_id in ids:
                    experiment_config(resolve(args.suite), experiment_id)
                for experiment_id in ids:
                    print(train(resolve(args.suite), experiment_id))
        elif args.command == "evaluate":
            from .evaluation import evaluate_validation
            print(evaluate_validation(args.run, args.published, resolve(args.suite), args.operating_conf))
        elif args.command == "freeze":
            from .evaluation import freeze_selection
            print(freeze_selection(args.evaluation, args.rationale))
        elif args.command == "test":
            from .evaluation import evaluate_test
            print(evaluate_test(resolve(args.selection)))
        elif args.command == "compare":
            from .analysis import compare_runs
            print(compare_runs([resolve(path) for path in args.runs]))
        elif args.command == "predict":
            from .inference import predict
            print(predict(args.source, args.run, args.confidence, args.iou, args.imgsz, args.device))
    except (ValueError, RuntimeError, FileNotFoundError, FileExistsError, KeyError, OSError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    return 0
