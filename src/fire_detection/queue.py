"""Finite Linux queue with detached workers, immutable inputs and explicit run receipts.

The queue stops for failures and after the authorized core experiments. E6 and Test
require analysis; this module never selects their settings or launches them.
"""
from __future__ import annotations

import argparse
import fcntl
import os
import signal
import subprocess
import sys
import time
import traceback
from contextlib import contextmanager
from pathlib import Path

import psutil

from .common import (ROOT, configure_plotting, file_sha256, git_state, inside,
                     object_sha256, read_json, resolve, runs_root, utc_now, write_json)
from .config import experiment_config
from .dataset import verify_dataset
from .model import spec_for, verify_model

CORE_IDS = ("E1", "E2-A", "E2-C", "E3-HSV", "E3-Mosaic", "E4-800")
SCRIPT = ROOT / "scripts/training_queue.py"
PYTHON = ROOT / ".venv/bin/python"
DEFAULT_STATE = ROOT / "work/server-queue.json"
EXPECTED_DATA = "c3d63dfa73314a36b5522f4651f5fd7d784bd8a054e78916f821160729862e2b"
EXPECTED_COCO = "f59b3d833e2ff32e194b5bb8e08d211dc7c5bdf144b90d2c8412c47ccfc83b36"
E0_TRAIN = ROOT / "experiments/runs/E0_20261103T164735Z_40e89a6c"
E0_VAL = ROOT / "experiments/runs/val-E0_20261103T221126Z_f11e3517"


def identity(pid=None):
    process = psutil.Process(pid or os.getpid())
    return {"pid": process.pid, "process_created_at": process.create_time()}


def alive(info):
    if not info or not info.get("pid"):
        return False
    try:
        process = psutil.Process(info["pid"])
        return (abs(process.create_time() - info["process_created_at"]) < .01
                and process.status() != psutil.STATUS_ZOMBIE)
    except psutil.NoSuchProcess:
        return False


@contextmanager
def exclusive(path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+") as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError(f"Another queue/worker holds lock: {path}") from exc
        try:
            yield handle
        finally:
            # Closing releases our descriptor; do not explicitly unlock a descriptor
            # inherited by a still-running child after an interrupted manager.
            pass


def detached(command, log_path, pass_fds=()):
    Path(log_path).parent.mkdir(parents=True, exist_ok=True)
    env = {**os.environ, "MPLBACKEND": "Agg", "PYTHONUNBUFFERED": "1"}
    with Path(log_path).open("ab", buffering=0) as log:
        return subprocess.Popen(command, cwd=ROOT, env=env, stdin=subprocess.DEVNULL,
                                stdout=log, stderr=subprocess.STDOUT, start_new_session=True,
                                close_fds=True, pass_fds=pass_fds)


def classify(error):
    text = str(error).lower()
    if any(word in text for word in ("checksum", "fingerprint", "integrity", "changed", "mismatch")):
        return "integrity"
    if "out of memory" in text or "outofmemory" in text:
        return "oom"
    if any(word in text for word in ("keyboardinterrupt", "signal", "interrupted")):
        return "interrupted"
    return "failure"


def save_state(path, state):
    state["updated_at_utc"] = utc_now()
    write_json(path, state)


def validate_training(run, config, fingerprint, require_completed=True):
    run = inside(run, runs_root())
    record = read_json(run / "training.json")
    if (record["config"] != config or record["dataset_fingerprint"] != fingerprint
            or record["initial_model"]["sha256"] != EXPECTED_COCO
            or record["attempts"][0]["checkpoint_sha256"] != EXPECTED_COCO):
        raise ValueError("Training configuration/model/dataset integrity mismatch")
    if require_completed and (record["status"] != "completed"
                              or record["epochs_completed"] != config["training"]["epochs"]):
        raise ValueError("Training did not complete its requested epochs")
    entries = read_json(run / "checkpoints.json")
    for name in ("last", "best"):
        checkpoint = run / "train/weights" / f"{name}.pt"
        if Path(entries[name]["path"]).resolve() != checkpoint.resolve():
            raise ValueError("Checkpoint path integrity mismatch")
        if file_sha256(checkpoint) != entries[name]["sha256"]:
            raise ValueError(f"{name} checkpoint checksum mismatch")
    if require_completed and (Path(record["best_path"]).resolve() != (run / "train/weights/best.pt").resolve()
                              or record["best_sha256"] != entries["best"]["sha256"]):
        raise ValueError("Best checkpoint integrity mismatch")
    return record


def validate_evaluation(run, training_run, training):
    run = inside(run, runs_root())
    record = read_json(run / "evaluation.json")
    if ((run / "invalidated.json").exists() or record["status"] != "completed"
            or record["split"] != "val" or record["model_role"] != "trained"
            or Path(record["training_run"]).resolve() != Path(training_run).resolve()
            or record["config"] != training["config"]
            or record["dataset_fingerprint"] != training["dataset_fingerprint"]
            or record["checkpoint_sha256"] != training["best_sha256"]
            or Path(record["checkpoint"]).resolve() != Path(training["best_path"]).resolve()
            or record["operating"]["settings"] != training["config"]["evaluation"]):
        raise ValueError("Validation configuration/checkpoint/dataset integrity mismatch")
    if file_sha256(record["checkpoint"]) != record["checkpoint_sha256"]:
        raise ValueError("Validation checkpoint checksum mismatch")
    return record


def assert_no_active_training():
    for path in runs_root().glob("*/training.json"):
        record = read_json(path)
        if record["status"] == "running" and any(alive(attempt) for attempt in record["attempts"]):
            raise RuntimeError(f"Training process is still active: {path.parent}")
    # Account for manually launched CLI/queue workers even before their run is created.
    for process in psutil.process_iter(["pid", "cmdline"]):
        if process.pid == os.getpid():
            continue
        args = process.info["cmdline"] or []
        if any(Path(arg).name == "firelab.py" for arg in args) and any(
                action in args for action in ("train", "batch", "resume", "evaluate", "test")):
            raise RuntimeError(f"Another GPU command is active: PID {process.pid}")
        if any(Path(arg).name == "training_queue.py" for arg in args) and "worker" in args:
            raise RuntimeError(f"A queue GPU worker is still active: PID {process.pid}")


def make_plan():
    suite = ROOT / "configs/experiments.json"
    configs = {key: experiment_config(suite, key) for key in CORE_IDS}
    baseline = read_json(E0_TRAIN / "training.json")
    if baseline["config"] != experiment_config(suite, "E0"):
        raise ValueError("Server E0 baseline configuration mismatch")
    validate_training(E0_TRAIN, baseline["config"], EXPECTED_DATA)
    validate_evaluation(E0_VAL, E0_TRAIN, baseline)
    if baseline["best_sha256"] != "793526814adbd0c2a9d31054c1c37563c3c77fdef6cded429f062b964a4ee85d":
        raise ValueError("Server E0 checkpoint checksum mismatch")
    paths = [suite, ROOT / "configs/models.json", ROOT / "configs/dataset.json", SCRIPT]
    paths += sorted((ROOT / "src/fire_detection").glob("*.py"))
    plan = {"kind": "authorized_core_queue", "created_at_utc": utc_now(),
            "ids": list(CORE_IDS), "configs": configs,
            "dataset_fingerprint": EXPECTED_DATA, "coco_sha256": EXPECTED_COCO,
            "baseline_training_run": str(E0_TRAIN), "baseline_validation_run": str(E0_VAL),
            "baseline_training_sha256": file_sha256(E0_TRAIN / "training.json"),
            "baseline_validation_sha256": file_sha256(E0_VAL / "evaluation.json"),
            "pinned_files": {str(path.relative_to(ROOT)): file_sha256(path) for path in paths},
            "git": git_state(), "after_queue": "awaiting_analysis",
            "retry_policy": "No automatic retry; explicit --resume only after interruption and checksum verification"}
    plan["plan_sha256"] = object_sha256(plan)
    verify_plan(plan)
    return plan


def verify_plan(plan):
    payload = {k: v for k, v in plan.items() if k != "plan_sha256"}
    if object_sha256(payload) != plan["plan_sha256"] or plan["ids"] != list(CORE_IDS):
        raise ValueError("Queue plan checksum/authorized IDs mismatch")
    for path, digest in plan["pinned_files"].items():
        if file_sha256(resolve(path)) != digest:
            raise ValueError(f"Pinned source/config checksum mismatch: {path}")
    spec = spec_for("coco")
    if spec["sha256"] != plan["coco_sha256"]:
        raise ValueError("COCO model specification checksum mismatch")
    verify_model(resolve(spec["local_path"]), spec)
    manifest = verify_dataset(resolve(plan["configs"]["E1"]["dataset_config"]["prepared_root"]))
    if manifest["fingerprint"] != plan["dataset_fingerprint"]:
        raise ValueError("Queue dataset fingerprint mismatch")
    for prefix, filename in (("training", "training.json"), ("validation", "evaluation.json")):
        path = Path(plan[f"baseline_{prefix}_run"]) / filename
        if file_sha256(path) != plan[f"baseline_{prefix}_sha256"]:
            raise ValueError("Baseline record checksum mismatch")
    baseline = read_json(Path(plan["baseline_training_run"]) / "training.json")
    validate_training(plan["baseline_training_run"], baseline["config"], plan["dataset_fingerprint"])
    validate_evaluation(plan["baseline_validation_run"], plan["baseline_training_run"], baseline)


def existing_runs(experiment_id):
    return [path.parent for path in runs_root().glob("*/training.json")
            if read_json(path).get("experiment_id") == experiment_id]


def progress(task):
    run = task.get("training_run")
    if run and (Path(run) / "checkpoints.json").exists():
        entries = read_json(Path(run) / "checkpoints.json")
        task["last_completed_epoch"] = entries.get("last", {}).get("epoch", 0)


def receipt_run(task, stage):
    receipt = read_json(task[f"{stage}_receipt"])
    run = receipt.get("run")
    if run:
        task["training_run" if stage == "train" else "validation_run"] = str(inside(run, runs_root()))
    return receipt


def execute_stage(state_path, state, task, stage, lock_fd, resume_run=None):
    folder = state_path.parent / (state_path.stem + "-jobs") / task["id"]
    folder.mkdir(parents=True, exist_ok=True)
    number = len(task.setdefault("workers", [])) + 1
    receipt = folder / f"{stage}-{number}.json"
    log = folder / f"{stage}-{number}.log"
    task[f"{stage}_receipt"] = str(receipt)
    task[f"{stage}_log"] = str(log)
    command = [str(PYTHON), "-u", str(SCRIPT), "worker", "--stage", stage,
               "--id", task["id"], "--plan", state["plan_path"], "--receipt", str(receipt)]
    if stage == "val" or resume_run:
        command += ["--training-run", str(resume_run or task["training_run"])]
    if resume_run:
        command.append("--resume-training")
    process = detached(command, log, pass_fds=(lock_fd,))
    info = {**identity(process.pid), "stage": stage, "log": str(log),
            "receipt": str(receipt), "started_at_utc": utc_now()}
    task["workers"].append(info)
    task["worker"] = info
    task["status"] = state["status"] = "training" if stage == "train" else "validating"
    state["current_experiment"] = task["id"]
    state["next_action"] = "verify training then common Validation" if stage == "train" else "verify Validation then comparison report"
    save_state(state_path, state)
    while process.poll() is None:
        if receipt.exists():
            receipt_run(task, stage)
        progress(task)
        save_state(state_path, state)
        time.sleep(5)
    if receipt.exists():
        result = receipt_run(task, stage)
    else:
        result = {"status": "failed", "error": f"Worker exited without receipt (signal/exit {process.returncode})"}
    info.update(exit_code=process.returncode, finished_at_utc=utc_now())
    progress(task)
    save_state(state_path, state)
    if process.returncode != 0 or result["status"] != "completed":
        message = result.get("error", f"Worker exited with signal/code {process.returncode}")
        raise RuntimeError(message)
    return result


def report_completed(state, task):
    from .analysis import compare_runs
    previous = [state["plan"]["baseline_validation_run"]]
    previous += [entry["validation_run"] for entry in state["tasks"] if entry["status"] == "completed"]
    output = compare_runs([*map(Path, previous), Path(task["validation_run"])])
    task["comparison_output"] = str(output)
    plan = state["plan"]
    run = Path(task["training_run"])
    training = validate_training(run, plan["configs"][task["id"]], plan["dataset_fingerprint"])
    evaluation = validate_evaluation(task["validation_run"], run, training)
    reference_id = "E0" if task["id"] == "E1" else "E1"
    reference_path = Path(plan["baseline_validation_run"] if reference_id == "E0" else
                          next(item["validation_run"] for item in state["tasks"] if item["id"] == "E1"))
    reference = read_json(reference_path / "evaluation.json")
    delta = 100 * (evaluation["metrics"]["map50_95"] - reference["metrics"]["map50_95"])
    op = evaluation["operating"]
    summary = {"experiment_id": task["id"], "training_run": str(run), "validation_run": task["validation_run"],
               "training_record_sha256": file_sha256(run / "training.json"),
               "validation_record_sha256": file_sha256(Path(task["validation_run"]) / "evaluation.json"),
               "reference_id": reference_id, "map50_95_delta_percentage_points": delta,
               "comparison_output": str(output), "training": training, "validation": evaluation}
    report = ROOT / "reports" / f"010_server_{task['id'].lower()}"
    write_json(report.with_suffix(".json"), summary)
    lines = [f"# 서버 {task['id']} 학습 및 공통 Validation", "", f"실제 {training['epochs_completed']} epochs 학습과 공통 Validation을 완료했다.", "",
             f"- 학습: `{run}`", f"- 평가: `{task['validation_run']}`",
             f"- 모델 SHA256: `{training['best_sha256']}`", f"- 데이터 fingerprint: `{training['dataset_fingerprint']}`",
             f"- 기준 {reference_id} 대비 mAP50-95 변화: {delta:+.3f}%p (단일 seed 관측).",
             f"- 공통 mAP50: {evaluation['metrics']['map50']:.5f}; mAP50-95: {evaluation['metrics']['map50_95']:.5f}.",
             f"- confidence 0.25 Fire Recall: {op['classes']['fire']['recall']:.5f}; "
             f"TP/FP/FN: {op['classes']['fire']['tp']}/{op['classes']['fire']['fp']}/{op['classes']['fire']['fn']}.",
             f"- Negative 오탐 이미지: {op['negative_images_with_fp']}/{op['negative_images']}; 검출 {op['negative_fp_detections']}개.",
             f"- 학습 시간: {training['training_wall_seconds']:.1f}초; 배치 1 FP32 추론 {op['inference_ms_per_image']:.3f}ms/장.",
             f"- 비교 CSV·JSON·loss/AP 곡선·그래프: `{output}`.", "",
             "AP의 P/R은 최대 평균 F1 지점이며 confidence 0.25 운영 P/R과 구분한다. "
             "학습 내부 Validation의 padding/batch 차이 때문에 후보 비교에는 공통 Validation만 사용한다.", "",
             "세부 effective 인자·optimizer/LR·버전·Git 상태·GPU 메모리·클래스별 AP 및 운영 지표는 "
             "동명 JSON과 run 원본에 보존한다. 하드웨어/버전별 속도 차이와 단일 seed 한계가 있다. "
             "출처별 video/session ID가 없어 근접 프레임 누수를 보장하지 못하며 공개 모델 split과 다르다. "
             "Test는 평가하지 않았다.", ""]
    report.with_suffix(".md").write_text("\n".join(lines), encoding="utf-8")
    task["report"] = str(report.with_suffix(".md"))
    with (ROOT / "docs/PROGRESS.md").open("a", encoding="utf-8") as handle:
        handle.write(f"\n## {utc_now()[:10]}: 서버 {task['id']} 학습·Validation 완료\n\n"
                     f"- 실제 완료 결과와 {reference_id} 비교: [상세 보고서](../reports/{report.name}.md).\n"
                     f"- 공통 mAP50-95 {evaluation['metrics']['map50_95']:.5f}, 기준 대비 {delta:+.3f}%p.\n"
                     "- 모델·입력·평가 연결 검증 후 다음 승인 조건으로 진행한다. Test는 미평가다.\n")


def recover_task(task, plan):
    """Adopt only the exact recorded runs; never guess a run from modification time."""
    for stage in ("train", "val"):
        if task.get(f"{stage}_receipt") and Path(task[f"{stage}_receipt"]).exists():
            receipt_run(task, stage)
    if not task.get("training_run"):
        raise RuntimeError("Interrupted task has no recorded training run; inspect before restarting")
    record = validate_training(task["training_run"], plan["configs"][task["id"]],
                               plan["dataset_fingerprint"], require_completed=False)
    if record["status"] == "completed":
        validate_training(task["training_run"], plan["configs"][task["id"]], plan["dataset_fingerprint"])
        if task.get("validation_run"):
            val = read_json(Path(task["validation_run"]) / "evaluation.json")
            if val["status"] == "completed":
                validate_evaluation(task["validation_run"], task["training_run"], record)
                return "report"
        return "val"
    if record["status"] == "failed" and classify(record["attempts"][-1].get("error", "")) != "interrupted":
        raise RuntimeError("Non-transient training failure; refusing automatic resume")
    # Checkpoint's resumable optimizer/epoch is also checked by the existing train API.
    return "resume"


def run_queue(state_path, resume=False):
    signal.signal(signal.SIGHUP, signal.SIG_IGN)
    with exclusive(ROOT / "work/server-queue.lock") as lock:
        state = read_json(state_path)
        if alive(state.get("manager")):
            raise RuntimeError("Queue manager is still active")
        for task in state["tasks"]:
            if any(alive(worker) for worker in task.get("workers", [])):
                raise RuntimeError("Recorded worker is still active; do not start a duplicate manager")
        if state["status"] != "prepared" and not resume:
            raise RuntimeError("Existing queue requires explicit --resume after inspection")
        if state["status"] == "awaiting_analysis":
            raise RuntimeError("Core queue already completed; E6 requires analysis")
        if resume and state.get("failure_kind") in ("integrity", "oom", "failure"):
            raise RuntimeError("Non-transient queue failure requires investigation; no automatic retry")
        state.setdefault("started_at_utc", utc_now())
        if state.get("error"):
            state.setdefault("failures", []).append({key: state.get(key) for key in
                                                      ("error", "failure_kind", "traceback", "finished_at_utc")})
        for key in ("error", "failure_kind", "traceback", "finished_at_utc"):
            state.pop(key, None)
        state.update(manager={**identity(), "sid": os.getsid(0)}, status="preflight")
        save_state(state_path, state)
        try:
            assert_no_active_training()
            plan = read_json(state["plan_path"])
            if plan["plan_sha256"] != state["plan_sha256"]:
                raise ValueError("State/plan checksum mismatch")
            state["plan"] = plan
            verify_plan(plan)
            for index, task in enumerate(state["tasks"]):
                if task["status"] == "completed":
                    training = validate_training(task["training_run"], plan["configs"][task["id"]], plan["dataset_fingerprint"])
                    validate_evaluation(task["validation_run"], task["training_run"], training)
                    continue
                state["current_experiment"] = task["id"]
                state["next_experiment"] = CORE_IDS[index + 1] if index + 1 < len(CORE_IDS) else "E6 analysis"
                verify_plan(plan)
                action = "new" if task["status"] == "pending" else recover_task(task, plan)
                task.setdefault("started_at_utc", utc_now())
                if action == "new":
                    if existing_runs(task["id"]):
                        raise RuntimeError(f"Existing {task['id']} training run found; explicit adoption required")
                    execute_stage(state_path, state, task, "train", lock.fileno())
                elif action == "resume":
                    execute_stage(state_path, state, task, "train", lock.fileno(), resume_run=task["training_run"])
                training = validate_training(task["training_run"], plan["configs"][task["id"]], plan["dataset_fingerprint"])
                verify_plan(plan)
                if action != "report":
                    execute_stage(state_path, state, task, "val", lock.fileno())
                validate_evaluation(task["validation_run"], task["training_run"], training)
                verify_plan(plan)
                task["status"] = state["status"] = "reporting"
                save_state(state_path, state)
                report_completed(state, task)
                task.update(status="completed", finished_at_utc=utc_now())
                save_state(state_path, state)
            state.update(status="awaiting_analysis", finished_at_utc=utc_now(),
                         current_experiment=None, next_action="Analyze core Validation, decide E5 need and E6 overrides; do not evaluate Test before freeze")
            save_state(state_path, state)
        except BaseException as exc:
            reason = f"{type(exc).__name__}: {exc}"
            state.update(status="failed", error=reason, failure_kind=classify(reason),
                         traceback=traceback.format_exc(), finished_at_utc=utc_now(),
                         next_action="Inspect failure and logs; preserve all runs. No automatic retry or setting changes.")
            for task in state["tasks"]:
                if task["id"] == state.get("current_experiment") and task["status"] != "completed":
                    task.update(status="failed", error=reason, finished_at_utc=utc_now())
            save_state(state_path, state)
            raise


def worker(args):
    signal.signal(signal.SIGHUP, signal.SIG_IGN)
    configure_plotting()
    receipt = {**identity(), "status": "starting", "stage": args.stage, "experiment_id": args.id,
               "started_at_utc": utc_now(), "run": None}
    write_json(args.receipt, receipt)

    def created(run):
        receipt.update(run=str(run), status="running")
        write_json(args.receipt, receipt)

    try:
        with exclusive(ROOT / "work/server-gpu.lock"):
            assert_no_active_training()
            plan = read_json(args.plan)
            verify_plan(plan)
            if args.stage == "train":
                from .training import train
                run = train(ROOT / "configs/experiments.json", args.id,
                            resume_run=args.training_run if args.resume_training else None, on_run_created=created)
            else:
                from .evaluation import evaluate_validation
                training = validate_training(args.training_run, plan["configs"][args.id], plan["dataset_fingerprint"])
                run = evaluate_validation(training_run=args.training_run, on_run_created=created)
                validate_evaluation(run, args.training_run, training)
            receipt.update(status="completed", run=str(run))
    except BaseException as exc:
        receipt.update(status="failed", error=f"{type(exc).__name__}: {exc}", traceback=traceback.format_exc())
        raise
    finally:
        receipt["finished_at_utc"] = utc_now()
        write_json(args.receipt, receipt)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    prepare = sub.add_parser("prepare")
    prepare.add_argument("--state", type=Path, default=DEFAULT_STATE)
    for name in ("start", "run", "status"):
        command = sub.add_parser(name)
        command.add_argument("--state", type=Path, default=DEFAULT_STATE)
        if name != "status":
            command.add_argument("--resume", action="store_true")
    child = sub.add_parser("worker")
    child.add_argument("--stage", choices=("train", "val"), required=True)
    child.add_argument("--id", choices=CORE_IDS, required=True)
    child.add_argument("--plan", type=Path, required=True)
    child.add_argument("--receipt", type=Path, required=True)
    child.add_argument("--training-run", type=Path)
    child.add_argument("--resume-training", action="store_true")
    probe = sub.add_parser("probe")
    probe.add_argument("--output", type=Path, required=True)
    probe.add_argument("--delay", type=float, default=3)
    args = parser.parse_args(argv)
    if args.command == "prepare":
        with exclusive(ROOT / "work/server-queue.lock"):
            if args.state.exists():
                raise FileExistsError("Queue state already exists; inspect it instead of overwriting")
            assert_no_active_training()
            plan = make_plan()
            for experiment_id in CORE_IDS:
                if existing_runs(experiment_id):
                    raise RuntimeError(f"Existing {experiment_id} run found; explicit adoption required")
            plan_path = args.state.with_name(args.state.stem + "-plan.json").resolve()
            write_json(plan_path, plan)
            save_state(args.state, {"kind": "detached_core_queue", "status": "prepared", "plan_path": str(plan_path),
                                   "plan_sha256": plan["plan_sha256"], "created_at_utc": utc_now(),
                                   "tasks": [{"id": key, "status": "pending", "training_run": None,
                                              "validation_run": None, "last_completed_epoch": 0} for key in CORE_IDS],
                                   "next_action": "Start verified detached queue", "manager": None})
        print(args.state)
    elif args.command == "start":
        state = read_json(args.state)
        if alive(state.get("manager")) or any(alive(worker) for task in state["tasks"] for worker in task.get("workers", [])):
            raise RuntimeError("Queue manager/worker is still active")
        if state["status"] != "prepared" and not args.resume:
            raise RuntimeError("Queue is not prepared; inspect state before explicit --resume")
        log = args.state.with_suffix(".log").resolve()
        command = [str(PYTHON), "-u", str(SCRIPT), "run", "--state", str(args.state.resolve())]
        if args.resume:
            command.append("--resume")
        process = detached(command, log)
        print(f"Detached queue PID {process.pid}; state {args.state}; log {log}")
    elif args.command == "run":
        run_queue(args.state, args.resume)
    elif args.command == "status":
        state = read_json(args.state)
        print(f"status={state['status']} manager_alive={alive(state.get('manager'))} manager={state.get('manager')}")
        print(f"updated={state['updated_at_utc']} next={state.get('next_action')}")
        for task in state["tasks"]:
            print(f"{task['id']}: {task['status']} epoch={task['last_completed_epoch']} worker_alive={alive(task.get('worker'))}")
            print(f"  train={task.get('training_run')} val={task.get('validation_run')}")
            if task.get("worker"):
                print(f"  log={task['worker']['log']}")
        if state.get("error"):
            print(f"error={state['error']}")
    elif args.command == "worker":
        worker(args)
    elif args.command == "probe":
        signal.signal(signal.SIGHUP, signal.SIG_IGN)
        write_json(args.output, {"status": "running", **identity(), "sid": os.getsid(0), "started_at_utc": utc_now()})
        time.sleep(args.delay)
        write_json(args.output, {"status": "completed", **identity(), "sid": os.getsid(0), "ppid": os.getppid(),
                                 "stdin": os.readlink('/proc/self/fd/0'), "stdout": os.readlink('/proc/self/fd/1'),
                                 "sighup_ignored": signal.getsignal(signal.SIGHUP) == signal.SIG_IGN,
                                 "finished_at_utc": utc_now(), "note": "Non-training detachment probe; no model result"})
    return 0
