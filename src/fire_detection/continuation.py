"""Explicit E6 continuation of an archived, successfully completed core queue."""
from __future__ import annotations

import copy
from pathlib import Path

from .common import ROOT, file_sha256, inside, object_sha256, read_json, utc_now, write_json
from .config import experiment_config


def read_completed_archive(folder, expected_sha256):
    from .queue import CORE_IDS

    folder = inside(folder, ROOT / "work")
    if file_sha256(folder / "archive.json") != expected_sha256:
        raise ValueError("Core archive manifest checksum mismatch")
    archive = read_json(folder / "archive.json")
    if archive.get("kind") != "completed_core_queue_archive":
        raise ValueError("Expected a completed core queue archive")
    for relative, digest in archive["files"].items():
        if file_sha256(inside(folder / relative, folder)) != digest:
            raise ValueError(f"Archived core file checksum mismatch: {relative}")
    state, plan = read_json(folder / "state.json"), read_json(folder / "plan.json")
    if (object_sha256({k: v for k, v in plan.items() if k != "plan_sha256"}) != plan["plan_sha256"]
            or plan["plan_sha256"] != archive["previous_plan_sha256"]
            or state["plan_sha256"] != plan["plan_sha256"]
            or state["status"] != "awaiting_analysis"
            or plan["ids"] != list(CORE_IDS)
            or [task["id"] for task in state["tasks"]] != list(CORE_IDS)
            or any(task["status"] != "completed" for task in state["tasks"])
            or file_sha256(folder / "state.json") != archive["previous_state_sha256"]
            or file_sha256(folder / "plan.json") != archive["previous_plan_file_sha256"]):
        raise ValueError("Archived completed queue integrity mismatch")
    for relative, digest in plan["pinned_files"].items():
        if file_sha256(inside(folder / "sources" / relative, folder)) != digest:
            raise ValueError(f"Archived pinned source checksum mismatch: {relative}")
    return archive, state, plan


def verify_e6_extension(plan):
    from .queue import CORE_IDS, validate_evaluation, validate_training

    extension = plan["extension"]
    archive, old_state, old_plan = read_completed_archive(
        extension["archive"], extension["archive_manifest_sha256"])
    if (plan["kind"] != "authorized_core_queue_with_selected_e6"
            or plan["ids"] != [*CORE_IDS, "E6"]
            or plan["after_queue"] != "awaiting_final_selection"
            or extension["previous_plan_sha256"] != old_plan["plan_sha256"]
            or extension["previous_state_sha256"] != archive["previous_state_sha256"]
            or not extension.get("authorized_at_utc")):
        raise ValueError("E6 continuation authorization/archive integrity mismatch")
    for key in CORE_IDS:
        if plan["configs"][key] != old_plan["configs"][key]:
            raise ValueError(f"Completed core configuration changed: {key}")
    selected = plan["configs"]["E6"]
    if selected != experiment_config(ROOT / "configs/experiments.json", "E6"):
        raise ValueError("Selected E6 configuration integrity mismatch")
    if (selected["training"]["epochs"] != 100 or selected["training"]["patience"] != 0
            or not selected["experiment"].get("selection_rationale")):
        raise ValueError("E6 requires an explicit 100-epoch selection rationale")
    if any(selected["training"] == old_plan["configs"][key]["training"] for key in CORE_IDS):
        raise ValueError("E6 duplicates a completed condition; reuse that run")
    anchors = {entry["id"]: entry for entry in archive["completed"]}
    if set(anchors) != set(CORE_IDS):
        raise ValueError("Incomplete core validation evidence")
    for task in old_state["tasks"]:
        anchor = anchors[task["id"]]
        if (anchor["training_run"] != task["training_run"]
                or anchor["validation_run"] != task["validation_run"]
                or file_sha256(Path(task["training_run"]) / "training.json") != anchor["training_record_sha256"]
                or file_sha256(Path(task["validation_run"]) / "evaluation.json") != anchor["validation_record_sha256"]):
            raise ValueError("Completed core evidence checksum/path mismatch")
        training = validate_training(task["training_run"], old_plan["configs"][task["id"]],
                                     old_plan["dataset_fingerprint"])
        evaluation = validate_evaluation(task["validation_run"], task["training_run"], training)
        if evaluation["checkpoint_sha256"] != anchor["best_sha256"]:
            raise ValueError("Selected validation model checksum mismatch")
    evidence = selected["experiment"]["selection_evidence"]
    if not isinstance(evidence, list) or not evidence:
        raise ValueError("E6 requires pinned validation evidence")
    evidence_ids = [entry["experiment_id"] for entry in evidence]
    if len(set(evidence_ids)) != len(evidence_ids) or "E1" not in evidence_ids or len(evidence_ids) < 2:
        raise ValueError("E6 needs distinct candidate and E1 validation evidence")
    for entry in evidence:
        if entry["experiment_id"] not in anchors:
            raise ValueError("Only completed core Validation can select E6")
        anchor = anchors[entry["experiment_id"]]
        if ((ROOT / entry["validation_run"]).resolve() != Path(anchor["validation_run"]).resolve()
                or entry["validation_record_sha256"] != anchor["validation_record_sha256"]
                or entry["training_record_sha256"] != anchor["training_record_sha256"]
                or entry["checkpoint_sha256"] != anchor["best_sha256"]):
            raise ValueError("E6 selection evidence checksum mismatch")


def prepare_e6_extension(state_path, archive_path, archive_sha256):
    from .queue import (CORE_IDS, alive, assert_no_active_training, exclusive,
                        existing_runs, make_plan, save_state, verify_plan)

    state_path = inside(state_path, ROOT / "work")
    with exclusive(ROOT / "work/server-queue.lock"), exclusive(ROOT / "work/server-gpu.lock"):
        state = read_json(state_path)
        if alive(state.get("manager")) or any(
                alive(worker) for task in state["tasks"] for worker in task.get("workers", [])):
            raise RuntimeError("Queue manager/worker is still active; refusing E6 migration")
        assert_no_active_training()
        archive, old_state, old_plan = read_completed_archive(archive_path, archive_sha256)
        if file_sha256(state_path) != archive["previous_state_sha256"] or state != old_state:
            raise ValueError("Current queue changed since the completed archive")
        if existing_runs("E6"):
            raise RuntimeError("Existing E6 run found; inspect/adopt it instead of retraining")
        plan_path = state_path.with_name(state_path.stem + "-e6-plan.json")
        if plan_path.exists():
            raise FileExistsError("E6 plan already exists; inspect it instead of overwriting")
        plan = make_plan()
        if any(plan["configs"][key] != old_plan["configs"][key] for key in CORE_IDS):
            raise ValueError("Completed core configuration changed during E6 migration")
        plan.update(kind="authorized_core_queue_with_selected_e6", ids=[*CORE_IDS, "E6"],
                    after_queue="awaiting_final_selection")
        plan["configs"]["E6"] = experiment_config(ROOT / "configs/experiments.json", "E6")
        plan["extension"] = {"authorized_at_utc": utc_now(), "archive": str(Path(archive_path).resolve()),
                             "archive_manifest_sha256": archive_sha256,
                             "previous_plan_sha256": old_plan["plan_sha256"],
                             "previous_state_sha256": archive["previous_state_sha256"]}
        plan.pop("plan_sha256", None)
        plan["plan_sha256"] = object_sha256(plan)
        verify_plan(plan)
        updated = copy.deepcopy(state)
        updated["tasks"].append({"id": "E6", "status": "pending", "training_run": None,
                                 "validation_run": None, "last_completed_epoch": 0})
        updated.update(kind="detached_core_queue_with_selected_e6", status="prepared", manager=None,
                       plan_path=str(plan_path), plan_sha256=plan["plan_sha256"], plan=plan,
                       current_experiment=None, next_experiment="E6",
                       next_action="Train selected E6 from COCO, common Validation and report; no automatic Test",
                       extension_prepared_at_utc=utc_now(), extension=plan["extension"],
                       core_finished_at_utc=state.get("finished_at_utc"))
        updated.pop("finished_at_utc", None)
        write_json(plan_path, plan)
        save_state(state_path, updated)
        return updated
