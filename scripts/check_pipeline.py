"""Real-data GPU integration check, including interruption/resume; not an E0 result.

Run with .\.venv\Scripts\python.exe scripts/check_pipeline.py.
Only training/validation images are sampled. The held-out test images are never evaluated.
"""
from __future__ import annotations

import random
import shutil
import sys
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from fire_detection import training
from fire_detection.common import ROOT, file_sha256, new_run, read_json, write_json, yolo_class
from fire_detection.dataset import fingerprint, verify_dataset
from fire_detection.evaluation import evaluate_validation


def main():
    import yaml
    base = ROOT / read_json(ROOT / "configs/dataset.json")["prepared_root"]
    source = verify_dataset(base)
    output = new_run("pipeline-check", ROOT / "work")
    prepared = ROOT / "data/processed" / output.name
    prepared.mkdir()
    config = {**source["config"], "prepared_root": prepared.relative_to(ROOT).as_posix(),
              "integration_check": True, "source_fingerprint": source["fingerprint"]}
    config["split_files"] = {}
    manifest = {"config": config, "split_sources": {}, "records": [], "summary": {}}
    rng = random.Random(42)
    for split, per_group in (("train", 16), ("val", 4)):
        groups = {key: [] for key in ((), (0,), (1,), (0, 1))}
        for row in source["records"]:
            if row["split"] == split:
                groups[tuple(sorted({int(b[0]) for b in row["boxes"]}))].append(row)
        selected = []
        for key in groups:
            candidates = sorted(groups[key], key=lambda r: r["prepared_image"])
            selected.extend(rng.sample(candidates, per_group))
        paths = []
        for row in selected:
            for key in ("prepared_image", "prepared_label"):
                target = prepared / row[key]
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(base / row[key], target)
            paths.append((prepared / row["prepared_image"]).as_posix())
            manifest["records"].append(row)
        listing = prepared / f"{split}.txt"
        listing.write_text("\n".join(paths) + "\n", encoding="utf-8")
        config["split_files"][split] = listing.relative_to(ROOT).as_posix()
        manifest["split_sources"][split] = {"path": config["split_files"][split], "sha256": file_sha256(listing), "count": len(paths)}
        manifest["summary"][split] = {"images": len(paths), "negative_images": per_group}
    (prepared / "test.txt").write_text("", encoding="utf-8")
    common = {"path": prepared.as_posix(), "train": "train.txt", "val": "val.txt", "names": {0: "smoke", 1: "fire"}}
    (prepared / "data.yaml").write_text(yaml.safe_dump(common), encoding="utf-8")
    (prepared / "test.yaml").write_text(yaml.safe_dump(common), encoding="utf-8")
    manifest["artifact_sha256"] = {name: file_sha256(prepared / name) for name in
                                  ("train.txt", "val.txt", "test.txt", "data.yaml", "test.yaml")}
    manifest["fingerprint"] = fingerprint(manifest)
    write_json(prepared / "manifest.json", manifest)
    dataset_path = output / "dataset.json"
    write_json(dataset_path, config)
    suite = read_json(ROOT / "configs/experiments.json")
    suite["dataset_config"] = dataset_path.relative_to(ROOT).as_posix()
    suite["training"].update(epochs=2, imgsz=160, batch=4, close_mosaic=0, warmup_epochs=0.)
    suite["evaluation"].update(batch=4, max_case_images=2)
    suite["experiments"] = [{"id": "PIPELINE-CHECK", "stage": "integration", "reference": None, "overrides": {}}]
    suite_path = output / "suite.json"
    write_json(suite_path, suite)
    YOLO = yolo_class()

    def interrupt_second_epoch(trainer):
        if trainer.epoch == 1:
            raise KeyboardInterrupt("Intentional integration-check interruption after the first saved epoch")

    def instrumented_yolo(*args, **kwargs):
        model = YOLO(*args, **kwargs)
        model.add_callback("on_train_epoch_start", interrupt_second_epoch)
        return model

    previous_runs = set((ROOT / "experiments/runs").glob("PIPELINE-CHECK_*"))
    with patch.object(training, "yolo_class", return_value=instrumented_yolo):
        try:
            training.train(suite_path, "PIPELINE-CHECK")
        except KeyboardInterrupt:
            pass
        else:
            raise AssertionError("Expected the intentional interruption")
    created = set((ROOT / "experiments/runs").glob("PIPELINE-CHECK_*")) - previous_runs
    if len(created) != 1:
        raise AssertionError("Could not identify the integration training run")
    run = created.pop()
    if read_json(run / "training.json")["status"] != "interrupted":
        raise AssertionError("Interruption was not recorded")
    training.train(resume_run=run)
    evaluation = evaluate_validation(run)
    result = {"kind": "real_data_gpu_integration_check", "not_a_baseline_result": True,
              "train_images": 64, "val_images": 16, "epochs": 2, "imgsz": 160, "batch": 4,
              "interruption_and_resume": "passed", "training_run": str(run), "evaluation_run": str(evaluation),
              "source_dataset_fingerprint": source["fingerprint"], "test_evaluated": False}
    write_json(output / "check.json", result)
    print(output / "check.json")


if __name__ == "__main__":
    main()
