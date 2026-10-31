"""Aggregate only measured runs and export comparison tables and figures."""
from __future__ import annotations

import csv
import statistics
from collections import defaultdict
from pathlib import Path

from .common import ROOT, new_run, object_sha256, read_json, write_json


def compare_runs(run_paths):
    evaluations, trainings = [], []
    for folder in run_paths:
        folder = Path(folder)
        if (folder / "evaluation.json").is_file():
            evaluations.append((folder, read_json(folder / "evaluation.json")))
        if (folder / "training.json").is_file():
            trainings.append((folder, read_json(folder / "training.json")))
    # Integration checks are not hyperparameter experiment results.
    evaluations = [(p, r) for p, r in evaluations if r["status"] == "completed" and
                   r["config"]["experiment"]["stage"] != "integration"]
    if not evaluations:
        raise ValueError("No completed experiment evaluations; no scores or plots were fabricated.")
    protocols = {(r["split"], r["dataset_fingerprint"], r["config"]["evaluation"]["operating_conf"],
                  r["config"]["evaluation"]["ap_conf"], r["config"]["evaluation"]["nms_iou"],
                  r["config"]["evaluation"]["match_iou"], r["config"]["evaluation"]["max_det"])
                 for _, r in evaluations}
    if len(protocols) != 1:
        raise ValueError("Compare only the same split, dataset fingerprint, confidence, NMS and matching settings.")
    output = new_run("comparison", ROOT / "outputs")
    rows = []
    for folder, record in evaluations:
        config, metrics, op = record["config"], record["metrics"], record["operating"]
        train_path = Path(record["training_run"]) if record["training_run"] else None
        training = read_json(train_path / "training.json") if train_path else {}
        row = {"experiment_id": record["experiment_id"], "role": record["model_role"],
               "split": record["split"], "evaluation_path": str(folder),
               "checkpoint_sha256": record["checkpoint_sha256"], "dataset_fingerprint": record["dataset_fingerprint"],
               "stage": config["experiment"]["stage"], "reference": config["experiment"].get("reference"),
               **{k: config["training"][k] if record["model_role"] == "trained" else None
                  for k in ("epochs", "lr0", "optimizer", "batch", "seed", "hsv_s", "hsv_v", "mosaic")},
               "imgsz": config["training"]["imgsz"],
               **{k: metrics[k] for k in ("precision", "recall", "map50", "map50_95", "f1_macro")},
               "smoke_ap50_95": metrics["classes"]["smoke"]["ap50_95"],
               "fire_ap50_95": metrics["classes"]["fire"]["ap50_95"],
               "smoke_recall_fixed": op["classes"]["smoke"]["recall"],
               "fire_recall_fixed": op["classes"]["fire"]["recall"],
               "negative_fp_detections": op["negative_fp_detections"],
               "inference_ms": op["inference_ms_per_image"], "fps": op["inference_fps"],
               "peak_allocated_bytes": record["peak_allocated_bytes"],
               "training_wall_seconds": training.get("training_wall_seconds")}
        rows.append(row)
    with (output / "results.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    write_json(output / "results.json", rows)
    groups = defaultdict(list)
    evaluation_records = {str(path): record for path, record in evaluations}
    # Repeated evaluations of a checkpoint must not masquerade as repeated seeds.
    unique_checkpoints = set()
    for row in rows:
        if row["role"] == "trained" and row["checkpoint_sha256"] not in unique_checkpoints:
            matched = evaluation_records[row["evaluation_path"]]
            training_settings = {k: v for k, v in matched["config"]["training"].items() if k != "seed"}
            key = row["experiment_id"] + ":" + object_sha256(training_settings)[:12]
            # Repeated runs with one seed do not count as independent seeds.
            if row["seed"] not in {r["seed"] for r in groups[key]}:
                groups[key].append(row)
            unique_checkpoints.add(row["checkpoint_sha256"])
    repeated = {key: {"n": len(values), "seeds": [r["seed"] for r in values],
                      "map50_95_mean": statistics.mean(r["map50_95"] for r in values),
                      "map50_95_sample_std": statistics.stdev(r["map50_95"] for r in values) if len(values) > 1 else None}
                for key, values in groups.items()}
    write_json(output / "seed-summary.json", repeated)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    labels = [r["experiment_id"] + "\n" + Path(r["evaluation_path"]).name[-8:] for r in rows]
    for filename, columns, ylabel in (
        ("accuracy", ("map50", "map50_95"), "AP"),
        ("precision-recall", ("precision", "recall"), "Best-F1 operating point"),
        ("fixed-recall", ("smoke_recall_fixed", "fire_recall_fixed"), "Fixed-threshold recall"),
    ):
        fig, ax = plt.subplots(figsize=(max(7, len(rows) * 1.2), 4))
        for index, key in enumerate(columns):
            ax.bar([x + index * .36 for x in range(len(rows))], [r[key] for r in rows], width=.36, label=key)
        ax.set_xticks([x + .18 for x in range(len(rows))], labels, rotation=25, ha="right")
        ax.set_ylabel(ylabel)
        ax.set_ylim(0, 1)
        ax.legend()
        fig.tight_layout()
        fig.savefig(output / (filename + ".png"), dpi=160)
        plt.close(fig)
    for stage, xkey in (("lr", "lr0"), ("augmentation", "experiment_id"), ("resolution", "imgsz")):
        selected = [r for r in rows if r["role"] == "trained" and (r["stage"] == stage or r["experiment_id"] == "E1")]
        if len(selected) < 2:
            continue
        fig, axes = plt.subplots(1, 2 if stage == "resolution" else 1, figsize=(9, 4), squeeze=False)
        axes[0, 0].bar([str(r[xkey]) for r in selected], [r["map50_95"] for r in selected])
        axes[0, 0].set(xlabel=xkey, ylabel="mAP50-95")
        if stage == "resolution":
            axes[0, 1].bar([str(r[xkey]) for r in selected], [r["inference_ms"] for r in selected])
            axes[0, 1].set(xlabel=xkey, ylabel="Inference ms/image")
        fig.tight_layout()
        fig.savefig(output / f"{stage}.png", dpi=160)
        plt.close(fig)
    # Loss/AP curves come from actual Ultralytics epoch CSVs, including unsuccessful experiments.
    training_paths = {p.resolve() for p, _ in trainings}
    training_paths.update(Path(r["training_run"]).resolve() for _, r in evaluations if r["training_run"])
    for folder in training_paths:
        csv_path = folder / "train/results.csv"
        if not csv_path.is_file():
            continue
        with csv_path.open(encoding="utf-8-sig") as handle:
            history = [{key.strip(): value.strip() for key, value in row.items()} for row in csv.DictReader(handle)]
        if not history:
            continue
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        epochs = [float(r["epoch"]) for r in history]
        for key in history[0]:
            if "loss" in key:
                axes[0].plot(epochs, [float(r[key]) for r in history], label=key)
            if key in ("metrics/mAP50(B)", "metrics/mAP50-95(B)"):
                axes[1].plot(epochs, [float(r[key]) for r in history], label=key)
        for ax in axes:
            ax.set_xlabel("Epoch")
            ax.legend(fontsize=7)
        fig.tight_layout()
        fig.savefig(output / f"curves-{folder.name}.png", dpi=160)
        plt.close(fig)
    lines = ["# 실험 결과 비교", "", "실제 실행된 평가만 수록한다. 주 선택 기준은 validation mAP50-95이다.", "",
             "| 실험 | 역할 | mAP50-95 | Fire recall (고정) | Negative FP | 추론 ms |", "|---|---|---:|---:|---:|---:|"]
    for row in rows:
        lines.append(f"| {row['experiment_id']} | {row['role']} | {row['map50_95']:.4f} | {row['fire_recall_fixed']:.4f} | {row['negative_fp_detections']} | {row['inference_ms']:.3f} |")
    lines += ["", "## 해석 및 제한", "", "- 측정 사실: 표는 동일 fingerprint와 동일 confidence/NMS 설정의 결과이다.",
              "- 학습곡선의 loss 및 AP 추이를 확인하고 과적합 여부와 설정별 차이를 해석해야 한다.",
              "- 변화의 원인은 통제 실험과 반복 seed 결과가 확보되기 전까지 추정으로 취급한다.",
              "- 공개 모델의 게시 수치(mAP50-95 0.430)는 로컬 측정과 분리한다. 학습/검증 split 동일성은 미확인이다.",
              "- 로컬 라벨 보정, 출처별 session 정보 부재, 실제 화재 환경과의 도메인 차이가 비교의 한계다.",
              "- 모델 선택은 validation에서 수행한다. test 결과는 선택 근거에 사용하지 않는다.",
              "- 장치/버전/해상도/배치가 다른 속도 수치는 동일 조건의 속도 우위로 해석할 수 없다.", ""]
    (output / "report.md").write_text("\n".join(lines), encoding="utf-8")
    return output
