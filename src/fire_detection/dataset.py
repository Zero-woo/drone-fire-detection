"""Strict D-Fire validation and immutable, raw-preserving training copies."""
from __future__ import annotations

import hashlib
import math
import shutil
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from .common import (ROOT, environment, file_sha256, git_state, inside, new_run, object_sha256,
                     read_json, resolve, utc_now, write_json)

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def read_annotations(path, policy="strict"):
    boxes, changes = [], []
    for number, line in enumerate(Path(path).read_text(encoding="utf-8-sig").splitlines(), 1):
        if not line.strip():
            continue
        fields = line.split()
        if len(fields) != 5:
            raise ValueError(f"{path}:{number}: expected class x y w h")
        try:
            cls = int(fields[0])
            x, y, width, height = map(float, fields[1:])
        except ValueError as exc:
            raise ValueError(f"{path}:{number}: invalid number") from exc
        if cls not in (0, 1) or not all(math.isfinite(v) for v in (x, y, width, height)):
            raise ValueError(f"{path}:{number}: invalid class or non-finite coordinate")
        original = [cls, x, y, width, height]
        if width < 0 or height < 0:
            raise ValueError(f"{path}:{number}: negative box dimension")
        invalid_range = not (0 <= x <= 1 and 0 <= y <= 1 and 0 < width <= 1 and 0 < height <= 1)
        outside = min(x - width / 2, y - height / 2) < -1e-6 or max(x + width / 2, y + height / 2) > 1 + 1e-6
        if invalid_range or outside:
            if policy != "clip_drop_degenerate":
                raise ValueError(f"{path}:{number}: invalid range or box extends outside image")
            left, top = max(0., min(1., x - width / 2)), max(0., min(1., y - height / 2))
            right, bottom = max(0., min(1., x + width / 2)), max(0., min(1., y + height / 2))
            if right <= left or bottom <= top:
                changes.append({"line": number, "action": "drop_degenerate", "before": original, "after": None})
                continue
            x, y, width, height = (left + right) / 2, (top + bottom) / 2, right - left, bottom - top
            changes.append({"line": number, "action": "clip", "before": original, "after": [cls, x, y, width, height]})
        boxes.append([cls, x, y, width, height])
    if len({tuple(box) for box in boxes}) != len(boxes):
        raise ValueError(f"{path}: duplicate annotation rows")
    return boxes, changes


def parse_label(path):
    return read_annotations(path)[0]


def inspect_pair(pair):
    from PIL import Image
    image, label, policy = pair
    try:
        with Image.open(image) as im:
            im.verify()
        with Image.open(image) as im:
            rgb = im.convert("RGB")
            width, height = rgb.size
            if min(width, height) < 10:
                raise ValueError("Image is smaller than 10 pixels")
            pixel_hash = hashlib.sha256(f"{width}x{height}:".encode() + rgb.tobytes()).hexdigest()
        record = {"image": str(image), "label": str(label), "image_sha256": file_sha256(image),
                  "label_sha256": file_sha256(label), "source_label_sha256": file_sha256(label),
                  "pixel_sha256": pixel_hash, "width": width, "height": height}
        try:
            boxes, changes = read_annotations(label, policy)
            record.update(boxes=boxes, annotation_changes=changes)
            if changes:
                text = "".join(str(int(b[0])) + " " + " ".join(format(v, ".17g") for v in b[1:]) + "\n" for b in boxes)
                record["effective_label_text"] = text
                record["label_sha256"] = hashlib.sha256(text.encode("utf-8")).hexdigest()
        except (OSError, ValueError) as exc:
            record.update(error=str(exc), boxes=[], annotation_changes=[])
        return record
    except (OSError, ValueError) as exc:
        return {"image": str(image), "error": str(exc)}


def fingerprint(manifest):
    return object_sha256({key: manifest[key] for key in ("config", "split_sources", "records")})


def prepare_dataset(config_path):
    config = read_json(config_path)
    if config["names"] != ["smoke", "fire"]:
        raise ValueError("D-Fire class mapping must be 0=smoke, 1=fire.")
    if config["cross_split_duplicates"] not in ("error", "quarantine"):
        raise ValueError("cross_split_duplicates must be error or quarantine")
    policy = config.get("annotation_policy", "strict")
    if policy not in ("strict", "clip_drop_degenerate"):
        raise ValueError("Unknown annotation_policy")
    raw = resolve(config["raw_root"])
    destination = inside(resolve(config["prepared_root"]), ROOT / "data/processed")
    if destination.exists():
        raise FileExistsError(f"Prepared dataset already exists: {destination}. Verify it or choose a new prepared_root.")
    audit_dir = new_run("dataset-audit")
    errors, warnings, pairs, stems = [], [], {}, set()
    images = sorted(p for p in raw.rglob("*") if p.suffix.lower() in IMAGE_SUFFIXES)
    if not images:
        errors.append(f"No images under {raw}")
    for image in images:
        if image.parent.name != "images":
            errors.append(f"Unexpected image layout: {image}")
            continue
        if image.name in pairs:
            errors.append(f"Ambiguous image filename: {image.name}")
        stem_key = (str(image.parent), image.stem.lower())
        if stem_key in stems:
            errors.append(f"Multiple images share a label stem: {image}")
        stems.add(stem_key)
        label = image.parent.parent / "labels" / (image.stem + ".txt")
        if not label.is_file():
            errors.append(f"Missing label: {label}")
        else:
            pairs[image.name] = (image, label, policy)
    expected_labels = {label.resolve() for _, label, _ in pairs.values()}
    for label in raw.glob("*/labels/*.txt"):
        if label.resolve() not in expected_labels:
            errors.append(f"Orphan label: {label}")
    splits, split_sources = {}, {}
    for split in ("train", "val", "test"):
        source = resolve(config["split_files"][split])
        entries = [line.strip() for line in source.read_text(encoding="utf-8-sig").splitlines() if line.strip()]
        if any(Path(name).name != name or "/" in name or "\\" in name for name in entries):
            raise ValueError(f"Split lists must contain bare filenames: {source}")
        if not entries or len(set(entries)) != len(entries):
            errors.append(f"Empty or repeated entries in {source}")
        missing = sorted(set(entries) - set(pairs))
        errors.extend(f"{split}: missing image/label {name}" for name in missing)
        splits[split] = entries
        split_sources[split] = {"path": config["split_files"][split], "sha256": file_sha256(source), "count": len(entries)}
    memberships = defaultdict(list)
    for split, entries in splits.items():
        for name in entries:
            memberships[name].append(split)
    errors.extend(f"Filename crosses splits: {name}: {where}" for name, where in memberships.items() if len(where) > 1)
    errors.extend(f"Image not assigned to a split: {name}" for name in sorted(set(pairs) - set(memberships)))
    print(f"Auditing {len(pairs)} image/label pairs...", flush=True)
    with ThreadPoolExecutor(max_workers=8) as pool:
        scanned = list(pool.map(inspect_pair, pairs.values()))
    records = {}
    for record in scanned:
        if "error" in record:
            errors.append(f"{record['image']}: {record['error']}")
        if "pixel_sha256" in record:
            name = Path(record["image"]).name
            record["image"] = Path(record["image"]).relative_to(ROOT).as_posix()
            record["label"] = Path(record["label"]).relative_to(ROOT).as_posix()
            records[name] = record
    by_hash = defaultdict(list)
    for name, record in records.items():
        by_hash[record["pixel_sha256"]].append(name)
    duplicate_groups, quarantined = [], []
    priority = {"test": 0, "val": 1, "train": 2}
    for digest, names in by_hash.items():
        if len(names) < 2:
            continue
        where = {split for name in names for split in memberships[name]}
        group = {"pixel_sha256": digest, "images": sorted(names), "splits": sorted(where)}
        duplicate_groups.append(group)
        if len(where) > 1:
            if config["cross_split_duplicates"] == "error":
                errors.append(f"Identical pixels cross splits: {names}")
            else:
                keep_split = min(where, key=priority.get)
                for split in where - {keep_split}:
                    for name in sorted(set(names) & set(splits[split])):
                        splits[split].remove(name)
                        quarantined.append({"split": split, "image": name, "kept_split": keep_split, "pixel_sha256": digest})
    if not config["source_group_identity_available"]:
        warnings.append("Source video/session IDs unavailable: exact-pixel checks cannot rule out near-frame leakage.")
    if not config["published_split_identity_verified"]:
        warnings.append("Publisher split identity unverified: published scores are contextual references only.")
    summary = {}
    for split, entries in splits.items():
        valid = [records[name] for name in entries if name in records and "error" not in records[name]]
        counts = Counter(int(box[0]) for record in valid for box in record["boxes"])
        summary[split] = {"images": len(entries), "valid_images": len(valid), "negative_images": sum(not r["boxes"] for r in valid),
                          "objects": {name: counts[i] for i, name in enumerate(config["names"])},
                          "images_with_class": {name: sum(any(b[0] == i for b in r["boxes"]) for r in valid)
                                                for i, name in enumerate(config["names"])}}
    report = {"created_at_utc": utc_now(), "git": git_state(), "environment": environment(), "config": config,
              "status": "failed" if errors else "passed", "summary": summary,
              "split_sources": split_sources, "errors": errors, "warnings": warnings,
              "duplicate_groups": duplicate_groups, "quarantined": quarantined}
    report["annotation_changes"] = [{"image": name, "source_label_sha256": r["source_label_sha256"],
                                      "effective_label_sha256": r["label_sha256"], "changes": r["annotation_changes"]}
                                     for name, r in records.items() if r["annotation_changes"]]
    report["annotation_change_counts"] = dict(Counter(change["action"] for r in records.values() for change in r["annotation_changes"]))
    write_json(audit_dir / "audit.json", report)
    print(f"Audit: {audit_dir / 'audit.json'}; errors={len(errors)}, duplicate_groups={len(duplicate_groups)}", flush=True)
    if errors:
        raise ValueError(f"Dataset validation failed with {len(errors)} errors; see {audit_dir / 'audit.json'}")
    destination.mkdir(parents=True, exist_ok=False)
    manifest = {"config": config, "split_sources": split_sources, "records": [],
                "summary": summary, "audit": str(audit_dir / "audit.json"), "quarantined": quarantined}
    try:
        for split, entries in splits.items():
            paths = []
            for name in entries:
                record = records[name]
                target_image = destination / split / "images" / name
                target_label = destination / split / "labels" / (Path(name).stem + ".txt")
                target_image.parent.mkdir(parents=True, exist_ok=True)
                target_label.parent.mkdir(parents=True, exist_ok=True)
                # Copies protect raw even if a third-party library repairs a JPEG.
                shutil.copyfile(resolve(record["image"]), target_image)
                if "effective_label_text" in record:
                    target_label.write_bytes(record["effective_label_text"].encode("utf-8"))
                else:
                    shutil.copyfile(resolve(record["label"]), target_label)
                paths.append(target_image.as_posix())
                manifest["records"].append({**{k: v for k, v in record.items() if k != "effective_label_text"}, "split": split,
                    "prepared_image": target_image.relative_to(destination).as_posix(),
                    "prepared_label": target_label.relative_to(destination).as_posix()})
            (destination / f"{split}.txt").write_text("\n".join(paths) + "\n", encoding="utf-8")
        import yaml
        common = {"path": destination.as_posix(), "train": "train.txt", "val": "val.txt", "names": {0: "smoke", 1: "fire"}}
        # Training YAML deliberately has no test entry.
        (destination / "data.yaml").write_text(yaml.safe_dump(common), encoding="utf-8")
        (destination / "test.yaml").write_text(yaml.safe_dump({**common, "test": "test.txt"}), encoding="utf-8")
        manifest["artifact_sha256"] = {name: file_sha256(destination / name) for name in
                                      ("train.txt", "val.txt", "test.txt", "data.yaml", "test.yaml")}
        manifest["fingerprint"] = fingerprint(manifest)
        write_json(destination / "manifest.json", manifest)
        write_json(destination / "audit.json", report)
    except BaseException:
        # Preserve incomplete copies for inspection; absence of manifest prevents training.
        raise
    return destination


def verify_dataset(path):
    path = Path(path).resolve()
    manifest = read_json(path / "manifest.json")
    if fingerprint(manifest) != manifest["fingerprint"]:
        raise ValueError("Dataset manifest fingerprint mismatch")
    for name, digest in manifest["artifact_sha256"].items():
        if file_sha256(inside(path / name, path)) != digest:
            raise ValueError(f"Dataset artifact checksum mismatch: {name}")
    def verify_record(record):
        for key, digest_key in (("prepared_image", "image_sha256"), ("prepared_label", "label_sha256")):
            file = inside(path / record[key], path)
            if file_sha256(file) != record[digest_key]:
                raise ValueError(f"Dataset content checksum mismatch: {file}")
    with ThreadPoolExecutor(max_workers=8) as pool:
        list(pool.map(verify_record, manifest["records"]))
    return manifest
