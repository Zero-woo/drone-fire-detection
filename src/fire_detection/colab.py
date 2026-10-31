"""Move exact validated inputs across machines while keeping dataset identity fixed."""
from __future__ import annotations

import json
import re
import shutil
import stat
import zipfile
from pathlib import Path, PurePosixPath
from uuid import uuid4

from .common import (ROOT, file_sha256, git_state, inside, new_run, read_json,
                     resolve, utc_now, write_json)
from .dataset import fingerprint, verify_dataset


def export_data(config_path, output=None):
    config = read_json(config_path)
    prepared = resolve(config["prepared_root"])
    manifest = verify_dataset(prepared)
    if config != manifest["config"]:
        raise ValueError("Configured dataset differs from prepared data")
    if output is None:
        output = new_run("colab-data", ROOT / "outputs") / "dfire-fold1-v2.zip"
    output = inside(output, ROOT / "outputs")
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite bundle: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    files = {"manifest.json", *manifest["artifact_sha256"]}
    if (prepared / "audit.json").is_file():
        files.add("audit.json")
    for row in manifest["records"]:
        files.update((row["prepared_image"], row["prepared_label"]))
    metadata = {"schema": 1, "created_at_utc": utc_now(), "source_git": git_state(),
                "dataset_fingerprint": manifest["fingerprint"], "dataset_config": config,
                "input_files": len(files), "images": len(manifest["records"])}
    temporary = output.with_name(output.name + "." + uuid4().hex + ".part")
    try:
        with zipfile.ZipFile(temporary, "x", compression=zipfile.ZIP_STORED, allowZip64=True) as archive:
            archive.writestr("bundle.json", json.dumps(metadata, ensure_ascii=False, indent=2))
            for name in sorted(files):
                archive.write(inside(prepared / name, prepared), "dataset/" + name)
        if output.exists():
            raise FileExistsError(f"Bundle appeared during export: {output}")
        temporary.rename(output)
    finally:
        temporary.unlink(missing_ok=True)
    result = {**metadata, "archive": str(output), "archive_size_bytes": output.stat().st_size,
              "archive_sha256": file_sha256(output)}
    write_json(output.with_suffix(".json"), result)
    return result


def import_data(archive_path, expected_sha256, config_path):
    """Import a pinned bundle and rebuild only machine-dependent paths/YAML checksums."""
    archive_path = Path(archive_path)
    if not re.fullmatch(r"[0-9a-f]{64}", expected_sha256):
        raise ValueError("Supply the 64-character SHA256 printed by export-colab")
    if file_sha256(archive_path) != expected_sha256:
        raise ValueError("Colab archive checksum mismatch; refusing to extract")
    config = read_json(config_path)
    destination = inside(resolve(config["prepared_root"]), ROOT / "data/processed")
    with zipfile.ZipFile(archive_path) as archive:
        metadata = json.loads(archive.read("bundle.json"))
        if metadata.get("schema") != 1 or metadata["dataset_config"] != config:
            raise ValueError("Bundle version/configuration differs from this repository")
        if destination.exists():
            existing = verify_dataset(destination)
            if existing["fingerprint"] != metadata["dataset_fingerprint"]:
                raise ValueError("Existing dataset differs from the Colab bundle")
            return destination
        stage = new_run("colab-import", ROOT / "work")
        seen = set()
        for member in archive.infolist():
            name = PurePosixPath(member.filename)
            if (member.filename in seen or "\\" in member.filename or ":" in member.filename
                    or name.is_absolute() or ".." in name.parts
                    or stat.S_ISLNK(member.external_attr >> 16)
                    or (member.filename != "bundle.json" and (not name.parts or name.parts[0] != "dataset"))):
                raise ValueError(f"Unsafe or repeated archive path: {member.filename}")
            seen.add(member.filename)
            target = inside(stage / name.as_posix(), stage)
            if member.is_dir():
                target.mkdir(parents=True, exist_ok=True)
            else:
                target.parent.mkdir(parents=True, exist_ok=True)
                with archive.open(member) as source, target.open("xb") as output:
                    shutil.copyfileobj(source, output, length=1024 * 1024)
    extracted = inside(stage / "dataset", ROOT / "work")
    manifest = verify_dataset(extracted)
    if manifest["fingerprint"] != metadata["dataset_fingerprint"]:
        raise ValueError("Extracted dataset fingerprint mismatch")
    # Windows absolute paths are not executable input lists on Linux. Image/label bytes stay exact.
    import yaml
    for split in ("train", "val", "test"):
        paths = [(destination / row["prepared_image"]).as_posix()
                 for row in manifest["records"] if row["split"] == split]
        (extracted / f"{split}.txt").write_bytes(("\n".join(paths) + "\n").encode("utf-8"))
    common = {"path": destination.as_posix(), "train": "train.txt", "val": "val.txt", "names": {0: "smoke", 1: "fire"}}
    (extracted / "data.yaml").write_bytes(yaml.safe_dump(common).encode("utf-8"))
    (extracted / "test.yaml").write_bytes(yaml.safe_dump({**common, "test": "test.txt"}).encode("utf-8"))
    previous_artifacts = manifest["artifact_sha256"]
    manifest["artifact_sha256"] = {name: file_sha256(extracted / name) for name in previous_artifacts}
    manifest["relocation"] = {"created_at_utc": utc_now(), "bundle_sha256": expected_sha256,
                              "original_artifact_sha256": previous_artifacts}
    if fingerprint(manifest) != metadata["dataset_fingerprint"]:
        raise ValueError("Relocation changed dataset identity")
    write_json(extracted / "manifest.json", manifest)
    destination.parent.mkdir(parents=True, exist_ok=True)
    # Both absolute paths have been constrained above to work/ and data/processed/ respectively.
    extracted.rename(destination)
    verify_dataset(destination)
    return destination
