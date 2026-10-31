"""Download checksummed immutable model artifacts; never accept silent replacement."""
from __future__ import annotations

import urllib.request
from pathlib import Path
from uuid import uuid4

from .common import ROOT, file_sha256, inside, read_json, utc_now, write_json


def spec_for(name):
    return read_json(ROOT / "configs/models.json")[name]


def verify_model(path, spec):
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Model missing: {path}; run fetch-models first.")
    if path.stat().st_size != spec["size_bytes"] or file_sha256(path) != spec["sha256"]:
        raise ValueError(f"Model checksum/size mismatch: {path}; refusing to load or overwrite.")
    return path


def fetch_model(name):
    spec = spec_for(name)
    path = inside(ROOT / spec["local_path"], ROOT / "models")
    if path.exists():
        return verify_model(path, spec)
    if not spec["url"].startswith(("https://huggingface.co/", "https://github.com/ultralytics/")):
        raise ValueError("Unexpected model download host.")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + "." + uuid4().hex + ".part")
    try:
        request = urllib.request.Request(spec["url"], headers={"User-Agent": "firelab/0.2"})
        with urllib.request.urlopen(request, timeout=60) as response, temporary.open("xb") as output:
            size = 0
            while chunk := response.read(1024 * 1024):
                size += len(chunk)
                if size > spec["size_bytes"]:
                    raise ValueError("Downloaded model exceeds registered size.")
                output.write(chunk)
        verify_model(temporary, spec)
        if path.exists():
            verify_model(path, spec)
        else:
            temporary.rename(path)
        write_json(path.with_suffix(".metadata.json"), {**spec, "downloaded_at_utc": utc_now()})
        return path
    finally:
        temporary.unlink(missing_ok=True)
