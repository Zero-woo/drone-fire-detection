"""Synthetic filesystem fixtures, not measured model experiments."""
import json
import sys
import tempfile
import unittest
from contextlib import ExitStack
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from fire_detection import common, dataset
from fire_detection.common import file_sha256, read_json, write_json


class DatasetTests(unittest.TestCase):
    def setUp(self):
        from PIL import Image
        (common.ROOT / "work").mkdir(exist_ok=True)
        self.tmp = tempfile.TemporaryDirectory(dir=common.ROOT / "work")
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        self.stack.enter_context(patch.object(common, "ROOT", self.root))
        self.stack.enter_context(patch.object(dataset, "ROOT", self.root))
        self.config = {"raw_root": "raw", "prepared_root": "data/processed/fixture", "split_files": {},
                       "names": ["smoke", "fire"], "annotation_policy": "strict",
                       "cross_split_duplicates": "error", "source_group_identity_available": False,
                       "published_split_identity_verified": False}
        for index, split in enumerate(("train", "val", "test")):
            for sub in ("images", "labels"):
                (self.root / "raw" / split / sub).mkdir(parents=True)
            Image.new("RGB", (20, 20), (index * 70, 10, 20)).save(self.root / "raw" / split / "images" / f"{split}.png")
            self.label(split).write_text("0 0.5 0.5 0.4 0.4\n", encoding="utf-8")
            listing = self.root / f"{split}.txt"
            listing.write_text(f"{split}.png\n", encoding="utf-8")
            self.config["split_files"][split] = listing.name
        self.config_path = self.root / "config.json"

    def label(self, split):
        return self.root / "raw" / split / "labels" / f"{split}.txt"

    def prepare(self):
        write_json(self.config_path, self.config)
        return dataset.prepare_dataset(self.config_path)

    def test_empty_label_is_negative_and_raw_is_unchanged(self):
        self.label("val").write_text("", encoding="utf-8")
        before = {str(p): file_sha256(p) for p in (self.root / "raw").rglob("*") if p.is_file()}
        prepared = self.prepare()
        manifest = dataset.verify_dataset(prepared)
        self.assertEqual(manifest["summary"]["val"]["negative_images"], 1)
        self.assertEqual(before, {str(p): file_sha256(p) for p in (self.root / "raw").rglob("*") if p.is_file()})
        self.assertNotIn("test:", (prepared / "data.yaml").read_text())

    def test_prepared_label_tampering_fails(self):
        prepared = self.prepare()
        (prepared / "train/labels/train.txt").write_text("", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "checksum mismatch"):
            dataset.verify_dataset(prepared)

    def test_split_artifact_tampering_fails(self):
        prepared = self.prepare()
        (prepared / "val.txt").write_text("train.png\n", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "artifact checksum mismatch"):
            dataset.verify_dataset(prepared)

    def test_manifest_tampering_fails(self):
        prepared = self.prepare()
        manifest = read_json(prepared / "manifest.json")
        manifest["records"][0]["image_sha256"] = "0" * 64
        write_json(prepared / "manifest.json", manifest)
        with self.assertRaisesRegex(ValueError, "fingerprint mismatch"):
            dataset.verify_dataset(prepared)

    def test_prepared_dataset_is_not_overwritten(self):
        self.prepare()
        with self.assertRaises(FileExistsError):
            self.prepare()

    def test_missing_label_blocks_preparation(self):
        self.label("train").unlink()
        with self.assertRaisesRegex(ValueError, "validation failed"):
            self.prepare()

    def test_orphan_label_blocks_preparation(self):
        (self.label("train").parent / "orphan.txt").write_text("")
        with self.assertRaisesRegex(ValueError, "validation failed"):
            self.prepare()

    def test_corrupt_image_blocks_preparation(self):
        (self.root / "raw/train/images/train.png").write_bytes(b"not an image")
        with self.assertRaisesRegex(ValueError, "validation failed"):
            self.prepare()

    def test_pixel_identical_different_files_crossing_splits_fail(self):
        from PIL import Image
        image = self.root / "raw/train/images/train.png"
        target = self.root / "raw/val/images/val.png"
        with Image.open(image) as im:
            im.save(target, compress_level=0)
        self.assertNotEqual(file_sha256(image), file_sha256(target))
        with self.assertRaisesRegex(ValueError, "validation failed"):
            self.prepare()

    def test_filename_crossing_splits_fails(self):
        (self.root / "val.txt").write_text("train.png\n")
        with self.assertRaisesRegex(ValueError, "validation failed"):
            self.prepare()

    def test_clip_and_drop_are_traced_without_raw_changes(self):
        self.config["annotation_policy"] = "clip_drop_degenerate"
        self.label("train").write_text("1 0.98 0.5 0.1 0.2\n0 0.5 0.5 0 0.1\n")
        digest = file_sha256(self.label("train"))
        prepared = self.prepare()
        effective = dataset.parse_label(prepared / "train/labels/train.txt")
        self.assertEqual(len(effective), 1)
        self.assertAlmostEqual(effective[0][1] + effective[0][3] / 2, 1)
        self.assertEqual(file_sha256(self.label("train")), digest)
        report = read_json(prepared / "audit.json")
        self.assertEqual(report["annotation_change_counts"], {"clip": 1, "drop_degenerate": 1})
        dataset.verify_dataset(prepared)

    def test_strict_policy_rejects_invalid_geometry(self):
        for line in ("0 0.5 0.5 nan 0.1", "2 0.5 0.5 0.1 0.1", "0 0.5 0.5 0 0.1",
                     "0 0.99 0.5 0.1 0.1", "0 0.5 0.5 -0.1 0.1", "0 0.5"):
            with self.subTest(line=line):
                self.label("train").write_text(line)
                with self.assertRaises(ValueError):
                    dataset.parse_label(self.label("train"))

    def test_jpeg_trailer_is_removed_without_pixel_change_or_ultralytics_rewrite(self):
        from PIL import Image
        png = self.root / "raw/train/images/train.png"
        jpeg = png.with_suffix(".jpg")
        with Image.open(png) as im:
            im.save(jpeg, "JPEG")
        png.unlink()
        jpeg.write_bytes(jpeg.read_bytes() + b"MediaType=default\n")
        (self.root / "train.txt").write_text("train.jpg\n")
        source_hash = file_sha256(jpeg)
        raw_pixels = dataset.inspect_image(jpeg)["pixel_sha256"]
        prepared = self.prepare()
        target = prepared / "train/images/train.jpg"
        self.assertTrue(target.read_bytes().endswith(b"\xff\xd9"))
        self.assertNotEqual(file_sha256(target), source_hash)
        self.assertEqual(dataset.inspect_image(target)["pixel_sha256"], raw_pixels)
        self.assertEqual(file_sha256(jpeg), source_hash)
        manifest = dataset.verify_dataset(prepared)
        row = next(r for r in manifest["records"] if r["split"] == "train")
        self.assertEqual(row["source_image_sha256"], source_hash)
        self.assertEqual(row["image_changes"][0]["removed_bytes"], len(b"MediaType=default\n"))
        # Exercise the exact installed third-party path that rewrote the original copies.
        common.yolo_class()
        from ultralytics.data.utils import verify_image_label
        digest = file_sha256(target)
        for _ in range(2):
            result = verify_image_label((str(target), str(prepared / "train/labels/train.txt"), "", False, 2, 0, 0, False))
            self.assertEqual(result[8], 0, result[9])
            self.assertNotIn("restored", result[9])
            self.assertEqual(file_sha256(target), digest)
        dataset.verify_unchanged_dataset(prepared, manifest["fingerprint"])

    def test_missing_jpeg_eoi_is_not_silently_reencoded(self):
        from PIL import Image
        image = self.root / "broken.jpg"
        Image.new("RGB", (20, 20)).save(image)
        image.write_bytes(image.read_bytes()[:-2])
        before = file_sha256(image)
        with self.assertRaises((OSError, ValueError)):
            dataset.inspect_image(image)
        self.assertEqual(file_sha256(image), before)

    def test_execution_guard_rejects_rewritten_image(self):
        prepared = self.prepare()
        manifest = dataset.verify_dataset(prepared)
        (prepared / "train/images/train.png").write_bytes(b"unexpected rewrite")
        with self.assertRaisesRegex(ValueError, "content checksum mismatch"):
            dataset.verify_unchanged_dataset(prepared, manifest["fingerprint"])


if __name__ == "__main__":
    unittest.main()
