"""Colab transfer fixtures; no remote GPU execution or model performance is claimed."""
import ast
import json
import os
import subprocess
import sys
import unittest
import tempfile
import zipfile
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import test_dataset as dataset_fixtures
from fire_detection import colab, common, dataset
from fire_detection.common import file_sha256, read_json, write_json


class ColabTests(unittest.TestCase):
    def setUp(self):
        self.fixture = dataset_fixtures.DatasetTests("test_empty_label_is_negative_and_raw_is_unchanged")
        self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)
        self.root = self.fixture.root
        self.addCleanup(patch.stopall)
        patch.object(colab, "ROOT", self.root).start()
        self.prepared = self.fixture.prepare()
        self.archive = self.root / "outputs/fixture.zip"

    def export(self):
        return colab.export_data(self.fixture.config_path, self.archive)

    def destination(self):
        destination = self.root / "another-machine"
        destination.mkdir()
        config_path = destination / "config.json"
        write_json(config_path, self.fixture.config)
        return destination, config_path

    def test_transfer_preserves_exact_inputs_and_rebuilds_absolute_paths(self):
        source = dataset.verify_dataset(self.prepared)
        bundle = self.export()
        target_root, config = self.destination()
        with patch.object(common, "ROOT", target_root), patch.object(colab, "ROOT", target_root):
            imported = colab.import_data(self.archive, bundle["archive_sha256"], config)
            manifest = dataset.verify_dataset(imported)
            self.assertEqual(manifest["fingerprint"], source["fingerprint"])
            self.assertEqual(manifest["records"], source["records"])
            self.assertIn(imported.as_posix(), (imported / "train.txt").read_text())
            self.assertNotEqual(manifest["artifact_sha256"], source["artifact_sha256"])
            self.assertEqual(colab.import_data(self.archive, bundle["archive_sha256"], config), imported)
        with zipfile.ZipFile(self.archive) as archive:
            self.assertTrue(all(name == "bundle.json" or name.startswith("dataset/") for name in archive.namelist()))

    def test_archive_checksum_mismatch_fails_before_extraction(self):
        self.export()
        target_root, config = self.destination()
        with patch.object(common, "ROOT", target_root), patch.object(colab, "ROOT", target_root):
            with self.assertRaisesRegex(ValueError, "archive checksum mismatch"):
                colab.import_data(self.archive, "0" * 64, config)
            self.assertFalse((target_root / "data/processed").exists())

    def test_export_does_not_overwrite_archive(self):
        self.export()
        with self.assertRaises(FileExistsError):
            self.export()

    def test_changed_input_is_not_exported(self):
        (self.prepared / "train/images/train.png").write_bytes(b"changed fixture")
        with self.assertRaisesRegex(ValueError, "checksum mismatch"):
            self.export()

    def test_archive_parent_traversal_is_rejected(self):
        self.export()
        with zipfile.ZipFile(self.archive, "a") as archive:
            archive.writestr("dataset/../../escape.txt", "fixture")
        digest = file_sha256(self.archive)
        target_root, config = self.destination()
        with patch.object(common, "ROOT", target_root), patch.object(colab, "ROOT", target_root):
            with self.assertRaisesRegex(ValueError, "Unsafe"):
                colab.import_data(self.archive, digest, config)
            self.assertFalse((target_root / "escape.txt").exists())

    def test_persistent_run_root_is_explicit_and_keeps_default(self):
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("FIRELAB_RUNS_DIR", None)
            self.assertEqual(common.runs_root(), self.root / "experiments/runs")
            persistent = self.root / "drive/runs"
            os.environ["FIRELAB_RUNS_DIR"] = str(persistent)
            created = common.new_run("fixture")
            self.assertEqual(created.parent, persistent.resolve())

    def test_colab_notebook_code_is_parseable_and_has_no_saved_execution(self):
        path = Path(__file__).resolve().parents[1] / "notebooks/colab_training.ipynb"
        if not path.exists():
            self.skipTest("Notebook is generated in the next preparation step")
        notebook = read_json(path)
        self.assertEqual(notebook["nbformat"], 4)
        for cell in notebook["cells"]:
            if cell["cell_type"] == "code":
                ast.parse("".join(cell["source"]))
                self.assertIsNone(cell["execution_count"])
                self.assertEqual(cell["outputs"], [])


class PlotEnvironmentTests(unittest.TestCase):
    def test_fresh_process_renders_png_with_inherited_colab_inline_backend(self):
        root = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory(dir=root / "work") as temporary:
            png = Path(temporary) / "regression.png"
            env = os.environ.copy()
            env["MPLBACKEND"] = "module://matplotlib_inline.backend_inline"
            script = (
                "import sys,json;sys.path.insert(0,sys.argv[1]);"
                "from fire_detection.common import configure_plotting;configure_plotting();"
                "import matplotlib;import matplotlib.pyplot as plt;"
                "fig,ax=plt.subplots();ax.plot([0,1],[0,1]);fig.savefig(sys.argv[2]);plt.close(fig);"
                "print(json.dumps({'backend':matplotlib.get_backend()}))"
            )
            result = subprocess.run([sys.executable, "-c", script, str(root / "src"), str(png)],
                                    env=env, capture_output=True, text=True, timeout=60)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(json.loads(result.stdout)["backend"].lower(), "agg")
            self.assertTrue(png.read_bytes().startswith(b"\x89PNG\r\n\x1a\n"))

    def test_notebook_sanitizes_child_environment_without_changing_kernel_settings(self):
        path = Path(__file__).resolve().parents[1] / "notebooks/colab_training.ipynb"
        notebook = read_json(path)
        source = next("".join(c["source"]) for c in notebook["cells"]
                      if c["cell_type"] == "code" and "def run(arguments" in "".join(c["source"]))
        namespace = {}
        exec(compile(source, str(path), "exec"), namespace)
        inherited = {"MPLBACKEND": "module://matplotlib_inline.backend_inline",
                     "CUDA_VISIBLE_DEVICES": "0", "FIRELAB_RUNS_DIR": "/drive/runs"}
        with patch("subprocess.Popen") as popen:
            popen.return_value.stdout = iter(())
            popen.return_value.wait.return_value = 0
            namespace["run"](["fixture-python"], env=inherited)
            child = popen.call_args.kwargs["env"]
        self.assertEqual(child["MPLBACKEND"], "Agg")
        self.assertEqual(child["CUDA_VISIBLE_DEVICES"], "0")
        self.assertEqual(child["FIRELAB_RUNS_DIR"], "/drive/runs")
        self.assertEqual(inherited["MPLBACKEND"], "module://matplotlib_inline.backend_inline")


if __name__ == "__main__":
    unittest.main()
