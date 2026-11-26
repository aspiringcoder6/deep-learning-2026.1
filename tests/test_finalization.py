import csv
import hashlib
import io
import json
import tempfile
import unittest
from argparse import Namespace
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

from essay_scoring.__main__ import evaluate, finalize
from essay_scoring.data import create_manifest


class TestHoldoutGate(unittest.TestCase):
    def test_legacy_run_and_manifest_hashes_survive_windows_line_endings(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data = root / "essays.tsv"
            with data.open("w", newline="", encoding="utf-8") as file:
                writer = csv.writer(file, delimiter="\t")
                writer.writerow(["essay_id", "essay_set", "essay", "domain1_score"])
                for index in range(20):
                    writer.writerow([index, 1, f"Essay number {index}.", 8])
            manifest, _ = create_manifest(data, 1, 42)
            manifest_file = root / "split.json"
            manifest_bytes = (json.dumps(manifest, indent=2) + "\n").encode().replace(b"\n", b"\r\n")
            manifest_file.write_bytes(manifest_bytes)
            run_dir = root / "ridge"
            run_dir.mkdir()
            record = {
                "essay_set": 1, "model": "ridge", "validation_metrics": {"qwk": None, "mse": 0.0},
                "dataset_sha256": manifest["dataset_sha256"], "splits": manifest["splits"],
                "manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
            }
            run_bytes = (json.dumps(record, indent=2) + "\n").encode().replace(b"\n", b"\r\n")
            (run_dir / "run.json").write_bytes(run_bytes)
            (run_dir / "model.pkl").write_bytes(b"checkpoint fixture")
            with redirect_stdout(io.StringIO()):
                finalize(Namespace(run_dir=run_dir, note="Selected using synthetic validation."))
            selection_file = root / "final-selection.json"
            selection = json.loads(selection_file.read_text(encoding="utf-8"))
            selection["run_sha256"] = hashlib.sha256(run_bytes).hexdigest()
            selection_file.write_text(json.dumps(selection), encoding="utf-8")
            (run_dir / "run.json").write_bytes(run_bytes.replace(b"\r\n", b"\n"))
            manifest_file.write_bytes(manifest_bytes.replace(b"\r\n", b"\n"))
            args = Namespace(run_dir=run_dir, final_model=True, data=data, manifest=manifest_file)
            with patch("essay_scoring.__main__._predict", return_value=[8.0, 8.0]) as predictor:
                output = io.StringIO()
                with redirect_stdout(output):
                    evaluate(args)
            self.assertEqual(json.loads(output.getvalue())["test_metrics"]["mse"], 0.0)
            predictor.assert_called_once()
            with redirect_stdout(io.StringIO()):
                finalize(Namespace(run_dir=run_dir, note="Selected using synthetic validation."))

    def test_test_stays_sealed_and_finalized_checkpoint_is_frozen(self):
        with tempfile.TemporaryDirectory() as directory:
            run_dir = Path(directory) / "ridge"
            run_dir.mkdir()
            (run_dir / "run.json").write_text(
                json.dumps({"essay_set": 1, "model": "ridge", "validation_metrics": {"qwk": 0.5, "mse": 1.0}}),
                encoding="utf-8",
            )
            model_file = run_dir / "model.pkl"
            model_file.write_bytes(b"checkpoint")
            with self.assertRaisesRegex(ValueError, "sealed"):
                evaluate(Namespace(run_dir=run_dir, final_model=False))
            with redirect_stdout(io.StringIO()):
                finalize(Namespace(run_dir=run_dir, note="Selected on validation"))
            self.assertTrue((run_dir.parent / "final-selection.json").exists())
            model_file.write_bytes(b"changed")
            with self.assertRaisesRegex(ValueError, "changed after finalization"):
                evaluate(Namespace(run_dir=run_dir, final_model=True))


if __name__ == "__main__":
    unittest.main()
