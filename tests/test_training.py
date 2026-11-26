import csv
import io
import json
import tempfile
import unittest
from argparse import Namespace
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

from essay_scoring.__main__ import _predict, train
from essay_scoring.data import create_manifest


class TrainingCommandTest(unittest.TestCase):
    def test_reload_accepts_roundoff_and_rejects_changed_predictions(self):
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
            manifest_file.write_text(json.dumps(manifest), encoding="utf-8")
            for delta, succeeds in ((1e-7, True), (1.0, False)):
                with self.subTest(delta=delta):
                    run_dir = root / f"ridge-{delta}"
                    args = Namespace(
                        data=data, manifest=manifest_file, essay_set=1, seed=42,
                        model="ridge", run_dir=run_dir,
                    )
                    def shifted_predictions(*arguments):
                        return [score + delta for score in _predict(*arguments)]

                    with patch("essay_scoring.__main__._predict", side_effect=shifted_predictions):
                        with redirect_stdout(io.StringIO()):
                            if succeeds:
                                train(args)
                            else:
                                with self.assertRaisesRegex(ValueError, "changed validation predictions"):
                                    train(args)
                    self.assertEqual(run_dir.exists(), succeeds)
                    if succeeds:
                        record = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
                        self.assertGreater(record["validation_metrics"]["mse"], 0.0)

    def test_prompt_eight_with_batch_eight_saves_a_reloadable_run(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data = root / "essays.tsv"
            with data.open("w", newline="", encoding="utf-8") as file:
                writer = csv.writer(file, delimiter="\t")
                writer.writerow(["essay_id", "essay_set", "essay", "domain1_score"])
                for index in range(1280):
                    words = " ".join(f"word{j % 17}" for j in range(8 + index % 40))
                    text = f"A varied essay {index}. {words}."
                    writer.writerow([index, 8, text, 60 if index % 2 else 0])
            manifest, _ = create_manifest(data, 8, 0)
            manifest_file = root / "split.json"
            manifest_file.write_text(json.dumps(manifest), encoding="utf-8")
            run_dir = root / "han"
            args = Namespace(
                data=data, manifest=manifest_file, essay_set=8, seed=0, model="han",
                epochs=1, batch_size=8, run_dir=run_dir,
            )
            with redirect_stdout(io.StringIO()):
                train(args)
            record = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
            self.assertEqual(record["model_settings"]["batch_size"], 8)
            verification = record["checkpoint_verification"]
            self.assertEqual(record["validation_metrics"], verification["reloaded_validation_metrics"])
            self.assertLessEqual(verification["max_absolute_reload_difference"], 1e-6)
            prediction, = _predict(run_dir, record, ["A fresh essay. It has another sentence."])
            self.assertTrue(0 <= prediction <= 60)
