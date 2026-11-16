import io
import json
import tempfile
import unittest
from argparse import Namespace
from contextlib import redirect_stdout
from pathlib import Path

from essay_scoring.__main__ import evaluate, finalize


class TestHoldoutGate(unittest.TestCase):
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
