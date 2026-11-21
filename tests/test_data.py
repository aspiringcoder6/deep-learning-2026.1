import json
import tempfile
import unittest
from pathlib import Path

from essay_scoring.data import create_manifest, load_asap_with_audit, load_training_split, read_manifest


class AsapDataTest(unittest.TestCase):
    def test_labeled_columns_cleaning_and_frozen_split(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "tiny.tsv"
            path.write_text(
                "essay_id\tessay_set\tessay\tdomain1_score\n"
                "1\t1\tFirst essay\t8\n"
                "2\t1\t second essay \t7\n"
                "3\t1\tSECOND  ESSAY\t7\n"
                "4\t1\tThird essay\t6\n"
                "5\t1\tFourth essay\t9\n"
                "6\t1\tFifth essay\t10\n"
                "7\t1\t  \t8\n"
                "8\t2\tOther prompt\t5\n",
                encoding="utf-8",
            )
            essays, fingerprint, audit = load_asap_with_audit(path, 1)
            self.assertEqual([essay.essay_id for essay in essays], [1, 2, 4, 5, 6])
            self.assertEqual(audit["duplicate_text_essay_ids"], [3])
            self.assertEqual(audit["blank_essay_ids"], [7])
            manifest, _ = create_manifest(path, 1, 42)
            manifest_file = Path(directory) / "split.json"
            manifest_file.write_text(json.dumps(manifest), encoding="utf-8")
            split = read_manifest(manifest_file, essays, fingerprint, 1)
            self.assertEqual(sum(map(len, split.values())), 5)
            self.assertEqual(len({essay.essay_id for group in split.values() for essay in group}), 5)
            training, same_fingerprint = load_training_split(path, manifest_file, 1)
            self.assertEqual(set(training), {"train", "validation"})
            self.assertEqual(same_fingerprint, fingerprint)
            self.assertEqual(training["train"], split["train"])

            manifest["splits"]["test"][0] = manifest["splits"]["train"][0]
            manifest_file.write_text(json.dumps(manifest), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "overlapping"):
                read_manifest(manifest_file, essays, fingerprint, 1)

    def test_scoped_loader_skips_test_label_and_text_before_validation(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "tiny.tsv"
            path.write_text("essay_id\tessay_set\tessay\tdomain1_score\n"
                            "1\t1\tTrain essay\t8\n2\t1\t\tSEALED\n", encoding="utf-8")
            essays, _, _ = load_asap_with_audit(path, 1, allowed_ids={1})
            self.assertEqual([essay.essay_id for essay in essays], [1])


if __name__ == "__main__":
    unittest.main()
