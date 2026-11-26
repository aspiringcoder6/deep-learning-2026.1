import copy
import csv
import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from essay_scoring.data import (
    create_manifest, load_asap_with_audit, load_training_split, read_manifest, split_essays,
)


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

    def test_training_rejects_incomplete_or_unknown_manifest_ids(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "tiny.tsv"
            with path.open("w", newline="", encoding="utf-8") as file:
                writer = csv.writer(file, delimiter="\t")
                writer.writerow(["essay_id", "essay_set", "essay", "domain1_score"])
                for index in range(20):
                    writer.writerow([2 * index + 2, 1, f"Essay number {index}.", 8])
            manifest, essays = create_manifest(path, 1, 42)
            incomplete = copy.deepcopy(manifest)
            incomplete["splits"] = {
                name: [essay.essay_id for essay in group]
                for name, group in split_essays(essays[:10], 42).items()
            }
            unknown = copy.deepcopy(manifest)
            unknown["splits"]["test"][0] += 1
            manifest_file = Path(directory) / "split.json"
            for invalid in (incomplete, unknown):
                with self.subTest(splits=invalid["splits"]):
                    manifest_file.write_text(json.dumps(invalid), encoding="utf-8")
                    with self.assertRaisesRegex(ValueError, "missing or unknown"):
                        load_training_split(path, manifest_file, 1)

    def test_training_coverage_check_keeps_test_labels_sealed(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "tiny.tsv"
            rows = [[i, 1, f"Essay number {i}.", 8] for i in range(20)]
            header = ["essay_id", "essay_set", "essay", "domain1_score"]
            with path.open("w", newline="", encoding="utf-8") as file:
                writer = csv.writer(file, delimiter="\t")
                writer.writerow(header)
                writer.writerows(rows)
            manifest, _ = create_manifest(path, 1, 42)
            test_ids = set(manifest["splits"]["test"])
            for row in rows:
                if row[0] in test_ids:
                    row[2:] = ["", "SEALED"]
            with path.open("w", newline="", encoding="utf-8") as file:
                writer = csv.writer(file, delimiter="\t")
                writer.writerow(header)
                writer.writerows(rows)
            manifest["dataset_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
            manifest_file = Path(directory) / "split.json"
            manifest_file.write_text(json.dumps(manifest), encoding="utf-8")
            training, _ = load_training_split(path, manifest_file, 1)
            self.assertEqual(set(training), {"train", "validation"})
            self.assertTrue(all(essay.essay_id not in test_ids for group in training.values() for essay in group))


if __name__ == "__main__":
    unittest.main()
