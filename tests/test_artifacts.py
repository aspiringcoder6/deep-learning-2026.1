import hashlib
import tempfile
import unittest
from pathlib import Path

from essay_scoring.artifacts import matches_text_sha256, text_sha256


class ArtifactChecksumTest(unittest.TestCase):
    def test_line_endings_preserve_new_and_legacy_checksums(self):
        lf = b'{\n  "seed": 42\n}\n'
        crlf = lf.replace(b"\n", b"\r\n")
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "run.json"
            for content in (lf, crlf):
                with self.subTest(content=content):
                    path.write_bytes(content)
                    self.assertEqual(text_sha256(path), hashlib.sha256(lf).hexdigest())
                    for original in (lf, crlf):
                        self.assertTrue(matches_text_sha256(path, hashlib.sha256(original).hexdigest()))
            path.write_bytes(lf.replace(b"42", b"43"))
            self.assertFalse(matches_text_sha256(path, hashlib.sha256(lf).hexdigest()))
