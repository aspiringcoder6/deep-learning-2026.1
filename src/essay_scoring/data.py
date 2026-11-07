"""Read labeled ASAP essays and build a deterministic split for one prompt."""

import csv
import hashlib
import io
import random
from dataclasses import dataclass
from pathlib import Path

# Domain 1 rubric bounds for ASAP-AES (Xie et al., COLING 2022, Table 1).
SCORE_RANGES = {
    1: (2, 12),
    2: (1, 6),
    3: (0, 3),
    4: (0, 3),
    5: (0, 4),
    6: (0, 4),
    7: (0, 30),
    8: (0, 60),
}
DEFAULT_DATA_PATH = Path("data/raw/training_set_rel3.tsv")
SPLIT_VERSION = 1


@dataclass(frozen=True)
class Essay:
    essay_id: int
    text: str
    score: int


def load_asap(path: Path, essay_set: int) -> tuple[list[Essay], str]:
    raw = path.read_bytes()
    try:
        text = raw.decode("utf-8-sig")
    except UnicodeDecodeError:
        text = raw.decode("latin-1")

    rows = csv.DictReader(io.StringIO(text, newline=""), delimiter="\t")
    required = {"essay_id", "essay_set", "essay", "domain1_score"}
    missing = required.difference(rows.fieldnames or [])
    if missing:
        raise ValueError(f"Missing ASAP columns in {path}: {', '.join(sorted(missing))}")

    low, high = SCORE_RANGES[essay_set]
    essays = {}
    for row in rows:
        try:
            if int(row["essay_set"]) != essay_set:
                continue
            essay_id = int(row["essay_id"])
            score = int(row["domain1_score"])
        except (TypeError, ValueError) as error:
            raise ValueError(f"Invalid ASAP record on line {rows.line_num} of {path}.") from error

        if essay_id in essays:
            raise ValueError(f"Duplicate essay ID {essay_id} in set {essay_set}.")
        if not low <= score <= high:
            raise ValueError(f"Essay {essay_id} has score {score} outside [{low}, {high}].")
        essays[essay_id] = Essay(essay_id, (row["essay"] or "").strip(), score)

    # Keep the lowest ID for each case- and whitespace-normalized text.
    unique_texts = {}
    for essay in sorted(essays.values(), key=lambda essay: essay.essay_id):
        key = " ".join(essay.text.casefold().split())
        if key:
            unique_texts.setdefault(key, essay)

    return list(unique_texts.values()), hashlib.sha256(raw).hexdigest()


def split_essays(essays: list[Essay], seed: int) -> dict[str, list[Essay]]:
    if len(essays) < 3:
        raise ValueError("At least three distinct, nonempty essays are needed to create a split.")
    shuffled = list(essays)
    random.Random(seed).shuffle(shuffled)
    holdout_size = max(1, len(shuffled) // 10)
    return {
        "train": shuffled[:-2 * holdout_size],
        "validation": shuffled[-2 * holdout_size:-holdout_size],
        "test": shuffled[-holdout_size:],
    }
