"""Read labeled ASAP essays and build a deterministic split for one prompt."""

import csv
import hashlib
import io
import json
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
DEFAULT_SPLIT_DIR = Path("data/splits")
SPLIT_VERSION = 1


@dataclass(frozen=True)
class Essay:
    essay_id: int
    text: str
    score: int


def canonical_text(text: str) -> str:
    """Use the same exact-duplicate rule during cleaning and split checks."""
    return " ".join(text.casefold().split())


def load_asap_with_audit(path: Path, essay_set: int) -> tuple[list[Essay], str, dict]:
    if essay_set not in SCORE_RANGES:
        raise ValueError(f"Unknown essay set: {essay_set}.")
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
    source_rows = 0
    for row in rows:
        try:
            if int(row["essay_set"]) != essay_set:
                continue
            source_rows += 1
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
    blank_ids = []
    duplicate_ids = []
    conflicting_score_ids = []
    for essay in sorted(essays.values(), key=lambda essay: essay.essay_id):
        key = canonical_text(essay.text)
        if not key:
            blank_ids.append(essay.essay_id)
        elif key in unique_texts:
            duplicate_ids.append(essay.essay_id)
            if essay.score != unique_texts[key].score:
                conflicting_score_ids.append(essay.essay_id)
        else:
            unique_texts[key] = essay

    audit = {
        "source_rows": source_rows,
        "kept_rows": len(unique_texts),
        "blank_essay_ids": blank_ids,
        "duplicate_text_essay_ids": duplicate_ids,
        "conflicting_score_essay_ids": conflicting_score_ids,
    }
    return list(unique_texts.values()), hashlib.sha256(raw).hexdigest(), audit


def load_asap(path: Path, essay_set: int) -> tuple[list[Essay], str]:
    essays, fingerprint, _ = load_asap_with_audit(path, essay_set)
    return essays, fingerprint


def manifest_path(essay_set: int) -> Path:
    return DEFAULT_SPLIT_DIR / f"set-{essay_set}-v{SPLIT_VERSION}.json"


def create_manifest(path: Path, essay_set: int, seed: int) -> tuple[dict, list[Essay]]:
    essays, fingerprint, audit = load_asap_with_audit(path, essay_set)
    split = split_essays(essays, seed)
    manifest = {
        "version": SPLIT_VERSION,
        "essay_set": essay_set,
        "seed": seed,
        "dataset_sha256": fingerprint,
        "cleaning": "strip; discard blank; keep lowest essay_id for casefolded, whitespace-collapsed duplicate text",
        "audit": audit,
        "splits": {name: [essay.essay_id for essay in group] for name, group in split.items()},
    }
    validate_manifest(manifest, essays, fingerprint)
    return manifest, essays


def validate_manifest(manifest: dict, essays: list[Essay], fingerprint: str) -> dict[str, list[Essay]]:
    if manifest.get("version") != SPLIT_VERSION:
        raise ValueError("Unsupported split manifest version.")
    if manifest.get("dataset_sha256") != fingerprint:
        raise ValueError("Split manifest does not match the dataset SHA-256.")
    if isinstance(manifest.get("seed"), bool) or not isinstance(manifest.get("seed"), int):
        raise ValueError("Split manifest must have an integer seed.")
    by_id = {essay.essay_id: essay for essay in essays}
    ids = manifest.get("splits", {})
    if set(ids) != {"train", "validation", "test"} or any(not ids[name] for name in ids):
        raise ValueError("Manifest must contain nonempty train, validation, and test splits.")
    all_ids = [essay_id for name in ("train", "validation", "test") for essay_id in ids[name]]
    if len(all_ids) != len(set(all_ids)) or set(all_ids) != set(by_id):
        raise ValueError("Manifest has overlapping, missing, or unknown essay IDs.")
    expected = split_essays(essays, manifest["seed"])
    if any(ids[name] != [essay.essay_id for essay in expected[name]] for name in expected):
        raise ValueError("Split IDs do not match the recorded seed and split version.")
    split = {name: [by_id[essay_id] for essay_id in ids[name]] for name in ids}
    text_sets = {name: {canonical_text(essay.text) for essay in group} for name, group in split.items()}
    if any(text_sets[a] & text_sets[b] for a, b in (("train", "validation"), ("train", "test"), ("validation", "test"))):
        raise ValueError("Duplicate essay text crosses split boundaries.")
    return split


def read_manifest(path: Path, essays: list[Essay], fingerprint: str, essay_set: int) -> dict[str, list[Essay]]:
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest.get("essay_set") != essay_set:
        raise ValueError("Split manifest belongs to another essay set.")
    return validate_manifest(manifest, essays, fingerprint)


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
