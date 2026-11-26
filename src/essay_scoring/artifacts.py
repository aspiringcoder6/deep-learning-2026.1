"""Portable checksums for JSON experiment records and split manifests."""

import hashlib
from pathlib import Path


def text_sha256(path: Path) -> str:
    content = path.read_bytes().replace(b"\r\n", b"\n")
    return hashlib.sha256(content).hexdigest()


def matches_text_sha256(path: Path, expected: str) -> bool:
    """Also accept historical checksums written with Windows line endings."""
    original = path.read_bytes()
    normalized = original.replace(b"\r\n", b"\n")
    variants = (original, normalized, normalized.replace(b"\n", b"\r\n"))
    return any(hashlib.sha256(content).hexdigest() == expected for content in variants)
