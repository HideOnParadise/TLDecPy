"""Provenance helpers for deterministic dataset hashing."""

from __future__ import annotations

from pathlib import Path
import hashlib


def file_sha256_hex(path: str | Path, chunk_size: int = 1 << 20) -> str:
    """Return SHA-256 hex digest for a file path."""
    file_path = Path(path)
    hasher = hashlib.sha256()
    with file_path.open("rb") as handle:
        while True:
            block = handle.read(chunk_size)
            if not block:
                break
            hasher.update(block)
    return hasher.hexdigest()

