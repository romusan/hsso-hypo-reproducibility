#!/usr/bin/env python
"""Verify SHA-256 checksums for the canonical code and principal inputs."""
from __future__ import annotations

import hashlib
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TEXT_SUFFIXES = {
    ".conf", ".csv", ".inp", ".json", ".md", ".py", ".tex", ".txt", ".yaml", ".yml"
}


def canonical_bytes(path: Path) -> bytes:
    """Return platform-independent bytes for checksum comparison."""
    data = path.read_bytes()
    if path.suffix.lower() in TEXT_SUFFIXES:
        data = data.replace(b"\r\n", b"\n")
    return data


def main() -> None:
    failures = []
    for line in (ROOT / "MANIFEST.sha256").read_text(encoding="utf-8").splitlines():
        expected, relative = line.split(maxsplit=1)
        path = ROOT / relative
        observed = hashlib.sha256(canonical_bytes(path)).hexdigest()
        status = "OK" if observed == expected else "FAIL"
        print(f"{status}  {relative}")
        if status == "FAIL":
            failures.append(relative)
    if failures:
        raise SystemExit(f"Checksum mismatch: {', '.join(failures)}")
    print("manifest verification: PASS")


if __name__ == "__main__":
    main()
