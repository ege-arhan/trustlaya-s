"""Fetch pinned Tensor Trust data into the git-ignored local cache and verify SHA-256.

Raw data is never committed: files land in data/external/tensor_trust/<revision>/.
Checksums live in data/v5_source_pins.json. A cached or downloaded file whose
hash differs from its pin fails the run. `--record` only fills pins that are
missing; it never replaces an existing pin.

Upstream: https://github.com/HumanCompatibleAI/tensor-trust-data (paper
arXiv:2311.01011). The data repository has no LICENSE file; the game code
repository is BSD-2-Clause. That status is reported as-is, not assumed.

Run: .venv/bin/python scripts/fetch_tensor_trust.py [--record]
"""

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PINS = ROOT / "data/v5_source_pins.json"
REPO = "HumanCompatibleAI/tensor-trust-data"
REVISION = "747a75e096761ebc01bd3970158827326b4add23"
CACHE = ROOT / "data/external/tensor_trust" / REVISION
FILES = (
    "benchmarks/hijacking-robustness/v1/hijacking_robustness_dataset.jsonl",
    "benchmarks/extraction-robustness/v1/extraction_robustness_dataset.jsonl",
    "detecting-extractions/v1/prompt_extraction_detection.jsonl",
    "raw-data/v1/raw_dump_attacks.jsonl.bz2",
    "raw-data/v1/raw_dump_defenses.jsonl.bz2",
    "raw-data/v2/raw_dump_attacks.jsonl.bz2",
    "raw-data/v2/raw_dump_defenses.jsonl.bz2",
)


class ChecksumError(RuntimeError):
    pass


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def load_pins():
    return json.loads(PINS.read_text()) if PINS.exists() else {}


def save_pins(pins):
    PINS.write_text(json.dumps(pins, indent=2, sort_keys=True) + "\n")


def fetch(url, dest, pinned):
    """Download once, reuse the cache, and fail on any mismatch with the pin."""
    dest = Path(dest)
    if not dest.exists():
        dest.parent.mkdir(parents=True, exist_ok=True)
        partial = dest.with_suffix(dest.suffix + ".partial")
        # curl, not urllib: the arXiv CDN answers Python's TLS client with 406.
        subprocess.run(["curl", "-fsSL", "--retry", "3", "-o", str(partial), url], check=True)
        partial.rename(dest)
    actual = sha256(dest)
    if pinned is not None and actual != pinned:
        raise ChecksumError(f"checksum mismatch for {dest.name}: {actual} != pinned {pinned}")
    return actual


def fetch_pinned(source, repo_url, revision, files, cache, *, record=False):
    """Fetch `files` of one source; pins are keyed by source name and path."""
    pins = load_pins()
    entry = pins.setdefault(source, {"url": repo_url, "revision": revision, "files": {}})
    if entry["revision"] != revision:
        raise ChecksumError(f"{source}: pinned revision {entry['revision']} != {revision}")
    changed = False
    for path, url in files.items():
        pinned = entry["files"].get(path)
        if pinned is None and not record:
            raise ChecksumError(f"{source}:{path} has no pin; rerun with --record once")
        actual = fetch(url, cache / path, pinned)
        if pinned is None:
            entry["files"][path] = actual
            changed = True
    if changed:
        save_pins(pins)
    return {path: cache / path for path in files}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--record", action="store_true", help="pin files that have no checksum yet")
    args = parser.parse_args()
    files = {path: f"https://raw.githubusercontent.com/{REPO}/{REVISION}/{path}" for path in FILES}
    try:
        paths = fetch_pinned("tensor_trust", f"https://github.com/{REPO}", REVISION, files,
                             CACHE, record=args.record)
    except ChecksumError as exc:
        sys.exit(f"FAIL: {exc}")
    for path in paths.values():
        print(f"ok {path.relative_to(ROOT)} {path.stat().st_size} bytes")


if __name__ == "__main__":
    main()
