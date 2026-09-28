"""Fetch the pinned, permitted v3 training/development sources."""
import hashlib
import urllib.request
from pathlib import Path

from huggingface_hub import hf_hub_download

from train_v3_external import RAW, GANDALF_SHA256, PROMPTS_SHA256

PROMPTS_REV = "f78a1c5136fa080155d928e0d7e2b4a41ddef03e"
GANDALF_REV = "04737b65e90a6794ec227012e4a255a7def6344b"


def main():
    RAW.mkdir(parents=True, exist_ok=True)
    prompt_path = RAW / "prompts.csv"
    if not prompt_path.exists():
        url = f"https://raw.githubusercontent.com/f/prompts.chat/{PROMPTS_REV}/prompts.csv"
        urllib.request.urlretrieve(url, prompt_path)
    assert hashlib.sha256(prompt_path.read_bytes()).hexdigest() == PROMPTS_SHA256
    for split in ("train", "validation", "test"):
        filenames = {"train": "train-00000-of-00001-ded53be747ff55cd.parquet",
                     "validation": "validation-00000-of-00001-94481a2a09ff2fff.parquet",
                     "test": "test-00000-of-00001-bc92128b9288a6d1.parquet"}
        filename = f"data/{filenames[split]}"
        path = Path(hf_hub_download("Lakera/gandalf_ignore_instructions", filename,
                                    repo_type="dataset", revision=GANDALF_REV,
                                    local_dir=RAW / "gandalf"))
        assert hashlib.sha256(path.read_bytes()).hexdigest() == GANDALF_SHA256[split]
    print("Pinned v3 training/development data verified in", RAW)


if __name__ == "__main__": main()
