"""Copy verified v2 row predictions into immutable v3 comparison inputs.

Run `scripts/run_external_real.py` first. This never rewrites v2 weights.
"""
import hashlib
import json
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LOCAL = ROOT / "benchmarks/external/predictions"
EXPECTED_MODEL = "99a8527de00fed3a520d136d26cdda9acc79dff2fae5c725ef773159b565563c"


def main():
    model = ROOT / "models/trustlaya-s-v2/model.safetensors"
    assert hashlib.sha256(model.read_bytes()).hexdigest() == EXPECTED_MODEL, "Not the frozen v2 model"
    for source, target, count in (("tab_predictions.json", "v2_frozen_tab_predictions.json", 2079),
                                  ("jailbreak_predictions.json", "v2_frozen_jailbreak_predictions.json", 5888)):
        rows = json.loads((LOCAL / source).read_text())
        assert len(rows) == count and len({r["sample_id"] for r in rows}) == count
        assert all(r["model_version"] == "TrustLaya-S v2 frozen" for r in rows)
        destination = LOCAL / target
        if destination.exists():
            assert destination.read_bytes() == (LOCAL / source).read_bytes(), f"Existing {target} differs; do not overwrite"
        else:
            shutil.copyfile(LOCAL / source, destination)
        print(target, hashlib.sha256(destination.read_bytes()).hexdigest())


if __name__ == "__main__": main()
