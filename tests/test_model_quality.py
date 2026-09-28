"""Real-V2 behavior that must stay visible. A change here needs a report update."""

import json
from pathlib import Path

import pytest

from trustlaya.inference import ROOT, Analyzer

V2 = ROOT / "models/exported/v2/trustlaya_s.onnx"


@pytest.mark.skipif(not V2.exists(), reason="V2 ONNX not downloaded")
def test_long_benign_note_block_is_still_a_known_failure():
    fixtures = json.loads((ROOT / "data/firewall_fixtures.json").read_text())["fixtures"]
    note = next(f for f in fixtures if f["id"] == "model_failure_long_benign_note")
    analyzer = Analyzer("onnx", model_dir=ROOT / "models/trustlaya-s-v2", onnx_path=V2)
    result = analyzer.analyze(note["text"], {"agent": True, "database": True})
    # Known V2 failure: benign note blocked. If this ever passes, update the
    # fixture's known_failure and the reports instead of deleting the case.
    assert result["action"] == "BLOCK" and result["policy_reason"] == "prompt_injection"
    assert result["coverage"]["truncated"] is True
