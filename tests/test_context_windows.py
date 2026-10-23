from trustlaya.context_windows import read_windows


def test_windows_cover_end_and_stay_bounded():
    original = list(range(251))
    windows = read_windows(original)
    assert all(len(w) <= 94 for parts in windows.values() for w in parts)
    assert windows["HEAD"][0] == original[:94]
    assert windows["TAIL"][0] == original[-94:]
    assert windows["SLIDING"][-1] == original[-94:]
    assert windows["HEAD_TAIL"][0] == original[:47] + original[-47:]


def test_short_input_and_invalid_geometry():
    assert read_windows([1, 2, 3])["SLIDING"] == [[1, 2, 3]]
    import pytest
    with pytest.raises(ValueError): read_windows([1], capacity=94, overlap=94)


import pytest
from pathlib import Path


@pytest.mark.skipif(not (Path(__file__).resolve().parents[1] / "models/trustlaya-s-v2/tokenizer.json").exists(),
                    reason="V2 model files are local-only")
def test_repository_tokenizer_low_level_truncation_is_explicitly_disabled():
    from pathlib import Path
    from transformers import AutoTokenizer
    root = Path(__file__).resolve().parents[1]
    tok = AutoTokenizer.from_pretrained(root / "models/trustlaya-s-v2")
    tok.backend_tokenizer.no_truncation()
    tok.backend_tokenizer.no_padding()
    assert len(tok.backend_tokenizer.encode("hi", add_special_tokens=False).ids) == 1
    assert len(tok.backend_tokenizer.encode("hello " * 1000, add_special_tokens=False).ids) > 512
