"""Pure token-window construction for the isolated v4 research experiment."""
from __future__ import annotations


def read_windows(token_ids: list[int], capacity: int = 94, overlap: int = 47) -> dict[str, list[list[int]]]:
    if capacity < 2 or not 0 <= overlap < capacity:
        raise ValueError("invalid window geometry")
    if not token_ids:
        token_ids = []
    head = token_ids[:capacity]
    tail = token_ids[-capacity:]
    left = capacity // 2
    head_tail = token_ids[:left] + token_ids[-(capacity - left):] if len(token_ids) > capacity else head
    starts = list(range(0, max(len(token_ids) - capacity + 1, 1), capacity - overlap))
    if starts[-1] != max(len(token_ids) - capacity, 0):
        starts.append(max(len(token_ids) - capacity, 0))
    sliding = [token_ids[i:i + capacity] for i in starts]
    return {"HEAD": [head], "TAIL": [tail], "HEAD_TAIL": [head_tail],
            "SLIDING": sliding, "WINDOW_MAX": sliding, "WINDOW_LOGIT_POOL": sliding}
