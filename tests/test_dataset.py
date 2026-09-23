from trustlaya.dataset import make_rows

def test_group_split_and_count():
    splits=make_rows(10000);assert sum(map(len,splits.values()))==10000
    groups=[{r['family'] for r in rows} for rows in splits.values()]
    assert not(groups[0]&groups[1] or groups[0]&groups[2] or groups[1]&groups[2])
