import pytest
from .elevation_review import export_records


def test_low_tags_never_automatically_exclude():
    rows=[dict(id='a',image={'sha256':'aaa'},elevation='low'),dict(id='b',image={'sha256':'bbb'},elevation='unknown')]
    result=export_records(rows,{})
    assert all(r['decision']=='unreviewed' and r['reason'] is None for r in result)
    result=export_records(rows,{'a':'exclude'})
    assert result[0]['reason']=='proposed_extreme_low_elevation' and result[1]['decision']=='unreviewed'
    assert rows[0]['elevation']=='low' and 'decision' not in rows[0]


def test_unknown_ids_and_decisions_rejected():
    rows=[dict(id='a',image={'sha256':'aaa'})]
    with pytest.raises(ValueError):export_records(rows,{'other':'exclude'})
    with pytest.raises(ValueError):export_records(rows,{'a':'delete'})
