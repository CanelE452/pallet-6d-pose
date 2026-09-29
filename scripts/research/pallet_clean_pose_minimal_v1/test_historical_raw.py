import pytest
from . import historical_raw as H


def test_frozen_name_preserved_not_error_minimum():
    row = dict(current={'available': False}, hypotheses=[dict(name='a', pose={'error': 99}), dict(name='b', pose={'error': 0})])
    assert H.candidate_by_name(row, 'a') == {'error': 99}
    assert H.candidate_by_name(row, None) == {'available': False}
    row['hypotheses'].append(dict(name='a', pose={}))
    with pytest.raises(AssertionError):
        H.candidate_by_name(row, 'a')


def test_new_separate_outputs_and_three_selectors():
    assert H.SELECTORS == ('D9', 'GEO', 'NEWGEO')
    assert H.PRIVATE.is_relative_to(H.C.RAW) and H.RESULT.is_relative_to(H.C.DOC)
    assert not H.PRIVATE.is_relative_to(H.C.OLD.RAW)
