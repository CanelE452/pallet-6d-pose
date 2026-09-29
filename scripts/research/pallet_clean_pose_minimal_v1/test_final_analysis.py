import pytest
from .final_analysis import merge_equal, pairs_for


def test_equal_merge_rejects_historical_change():
    out = {'a': {'value': 1}}
    merge_equal(out, {'a': {'value': 1}, 'b': 2})
    with pytest.raises(AssertionError):
        merge_equal(out, {'a': {'value': 9}})


def test_full_pair_contract_without_cross_seed_scorer_mix():
    prefixes = ['R0', 'OLD_RAW', 'OLD_REF'] + [f'{t}_{c}_S{s}' for t in ('RAW', 'REF') for c in ('CLEAR', 'OCC') for s in (42, 43)]
    names = [p + '_' + selector for p in prefixes for selector in ('D9', 'GEO', 'NEWGEO')]
    pairs = pairs_for(names)
    assert ('OLD_RAW_GEO', 'OLD_REF_GEO') in pairs
    assert ('RAW_CLEAR_S43_NEWGEO', 'REF_CLEAR_S43_NEWGEO') in pairs
    assert ('REF_CLEAR_S43_GEO', 'REF_OCC_S43_GEO') in pairs
    assert ('REF_OCC_S42_NEWGEO', 'REF_OCC_S43_GEO') not in pairs
