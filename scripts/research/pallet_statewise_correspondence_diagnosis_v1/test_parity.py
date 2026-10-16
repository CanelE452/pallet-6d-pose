from copy import deepcopy
from .parity import compare, _compact_outcomes


def test_tolerance_is_absolute_fixed_and_exact_flags_preserved():
    assert compare({'value':1.,'accepted':True}, {'value':1.+5e-10,'accepted':True}) == []
    assert compare({'value':1.}, {'value':1.+2e-9})
    assert compare(True, 1)
    assert compare(None, False)
    assert compare([1.,2.],[1.,2.,3.])


def test_journal_compaction_preserves_all_candidates_and_scalar_values():
    original=[dict(initialization='a',objective=3.,visible_measurement_audit=[1,2],domain_audit=[])]
    before=deepcopy(original)
    compact=_compact_outcomes(original)
    assert compact == [dict(initialization='a',objective=3.,visible_measurement_audit_count=2,domain_audit_count=0)]
    assert original==before
