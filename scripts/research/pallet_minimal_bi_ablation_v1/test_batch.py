"""Batch boundary checks: fixed score-free pilot and retained no-pose rows."""
from copy import deepcopy
import hashlib
from scripts.research.pallet_minimal_bi_ablation_v1 import batch


def test_pilot_is_balanced_deterministic_and_ignores_nonCAL_fields():
    frames=[dict(id=f'{s}:{i}',partition='CAL',session=s,irrelevant_score=i)
        for s in ('a','b','c','d') for i in range(5)]
    snapshot=deepcopy(frames)
    ids=batch.pilot_ids(frames)
    assert ids==batch.pilot_ids(list(reversed(frames)))
    assert frames==snapshot and len(ids)==6
    assert [x.split(':')[0] for x in ids]==list('abcdab')
    for s in 'abcd':
        expected=min((f['id'] for f in frames if f['session']==s),key=lambda x:hashlib.sha256(x.encode()).hexdigest())
        assert next(x for x in ids if x.startswith(s+':'))==expected


def test_baseline_unavailable_preserves_failure_with_NA_numerical_acceptance():
    source=dict(coarse_pose=dict(available=False,reason='insufficient_original_detections'))
    snapshot=deepcopy(source)
    row=batch._baseline(source)
    assert row['actual_pose']['available'] is False
    assert row['status']=='no_initial_pose'
    assert row['accepted'] is None and row['fallback'] is None
    assert source==snapshot


def test_active_stage_is_in_resource_cap(tmp_path):
    ledger=batch.Budget(tmp_path)
    ledger.record('old','gpu_process',3599.)
    ledger.check('gpu_process',.5)
    import pytest
    with pytest.raises(RuntimeError,match='RESOURCE_CAP'):
        ledger.check('gpu_process',1.1)


def test_parity_requires_exact_discrete_branch():
    import pytest
    with pytest.raises(AssertionError):
        batch.compare(dict(accepted=True),dict(accepted=False))
    with pytest.raises(AssertionError):
        batch.compare(dict(states=['B','S']),dict(states=['S','B']))
