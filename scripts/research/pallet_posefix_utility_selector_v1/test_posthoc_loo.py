"""Sequential selection cannot revive rejected corrections or move the centre."""
from unittest.mock import patch
import numpy as np
from . import posthoc_loo as L


def run(mask, evidence, residuals):
    n2 = np.arange(18, dtype=float).reshape(9, 2)
    selected = n2 + 20
    selected[8] = n2[8]
    geometry = dict(chosen_hypothesis='fixed', projected_diagonal_px=100.)
    with patch.object(L.G, 'geometry_details', side_effect=[geometry]+[
        dict(per_corner_remove=r) for r in residuals]), patch.object(
            L.G, 'decide_pairs', return_value=dict(pair_accept=evidence)):
        out, diag = L.filter_selected(n2, selected, mask, np.ones(9, bool),
                                     np.eye(3), {}, 1., np.ones(9))
    return n2, selected, out, diag


def test_cannot_revive_learned_rejected_pairs():
    n2, selected, out, diag = run([True,False,False,False], [True]*4, [[.01]*8])
    assert diag['pair_accept'] == [True,False,False,False]
    assert np.array_equal(out[[0,3]], selected[[0,3]])
    assert np.array_equal(out[[1,2,4,5,6,7,8]], n2[[1,2,4,5,6,7,8]])


def test_one_bad_endpoint_rejects_both():
    residual = [.01]*8
    residual[0] = .051
    n2, _, out, diag = run([True,False,False,False], [True]*4, [residual])
    assert diag['pair_accept'] == [False]*4
    assert np.array_equal(out, n2)


def test_initial_loo_rejection_and_empty_selection():
    for mask in ([False]*4, [True]*4):
        n2, _, out, diag = run(mask, [False]*4, [])
        assert not any(diag['pair_accept'])
        assert np.array_equal(out, n2)


def test_mixed_structure_rechecked_until_stable():
    first = [.01]*8
    first[0] = .06
    second = [.01]*8
    second[1] = float('inf')
    n2, _, out, diag = run([True,True,False,False], [True]*4, [first,second])
    assert len(diag['recheck_history']) == 2
    assert not any(diag['pair_accept'])
    assert np.array_equal(out, n2)
