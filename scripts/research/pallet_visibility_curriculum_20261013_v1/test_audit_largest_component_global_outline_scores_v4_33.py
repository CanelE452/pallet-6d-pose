from copy import deepcopy

import pytest

from . import audit_largest_component_global_outline_scores_v4_33 as V


def test_GO_and_both_SHA_before_IO(monkeypatch):
    def forbidden(*args,**kwargs):raise AssertionError('IO beforeGO/SHA')
    monkeypatch.setattr(V.A,'sha',forbidden)
    with pytest.raises(PermissionError):V.run()
    with pytest.raises(ValueError,match='before IO'):V.run(root_score_review_GO=True,expected_score_seal_sha256='a'*64)


def rows():
    return [dict(id='SYNTHETIC',metrics={a:dict(translation_cm=5.,rotation_deg=5.,pose_available=True) for a in V.ARMS},
        accepted={a:False for a in V.ARMS},pose_available={a:True for a in V.ARMS})]


def test_native_e_entire_metrics_and_status_invariance_only():
    old=rows();new=deepcopy(old)
    new[0]['metrics'][V.ARMS[2]]['translation_cm']=9.
    q=V.A.Checks();V.compare_invariants(old,new,q);assert not q.failures
    new[0]['metrics'][V.ARMS[1]]['translation_cm']=5.01
    q=V.A.Checks();V.compare_invariants(old,new,q);assert q.failures


def test_paired_threshold_losses_gains_and_missing_retained():
    old=rows();new=deepcopy(old);arm=V.ARMS[2]
    new[0]['metrics'][arm].update(translation_cm=None,rotation_deg=None,pose_available=False)
    q=V.paired_C_changes(old,new,arm)
    assert len(q['translation_threshold_losses'])==1 and not q['translation_threshold_gains']
    q=V.paired_C_changes(new,old,arm)
    assert len(q['translation_threshold_gains'])==1 and not q['translation_threshold_losses']
    with pytest.raises(ValueError):V.paired_C_changes(old,[],arm)
