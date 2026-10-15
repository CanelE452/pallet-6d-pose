"""Small synthetic independent accounting contracts; no actual score IO."""
from copy import deepcopy
import pytest

from . import audit_clean_fresh_BI_scores_v4_32 as A


def test_root_GO_precedes_IO(monkeypatch):
    def forbidden(*args,**kwargs):raise AssertionError('No IO before explicit root GO')
    monkeypatch.setattr(A,'sha',forbidden);monkeypatch.setattr(A,'read',forbidden)
    with pytest.raises(PermissionError):A.run(expected_score_seal_sha256='a'*64)


def test_missing_whole_denominator_moments_are_not_finite_subset_moments():
    q=A.moments([None,2.,4.])
    assert q['count']==3 and q['missing_or_undefined_count']==1
    assert q['mean']is None and q['max']is None
    assert q['finite_only_diagnostic']==dict(count=2,mean=3.,max=4.)
    with pytest.raises(ValueError):A.moments([float('inf')])


def test_inclusive_boundary_fallback_and_joint_counts():
    arm=A.ARMS[0]
    summary=dict(denominator=1,translation_le5=1,successes_5cm_5deg=1,missing_pose_count=0,
        rotation_unresolved=0,fallback_count=1,unaccepted_count=1,max_translation_id='TOY',
        translation_cm=dict(mean=5.,max=5.),rotation_deg=dict(mean=5.,max=5.))
    row=dict(id='TOY',metrics={arm:dict(translation_cm=5.,rotation_deg=5.,pose_available=True,
        translation_le5=True,success_5cm_5deg=True)},accepted={arm:False},pose_available={arm:True})
    checks=A.Checks();result=A.aggregate(dict(count=1,rows=[row],summary={arm:summary}),[arm],checks)[arm]
    assert not checks.failures
    assert result['translation_le5']==result['rotation_le5']==result['successes_5cm_5deg']==1
    assert result['fallback_count']==1


def test_exact_e_metric_and_legacy_boolean_scope():
    arm=A.ARMS[0];old=dict(id='TOY',accepted={arm:False},metrics={arm:dict(
        translation_cm=5.,rotation_deg=5.,ADDsym_cm=1.,success_5cm_5deg=False)})
    new=deepcopy(old);new['metrics'][arm].update(success_5cm_5deg=True,legacy_strict_success_5cm_5deg=False)
    checks=A.Checks();A.compare_e([old],[new],checks);assert not checks.failures
    new['metrics'][arm]['translation_cm']+=.01
    checks=A.Checks();A.compare_e([old],[new],checks)
    assert checks.failures==['V30 e exact metric TOY/translation_cm']


def test_threshold_regressions_and_any_increase_are_separate():
    e,a=A.ARMS[:2];rows=[]
    for i,t in enumerate((6.,4.5,None)):
        rows.append(dict(id=str(i),metrics={e:dict(translation_cm=4.,rotation_deg=2.,success_5cm_5deg=True),
            a:dict(translation_cm=t,rotation_deg=2. if t is not None else None,
                   success_5cm_5deg=t is not None and t<=5.,pose_available=t is not None)},accepted={a:t is not None}))
    q=A.regressions(rows,a)
    assert [r['id']for r in q['translation_threshold_regressions']]==['0','2']
    assert [r['id']for r in q['translation_error_increases']]==['0','1','2']
    assert [r['id']for r in A.largest_errors(rows,a)]==['2','0','1']
