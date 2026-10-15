"""Behavioral tests of the predeclared CAL-only bounded candidate selector."""
from copy import deepcopy

import pytest

from . import score as S


def rows():
    result=[]
    for i in range(46):
        metrics={}
        statuses={}
        for arm in S.ARMS:
            success=i < 10
            metrics[arm]=dict(pose_available=True, translation_cm=2. if success else 8.,
                rotation_deg=1., ADDsym_cm=2., success_le5=success,
                success_strict_lt5=success, reference_rotation_available=True)
            statuses[arm]=dict(accepted=None if arm.endswith('0') else True,
                fallback=None if arm.endswith('0') else False, status='fixture', reason=None)
        result.append(dict(id='frame:'+str(i), partition='CAL', session='session'+str(i%4),
            cluster_id='recording'+str(i%4), metrics=metrics, prediction_status=statuses))
    return result


def success_at(values, arm, ids, *, mean_error=None):
    for i,r in enumerate(values):
        ok=i in ids
        m=r['metrics'][arm]
        m.update(success_le5=ok, success_strict_lt5=ok,
            translation_cm=(2. if ok else 8.) if mean_error is None else mean_error)


def test_unconstrained_1_is_never_a_selection_candidate():
    values=rows()
    for arm in ('R1','N1','E1'):
        success_at(values,arm,set(range(46)))
    result=S.selection(values)
    assert result['per_initialization']==dict(R='R0',N='N0',E='E0')
    assert result['selected_arm']=='E0'
    assert all(not v['eligible'] for v in result['eligibility'].values())


def test_net_gain_with_any_own_baseline_damage_is_ineligible():
    values=rows()
    success_at(values,'RA',set(range(1,20)))
    success_at(values,'RB',set(range(11)))
    result=S.selection(values)
    assert result['eligibility']['RA']==dict(eligible=False,recovery=10,damage=1)
    assert result['eligibility']['RB']==dict(eligible=True,recovery=1,damage=0)
    assert result['per_initialization']['R']=='RB'
    assert result['selected_arm']=='RB'


def test_preserving_success_without_recovery_keeps_own_baseline():
    values=rows()
    for r in values:
        r['metrics']['EA']['translation_cm']=.01
        r['metrics']['EB']['rotation_deg']=.01
    result=S.selection(values)
    assert result['per_initialization']['E']=='E0'
    assert result['eligibility']['EA']==dict(eligible=False,recovery=0,damage=0)


def test_local_success_count_precedes_continuous_error():
    values=rows()
    success_at(values,'NA',set(range(12)),mean_error=4.)
    success_at(values,'NB',set(range(11)),mean_error=.1)
    assert S.selection(values)['per_initialization']['N']=='NA'


def test_local_continuous_tie_break_then_simpler_A():
    values=rows()
    success_at(values,'NA',set(range(11)),mean_error=3.)
    success_at(values,'NB',set(range(11)),mean_error=2.)
    assert S.selection(values)['per_initialization']['N']=='NB'
    for r in values:
        r['metrics']['NB']['translation_cm']=3.
        r['metrics']['NA']['rotation_deg']=.5
    assert S.selection(values)['per_initialization']['N']=='NA'
    for r in values:
        r['metrics']['NB']['rotation_deg']=.5
    assert S.selection(values)['per_initialization']['N']=='NA'


def test_global_ties_use_fixed_E_N_RAW_priority_not_frame_identity():
    values=rows()
    for arm in ('RA','NA','EA'):
        success_at(values,arm,set(range(11)))
    before=deepcopy(values)
    first=S.selection(values)
    assert first['selected_arm']=='EA'
    assert first['per_initialization']==dict(R='RA',N='NA',E='EA')
    for i,r in enumerate(values):
        r['id']='unrelated:'+str(1000-i)
        r['session']='different-session'+str(i%4)
        r['cluster_id']='different-recording'+str(i%4)
    assert S.selection(list(reversed(values)))==first
    assert before[0]['metrics']==values[0]['metrics']


def test_global_success_count_precedes_fixed_initialization_priority():
    values=rows()
    success_at(values,'RA',set(range(12)))
    success_at(values,'EA',set(range(11)))
    assert S.selection(values)['selected_arm']=='RA'


def test_no_pose_keeps_full_denominator_and_counts_as_failure():
    values=rows()
    m=values[0]['metrics']['RA']
    m.update(pose_available=False,translation_cm=None,rotation_deg=None,ADDsym_cm=None,
        success_le5=False,success_strict_lt5=False)
    success_at(values[10:11],'RA',{0})
    result=S.selection(values)
    summary=S.summarize(values)['arms']['RA']
    assert summary['denominator']==46 and summary['no_pose_n']==1
    assert summary['errors']['translation_cm']['n']==45
    assert result['eligibility']['RA']==dict(eligible=False,recovery=1,damage=1)


@pytest.mark.parametrize('change', ['primary','missing','duplicate'])
def test_selection_rejects_non_CAL46_and_duplicate_id_input(change):
    values=rows()
    if change=='primary':
        values[0]['partition']='TEST'
    elif change=='missing':
        values.pop()
    else:
        values[1]['id']=values[0]['id']
    with pytest.raises(ValueError):
        S.selection(values)
