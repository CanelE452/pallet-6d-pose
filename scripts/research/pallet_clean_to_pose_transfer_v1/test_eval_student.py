"""평가/후보진단의 분모·개입비교·참조분리 회귀검사."""
import inspect
import pytest

from . import eval_student as E


def metadata():
    return [dict(id=str(i),recording='REC_A' if i<64 else 'REC_B',
        severity='CLEAN' if i<29 else 'MODERATE_OCCLUSION' if i<50 else 'SEVERE_OCCLUSION',
        image=dict(path=f'image{i}.png',sha256='unused'),K=[[1.,0.,0.],[0.,1.,0.],[0.,0.,1.]],
        xyz=[1.1,.11,1.3],hw=[100,100]) for i in range(128)]


def test_fixed_population_denominators_and_natural_pool():
    rows=metadata()
    groups=E.group_ids(rows)
    assert [len(groups[k]) for k in ('FULL128','CLEAN29','MOD21','SEV78','NATURAL99')] == [128,29,21,78,99]
    assert set(groups['NATURAL99']) == set(groups['MOD21']) | set(groups['SEV78'])


def test_reference_fields_rejected_before_inference():
    rows=metadata()
    rows[0]['GT_keypoints']=[[1.,2.]]
    with pytest.raises(AssertionError):
        E.validate_membership(rows)


def test_duplicate_or_wrong_membership_fails_closed():
    rows=metadata()
    rows[1]['id']=rows[0]['id']
    with pytest.raises(AssertionError):
        E.validate_membership(rows)
    rows=metadata()
    rows[0]['severity']='MODERATE_OCCLUSION'
    with pytest.raises(AssertionError):
        E.validate_membership(rows)


def test_six_causal_contrasts_preserve_both_factors_and_baselines():
    assert E.PAIRS == (
        ('CLEAN_REF_CLEAR','CLEAN_REF_OCC'),('CLEAN_RAW_CLEAR','CLEAN_RAW_OCC'),
        ('CLEAN_RAW_CLEAR','CLEAN_REF_CLEAR'),('CLEAN_RAW_OCC','CLEAN_REF_OCC'),
        ('OLD_REF','CLEAN_REF_OCC'),('R0','CLEAN_REF_OCC'))


def test_oracle_does_not_combine_two_different_candidates():
    metric=lambda t,r:dict(available=True,translation_cm=t,rotation_deg=r)
    baseline={'a':metric(10.,10.)}
    choices={'a':[dict(name='Tbest',metric=metric(5.,20.)),dict(name='Rbest',metric=metric(20.,5.))]}
    assert E.same_candidate_joint_count(['a'],choices,baseline)==0
    choices['a'].append(dict(name='both',metric=metric(9.,9.)))
    assert E.same_candidate_joint_count(['a'],choices,baseline)==1


def test_joint_gain_does_not_accept_coverage_loss_or_one_axis_gain():
    row=lambda t,r,n=99:dict(conditional={'translation_cm':{'median':t},'rotation_deg':{'median':r}},valid_pose=n,frames=99)
    rows={arm:row(10.,10.) for arm in (*E.ARMS,*E.BASELINES)}
    rows['CLEAN_REF_OCC']=row(9.,9.,98)
    result=E.classify({'NATURAL99':rows})
    assert result['versus_R0']['eligible_joint'] is False
    rows['CLEAN_REF_OCC']=row(9.,11.)
    result=E.classify({'NATURAL99':rows})
    assert result['versus_R0']['label']=='TRADEOFF'


def test_reference_open_is_in_score_only_not_infer():
    source=inspect.getsource(E.infer)
    assert "metadata('REAL_DEV')" not in source
    assert 'O.P.TRUTH' not in source
    source=inspect.getsource(E.score)
    assert source.index('prediction_lock(seed)') < source.index('C.read(O.P.TRUTH)')
    source=inspect.getsource(E.oracle_score)
    assert source.index("C.read(p['candidate_lock'])") < source.index("metadata('REAL_DEV')")


def test_paths_are_new_namespace_only():
    p=E.paths(42)
    assert all(path.is_relative_to(E.C.RAW) or path.is_relative_to(E.C.DOC) for path in p.values())
    with pytest.raises(AssertionError):
        E.paths(-1)
