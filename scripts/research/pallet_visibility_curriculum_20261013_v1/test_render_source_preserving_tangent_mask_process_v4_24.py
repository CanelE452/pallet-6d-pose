from copy import deepcopy
import pytest
from . import render_source_preserving_tangent_mask_process_v4_24 as R


def arm(continuous):
    solution=dict(native_observation_objective=10.,regularizer_costs=[dict(kind=k,weighted_cost=v)for k,v in zip(R.EXP.P.REGULARIZER_KINDS,(2.,3.))],
        total_regularized_objective=15.,objective=21. if continuous else 15.)
    if continuous:solution.update(weighted_mask_objective=6.,global_mask_weight=1.5,global_mask_cost=4.,mask_count_in_optimizer=1,continuous_mask_gradient_active=True)
    return dict(accepted=True,solution=solution,candidate_selection=dict(selected_name='one',
        mask_only_machine_numeric_regularized_energy_ties=not continuous,
        candidates=[dict(name='one',global_mask_cost=4.,total_energy=21. if continuous else 15.)]))


def test_five_fixed_methods_and_ten_stages_without_metric_or_data_reads():
    assert len(R.METHODS)==5 and R.METHODS==tuple(R.EXP.ARMS[i]for i in (0,1,3,5,6))
    assert R.CASES==R.PREVIOUS.CASES and len(R.CASES)==4
    assert R.AUDIT_SUBDIR=='independent_root_audit_v1'
    assert len(R.TITLES)==len(R.CAPTIONS)==10
    with pytest.raises(ValueError,match='explicit root'):R.render()


def test_energy_ledger_keeps_N_O_S_and_mask_exactly_once():
    a=arm(True);saved=deepcopy(a);e=R._energies(a,R.MASK_FIT)
    assert a==saved and e['total']==21. and e['weighted_mask']==6. and e['mask_count_in_optimizer']==1
    assert e['native']==10. and e['O']==2. and e['S']==3.
    b=arm(False);e=R._energies(b,R.SOURCE_NATIVE)
    assert e['total']==15. and e['weighted_mask']==0. and e['native_mask_tie_break_only']
    bad=arm(True);bad['solution']['objective']=27.
    with pytest.raises(ValueError,match='exactlyonce'):R._energies(bad,R.LOCKED)


def test_e_and_failed_joint_do_not_invent_comparable_energies():
    assert not R._energies({},R.BASELINE)['applied']
    rejected=R._energies(dict(accepted=False),R.MASK_FIT)
    assert rejected['native']is None and rejected['total']is None and rejected['failed_joint']
