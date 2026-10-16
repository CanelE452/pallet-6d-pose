"""Pure V31 postscore diagnostic adaptation; no actual data or model IO."""
from copy import deepcopy
import gzip
import json

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from . import diagnose_frozen_e_pose_under_v31_v4_31b as D
from .test_diagnose_frozen_e_pose_under_v29_v4_29b import scene, DummyProblem


def test_exact_new_paths_four_profiles_and_private_M31_modules():
    assert D.OUTPUT.name=='frozen_e_geometry_guided_objective_diagnosis_v4_31b'
    assert tuple(D.profiles())==D.ARMS==tuple(D.EXP.ARMS[1:])
    assert [p['containment']for p in D.profiles().values()]==[False,True,False,True]
    assert D.M._problem is D.EXP._problem and D.M._models is D.EXP._models
    assert D.M.PREP.G is D.EXP.G and D.M.CAL.M.CORE is D.EXP.G


def test_all_pure_numerical_function_codeobjects_and_parent_globals_unchanged():
    before=dict(D.PARENT.__dict__)
    namespace=D._facade('a'*64)
    for name in('validate_saved_pose_basis','transport_diagnostic_pose','source_inventory',
            'plan_summary','native_buckets','evaluate_parts','comparison','make_problem'):
        function=getattr(namespace,name)
        assert function.__code__ is getattr(D.PARENT,name).__code__
        assert function.__globals__['M'] is D.M
        assert function.__globals__ is not getattr(D.PARENT,name).__globals__
    assert all(D.PARENT.__dict__[key]is value for key,value in before.items())


def test_replay_AST_changes_only_the_one_output_schema_constant():
    import ast,inspect
    original=ast.parse(inspect.getsource(D.PARENT.run))
    constants=[node for node in ast.walk(original)if isinstance(node,ast.Constant)
        and node.value=='POSTSCORE_FROZEN_E_V29_OBJECTIVE_DIAGNOSIS_V29B']
    assert len(constants)==1
    code=D._run_code()
    assert D.SCHEMA in code.co_consts
    assert 'POSTSCORE_FROZEN_E_V29_OBJECTIVE_DIAGNOSIS_V29B'not in code.co_consts


def test_noncausal_e_feasibility_cost_preference_gap_and_fallback_classifications():
    good=dict(accepted_current_guards=True,finite_objective=True,total=1.)
    assert 'search_coverage_gap'in D.comparison(good,{**good,'total':2.},selected_accepted=True)
    assert 'objective_prefers_selected'in D.comparison({**good,'total':3.},good,selected_accepted=True)
    assert 'hard_or_numeric_guards'in D.comparison({**good,'accepted_current_guards':False},good,selected_accepted=True)
    assert 'own_fallback'in D.comparison(good,good,selected_accepted=False)


def test_frozen_parts_and_evaluate_only_preserve_native_O_S_P0_C_once():
    _,R,t,_,_=scene()
    actual=D.evaluate_parts(DummyProblem(),R,t)
    assert actual['total']==14. and actual['native']==10.
    assert [actual[k]for k in('O','S','P','C','C_weight','weighted_C')]==[.2,.3,0.,5.,.7,3.5]
    assert actual['maximum_internal_ledger_difference']==0.
    assert actual['pose_optimization_or_runtime_candidate_insertions']==0


def test_original_native_WD_and_unordered_physical_transport_do_not_change_channel_claim():
    xyz,R,t,family,pose=scene();Q=Rotation.from_euler('y',90,degrees=True).as_matrix()
    permutation=[1,0,3,2,5,4,7,6]
    other=dict(name='SWAPPED_WD',native_xyz=(xyz[permutation]@Q.T).tolist(),physical_registration_Q=Q.tolist())
    before=deepcopy(pose)
    moved,translation,audit=D.transport_diagnostic_pose(pose,family,other)
    assert np.allclose(moved@Q,R)and np.array_equal(translation,t)
    assert audit['available']and audit['not_the_original_saved_e_channel_correspondence']
    assert not np.allclose(xyz@R.T+t,np.asarray(other['native_xyz'])@moved.T+translation)
    assert pose==before


def test_no_actual_prepare_or_replay_IO_without_postscore_GO(monkeypatch):
    monkeypatch.setattr(D.C,'read',lambda *a:pytest.fail('Actual metadata/row IO before GO'))
    monkeypatch.setattr(D.C,'sha',lambda *a:pytest.fail('Actual source/input IO before GO'))
    monkeypatch.setattr(D.EXP,'_verify_plan',lambda *a:pytest.fail('Actual plan IO before GO'))
    for action in(D.prepare,D.run):
        with pytest.raises(ValueError,match='Separate root'):
            action(expected_central_plan_sha256='a'*64)


def test_explicit_central_gate_precedes_postscore_protocol_or_row_read(monkeypatch):
    def blocked(value):raise ValueError('central missing')
    monkeypatch.setattr(D.EXP,'_verify_plan',blocked)
    monkeypatch.setattr(D.C,'read',lambda *a:pytest.fail('Protocol read before explicit central'))
    with pytest.raises(ValueError,match='central missing'):
        D.run(root_postscore_GO=True)


def test_prefix49_skips_CAL_and_REMAINDER_JSON_and_keeps_all_TEST(tmp_path):
    path=tmp_path/'rows.gz';ids=['TEST:'+str(i)for i in range(49)]
    with gzip.open(path,'wt')as stream:
        for _ in range(46):stream.write('CAL not JSON\n')
        for ident in ids:stream.write(json.dumps(dict(id=ident,partition='TEST'))+'\n')
        stream.write('REMAINDER not parsed\n')
    assert [r['id']for r in D._TEST_prefix(path,ids)]==ids


def test_typed_prepared_plan_cannot_use_old_profile_before_frozen_replay(monkeypatch,tmp_path):
    monkeypatch.setattr(D.EXP,'_verify_plan',lambda expected:{})
    D.C.write(tmp_path/'PLAN.json',dict(central_plan_sha256='a'*64,selection_profile='OLD'))
    with pytest.raises(ValueError,match='newly typed V31'):
        D.run(tmp_path,root_postscore_GO=True,expected_central_plan_sha256='a'*64)
