"""Fake poses/references/metadata only; no saved prediction or reference IO."""
from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest

from . import score_largest_component_global_outline_poses_v4_33 as S
from .test_score_clean_fresh_BI_poses_v4_32 import prediction, toy_truth, cohort, LEGACY
from .test_prepare_geometry_guided_native_metadata_v4_32 import Poison


def test_all_pure_numeric_and_fixed_reference_functions_are_original_exact_aliases():
    for name in ('prediction_pose','compact_predictions','pose_metric','distribution','score_rows',
                 '_reference_metadata','_read_references'):
        assert getattr(S,name) is getattr(S.P,name)
        assert getattr(S,name).__code__ is getattr(S.P,name).__code__
    assert S.REFERENCE_PATHS is S.P.REFERENCE_PATHS
    assert S.ANCHOR==S.P.ANCHOR and S.ANCHOR_SHA256==S.P.ANCHOR_SHA256
    assert S.PUBLIC_MANIFEST_SHA256==S.P.PUBLIC_MANIFEST_SHA256
    assert S.ARMS==S.P.ARMS and S.PARTITIONS==S.P.PARTITIONS
    assert S.score.__kwdefaults__['output']==S.M.OUTPUT!=S.P.M.OUTPUT
    S._pins()  # Source hashes only, no experiment metadata.


def test_cm_and_inclusive_headline_exact_old_numeric_errors_and_sample_statistics():
    pose=dict(R_physical=np.eye(3).tolist(),centroid=[.05,0.,0.])
    truth=dict(R=np.eye(3).tolist(),t=[0.,0.,0.],xyz=[.8,.4,.6],order=2)
    value=S.pose_metric(pose,truth)
    assert value['translation_cm']==S.P.METRIC.pose_metric(pose,truth)['translation_cm']==5.
    assert value['success_5cm_5deg'] and not value['legacy_strict_success_5cm_5deg']
    stats=S.distribution([1.,2.,4.])
    assert stats['std']==S.P.distribution([1.,2.,4.])['std']
    assert np.isclose(stats['std'],np.std([1.,2.,4.],ddof=1))


@pytest.mark.parametrize('family',S.P.M.FAMILIES)
def test_native_translation_center_Q_and_physical_rotation_remain_V32(family):
    arm=S.ARMS[1];entry=prediction(arm,mean=(.2,-.1,.3),family=family)
    pose=S.prediction_pose(entry,arm=arm)
    assert not np.allclose(pose['native_t'],pose['centroid'])
    assert not np.allclose(pose['R_cf'],pose['R_physical'])
    assert S.pose_metric(pose,toy_truth(pose))['translation_cm']==0.
    wrong=deepcopy(entry);wrong['actual_pose']['centroid']=deepcopy(pose['native_t'])
    wrong['actual_pose']['model_center_camera_xyz']=deepcopy(pose['native_t'])
    with pytest.raises(ValueError,match='center'):
        S.prediction_pose(wrong,arm=arm)
    wrong=deepcopy(entry)
    wrong['new_geometry_edge_cost_plans'][family]['selection_profile']=S.SELECTION_PROFILE
    with pytest.raises(ValueError,match='typed V32'):
        S.prediction_pose(wrong,arm=arm)  # No false relabel of native population to LCC33.


def test_all238_five_arms_full_denominators_missing_and_fallback_not_dropped():
    rows,comparisons,refs=cohort();arm=S.ARMS[1]
    rows[46]['arms'][arm]=dict(accepted=False,pose_available=False,actual_pose=Poison())
    rows[47]['arms'][arm]['accepted']=False
    refs['REMAINDER']['REMAINDER:0']['R']=None
    compact=S.compact_predictions(rows,comparisons,[row['id'] for row in rows])
    reports=S.score_rows(compact,refs)
    assert {part:report['count'] for part,report in reports.items()}==dict(CAL=46,TEST=49,REMAINDER=143,PRIMARY=192)
    for part,report in reports.items():
        assert set(report['summary'])==set(S.ARMS)
        assert all(summary['denominator']==report['count'] for summary in report['summary'].values())
    test=reports['TEST']['summary'][arm]
    assert test['missing_pose_count']==test['fallback_count']==1
    assert test['translation_cm']['mean'] is None
    assert test['translation_cm']['max_status']=='POSITIVE_INFINITY_NO_POSE_FAILURE'
    assert test['translation_le5']==test['successes_5cm_5deg']==48
    primary=reports['PRIMARY']['summary'][arm]
    assert primary['denominator']==192 and primary['translation_le5']==191 and primary['successes_5cm_5deg']==190
    assert primary['rotation_unresolved']==2
    assert reports['CAL']['summary'][arm]['translation_le5']==46


@pytest.mark.parametrize('change',['e','missing_arm','missing_row','wrong_partition'])
def test_order_and_exact_E_and_all_arms_remain_pre_reference_gates(change):
    rows,comparisons,_=cohort();ids=[row['id'] for row in rows]
    if change=='e':rows[0]['arms'][S.ARMS[0]]['accepted']=False
    elif change=='missing_arm':del rows[0]['arms'][S.ARMS[-1]]
    elif change=='missing_row':rows.pop()
    else:rows[0]['partition']=comparisons[0]['partition']='TEST'
    with pytest.raises(ValueError):
        S.compact_predictions(rows,comparisons,ids)


def test_score_GO_explicit_SHAs_and_source_peer_binding_precede_all_IO(monkeypatch):
    def no(*args,**kwargs):raise AssertionError('IO before GO and explicit bindings')
    monkeypatch.setattr(S.C,'read',no);monkeypatch.setattr(S.C,'sha',no)
    with pytest.raises(PermissionError):S.score()
    for central,pred,peer in ((None,'a'*64,{'path':'p','sha256':'b'*64}),
                               ('c'*64,'a'*64,None),('c'*64,'a'*64,{'path':'p','sha256':'INVALID'})):
        with pytest.raises(ValueError,match='before IO'):
            S.score(root_score_GO=True,expected_central_plan_sha256=central,
                expected_prediction_seal_sha256=pred,source_peer_binding=peer)


def gate(monkeypatch):
    output=S.M.B/'SYNTHETIC_LCC_NOT_REAL'
    ids=[str(i) for i in range(238)]
    protocol=dict(ordered_ids=ids,central_plan_sha256='c'*64)
    seal=dict(rows=238,ordered_ids=ids,arms=list(S.ARMS),arm_count=5,all_declared_arms_complete=True,
        no_population_drop=True,before_fixed_reference_scoring=True,default='SAME_FROZEN_E unchanged',
        outline_selection_profile=S.SELECTION_PROFILE,native_CAL32_model_relabelled=False,
        **{key:'b'*64 for key in ('predictions_sha256','protocol_sha256','input_seal_sha256',
            'execution_protocol_sha256','row_index_sha256','separate_e_comparison_sha256','native_and_e_parity_sha256')})
    parity=dict(rows=238,native_arm_count=2,native_arm_payload_equalities=476,e_payload_equalities=238,
        failure_count=0,numeric_tolerance=0.)
    path=output/'independent_root_audit_v1/AUDIT.json'
    audit=dict(rows=238,failure_count=0,failures=[],prediction_seal=dict(path=S.M._relative(output/'PREDICTION_SEAL.json'),sha256='a'*64),
        outline_selection_profile=S.SELECTION_PROFILE,native_observation_family=S.M.M.G.OBSERVATION_FAMILY,
        native_noise_population=S.M.M.G.POPULATION,native_selection_profile=S.M.M.G.SELECTION_PROFILE,
        source_sha256={'fake.py':'1'*64},input_sha256={'fake.json':'2'*64},protected_user_sha256={'fakeuser':'3'*64})
    aseal=dict(receipt_sha256='d'*64)
    docs={output/'PREDICTION_SEAL.json':seal,output/'NATIVE_AND_E_PARITY_V1.json':parity,
        path:audit,path.with_name('AUDIT_SEAL.json'):aseal}
    reads=[]
    def read(p):reads.append(Path(p));return docs[Path(p)]
    monkeypatch.setattr(S.C,'read',read)
    hashes={output/'PREDICTION_SEAL.json':'a'*64,S.M.DECLARATION:'c'*64,path:'d'*64}
    monkeypatch.setattr(S.C,'sha',lambda p:hashes.get(Path(p),'b'*64))
    monkeypatch.setattr(S,'_pins',lambda:None)
    monkeypatch.setattr(S.M,'_load_prepared',lambda folder:protocol)
    checked=[];monkeypatch.setattr(S.M,'_verify',lambda *maps:checked.extend(maps))
    return output,protocol,seal,parity,audit,aseal,reads,checked


def test_complete_LCC33_prediction_native_E_parity_and_current_independent_audit(monkeypatch):
    output,protocol,_,_,audit,_,_,checked=gate(monkeypatch)
    assert S._prediction_gate(output,'c'*64,'a'*64) is protocol
    assert checked==[audit[key] for key in ('source_sha256','input_sha256','protected_user_sha256')]


def test_incomplete_prediction_stops_before_parity_audit_or_reference_documents(monkeypatch):
    output,_,seal,_,_,_,reads,_=gate(monkeypatch)
    seal['rows']=237
    with pytest.raises(ValueError,match='before reference'):
        S._prediction_gate(output,'c'*64,'a'*64)
    assert reads==[output/'PREDICTION_SEAL.json']


@pytest.mark.parametrize('change',['outline','central','relabel','parity','audit_failure','audit_pred','audit_population','audit_receipt'])
def test_new_profile_and_native32_population_and_parity_and_zeroaudit_fail_closed(change,monkeypatch):
    output,protocol,seal,parity,audit,aseal,_,_=gate(monkeypatch)
    if change=='outline':seal['outline_selection_profile']='OLD'
    elif change=='central':protocol['central_plan_sha256']='0'*64
    elif change=='relabel':seal['native_CAL32_model_relabelled']=True
    elif change=='parity':parity['native_arm_payload_equalities']=475
    elif change=='audit_failure':audit['failure_count']=1
    elif change=='audit_pred':audit['prediction_seal']['sha256']='0'*64
    elif change=='audit_population':audit['native_noise_population']='FAKE_V33_NOISE'
    else:aseal['receipt_sha256']='0'*64
    with pytest.raises(ValueError):
        S._prediction_gate(output,'c'*64,'a'*64)


def test_toy_score_writes_execution_protocol_before_any_reference_then_all_four_reports(monkeypatch,tmp_path):
    rows,comparisons,refs=cohort();ids=[row['id'] for row in rows]
    monkeypatch.setattr(S.C,'ROOT',tmp_path)
    output=tmp_path/'score';output.mkdir();peer=tmp_path/'peer.json';peer.write_text('{}')
    S.C.save_rows(output/'PREDICTIONS.jsonl.gz',rows)
    S.C.save_rows(output/'E_COMPARISON.jsonl.gz',comparisons)
    for name in ('PREDICTION_SEAL.json','PROTOCOL.json','INPUT_SEAL.json','PREDICTION_EXECUTION_PROTOCOL.json',
                 'PREDICTION_ROW_INDEX.json','NATIVE_AND_E_PARITY_V1.json','independent_root_audit_v1/AUDIT.json',
                 'independent_root_audit_v1/AUDIT_SEAL.json'):
        path=output/name;path.parent.mkdir(exist_ok=True);path.write_text('{}')
    protocol=dict(ordered_ids=ids,source_sha256={},input_sha256={},protected_user_sha256={})
    bindings={part:dict(path='synthetic/'+part,sha256='a'*64) for part in S.PARTITIONS}
    monkeypatch.setattr(S,'_source_peer',lambda binding:peer)
    gate_calls=[]
    monkeypatch.setattr(S,'_prediction_gate',lambda *args:gate_calls.append(True) or protocol)
    monkeypatch.setattr(S,'_reference_metadata',lambda *args:(bindings,dict(source_sha256={},protected_user_sha256={}),{}))
    monkeypatch.setattr(S,'_sources',lambda:{'synthetic_score.py':'b'*64})
    monkeypatch.setattr(S.M,'_verify',lambda *maps:None)
    def references(compact,refplan):
        assert len(compact)==238
        saved=S.C.read(output/'SCORING_PROTOCOL.json')
        assert saved['numeric_reference_decodes_before_this_protocol']==0
        assert saved['native_CAL32_models_relabelled'] is False
        return refs
    monkeypatch.setattr(S,'_read_references',references)
    result=S.score(root_score_GO=True,expected_central_plan_sha256='c'*64,
        expected_prediction_seal_sha256='d'*64,source_peer_binding=dict(path='peer.json',sha256=S.C.sha(peer)),output=output)
    assert result['path']=='score/SCORE_SEAL.json' and len(gate_calls)==2
    assert S.C.read(output/'PRIMARY_SCORE.json')['count']==192
    assert S.C.read(output/'TEST_SCORE.json')['summary'][S.ARMS[0]]['denominator']==49
    seal=S.C.read(output/'SCORE_SEAL.json')
    assert seal['reference_bindings']==bindings and len(seal['score_sha256'])==4
    assert seal['outline_selection_profile']==S.SELECTION_PROFILE
    assert seal['default']=='SAME_FROZEN_E unchanged' and seal['all_rows_kept']
    with pytest.raises(FileExistsError,match='Preserve'):
        S.score(root_score_GO=True,expected_central_plan_sha256='c'*64,
            expected_prediction_seal_sha256='d'*64,source_peer_binding=dict(path='peer.json',sha256='a'*64),output=output)
