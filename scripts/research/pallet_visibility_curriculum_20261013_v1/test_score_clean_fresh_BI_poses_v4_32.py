"""Fake prediction/reference fixtures only; actual score/labels/preds IO=0."""
from copy import deepcopy

import numpy as np
import pytest

from . import score_clean_fresh_BI_poses_v4_32 as P
from .test_prepare_geometry_guided_native_metadata_v4_32 import Poison


LEGACY=dict(centroid_field='NATIVE_T_EQUAL_MODEL_CENTER_FOR_CENTERED_XYZ',
    original_native_xyz_zero_mean_verified=True,comparison_only=True,numeric_worker_exposure=False)


def prediction(arm, *, mean=(0.,0.,0.),family='ORIGINAL_WD',accepted=True):
    xyz=P.C.cuboid(.8,.4,.6)+np.asarray(mean)
    Q=P.METRIC.rotation_y(.23);R=P.METRIC.rotation_y(.12)
    coarse=dict(R=R.tolist(),t=[0.,0.,3.])
    families=P.M.CLEAN._families(xyz,coarse,Q)
    f=families[family];pose=P.M._actual_pose(f,f['coarse_pose']['R'],f['coarse_pose']['t'])
    if arm==P.ARMS[0]:return dict(accepted=accepted,actual_pose=pose)
    policy=next(p for p,arms in P.M.POLICY_ARMS.items()if arm in arms)
    plan=dict(source_policy=policy,observation_family=P.M.G.OBSERVATION_FAMILY,
        population=P.M.G.POPULATION,selection_profile=P.M.G.SELECTION_PROFILE,
        initial_model_ray_binding=dict(geometry=dict(native_xyz=xyz.tolist(),physical_registration_Q=Q.tolist())))
    return dict(accepted=accepted,pose_available=True,actual_pose=pose,WD_family=family,
        R=f['coarse_pose']['R'],t=f['coarse_pose']['t'],new_geometry_edge_cost_plans={family:plan})


def toy_truth(pose, *, remainder=False):
    value=dict(R=pose['R_physical'],t=pose['centroid'],xyz=[.8,.4,.6],order=2)
    if remainder:value.update(alternatives=[])
    return value


def cohort():
    rows=[];comparisons=[];references={p:{}for p in P.PARTITIONS}
    for part,count in P.PARTITIONS.items():
        for i in range(count):
            ident=part+':'+str(i)
            arms={a:prediction(a)for a in P.ARMS}
            row=dict(id=ident,partition=part,arms=arms,legacy_e_pose_semantics=deepcopy(LEGACY))
            rows.append(row)
            comparisons.append(dict(id=ident,partition=part,prediction=deepcopy(arms[P.ARMS[0]]),legacy_e_pose_semantics=deepcopy(LEGACY)))
            references[part][ident]=toy_truth(arms[P.ARMS[0]]['actual_pose'],remainder=part=='REMAINDER')
    return rows,comparisons,references


def test_cm_not_meters_and_headline_inclusive_boundary_preserves_old_numeric_errors():
    pose=dict(R_physical=np.eye(3).tolist(),centroid=[.05,0.,0.])
    truth=dict(R=np.eye(3).tolist(),t=[0.,0.,0.],xyz=[.8,.4,.6],order=2)
    value=P.pose_metric(pose,truth)
    old=P.METRIC.pose_metric(pose,truth)
    assert value['translation_cm']==old['translation_cm']==5.
    assert value['rotation_deg']==old['rotation_deg']==0.
    assert value['success_5cm_5deg']is True and value['legacy_strict_success_5cm_5deg']is False
    pose['centroid']=[.01,0.,0.]
    assert P.pose_metric(pose,truth)['translation_cm']==1.


@pytest.mark.parametrize('family',P.M.FAMILIES)
def test_nonzero_native_origin_and_Q_do_not_become_center_or_physical_R(family):
    arm=P.ARMS[1];entry=prediction(arm,mean=(.2,-.1,.3),family=family)
    before=deepcopy(entry)
    pose=P.prediction_pose(entry,arm=arm)
    value=P.pose_metric(pose,toy_truth(pose))
    assert value['translation_cm']==0. and value['rotation_deg']<1e-5
    assert not np.allclose(pose['native_t'],pose['centroid'])
    assert not np.allclose(pose['R_cf'],pose['R_physical'])
    assert entry==before
    wrong=deepcopy(entry);wrong['actual_pose']['centroid']=deepcopy(pose['native_t'])
    wrong['actual_pose']['model_center_camera_xyz']=deepcopy(pose['native_t'])
    with pytest.raises(ValueError,match='center'):P.prediction_pose(wrong,arm=arm)
    wrong=deepcopy(entry);wrong['actual_pose']['R_physical']=deepcopy(pose['R_cf'])
    with pytest.raises(ValueError,match='R-Q'):P.prediction_pose(wrong,arm=arm)


def test_missing_pose_gate_before_payload_and_reference_and_fallback_pose_still_scores():
    entry=dict(accepted=False,pose_available=False,actual_pose=Poison(),new_geometry_edge_cost_plans=Poison())
    assert P.prediction_pose(entry,arm=P.ARMS[1])is None
    failed=P.pose_metric(None,Poison())
    assert failed['success_5cm_5deg']is False and failed['translation_cm']is None
    fallback=prediction(P.ARMS[1],accepted=False)
    pose=P.prediction_pose(fallback,arm=P.ARMS[1])
    assert P.pose_metric(pose,toy_truth(pose))['success_5cm_5deg']is True


def test_all238_fivearms_allfour_populations_retained_and_no_pose_fallback_counts():
    rows,comparisons,refs=cohort()
    arm=P.ARMS[1];test=next(r for r in rows if r['partition']=='TEST')
    test['arms'][arm]=dict(accepted=False,pose_available=False,actual_pose=Poison())
    rows[47]['arms'][arm]['accepted']=False  # A finite fallback remains scored.
    refs['REMAINDER']['REMAINDER:0']['R']=None
    compact=P.compact_predictions(rows,comparisons,[r['id']for r in rows])
    reports=P.score_rows(compact,refs)
    assert {p:r['count']for p,r in reports.items()}==dict(CAL=46,TEST=49,REMAINDER=143,PRIMARY=192)
    for p,report in reports.items():
        assert len(report['rows'])==report['count']and set(report['summary'])==set(P.ARMS)
        assert all(s['denominator']==report['count']for s in report['summary'].values())
    s=reports['TEST']['summary'][arm]
    assert s['missing_pose_count']==1 and s['fallback_count']==1 and s['unaccepted_count']==2
    assert s['translation_le5']==s['successes_5cm_5deg']==48
    assert s['translation_cm']['mean']is None and s['translation_cm']['max']is None
    assert s['translation_cm']['n']==49 and s['translation_cm']['finite_count']==48
    assert s['translation_cm']['max_status']=='POSITIVE_INFINITY_NO_POSE_FAILURE'
    assert s['translation_cm']['finite_only_diagnostic']['mean']==0.
    assert s['max_translation_id']=='TEST:0'
    s=reports['PRIMARY']['summary'][arm]
    assert s['translation_le5']==191 and s['successes_5cm_5deg']==190 and s['denominator']==192
    assert s['reference_rotation_unresolved_count']==1
    assert reports['CAL']['summary'][arm]['translation_cm']['mean']==0.


def test_finite_statistics_match_existing_sample_std_and_missing_never_filters_headline():
    values=[1.,2.,4.];stats=P.distribution(values)
    old=P.METRIC.distribution(values)
    for key in('n','mean','median','P90','std','max'):assert stats[key]==old[key]
    assert np.isclose(stats['std'],np.std(values,ddof=1))
    missing=P.distribution([1.,None,4.],missing_pose_count=1)
    assert missing['n']==3 and missing['mean']is None and missing['median']is None
    assert missing['finite_only_diagnostic']['mean']==2.5


@pytest.mark.parametrize('change',['e','arm','order','missing','duplicate','partition','partition_swap'])
def test_identity_or_e_alias_or_any_arm_dropout_fails_before_references(change):
    rows,comparisons,_=cohort();ids=[r['id']for r in rows]
    if change=='e':rows[0]['arms'][P.ARMS[0]]['accepted']=False
    elif change=='arm':del rows[0]['arms'][P.ARMS[-1]]
    elif change=='order':rows[0],rows[1]=rows[1],rows[0]
    elif change=='missing':rows.pop()
    elif change=='duplicate':rows[-1]['id']=rows[-2]['id']
    elif change=='partition_swap':
        rows[0]['partition']=comparisons[0]['partition']='TEST'
        rows[46]['partition']=comparisons[46]['partition']='CAL'
    else:rows[-1]['partition']=comparisons[-1]['partition']='TEST'
    with pytest.raises(ValueError):P.compact_predictions(rows,comparisons,ids)


def test_root_score_and_explicit_SHA_gate_before_any_IO(monkeypatch):
    def no(*a,**k):raise AssertionError('IO before root score and explicit hashes')
    monkeypatch.setattr(P.C,'read',no);monkeypatch.setattr(P.C,'sha',no)
    with pytest.raises(PermissionError):P.score()
    for value in(None,'','A'*64):
        with pytest.raises(ValueError,match='before IO'):
            P.score(root_score_GO=True,expected_central_plan_sha256=value,expected_prediction_seal_sha256='a'*64)


def test_prePRED_gate_rejects_partialseal_before_audit_or_reference_decoder(monkeypatch):
    output=P.M.B/'SYNTHETIC_NOT_A_REAL_OUTPUT';ids=[str(i)for i in range(238)]
    seal=dict(rows=237,arms=list(P.ARMS),arm_count=5,all_declared_arms_complete=True,no_population_drop=True,ordered_ids=ids)
    docs={output/'PREDICTION_SEAL.json':seal,output/'PROTOCOL.json':dict(ordered_ids=ids,central_plan_sha256='c'*64)}
    def read(path):
        if path not in docs:raise AssertionError('Audit/reference metadata read before full seal')
        return docs[path]
    monkeypatch.setattr(P.C,'read',read)
    hashes={P.Path(P.M.__file__):P.CONTROLLER_SHA256,output/'PREDICTION_SEAL.json':'a'*64}
    monkeypatch.setattr(P.C,'sha',lambda path:hashes.get(P.Path(path),'b'*64))
    with pytest.raises(ValueError,match='before reference'):
        P._prediction_gate(output,'c'*64,'a'*64)


def test_old_generation_costplan_and_legacy_e_center_attestation_rejected():
    arm=P.ARMS[1];entry=prediction(arm)
    entry['new_geometry_edge_cost_plans'][entry['WD_family']]['selection_profile']='V31'
    with pytest.raises(ValueError,match='typed'):P.prediction_pose(entry,arm=arm)
    with pytest.raises(ValueError,match='legacy-e'):
        P.prediction_pose(prediction(P.ARMS[0]),arm=P.ARMS[0],legacy_semantics={})


def test_unresolved_R_reuses_derived_reference_numeric_translation_without_false_joint():
    pose=dict(R_physical=np.eye(3).tolist(),centroid=[.01,0.,0.])
    ref=dict(t=[0.,0.,0.],R=None,xyz=[.8,.4,.6],order=2,alternatives=[])
    value=P.pose_metric(pose,ref,remainder=True)
    assert value['translation_cm']==1. and value['rotation_deg']is None
    assert value['success_5cm_5deg']is None and value['translation_le5']


def gate_fixture(monkeypatch):
    output=P.M.B/'SYNTHETIC_NOT_A_REAL_OUTPUT';ids=[str(i)for i in range(238)]
    seal=dict(rows=238,arms=list(P.ARMS),arm_count=5,all_declared_arms_complete=True,
        no_population_drop=True,before_fixed_reference_scoring=True,
        default='SAME_FROZEN_E unchanged',ordered_ids=ids)
    links={'predictions_sha256':'PREDICTIONS.jsonl.gz','protocol_sha256':'PROTOCOL.json',
        'input_seal_sha256':'INPUT_SEAL.json','execution_protocol_sha256':'PREDICTION_EXECUTION_PROTOCOL.json',
        'row_index_sha256':'PREDICTION_ROW_INDEX.json','separate_e_comparison_sha256':'E_COMPARISON.jsonl.gz'}
    seal.update({k:'b'*64 for k in links})
    protocol=dict(ordered_ids=ids,central_plan_sha256='c'*64,
        source_sha256={'fake_source.py':'1'*64},input_sha256={'fake_input.json':'2'*64},
        protected_user_sha256={'fake_user.json':'3'*64},original_CAL_file_sha256={'fake_CAL53.json':'4'*64})
    path=output/'independent_root_audit_v1/AUDIT.json'
    audit=dict(rows=238,failure_count=0,failures=[],
        prediction_seal=dict(path=str((output/'PREDICTION_SEAL.json').relative_to(P.C.ROOT)),sha256='a'*64))
    aseal=dict(receipt_sha256='d'*64)
    docs={output/'PREDICTION_SEAL.json':seal,output/'PROTOCOL.json':protocol,
        path:audit,path.with_name('AUDIT_SEAL.json'):aseal}
    monkeypatch.setattr(P.C,'read',lambda p:docs[p])
    hashes={P.Path(P.M.__file__):P.CONTROLLER_SHA256,output/'PREDICTION_SEAL.json':'a'*64,
        P.M.OBS.DECLARATION:'c'*64,path:'d'*64}
    monkeypatch.setattr(P.C,'sha',lambda p:hashes.get(P.Path(p),'b'*64))
    checked=[];monkeypatch.setattr(P.M,'_verify',lambda *maps:checked.extend(maps))
    return output,seal,protocol,audit,aseal,checked


def test_fullprediction_and_current_zero_failure_audit_required_before_reference(monkeypatch):
    output,seal,protocol,audit,aseal,checked=gate_fixture(monkeypatch)
    assert P._prediction_gate(output,'c'*64,'a'*64)is protocol
    assert checked==[protocol[k]for k in('source_sha256','input_sha256','protected_user_sha256','original_CAL_file_sha256')]
    audit['failure_count']=1
    with pytest.raises(ValueError,match='zero-failure'):P._prediction_gate(output,'c'*64,'a'*64)
    audit['failure_count']=0;audit['prediction_seal']['sha256']='0'*64
    with pytest.raises(ValueError,match='prediction seal'):P._prediction_gate(output,'c'*64,'a'*64)
    audit['prediction_seal']['sha256']='a'*64;aseal['receipt_sha256']='0'*64
    with pytest.raises(ValueError,match='zero-failure'):P._prediction_gate(output,'c'*64,'a'*64)


def test_reference_metadata_uses_exact_published_paths_SHA_and_all_annotation_inputs(monkeypatch):
    ids=['REMAINDER:'+str(i)for i in range(143)]
    compact=[dict(id=i,partition='REMAINDER')for i in ids]
    bindings={p:dict(path=str(v.relative_to(P.C.ROOT)),sha256='a'*64)for p,v in P.REFERENCE_PATHS.items()}
    annotations=[dict(id=i,annotation='synthetic/'+str(n)+'.json',annotation_sha256='b'*64)for n,i in enumerate(ids)]
    refplan=dict(schema='DERIVED_FINAL143_REFERENCE_SCORE_PLAN_V4_14',rows=143,severe=0,ids=ids,annotations=annotations,
        annotation_coordinate_or_pose_values_decoded_at_preparation=False,
        no_reference_used_to_select_prediction_family_or_pose=True,
        all143_translation_rows_even_if_rotation_unresolved=True,
        input_sha256={a['annotation']:a['annotation_sha256']for a in annotations},
        source_sha256={'fake_ref.py':'c'*64},protected_user_sha256={'fake_user.json':'d'*64})
    docs={P.PUBLIC_MANIFEST:dict(files=[dict(path=str(P.ANCHOR.relative_to(P.C.ROOT)),sha256=P.ANCHOR_SHA256)]),
        P.ANCHOR:dict(reference_bindings=bindings),P.REFERENCE_PATHS['REMAINDER']:refplan}
    monkeypatch.setattr(P.C,'read',lambda path:docs[path])  # Numeric reference docs are deliberately absent.
    hashes={P.ANCHOR:P.ANCHOR_SHA256,P.PUBLIC_MANIFEST:P.PUBLIC_MANIFEST_SHA256}
    monkeypatch.setattr(P.C,'sha',lambda path:hashes.get(path,'a'*64))
    checked=[];monkeypatch.setattr(P.M,'_verify',lambda *maps:checked.extend(maps))
    actual,plan,inputs=P._reference_metadata(dict(input_sha256={'fake_prediction_input.json':'e'*64}),compact)
    assert actual==bindings and plan is refplan
    assert all(inputs[a['annotation']]==a['annotation_sha256']for a in annotations)
    assert checked[1:]==[refplan['source_sha256'],refplan['protected_user_sha256']]
    annotations[0]['annotation']='../forbidden.json'
    with pytest.raises(ValueError,match='path/SHA'):P._reference_metadata(dict(input_sha256={}),compact)


def test_frozen_R2_controller_default_and_output_override():
    assert P.M.__name__.endswith('evaluate_clean_fresh_BI_poses_v4_32_r2')
    assert P.M.OUTPUT.name=='clean_fresh_BI_pose_eval_v4_32_r2'
    assert P.score.__kwdefaults__['output']==P.M.OUTPUT
