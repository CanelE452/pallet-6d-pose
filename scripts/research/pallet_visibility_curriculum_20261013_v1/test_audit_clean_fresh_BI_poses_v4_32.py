"""Toy-only selected audit checks; no actual packet/model/reference reads."""
from copy import deepcopy
import inspect

import numpy as np
import pytest

from . import audit_clean_fresh_BI_poses_v4_32 as A


def forbidden(*args, **kwargs):
    raise AssertionError('Forbidden regeneration/IO/current objective called')


@pytest.fixture(scope='module')
def genuine():
    # Synthetic geometry, fresh toy SQPnP and two-iteration toy refinements
    # generate a realistic saved result. None is called by the audit itself.
    from .test_evaluate_clean_fresh_BI_poses_v4_32 import packet, V
    _,_,_,value = packet()
    result = V.solve_worker_packet(value)
    identifiers = value['identifiers']
    comparison = dict(id=identifiers['id'], partition=identifiers['partition'],
        prediction=dict(accepted=True, actual_pose=dict(centroid=[0.,0.,3.])),
        legacy_e_pose_semantics=dict(numeric_worker_exposure=False, comparison_only=True))
    row = {**result, 'id':identifiers['id'], 'partition':identifiers['partition'],
        'image_sha256':identifiers['image_sha256'], 'legacy_e_pose_semantics':comparison['legacy_e_pose_semantics'],
        'e_comparison_parent_only_no_numeric_exposure':True}
    row['arms'] = {'SAME_FROZEN_E':deepcopy(comparison['prediction']), **row['arms']}
    return value, comparison, row


def test_GO_and_explicit_pins_before_any_metadata_IO(monkeypatch):
    monkeypatch.setattr(A.C, 'read', forbidden); monkeypatch.setattr(A.C, 'sha', forbidden)
    with pytest.raises(PermissionError):
        A.main()
    with pytest.raises(ValueError, match='before IO'):
        A.main(root_independent_audit_GO=True)


def test_source_interface_no_original_task_bank_or_e_numeric_argument():
    assert set(inspect.signature(A.audit_row).parameters) == {'packet','comparison','row','models'}
    assert not any(name in A.__dict__ for name in ('G','AUTH','SOLVER','CLEAN','FACTORY'))
    source = inspect.getsource(A)
    assert 'INDEPENDENT.recompute_native(' in source
    assert 'generate_fresh_observed_BI_pnp_proposals(' not in source
    with pytest.raises(ValueError, match='seven-key'):
        A.audit_row({'baseline':None}, {}, {})


def test_original_BI_archive_gates_unknown_and_S_payload_before_coordinates():
    class Poison(dict):
        def __getitem__(self, key):
            if key == 'native_corner': return dict.__getitem__(self, key)
            raise AssertionError('Excluded coordinate payload')
    source = dict(states=['S','U','O','T','B','I','S','U'], initial_self_hidden=[True,False,False,False,False,False,True,False],
        visible_measurements=[Poison(native_corner=k) for k in (0,1,2,3,6,7)]+[
            dict(native_corner=4,xy=[3.,4.],source='RGB'),dict(native_corner=5,xy=[5.,6.],source='UNKNOWN')])
    assert [r['native_corner'] for r in A.original_BI_archive(source)] == [4,5]


def test_genuine_selected_native_O_S_C_and_start_readout_with_no_new_consumer_calls(genuine, monkeypatch):
    from .test_evaluate_clean_fresh_BI_poses_v4_32 import V
    value, comparison, row = deepcopy(genuine)
    before = deepcopy((value,comparison,row))
    monkeypatch.setattr(V, 'build_clean_native_cores', forbidden)
    monkeypatch.setattr(V.CLEAN, 'prepare_clean_fresh_BI_task', forbidden)
    monkeypatch.setattr(V.G.GeometryGuidedEdgeProblem, 'evaluate', forbidden)
    monkeypatch.setattr(V.G.GeometryGuidedEdgeProblem, 'objective', forbidden)
    result = A.audit_row(value,comparison,row)
    assert not [r for r in result['checks'] if not r['passed']]
    assert result['accepted'] + result['fallback'] == 4
    assert (value,comparison,row) == before


def test_family_reconstruction_preserves_shifted_body_and_physical_Q(genuine):
    value = deepcopy(genuine[0]);source = value['clean_input']
    xyz = np.asarray(source['native_xyz']) + [.4,-.2,.3]
    source['native_xyz'] = xyz.tolist()
    families = A.geometry_families(value)
    original = families['ORIGINAL_WD']; swapped = families['SWAPPED_WD']
    assert original['native_xyz'] == xyz.tolist()
    assert A.REPLAY.check_initial_physical_body(value, swapped)['physical_registration_and_geometric_center_preserved']
    assert A.fresh_input_digest(source, original) != A.fresh_input_digest(source, swapped)


@pytest.mark.parametrize('what', ['e','native','physical_t','readout','start_hash','selected_min','prior'])
def test_saved_corruptions_are_detected_without_row_dropping(genuine, what):
    value, comparison, row = deepcopy(genuine)
    arm = next(name for name in A.NEW_ARMS if row['arms'][name]['accepted'])
    entry = row['arms'][arm]
    if what == 'e': row['arms']['SAME_FROZEN_E']['accepted'] = False
    elif what == 'native': entry['solution']['native_costs'][0]['weighted_robust_cost'] += 2.
    elif what == 'physical_t': entry['actual_pose']['native_t'][2] += .1
    elif what == 'readout':
        entry['readout']['point_sources'][0] = 'FABRICATED_POINT_SOURCE'
    elif what == 'start_hash': entry['family_refinements']['ORIGINAL_WD']['search']['same_clean_start_packet_sha256'] = '0'*64
    elif what == 'prior': entry['solution']['regularizer_costs'][0]['weighted_cost'] += 1.
    else:
        selection = entry['candidate_selection']; selection['selected_name'] = 'MISSING'
    result = A.audit_row(value,comparison,row)
    assert any(not c['passed'] for c in result['checks'])


def test_selector_stable_numeric_ties_and_false_fallback():
    def candidate(name,index,total):
        return dict(name='ORIGINAL_WD/'+name,index=index,family='ORIGINAL_WD',native_energy=total,
            regularizer_energy=0.,global_containment_cost=0.,weighted_containment_objective=0.,total_energy=total)
    selection = dict(old_full_mask_cost_or_numeric_tie_break_used=False,e_numeric_pose_prior_or_fallback=False,
        native_Pareto_audit_not_prefilter=True,containment_weight=0.,selected_name='ORIGINAL_WD/A',selected_family='ORIGINAL_WD',
        candidates=[candidate('B',2,1.),candidate('A',1,1.+np.finfo(float).eps)])
    checks=[]; check=lambda ok,reason:checks.append(bool(ok))
    assert A._selection_checks(selection, True, A.POLICIES[0], False, 0., check)['name']=='ORIGINAL_WD/A'
    assert all(checks)
    checks.clear();A._selection_checks(selection, False, A.POLICIES[0], False, 0., check)
    assert not all(checks)


def test_invalid_coarse_four_no_pose_arms_retained_without_other_payload(genuine):
    from .test_evaluate_clean_fresh_BI_poses_v4_32 import V
    value, comparison, _ = deepcopy(genuine)
    value['clean_input']['initial_coarse_pose'] = dict(source=A.COARSE_SOURCE, independent_of_cached_pose_bank=True)
    value['source_contract']['ready'] = False
    value['physics'] = value['typed_bundles'] = value['regularizers'] = None
    result = V.solve_worker_packet(value);ids=value['identifiers']
    row={**result,'id':ids['id'],'partition':ids['partition'],'image_sha256':ids['image_sha256'],
         'legacy_e_pose_semantics':comparison['legacy_e_pose_semantics'], 'e_comparison_parent_only_no_numeric_exposure':True}
    row['arms']={'SAME_FROZEN_E':comparison['prediction'],**row['arms']}
    audit=A.audit_row(value,comparison,row)
    assert audit['no_pose'] and audit['accepted']==audit['fallback']==0
    assert all(c['passed'] for c in audit['checks'])


def test_new_model_tags_digest_zero_floor_and_old_noise_rejection():
    model=dict(schema='CAL_INITIAL_ONLY_OBSERVED_NATIVE_RAY_TANGENT_SCALE_V32',fit_partition='CAL',
        observation_generation=A.REPLAY.FAMILY,noise_population=A.REPLAY.POPULATION,selection_profile=A.REPLAY.PROFILE,
        raw_observation_generation=A.RAW_GENERATION,source_policy=A.POLICIES[0],newly_fitted_selection_population=True,
        raw_observation_generation_unchanged=True,prior_V29_fitted_model_relabelled=False,
        prior_V30_fitted_model_relabelled=False,prior_V31_fitted_model_relabelled=False,
        gamma=0.,pixel_floor_variance_px2=1.,bias_correction_px=0.)
    model['model_digest_sha256']=A.digest(model)
    assert A.check_model(model,A.POLICIES[0],tangent=True)
    assert A.check_model(dict(status='FITTED_CAL_ONLY',model=model),A.POLICIES[0],tangent=True)
    for field,value in [('selection_profile','OLD'),('noise_population','OLD'),('gamma',4.)]:
        changed=deepcopy(model);changed[field]=value
        with pytest.raises(ValueError):A.check_model(changed,A.POLICIES[0],tangent=True)
    assert not A.check_model(dict(model=None,status='MODEL_UNAVAILABLE_EMPTY',old_model_substituted=False),A.POLICIES[0])


def test_actual_public_model_schema_four_files_sixteen_wrapped_LOGO_entries_on_toy_metadata(tmp_path):
    # No actual saved model bytes or errors are opened. The frozen fitter's
    # full-file / nested-status interface is reproduced with scalar fixtures.
    def model(policy,tangent=False):
        value=dict(schema=('CAL_INITIAL_ONLY_OBSERVED_NATIVE_RAY_TANGENT_SCALE_V32' if tangent else
            'CAL_INITIAL_ONLY_OBSERVED_NATIVE_RAY_EDGE_NOISE_V32'),fit_partition='CAL',
            observation_generation=A.REPLAY.FAMILY,noise_population=A.REPLAY.POPULATION,selection_profile=A.REPLAY.PROFILE,
            raw_observation_generation=A.RAW_GENERATION,source_policy=policy,newly_fitted_selection_population=True,
            raw_observation_generation_unchanged=True,prior_V29_fitted_model_relabelled=False,
            prior_V30_fitted_model_relabelled=False,prior_V31_fitted_model_relabelled=False)
        value.update(dict(gamma=.2,pixel_floor_variance_px2=1.,bias_correction_px=0.) if tangent else
                     dict(normal_variance_component_px2=2.,angular_variance_component_sine2=.01))
        value['model_digest_sha256']=A.digest(value)
        return value
    report={};assets={}
    for policy in A.POLICIES:
        (tmp_path/policy).mkdir()
        for name,tangent in [('MODEL.json',False),('TANGENT_MODEL.json',True)]:
            A.C.write(tmp_path/policy/name,model(policy,tangent))
        (tmp_path/policy/'CAL_ERRORS.jsonl.gz').write_bytes(b'OPAQUE synthetic journal; never decoded')
        report[policy]=dict(inventory46=46,compatible44=44,LOGO_folds=[dict(heldout_session=str(i),
            line=dict(status='FITTED_CAL_ONLY',model=model(policy)),
            tangent=dict(status='FITTED_CAL_ONLY',model=model(policy,True))) for i in range(4)])
    A.C.write(tmp_path/'CAL_LOGO_REPORT.json',report)
    A.C.write(tmp_path/'PREPARATION_SEAL.json',{'synthetic':True})
    for p in tmp_path.rglob('*'):
        if p.is_file() and p.name!='PREPARATION_SEAL.json':assets[str(p.relative_to(tmp_path))]=A.C.sha(p)
    seal=dict(fit_partition='CAL',CAL_inventory=46,compatible_inventory=44,
        observation_generation=A.REPLAY.FAMILY,noise_population=A.REPLAY.POPULATION,
        selection_profile=A.REPLAY.PROFILE,raw_observation_generation=A.RAW_GENERATION,
        source_sha256={},input_sha256={},assets=assets,preparation_seal_sha256=A.C.sha(tmp_path/'PREPARATION_SEAL.json'))
    A.C.write(tmp_path/'MODEL_SEAL.json',seal)
    models,_=A._load_models(tmp_path)
    assert set(models)==set(A.POLICIES) and all(v['available'] for v in models.values())
    report[A.POLICIES[0]]['LOGO_folds'][0]['line']['model']['selection_profile']='OLD'
    A.C.write(tmp_path/'CAL_LOGO_REPORT.json',report)
    seal['assets']['CAL_LOGO_REPORT.json']=A.C.sha(tmp_path/'CAL_LOGO_REPORT.json')
    A.C.write(tmp_path/'MODEL_SEAL.json',seal)
    with pytest.raises(ValueError,match='newly fitted'):A._load_models(tmp_path)


def test_ordered_all238_streaming_join_and_e_worker_separation_no_pose_rows():
    packets=[];comparisons=[];predictions=[];records=[];indices=[];ids=[]
    for i in range(238):
        identifier=f'SYNTHETIC:木{i}';partition='CAL' if i<46 else 'TEST' if i<95 else 'REMAINDER'
        ids.append(identifier)
        packet=dict(identifiers=dict(id=identifier,partition=partition),source_contract=dict(ready=False))
        worker=dict(id=identifier,partition=partition,arms={name:dict(accepted=False,pose_available=False) for name in A.NEW_ARMS})
        row={**worker,'arms':{'SAME_FROZEN_E':dict(accepted=i%2==0),**worker['arms']},
             'e_comparison_parent_only_no_numeric_exposure':True,'legacy_e_pose_semantics':{'comparison_only':True}}
        comparison=dict(id=identifier,partition=partition,prediction=row['arms']['SAME_FROZEN_E'])
        packets.append(packet);comparisons.append(comparison);predictions.append(row)
        records.append(dict(id=identifier,partition=partition,ready=False,positive_packet_sha256=A.digest(packet)))
        indices.append(dict(id=identifier,partition=partition,arm_count=4,canonical_row_sha256=A.digest(worker)))
    assert len(list(A.joined_rows(iter(packets),iter(comparisons),iter(predictions),records,indices,ids)))==238
    changed=deepcopy(predictions);changed[-1]['arms'][A.NEW_ARMS[0]]['accepted']=True
    with pytest.raises(ValueError,match='canonical worker'):
        list(A.joined_rows(packets,comparisons,changed,records,indices,ids))
    with pytest.raises(ValueError,match='All238 ordered'):
        list(A.joined_rows(packets,comparisons[:-1],predictions,records,indices,ids))
    changed=deepcopy(indices);changed[0]['partition']='TEST'
    with pytest.raises(ValueError,match='partition'):
        list(A.joined_rows(packets,comparisons,predictions,records,changed,ids))


def test_new_model_binding_even_when_current_frame_has_zero_active_lines(genuine):
    value,comparison,row=deepcopy(genuine)
    for policy in A.POLICIES:
        value['source_contract']['model_availability'][policy]=False
    # Strong mismatch is detected before a zero-evidence frame could make
    # missing current availability/digest binding invisible.
    seen=[]
    policy=A.POLICIES[0];name=A.FAMILIES[0]
    plan=row['arms'][A.NEW_ARMS[0]]['new_geometry_edge_cost_plans'][name]
    models={policy:dict(available=True)}
    A._plan_binding_checks(value,plan,policy,name,lambda ok,reason:seen.append((bool(ok),reason)),models)
    assert any(not ok and 'availability exact' in reason for ok,reason in seen)


def test_ready_own_coarse_fallback_makes_no_new_edge_likelihood_claim(genuine, monkeypatch):
    from .test_evaluate_clean_fresh_BI_poses_v4_32 import V
    value,comparison,row=deepcopy(genuine)
    clean,cores=V.build_clean_native_cores(value)
    for policy,arms in A.POLICY_ARMS.items():
        for arm in arms:
            entry=row['arms'][arm];family=clean['families']['ORIGINAL_WD'];pose=family['coarse_pose']
            entry.update(accepted=False,R=pose['R'],t=pose['t'],WD_family='ORIGINAL_WD',
                actual_pose=V._actual_pose(family,pose['R'],pose['t']),
                centroid=family['coarse_model_center_camera_xyz'],R_physical=pose['R_physical'],
                physical_registration_Q=family['physical_registration_Q'],
                selected_endpoint_origin='OWN_INDEPENDENT_INITIAL_COARSE_FALLBACK',
                fallback_audit=dict(own_independent_original_coarse_only=True,old_method_or_baseline_fallback_used=False,cached_pose_inputs=0),
                solution=dict(accepted=False,R=pose['R'],t=pose['t'],objective=None,native_costs=[],regularizer_costs=[]))
            entry['candidate_selection'].update(candidates=[],selected_name=None,selected_family=None)
            entry['readout']=V._readout(value,clean,cores[policy]['ORIGINAL_WD'],entry)
    monkeypatch.setattr(A.INDEPENDENT,'recompute_native',forbidden)
    audit=A.audit_row(value,comparison,row)
    assert audit['accepted']==0 and audit['fallback']==4 and not audit['no_pose']
    assert all(c['passed'] for c in audit['checks'])


def test_S_projection_and_unknown_region_readout_are_distinct_model_information():
    source=dict(states=['S','U','O','T','B','I','B','I'],initial_self_hidden=[True]+[False]*7,
        K=[[100.,0.,50.],[0.,100.,50.],[0.,0.,1.]],
        visible_measurements=[dict(native_corner=k,xy=[10.+k,20.],source='RGB') for k in range(4,8)])
    xyz=A.C.cuboid(1.,.5,.7);family=dict(native_xyz=xyz.tolist())
    entry=dict(R=np.eye(3).tolist(),t=[0.,0.,4.],accepted=True)
    hom=(xyz+entry['t'])@np.asarray(source['K']).T;points=[None]*8;points[0]=(hom[0,:2]/hom[0,2]).tolist()
    for r in source['visible_measurements']:points[r['native_corner']]=r['xy']
    sources=['MODEL_PNP_PROJECTION_ONLY_NO_S_IMAGE_TERM']+['APPROXIMATE_REGION_ONLY_NO_POINT_CORRESPONDENCE']*3+['RGB']*4
    readout=dict(states=source['states'],current_complete_BI_ids=list(range(4,8)),current_active_owner_ids=[],
        u_regions=[{'native_corner':1,'approximate':True}],U_range_only=True,S_image_raw_2D_or_region_terms=0,
        new_full_2D_corner_measurements=0,OT_solved=False,OT_unique_solution_claimed=False,used_in_accepted_pose=True,
        accepted_fresh_or_refined_pose=True,S_model_projection_ids=[0],reported_points=points,point_sources=sources)
    packet=dict(clean_input=source,physics=dict(u_regions=readout['u_regions']));entry['readout']=readout
    native=dict(native_costs=[dict(kind='COMPLETE_BI',native_corner=k) for k in range(4,8)])
    checks=[];check=lambda ok,reason:checks.append(bool(ok))
    A._readout_checks(packet,entry,family,native,dict(new_owner_ids=[]),check)
    assert all(checks)
    readout['reported_points'][1]=[12.,34.];checks.clear()
    A._readout_checks(packet,entry,family,native,dict(new_owner_ids=[]),check)
    assert not all(checks)


def test_unavailable_both_current_policy_models_keep_empty_typed_evidence_and_original_baseline():
    from .test_evaluate_clean_fresh_BI_poses_v4_32 import packet,V
    _,_,_,value=packet(unavailable=True);result=V.solve_worker_packet(value);ids=value['identifiers']
    comparison=dict(id=ids['id'],partition=ids['partition'],prediction=dict(accepted=False),
        legacy_e_pose_semantics=dict(comparison_only=True,numeric_worker_exposure=False))
    row={**result,'id':ids['id'],'partition':ids['partition'],'image_sha256':ids['image_sha256'],
        'legacy_e_pose_semantics':comparison['legacy_e_pose_semantics'],'e_comparison_parent_only_no_numeric_exposure':True}
    row['arms']={'SAME_FROZEN_E':comparison['prediction'],**row['arms']}
    models={policy:dict(available=False)for policy in A.POLICIES}
    audit=A.audit_row(value,comparison,row,models=models)
    assert audit['accepted']+audit['fallback']==4
    assert all(check['passed']for check in audit['checks'])
    assert all(not p['active_groups'] and not p['new_owner_ids'] for arm in A.NEW_ARMS
        for p in row['arms'][arm]['new_geometry_edge_cost_plans'].values())
