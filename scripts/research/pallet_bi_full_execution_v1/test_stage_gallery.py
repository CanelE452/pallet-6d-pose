"""Constructed metadata only: no dataset image/reference/scorer/solve/fit."""
from copy import deepcopy
from pathlib import Path
from tempfile import TemporaryDirectory
from . import stage_gallery as G


def tests():
    def metric(success=False,t=1.,r=2.):return dict(success=success,pose_available=True,translation_cm=t,rotation_deg=r)
    def row(fid,partition='TEST',category='BOTH_FAILURE'):
        selected={str(s):dict(transition=category,metric=metric(category in ('RECOVERY','BOTH_SUCCESS')),
            baseline_metric=metric(category in ('DAMAGE','BOTH_SUCCESS')),failure_stage='SELECTION_LOSS',fallback=False)for s in G.SEEDS}
        return dict(arm='CONSTRUCTED',id=fid,partition=partition,cluster_id='REC_001',selected=selected)
    a=row('a');a['selected'][str(G.SEEDS[0])]['transition']='RECOVERY';a['selected'][str(G.SEEDS[1])]['transition']='DAMAGE'
    b=row('b');b['selected'][str(G.SEEDS[2])]['transition']='RECOVERY'
    normal=row('normal',category='BOTH_SUCCESS');normal['cluster_id']='REC_002'
    cal=row('private_CAL','CAL','DAMAGE')
    chosen,transition_ids=G.choose_cases([a,b,normal,cal])
    assert transition_ids==['a','b']
    assert not any(key[1]=='private_CAL'for key in chosen)
    assert {v['category']for v in chosen[('CONSTRUCTED','a')]} >= {'RECOVERY','DAMAGE'}
    assert ('CONSTRUCTED','normal')in chosen
    selection=G.choose_cases([cal,normal,b,a]);assert selection==(chosen,transition_ids)
    fixed=G.choose_cases([a,b,normal,cal],[dict(arm='CONSTRUCTED',id='normal',reason='PF_new_good3')])[0]
    assert any(v['category']=='FIXED_POSTHOC_DIAGNOSTIC/PF_new_good3'for v in fixed[('CONSTRUCTED','normal')])
    for fid in ('private_CAL','absent'):
        try:G.choose_cases([a,b,normal,cal],[dict(arm='CONSTRUCTED',id=fid,reason='PF_new_good3')])
        except ValueError:pass
        else:raise AssertionError('Missing/CAL fixed diagnostic example accepted')
    paired=row('paired')
    alternatives={'paired':dict(metrics={'ALT_s'+str(s):metric(True)for s in G.SEEDS})}
    recast=G.comparison_coverage([paired],alternatives,'ALT')
    assert all(recast[0]['selected'][str(s)]['transition']=='DAMAGE'for s in G.SEEDS)
    assert all(paired['selected'][str(s)]['transition']=='BOTH_FAILURE'for s in G.SEEDS)
    packet=dict(packet=dict(solver_input=dict(states=['B','I','S'],visible=[
        dict(native_corner=0,state='B',xy=[2,3],covariance_px2=[[1,0],[0,1]],physical_source_ids=['actual-pixel-a']),
        dict(native_corner=1,state='I',xy=[999,999],covariance_px2=[[1,0],[0,1]],physical_source_ids=['archive-only']),
        dict(native_corner=2,state='S',xy=[4,5],covariance_px2=[[1,0],[0,1]])],boundary_observations=[
        dict(native_corner=1,state='I',lines=[dict(normal_unit_xy=[1,0],offset_px=-4,normal_variance_px2=2,
            support_tangent_interval_px=[10000,10001],physical_source_id='actual-line',
            provenance=dict(line_record=dict(observed_points=[[4,1],[4,2]],inlier_mask=[True,False])))])])))
    observations=G.observation_records(packet)
    assert [r['native_corner']for r in observations['full_points']]==[0]
    assert 'xy'not in observations['revoked_archives'][0]
    assert observations['normal_factors'][0]['raw_image_support_xy']==[[4,1]]
    assert observations['S_point_observations']==0
    changed=deepcopy(packet);changed['packet']['solver_input']['visible'][1]['xy']=[-1000,2222]
    other=G.observation_records(changed)
    assert observations['full_points']==other['full_points']and observations['normal_factors']==other['normal_factors']
    assert observations['revoked_archives'][0]['revoked_archive_payload_sha256']!=other['revoked_archives'][0]['revoked_archive_payload_sha256']
    segment=G.line_segment([1,0],-4,8,6);assert segment.tolist()==[[4.,0.],[4.,6.]]
    # The display crop is bound to saved initialization/automatic pixels only.
    # No actual image, final prediction, score or reference is read here.
    cube=[[x,y,z]for x in (-.5,.5)for y in (-.5,.5)for z in (-.5,.5)]
    eye=[[1,0,0],[0,1,0],[0,0,1]]
    initial_frame=dict(raw_hw=[480,640],K=[[400,0,320],[0,400,240],[0,0,1]],
        initial_pose=dict(available=True,R_physical=eye,centroid=[0,0,4]),
        initial_points=[[250,170],[390,170],[390,310],[250,310]]*2)
    initial_result=dict(families=dict(ORIGINAL_WD=dict(native_xyz=cube,physical_registration_Q=eye)),
        packet=dict(solver_input=dict(outline_observation=dict(observed_boundary_samples_xy=[[270,190],[370,290]]))))
    roi=G.display_roi(initial_frame,initial_result);assert roi['mode']=='PREDICTION_ROI'
    x0,y0,x1,y1=roi['xyxy_raw'];assert 0<x0<250<390<x1<640 and 0<y0<170<310<y1<480
    altered_frame=deepcopy(initial_frame);altered_result=deepcopy(initial_result)
    altered_frame.update(reference_pose=dict(R=eye,t=[999,999,999]),scored_error=1e6,
        selected_pose=dict(R_physical=eye,centroid=[-999,-999,999]))
    altered_result.update(reference_pose=dict(R=eye,t=[-999,999,999]),selected_candidate_id='bad-final',
        candidates=[dict(R_physical=eye,centroid=[999,-999,999])])
    assert G.display_roi(altered_frame,altered_result)==roi
    outside=deepcopy(initial_result);outside['packet']['solver_input']['outline_observation']['observed_boundary_samples_xy'].append([40,40])
    outside_roi=G.display_roi(initial_frame,outside)
    assert outside_roi['mode']=='PREDICTION_ROI'and outside_roi['xyxy_raw']==roi['xyxy_raw']
    assert outside_roi['saved_automatic_outline_outside_ROI_count']==1
    assert outside_roi['saved_automatic_outline_outside_ROI']and not outside_roi['outline_is_GT']
    behind=deepcopy(initial_frame);behind['initial_pose']['centroid']=[0,0,-4]
    assert G.display_roi(behind,initial_result)['reason']=='INVALID_OR_BEHIND_CAMERA_N0_PROJECTION'
    missing=deepcopy(initial_frame);missing.pop('initial_points')
    assert G.display_roi(missing,initial_result)['mode']=='FULL_CONTEXT'
    assert roi['context_always_retained']and not roi['GT_or_score_or_selected_pose_used']
    pose=dict(available=True,R_physical=[[1,0,0],[0,1,0],[0,0,1]],centroid=[0,0,4])
    rank_input=dict(selected_score=2.,actions=[dict(candidate_id='selected',source_aliases=[],score=2.),
        dict(candidate_id='old_dedup',source_aliases=['good_alias'],score=1.),
        dict(candidate_id='good',source_aliases=[],score=1.)])
    rank_features=dict(candidates=[dict(candidate_id='old_dedup',actual_pose=pose)])
    rank,overlay=G.saved_action_rank(dict(good_eligible_ids=['good','good_alias']),rank_input,rank_features)
    assert rank['best_good_action_id']=='old_dedup'and rank['best_good_action_rank']==2
    assert rank['best_good_logit_minus_selected']==-1 and overlay==pose
    assert rank['posthoc_good_actions']==2 and not rank['logit_is_calibrated_confidence']
    G.public_safe(rank)
    result=dict(candidates=[dict(candidate_id='G13/ORIGINAL_WD/selected',R_physical=pose['R_physical'],centroid=pose['centroid'])],
        selected_family='ORIGINAL_WD',stage_results=dict(G13=dict(families=[dict(name='ORIGINAL_WD',solution=dict(objective=3.,minimum_joint_slack=.2,
            objective_components=dict(full=1.,partial=2.,projected_points=[[1,2]],R=pose['R_physical'])))])))
    families=G.family_readouts(result)
    assert families['ORIGINAL_WD']['public']['stored_stage_costs_and_constraints']['objective']==3.
    assert families['SWAPPED_WD']['pose']is None and not families['SWAPPED_WD']['public']['available']
    G.public_safe({key:value['public']for key,value in families.items()})
    for forbidden in ('reference','R_physical','centroid','references'):
        try:G.public_safe(dict(nested={forbidden:pose}))
        except ValueError:pass
        else:raise AssertionError('Public JSON pose/reference guard failed')
    with TemporaryDirectory(prefix='stage-gallery-metadata-')as temporary:
        root=Path(temporary);d1=root/'baseline';pred=root/'new';ed=root/'evaluation';cd=root/'coverage'
        no_pose=dict(available=False);base=[];new=[];features=[];scored=[];coverage=[]
        for i in range(238):
            partition='TEST'if i<49 else'REMAINDER'if i<192 else'CAL';fid=f'constructed_{i:04d}'
            identity=dict(id=fid,partition=partition,session='constructed',cluster_id='REC_'+str(i%6))
            oldarms={'D1_H1_MLP_s'+str(s):dict(actual_pose=no_pose)for s in G.SEEDS}
            newarms={'CONSTRUCTED_s'+str(s):dict(actual_pose=no_pose,selected_candidate_id=None)for s in G.SEEDS}
            base.append(dict(identity,predictions=oldarms));new.append(dict(identity,predictions=newarms));features.append(dict(identity,candidates=[]))
            metrics={name:dict(success=False,pose_available=False,translation_cm=None,rotation_deg=None)for name in list(oldarms)+list(newarms)}
            scored.append(dict(identity,poses={name:no_pose for name in metrics},metrics=metrics))
            coverage.append(dict(identity,arm='CONSTRUCTED',generated_good=False,eligible_good=False,selected={str(s):dict(
                transition='BOTH_FAILURE',metric=metrics['CONSTRUCTED_s'+str(s)],baseline_metric=metrics['D1_H1_MLP_s'+str(s)],
                failure_stage='GENERATED_ABSENT',fallback=False,selected_candidate_id=None)for s in G.SEEDS}))
        for folder,values in ((d1,base),(pred,new)):
            G.U.save(folder/'PREDICTIONS.jsonl.gz',values)
            G.U.write(folder/'PREDICTION_SEAL.json',dict(complete=True,predictions_sha256=G.U.sha(folder/'PREDICTIONS.jsonl.gz')))
        G.U.save(root/'FEATURES.jsonl.gz',features);G.U.write(root/'FEATURE_SEAL.json',dict(complete=True,head_features_sha256=G.U.sha(root/'FEATURES.jsonl.gz')))
        manifest=dict(arms=dict(CONSTRUCTED=dict(backend='original_union',features=str(root/'FEATURES.jsonl.gz'),feature_seal=str(root/'FEATURE_SEAL.json'),prediction_dir=str(pred))))
        G.U.write(root/'MANIFEST.json',manifest)
        G.U.save(ed/'SCORED_ROWS.jsonl.gz',scored)
        prediction_bindings=[dict(path=str(folder/'PREDICTIONS.jsonl.gz'),sha256=G.U.sha(folder/'PREDICTIONS.jsonl.gz'))for folder in (d1,pred)]
        G.U.write(ed/'EVALUATION_SEAL.json',dict(complete=True,prediction_bindings=prediction_bindings,artifacts={'SCORED_ROWS.jsonl.gz':G.U.sha(ed/'SCORED_ROWS.jsonl.gz')}))
        G.U.save(cd/'BANK_COVERAGE_ROWS.jsonl.gz',coverage)
        _,new_binding=G.predictions(pred)
        G.U.write(cd/'INPUT_BINDINGS.json',dict(CONSTRUCTED=dict(features=dict(sha256=G.U.sha(root/'FEATURES.jsonl.gz')),
            feature_seal=dict(sha256=G.U.sha(root/'FEATURE_SEAL.json')),prediction=new_binding)))
        G.U.write(cd/'ANALYSIS_SEAL.json',dict(complete=True,manifest_sha256=G.U.sha(root/'MANIFEST.json'),artifacts={
            'BANK_COVERAGE_ROWS.jsonl.gz':G.U.sha(cd/'BANK_COVERAGE_ROWS.jsonl.gz'),'INPUT_BINDINGS.json':G.U.sha(cd/'INPUT_BINDINGS.json')}))
        args=(root/'MANIFEST.json',d1,ed,cd)
        validated=G.validate_inputs(*args);assert len(validated[3])==238
        # A completed COMMON can be the explicit paired baseline for F_BI.
        # This fixture uses one metadata arm on both sides, with no evaluation.
        ps=G.U.read(pred/'PREDICTION_SEAL.json');ps.update(schema='NEW_CAMPAIGN_H1_PREDICTIONS_V1',
            features_sha256=G.U.sha(root/'FEATURES.jsonl.gz'),feature_seal_sha256=G.U.sha(root/'FEATURE_SEAL.json'))
        (pred/'PREDICTION_SEAL.json').write_text(G.U.canonical(ps))
        matched=deepcopy(manifest);matched['comparison_baseline']=dict(matched['arms']['CONSTRUCTED'],arm='CONSTRUCTED')
        G.U.write(root/'MATCHED_MANIFEST.json',matched)
        matched_result=G.validate_inputs(root/'MATCHED_MANIFEST.json',d1,ed,cd)
        assert matched_result[5]['comparison_baseline']['arm']=='CONSTRUCTED'
        assert not matched_result[5]['same_manifest_as_coverage']
        # An exact source-sealed set of incorrect coverage labels is rejected
        # by recomputing only logical categories from already scored bits.
        coverage[0]['selected'][str(G.SEEDS[0])]['failure_stage']='SUCCESS'
        bad=root/'bad_coverage';G.U.save(bad/'BANK_COVERAGE_ROWS.jsonl.gz',coverage)
        G.U.write(bad/'INPUT_BINDINGS.json',G.U.read(cd/'INPUT_BINDINGS.json'))
        G.U.write(bad/'ANALYSIS_SEAL.json',dict(complete=True,manifest_sha256=G.U.sha(root/'MANIFEST.json'),artifacts={
            'BANK_COVERAGE_ROWS.jsonl.gz':G.U.sha(bad/'BANK_COVERAGE_ROWS.jsonl.gz'),'INPUT_BINDINGS.json':G.U.sha(bad/'INPUT_BINDINGS.json')}))
        try:G.validate_inputs(root/'MANIFEST.json',d1,ed,bad)
        except ValueError:pass
        else:raise AssertionError('Wrong coverage failure attribution accepted')
        unbound=root/'unbound_coverage';G.U.save(unbound/'BANK_COVERAGE_ROWS.jsonl.gz',coverage)
        G.U.write(unbound/'INPUT_BINDINGS.json',G.U.read(cd/'INPUT_BINDINGS.json'))
        G.U.write(unbound/'ANALYSIS_SEAL.json',dict(complete=True,manifest_sha256=G.U.sha(root/'MANIFEST.json'),artifacts={
            'BANK_COVERAGE_ROWS.jsonl.gz':G.U.sha(unbound/'BANK_COVERAGE_ROWS.jsonl.gz')}))
        try:G.validate_inputs(root/'MANIFEST.json',d1,ed,unbound)
        except ValueError as error:assert 'authenticated INPUT_BINDINGS'in str(error)
        else:raise AssertionError('Unbound old input-closure receipt accepted')
        # A completed prediction re-seal cannot change a scored input binding.
        changed=new[:237];G.U.save(root/'changed'/'PREDICTIONS.jsonl.gz',changed)
        G.U.write(root/'changed'/'PREDICTION_SEAL.json',dict(complete=True,predictions_sha256=G.U.sha(root/'changed'/'PREDICTIONS.jsonl.gz')))
        try:G.predictions(root/'changed')
        except ValueError:pass
        else:raise AssertionError('Partial238 prediction population accepted')
    return dict(complete=True,constructed_metadata_only=True,dataset_rows_read=0,images_opened=0,numerical_references_opened=0,
        new_metric_evaluations=0,new_individual_costs=0,new_pose_solves=0,new_fits=0,
        all3seed_transitions_selected=True,CAL_excluded=True,selection_order_invariant=True,
        explicit_posthoc_fixed_examples_included_and_invalid_or_CAL_rejected=True,
        saved_logits_rank_and_good_aliases_keep_original_tie_order=True,
        prediction_ROI_encloses_initial_queries_and_native_projection=True,
        prediction_ROI_independent_of_GT_scores_and_selected_pose=True,
        automatic_outline_outside_ROI_disclosed_without_changing_valid_prediction_crop=True,
        invalid_N0_falls_back_to_full_context=True,
        full_context_always_retained=True,
        revoked_archived2D_and_S_not_plotted_as_points=True,known_sources_retained=True,
        family_readouts_use_actual_saved_results=True,missing_family_is_explicit_NA=True,
        public_JSON_pose_reference_guard=True,full238_completed_bindings_checked=True,
        post_bank_V2_input_bindings_must_be_authenticated=True,
        bad_coverage_category_and_partial_prediction_rejected=True,source_sha256=G.U.sha(G.__file__))


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
    receipt=tests();G.U.write(args.output,receipt);print(G.U.canonical(receipt),flush=True)
