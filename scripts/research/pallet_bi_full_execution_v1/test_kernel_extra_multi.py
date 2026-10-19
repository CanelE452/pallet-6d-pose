"""Constructed two-parent union tests; no campaign dataset/reference/fit calls."""
from copy import deepcopy
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import mock
import numpy as np
from . import kernel_extra as K
from .test_kernel_extra import constructed,endpoint,_bound,_fresh_union

def test_multi():
    frame,result,old,config=constructed()
    pose=deepcopy(old['candidates'][0]['actual_pose']);shift=deepcopy(pose);shift['centroid'][0]+=.12
    distinct=deepcopy(pose);distinct['centroid'][0]+=.23
    c2=deepcopy(shift);c2['R_physical']=(np.asarray(shift['R_physical'])@np.diag([-1.,1.,-1.])).tolist()
    if 'physical_registration_Q'in c2:c2['R_cf']=(np.asarray(c2['R_physical'])@np.asarray(c2['physical_registration_Q']).T).tolist()
    pf=[endpoint(frame,pose,'PF_ACTION','POSEFIX_SUBPIX_PF_ANCHOR'),endpoint(frame,shift,'YOLO_ACTION','POSEFIX_SUBPIX_YOLO_ANCHOR')]
    material=[endpoint(frame,shift,'S4_PL/start0/update1','S4_PL'),endpoint(frame,distinct,'S4_PL/start0/update2','S4_PL'),
        endpoint(frame,c2,'S4_PL/start0/update3','S4_PL'),endpoint(frame,pose,'S4_PL/start0/rejected','S4_PL',False)]
    with TemporaryDirectory(prefix='constructed-kernel-multiple-')as tmp:
        root=Path(tmp);core=root/'core';folder=core/'CONSTRUCTED'/'synthetic'/'0000'
        _,lock,seal,_,_=_bound(frame,result,old,config,folder)
        K.U.write(core/'PRODUCER_LOCK.json',lock);K.U.write(folder/'FRAME_SEAL.json',seal)
        K.U.save(root/'INPUTS.jsonl.gz',[frame]);K.U.write(root/'CONFIG.json',config);K.U.write(root/'CAMPAIGN_PLAN.json',dict(constructed_only=True))
        specs=[]
        for name,values,methods in [('PF',pf,list(K.PF_METHODS)),('MATERIAL',material,['S4_PL'])]:
            p=root/(name+'.jsonl.gz');s=root/(name+'_SEAL.json');K.U.save(p,values)
            K.U.write(s,dict(complete=True,predictions_sha256=K.U.sha(p)))
            specs.append(dict(path=p.name,seal=s.name,methods=sorted(methods)))
        mp=root/'MULTIPLE.json';K.U.write(mp,dict(extra_inputs=specs))
        extras,parents,composite,multi=K.extra_inputs_for_frames([frame['id']],extra_inputs_manifest=mp)
        assert multi and extras[frame['id']]==pf+material and len(parents[frame['id']])==2
        assert [p['input_index']for p in composite['ordered_inputs']]==[0,1]
        assert all(p['producer_seal_sha256']for p in composite['ordered_inputs'])
        args=(root/'INPUTS.jsonl.gz',core,'CONSTRUCTED','synthetic',None,root/'CONFIG.json',root/'multi',root)
        with mock.patch.object(K.B,'compile_models',wraps=K.B.compile_models)as compiled,mock.patch.object(K.S,'candidate_scores',wraps=K.S.candidate_scores)as scored:
            receipt=K.batch_features(*args,root_GO=True,extra_inputs_manifest=mp)
        assert compiled.call_count==1 and scored.call_count==3,(compiled.call_count,scored.call_count)
        union=next(K.U.rows(root/'multi/HEAD_FEATURES.jsonl.gz'));info=union['kernel_extra']
        assert info['visited_extra_endpoints']==6 and info['new_unique_complete_actions']==3
        assert len(info['complete_action_duplicates'])==2 and len(info['complete_action_rejections'])==1
        assert info['old_individual_cost_evaluations']==0
        shifted=next(c for c in union['candidates']if c['candidate_id']=='YOLO_ACTION')
        assert {'YOLO_ACTION','S4_PL/start0/update1','S4_PL'}<=set(shifted['source_aliases'])
        assert any(c['candidate_id']=='S4_PL/start0/update3'for c in union['candidates'])
        for a,b in zip(old['candidates'],union['candidates']):
            assert a['features'][:26]==b['features'][:26]and a['features'][31:]==b['features'][31:]
            assert set(a['source_aliases'])<=set(b['source_aliases'])and a['source_baseline_aliases']==b['source_baseline_aliases']
        fresh=_fresh_union(frame,result,union,config)
        error=float(np.max(np.abs(np.asarray([c['features']for c in union['candidates']])-np.asarray([c['features']for c in fresh['candidates']]))))
        assert error<K.U.FEATURE_ATOL,error
        normalizer=dict(mean=[0.]*31,scale=[1.]*31,valid_train_counts=[1]*31)
        a,aa=K.D.inputs(union,normalizer,'H1');b,bb=K.D.inputs(fresh,normalizer,'H1')
        assert a.shape[1]==188 and [c['candidate_id']for c in aa]==[c['candidate_id']for c in bb]
        np.testing.assert_allclose(a,b,atol=2e-6,rtol=0)
        # Independent full bank all-family minima,not the optimized gap path.
        models=K.B.compile_models(result['packet'],config)
        scores=[K.S.candidate_scores(K._raw(c),models)for c in union['candidates']]
        minima=K._minima(scores,models)
        for mode in K.ACTIVE_MODES:np.testing.assert_allclose(info['union_minima'][mode],minima[mode],atol=1e-12,rtol=0)
        rs=K.U.read(root/'multi/frames/0000/SEAL.json')
        assert 'EXTRA_INPUT_FRAME_BINDINGS.jsonl.gz'in rs['artifacts']and rs['dependencies']['ordered_extra_inputs_frame_sha256']==info['ordered_extra_input_frame_sha256']
        assert next(K.U.rows(root/'multi/frames/0000/ALL_EXTRA_ENDPOINTS.jsonl.gz'))==pf[0]
        assert K.D.bound(root/'multi/HEAD_FEATURES.jsonl.gz',root/'multi/FEATURE_SEAL.json')==receipt['head_features_sha256']
        with mock.patch.object(K.S,'candidate_scores',side_effect=AssertionError('Resume recomputed costs')):
            assert K.batch_features(*args,root_GO=True,extra_inputs_manifest=mp)==receipt
        # Legacy single archive still executes the same union in one pass.
        K.U.save(root/'ALL.jsonl.gz',pf+material);K.U.write(root/'ALL_SEAL.json',dict(complete=True,predictions_sha256=K.U.sha(root/'ALL.jsonl.gz')))
        single=list(args);single[4]=root/'ALL.jsonl.gz';single[6]=root/'legacy'
        K.batch_features(*single,root_GO=True,extra_seal_path=root/'ALL_SEAL.json')
        legacy=next(K.U.rows(root/'legacy/HEAD_FEATURES.jsonl.gz'))
        assert [c for c in legacy['candidates']]==union['candidates']
        # Reversed parent policy and archive/producer metadata tampering must
        # fail before any feature model is compiled or candidate cost called.
        reversed_specs=[dict(v,path=str(root/v['path']),seal=str(root/v['seal']))for v in reversed(specs)]
        try:K.extra_inputs_for_frames([frame['id']],extra_inputs=reversed_specs)
        except ValueError:pass
        else:raise AssertionError('Material-before-PF input accepted')
        for index in range(2):
            path=root/specs[index]['path'];saved=path.read_bytes();path.write_bytes(saved+b'corruption')
            with mock.patch.object(K.B,'compile_models',side_effect=AssertionError('Tamper reached feature work')):
                try:K.batch_features(*args,root_GO=True,extra_inputs_manifest=mp)
                except ValueError:pass
                else:raise AssertionError('Changed parent archive accepted')
            path.write_bytes(saved)
        sp=root/specs[1]['seal'];saved=sp.read_bytes();value=K.U.read(sp);value['changed_parent_metadata']=True;sp.write_text(K.U.canonical(value))
        with mock.patch.object(K.B,'compile_models',side_effect=AssertionError('Changed receipt reached feature work')):
            try:K.batch_features(*args,root_GO=True,extra_inputs_manifest=mp)
            except ValueError:pass
            else:raise AssertionError('Changed parent seal accepted by completed union')
        sp.write_bytes(saved)
        # Parent ordering and selected rows are part of each frame's digest.
        assert parents[frame['id']][0]['selected_rows_payload_sha256']==K.U.payload_sha(pf)
        assert parents[frame['id']][1]['selected_rows_payload_sha256']==K.U.payload_sha(material)
    return dict(PASS=True,constructed_only=True,fixture_pose_optimizer_calls=result['operations']['pose_optimizer_calls'],
        actual_dataset_rows_read=0,actual_dataset_cost_or_pose_calls=0,real_reference_reads=0,scientific_forwards_or_fits=0,
        two_independently_verified_archives=True,PF_first_material_second=True,single_model_compile=True,new_unique_candidate_cost_calls=3,
        same_pose_cross_parent_aliases_preserved=True,distinct_C2_phase_preserved=True,all_raw_visited_rows_preserved=True,
        first26_old_scalars_and31_masks_exact=True,all_active_minima_and_full188_fresh_parity=True,feature_max_error=error,
        tampered_parent_bytes_and_resume_parent_receipt_rejected=True,legacy_single_archive_candidates_exact=True,
        completed_multi_resume_no_new_costs=True,source_sha256=K.U.sha(K.__file__))
if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args();K.cv2.setNumThreads(1);r=test_multi();K.U.write(a.output,r);print(K.U.canonical(r))
