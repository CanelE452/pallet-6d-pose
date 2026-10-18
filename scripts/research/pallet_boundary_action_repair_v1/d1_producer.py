"""One actual automatic state/observation/e producer for real and synthetic RGB.

The same callable executes both domains. Supervision is neither a function
argument nor an import. Original e uses P1 and two-WD0 seeds, numeric G10 family
selection, typed G13 refinement, then its *G10-selected* family readout.
"""
from __future__ import annotations

from collections import Counter
from copy import deepcopy
from pathlib import Path
import time
import numpy as np

from scripts.research.pallet_statewise_end_to_end_recovery_v1.common import read,sha,write,save_rows
from .pipeline_contract import validate_input


def load_models(device='cuda'):
    """Load unchanged descriptors without original experiment/scorer loaders."""
    import torch,cv2
    from scripts.research.pallet_visibility_curriculum_20261013_v1 import common as C
    from scripts.research.pallet_visibility_curriculum_20261013_v1 import observe_remainder_rgb_v4_14 as O
    from scripts.research.pallet_visibility_curriculum_20261013_v1.statewise_masks_v2 import StatewiseMaskProvider
    from scripts.research.pallet_visibility_curriculum_20261013_v1.state_visibility_v2 import load_state_model
    from scripts.research.pallet_visibility_curriculum_20261013_v1.statewise_uncertainty_calibration_v4 import load_uncertainty_model
    from scripts.research.pallet_visibility_curriculum_20261013_v1.truncation_interval_v4_2 import load_truncation_interval_model
    cv2.setNumThreads(1);torch.set_num_threads(1)
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    torch.backends.cudnn.allow_tf32=False;torch.backends.cuda.matmul.allow_tf32=False
    study=C.DOC/'nonsevere_v1/statewise_joint_ot_20261014_v2'
    runtime_path=study/'RUNTIME_CONFIG_V3.json'; runtime=read(runtime_path)
    paths={key:study/runtime[key]for key in ('state_model','OT_model','mask_descriptor','boundary_mask_config')}
    bindings={str(runtime_path):sha(runtime_path)}
    for key,path in paths.items():
        if sha(path)!=runtime['bindings_sha256'][key]:raise ValueError('Frozen descriptor changed: '+key)
        bindings[str(path)]=sha(path)
    for key in ('state_model','mask_descriptor'):
        desc=read(paths[key]);arr=paths[key].parent/desc['arrays_file']
        if sha(arr)!=desc['arrays_sha256']:raise ValueError('State head arrays changed')
        bindings[str(arr)]=sha(arr)
        for role in ('backbone','pallet_segmenter'):
            b=desc[role];p=C.ROOT/b['weights_repo_relative']
            if sha(p)!=b['weights_sha256']:raise ValueError('Frozen '+role+' changed')
            bindings[str(p)]=sha(p)
    boundary=read(paths['boundary_mask_config']);checkpoint=(paths['boundary_mask_config'].parent/boundary['mobile_sam_checkpoint']).resolve()
    if sha(checkpoint)!=boundary['model_sha256']['mobile_sam_checkpoint']:raise ValueError('SAM checkpoint changed')
    bindings[str(checkpoint)]=sha(checkpoint)
    for p in (O.UNCERTAINTY_MODEL,O.INTERVAL_MODEL,O.LEGACY_PROTOCOL):bindings[str(p)]=sha(p)
    bundle=dict(head=load_state_model(str(paths['state_model']),device),ot=read(paths['OT_model']),
        tight=O.RecordingMaskProvider(StatewiseMaskProvider.from_configs(read(paths['mask_descriptor']),paths['boundary_mask_config'],device)),
        full=O.MASK.GeometryPromptSAMCapture.from_configs(read(paths['mask_descriptor']),paths['boundary_mask_config'],device),
        model=load_uncertainty_model(O.UNCERTAINTY_MODEL),interval=load_truncation_interval_model(O.INTERVAL_MODEL),
        config=read(O.LEGACY_PROTOCOL)['solver_configuration'],device=device,bindings=bindings)
    if bundle['config']['prior_weight']!=1. or bundle['config']['bias_strength']!=1.:
        raise ValueError('Original e P1/bias1 seed configuration required')
    return bundle


def observe(frame,bundle):
    """Domain-independent generalization of original v4_14 observer.

    Only metadata acceptance is generalized; all image operators, numerical
    settings, original CAL covariance inheritance, and gate order are unchanged.
    """
    import cv2
    from scripts.research.pallet_visibility_curriculum_20261013_v1 import common as C
    from scripts.research.pallet_visibility_curriculum_20261013_v1 import observe_remainder_rgb_v4_14 as O
    from scripts.research.pallet_visibility_curriculum_20261013_v1.statewise_joint_api_v2 import registered_geometry
    validate_input(frame);image=cv2.imread(frame['image'])
    if image is None or list(image.shape[:2])!=frame['raw_hw']:raise ValueError('Exact actual RGB canvas required')
    geometry=registered_geometry(frame['raw_hw'],frame['K'],frame['native_xyz'],frame['initial_pose'])
    gated=deepcopy(frame['initial_points'])
    for k,hidden in enumerate(geometry['initial_self_hidden']):
        if hidden:gated[k]=[-1.,-1.]
    bundle['tight'].last=None
    api=O.StatewiseIntervalV4_2(bundle['head'],bundle['ot'],bundle['tight'],
        uncertainty_model=bundle['model'],truncation_model=bundle['interval'],
        config=dict(prediction_source='N3_SUBPIX',record_id=frame['id'],device=bundle['device'],solver=bundle['config']))
    source=api.observe(image,gated,frame['K'],frame['native_xyz'],frame['initial_pose'],crop_transform=frame['crop_transform'])
    if source['initial_geometry']!=C.finite(geometry):raise ValueError('Observer changed initial registration')
    tight=bundle['tight'].last
    if tight is None:
        zero=np.zeros(image.shape[:2],np.float32)
        tight=dict(pallet_mask=zero,state_pallet_mask=zero.copy(),provenance=dict(reason='frozen_API_no_gated_RGB_queries',mask_forward_calls=0))
    prompt=O.MASK.build_geometry_prompt(geometry,source['original_raw_points'],source['states'])
    bundle['full'].predictor=bundle['tight'].delegate.predictor
    full=bundle['full'](image,prompt=prompt,state_pallet_mask=tight['state_pallet_mask'])
    if not np.array_equal(full['state_pallet_mask'],tight['state_pallet_mask']):raise ValueError('Tiny mask changed')
    original_fit=O.apply_calibrated_measurements(source['frozen_visible_measurements'],bundle['model'],states=source['states'],
        nonself_scale_px=source['ot_domains']['nonself_prediction_scale_px'])
    source['calibrated_visible_measurements']=deepcopy(original_fit)
    correction,stages=O.recover_stages(image,source,tight['pallet_mask'],full['pallet_mask_clean'])
    fit=O.fixed_fit_measurements(original_fit,correction)
    regions=O.build_u_regions(source,bundle['interval'])
    partial=O.observe_shared_supported_b_lines(correction,geometry,source['states'],fit,
        localization_model=bundle['interval'],nonself_scale_px=source['ot_domains']['nonself_prediction_scale_px'],
        image_binding=dict(sha256=frame['image_sha256'],image_hw=geometry['image_hw']))
    row=dict(schema='D1_FRESH_AUTOMATIC_RGB_OBSERVATIONS_V1',id=frame['id'],partition=frame['partition'],
        split=frame['split'],session=frame['session'],cluster_id=frame['cluster_id'],image=frame['image'],
        image_sha256=frame['image_sha256'],states=source['states'],runtime={'initial_geometry':geometry},
        _original_source=source,old_correction=source['visible_correction'],correction=correction,
        frozen_BI_fit=fit,original_CAL_fit=original_fit,partial_boundary_observation=partial,
        u_region_observation=regions,recovery_stage_audit=stages,
        full_mask_provenance=full['provenance'],tight_mask_provenance=tight['provenance'],
        contract=dict(GT_inputs=False,actual_production_state_classifier_stage=True,external_O_placeholder=False,
            same_callable_for_both_domains=True,pose_solve_calls=0,S_image_queries=0))
    arrays={k:v for k,v in full.items()if k!='provenance'};arrays['pallet_mask_tight']=tight['pallet_mask']
    return C.finite(row),arrays


def _physical_candidate(pose,family,name,source):
    if pose is None:return None
    if 'R' in pose:R=np.asarray(pose['R']);t=np.asarray(pose['t'])
    elif pose.get('available'):R=np.asarray(pose['R_cf']);t=np.asarray(pose['centroid'])
    else:return None
    if R.shape!=(3,3)or t.shape!=(3,)or not np.isfinite(R).all()or not np.isfinite(t).all():return None
    return dict(candidate_id=name,source=source,R_physical=(R@np.asarray(family['physical_registration_Q'])).tolist(),
        centroid=t.tolist(),original_native_family=family['name'])


def solve_e(row,frame,mask,config,model,interval):
    """Execute the original hierarchical e, preserving actual visited endpoints."""
    from scripts.research.pallet_visibility_curriculum_20261013_v1 import common as C
    from scripts.research.pallet_visibility_curriculum_20261013_v1 import statewise_partial_boundary_u_solver_v4_6 as P
    from scripts.research.pallet_visibility_curriculum_20261013_v1 import statewise_two_wd_family_v4_9 as W
    from scripts.research.pallet_visibility_curriculum_20261013_v1 import statewise_global_outline_solver_v4_10 as G
    from scripts.research.pallet_visibility_curriculum_20261013_v1 import face_aware_partial_bi_observations_v4_13d as FACE
    from scripts.research.pallet_visibility_curriculum_20261013_v1 import evaluate_visibility_pose_v4_8 as SPEC
    from scripts.research.pallet_statewise_correspondence_diagnosis_v1 import typed_replay as T
    before=C.finite(row);geometry=row['runtime']['initial_geometry'];spec=SPEC._spec(row,config)
    prior=P.solve_boundary_u_region_joint_ot_pnp(**spec);prior.pop('parent_baseline_solution',None)
    cfg=deepcopy(config);cfg['prior_weight']=0.;spec=SPEC._spec(row,cfg)
    wd=W.solve_two_wd_families(spec['xyz'],spec['K'],spec['states'],spec['visible'],spec['domains'],spec['image_hw'],
        spec['coarse_pose'],initializations=spec['initializations'],config=cfg,
        boundary_observations=spec['boundary_observations'],u_regions=spec['u_regions'],solver_callable=P.solve_boundary_u_region_joint_ot_pnp)
    families={f['name']:f for f in wd['families']};ledger=[]
    prior_R,prior_t,_=T.choose_readout(prior,spec['coarse_pose'])
    outline=G.observe_global_outline(mask,spec['image_hw'],image_binding=dict(image_sha256=frame['image_sha256'],D1_current_RGB=True))
    import cv2
    image=cv2.imread(frame['image'])
    face=FACE.observe_face_aware_partial_bi(image,mask,geometry,row['states'],row['frozen_BI_fit'],
        raw_native_points=row['correction']['raw_points'],localization_model=interval,
        nonself_scale_px=row['partial_boundary_observation']['unchanged_nonself_prediction_scale_px'],
        image_binding=dict(sha256=frame['image_sha256'],image_hw=frame['raw_hw']))
    legacy=spec['boundary_observations'];merged={r['native_corner']:deepcopy(r)for r in legacy}
    for item in face['boundary_observations']:merged[item['native_corner']]=deepcopy(item)
    typed=[merged[k]for k in sorted(merged)]
    def stage(which,boundary):
        def solve(xyz,K,states,visible,domains,hw,coarse,**kw):
            family=next(f for f in families.values()if np.allclose(f['native_xyz'],xyz,atol=1e-12,rtol=0))
            solutions=[family['solution']];initial=[]
            if family['name']=='ORIGINAL_WD':solutions.append(prior)
            else:initial=[dict(R=(np.asarray(prior_R)@W.RY90.T).tolist(),t=prior_t,name='PRIOR_PHYSICAL_POSE_TRANSFORMED_SEED_ONLY')]
            backend=T._backend('statewise_global_outline_solver_v4_10' if which=='G10' else 'statewise_partial_BI_outline_solver_v4_13',
                T._ledger_sink(ledger,which,family['name']))
            return backend.refine_global_outline_joint_pose(xyz,K,states,visible,domains,hw,coarse,
                outline_observation=outline,initial_solutions=solutions,initializations=initial,
                boundary_observations=boundary,u_regions=spec['u_regions'],config=kw['config'])
        return W.solve_two_wd_families(spec['xyz'],spec['K'],spec['states'],spec['visible'],spec['domains'],spec['image_hw'],
            spec['coarse_pose'],config=cfg,boundary_observations=boundary,u_regions=spec['u_regions'],solver_callable=solve)
    g10=stage('G10',legacy)
    g13=stage('G13',typed)if face['boundary_observations']else deepcopy(g10)
    name=g10['selected_family'];selected=next(f for f in g13['families']if f['name']==name)
    e0=T.actual_pose(selected['solution'],selected);e0['source']='D1_E0_REGENERATED_SAME_ORIGINAL_HIERARCHY'
    finalfamilies={f['name']:{k:deepcopy(f[k])for k in ('name','native_xyz','cf_extents','physical_registration_Q','coarse_pose','initializations')}for f in g13['families']}
    packet=dict(id=frame['id'],partition=frame['partition'],image=frame['image'],image_sha256=frame['image_sha256'],
        solver_input=dict(xyz=spec['xyz'],K=spec['K'],states=spec['states'],visible=spec['visible'],domains=spec['domains'],
            image_hw=spec['image_hw'],coarse_pose=spec['coarse_pose'],boundary_observations=typed,u_regions=spec['u_regions'],
            outline_observation=outline,config=cfg),original_e_legacy_boundary_observations=legacy,
        original_e_stage1_family_seed_bank={n:dict(family=f)for n,f in finalfamilies.items()},registered_family=finalfamilies[name],
        provenance=dict(GT_inputs=False,original_mask_binding=None,D1_auto_pipeline=True))
    candidates=[dict(candidate_id='N0',source='N0',R_physical=frame['initial_pose']['R_physical'],centroid=frame['initial_pose']['centroid']),
        dict(candidate_id='E0_REGENERATED',source='E0',R_physical=e0['R_physical'],centroid=e0['centroid'])]
    for which,records in [('P1',[dict(families['ORIGINAL_WD'],solution=prior)]),('WD0',wd['families']),('G10',g10['families']),('G13',g13['families'])]:
        for fam in records:
            sol=fam['solution']
            if sol.get('accepted'):
                item=_physical_candidate(sol,fam,which+'/'+fam['name']+'/selected','D1_'+which)
                if item:candidates.append(item)
            for j,sample in enumerate(sol.get('feasible_pose_samples',[])):
                item=_physical_candidate(sample,fam,which+'/'+fam['name']+'/sample/'+str(j),'D1_'+which)
                if item:candidates.append(item)
    for j,endpoint in enumerate(ledger):
        item=_physical_candidate(endpoint,finalfamilies[endpoint['family']],endpoint['stage']+'/'+endpoint['family']+'/endpoint/'+str(j),'D1_'+endpoint['stage']+'_ENDPOINT')
        if item:
            item['execution_record']=endpoint['original_endpoint_record'];candidates.append(item)
    if row!=before:raise ValueError('Original e mutated immutable observations')
    return C.finite(dict(id=frame['id'],packet=packet,families=finalfamilies,candidates=candidates,fallback=frame['initial_pose'],
        e0_regenerated=e0,selected_family=name,stage_summaries=dict(G10_selected=g10['selected_family'],G13_selected=g13['selected_family'],
            G13_executed=bool(face['boundary_observations']),G13_cache_reuse_without_typed=not bool(face['boundary_observations'])),
        face_observation=face,endpoint_journal=ledger,old_full_run_reused=False,
        proof=dict(e0_independently_executed=True,N0_copied_as_E0=False,history_only_union=False,
            new_B_or_A_candidates=False,reference_reads=0,same_procedure_for_both_domains=True)))
