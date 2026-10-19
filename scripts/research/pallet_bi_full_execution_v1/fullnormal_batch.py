"""Resumable all-row new full-normal hierarchy and original 188-input features."""
from __future__ import annotations
import argparse
from collections import Counter
from copy import deepcopy
from functools import lru_cache
from pathlib import Path
import time
import json

import cv2
import numpy as np

from scripts.research.pallet_bi_full_execution_v1 import qcov as Q, calibration as CAL
from scripts.research.pallet_bi_full_execution_v1 import fullnormal_backend as B
from scripts.research.pallet_visibility_curriculum_20261013_v1 import statewise_two_wd_family_v4_9 as WD
from scripts.research.pallet_boundary_action_repair_v1 import d1_batch as D1
from scripts.research.pallet_statewise_end_to_end_recovery_v1 import solver as S


def jsonrow(path):
    values=Q.P.B.rows(path)
    if len(values)!=1:raise ValueError('Exactly one frozen observation row required')
    return values[0]


def physical(solution,family,frame):
    if solution.get('accepted'):
        return dict(available=True,R_cf=solution['R'],R_physical=solution['R_physical'],centroid=solution['t'],
            cf_extents=family['cf_extents'],physical_registration_Q=family['physical_registration_Q'],source='MATCHED_FULL_NORMAL_E0')
    return dict(deepcopy(frame['initial_pose']),source='MATCHED_FULL_NORMAL_N0_FALLBACK')


def solve_frame(frame,selection,observation,mask,original_config):
    start=time.monotonic();selection,dedup=CAL.common_effective_selection(selection)
    geometry=Q.S.RT.registered_geometry(frame['raw_hw'],frame['K'],frame['native_xyz'],frame['initial_pose'])
    coarse=dict(R=frame['initial_pose']['R_cf'],t=frame['initial_pose']['centroid'],
        R_physical=frame['initial_pose']['R_physical'],physical_registration_Q=geometry['physical_registration_Q'].tolist())
    cfg=B.configuration(original_config);cfg['prior_weight']=1.
    domains=deepcopy(observation['_original_source']['ot_domains']['domains'])
    regions=deepcopy(observation['u_region_observation']['u_regions'])
    if selection['states']!=observation['states']:raise ValueError('Immutable upstream states differ')
    spec=dict(xyz=frame['native_xyz'],K=frame['K'],states=selection['states'],visible=selection['full_points'],domains=domains,
        image_hw=frame['raw_hw'],coarse_pose=coarse,initializations=None,config=cfg,
        boundary_observations=selection['partial_lines'],u_regions=regions)
    prior=B.solve_boundary_u_region_joint_ot_pnp(**spec);cfg0=deepcopy(cfg);cfg0['prior_weight']=0.
    wd=WD.solve_two_wd_families(spec['xyz'],spec['K'],spec['states'],spec['visible'],domains,spec['image_hw'],coarse,
        config=cfg0,boundary_observations=spec['boundary_observations'],u_regions=regions,solver_callable=B.solve_boundary_u_region_joint_ot_pnp)
    families={f['name']:f for f in wd['families']}
    outline=B.G10.observe_global_outline(mask,frame['raw_hw'],image_binding=dict(image_sha256=frame['image_sha256'],new_campaign_fixed_original_mask=True))
    prior_R=np.asarray(prior['R']if prior.get('accepted')else prior['fallback_pose']['R'])
    prior_t=prior['t']if prior.get('accepted')else prior['fallback_pose']['t']
    def refine(xyz,K,states,visible,domains,hw,coarse,**kw):
        family=next(f for f in families.values()if np.array_equal(np.asarray(f['native_xyz']),np.asarray(xyz)))
        solutions=[family['solution']];extra=[]
        if family['name']=='ORIGINAL_WD':solutions.append(prior)
        else:extra=[dict(R=(prior_R@WD.RY90.T).tolist(),t=prior_t,name='PRIOR_PHYSICAL_POSE_TRANSFORMED_SEED_ONLY')]
        return B.refine_global_outline_joint_pose(xyz,K,states,visible,domains,hw,coarse,
            outline_observation=outline,initial_solutions=solutions,initializations=extra,
            boundary_observations=kw['boundary_observations'],u_regions=regions,config=kw['config'])
    g10=WD.solve_two_wd_families(spec['xyz'],spec['K'],spec['states'],spec['visible'],domains,spec['image_hw'],coarse,
        config=cfg0,boundary_observations=spec['boundary_observations'],u_regions=regions,solver_callable=refine)
    # Both stages now consume exactly the same typed normal-only package,
    # same analytic outline factor, and same P1/WD0 starts. G13 is an exact
    # alias of G10, with source aliases retained in the complete bank.
    g13=deepcopy(g10);selected=g10['selected_family'];chosen=next(f for f in g13['families']if f['name']==selected)
    e0=physical(chosen['solution'],chosen,frame)
    finalfamilies={f['name']:{k:deepcopy(f[k])for k in ('name','native_xyz','cf_extents','physical_registration_Q','coarse_pose','initializations')}for f in g13['families']}
    packet=dict(id=frame['id'],partition=frame['partition'],solver_input=dict(xyz=spec['xyz'],K=spec['K'],states=spec['states'],
        visible=spec['visible'],domains=domains,image_hw=spec['image_hw'],coarse_pose=coarse,
        boundary_observations=spec['boundary_observations'],u_regions=regions,outline_observation=outline,config=cfg0),
        original_e_stage1_family_seed_bank={name:dict(family=fam)for name,fam in finalfamilies.items()},
        registered_family=finalfamilies[selected],original_e_legacy_boundary_observations=deepcopy(spec['boundary_observations']),
        provenance=dict(GT_inputs=False,new_matched_normal_only_hierarchy=True,original_D1_parity_claimed=False))
    candidates=[dict(candidate_id='N0',source='N0',R_physical=frame['initial_pose']['R_physical'],centroid=frame['initial_pose']['centroid']),
        dict(candidate_id='E0_REGENERATED',source='E0',R_physical=e0['R_physical'],centroid=e0['centroid'])]
    counts=Counter();all_stage_endpoints=[]
    for stage,records in [('P1',[dict(families['ORIGINAL_WD'],solution=prior)]),('WD0',wd['families']),('G10',g10['families']),('G13',g13['families'])]:
        for fam in records:
            sol=fam['solution'];Qmatrix=np.asarray(fam['physical_registration_Q'])
            def add(R,t,name,record=None):
                if R is None or t is None:return
                candidates.append(dict(candidate_id=name,source='D1_'+stage,R_physical=(np.asarray(R)@Qmatrix).tolist(),
                    centroid=np.asarray(t).tolist(),original_native_family=fam['name'],execution_record=deepcopy(record)))
            if sol.get('accepted'):add(sol['R'],sol['t'],stage+'/'+fam['name']+'/selected')
            for index,sample in enumerate(sol.get('feasible_pose_samples',[])):add(sample['R'],sample['t'],stage+'/'+fam['name']+'/sample/'+str(index),sample)
            for index,endpoint in enumerate(sol.get('all_visited_endpoints',[])):
                add(endpoint['R'],endpoint['t'],stage+'/'+fam['name']+'/endpoint/'+str(index),endpoint)
                all_stage_endpoints.append(dict(stage=stage,family=fam['name'],exact_reused=stage=='G13',record=endpoint))
            if stage!='G13':counts['pose_optimizer_calls']+=sol.get('optimizer_calls',0)
    return Q.S._json(dict(id=frame['id'],packet=packet,families=finalfamilies,candidates=candidates,fallback=frame['initial_pose'],
        e0_regenerated=e0,selected_family=selected,all_visited_endpoints=all_stage_endpoints,
        stage_results=dict(P1=prior,WD0=wd,G10=g10,G13=g13),operations=dict(counts),elapsed_seconds=time.monotonic()-start,
        stage_dependency_payloads=dict(P1=dict(spec=spec,ordered_initializations=prior.get('ordered_actual_initializations',[]),
            ordered_joint_initializations=prior.get('ordered_actual_joint_initializations',[])),
            WD0=dict(configuration=cfg0,complete_family_models=WD.make_wd_families(spec['xyz'],coarse),
                ordered_family_initializations={f['name']:f['solution'].get('ordered_actual_initializations',[])for f in wd['families']},
                ordered_family_joint_initializations={f['name']:f['solution'].get('ordered_actual_joint_initializations',[])for f in wd['families']}),
            G10=dict(configuration=cfg0,outline_observation=outline,outline_sha256=Q.P.digest(outline),
                complete_P1_result=prior,complete_WD0_result=wd,
                ordered_initializations={f['name']:f['solution'].get('ordered_actual_initializations',[])for f in g10['families']},
                complete_saved_initial_solutions={f['name']:f['solution'].get('full_saved_initial_solutions',[])for f in g10['families']}),
            G13=dict(exact_alias='G10',reason='identical effective typed normal input, outline, ordered P1/WD0 starts, solver configuration')),
        proof=dict(original_D1_parity_claimed=False,matched_new_backend=True,
            G13_exact_reuses_G10=True,G13_reuse_reason='identical typed normal graph, analytic outline, P1/WD0 starts and configuration',
            G10_family_readout_preserved=True,original_search_numeric_caps_preserved=True,
            revoked_2D_seed_or_guard_terms=0,partial_tangent_constraints=0,source_deduplication=dedup,
            new_full_search_calls=True,all_actual_endpoints_including_infeasible_preserved=True)))


def feature_frame(frame,result,old_config):
    models=B.compile_models(result['packet'],old_config);scale=max(m.ctx.scale for m in models.values());admitted=[];rejected=[]
    for candidate in result['candidates']:
        R=np.asarray(candidate['R_physical']);t=np.asarray(candidate['centroid']);proper=bool(np.isfinite(R).all()and np.isfinite(t).all()and np.allclose(R.T@R,np.eye(3),atol=1e-6,rtol=0)and abs(np.linalg.det(R)-1)<1e-6)
        ctx=models['ORIGINAL_WD'].ctx;native=R@ctx.Q.T;depth=((ctx.xyz-ctx.xyz.mean(0))@native.T+t)[:,2]
        if proper and depth.min()>1e-5:admitted.append(candidate)
        else:rejected.append(dict(candidate_id=candidate['candidate_id'],reason='INVALID_OR_NONPOSITIVE_DEPTH'))
    bank=S.unique_poses(admitted,scale);scores=[S.candidate_scores(candidate,models)for candidate in bank]
    inp=dict(frame,packet=result['packet'],families=result['families'],fallback=frame['initial_pose'])
    features=S.features_for_bank(inp,models,bank,scores)
    missing_indices=[S.FEATURE_NAMES.index(name)for name in ('B_energy','edge_energy','score_gap_B','score_gap_edge')]
    for feature,candidate in zip(features,bank):
        # The old boundary-profile producer queries revoked raw coordinates.
        # It is absent for every matched arm; missing energy uses the original
        # validity convention, never a confident fabricated zero cost.
        for index in missing_indices:feature['features'][index]=0.;feature['features'][index+len(S.FEATURE_NAMES)]=0.
        aliases=[s for s in admitted if np.max(np.abs(np.asarray(s['R_physical'])-candidate['R_physical']))<=1e-7 and np.max(np.abs(np.asarray(s['centroid'])-candidate['centroid']))<=1e-7*scale]
        feature.update(source_aliases=sorted({s['candidate_id']for s in aliases}),source_baseline_aliases=sorted({s['source']for s in aliases if s['source']in('N0','E0')}),object_scale_m=scale)
        if feature['source_baseline_aliases']:feature['eligible']=True
    return dict(id=frame['id'],partition=frame['partition'],split=frame['split'],session=frame['session'],cluster_id=frame['cluster_id'],
        candidates=features,fallback=frame['initial_pose'],e0_regenerated=result['e0_regenerated'],object_scale_m=scale,
        feature_names=list(S.FEATURE_NAMES),transformation='signed_log1p31_plus31_explicit_validity_bits',
        state_counts=dict(Counter(result['packet']['solver_input']['states'])),candidate_sources_before_dedup=len(result['candidates']),
        projection_rejections=rejected,proof=result['proof'],
        B_profile_features=dict(available=False,reason='old bundle would query archived revoked raw points; absent identically all matched arms',energy_indices_masked=missing_indices))


@lru_cache(maxsize=1)
def source_binding():
    result=D1.source_bindings();root=Path(__file__).resolve().parents[3]
    paths=[Path(B.__file__),Path(__file__),Path(CAL.__file__),Path(Q.__file__),Path(Q.S.__file__)]
    local=Path(Q.S.__file__).parent
    paths.extend(local/name for name in ('predict.py','bank.py','observer.py','models.py','features.py','registry.py','supervision.py','evaluate.py'))
    for path in paths:result[str(path.relative_to(root))]=Q.sha(path)
    return result


def cache_payload(frame,selection,observation,mask_sha,original_config,old_config):
    selected,_=CAL.common_effective_selection(selection)
    for point in selected['full_points']:point.pop('uncertainty_calibration',None)
    return dict(frame={k:frame[k]for k in ('K','native_xyz','raw_hw','initial_pose','image_sha256')},selection=selected,
        domains=observation['_original_source']['ot_domains']['domains'],u_regions=observation['u_region_observation']['u_regions'],
        mask_sha256=mask_sha,configuration=B.configuration(original_config),feature_configuration=old_config,source=source_binding())


def run(prior_run,calibration_root,output,*,pilot=False,arms=('C0','CP','CB'),root_GO=False):
    if root_GO is not True:raise PermissionError('Explicit new full-normal producer GO required')
    cv2.setNumThreads(1);prior_run,calibration_root,output=map(Path,(prior_run,calibration_root,output));output.mkdir(parents=True,exist_ok=True)
    original_config,_,_=D1._cpu_models();old_run=prior_run.parent/'pallet-statewise-end-to-end-recovery-v1';old_config=Q.read(old_run/'CONFIG_FROZEN.json')
    original_config=B.configuration(original_config)
    manifest=[];wall=time.monotonic();p0=Q.read(prior_run/'P0_INPUT_CONTRACT.json');bindings=source_binding();calls=0;reuses=0
    producer_lock=dict(source_binding=bindings,original_numeric_configuration=original_config,feature_configuration=old_config,
        calibration_packet_seal_sha256=Q.sha(calibration_root/'PACKET_SEAL.json'),arms=list(arms),pilot=pilot,
        original_D1_parity_claimed=False,new_matched_full_normal_backend=True)
    lock_path=output/'PRODUCER_LOCK.json'
    if lock_path.exists()and Q.read(lock_path)!=producer_lock:raise ValueError('Frozen new producer lock changed')
    Q.write(lock_path,producer_lock)
    for domain in ('real','synthetic'):
        frames,upstream=CAL.load_bound_frames(prior_run,domain);selected=list(enumerate(frames))
        if pilot:
            selected=sorted(selected,key=lambda item:item[1]['id'])[:12]
            Q.write(output/f'{domain}_PILOT_SELECTION.json',dict(rule='first12 sorted original IDs, frozen campaign plan',
                ids=[f['id']for _,f in selected],ordinals=[i for i,_ in selected],reference_scores_used=False))
        for ordinal,frame in selected:
            packet_path=calibration_root/'packets'/domain/f'{ordinal:04d}.json.gz';packet=Q.read(packet_path)
            if packet['id']!=frame['id']:raise ValueError('S2 packet identity mismatch')
            obs=Path(p0[domain]['inventory'][ordinal]['observation_folder']);observation=jsonrow(obs/'OBSERVATION.jsonl.gz')
            mask_path=obs/'MASK.npz'
            with np.load(mask_path)as arrays:mask=arrays['pallet_mask_clean'].copy()
            cache={}
            for arm in arms:
                folder=output/arm/domain/f'{ordinal:04d}';folder.mkdir(parents=True,exist_ok=True);seal_path=folder/'FRAME_SEAL.json'
                payload=cache_payload(frame,packet['selections'][arm],observation,Q.sha(mask_path),original_config,old_config);signature=Q.P.digest(payload)
                if seal_path.exists():
                    seal=Q.read(seal_path)
                    if seal['cache_payload_sha256']!=signature or seal['source_binding']!=bindings:raise ValueError('Completed full-normal row dependency changed')
                    for name,value in seal['artifacts'].items():
                        if Q.sha(folder/name)!=value:raise ValueError('Completed artifact changed')
                    result=Q.read(folder/'RESULT.json.gz');feature=Q.read(folder/'FEATURE.json.gz');cache[signature]=(payload,result,feature,arm)
                    manifest.append(dict(domain=domain,ordinal=ordinal,arm=arm,id=frame['id'],seal=str(seal_path),resource=seal['resource'],exact_reused=seal['exact_reused']));continue
                tick=time.monotonic();cpu=time.process_time()
                if signature in cache:
                    cached,result,feature,origin=cache[signature]
                    if cached!=payload:raise AssertionError('Cache collision')
                    reused=True;actual_calls=0;reuses+=1
                else:
                    result=solve_frame(frame,packet['selections'][arm],observation,mask,original_config)
                    feature=feature_frame(frame,result,old_config);cache[signature]=(payload,result,feature,arm);origin=arm;reused=False
                    actual_calls=result['operations']['pose_optimizer_calls'];calls+=actual_calls
                Q.write(folder/'RESULT.json.gz',result);Q.write(folder/'FEATURE.json.gz',feature)
                seal=dict(id=frame['id'],arm=arm,domain=domain,ordinal=ordinal,complete=True,source_binding=bindings,
                    cache_payload_sha256=signature,exact_reused=reused,cache_origin_arm=origin,
                    artifacts={name:Q.sha(folder/name)for name in ('RESULT.json.gz','FEATURE.json.gz')},
                    resource=dict(wall_seconds=time.monotonic()-tick,CPU_seconds=time.process_time()-cpu,actual_pose_optimizer_calls=actual_calls),
                    numeric_reference_inputs=False,selected_reference_inputs=False)
                Q.write(seal_path,seal);manifest.append(dict(domain=domain,ordinal=ordinal,arm=arm,id=frame['id'],seal=str(seal_path),resource=seal['resource'],exact_reused=reused))
            print(json.dumps(dict(stage='S2_MATCHED_FULL_NORMAL',domain=domain,ordinal=ordinal,completed_rows=len(manifest),wall_seconds=time.monotonic()-wall)),flush=True)
    # Stable existing H1 input shape: one frame record per original ID.
    for domain in ('real','synthetic'):
        for arm in arms:
            records=[Q.read(output/arm/domain/f"{item['ordinal']:04d}"/'FEATURE.json.gz')for item in manifest if item['domain']==domain and item['arm']==arm]
            Q.P.save_rows(output/f'{arm}_{domain}_FEATURES.jsonl.gz',records)
    summary=dict(status='COMPLETE_NEW'if not pilot else'DIAGNOSTIC_ONLY_TECHNICAL_PILOT',rows=len(manifest),arms=list(arms),pilot=pilot,
        original_D1_parity_claimed=False,matched_full_hierarchy=True,source_binding=bindings,manifest=manifest,
        new_optimizer_calls=calls,new_exact_packet_reuses=reuses,wall_seconds=time.monotonic()-wall,numeric_reference_inputs=False)
    Q.write(output/'SUMMARY.json',summary);return summary


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--prior-run',required=True);parser.add_argument('--calibration-root',required=True);parser.add_argument('--output',required=True)
    parser.add_argument('--pilot',action='store_true');parser.add_argument('--arms',nargs='+',default=['C0','CP','CB']);parser.add_argument('--root-go',action='store_true')
    args=parser.parse_args();result=run(args.prior_run,args.calibration_root,args.output,pilot=args.pilot,arms=args.arms,root_GO=args.root_go)
    print({k:v for k,v in result.items()if k!='manifest'})


if __name__=='__main__':main()
