"""Fixed CAL mean/noise profiles with OLD V47 B/I observations, all95.

Three profiles use one shared CAL total noise and differ only in mean bias.
Predictions for all original frames and both W/D families precede scoring.
No default adoption, held-out generalization or global optimality is claimed.
"""
import os
for _key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[_key]='1'
from copy import deepcopy
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor,as_completed
from multiprocessing import get_context
import argparse
import shutil
import time
import numpy as np

from . import common as C
from . import statewise_evaluation_v4 as E
from . import evaluate_global_outline_pose_v4_10 as OLD
from . import statewise_global_outline_solver_v4_12 as G
from . import global_outline_calibrated_factor_v4_12 as FACTOR
from . import geometry_conditioned_outline_calibration_v4_12 as CAL
from .scoring import pose_metric,distribution

W=OLD.W
MASKS=OLD.MASKS
BASELINE=OLD.OUTPUT.parent/'global_outline_all95_eval_v4_10'
OUTPUT=OLD.OUTPUT.parent/'calibrated_global_outline_eval_v4_12'
PROFILES=CAL.PROFILES
FAMILIES=('ORIGINAL_WD','SWAPPED_WD','SELECTED_JOINT_OUTLINE')


def _load():
    rows,model,config,bindings,parent=OLD._load('all95')
    sealpath=CAL.OUTPUT/'MODEL_SEAL.json';seal=C.read(sealpath)
    calmodel=C.read(CAL.OUTPUT/'MODEL.json')
    if C.sha(CAL.OUTPUT/'MODEL.json')!=seal['model_sha256'] or C.sha(CAL.OUTPUT/'PROTOCOL.json')!=seal['protocol_sha256']:
        raise ValueError('Frozen CAL geometry model binding differs')
    CAL._verify(seal['source_sha256']);CAL._verify(seal['input_sha256'])
    for path in (sealpath,CAL.OUTPUT/'MODEL.json',CAL.OUTPUT/'PROTOCOL.json'):
        bindings[str(path.relative_to(C.ROOT))]=C.sha(path)
    bseal=C.read(BASELINE/'PREDICTION_SEAL.json');bp=C.read(BASELINE/'PROTOCOL.json')
    if bseal['rows']!=95 or not bseal['before_new_reference_metric_read'] or C.sha(BASELINE/'PREDICTIONS.jsonl.gz')!=bseal['predictions_sha256'] or C.sha(BASELINE/'PROTOCOL.json')!=bseal['protocol_sha256']:
        raise ValueError('Exact whole95 OLDq continuous baseline required')
    for key in ('source_sha256','input_sha256','protected_user_sha256'):
        E._verify_sources(bp[key])
    baseline=list(C.rows(BASELINE/'PREDICTIONS.jsonl.gz'))
    if [r['id'] for r in baseline]!=[r['id'] for r in rows]:
        raise ValueError('Exact ordered original95 baseline required')
    for row,b in zip(rows,baseline):
        if any(b[key]!=row[key] for key in ('states','image_sha256','frozen_BI_fit','partial_boundary_observation','u_region_observation')):
            raise ValueError('OLDq baseline observation mismatch')
        row['_CONTINUOUS_BASELINE']=b
    for path in (BASELINE/'PREDICTIONS.jsonl.gz',BASELINE/'PREDICTION_SEAL.json',BASELINE/'PROTOCOL.json'):
        bindings[str(path.relative_to(C.ROOT))]=C.sha(path)
    parent['source_sha256']={**parent['source_sha256'],**seal['source_sha256']}
    return rows,model,config,bindings,parent,calmodel


def _run(task):
    row,model,config,calmodel=task;started=time.monotonic()
    spec=W.PREVIOUS._spec(row,config);before=E._digest(spec);record=row['_MASK']
    with np.load(MASKS/record['mask_npz']) as arrays:
        mask=arrays['pallet_mask_clean'].copy()
    basic=G.observe_global_outline(mask,spec['image_hw'])
    # The hull-diagonal statistic is the same one fitted on CAL; not a GT size.
    if basic.get('supported'):
        diagonal=float(np.linalg.norm(np.ptp(np.asarray(basic['polygon_xy']),axis=0)))
    else:
        y,x=np.nonzero(mask>=.5)
        diagonal=float(np.linalg.norm([np.ptp(x),np.ptp(y)])) if len(x)>1 else 1.
        diagonal=max(diagonal,1.)
    extents=row['runtime']['initial_geometry']['cf_extents']
    prior=row['_cached_V48']['arms'][W.PREVIOUS.ARMS[0]]
    outputs={};arms={}
    for profile in PROFILES:
        params=CAL.geometry_outline_parameters(calmodel,extents,diagonal,profile=profile)
        ocfg={**G.DEFAULT_OUTLINE_CONFIGURATION,'normal_sigma_px':params['total_normal_sigma_px']}
        observation=G.observe_global_outline(mask,spec['image_hw'],configuration=ocfg,image_binding=dict(
            image_sha256=row['image_sha256'],mask_file_sha256=record['mask_file_sha256'],mask_npz=record['mask_npz']))
        observation.update(normal_bias_px=params['outward_mask_bias_px'],outline_noise_model_sha256=params['model_sha256'])
        calls={}
        def solve(xyz,K,states,visible,domains,hw,coarse,**kw):
            family=next(f for f in row['_WD']['families'] if np.allclose(f['native_xyz'],xyz,atol=1e-12,rtol=0))
            name=family['name'];solutions=[family['solution']];initial=[]
            old_cont=next(f for f in row['_CONTINUOUS_BASELINE']['families'] if f['name']==name)
            solutions.append(old_cont['solution'])
            if name=='ORIGINAL_WD':
                solutions.append(prior['solution'])
            else:
                initial.append(dict(R=(np.asarray(prior['actual_pose']['R_cf'])@W.FAMILY.RY90.T).tolist(),
                    t=prior['actual_pose']['centroid'],name='OLD_PRIOR_PHYSICAL_POSE_TRANSFORMED_SEED_ONLY'))
            applicable=bool(observation.get('supported') and len(visible)>=2 and
                (len(visible)>2 or domains or kw.get('u_regions') or kw.get('boundary_observations')))
            if applicable:
                answer=G.refine_global_outline_joint_pose(xyz,K,states,visible,domains,hw,coarse,
                    outline_observation=observation,outline_configuration=ocfg,
                    initial_solutions=solutions,initializations=initial,config=kw['config'],
                    boundary_observations=kw.get('boundary_observations'),u_regions=kw.get('u_regions'))
            else:
                # Fresh current-input solve also closes old cache objective metadata.
                answer=G.P.solve_boundary_u_region_joint_ot_pnp(xyz,K,states,visible,domains,hw,coarse,
                    initializations=kw.get('initializations'),config=kw['config'],
                    boundary_observations=kw.get('boundary_observations'),u_regions=kw.get('u_regions'))
            calls[name]=dict(outline_applied=applicable,fresh_current_input_solve=True,
                reason=None if applicable else 'unsupported_outline_or_insufficient_visible_support')
            return answer
        result=W.FAMILY.solve_two_wd_families(spec['xyz'],spec['K'],spec['states'],spec['visible'],spec['domains'],
            spec['image_hw'],spec['coarse_pose'],config=spec['config'],boundary_observations=spec['boundary_observations'],
            u_regions=spec['u_regions'],solver_callable=solve)
        families=result['families'];chosen=families[result['selected_family_index']]
        par={f['name']:W._family_finish(row,model,f,profile+'__'+f['name']) for f in families}
        par['SELECTED_JOINT_OUTLINE']=W._family_finish(row,model,chosen,profile+'__SELECTED_JOINT_OUTLINE',result['solution'])
        for name,arm in par.items():
            family_name=chosen['name'] if name=='SELECTED_JOINT_OUTLINE' else name
            arm['outline_applicability']=deepcopy(calls[family_name])
            if arm['accepted']:
                arm['actual_pose']['source']=('CALIBRATED_GLOBAL_OUTLINE_BI_PARTIAL_OT_U_JOINT_SE3'
                    if calls[family_name]['outline_applied'] else 'FRESH_DATA_ONLY_BI_PARTIAL_OT_U_JOINT_SE3')
            arms[profile+'__'+name]=arm
        for f in families:f['solution']=E._compact_solution(f['solution'])
        outputs[profile]=dict(parameters=params,mask_observation={k:v for k,v in observation.items()
            if k not in ('observed_boundary_samples_xy',)},families=families,
            selected_family=result['selected_family'],family_ambiguity=result['family_ambiguity'],applicability=calls,
            selection_rule='minimum joint-feasible BI/partial plus CAL-centered arc-normalized outline loss; exact tie ORIGINAL_WD')
    if before!=E._digest(spec):raise RuntimeError('Fixed mathematical input mutated')
    return C.finite(dict(id=row['id'],partition=row['partition'],session=row['session'],image=row['image'],
        image_sha256=row['image_sha256'],states=row['states'],frozen_BI_fit=row['frozen_BI_fit'],
        partial_boundary_observation=row['partial_boundary_observation'],u_region_observation=row['u_region_observation'],
        profiles=outputs,arms=arms,mathematical_source_sha256=before,
        contract=dict(all_profiles_same_shared_CAL_sigma=True,only_profile_mean_changes=True,
            known_unordered_geometry_runtime_key=True,session_or_frame_routing=False,
            S_local_queries=0,U_local_queries=0,raw_hidden_coordinates_read=False,new_image_forwards=0,
            every_OT_U_and_partial_tangent_guard=True,BI_q_covariances_reliability_unchanged=True,
            GT_runtime_inputs=False,prior_weight=0.),elapsed_seconds=time.monotonic()-started))


def predict(output=OUTPUT,workers=3):
    output=Path(output).resolve()
    if output.exists():raise FileExistsError('Preserve all previous prediction evidence')
    rows,model,config,bindings,parent,calmodel=_load();sources=deepcopy(parent['source_sha256'])
    for module in (G,FACTOR,FACTOR.BASE,CAL,OLD):
        path=Path(module.__file__);sources[str(path.relative_to(C.ROOT))]=C.sha(path)
    path=Path(__file__);sources[str(path.relative_to(C.ROOT))]=C.sha(path)
    protected=parent['protected_user_sha256']
    for value in (sources,bindings,protected):E._verify_sources(value)
    output.mkdir(parents=True)
    protocol=dict(schema='CALIBRATED_GLOBAL_OUTLINE_OLD_BI_ALL95_V4_12',ids=[r['id'] for r in rows],count=95,
        population='original CAL46 plus TEST49; every flagged/fallback case retained',severe=0,
        fixed_profiles=list(PROFILES),solver_configuration=config,outline_configuration=G.DEFAULT_OUTLINE_CONFIGURATION,
        outline_runtime_sigma='shared frozen CAL total scatter times observed mask diagonal; minimum1px',
        outline_runtime_mean='fixed profile, known unordered size only, no frame/material/session routing',
        source_sha256=sources,input_sha256=bindings,protected_user_sha256=protected,
        CAL_model_sha256=calmodel['model_sha256'],CAL_geometry_one_session_limit_disclosed=True,
        frozen_old_BI=True,all95_no_GT_runtime_inputs=True,labels_before_prediction_seal=False,
        prior_poses_and_old_global_search_used_as_starts_only=True,
        TEST_is_reused_iterative_exploratory=True,default_adopted=False,workers=workers)
    C.write(output/'PROTOCOL.json',protocol)
    for relative,digest in sources.items():
        target=output/'code_snapshot'/relative;target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(C.ROOT/relative,target)
        if C.sha(target)!=digest:raise RuntimeError('Snapshot source differs')
    started=time.monotonic();done={}
    with ProcessPoolExecutor(max_workers=workers,mp_context=get_context('spawn')) as pool:
        futures={pool.submit(_run,(r,model,config,calmodel)):r['id'] for r in rows}
        for future in as_completed(futures):
            value=future.result();done[value['id']]=value
            C.write(output/'frames'/(value['id'].replace(':','__')+'.json'),value)
            print(f'{len(done)}/95 {value["id"]}',flush=True)
    for value in (sources,bindings,protected):E._verify_sources(value)
    C.save_rows(output/'PREDICTIONS.jsonl.gz',[done[r['id']] for r in rows])
    C.write(output/'PREDICTION_SEAL.json',dict(rows=95,predictions_sha256=C.sha(output/'PREDICTIONS.jsonl.gz'),
        protocol_sha256=C.sha(output/'PROTOCOL.json'),before_new_reference_metric_read=True,
        source_sha256=sources,input_sha256=bindings,elapsed_seconds=time.monotonic()-started))


def score(output=OUTPUT):
    output=Path(output).resolve();protocol=C.read(output/'PROTOCOL.json');seal=C.read(output/'PREDICTION_SEAL.json')
    if not seal['before_new_reference_metric_read'] or seal['rows']!=95 or C.sha(output/'PREDICTIONS.jsonl.gz')!=seal['predictions_sha256'] or C.sha(output/'PROTOCOL.json')!=seal['protocol_sha256']:
        raise ValueError('Complete original95 prediction seal required before scoring')
    for key in ('source_sha256','input_sha256','protected_user_sha256'):E._verify_sources(protocol[key])
    rows=list(C.rows(output/'PREDICTIONS.jsonl.gz'))
    if [r['id'] for r in rows]!=protocol['ids']:raise ValueError('Exact ordered95 required')
    arm_names=tuple(p+'__'+f for p in PROFILES for f in FAMILIES);reports={}
    for part,count in (('CAL',46),('TEST',49)):
        population=[r for r in rows if r['partition']==part]
        if len(population)!=count:raise ValueError('Original whole partition required')
        target=E.COHORT/('OOF_pose_v1/ISOLATED_CAL_SCORING.json' if part=='CAL' else 'test_reused_v1/ISOLATED_TEST_SCORING.json')
        labels=C.read(target)['rows'];scored=[]
        for row in population:
            scored.append(dict(id=row['id'],metrics={a:pose_metric(row['arms'][a]['actual_pose'],labels[row['id']]['target']['truth']) for a in arm_names},
                accepted={a:row['arms'][a]['accepted'] for a in arm_names},
                selected_family={p:row['profiles'][p]['selected_family'] for p in PROFILES}))
        report=dict(partition=part,count=count,rows=scored,all_original_flagged_and_fallback_kept=True,
            summary={a:dict(translation_cm=distribution([r['metrics'][a]['translation_cm'] for r in scored]),
                rotation_deg=distribution([r['metrics'][a]['rotation_deg'] for r in scored]),
                translation_le5=sum(r['metrics'][a]['translation_cm']<=5 for r in scored),
                successes=sum(r['metrics'][a]['success_5cm_5deg'] for r in scored),
                fallback_count=sum(not r['accepted'][a] for r in scored)) for a in arm_names},
            target_binding=E._binding(target),no_GT_runtime_choice_or_fit=True,TEST_is_reused_iterative_exploratory=True)
        C.write(output/f'{part}_SCORE.json',report);reports[part]=report
    C.write(output/'SCORE_SEAL.json',dict(source_prediction_seal=E._binding(output/'PREDICTION_SEAL.json'),
        scores={part:C.sha(output/f'{part}_SCORE.json') for part in reports}))
    print({part:{p:report['summary'][p+'__SELECTED_JOINT_OUTLINE'] for p in PROFILES} for part,report in reports.items()},flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('command',choices=('predict','score','preflight'))
    parser.add_argument('--output',type=Path,default=OUTPUT);parser.add_argument('--workers',type=int,default=3)
    args=parser.parse_args()
    if args.command=='preflight':
        rows,model,config,bindings,parent,calmodel=_load()
        print(dict(rows=len(rows),CAL=sum(r['partition']=='CAL' for r in rows),TEST=sum(r['partition']=='TEST' for r in rows),
            model_sha256=calmodel['model_sha256'],profiles=PROFILES),flush=True)
    elif args.command=='predict':predict(args.output,args.workers)
    else:score(args.output)
