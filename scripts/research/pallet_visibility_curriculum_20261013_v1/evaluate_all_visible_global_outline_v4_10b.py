"""Seal continuous two-W/D global-outline predictions before label scoring.

Use the sealed all-visible RGB corrections with both explicit W/D families and
the identical recipe. Frozen observations/covariances/regions/masks are used;
no GT, image forward, new uncertainty fit or pose reference enters prediction.
"""
import os
for _key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[_key]='1'
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
from . import evaluate_two_wd_families_v4_9 as W
from . import statewise_global_outline_solver_v4_10 as G
from .scoring import pose_metric,distribution

IDS=('eval_pallet07:1778652140531310080','eval_pallet07:1778652142480077056','eval_pallet09:1778653661653195264')
OUTPUT=W.OUTPUT.parent/'all_visible_global_outline_eval_v4_10b'
CORRECTIONS=W.OUTPUT.parent/'global_BI_edge_eval_v4_9c'
MASKS=W.OUTPUT.parent/'image_evidence_v4_6'


def _load(population):
    rows,model,config,bindings,prior=W._load();protocol=C.read(W.OUTPUT/'PROTOCOL.json')
    complete=C.read(W.OUTPUT/'PREDICTION_COMPLETE.json');assert complete['rows']==95
    for key in ('source_sha256','input_sha256','protected_user_sha256'):E._verify_sources(protocol[key])
    wd={}
    for part,count in (('CAL',46),('TEST',49)):
        path=W.OUTPUT/f'{part}_PREDICTION_SEAL.json';seal=C.read(path);journal=W.OUTPUT/f'{part}_PREDICTIONS.jsonl.gz'
        assert seal['rows']==count and C.sha(journal)==seal['predictions_sha256']
        assert C.sha(path)==complete['ordered_partition_prediction_seal_sha256'][part]
        for r in C.rows(journal):wd[r['id']]=r
        for path in (path,journal):bindings[str(path.relative_to(C.ROOT))]=C.sha(path)
    for path in (W.OUTPUT/'PROTOCOL.json',W.OUTPUT/'PREDICTION_COMPLETE.json'):
        bindings[str(path.relative_to(C.ROOT))]=C.sha(path)
    seal=C.read(MASKS/'SEAL.json');records={r['id']:r for r in C.read(MASKS/'RECORDS.json')}
    assert seal['records']==95 and not seal['GT_or_pose_labels_read']
    assert C.sha(MASKS/'RECORDS.json')==seal['records_sha256'] and C.sha(MASKS/'PROTOCOL.json')==seal['protocol_sha256']
    for path in (MASKS/'SEAL.json',MASKS/'RECORDS.json',MASKS/'PROTOCOL.json'):
        bindings[str(path.relative_to(C.ROOT))]=C.sha(path)
    selected=rows if population=='all95' else [r for r in rows if r['id'] in IDS]
    if len(selected)!=(95 if population=='all95' else 3):raise ValueError('Exact fixed population required')
    for row in selected:
        row['_WD']=wd[row['id']];row['_MASK']=records[row['id']]
        maskpath=MASKS/row['_MASK']['mask_npz'];bindings[str(maskpath.relative_to(C.ROOT))]=C.sha(maskpath)
        if C.sha(maskpath)!=row['_MASK']['mask_file_sha256'] or row['_MASK']['image_sha256']!=row['image_sha256']:
            raise ValueError('Actual current global mask binding mismatch')
    # Attach the complete presealed image correction inputs, never score rows.
    correction_protocol=C.read(CORRECTIONS/'PROTOCOL.json')
    correction_complete=C.read(CORRECTIONS/'PREDICTION_COMPLETE.json')
    if correction_complete['rows']!=95 or not correction_complete['all_corrections_and_poses_before_labels']:
        raise ValueError('Complete sealed all-visible95 required')
    for key in ('source_sha256','input_sha256','protected_user_sha256'):
        E._verify_sources(correction_protocol[key])
    corrections={}
    for part,count in (('CAL',46),('TEST',49)):
        journal=CORRECTIONS/f'{part}_POSE_PREDICTIONS.jsonl.gz'
        sealpath=CORRECTIONS/f'{part}_POSE_PREDICTION_SEAL.json';seal=C.read(sealpath)
        if seal['rows']!=count or not seal['before_annotations'] or C.sha(journal)!=seal['predictions_sha256'] or C.sha(CORRECTIONS/'PROTOCOL.json')!=seal['protocol_sha256']:
            raise ValueError('All-visible correction prediction seal mismatch')
        values=list(C.rows(journal));expected=[r['id'] for r in rows if r['partition']==part]
        if [r['id'] for r in values]!=expected:raise ValueError('Complete exact ordered95 corrections required')
        corrections.update({r['id']:r for r in values})
        for path in (journal,sealpath):bindings[str(path.relative_to(C.ROOT))]=C.sha(path)
    for path in (CORRECTIONS/'PROTOCOL.json',CORRECTIONS/'PREDICTION_COMPLETE.json'):
        bindings[str(path.relative_to(C.ROOT))]=C.sha(path)
    protocol['source_sha256']={**protocol['source_sha256'],**correction_protocol['source_sha256']}
    for row in selected:
        c=corrections[row['id']]
        old_boundary=c['partial_boundary_observation']['original_boundary_observations_reference']
        if (c['states']!=row['states'] or c['image_sha256']!=row['image_sha256']
            or c['original_BI_fit']!=row['frozen_BI_fit'] or c['u_region_observation']!=row['u_region_observation']
            or c['runtime']['initial_geometry']!=row['runtime']['initial_geometry']
            or old_boundary!=row['partial_boundary_observation']['boundary_observations']):
            raise ValueError('All-visible observations do not share original state/native/image input')
        row['frozen_BI_fit']=deepcopy(c['frozen_BI_fit'])
        row['partial_boundary_observation']=deepcopy(c['partial_boundary_observation'])
        row['_ALL_VISIBLE']=c
    return selected,model,config,bindings,protocol


def _run(task):
    row,model,config=task;start=time.monotonic();spec=W.PREVIOUS._spec(row,config);record=row['_MASK']
    with np.load(MASKS/record['mask_npz']) as arrays:mask=arrays['pallet_mask_clean'].copy()
    observation=G.observe_global_outline(mask,spec['image_hw'],image_binding=dict(
        image_sha256=row['image_sha256'],mask_file_sha256=record['mask_file_sha256'],mask_npz=record['mask_npz']))
    prior=row['_cached_V48']['arms'][W.PREVIOUS.ARMS[0]]
    def solve(xyz,K,states,visible,domains,hw,coarse,**kw):
        family=next(f for f in row['_WD']['families'] if np.allclose(f['native_xyz'],xyz,atol=1e-12,rtol=0))
        fresh=row['_ALL_VISIBLE']['arms']
        fresh_original=fresh['ALL_VISIBLE_GLOBAL_EDGE_PRIOR0']['solution']
        fresh_prior=fresh['ALL_VISIBLE_GLOBAL_EDGE_PRIOR1']['solution']
        solutions=[family['solution']];initial=[]
        if family['name']=='ORIGINAL_WD':
            solutions=[fresh_original,family['solution'],fresh_prior,prior['solution']]
        else:
            for name,arm in [('ORIGINAL_ALL_VISIBLE_NO_PRIOR',fresh['ALL_VISIBLE_GLOBAL_EDGE_PRIOR0']),
                             ('ORIGINAL_ALL_VISIBLE_PRIOR',fresh['ALL_VISIBLE_GLOBAL_EDGE_PRIOR1']),
                             ('OLD_PRIOR',prior)]:
                pose=arm['actual_pose']
                if pose.get('available'):
                    initial.append(dict(R=(np.asarray(pose['R_cf'])@W.FAMILY.RY90.T).tolist(),
                        t=pose['centroid'],name=name+'_PHYSICAL_TRANSFORMED_SEED_ONLY'))
        return G.refine_global_outline_joint_pose(xyz,K,states,visible,domains,hw,coarse,
            outline_observation=observation,initial_solutions=solutions,initializations=initial,
            config=kw['config'],boundary_observations=kw.get('boundary_observations'),u_regions=kw.get('u_regions'))
    before=E._digest(spec)
    result=W.FAMILY.solve_two_wd_families(spec['xyz'],spec['K'],spec['states'],spec['visible'],spec['domains'],spec['image_hw'],
        spec['coarse_pose'],config=spec['config'],boundary_observations=spec['boundary_observations'],
        u_regions=spec['u_regions'],solver_callable=solve)
    assert before==E._digest(spec)
    families=result['families'];arms={}
    for f in families:arms[f['name']]=W._family_finish(row,model,f,'CONTINUOUS_GLOBAL_OUTLINE_'+f['name'])
    selected=families[result['selected_family_index']]
    arms['SELECTED_JOINT_OUTLINE']=W._family_finish(row,model,selected,'SELECTED_JOINT_OUTLINE',result['solution'])
    for arm in arms.values():
        if arm['accepted']:arm['actual_pose']['source']='CONTINUOUS_GLOBAL_OUTLINE_BI_PARTIAL_OT_U_JOINT_SE3'
    for f in families:f['solution']=E._compact_solution(f['solution'])
    return C.finite(dict(id=row['id'],partition=row['partition'],session=row['session'],image=row['image'],
        image_sha256=row['image_sha256'],states=row['states'],frozen_BI_fit=row['frozen_BI_fit'],
        partial_boundary_observation=row['partial_boundary_observation'],u_region_observation=row['u_region_observation'],
        families=families,selected_family=result['selected_family'],selected_model_geometry=result['selected_model_geometry'],
        family_ambiguity=result['family_ambiguity'],arms=arms,
        family_selection=dict(rule='minimum BI/partial plus fixed arc-normalized robust whole-outline objective',
            stable_original_ties=True,GT_inputs=False,pure_mask_selector_only=False,continuous_global_mask_factor_used=True),
        mathematical_source_sha256=before,mask_observation={k:v for k,v in observation.items() if k not in ('observed_boundary_samples_xy',)},
        contract=dict(prior_weight=0.,observed_BI_q_and_both_covariances_and_reliability_unchanged=True,
            every_OT_U_and_partial_tangent_guard=True,S_local_queries=0,U_local_queries=0,
            raw_hidden_coordinates_read=False,new_image_forwards=0,GT_inputs=False),elapsed_seconds=time.monotonic()-start))


def predict(output=OUTPUT,population='targets',workers=3):
    output=Path(output).resolve()
    if output.exists():raise FileExistsError('Preserve old global-outline predictions')
    rows,model,config,bindings,parent=_load(population);sources=deepcopy(parent['source_sha256'])
    for path in (Path(__file__),Path(G.__file__)):sources[str(path.relative_to(C.ROOT))]=C.sha(path)
    protected=parent['protected_user_sha256']
    for value in (sources,bindings,protected):E._verify_sources(value)
    output.mkdir(parents=True)
    protocol=dict(schema='ALL_VISIBLE_RGB_PLUS_CONTINUOUS_GLOBAL_OUTLINE_TWO_WD_V4_10B',population=population,ids=[r['id'] for r in rows],
        count=len(rows),severe=0,solver_configuration=config,outline_configuration=G.DEFAULT_OUTLINE_CONFIGURATION,
        source_sha256=sources,input_sha256=bindings,protected_user_sha256=protected,
        old_prior_pose_used_as_initialization_only=True,original_global_pose_search_bound_to_WD_predictions=True,
        newly_searched_global_optimum_or_uniqueness_claimed=False,GT_inputs=False,labels_before_prediction_seal=False,
        existing_three_cases_selected_by_user_and_prior_errors_for_diagnosis=population=='targets',
        TEST_is_reused_iterative_exploratory=True,initial_S_and_all_states_unchanged=True,workers=workers,
        all_visible_RGB_corrections_presealed=True,all_visible_changed_full_points_replace_same_corner_partial=True,
        original_CAL_fit_covariances_and_reliability_unchanged=True,
        source_geometry_and_all_U_OT_regions_unchanged=True)
    C.write(output/'PROTOCOL.json',protocol)
    for relative,digest in sources.items():
        target=output/'code_snapshot'/relative;target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(C.ROOT/relative,target)
        assert C.sha(target)==digest
    started=time.monotonic();done={}
    with ProcessPoolExecutor(max_workers=workers,mp_context=get_context('spawn')) as pool:
        futures={pool.submit(_run,(r,model,config)):r['id'] for r in rows}
        for future in as_completed(futures):
            value=future.result();done[value['id']]=value
            C.write(output/'frames'/(value['id'].replace(':','__')+'.json'),value)
            print(f'{len(done)}/{len(rows)} {value["id"]} selected={value["selected_family"]}',flush=True)
    for value in (sources,bindings,protected):E._verify_sources(value)
    C.save_rows(output/'PREDICTIONS.jsonl.gz',[done[r['id']] for r in rows])
    C.write(output/'PREDICTION_SEAL.json',dict(rows=len(rows),predictions_sha256=C.sha(output/'PREDICTIONS.jsonl.gz'),
        protocol_sha256=C.sha(output/'PROTOCOL.json'),before_new_reference_metric_read=True,
        source_sha256=sources,input_sha256=bindings,elapsed_seconds=time.monotonic()-started))


def score(output=OUTPUT):
    output=Path(output).resolve();protocol=C.read(output/'PROTOCOL.json');seal=C.read(output/'PREDICTION_SEAL.json')
    assert seal['before_new_reference_metric_read'] and seal['rows']==protocol['count']
    assert C.sha(output/'PREDICTIONS.jsonl.gz')==seal['predictions_sha256'] and C.sha(output/'PROTOCOL.json')==seal['protocol_sha256']
    for key in ('source_sha256','input_sha256','protected_user_sha256'):E._verify_sources(protocol[key])
    rows=list(C.rows(output/'PREDICTIONS.jsonl.gz'));assert [r['id'] for r in rows]==protocol['ids']
    reports={};arms=('ORIGINAL_WD','SWAPPED_WD','SELECTED_JOINT_OUTLINE')
    for part in ('CAL','TEST'):
        population=[r for r in rows if r['partition']==part]
        if not population:continue
        target=E.COHORT/('OOF_pose_v1/ISOLATED_CAL_SCORING.json' if part=='CAL' else 'test_reused_v1/ISOLATED_TEST_SCORING.json')
        labels=C.read(target)['rows'];scored=[]
        for r in population:
            scored.append(dict(id=r['id'],selected_family=r['selected_family'],
                metrics={a:pose_metric(r['arms'][a]['actual_pose'],labels[r['id']]['target']['truth']) for a in arms},
                accepted={a:r['arms'][a]['accepted'] for a in arms},ambiguous={a:r['arms'][a]['ambiguity_audit']['material_ambiguity'] for a in arms},
                objectives={f['name']:f['objective'] for f in r['families']}))
        report=dict(partition=part,rows=scored,population=protocol['population'],all_selected_cases_kept=True,
            summary={a:dict(translation_cm=distribution([r['metrics'][a]['translation_cm'] for r in scored]),
                rotation_deg=distribution([r['metrics'][a]['rotation_deg'] for r in scored]),
                translation_le5=sum(r['metrics'][a]['translation_cm']<=5 for r in scored),
                successes=sum(r['metrics'][a]['success_5cm_5deg'] for r in scored),
                fallback_count=sum(not r['accepted'][a] for r in scored)) for a in arms},target_binding=E._binding(target),
            no_GT_pose_fit_or_runtime_choice=True)
        C.write(output/f'{part}_SCORE.json',report);reports[part]=report
    C.write(output/'SCORE_SEAL.json',dict(source_prediction_seal=E._binding(output/'PREDICTION_SEAL.json'),
        score_files={part:C.sha(output/f'{part}_SCORE.json') for part in reports},
        target_subset_result_is_not_TEST49_completion=protocol['population']=='targets'))
    print({p:r['summary'] for p,r in reports.items()},flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('command',choices=('predict','score'))
    parser.add_argument('--population',choices=('targets','all95'),default='targets');parser.add_argument('--output',type=Path,default=OUTPUT)
    parser.add_argument('--workers',type=int,default=3);args=parser.parse_args()
    predict(args.output,args.population,args.workers) if args.command=='predict' else score(args.output)
