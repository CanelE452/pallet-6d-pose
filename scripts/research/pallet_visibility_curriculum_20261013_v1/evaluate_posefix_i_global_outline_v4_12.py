"""Fixed cached PF / half-ensemble fallback-I controls on all original95.

Use the exact sealed9c observations and v4.11 analytic global outline recipe.
Only eligible original fallback-I xy changes. Two controls are fixed before
any new score, with original covariance/reliability and all B/OT/U/partial
constraints retained. Old fitted poses remain starts, not pseudo-observations.
No model/RGB/line forward, GT selector or noise fit occurs in prediction.
TEST is reused exploratory; previous four-case diagnostics were already seen.
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
from . import evaluate_all_visible_global_outline_v4_11 as G11
from .visible_i_posefix_alternatives_v4_12 import apply_posefix_fallback_I,MODES
from .statewise_partial_boundary_u_solver_v4_6 import solve_boundary_u_region_joint_ot_pnp
from .scoring import pose_metric,distribution

OUTPUT=G11.OUTPUT.parent/'posefix_fallback_I_outline_eval_v4_12'
PF_SOURCE=E.COHORT/'posefix_sub_comparison_v1'
ARMS=('SAME_G11',*MODES)


def _load():
    rows,model,config,bindings,parent=G11._load('all95')
    protocol=C.read(G11.OUTPUT/'PROTOCOL.json');seal=C.read(G11.OUTPUT/'PREDICTION_SEAL.json')
    path=G11.OUTPUT/'PREDICTIONS.jsonl.gz'
    if (seal['rows']!=95 or not seal['before_new_reference_metric_read'] or C.sha(path)!=seal['predictions_sha256']
            or C.sha(G11.OUTPUT/'PROTOCOL.json')!=seal['protocol_sha256']):raise ValueError('Complete sealed G11 baseline95 required')
    for key in ('source_sha256','input_sha256','protected_user_sha256'):E._verify_sources(protocol[key])
    baseline=list(C.rows(path))
    if [r['id'] for r in baseline]!=[r['id'] for r in rows]:raise ValueError('Exact ordered G11 population required')
    baselines={r['id']:r for r in baseline}
    for row in rows:
        b=baselines[row['id']]
        if (b['states']!=row['states'] or b['image_sha256']!=row['image_sha256']
                or b['frozen_BI_fit']!=row['frozen_BI_fit']
                or b['partial_boundary_observation']!=row['partial_boundary_observation']
                or b['u_region_observation']!=row['u_region_observation']):raise ValueError('Same G11 observation inputs required')
    pfseal=C.read(PF_SOURCE/'PREDICTION_SEAL.json');pfprotocol=C.read(PF_SOURCE/'PROTOCOL.json')
    pfpath=PF_SOURCE/'PREDICTIONS.jsonl.gz'
    if (not pfseal['complete'] or not pfseal['before_scoring'] or pfseal['rows']!=665
            or C.sha(pfpath)!=pfseal['predictions_sha256'] or C.sha(PF_SOURCE/'PROTOCOL.json')!=pfseal['protocol_sha256']):
        raise ValueError('Frozen PF comparison seal changed')
    pfs={r['id']:r for r in C.rows(pfpath) if r['method']=='POSEFIX_RAW'}
    if set(pfs)!={r['id'] for r in rows}:raise ValueError('Exact PF95 coverage required')
    # Original PF protocol source hashes are immutable except its explicitly
    # pre-inference namespace repair, which is pinned separately and not run.
    pfbindings=deepcopy(pfprotocol['source_bindings']);repairpath=PF_SOURCE/'PRE_INFERENCE_REPAIR.json'
    if repairpath.is_file():
        repair=C.read(repairpath);runner=str(PF_SOURCE/'compare.py')
        if (repair['protocol_sha256']!=C.sha(PF_SOURCE/'PROTOCOL.json')
                or repair['old_runner_sha256']!=pfbindings[runner]
                or repair['new_runner_sha256']!=C.sha(runner)):raise ValueError('PF historical repair binding mismatch')
        pfbindings[runner]=repair['new_runner_sha256']
    for name,digest in pfbindings.items():
        if C.sha(name)!=digest:raise ValueError('PF frozen input/source changed '+name)
        bindings[str(Path(name).relative_to(C.ROOT))]=digest
    pfinputs=C.read(PF_SOURCE/'BASELINE_INPUTS.json')['frames']
    input_by={r['frame']['id']:r['frame'] for r in pfinputs}
    for row in rows:
        f=input_by[row['id']];g=row['runtime']['initial_geometry'];pf=pfs[row['id']]
        if (pf['partition']!=row['partition'] or f['image_sha256']!=row['image_sha256']
                or f['image']!=row['image'] or f['K']!=g['K'] or f['raw_hw']!=g['image_hw']):raise ValueError('PF exact RGB/K/canvas binding mismatch')
    for p in (path,G11.OUTPUT/'PREDICTION_SEAL.json',G11.OUTPUT/'PROTOCOL.json',pfpath,
            PF_SOURCE/'PREDICTION_SEAL.json',PF_SOURCE/'PROTOCOL.json',PF_SOURCE/'BASELINE_PROVENANCE.json'):
        bindings[str(p.relative_to(C.ROOT))]=C.sha(p)
    if repairpath.is_file():bindings[str(repairpath.relative_to(C.ROOT))]=C.sha(repairpath)
    return rows,model,config,bindings,protocol,baselines,pfs


def _run(task):
    row,model,config,baseline,pf=task;started=time.monotonic();arms={ARMS[0]:deepcopy(baseline)};audits={};calls=0
    before=E._digest(row);g=row['runtime']['initial_geometry']
    for mode in MODES:
        adapted=apply_posefix_fallback_I(row['frozen_BI_fit'],g,row['states'],pf,mode)
        fresh=deepcopy(row);fresh['frozen_BI_fit']=adapted['visible_measurements']
        changed=adapted['changed_native_ids'];old={m['native_corner']:m for m in row['frozen_BI_fit']};new={m['native_corner']:m for m in fresh['frozen_BI_fit']}
        if set(old)!=set(new):raise ValueError('Visible IDs changed')
        for k in old:
            for field in ('covariance_px2','fit_covariance_px2','fit_reliability_weight'):
                if old[k].get(field)!=new[k].get(field):raise ValueError('Old observation noise or reliability changed')
            if k not in changed and old[k]!=new[k]:raise ValueError('Ineligible or unchanged B/I record modified')
        if not changed:
            result=deepcopy(baseline);cached=True;unsupported_fallback=False
        elif baseline['mask_observation']['supported']:
            result=G11._run((fresh,model,config));calls+=1;cached=False;unsupported_fallback=False
        else:
            # G11's unsupported-outline branch returns an old saved solution.
            # Changed I requires a genuine new data-only solve, never an old
            # cost/result labelled as if it fit the new observation.
            spec=G11.W.PREVIOUS._spec(fresh,config);cfg={**spec['config'],'prior_weight':0.}
            result_family=G11.W.FAMILY.solve_two_wd_families(spec['xyz'],spec['K'],spec['states'],spec['visible'],spec['domains'],spec['image_hw'],
                spec['coarse_pose'],config=cfg,boundary_observations=spec['boundary_observations'],u_regions=spec['u_regions'],
                solver_callable=solve_boundary_u_region_joint_ot_pnp)
            families=result_family['families'];inner={f['name']:G11.W._family_finish(fresh,model,f,'NO_OUTLINE_NEW_PF_I_DATA_ONLY_'+f['name']) for f in families}
            selected=families[result_family['selected_family_index']]
            inner['SELECTED_JOINT_OUTLINE']=G11.W._family_finish(fresh,model,selected,'NO_OUTLINE_NEW_PF_I_DATA_ONLY',result_family['solution'])
            for f in families:f['solution']=E._compact_solution(f['solution'])
            result=C.finite({**deepcopy(baseline), 'frozen_BI_fit':fresh['frozen_BI_fit'],'families':families,'arms':inner,
                'selected_family':result_family['selected_family'],'selected_model_geometry':result_family['selected_model_geometry'],
                'family_ambiguity':result_family['family_ambiguity'],
                'family_selection':dict(rule='unsupported_outline genuine changed-I data-only full solve; prior0',GT_inputs=False),
                'unsupported_outline_new_BI_solve':True})
            calls+=1;cached=False;unsupported_fallback=True
        arms[mode]=result
        audits[mode]=dict(observation_adapter=adapted,exact_whole_G11_cache_reuse=cached,
            changed_math_fresh_solver_calls=0 if cached else 1,unsupported_outline_new_data_only_solve=unsupported_fallback,
            all_B_and_already_RGB_corrected_I_and_partial_OT_U_geometry_K_Q_exact=True,
            source_old_effective_noise_and_reliability_exact=True,no_independent_fusion_precision_gain=True)
    if E._digest(row)!=before:raise ValueError('PF control mutated sealed original observations')
    return C.finite(dict(id=row['id'],partition=row['partition'],session=row['session'],image=row['image'],image_sha256=row['image_sha256'],
        states=row['states'],initial_geometry=g,arms=arms,alternative_audit=audits,new_arm_solve_calls=calls,
        elapsed_seconds=time.monotonic()-started,GT_inputs=False,S_PF_coordinate_reads=0,B_U_O_T_PF_coordinate_reads=0,
        new_RGB_mask_state_or_line_forward=0,PF_cached_full8_pose_not_used_as_new_initializer=True,
        original9th_center_and_all_nonnative_I_slots_not_consumed=True))


def predict(output=OUTPUT,workers=3):
    output=Path(output).resolve()
    if output.exists():raise FileExistsError('Preserve previous PF experiment '+str(output))
    rows,model,config,bindings,parent,baselines,pfs=_load();sources=deepcopy(parent['source_sha256'])
    helper=Path(__file__).with_name('visible_i_posefix_alternatives_v4_12.py')
    for path in (Path(__file__),helper):sources[str(path.relative_to(C.ROOT))]=C.sha(path)
    protected=parent['protected_user_sha256']
    for records in (sources,bindings,protected):E._verify_sources(records)
    output.mkdir(parents=True)
    protocol=dict(schema='FIXED_PF_FALLBACK_I_ANALYTIC_OUTLINE_PAIRED95_V4_12',arms=list(ARMS),ids=[r['id'] for r in rows],
        primary_population={'CAL':46,'TEST':49,'severe':0},source_sha256=sources,input_sha256=bindings,protected_user_sha256=protected,
        G11_solver_configuration=config,G11_outline_configuration=G11.G.DEFAULT_OUTLINE_CONFIGURATION,
        eligibility='currentI + initialnonS + VISIBLE_BACKBONE_FALLBACK + samePFextent/Q/native3Dslots + finite inframePFpoint',
        fixed_alphas={'I_BACKBONE_FALLBACK_PF':1.,'I_BACKBONE_FALLBACK_MEAN':.5},
        original_conditional_total_covariance_reliability_unchanged=True,existing_B_RGBcorrectedI_partial_OT_U_unchanged=True,
        old_full_family_solutions_are_seeds_not_new_observations=True,unsupported_outline_changedI_requires_new_data_only_full_solve=True,
        cached_PF_pose_used_only_for_CF_template_and_Q_metadata=True,PF_full8_cached_pose_not_new_initializer=True,
        no_training_or_model_or_RGB_or_line_forward=True,S_U_O_T_B_PF_coordinate_reads=0,
        all95_allarms_seal_before_new_score=True,TEST_is_reused_exploratory=True,previous4case_posthoc_diagnosis_already_seen=True,
        no_GT_or_frameID_inference_routing=True,workers=workers,CPU_threads_per_worker=1)
    C.write(output/'PROTOCOL.json',protocol)
    for relative,digest in sources.items():
        dest=output/'code_snapshot'/relative;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(C.ROOT/relative,dest)
        if C.sha(dest)!=digest:raise ValueError('Source snapshot mismatch')
    started=time.monotonic();done={}
    with ProcessPoolExecutor(max_workers=workers,mp_context=get_context('spawn')) as pool:
        futures={pool.submit(_run,(r,model,config,baselines[r['id']],pfs[r['id']])):r['id'] for r in rows}
        for future in as_completed(futures):
            value=future.result();done[value['id']]=value
            C.write(output/'frames'/(value['id'].replace(':','__')+'.json'),value)
            C.write(output/'LIVE_PROGRESS.json',dict(completed=len(done),total=95,latest_id=value['id'],elapsed_seconds=time.monotonic()-started))
            print(f'{len(done)}/95 {value["id"]} calls={value["new_arm_solve_calls"]}',flush=True)
    for records in (sources,bindings,protected):E._verify_sources(records)
    C.save_rows(output/'PREDICTIONS.jsonl.gz',[done[r['id']] for r in rows])
    C.write(output/'PREDICTION_SEAL.json',dict(schema='PF_FALLBACK_I_COMPLETE95_PREDICTION_SEAL_V4_12',rows=95,CAL=46,TEST=49,severe=0,
        predictions_sha256=C.sha(output/'PREDICTIONS.jsonl.gz'),protocol_sha256=C.sha(output/'PROTOCOL.json'),
        before_new_reference_metric_read=True,all3arms_complete_before_score=True,
        source_sha256=sources,input_sha256=bindings,elapsed_seconds=time.monotonic()-started,
        new_arm_solve_calls=sum(r['new_arm_solve_calls'] for r in done.values())))


def score(output=OUTPUT):
    output=Path(output).resolve();protocol=C.read(output/'PROTOCOL.json');seal=C.read(output/'PREDICTION_SEAL.json');journal=output/'PREDICTIONS.jsonl.gz'
    if (seal['rows']!=95 or not seal['all3arms_complete_before_score'] or not seal['before_new_reference_metric_read']
            or C.sha(journal)!=seal['predictions_sha256'] or C.sha(output/'PROTOCOL.json')!=seal['protocol_sha256']):raise ValueError('Seal all95 both fixed PF arms before labels')
    for key in ('source_sha256','input_sha256','protected_user_sha256'):E._verify_sources(protocol[key])
    rows=list(C.rows(journal));baseline={r['id']:r for r in C.rows(G11.OUTPUT/'PREDICTIONS.jsonl.gz')}
    if [r['id'] for r in rows]!=protocol['ids'] or any(r['arms'][ARMS[0]]!=baseline[r['id']] for r in rows):raise ValueError('Ordered all95 baseline parity failed')
    reports={}
    for part,count in (('CAL',46),('TEST',49)):
        target=E.COHORT/('OOF_pose_v1/ISOLATED_CAL_SCORING.json' if part=='CAL' else 'test_reused_v1/ISOLATED_TEST_SCORING.json')
        labels=C.read(target)['rows'];population=[r for r in rows if r['partition']==part]
        if len(population)!=count:raise ValueError('All-frame score population required')
        values=[]
        for r in population:
            metrics={};accepted={};ambiguous={};family={}
            for arm in ARMS:
                result=r['arms'][arm];pose=result['arms']['SELECTED_JOINT_OUTLINE']
                metrics[arm]=pose_metric(pose['actual_pose'],labels[r['id']]['target']['truth'])
                accepted[arm]=pose['accepted'];ambiguous[arm]=pose['ambiguity_audit']['material_ambiguity'];family[arm]=result['selected_family']
            values.append(dict(id=r['id'],metrics=metrics,accepted=accepted,ambiguous=ambiguous,selected_family=family,
                changed_I_ids={mode:r['alternative_audit'][mode]['observation_adapter']['changed_native_ids'] for mode in MODES}))
        summaries={}
        for arm in ARMS:
            summaries[arm]=dict(translation_cm=distribution([r['metrics'][arm]['translation_cm'] for r in values]),
                rotation_deg=distribution([r['metrics'][arm]['rotation_deg'] for r in values]),
                translation_le5=sum(r['metrics'][arm]['translation_cm']<=5 for r in values),
                successes_5cm_5deg=sum(r['metrics'][arm]['success_5cm_5deg'] for r in values),
                fallback_count=sum(not r['accepted'][arm] for r in values),ambiguous_count=sum(r['ambiguous'][arm] for r in values),
                all_translation_le5=all(r['metrics'][arm]['translation_cm']<=5 for r in values),
                translation_over5_rows=[r for r in values if r['metrics'][arm]['translation_cm']>5])
        report=dict(partition=part,count=count,all_fallbacks_and_native_incompatible_cases_kept=True,rows=values,summary=summaries,
            target_binding=E._binding(target),prediction_seal=E._binding(output/'PREDICTION_SEAL.json'),
            no_GT_or_frameID_runtime_selector=True,TEST_is_reused_exploratory=True,no_fitted_threshold_or_covariance=True)
        C.write(output/f'{part}_SCORE.json',report);reports[part]=report
    for key in ('source_sha256','input_sha256','protected_user_sha256'):E._verify_sources(protocol[key])
    C.write(output/'SCORE_SEAL.json',dict(prediction_seal=E._binding(output/'PREDICTION_SEAL.json'),
        score_sha256={part:C.sha(output/f'{part}_SCORE.json') for part in reports}))
    print(C.finite({p:{a:{k:v[k] for k in ('translation_cm','rotation_deg','translation_le5','successes_5cm_5deg','fallback_count')} for a,v in r['summary'].items()} for p,r in reports.items()}),flush=True)


def check():
    rows,model,config,bindings,parent,baselines,pfs=_load();counts={p:dict(fallback_I=0,PF_coordinates_read=0,changed=0) for p in ('CAL','TEST')}
    for row in rows:
        result=apply_posefix_fallback_I(row['frozen_BI_fit'],row['runtime']['initial_geometry'],row['states'],pfs[row['id']],MODES[0])
        counts[row['partition']]['fallback_I']+=sum(m['state']=='I' and m['source']=='VISIBLE_BACKBONE_FALLBACK' for m in row['frozen_BI_fit'])
        counts[row['partition']]['PF_coordinates_read']+=len(result['PF_coordinate_native_ids_read']);counts[row['partition']]['changed']+=len(result['changed_native_ids'])
    print('PF_FALLBACK_I_PREFLIGHT_PASS',len(rows),counts,flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--stage',choices=('check','predict','score','all'),default='all');parser.add_argument('--output',type=Path,default=OUTPUT);parser.add_argument('--workers',type=int,default=3);args=parser.parse_args()
    if args.stage=='check':check()
    elif args.stage=='predict':predict(args.output,args.workers)
    elif args.stage=='score':score(args.output)
    else:predict(args.output,args.workers);score(args.output)
