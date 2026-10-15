"""Cached ORIGINAL-model grazing-U P1/P0 paired95 evaluation.

The existing two-degree geometry tolerance defines a fixed initial uncertainty
band before any new local observation query. Historical V47 image observations
are cached: uncertain BI references and partial factors are not reused, nor
are factors propagated from an uncertain query endpoint. No RGB classifier,
mask network, correction or line fitter is run. Actual RGB is decoded only to
validate/rebuild retained OT image bindings. All95 predictions seal before GT
scoring. This is an iterative reused TEST experiment, not fresh generalization.
"""
from __future__ import annotations
import os
for _key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[_key]='1'

import argparse
from concurrent.futures import ProcessPoolExecutor,as_completed
from copy import deepcopy
from multiprocessing import get_context
from pathlib import Path
import shutil
import time

import cv2

from . import common as C
from . import statewise_evaluation_v4 as E
from . import evaluate_visibility_pose_v4_8 as V48
from . import evaluate_shared_edge_partial_b_v4_7 as V47
from . import evaluate_u_regions_v4_5 as P
from .geometric_visibility_uncertainty_v4_9 import apply_geometric_visibility_uncertainty,filter_grazing_observation_reuse
from .statewise_uncertain_solver_v4_bias import _configuration
from .statewise_partial_boundary_u_solver_v4_6 import solve_boundary_u_region_joint_ot_pnp
from .uncertain_u_regions_v4_5 import build_u_regions
from .ot_domains_v2 import build_ot_domains,validate_domain_bindings
from .statewise_uncertainty_calibration_v4 import _reference_alignment
from .scoring import pose_metric

ARMS=('SAME_V47','GRAZING_U_PRIOR1','GRAZING_U_PRIOR0')
OUTPUT=P.BOUNDARY/'grazing_visibility_pose_v4_9'
TARGET='eval_noapril:1775201430551447040'


def _inputs():
    rows,model,config,bindings,parent=V48._load()
    previous=C.read(V48.OUTPUT/'PROTOCOL.json');complete=C.read(V48.OUTPUT/'PREDICTION_COMPLETE.json')
    E._verify_sources(previous['source_sha256']);E._verify_sources(previous['input_sha256'])
    if complete['rows']!=95:raise ValueError('Complete cached V48 pair required')
    pairs={}
    for part,count in (('CAL',46),('TEST',49)):
        path=V48.OUTPUT/f'{part}_PREDICTIONS.jsonl.gz';seal_path=V48.OUTPUT/f'{part}_PREDICTION_SEAL.json';seal=C.read(seal_path)
        if (seal['rows']!=count or not seal['before_new_annotation_score']
                or C.sha(path)!=seal['predictions_sha256']
                or C.sha(V48.OUTPUT/'PROTOCOL.json')!=seal['protocol_sha256']):raise ValueError('V48 prediction seal changed')
        values=list(C.rows(path));expected=[r['id'] for r in rows if r['partition']==part]
        if [r['id'] for r in values]!=expected:raise ValueError('Exact ordered cached95 required')
        pairs.update({r['id']:r for r in values})
        for p in (path,seal_path):bindings[str(p.relative_to(C.ROOT))]=C.sha(p)
    for p in (V48.OUTPUT/'PROTOCOL.json',V48.OUTPUT/'PREDICTION_COMPLETE.json'):
        bindings[str(p.relative_to(C.ROOT))]=C.sha(p)
    ot_path=E.STUDY/'ot_domains/EXTERNAL_O_CAL_MAX_MODEL_N3_SUBPIX.json'
    ot_model=C.read(ot_path);region_model=C.read(P.REGION_MODEL)
    for p in (ot_path,P.REGION_MODEL):bindings[str(p.relative_to(C.ROOT))]=C.sha(p)
    return rows,model,config,bindings,previous,pairs,ot_model,region_model


def _raw_nonS(source):
    raw=source['original_raw_points'];hidden=source['initial_geometry']['initial_self_hidden']
    result=[]
    for k in range(8):
        result.append([-1.,-1.] if hidden[k] else list(raw[k]))
    if len(raw)==9:result.append(list(raw[8]))
    return result


def _adapt(row,config,ot_model,region_model):
    source=deepcopy(row['_original_source']);g=source['initial_geometry']
    gate=apply_geometric_visibility_uncertainty(g,source['states'],
        geometry_tolerance_deg=_configuration(config)['geometry_tolerance_deg'],
        state_records=source['state_observation']['records'])
    reuse=filter_grazing_observation_reuse(row['frozen_BI_fit'],
        row['partial_boundary_observation']['boundary_observations'],gate['states'],gate['new_U_native_ids'])
    source['states']=gate['states']
    source['state_observation']['states']=gate['states']
    partial={**deepcopy(row['partial_boundary_observation']),
        'boundary_observations':deepcopy(reuse['boundary_observations']),
        'grazing_geometry_reuse_filter':deepcopy(reuse)}
    if gate['new_U_native_ids']:
        image_path=C.ROOT/row['image']
        if C.sha(image_path)!=row['image_sha256']:raise ValueError('RGB binding changed')
        image=cv2.imread(str(image_path))
        if image is None or list(image.shape[:2])!=g['image_hw']:raise ValueError('Actual canvas changed')
        raw=_raw_nonS(source);old=row['_original_source']['ot_domains'];old_states=row['_original_source']['states']
        kwargs=dict(prediction_source=old['binding']['prediction_source'],record_id=old['binding']['record_id'],self_hidden=g['initial_self_hidden'])
        validate_domain_bindings(old,image,raw,old_states,g['K'],g['native_xyz'],**kwargs)
        sides={}
        for rec in source['state_observation']['records']:
            if gate['states'][rec['native_id']]=='T' and rec.get('physical_truncation_certified'):
                v=rec.get('outside_canvas_side',[]);v=[v] if isinstance(v,str) else v
                if len(v)==1:sides[rec['native_id']]=v[0]
        rebuilt=build_ot_domains(image,raw,gate['states'],g['K'],g['native_xyz'],ot_model,truncation_sides=sides,**kwargs)
        validate_domain_bindings(rebuilt,image,raw,gate['states'],g['K'],g['native_xyz'],**kwargs)
        if rebuilt['nonself_prediction_scale_px']!=old['nonself_prediction_scale_px']:raise ValueError('Frozen nonS scale changed')
        old_by={d['native_corner']:d for d in old['domains']};keys=('shape','center_xy','half_axes_px','angle_deg','halfplanes','outside_canvas_side','canvas_hw')
        for d in rebuilt['domains']:
            if d['native_corner'] not in old_by or any(d.get(k)!=old_by[d['native_corner']].get(k) for k in keys):
                raise ValueError('Retained OT numeric domain geometry changed')
        source['ot_domains']=rebuilt
    regions=build_u_regions(source,region_model)
    fit=deepcopy(reuse['visible'])
    if any(source['states'][m['native_corner']] not in ('B','I') for m in fit):raise ValueError('New U retains precise BI term')
    source['grazing_visibility_uncertainty']=gate
    return source,fit,partial,regions,gate,reuse


def _spec(source,fit,partial,regions,config):
    return {**E._physics_spec(source,fit,config),
        'boundary_observations':deepcopy(partial['boundary_observations']),
        'u_regions':deepcopy(regions['u_regions']),'boundary_factor_weight':1.}


def _math(spec):
    # Reporting/image binding metadata is independently sealed. Hash every
    # numeric solver-read input, including inactive observed Sigma and Q.
    result=P.mathematical_spec(spec)
    result['boundary_observations']=[{k:deepcopy(b[k]) for k in ('native_corner','state','lines','fit_reliability_weight','replaces_raw_B_likelihood')}
        for b in spec['boundary_observations']]
    for b in result['boundary_observations']:
        b['lines']=[{k:deepcopy(l[k]) for k in ('normal_unit_xy','offset_px','normal_variance_px2','support_tangent_interval_px','support_extrapolation_px')} for l in b['lines']]
    result['u_regions']=[{k:deepcopy(r[k]) for k in ('native_corner','state','shape','center_xy','half_axes_px','angle_deg')} for r in spec['u_regions']]
    return result


def _run(task):
    row,pair,model,config,ot_model,region_model=task;started=time.monotonic()
    source,fit,partial,regions,gate,reuse=_adapt(row,config,ot_model,region_model)
    arms={ARMS[0]:deepcopy(pair['arms'][V48.ARMS[0]])};audits={};calls=0
    for arm,weight,old_arm in ((ARMS[1],1.,V48.ARMS[0]),(ARMS[2],0.,V48.ARMS[1])):
        cfg={**config,'prior_weight':weight};spec=_spec(source,fit,partial,regions,cfg)
        old_spec=V48._spec(row,cfg);new_key=E._digest(_math(spec));old_key=E._digest(_math(old_spec))
        immutable=E._digest(spec)
        if new_key==old_key:solution=deepcopy(pair['arms'][old_arm]['solution']);cached=True
        else:solution=solve_boundary_u_region_joint_ot_pnp(**spec);calls+=1;cached=False
        if E._digest(spec)!=immutable:raise ValueError('Grazing pose solve modified immutable inputs')
        arms[arm]=V47._finish(source,fit,model,solution,regions['u_regions'],arm,partial)
        audits[arm]=dict(solver_mathematical_input_sha256=new_key,cached_original_input_sha256=old_key,exact_mathematical_input_reuse=cached,prior_weight=weight)
    return C.finite(dict(id=row['id'],partition=row['partition'],session=row['session'],cohort=row['cohort'],
        image=row['image'],image_sha256=row['image_sha256'],states=source['states'],original_states=row['states'],
        runtime={'initial_geometry':source['initial_geometry']},grazing_gate=gate,reuse_filter_audit=reuse,
        frozen_BI_fit=fit,partial_boundary_observation=partial,u_region_observation=regions,ot_domains=source['ot_domains'],
        arms=arms,solver_cache_audit=audits,pose_solver_calls=calls,elapsed_seconds=time.monotonic()-started,
        historical_local_queries_may_have_occurred=True,new_uncertain_RGB_or_line_records_reused=False,
        new_RGB_state_mask_correction_or_line_forwards=0,S_new_local_queries=0,U_new_local_queries=0,
        GT_inputs=False,original_coarse_Q_geometry_scale_and_retained_noise_unchanged=True))


def predict(output=OUTPUT,workers=3):
    output=Path(output).resolve()
    if output.exists():raise FileExistsError('Preserve previous run: '+str(output))
    rows,model,config,bindings,parent,pairs,ot_model,region_model=_inputs()
    sources=deepcopy(parent['source_sha256'])
    for name in (Path(__file__).name,'geometric_visibility_uncertainty_v4_9.py'):
        path=Path(__file__).parent/name;sources[str(path.relative_to(C.ROOT))]=C.sha(path)
    protected=parent['protected_user_sha256'];output.mkdir(parents=True)
    protocol=dict(schema='CACHED_GRAZING_VISIBILITY_U_PAIRED95_V4_9',arms=list(ARMS),ids=[r['id'] for r in rows],
        primary_population={'CAL':46,'TEST':49,'severe':0},source_sha256=sources,input_sha256=bindings,protected_user_sha256=protected,
        solver_configuration=config,uncertainty_tolerance_deg=_configuration(config)['geometry_tolerance_deg'],
        tolerance_origin='existing declared solver geometry tolerance; no new fitted cutoff',
        initial_S_fixed=True,certified_T_preserved=True,nonS_grazing_BI_OT_to_U=True,
        uncertainty_local_likelihood_and_propagated_source_query_factors_not_reused=True,
        observations_cached_not_fresh_full_RGB_API=True,historical_uncertain_queries_may_have_occurred=True,
        raw_nonS_U_circle_from_same_CAL264_max=True,no_U_exact_point_line_canvas_or_visibility_terms=True,
        retained_OT_rebuilt_for_changed_state_bindings=True,retained_noise_budgets_and_Q_and_scale_fixed=True,
        no_GT_or_frame_ID_routing=True,labels_read_only_after_complete95_seal=True,
        TEST_is_reused_iterative_exploratory=True,workers=workers,CPU_threads_per_worker=1)
    C.write(output/'PROTOCOL.json',protocol)
    for relative,digest in sources.items():
        dest=output/'code_snapshot'/relative;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(C.ROOT/relative,dest)
        if C.sha(dest)!=digest:raise ValueError('Source snapshot differs')
    started=time.monotonic();completed={}
    # Evaluate the requested case early, but retain the original full ordered
    # population in final journals. Target ID only changes execution order.
    tasks=sorted(rows,key=lambda r:r['id']!=TARGET)
    with ProcessPoolExecutor(max_workers=workers,mp_context=get_context('spawn')) as pool:
        futures={pool.submit(_run,(r,pairs[r['id']],model,config,ot_model,region_model)):r['id'] for r in tasks}
        for future in as_completed(futures):
            value=future.result();completed[value['id']]=value
            C.write(output/f"{value['partition']}_frames"/(value['id'].replace(':','__')+'.json'),value)
            C.write(output/'LIVE_PROGRESS.json',dict(completed=len(completed),total=95,elapsed_seconds=time.monotonic()-started,latest_id=value['id']))
            print(f"{len(completed)}/95 {value['id']} newU={value['grazing_gate']['new_U_native_ids']} calls={value['pose_solver_calls']}",flush=True)
    E._verify_sources(sources);E._verify_sources(bindings);E._verify_sources(protected);seals={}
    for p,count in (('CAL',46),('TEST',49)):
        values=[completed[r['id']] for r in rows if r['partition']==p];journal=output/f'{p}_PREDICTIONS.jsonl.gz';C.save_rows(journal,values)
        if len(values)!=count:raise ValueError('Complete primary counts required')
        C.write(output/f'{p}_PREDICTION_SEAL.json',dict(rows=count,predictions_sha256=C.sha(journal),protocol_sha256=C.sha(output/'PROTOCOL.json'),
            source_sha256=sources,input_sha256=bindings,before_new_annotation_score=True,all95_finished_before_seal=True))
        seals[p]=C.sha(output/f'{p}_PREDICTION_SEAL.json')
    C.write(output/'PREDICTION_COMPLETE.json',dict(rows=95,CAL=46,TEST=49,severe=0,elapsed_seconds=time.monotonic()-started,
        partition_seal_sha256=seals,pose_solver_calls=sum(r['pose_solver_calls'] for r in completed.values()),before_any_new_pose_scoring=True))


def score(output=OUTPUT):
    output=Path(output).resolve();protocol=C.read(output/'PROTOCOL.json');complete=C.read(output/'PREDICTION_COMPLETE.json')
    if complete['rows']!=95 or not complete['before_any_new_pose_scoring']:raise ValueError('Seal complete95 first')
    E._verify_sources(protocol['source_sha256']);E._verify_sources(protocol['input_sha256']);E._verify_sources(protocol['protected_user_sha256'])
    predictions={}
    for p,count in (('CAL',46),('TEST',49)):
        path=output/f'{p}_PREDICTIONS.jsonl.gz';seal_path=output/f'{p}_PREDICTION_SEAL.json';seal=C.read(seal_path)
        if (seal['rows']!=count or not seal['before_new_annotation_score'] or C.sha(path)!=seal['predictions_sha256']
                or C.sha(output/'PROTOCOL.json')!=seal['protocol_sha256'] or C.sha(seal_path)!=complete['partition_seal_sha256'][p]):raise ValueError('All95 seal mismatch')
        predictions[p]=list(C.rows(path))
        parent={r['id']:r for r in C.rows(V48.OUTPUT/f'{p}_PREDICTIONS.jsonl.gz')}
        if ([r['id'] for r in predictions[p]]!=[i for i in protocol['ids'] if i in parent]
                or any(r['arms'][ARMS[0]]!=parent[r['id']]['arms'][V48.ARMS[0]] for r in predictions[p])):raise ValueError('Ordered all-frame/cached baseline parity failed')
    reports={}
    for p in ('CAL','TEST'):
        target=E.COHORT/('OOF_pose_v1/ISOLATED_CAL_SCORING.json' if p=='CAL' else 'test_reused_v1/ISOLATED_TEST_SCORING.json');labels=C.read(target)['rows'];values=[]
        for r in predictions[p]:
            label=labels[r['id']];alignment=_reference_alignment(r,label)
            values.append(dict(id=r['id'],image=r['image'],states=r['states'],original_states=r['original_states'],new_U_native_ids=r['grazing_gate']['new_U_native_ids'],
                metrics={a:pose_metric(r['arms'][a]['actual_pose'],label['target']['truth']) for a in ARMS},accepted={a:r['arms'][a]['accepted'] for a in ARMS},
                ambiguous={a:r['arms'][a]['ambiguity_audit']['material_ambiguity'] for a in ARMS},physical_native_compatible=bool(alignment['compatible'])))
        report=dict(schema='GRAZING_VISIBILITY_U_POSTSEAL95_SCORE_V4_9',partition=p,frames=len(values),severe=0,
            all_fallbacks_and_native_incompatible_frames_kept=True,summary_all={a:V48._summary(values,a) for a in ARMS},rows=values,
            target=next((r for r in values if r['id']==TARGET),None),target_binding=E._binding(target),prediction_seal=E._binding(output/f'{p}_PREDICTION_SEAL.json'),
            TEST_is_reused_iterative_exploratory=True)
        C.write(output/f'{p}_POSE_SCORE.json',report);reports[p]=report
    E._verify_sources(protocol['source_sha256']);E._verify_sources(protocol['input_sha256']);E._verify_sources(protocol['protected_user_sha256'])
    C.write(output/'SCORE_SEAL.json',dict(CAL_score_sha256=C.sha(output/'CAL_POSE_SCORE.json'),TEST_score_sha256=C.sha(output/'TEST_POSE_SCORE.json'),prediction_complete_sha256=C.sha(output/'PREDICTION_COMPLETE.json'),no_GT_inputs_to_adapter_or_inference=True))
    print(C.finite({p:{'summary_all':r['summary_all'],'target':r['target']} for p,r in reports.items()}),flush=True)


def check():
    rows,model,config,bindings,parent,pairs,ot_model,region_model=_inputs();changed=[]
    for r in rows:
        source,fit,partial,regions,gate,reuse=_adapt(r,config,ot_model,region_model)
        if gate['new_U_native_ids']:changed.append((r['id'],gate['new_U_native_ids']))
    print('PREFLIGHT_PASS',len(rows),'changed',len(changed),'source',C.sha(__file__),flush=True)
    print(changed,flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--stage',choices=('check','predict','score','all'),default='all');p.add_argument('--output',type=Path,default=OUTPUT);p.add_argument('--workers',type=int,default=3);a=p.parse_args()
    if a.stage=='check':check()
    elif a.stage=='predict':predict(a.output,a.workers)
    elif a.stage=='score':score(a.output)
    else:predict(a.output,a.workers);score(a.output)


if __name__=='__main__':main()
