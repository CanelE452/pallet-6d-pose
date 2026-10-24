#!/usr/bin/env python3
"""TRAIN-only preserved SYNTH initializer geometry, metadata and math only.

Never opens RGB, masks, original annotation JSON, features, weights, current
data, or a solver. No jitter/uncertainty/crop policy is selected. Reference r is
the renderer cuboid reference proxy, not physical or operational semantic gold.
"""
from pathlib import Path
from collections import Counter,defaultdict
import csv
import hashlib
import json
import math
import sys
import time

import numpy as np

ROOT=Path(__file__).resolve().parent
BASE=ROOT.parent
AUDIT=BASE/'query_visibility_visible_preservation_20261023_v67/source_audit/SYNTH_REFERENCE_VISIBILITY_INTAKE.json'
AUDIT_SHA='28daf4ac39b168e00c8cb5b166e72dbb4affd8c4717434f4772acd591f65a524'
Q_LEVELS=(50,75,90,95,99)


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def ref(path):return dict(path=str(Path(path).resolve()),sha256=sha(path))
def read_json(path):return json.loads(Path(path).read_text())
def require(value,message):
    if not value:raise ValueError(message)
def boolean(x):
    if x in (True,'True','true',1,'1'):return True
    if x in (False,'False','false',0,'0'):return False
    raise ValueError('Explicit bool required')
def ratio(a,b):return int(a)/int(b) if b else None


def quantiles(values):
    a=np.asarray(values,np.float64);a=a[np.isfinite(a)]
    return dict(N=len(a),minimum=float(a.min()) if len(a) else None,
        **{f'p{p}':float(np.quantile(a,p/100,method='linear')) if len(a) else None for p in Q_LEVELS},
        maximum=float(a.max()) if len(a) else None,method='NumPy quantile linear; finite values only')


def point_flags(xy,hw):
    q=np.asarray(xy,np.float64);h,w=hw
    finite=bool(np.isfinite(q).all())
    continuous=finite and 0<=q[0]<w and 0<=q[1]<h
    rounded=continuous and 0<=math.floor(q[0]+.5)<w and 0<=math.floor(q[1]+.5)<h
    return finite,bool(continuous),bool(rounded)


def write_csv(path,rows):
    with Path(path).open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)


def residual_summary(rows):
    finite=[r for r in rows if r['finite_r_q_L']]
    return dict(N=len(rows),finite_residual_N=len(finite),
        norm_q0_minus_r_native_px=quantiles([r['residual_norm_px'] for r in finite]),
        abs_dx_native_px=quantiles([abs(r['dx_px']) for r in finite]),abs_dy_native_px=quantiles([abs(r['dy_px']) for r in finite]),
        norm_q0_minus_r_div_inputL=quantiles([r['residual_norm_over_L'] for r in finite]),
        abs_dx_div_inputL=quantiles([r['abs_dx_over_L'] for r in finite]),abs_dy_div_inputL=quantiles([r['abs_dy_over_L'] for r in finite]),
        r_center_inside_nominal_q_crop=sum(r['r_center_inside_nominal_q_crop'] for r in rows),
        r_center_inside_observed_q_crop=sum(r['r_center_inside_observed_q_crop'] for r in rows),
        point_coverage_scope='Reference coordinate only; no uncertainty bbox is set, so this is not FULL/PARTIAL/MISS availability or semantic visibility.')


def main():
    started=time.monotonic();output=ROOT/'SOURCE_INITIALIZER_RESIDUAL_DIAGNOSIS.json'
    require(not output.exists(),'Preserve previous diagnosis')
    require(sha(AUDIT)==AUDIT_SHA,'Frozen source intake metadata changed')
    audit=read_json(AUDIT)
    require(audit['completed'] is True and audit['new_render_training_forward_current_inference_PnP_or_pose_optimization']==0,'Source metadata intake required')
    paths={name:Path(audit['outputs'][name]['path']) for name in ('anchor_inventory','frame_manifest')}
    for name,path in paths.items():require(sha(path)==audit['outputs'][name]['sha256'],'Source inventory digest changed')
    source_binding=next(x for x in audit['source_bindings'] if Path(x['path']).name=='INPUTS.jsonl')
    inputs_path=Path(source_binding['path']);require(sha(inputs_path)==source_binding['sha256'],'Frozen source input metadata changed')
    with paths['anchor_inventory'].open(newline='') as f:inventory=[r for r in csv.DictReader(f) if r['split'].lower()=='train']
    with paths['frame_manifest'].open() as f:frames={r['id']:r for r in map(json.loads,f) if r['split'].lower()=='train'}
    with inputs_path.open() as f:inputs={r['id']:r for r in map(json.loads,f) if r['source']=='SYNTH' and r['split'].lower()=='train'}
    require(len(inventory)==6144 and len(frames)==len(inputs)==768,'Fixed SYNTH TRAIN metadata cohort required')
    require(set(frames)==set(inputs),'TRAIN frame identity mismatch')
    groups=defaultdict(list);rows=[];flag_mismatches=Counter()
    for saved in inventory:
        frame_id=saved['id'];frame=frames[frame_id];inp=inputs[frame_id];corner=int(saved['native_corner'])
        require(inp['raw_hw']==frame['raw_hw'] and inp['image_sha256']==frame['image_sha256'],'Source metadata native/image identity mismatch')
        require(saved['annotation_sha256']==frame['annotation_sha256'],'Reference annotation lineage mismatch')
        require(len(inp['query'])==8 and 0<=corner<8,'Original native8 query identity required')
        r=np.asarray([float(saved['reference_x']),float(saved['reference_y'])],np.float64)
        q=np.asarray([float(saved['initial_query_x']),float(saved['initial_query_y'])],np.float64)
        require(np.array_equal(q,np.asarray(inp['query'][corner],np.float64),equal_nan=True),'Stored initializer q0 changed')
        L=float(inp['side']);require(math.isfinite(L) and L>0,'Finite actual input L required')
        hw=list(map(int,frame['raw_hw']));rf,ri,rp=point_flags(r,hw);qf,qi,qp=point_flags(q,hw)
        z=boolean(saved['reference_camera_Z_positive']);finite=rf and qf
        dx,dy=(q-r).tolist();norm=float(np.linalg.norm(q-r))
        if finite:
            require(math.isclose(dx,float(saved['query_minus_reference_x']),rel_tol=1e-12,abs_tol=1e-12) and
                math.isclose(dy,float(saved['query_minus_reference_y']),rel_tol=1e-12,abs_tol=1e-12) and
                math.isclose(norm,float(saved['source_coordinate_displacement_px']),rel_tol=1e-12,abs_tol=1e-12),'Preserved residual metadata does not replay')
        inside=finite and q[0]-L/2<=r[0]<q[0]+L/2 and q[1]-L/2<=r[1]<q[1]+L/2
        for key,actual in (('reference_continuous_inframe',ri),('reference_nearestpixel_valid',rp),('initial_query_nearestpixel_valid',qp)):
            flag_mismatches[key]+=int(boolean(saved[key])!=actual)
        require(saved['certified_semantic_D_E_S']=='','No independently certified semantic state may be invented')
        row=dict(id=frame_id,split='TRAIN',native_corner=corner,native_H=hw[0],native_W=hw[1],input_L=L,
            reference_x=float(r[0]),reference_y=float(r[1]),initial_query_x=float(q[0]),initial_query_y=float(q[1]),
            dx_px=dx,dy_px=dy,residual_norm_px=norm,residual_norm_over_L=norm/L,abs_dx_over_L=abs(dx)/L,abs_dy_over_L=abs(dy)/L,
            finite_reference=rf,finite_query=qf,finite_r_q_L=finite,reference_continuous_inframe=ri,reference_nearest_valid=rp,
            query_continuous_inframe=qi,query_nearest_valid=qp,reference_positive_camera_Z=z,
            r_center_inside_nominal_q_crop=bool(inside),r_center_inside_observed_q_crop=bool(inside and ri),
            conditional_proxy_state=saved['diagnostic_conditional_proxy_state'],convex_self_visible_proxy=boolean(saved['convex_self_visible_proxy']),
            reference_external_9ray_fraction=float(saved['reference_external_9ray_fraction']),
            teacher_tier=saved['teacher_tier'],source_asset=frame['source_asset'],annotation_sha256=saved['annotation_sha256'])
        rows.append(row);groups[frame_id].append(row)
    require(all(len(g)==8 and {r['native_corner'] for r in g}==set(range(8)) for g in groups.values()),'Every TRAIN frame must retain original native8')
    cohorts={'ALL_TRAIN_NATIVE8':rows,
        'FINITE_R_Q': [r for r in rows if r['finite_r_q_L']],
        'REFERENCE_CONTINUOUS_INFRAME_ZPOS':[r for r in rows if r['reference_continuous_inframe'] and r['reference_positive_camera_Z']],
        'BOTH_REFERENCE_QUERY_CONTINUOUS_INFRAME_ZPOS':[r for r in rows if r['reference_continuous_inframe'] and r['query_continuous_inframe'] and r['reference_positive_camera_Z']],
        'BOTH_REFERENCE_QUERY_NEAREST_VALID_ZPOS':[r for r in rows if r['reference_nearest_valid'] and r['query_nearest_valid'] and r['reference_positive_camera_Z']]}
    for state in sorted({r['conditional_proxy_state'] for r in rows}):cohorts['CONDITIONAL_PROXY/'+state]=[r for r in rows if r['conditional_proxy_state']==state]
    summaries={name:residual_summary(rr) for name,rr in cohorts.items()}
    pairs=[]
    for frame_id,g in sorted(groups.items()):
        g=sorted(g,key=lambda r:r['native_corner'])
        for i,a in enumerate(g):
            for b in g[i+1:]:
                finite=a['finite_reference'] and b['finite_reference'];distance=math.hypot(a['reference_x']-b['reference_x'],a['reference_y']-b['reference_y']) if finite else float('nan')
                qdistance=math.hypot(a['initial_query_x']-b['initial_query_x'],a['initial_query_y']-b['initial_query_y'])
                state_pair='|'.join(sorted((a['conditional_proxy_state'],b['conditional_proxy_state'])))
                pairs.append(dict(id=frame_id,slot_a=a['native_corner'],slot_b=b['native_corner'],
                    conditional_proxy_a=a['conditional_proxy_state'],conditional_proxy_b=b['conditional_proxy_state'],
                    conditional_proxy_pair=state_pair,different_conditional_proxy_states=a['conditional_proxy_state']!=b['conditional_proxy_state'],
                    finite_references=finite,both_reference_continuous_inframe_Zpos=a['reference_continuous_inframe'] and b['reference_continuous_inframe'] and a['reference_positive_camera_Z'] and b['reference_positive_camera_Z'],
                    reference_pair_distance_px=distance,reference_pair_distance_over_L=distance/a['input_L'],
                    initial_query_pair_distance_px=qdistance,input_L=a['input_L'],
                    evidence='Conditional renderer cuboid/ray/convex proxy only; not distinct verified semantic states'))
    require(len(pairs)==768*28,'All same-image native8 unordered pairs required')
    def pair_summary(pp):
        return dict(N=len(pp),distance_px=quantiles([r['reference_pair_distance_px'] for r in pp]),distance_over_L=quantiles([r['reference_pair_distance_over_L'] for r in pp]))
    pair_cohorts={'ALL_FINITE_PAIRS':[r for r in pairs if r['finite_references']],
        'BOTH_REFERENCE_INFRAME_ZPOS':[r for r in pairs if r['both_reference_continuous_inframe_Zpos']],
        'BOTH_INFRAME_ZPOS_DIFFERENT_CONDITIONAL_STATES':[r for r in pairs if r['both_reference_continuous_inframe_Zpos'] and r['different_conditional_proxy_states']],
        'BOTH_INFRAME_ZPOS_SAME_CONDITIONAL_STATE':[r for r in pairs if r['both_reference_continuous_inframe_Zpos'] and not r['different_conditional_proxy_states']]}
    by_pair={key:pair_summary([r for r in pairs if r['conditional_proxy_pair']==key and r['both_reference_continuous_inframe_Zpos']]) for key in sorted({r['conditional_proxy_pair'] for r in pairs})}
    closest=sorted(pair_cohorts['BOTH_INFRAME_ZPOS_DIFFERENT_CONDITIONAL_STATES'],key=lambda r:(r['reference_pair_distance_px'],r['id'],r['slot_a'],r['slot_b']))[:20]
    point_path=ROOT/'SOURCE_INITIALIZER_RESIDUAL_DIAGNOSIS.csv';pair_path=ROOT/'SOURCE_INITIALIZER_REFERENCE_PAIRS.csv';summary_path=ROOT/'SOURCE_INITIALIZER_RESIDUAL_SUMMARY.csv'
    for p in (point_path,pair_path,summary_path):require(not p.exists(),'Preserve prior diagnostic output')
    write_csv(point_path,rows);write_csv(pair_path,pairs)
    flat=[]
    for cohort,s in summaries.items():
        for name,value in s.items():
            if isinstance(value,dict) and 'p50' in value:flat.append(dict(cohort=cohort,metric=name,**{k:v for k,v in value.items() if k!='method'}))
    write_csv(summary_path,flat)
    report=dict(schema='V69_SOURCE_TRAIN_ONLY_PRESERVED_INITIALIZER_METADATA_DIAGNOSIS_V1',completed=True,
        source='SYNTH TRAIN ONLY',frames=768,native8_reference_rows=6144,pairs=len(pairs),
        counts=dict(finite_reference=sum(r['finite_reference'] for r in rows),finite_query=sum(r['finite_query'] for r in rows),
            reference_continuous_inframe=sum(r['reference_continuous_inframe'] for r in rows),reference_nearest_valid=sum(r['reference_nearest_valid'] for r in rows),
            query_continuous_inframe=sum(r['query_continuous_inframe'] for r in rows),query_nearest_valid=sum(r['query_nearest_valid'] for r in rows),
            reference_positive_camera_Z=sum(r['reference_positive_camera_Z'] for r in rows),
            q_nearest_invalid_while_reference_continuous_inframe=sum(not r['query_nearest_valid'] and r['reference_continuous_inframe'] for r in rows),
            reference_center_inside_nominal_q_crop=sum(r['r_center_inside_nominal_q_crop'] for r in rows),
            reference_center_inside_observed_q_crop=sum(r['r_center_inside_observed_q_crop'] for r in rows)),
        residuals_by_cohort=summaries,residuals_by_native_slot={str(k):residual_summary([r for r in rows if r['native_corner']==k]) for k in range(8)},
        reference_pair_diagnostics={k:pair_summary(v) for k,v in pair_cohorts.items()},reference_pairs_by_conditional_proxy_state_pair=by_pair,
        closest20_different_conditional_proxy_pairs=closest,
        pair_limit='Small 2D separation of differing conditional states flags possible association ambiguity. It does not certify different semantic gold labels, identical model inputs, or an actual model failure.',
        metadata_flag_replay_mismatches=dict(flag_mismatches),
        input_L='Exact preserved per-frame V63 source input side; no L recomputation or selection.',
        nominal_crop_point_test='r in [q0_x-L/2,q0_x+L/2) x [q0_y-L/2,q0_y+L/2); observed additionally requires continuous r inside original H/W.',
        uncertainty_availability=dict(uncertainty_bbox_set=False,FULL_PARTIAL_MISS_computed=False,
            reason='No reference/cue uncertainty has been admitted. A center-point crop test is not uncertainty-region availability or semantic observability.'),
        reference_limit='Renderer projected cuboid reference proxy, camera_dynamic_0123_v4/native index; not physical mesh/corner or independently certified operational semantic GT.',
        conditional_states_limit='Legacy 9ray+convex classes only; semantic D/E/S gold remains zero.',
        source_initializer='Preserved actual N3/SubPix initial q0. No fresh jitter samples created.',
        jitter_recipe_selected=False,numeric_jitter_radius_selected=False,policy_or_threshold_selected=False,source_label_admission='HOLD',
        validation_or_current_residual_calibration=False,current_RGB_GT_scores_inputs_read=False,
        source_RGB_masks_depth_features_weights_or_original_annotation_JSON_reads=0,model_forward_calls=0,optimizer_updates=0,PnP_calls=0,
        previous_V69_draft_code_tests_unchanged=True,
        bindings=dict(source_intake_audit=ref(AUDIT),anchor_inventory=ref(paths['anchor_inventory']),frame_manifest=ref(paths['frame_manifest']),
            preserved_source_input_metadata=ref(inputs_path),code=ref(__file__)),
        outputs=[ref(p) for p in (point_path,pair_path,summary_path)],runtime=dict(python=sys.executable,numpy=np.__version__),seconds=time.monotonic()-started)
    output.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print(json.dumps(dict(completed=True,report=ref(output),counts=report['counts'],all_finite=summaries['FINITE_R_Q'])),flush=True)


if __name__=='__main__':main()
