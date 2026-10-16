"""Finite eight-arm CAL-only independent internal-face RGB coordinate pilot."""
from __future__ import annotations

import argparse
from collections import Counter
from copy import deepcopy
from itertools import product
from pathlib import Path
import time

import cv2
import numpy as np

from . import common as C
from .curriculum import solve_curriculum
from .internal_face_refine import build_internal_context, infer_from_context, DEFAULT_CONFIG
from .local_occlusion_cal import binding, no_scoring_reads
from .scoring import pose_metric, distribution, paired


ARMS = {f"FACE_C{contrast}_R{radius}_S{shift}": dict(minimum_face_contrast=float(contrast),
         normal_search_radius_px=float(radius), maximum_corner_shift_px=float(shift))
        for contrast, radius, shift in product((8,12),(12,20),(6,12))}
REFERENCE = "SEGMENTED_RGB_CURRICULUM"


def infer(args):
    out=Path(args.output);out.mkdir(parents=True,exist_ok=False)
    frames=C.read(args.inputs)['frames']
    if len(frames)!=76 or any(r['partition']!='CAL' or r['population_role']!='DEV' for r in frames):
        raise AssertionError('Isolated CAL76 only')
    seal=C.read(args.upstream/'PREDICTION_SEAL.json')
    if not seal['before_scoring'] or C.sha(args.upstream/'PREDICTIONS.jsonl.gz')!=seal['sha256']:
        raise AssertionError('Frozen primary prediction seal differs')
    primary={r['id']:r for r in C.rows(args.upstream/'PREDICTIONS.jsonl.gz') if r['method']==REFERENCE}
    if set(primary)!={r['id'] for r in frames}:raise AssertionError('Exact primary76 required')
    code=Path(__file__).parent
    paths=[Path(__file__),args.inputs,args.upstream/'PREDICTIONS.jsonl.gz']+[code/name for name in
           ('internal_face_refine.py','boundary_refine.py','appearance.py','curriculum.py','common.py')]
    protocol=dict(schema='CAL_internal_face_RGB_coordinate_grid_v1',frames=76,partition='CAL',arms=ARMS,
        fixed_configuration=DEFAULT_CONFIG,primary_method=REFERENCE,primary_upstream=str(args.upstream),
        primary_expected=dict(translation_cm=116.29147334738165,rotation_deg=19.290009317834865,success_count=37),
        source='Current original RGB, original predicted q8, frozen initial H, initial pose, registered xyz and K',
        target='Non-hull visible three-face cuboid junction; two face-palette transition lines produce an independent image intersection',
        face_visibility='Initial outward normal toward camera, margin2degrees, all face vertices nonself-hidden',
        face_palette='q polygon shrunk by scale.6 (20% per side); robust mode of RGB material-supported encoded-Lab pixels',
        face_centroid_contrast_unit='Euclidean OpenCV encoded-Lab distance8/12, not CIE DeltaE',
        edge_sampling='15..85% of original predicted cube edge; normal radius12/20; two visible faces occupy opposite sides',
        transition_evidence='Both sides must match their own face palette at relative likelihood>=.65 AND current RGB pallet-material support>=.55',
        line_fitting='Reuse outer-boundary robust line mathematics only; no material/background edge observations',
        intersection='At least2 incident supported line families, sine>=.25; reject third-line disagreement>3px',
        geometry_or_palette_failure='Retain original q; initial projection is only geometry/coherence prior, never a fit observation',
        pose='Frozen initial actual body family, H, original SEG appearance; same curriculum visible_threshold=.35 and other defaults; corrected independent RGB points only',
        current_initial_wrong_family_cases_preserved=True,all76_frames_preserved=True,GT_inputs=False,
        new_TEST_reads=0,FINAL_reads=0,declared_before_new_coordinate_or_pose_scoring=True,
        historical_CAL_reused=True,prior_failed_source_head_not_integrated=True,
        limitations=['Face brightness may not separate, holes/texture/occluders may dominate face seeds.',
                     'Palette likelihood and material support do not prove physical depth ownership.',
                     'RGB line intersection can still be a virtual cuboid vertex on rounded/nonconvex pallets.',
                     'Face identities come from supplied initial pose/order and can be wrong; coherence gates only reject detectable inconsistency.',
                     'Shared edge observations are correlated; this is finite CAL development, not independent confirmation.'],
        bindings=[binding(p) for p in paths])
    C.write(out/'PROTOCOL.json',protocol);cv2.setNumThreads(1);begin=time.monotonic();records=[]
    with no_scoring_reads():
        for index,frame in enumerate(frames):
            imagepath=C.ROOT/frame['image']
            if C.sha(imagepath)!=frame['image_sha256']:raise AssertionError('Original CAL RGB differs')
            image=cv2.imread(str(imagepath));original=np.asarray(frame['points']['N3_SUBPIX'],float)
            if image is None:raise FileNotFoundError(str(imagepath))
            q=original[:8];initial=frame['initial_poses']['N3_SUBPIX'];xyz=C.cuboid(*initial['cf_extents'])
            R,t,K=np.asarray(initial['R_cf']),np.asarray(initial['centroid']),np.asarray(frame['K'])
            H=C.self_hidden(xyz,R,t);base=primary[frame['id']]
            if not np.array_equal(q,np.asarray(base['original_points'])) or not np.array_equal(H,base['self_hidden']):
                raise AssertionError('Primary original q/H binding differs')
            reference=deepcopy(base);reference['method']='PRIMARY_V1';records.append(reference)
            context=build_internal_context(image,original,H,xyz,R,t,K)
            for name,config in ARMS.items():
                correction=infer_from_context(context,config);corrected=np.asarray(correction['points'])[:8]
                # Original appearance probabilities/order are deliberately frozen.
                solution=solve_curriculum(corrected,xyz,K,R,t,H,base['appearance'],dict(visible_threshold=.35))
                Q=R.T@np.asarray(initial['R_physical'])
                actual=dict(available=True,R_cf=solution['R'],
                    R_physical=(np.asarray(solution['R'])@Q).tolist(),centroid=solution['t'],
                    cf_extents=initial['cf_extents'],selected_hypothesis=initial['selected_hypothesis'])
                projected=C.project(xyz,solution['R'],solution['t'],K)
                native=original.copy()
                if solution['accepted']:
                    native[:8]=corrected;native[np.flatnonzero(H)]=projected[H]
                records.append(dict(id=frame['id'],session=frame['session'],arm='N3_SUBPIX',method=name,
                    actual_pose=actual,solution=solution,self_hidden=H.tolist(),appearance=deepcopy(base['appearance']),
                    original_points=q.tolist(),corrected_observation_points=corrected.tolist(),
                    native_points=native.tolist(),projected_points=projected.tolist(),correction=correction,
                    projected_points_used_as_observation=False,external_visibility_model_changed=False))
            if not np.array_equal(original,np.asarray(frame['points']['N3_SUBPIX'])):
                raise AssertionError('Original cached q mutated')
            if (index+1)%16==0:print('INTERNAL_FACE_CAL',index+1,round(time.monotonic()-begin,2),flush=True)
    if [binding(p) for p in paths]!=protocol['bindings']:raise AssertionError('Frozen inference dependency changed')
    C.save_rows(out/'PREDICTIONS.jsonl.gz',records)
    C.write(out/'PREDICTION_SEAL.json',dict(complete=True,before_scoring=True,rows=len(records),
        sha256=C.sha(out/'PREDICTIONS.jsonl.gz'),methods=dict(Counter(r['method'] for r in records)),
        original_RGB_images=76,original_q_unchanged=True,GT_reads_during_inference=0,new_TEST_reads=0,FINAL_reads=0,
        initial_pose_projection_as_fit_observation=False,seconds=time.monotonic()-begin))


def pose_summary(rows):
    return dict(frames=len(rows),accepted=sum(r['solution']['accepted'] for r in rows),
        success_count=sum(r['metric']['success_5cm_5deg'] for r in rows),
        metrics={name:distribution([r['metric'][name] for r in rows]) for name in
                 ('translation_cm','rotation_deg','ADDsym_cm')})


def coordinate_summary(rows,metadata):
    changes=[];all_direct=[];reasons=Counter();edges=Counter();faces=Counter()
    for row in rows:
        frame=metadata[row['id']];ann=frame['scoring_metadata'];target=frame['target']
        phase=np.asarray(ann['native_to_reference_corner_ids'],int)
        gt=np.asarray(target['gt'])[:8][phase];valid=np.asarray(target['valid'])[:8][phase]
        original=np.asarray(row['original_points']);corrected=np.asarray(row.get('corrected_observation_points',original))
        before=np.linalg.norm(original-gt,axis=1);after=np.linalg.norm(corrected-gt,axis=1)
        for k,state in enumerate(ann['human_states_native']):
            if state=='DIRECT_VISIBLE' and valid[k] and np.isfinite(before[k]):
                all_direct.append(dict(id=row['id'],corner=k,before=before[k],after=after[k]))
        correction=row.get('correction')
        if correction is None:continue
        for rec in correction['corner_records']:reasons[rec['reason']]+=1
        for rec in correction['edge_records']:edges[rec['reason']]+=1
        for rec in correction['face_records']:faces[rec['reason']]+=1
        for k in correction['observed_refined_ids']:
            changes.append(dict(id=row['id'],session=row['session'],corner=k,human_state=ann['human_states_native'][k],
                valid_reference=bool(valid[k]),before_error_px=float(before[k]) if valid[k] else None,
                after_error_px=float(after[k]) if valid[k] else None,
                shift_px=float(np.linalg.norm(corrected[k]-original[k])),
                original_xy=original[k].tolist(),corrected_RGB_xy=corrected[k].tolist()))
    tested=[r for r in changes if r['valid_reference']]
    direct=[r for r in tested if r['human_state']=='DIRECT_VISIBLE']
    def summary(points):
        return dict(points=len(points),before=distribution([r['before_error_px'] for r in points]),
                    after=distribution([r['after_error_px'] for r in points]),
                    improved=sum(r['after_error_px']<r['before_error_px']-1e-9 for r in points),
                    worsened=sum(r['after_error_px']>r['before_error_px']+1e-9 for r in points))
    return dict(refined_points=len(changes),refined_frames=len({r['id'] for r in changes}),
        refined_valid_reference=summary(tested),refined_human_direct_visible=summary(direct),
        all_human_direct_visible_before=distribution([r['before'] for r in all_direct]),
        all_human_direct_visible_after=distribution([r['after'] for r in all_direct]),
        refined_human_states=dict(Counter(r['human_state'] for r in changes)),
        corner_reasons=dict(reasons),edge_reasons=dict(edges),face_reasons=dict(faces),changes=changes,
        reference_phase='Fixed historical N3 native_to_reference mapping; no per-corner rematching')


def score(args):
    out=Path(args.output)
    if (out/'RESULTS.json').exists():raise FileExistsError('Preserve completed finite CAL experiment')
    seal=C.read(out/'PREDICTION_SEAL.json')
    if not seal['before_scoring'] or C.sha(out/'PREDICTIONS.jsonl.gz')!=seal['sha256']:
        raise AssertionError('Independent RGB coordinates must be sealed before score')
    labels=C.read(args.scoring)['frames']
    if len(labels)!=76 or any(r['partition']!='CAL' for r in labels):raise AssertionError('Only isolated CAL76 scoring')
    metadata={r['id']:r for r in labels};rows=list(C.rows(out/'PREDICTIONS.jsonl.gz'));groups={}
    for row in rows:
        row['metric']=pose_metric(row['actual_pose'],metadata[row['id']]['target']['truth'])
        groups.setdefault(row['method'],[]).append(row)
    assert all(len(g)==76 for g in groups.values())
    base=pose_summary(groups['PRIMARY_V1'])
    if abs(base['metrics']['translation_cm']['mean']-116.29147334738165)>1e-7 or base['success_count']!=37:
        raise AssertionError('Primary v1 pose baseline mismatch')
    sessions=sorted({r['session'] for r in rows});summaries={};coordinates={};per_session={};comparison={}
    reference={r['id']:r for r in groups['PRIMARY_V1']}
    for name,group in groups.items():
        summaries[name]=pose_summary(group);coordinates[name]=coordinate_summary(group,metadata)
        per_session[name]={s:dict(pose=pose_summary([r for r in group if r['session']==s]),
                                 coordinate=coordinate_summary([r for r in group if r['session']==s],metadata))
                           for s in sessions}
        if name!='PRIMARY_V1':comparison[name]=paired(group,reference)
    result=dict(complete=True,schema='CAL_internal_face_RGB_coordinate_results_v1',frames=76,arms=ARMS,
        summary=summaries,coordinates=coordinates,per_session=per_session,paired_against_primary=comparison,
        labels=binding(args.scoring),prediction_seal=binding(out/'PREDICTION_SEAL.json'),
        new_TEST_reads=0,FINAL_reads=0,visibility_model_changed=False,physical3D_corner_certified=False,
        no_projected_point_observations=True,integration_performed=False,
        interpretation='Finite CAL coordinate/pose development. RGB face separation may fail; supported line intersections may be virtual cuboid vertices. Fixed phase reference is reconstructed, not an independent physical sensor pose.')
    C.save_rows(out/'SCORED_ROWS.jsonl.gz',rows);C.write(out/'RESULTS.json',result)
    for name,item in summaries.items():
        coord=coordinates[name];d=coord['refined_human_direct_visible']
        print(name,'refined',coord['refined_points'],'direct',d['points'],
            'directError',d['before']['mean'],d['after']['mean'],
            'T',round(item['metrics']['translation_cm']['mean'],4),
            'R',round(item['metrics']['rotation_deg']['mean'],4),'success',item['success_count'],flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('stage',choices=('infer','score'))
    parser.add_argument('--inputs',type=Path,default=C.SCRATCH/'CAL_INPUTS.json')
    parser.add_argument('--upstream',type=Path,default=C.SCRATCH/'cal_segmentation_v2')
    parser.add_argument('--scoring',type=Path,default=C.SCRATCH/'ORACLE_CAL_INPUTS.json')
    parser.add_argument('--output',type=Path,default=C.SCRATCH/'internal_face_cal_v1')
    args=parser.parse_args();(infer if args.stage=='infer' else score)(args)


if __name__=='__main__':main()
