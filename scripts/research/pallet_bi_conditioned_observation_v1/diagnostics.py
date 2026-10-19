"""Postseal representative-BI input substitutions, never model fitting.

Synthetic state roles and projected corners are geometry references. They are
not independent physical-material annotations; projected-point recovery can be
circular because the same renderer pose defines its observations and metric.
"""
from __future__ import annotations

import argparse
from collections import Counter
from copy import deepcopy
import math
from pathlib import Path
import resource
import time

import cv2
import numpy as np
import torch

from . import bank as B
from . import models as M
from . import predict as P
from . import solver as S
from .supervision import align_reference

REPRESENTATIVE = 'OBS_BI_s20261018'
AUTOMATIC = 'AUTOMATIC_BI_REPRESENTATIVE'
STATE_REFERENCE = 'GEOMETRIC_ROLE_REFERENCE'
POINT_REFERENCE = 'PROJECTED_CORNER_REFERENCE_DIAGNOSTIC'


def sealed_gate(run_dir, expected_manifest_sha256, root_GO):
    """No target or real annotation IO occurs before this explicit seal gate."""
    if root_GO is not True:
        raise PermissionError('Explicit root GO after all automatic outputs are sealed is required')
    root = Path(run_dir); path = root/'PREDICTION_MANIFEST.json'
    if B.sha(path) != expected_manifest_sha256:
        raise ValueError('Exact current all-automatic prediction seal required')
    manifest = B.read(path)
    if (manifest.get('complete') is not True or manifest.get('rows') != 238
            or manifest.get('before_scoring') is not True or manifest.get('real_reference_inputs') is not False):
        raise ValueError('All automatic real238 outputs must precede diagnostic references')
    if manifest['artifacts']['PREDICTIONS.jsonl.gz'] != B.sha(root/'PREDICTIONS.jsonl.gz'):
        raise ValueError('Automatic predictions changed')
    plan = B.read(root/'EXPERIMENT_PLAN.json')
    if B.sha(root/'EXPERIMENT_PLAN.json') != manifest['experiment_plan_sha256']:
        raise ValueError('Frozen plan changed')
    if plan['diagnostic_plan']['representative'] != REPRESENTATIVE:
        raise ValueError('Only the frozen representative BI diagnostic is authorized')
    return manifest, plan


def geometric_roles(alignment):
    """Original runtime projected-hull rule, restricted later to automatic V."""
    if not alignment['available'] or not all(alignment['reference_valid8']):
        return None
    xy = np.asarray(alignment['aligned_reference_xy8'], float)
    hull = cv2.convexHull(xy.astype(np.float32)).reshape((-1,2))
    contour = hull.astype(np.float32).reshape((-1,1,2))
    return ['B' if abs(cv2.pointPolygonTest(contour,tuple(map(float,p)),True)) < 1e-3 else 'I' for p in xy]


def choose_selection(bank, features, model, normalizer, conditions=None):
    view = P._eligible_bank(bank)
    choices = M.choose(model, features, normalizer, condition_overrides=conditions)
    selected = {r['native_corner']:r['selected_candidate_id'] for r in choices}
    selection = S.selection_from_candidates(view,selected)
    return selection,choices


def replace_effective_points(selection, solver_result, reference_xy, reference_valid):
    """Replace effective full XY only; no partial archive, covariance or ID edit."""
    output = deepcopy(selection)
    partial_ids = {r['native_corner'] for r in selection['partial_lines']}
    if 'effective_full_points' in solver_result:
        effective = {r['native_corner'] for r in solver_result['effective_full_points']}
    else:
        points, _, _ = S._deduplicate(selection['full_points'],selection['partial_lines'])
        effective = {r['native_corner'] for r in points}
    changes = []
    for row in output['full_points']:
        k = row['native_corner']
        if k in partial_ids or k not in effective or not reference_valid[k]:
            continue
        before = deepcopy(row); row['xy'] = deepcopy(reference_xy[k])
        changes.append(dict(native_corner=k,original_xy=before['xy'],reference_xy=deepcopy(row['xy']),
                            numerically_changed=before['xy'] != row['xy']))
        other = deepcopy(row); other['xy'] = before['xy']
        if other != before:
            raise AssertionError('Point substitution changed covariance/source/metadata')
    changed = {r['native_corner']:r['reference_xy'] for r in changes}
    for row in output.get('selected_candidates',[]):
        if row['native_corner'] in changed:
            row['actual_full_xy'] = deepcopy(changed[row['native_corner']])
    if output['partial_lines'] != selection['partial_lines'] or output['states'] != selection['states']:
        raise AssertionError('Diagnostic changed typed partials or automatic backend states')
    return output,changes


def synthetic_metric(pose, gt):
    """Original trace/acos C1/C2 metric using the unmodified renderer matrix."""
    if not pose or not pose.get('available'):
        return dict(pose_available=False,translation_cm=None,rotation_deg=None,success=False)
    R = np.asarray(pose['R_physical'],float); t = np.asarray(pose['centroid'],float)
    G = np.asarray(gt['R_fixed'],float); center = np.asarray(gt['t'],float)
    if gt['symmetry_order'] not in (1,2):
        raise ValueError('Only original synthetic C1/C2 reference permitted')
    angles=[]
    for index in range(gt['symmetry_order']):
        angle=2*math.pi*index/gt['symmetry_order'];c,s=math.cos(angle),math.sin(angle)
        symmetry=np.array([[c,0,s],[0,1,0],[-s,0,c]])
        cosine=float((np.trace((G@symmetry).T@R)-1)/2)
        angles.append(math.degrees(math.acos(float(np.clip(cosine,-1.,1.)))))
    translation=float(np.linalg.norm(t-center)*100);rotation=min(angles)
    return dict(pose_available=True,translation_cm=translation,rotation_deg=rotation,
                success=bool(translation<=5 and rotation<=5))


def summarize(values):
    data=[float(v) for v in values if v is not None and math.isfinite(float(v))]
    return dict(n=len(data),mean=float(np.mean(data)) if data else None,
        sample_sd=float(np.std(data,ddof=1)) if len(data)>1 else None,max=max(data) if data else None)


def summarize_rows(rows):
    out={}
    for name in (AUTOMATIC,STATE_REFERENCE,POINT_REFERENCE):
        groups=[r['groups'][name] for r in rows]
        out[name]=dict(denominator=len(rows),successes=sum(g['metric']['success'] for g in groups),
            success_rate=sum(g['metric']['success'] for g in groups)/len(rows) if rows else None,
            no_pose=sum(not g['metric']['pose_available'] for g in groups),fallback=sum(g['fallback'] for g in groups),
            accepted=sum(g['accepted'] for g in groups),
            translation_cm=summarize([g['metric']['translation_cm'] for g in groups]),
            rotation_deg=summarize([g['metric']['rotation_deg'] for g in groups]),
            status_counts=dict(Counter(g['status'] for g in groups)))
    paired={}
    for name in (STATE_REFERENCE,POINT_REFERENCE):
        recoveries=[r['id'] for r in rows if not r['groups'][AUTOMATIC]['metric']['success'] and r['groups'][name]['metric']['success']]
        damages=[r['id'] for r in rows if r['groups'][AUTOMATIC]['metric']['success'] and not r['groups'][name]['metric']['success']]
        paired[name]=dict(recoveries=len(recoveries),damages=len(damages),recovery_ids=recoveries,damage_ids=damages,
            same_observations_as_automatic=sum(r['groups'][name]['same_observations_as_automatic'] for r in rows),
            success_rate_difference=(len(recoveries)-len(damages))/len(rows) if rows else None)
    return dict(groups=out,paired=paired)


def _load_representative(root,plan):
    folder=root/'training'/REPRESENTATIVE; path=folder/'best.pt';receipt=B.read(folder/'RUN_RECEIPT.json')
    if not receipt['complete'] or receipt['checkpoint_sha256']!=B.sha(path):
        raise ValueError('Frozen completed representative checkpoint required')
    checkpoint=torch.load(path,map_location='cpu',weights_only=False)
    norm=B.read(root/'TRAIN_ONLY_NORMALIZER.json');bound=checkpoint['bindings']['scientific']
    if (checkpoint['arm']!='BI' or checkpoint['seed']!=20261018 or checkpoint['normalizer']!=norm
            or bound['experiment_plan_sha256']!=B.sha(root/'EXPERIMENT_PLAN.json')
            or bound['model_source_sha256']!=B.sha(M.__file__)):
        raise ValueError('Representative model/normalizer/plan binding changed')
    model=M.Scorer('BI');model.load_state_dict(checkpoint['model_state_dict']);model.eval();model.requires_grad_(False)
    return model,norm,dict(checkpoint_sha256=B.sha(path),receipt_sha256=B.sha(folder/'RUN_RECEIPT.json'))


def _solve_cached(frame,selection,cache):
    key=P.exact_solve_key(frame,selection)
    packet={k:selection[k] for k in ('states','full_points','partial_lines')}
    if key in cache:
        old_packet,result=cache[key]
        if old_packet!=packet:raise AssertionError('Diagnostic cache hash collision')
        return deepcopy(result),key,True,0
    result=S.solve_frame(frame,selection)
    cache[key]=(deepcopy(packet),deepcopy(result))
    return result,key,False,result['optimizer_calls']


def run_synthetic(root,upstream_path,gt_path,bank_root,output,model,norm,plan):
    frames=B.rows(upstream_path);labels=B.rows(gt_path)
    if len(frames)!=1003 or [f['id'] for f in frames]!=[g['id'] for g in labels]:
        raise ValueError('Exact original synthetic1003 join required')
    p0=B.read(root/'P0_INPUT_CONTRACT.json')
    if B.sha(gt_path)!=p0['synthetic_targets']['sha256']:
        raise ValueError('P0 synthetic target binding changed')
    run=B.read(bank_root/'synthetic_all_all_RUN.json')
    if run['frames']!=1003 or run['reference_reads']!=0:
        raise ValueError('Complete common synthetic bank required')
    rows=[];optimizer_calls=0;selected_ids=[]
    for ordinal,(frame,gt) in enumerate(zip(frames,labels)):
        if frame['split']!='validation':continue
        bank_folder=bank_root/'frames'/'synthetic'/f'{ordinal:04d}'
        bank,feature,_,_=P._bind_frame(bank_folder,frame,run['binding'])
        before=P.digest(bank);alignment=align_reference(frame,gt);roles=geometric_roles(alignment)
        view=P._eligible_bank(bank);automatic_ids=[c['native_corner'] for c in view['corners']]
        scored_ids=[c['native_corner'] for c in view['corners'] if c['partial_line'] is None]
        overrides={k:roles[k] for k in scored_ids} if roles is not None else {}
        automatic,auto_choices=choose_selection(bank,feature,model,norm)
        # The production forward and its general model.choose helper must agree.
        production,_=P.infer_selection(bank,feature,model,norm)
        if automatic!=production:raise AssertionError('Diagnostic automatic forward differs from production')
        cache={};auto_result,auto_key,auto_cached,calls=_solve_cached(frame,automatic,cache);optimizer_calls+=calls
        state,state_choices=choose_selection(bank,feature,model,norm,overrides)
        if state['states']!=automatic['states']:
            raise AssertionError('State reference escaped the scorer condition')
        state_result,state_key,state_cached,calls=_solve_cached(frame,state,cache);optimizer_calls+=calls
        point,changes=replace_effective_points(automatic,auto_result,
            alignment['aligned_reference_xy8'],alignment['reference_valid8'])
        point_result,point_key,point_cached,calls=_solve_cached(frame,point,cache);optimizer_calls+=calls
        groups={}
        for name,selection,choices,result,key,cached in (
            (AUTOMATIC,automatic,auto_choices,auto_result,auto_key,auto_cached),
            (STATE_REFERENCE,state,state_choices,state_result,state_key,state_cached),
            (POINT_REFERENCE,point,auto_choices,point_result,point_key,point_cached)):
            groups[name]=dict(actual_pose=result['actual_pose'],accepted=result['accepted'],fallback=result['fallback'],
                status=result['status'],metric=synthetic_metric(result['actual_pose'],gt),observation_signature=key,
                exact_packet_cache_reused=cached,same_observations_as_automatic=key==auto_key,
                selected_candidate_ids=[r['selected_candidate_id'] for r in choices])
        detail=dict(id=frame['id'],split=frame['split'],ordinal=ordinal,alignment=alignment,
            automatic_visible_ids=automatic_ids,geometric_roles=roles,scorer_condition_overrides=overrides,
            changed_condition_ids=[k for k in scored_ids if roles is not None and roles[k]!=bank['states'][k]],
            partials_not_scored=[c['native_corner'] for c in view['corners'] if c['partial_line'] is not None],
            point_replacements=changes,groups=groups,
            selections={AUTOMATIC:automatic,STATE_REFERENCE:state,POINT_REFERENCE:point},
            solver_results={AUTOMATIC:auto_result,STATE_REFERENCE:state_result,POINT_REFERENCE:point_result},
            bank_frame_seal_sha256=B.sha(bank_folder/'SEAL.json'))
        if P.digest(bank)!=before:raise AssertionError('Diagnostic changed shared candidate bank')
        path=output/'frames'/'synthetic'/f'{ordinal:04d}.json.gz';B.write(path,detail)
        rows.append(dict(id=frame['id'],split='validation',groups=groups,
            changed_condition_ids=detail['changed_condition_ids'],point_replacement_count=len(changes),
            frame_result_path=str(path),frame_result_sha256=B.sha(path)))
        selected_ids.append(frame['id'])
        if len(rows)%32==0:print(dict(stage='SYNTHETIC_POSTSEAL_DIAGNOSTIC',done=len(rows),total=235),flush=True)
    if len(rows)!=235:raise ValueError('Every original validation235 frame must remain')
    P.save_rows(output/'SYNTHETIC_ROWS.jsonl.gz',rows)
    summary=dict(schema='REPRESENTATIVE_BI_SYNTHETIC_CONDITIONAL_DIAGNOSTICS_V1',rows=235,
        representative=REPRESENTATIVE,split='validation',result=summarize_rows(rows),
        state_condition_changed_frames=sum(bool(r['changed_condition_ids']) for r in rows),
        point_replacement_frames=sum(r['point_replacement_count']>0 for r in rows),
        state_reference='GEOMETRIC_ROLE_REFERENCE; cuboid projected hull, not material visibility authentication',
        point_reference='PROJECTED_CORNER_REFERENCE_DIAGNOSTIC; same GT defines observations and evaluation, circular/possibly trivial',
        physical_line_replacement='NA: no independently certified material-line labels',
        automatic_V_candidates_covariances_backend_states_fixed=True,partial_archives_fixed=True,
        actual_new_pose_optimizer_calls=optimizer_calls,ordered_ids=selected_ids,
        source_bindings=dict(upstream_sha256=B.sha(upstream_path),synthetic_gt_sha256=B.sha(gt_path),
            common_synthetic_run_sha256=B.sha(bank_root/'synthetic_all_all_RUN.json')))
    B.write(output/'SYNTHETIC_SUMMARY.json',summary)
    return summary


def physical_annotation_inventory(source_root):
    """Postseal inspect original label provenance, never synthesize labels."""
    root=Path(source_root);base=root/'_docs/experiments/pallet_visibility_curriculum_20261013_v1/nonsevere_v1'
    sources={'CAL':base/'OOF_pose_v1/ISOLATED_CAL_SCORING.json',
             'TEST':base/'test_reused_v1/ISOLATED_TEST_SCORING.json'}
    inventory=[]
    for partition,path in sources.items():
        labels=B.read(path)
        rows=labels['rows'];metadata=[r.get('metadata',{}) for r in rows.values()]
        examples=[dict(id=fid,metadata=row.get('metadata',{}),target_fields=sorted(row.get('target',{})))
                  for fid,row in list(rows.items())[:2]]
        inventory.append(dict(partition=partition,path=str(path),sha256=B.sha(path),frames=len(rows),
            container_fields=sorted(labels),metadata_fields=sorted(set().union(*(m.keys() for m in metadata))),
            human_state_counts=dict(Counter(s for m in metadata for s in m.get('human_states_native',[]))),
            examples=examples,independent_BI_labels=False,independent_line_labels=False))
    return dict(status='POSTSEAL_ORIGINAL_PROVENANCE_INSPECTION_REQUIRED',sources=inventory,
        point_diagnostic_status='PENDING_AUTHENTICATION: existing coordinate targets are not assumed physical clicks',
        B_I_state_diagnostic='NA unless original independent B/I annotations are authenticated',
        line_diagnostic='NA unless original physical-line correspondence annotations are authenticated')


def run_physical_points(root,upstream_path,reference_path,source_root,output):
    """Authenticate original manual_click DIRECT points and replace only those."""
    from scripts.research.pallet_statewise_end_to_end_recovery_v1.evaluator import _safe_metric
    source_root=Path(source_root);frames=B.rows(upstream_path)
    labels={};label_bindings={}
    base=source_root/'_docs/experiments/pallet_visibility_curriculum_20261013_v1/nonsevere_v1'
    for partition,relative in [('CAL','OOF_pose_v1/ISOLATED_CAL_SCORING.json'),
                               ('TEST','test_reused_v1/ISOLATED_TEST_SCORING.json')]:
        path=base/relative;labels[partition]=B.read(path)['rows'];label_bindings[str(path)]=B.sha(path)
    membership=source_root/'challenge/real_gt_v2/manifests/PAPER_EVAL_ALL_POS.json'
    members={r['frame_id']:r for r in B.read(membership)['items']}
    ancestry=source_root/'_docs/experiments/pallet_posefix_replay_diagnosis_v1/TARGET_SOURCE_BINDINGS.json'
    original_hashes={r['path']:r['sha256'] for r in B.read(ancestry)['files']}
    old_predictions=B.read(base/'PREDICTION_INPUTS.json')
    old_frames={r['id']:r for r in old_predictions['frames']}
    if B.sha(reference_path)!=B.read(root/'P0_INPUT_CONTRACT.json')['real_reference']['sha256']:
        raise ValueError('Canonical real metric reference changed')
    refs=B.read(reference_path)['references']
    rows=[];audit=[];source_counts=Counter();reason_counts=Counter();optimizer_calls=0
    original_files={};physical_points=0
    for ordinal,frame in enumerate(frames):
        if frame['partition'] not in ('CAL','TEST'):continue
        fid=frame['id'];label=labels[frame['partition']][fid]
        metadata=label['metadata'];target=label['target'];member=members[fid]
        old=old_frames[fid]
        if (old['points']['N3_SUBPIX']!=frame['initial_points'] or
                old['initial_poses']['N3_SUBPIX']!=frame['initial_pose']):
            raise ValueError('Old human/native annotation phase is not the exact current N0')
        relative=member['gt_v2_path'];annotation=source_root/relative
        if original_hashes.get(relative)!=B.sha(annotation):
            raise ValueError('Original keypoint annotation ancestry changed: '+relative)
        payload=B.read(annotation)
        if payload['schema_version']!='real_pallet_gt_v2' or len(payload['objects'])!=1:
            raise ValueError('Explicit original real keypoint contract required')
        obj=payload['objects'][0];entries=obj['keypoint_annotations']
        mapping=np.asarray(metadata['native_to_reference_corner_ids'],int)
        if mapping.shape!=(8,) or set(mapping.tolist())!=set(range(8)):
            raise ValueError('Fixed whole-object original native/reference bijection required')
        original_files[relative]=B.sha(annotation)
        source_counts.update(e.get('source','MISSING') for e in entries[:8])
        detail_path=root/'prediction_frames'/'real'/f'{ordinal:04d}'/'FRAME_PREDICTIONS.json.gz'
        saved=B.read(detail_path);details=saved['current_observation_and_solver_records'][REPRESENTATIVE]
        automatic=details['selection'];auto_result=details['solver_result']
        auto_pose=saved['ledger']['predictions'][REPRESENTATIVE]['actual_pose']
        if auto_result['actual_pose']!=auto_pose:
            raise AssertionError('Saved automatic counterpart and solver pose differ')
        partial_ids={r['native_corner'] for r in automatic['partial_lines']}
        effective={r['native_corner'] for r in auto_result.get('effective_full_points',[])}
        valid=[False]*8;xy=[None]*8;point_audit=[]
        for k,j in enumerate(mapping):
            entry=entries[int(j)];reason='NOT_EFFECTIVE_FULL_AUTOMATIC_OBSERVATION'
            if k in partial_ids:reason='FIXED_PARTIAL_PATH'
            elif k in effective:
                if metadata['human_states_native'][k]!='DIRECT_VISIBLE':reason='NOT_HUMAN_DIRECT_VISIBLE'
                elif entry.get('source')!='manual_click':reason='NOT_AUTHENTICATED_MANUAL_CLICK'
                elif entry.get('visibility')!=2:reason='NOT_DIRECT_VISIBLE_ORIGINAL_ANNOTATION'
                elif not target['valid'][int(j)]:reason='ORIGINAL_TARGET_LOCATION_INVALID'
                else:
                    point=np.asarray(entry.get('xy'),float)
                    if point.shape!=(2,) or not np.isfinite(point).all():reason='MISSING_ORIGINAL_CLICK_COORDINATE'
                    elif not np.array_equal(point,np.asarray(target['gt'][int(j)],float)):
                        raise ValueError('Saved isolated coordinate differs from original manual click')
                    elif not (0<=point[0]<frame['raw_hw'][1] and 0<=point[1]<frame['raw_hw'][0]):
                        reason='DIRECT_CLICK_OUTSIDE_ORIGINAL_RGB'
                    else:xy[k]=point.tolist();valid[k]=True;reason='AUTHENTICATED_ORIGINAL_DIRECT_MANUAL_CLICK'
            reason_counts[reason]+=1
            point_audit.append(dict(native_corner=k,reference_corner=int(j),original_source=entry.get('source'),
                human_state=metadata['human_states_native'][k],eligible=valid[k],reason=reason))
        audit.append(dict(id=fid,partition=frame['partition'],original_annotation=relative,
            original_annotation_sha256=B.sha(annotation),point_audit=point_audit,
            eligible_direct_manual_point_count=sum(valid)))
        if not any(valid):continue
        replacement,changes=replace_effective_points(automatic,auto_result,xy,valid)
        cache={};signature=P.exact_solve_key(frame,automatic)
        cache[signature]=(deepcopy({k:automatic[k] for k in ('states','full_points','partial_lines')}),deepcopy(auto_result))
        result,key,reused,calls=_solve_cached(frame,replacement,cache);optimizer_calls+=calls
        reference=refs[frame['partition']][fid]
        auto_metric=_safe_metric(auto_pose,reference);changed_metric=_safe_metric(result['actual_pose'],reference)
        record=dict(id=fid,partition=frame['partition'],original_annotation=relative,
            direct_point_replacements=changes,automatic_actual_pose=auto_pose,
            physical_point_actual_pose=result['actual_pose'],automatic_metric=auto_metric,
            physical_point_metric=changed_metric,automatic_status=auto_result['status'],
            physical_point_status=result['status'],automatic_fallback=auto_result['fallback'],
            physical_point_fallback=result['fallback'],exact_packet_cache_reused=reused,
            automatic_observation_signature=signature,physical_point_observation_signature=key)
        path=output/'frames'/'real'/f'{ordinal:04d}.json.gz'
        B.write(path,dict(row=record,automatic_selection=automatic,reference_point_selection=replacement,
                          automatic_solver_result=auto_result,physical_point_solver_result=result))
        rows.append(record);physical_points+=len(changes)
    if len(audit)!=95:raise ValueError('Original CAL46 and TEST49 annotation audit must be complete')
    P.save_rows(output/'PHYSICAL_POINT_ROWS.jsonl.gz',rows)
    P.save_rows(output/'PHYSICAL_POINT_SOURCE_AUDIT.jsonl.gz',audit)
    summaries={}
    for partition in ('CAL','TEST','CAL_TEST_AVAILABLE_SUBSET'):
        rr=[r for r in rows if partition=='CAL_TEST_AVAILABLE_SUBSET' or r['partition']==partition]
        summaries[partition]=dict(denominator=len(rr),
            automatic_successes=sum(r['automatic_metric']['success'] for r in rr),
            physical_point_successes=sum(r['physical_point_metric']['success'] for r in rr),
            recoveries=sum(not r['automatic_metric']['success'] and r['physical_point_metric']['success'] for r in rr),
            damages=sum(r['automatic_metric']['success'] and not r['physical_point_metric']['success'] for r in rr),
            automatic_translation_cm=summarize([r['automatic_metric']['translation_cm'] for r in rr]),
            physical_point_translation_cm=summarize([r['physical_point_metric']['translation_cm'] for r in rr]),
            automatic_rotation_deg=summarize([r['automatic_metric']['rotation_deg'] for r in rr]),
            physical_point_rotation_deg=summarize([r['physical_point_metric']['rotation_deg'] for r in rr]))
    report=dict(status='COMPLETE' if rows else 'NA_NO_AUTHENTICATED_EFFECTIVE_DIRECT_MANUAL_POINTS',
        real_independent_BI_state_reference='NA: original sources distinguish DIRECT/SELF/EXTERNAL, not B/I',
        real_physical_line_reference='NA: original keypoint annotations supply no material-line ownership reference',
        physical_point_reference_source='original real_pallet_gt_v2 keypoint_annotations with source=manual_click,visibility=2; same saved isolated target.gt and human DIRECT_VISIBLE',
        pose_metric_limitation='Canonical annotation-derived C2 pose reference; physical 2D clicks do not make the pose reference independent physical metrology',
        audited_frames=95,eligible_subset_frames=len(rows),eligible_replaced_points=physical_points,
        eligible_subset_ids=[r['id'] for r in rows],original_keypoint_source_counts=dict(source_counts),
        eligibility_reason_counts=dict(reason_counts),results=summaries,
        metric_subset_is_full_PRIMARY192=False,automatic_V_states_source_ids_covariances_partial_archives_fixed=True,
        actual_new_pose_optimizer_calls=optimizer_calls,original_annotation_sha256=original_files,
        bindings=dict(label_sources=label_bindings,membership_sha256=B.sha(membership),
            original_annotation_ancestry_sha256=B.sha(ancestry),reference_sha256=B.sha(reference_path),
            N0_native_phase_input_sha256=B.sha(base/'PREDICTION_INPUTS.json')),
        evidence_code=dict(annotation_writer='scripts/annotate/annotate_io.py:_normalise_keypoint_annotations',
            annotation_loader='challenge/evaluation_v2/paper_real_eval.py:_legacy_forbidden_target',
            target_provision='scripts/research/pallet_posefix_replay_diagnosis_v1/run.py:prepare'))
    B.write(output/'PHYSICAL_POINT_SUMMARY.json',report)
    return report


def self_test():
    alignment=dict(available=True,reference_valid8=[True]*8,
        aligned_reference_xy8=[[0,0],[2,0],[2,2],[0,2],[.5,.5],[1.5,.5],[1.5,1.5],[.5,1.5]])
    assert geometric_roles(alignment)==['B']*4+['I']*4
    selection=dict(states=['B','B']+['U']*6,
        full_points=[dict(native_corner=0,xy=[1.,2.],covariance_px2=[[2.,0.],[0.,3.]],source='a'),
                     dict(native_corner=1,xy=[3.,4.],covariance_px2=[[4.,0.],[0.,5.]],source='b')],
        partial_lines=[dict(native_corner=1,lines=[dict(normal=[1.,0.])])])
    before=deepcopy(selection)
    changed,records=replace_effective_points(selection,dict(effective_full_points=selection['full_points']),
        [[7.,8.],[9.,10.]]+[None]*6,[True,True]+[False]*6)
    assert changed['full_points'][0]['xy']==[7.,8.] and changed['full_points'][1]==selection['full_points'][1]
    assert changed['partial_lines']==selection['partial_lines'] and selection==before and len(records)==1
    return dict(status='PASS',checks=['original_projected_hull_roles','effective_full_XY_only','partial_archive_unchanged','input_immutable'],
                actual_pose_optimizer_calls=0,reference_file_reads=0)


def audit_legacy_ancestry(source_root,diagnostic_output):
    """Follow all migrated CAL/TEST labels to original per-point provenance."""
    root=Path(source_root);output=Path(diagnostic_output)
    proof=output/'PHYSICAL_LEGACY_ANCESTRY_AUDIT.json'
    if proof.exists():raise FileExistsError(proof)
    completed=B.read(output/'DIAGNOSTIC_SEAL.json')
    rows=B.rows(output/'PHYSICAL_POINT_SOURCE_AUDIT.jsonl.gz')
    if len(rows)!=95:raise ValueError('Complete original CAL46/TEST49 source audit required')
    counts=Counter();review_reasons=Counter();object_keys=set();details=[]
    for row in rows:
        current=root/row['original_annotation']
        if B.sha(current)!=row['original_annotation_sha256']:
            raise ValueError('Audited migrated annotation changed')
        migrated=B.read(current);obj=migrated['objects'][0]
        migration=migrated['real_gt_v2_migration'];path=root/migration['source_label']
        if B.sha(path)!=migration['source_sha256']:
            raise ValueError('Original legacy annotation hash differs from migration binding')
        legacy=B.read(path)['objects'][0];manual=legacy.get('manual_kps');mask=legacy.get('extrapolated_mask')
        authenticated=(isinstance(manual,(list,tuple)) and len(manual)==9 and
            isinstance(mask,(list,tuple)) and len(mask)==9 and all(type(v)is bool for v in mask))
        counts['frames']+=1;counts['original_source_hash_exact']+=1
        counts['has_legacy_manual_kps9']+=isinstance(manual,list) and len(manual)==9
        counts['has_boolean_extrapolated_mask9']+=isinstance(mask,list) and len(mask)==9 and all(type(v)is bool for v in mask)
        counts['authenticatable_per_point_source_frames']+=authenticated
        counts['object_gt_source_manual']+=legacy.get('gt_source')=='manual'
        counts.update('keypoint_source_'+str(e.get('source','missing')) for e in obj['keypoint_annotations'][:8])
        object_keys.update(obj);review_reasons.update(obj.get('manual_review_reasons',[]))
        details.append(dict(id=row['id'],partition=row['partition'],original_source_label=migration['source_label'],
            original_source_sha256=B.sha(path),original_source_hash_exact=True,
            has_manual_kps9=isinstance(manual,list) and len(manual)==9,
            has_boolean_extrapolated_mask9=isinstance(mask,list) and len(mask)==9 and all(type(v)is bool for v in mask),
            authenticatable_per_point_provenance=authenticated,migration_status=obj.get('migration_status')))
    code=[root/'scripts/annotate/migrate_real_gt_v2.py',root/'scripts/annotate/annotate_io.py',
          root/'challenge/evaluation_v2/paper_real_eval.py',root/'scripts/research/pallet_posefix_replay_diagnosis_v1/run.py']
    has_sources=counts['authenticatable_per_point_source_frames']>0
    conclusion=('Original legacy files contain per-point manual/extrapolation provenance; the exact eligible subset needs explicit transport before a physical-point NA conclusion.'
        if has_sources else
        'Existing manual_kps/object gt_source=manual do not distinguish original clicks from extrapolated/PnP fallback coordinates. All audited per-corner sources remain unknown; no authenticated physical-point oracle subset exists.')
    result=dict(schema='ORIGINAL_CAL_TEST_PHYSICAL_PROVENANCE_ANCESTRY_AUDIT_V1',status='NEEDS_LEGACY_POINT_TRANSPORT' if has_sources else 'COMPLETE',
        counts=dict(counts),migration_manual_review_reason_counts=dict(review_reasons),
        all_original_object_fields=sorted(object_keys),rows=details,
        conclusion=conclusion,
        integer_coordinate_heuristic_used=False,small_prediction_error_as_provenance_used=False,
        human_DIRECT_as_coordinate_source_certificate_used=False,real_state_BI_reference_available=False,
        real_material_line_reference_available=False,new_pose_optimizer_calls=0,new_model_fits=0,
        original_diagnostic_seal_sha256=B.sha(output/'DIAGNOSTIC_SEAL.json'),
        numerical_diagnostic_source_sha256=completed['source_bindings']['diagnostics.py'],
        metadata_extension_source_sha256=B.sha(__file__),
        evidence_source_sha256={str(p.relative_to(root)):B.sha(p) for p in code})
    B.write(proof,result)
    return result


def publish_diagnostics(run_dir,public):
    """Project sealed scalar diagnostic evidence and a standalone figure only."""
    import ast
    import csv
    import gzip
    import io
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from .artifacts import clean
    root=Path(run_dir);public=Path(public);private=root/'diagnostics_v1'
    destinations=[public/'DIAGNOSTIC_RESULTS.md',public/'DIAGNOSTIC_SUMMARY_PUBLIC.json',
                  public/'DIAGNOSTIC_ROWS.csv.gz',public/'figures/reference_replacement_diagnostics.png']
    if any(p.exists() for p in destinations):raise FileExistsError('Preserve existing public diagnostic artifacts')
    public.mkdir(parents=True,exist_ok=True);(public/'figures').mkdir(exist_ok=True)
    summary=B.read(private/'SYNTHETIC_SUMMARY.json');physical=B.read(private/'PHYSICAL_POINT_SUMMARY.json')
    ancestry=B.read(private/'PHYSICAL_LEGACY_ANCESTRY_AUDIT.json');seal=B.read(private/'DIAGNOSTIC_SEAL.json')
    values=B.rows(private/'SYNTHETIC_ROWS.jsonl.gz')
    for name,digest in seal['artifacts'].items():
        if B.sha(private/name)!=digest:raise ValueError('Sealed diagnostic artifact changed: '+name)
    if len(values)!=235:raise ValueError('All original validation235 diagnostic rows required')
    snapshot=root/'source_attempts/diagnostics_numerical_v1.py'
    if B.sha(snapshot)!=seal['source_bindings']['diagnostics.py']:
        raise ValueError('Numerical diagnostic source snapshot differs from execution seal')
    def functions(path):
        tree=ast.parse(path.read_text())
        return {node.name:ast.dump(node,include_attributes=False) for node in tree.body if isinstance(node,ast.FunctionDef)}
    before,after=functions(snapshot),functions(Path(__file__))
    changed=[name for name in before if before[name]!=after.get(name)]
    if changed:raise AssertionError('Numerical functions changed during metadata extension: '+str(changed))
    source_proof=dict(schema='DIAGNOSTIC_METADATA_AND_PUBLICATION_SOURCE_EXTENSION_V1',
        original_numerical_source_sha256=B.sha(snapshot),current_source_sha256=B.sha(__file__),
        original_functions_checked=sorted(before),changed_original_functions=changed,
        original_function_AST_equal=True,added_functions=sorted(set(after)-set(before)),
        added_scope='legacy provenance metadata audit and projection of sealed scalar evidence/figure only',
        new_model_fits=0,new_pose_solves_for_extension=0,numerical_diagnostic_seal_sha256=B.sha(private/'DIAGNOSTIC_SEAL.json'))
    proof_path=private/'DIAGNOSTIC_SOURCE_METADATA_EXTENSION.json'
    if proof_path.exists():raise FileExistsError(proof_path)
    B.write(proof_path,source_proof)
    flat=[]
    for frame in values:
        for name in (AUTOMATIC,STATE_REFERENCE,POINT_REFERENCE):
            group=frame['groups'][name]
            flat.append(dict(id=frame['id'],split='validation',reference_kind=name,
                **group['metric'],accepted=group['accepted'],fallback=group['fallback'],status=group['status'],
                exact_packet_cache_reused=group['exact_packet_cache_reused'],
                same_observations_as_automatic=group['same_observations_as_automatic'],
                changed_scorer_condition_corners_in_frame=len(frame['changed_condition_ids']),
                point_replacements_in_frame=frame['point_replacement_count']))
    with destinations[2].open('xb') as raw:
        with gzip.GzipFile(filename='',mode='wb',fileobj=raw,mtime=0) as compressed:
            with io.TextIOWrapper(compressed,newline='') as text:
                writer=csv.DictWriter(text,fieldnames=list(flat[0]));writer.writeheader();writer.writerows(flat)
    names=[AUTOMATIC,STATE_REFERENCE,POINT_REFERENCE]
    titles=['Automatic BI','Geometry-role condition','Projected-corner reference*']
    colors=['#334155','#9b59b6','#d97706'];stats=summary['result']['groups']
    fig,axes=plt.subplots(2,2,figsize=(12,8.8),constrained_layout=True)
    fig.suptitle('Conditional reference substitutions · representative BI · validation N=235',fontsize=14,weight='bold')
    ax=axes[0,0];bars=ax.bar(titles,[stats[n]['successes'] for n in names],color=colors,width=.62)
    bars[1].set_hatch('//');bars[2].set_hatch('..')
    ax.set_ylim(0,235);ax.set_ylabel('Joint pose successes / 235');ax.set_title('Diagnostic groups; separate from real automatic comparison')
    for bar,name in zip(bars,names):
        ax.text(bar.get_x()+bar.get_width()/2,bar.get_height()+5,
            f"{stats[name]['successes']}/235\n{100*stats[name]['success_rate']:.1f}%",ha='center',fontsize=10)
    ax.tick_params(axis='x',labelsize=8)
    ax=axes[0,1];maximum=[stats[n]['translation_cm']['max'] for n in names]
    bars=ax.bar(titles,maximum,color=colors,width=.62);bars[1].set_hatch('//');bars[2].set_hatch('..')
    ax.set_ylim(0,max(maximum)*1.25);ax.set_ylabel('Maximum translation error (cm)');ax.set_title('All finite extremes retained; no trimming')
    for bar,name in zip(bars,names):
        stat=stats[name]['translation_cm']
        ax.text(bar.get_x()+bar.get_width()/2,bar.get_height()+15,
            f"max {stat['max']:.2f}\nmean {stat['mean']:.2f}",ha='center',fontsize=9)
    ax.tick_params(axis='x',labelsize=8)
    for ax,axis,label in [(axes[1,0],'translation_cm','Translation error (cm), symmetric-log axis'),
                         (axes[1,1],'rotation_deg','Rotation error (degrees), original C1 convention')]:
        for name,title,color in zip(names,titles,colors):
            points=sorted(row['groups'][name]['metric'][axis] for row in values
                          if row['groups'][name]['metric'][axis] is not None)
            ax.step(points,np.arange(1,len(points)+1)/235,where='post',label=title,color=color,lw=1.8)
        ax.set_ylim(0,1.02);ax.set_xlabel(label);ax.set_ylabel('Cumulative fraction of all 235 frames');ax.grid(alpha=.2)
        if axis=='translation_cm':
            ax.set_xscale('symlog',linthresh=.01);ax.set_xlim(0,max(maximum)*1.08)
        else:ax.set_xlim(0,180)
        ax.legend(fontsize=8,loc='lower right')
    fig.text(.5,-.025,
        '* Same renderer reference defines substituted points and pose metric: circular, not physical-edge evidence.\n'
        'Validation uses the original C1-only held-out asset; no extra C2/90° symmetry. No-pose=0 in every group.',
        ha='center',fontsize=9)
    fig.savefig(destinations[3],dpi=170,bbox_inches='tight',facecolor='white');plt.close(fig)
    def process_cost(path):
        result={}
        for line in path.read_text().splitlines():
            for key in ('User time (seconds)','System time (seconds)','Elapsed (wall clock) time (h:mm:ss or m:ss)',
                        'Maximum resident set size (kbytes)','Exit status'):
                if line.strip().startswith(key+':'):result[key]=line.strip()[len(key)+1:].strip()
        return result
    cost=dict(numerical_phase_after_imports=seal['cost'],
        numerical_whole_process=process_cost(root/'execution/diagnostics_v1.process_cost.txt'),
        legacy_ancestry_whole_process=process_cost(root/'execution/diagnostics_physical_ancestry.process_cost.txt'),
        time_accounting='Numerical phase and its containing whole process overlap; do not add them. Legacy ancestry is a separate metadata-only process.',
        actual_new_pose_optimizer_calls=seal['actual_pose_optimizer_calls'],new_fits=0,new_backbone_forwards=0)
    public_summary=dict(schema='PUBLIC_REPRESENTATIVE_BI_CONDITIONAL_DIAGNOSTIC_SUMMARY_V1',
        representative=REPRESENTATIVE,synthetic_validation_N=235,synthetic_validation_asset='woodpallet_block_jtoastie_ccby.glb',
        synthetic_validation_symmetry_order=1,synthetic_summary={k:v for k,v in summary.items() if k!='ordered_ids'},
        physical_reference=dict(status=physical['status'],audited_frames=95,eligible_frames=0,eligible_points=0,
            original_keypoint_source_counts=physical['original_keypoint_source_counts'],
            eligibility_reason_counts=physical['eligibility_reason_counts'],legacy_ancestry_counts=ancestry['counts'],
            migration_manual_review_reason_counts=ancestry['migration_manual_review_reason_counts'],
            conclusion=ancestry['conclusion'],integer_coordinate_heuristic_used=False,
            independent_BI_state_reference_available=False,physical_material_line_reference_available=False,
            source_bindings=physical['bindings'],evidence_source_sha256=ancestry['evidence_source_sha256']),
        source_extension_proof=source_proof,cost=cost,
        row_ledger=dict(rows=705,frames=235,groups=3,GT_coordinates_or_matrices_exported=False,
                        file='DIAGNOSTIC_ROWS.csv.gz',sha256=B.sha(destinations[2])),
        artifact_sha256=dict(figure=B.sha(destinations[3]),numerical_diagnostic_seal=B.sha(private/'DIAGNOSTIC_SEAL.json'),
            synthetic_summary=B.sha(private/'SYNTHETIC_SUMMARY.json'),physical_summary=B.sha(private/'PHYSICAL_POINT_SUMMARY.json'),
            legacy_ancestry_audit=B.sha(private/'PHYSICAL_LEGACY_ANCESTRY_AUDIT.json')),
        limitations=['Conditional input replacement, not a retrained perfect-state upper bound',
            'Synthetic virtual projections are not physical material-edge annotations',
            'Validation was used for early stopping and is not an independent model test',
            'Automatic V, candidate arrays, covariance/source identities and backend states remain fixed',
            'Point replacement preserves partial archives and can be rejected by the fixed image-canvas guard'])
    B.write(destinations[1],clean(public_summary))
    def n(value):return f'{value:.3f}'
    table=['| 입력 | 성공/N | 위치 cm 평균±표본SD | 최대 cm | 회전 ° 평균±표본SD | 수락 / fallback |',
           '|---|---:|---:|---:|---:|---:|']
    labels=['자동 대표 BI','기하 역할 조건 참조','가상 투영 코너 좌표 참조']
    for name,title in zip(names,labels):
        s=stats[name];t=s['translation_cm'];r=s['rotation_deg']
        table.append(f"| {title} | {s['successes']}/235 | {n(t['mean'])} ± {n(t['sample_sd'])} | {n(t['max'])} | {n(r['mean'])} ± {n(r['sample_sd'])} | {s['accepted']} / {s['fallback']} |")
    text=['# 대표 BI의 조건부 참조 입력 진단','',
        '합성 validation235에서 상태 조건만 바꾸면 성공은44→44였고, 가상 투영 코너 좌표를 넣으면44→69였다. 후자는25회복·0손상이지만 위치 평균과 최대 오차가 악화했다. 자동 COMMON–BI 실사 성능표와 분리한 추가 학습 없는 입력 교체 진단이다. 대표 seed는20261018로 고정했다.','',
        '\n'.join(table),'',
        '모든 행은 같은235장이며 no-pose는0이다. 위치≤5cm AND 원 회전≤5°가 공동 성공이고 SD는ddof=1이다. 유한 극단값을 제외하거나 잘라내지 않았다. validation은 early stopping에 사용한 개발자료이며 독립 시험이 아니다. 전체 합성 계약의 원 asset별 C1/C2를 유지했고, 이번235장은 C1인 woodpallet_block_jtoastie_ccby.glb만 포함한다. 이 자산에 C2나90° 대칭을 추가하지 않았다. 원 renderer R을 정규직교화하지 않았다.','',
        '상태 진단은 정답 투영 직육면체의 원 projected-hull 규칙으로 만든 GEOMETRIC_ROLE_REFERENCE를 scorer의 두 조건값에만 넣었다. 가시 집합V·영상/후보·기하·후보별 covariance·source ID·부분선·후단 자동 상태는 고정했다. 실제 재료 코너의 가시성을 인증한 참조가 아니다.78장에서 적용 가능한 scorer 태그가 달라졌지만 실제 선택 관측은3장에서만 바뀌었고232장은 정확히 같은 관측 packet이었다. 회복0·손상0이다.','',
        '관측 진단은 자동 대표 BI가 선택한 유효 full point의 XY만 같은 whole-object 대응의 가상 참조로 바꿨다.171장은 좌표 교체 대상이 있었고64장은 동일 packet이었다. covariance·source ID·미관측·부분선 원장은 보존했다.25회복·0손상으로 성공은+10.638%p였지만 위치 평균은25.289→30.322cm, 최대는511.834→1105.873cm다.21장은 참조 코너가 영상 밖에 있어 고정 canvas 검사가 거부했으며 분모와 fallback에 남겼다. 부분선을 완전점으로 바꾸지 않았다. 같은 renderer GT에서 관측과 평가 자세를 만들기 때문에 회복은 순환적이거나 자명할 수 있으며 물리 정확도 증거가 아니다.','',
        '실사 물리 관측 참조는 CAL46+TEST49의 원95장까지 실제 역추적했다.760개 migrated corner의 source는 모두unknown이었다.95개 원본 migration source hash는 모두 정확히 일치했고,95개 모두 manual_kps와 객체 단위gt_source=manual은 있지만9개 bool의 per-point extrapolated_mask는0개다. 원 migration 코드는 이 구분 정보가 있어야 manual_click과 extrapolated/PnP fallback을 나눈다. 사람의 DIRECT_VISIBLE 판정이나 정수 좌표라는 추정으로 unknown 좌표를 직접 클릭 정답으로 승격하지 않았다. 따라서 인증된 물리점 교체 subset은0장/0점으로NA다. 독립적인 B/I 주석과 물리 선 소속 주석도 없어 두 실사 참조 진단은NA다. 기존 좌표가 틀렸다는 판정은 아니다.','',
        f"실제 진단은 새 pose optimizer 호출{seal['actual_pose_optimizer_calls']}회, 추가 fit/백본 forward0회였다. 숫자 계산 process wall32.37초(user32.03초/system0.36초)이며 함수 내부31.154초는 그 안에 포함되므로 합산하지 않는다. 원본 ancestry 검사는 별도 metadata process다. 모든235×3의 작은 스칼라 원수치와 상태는 [DIAGNOSTIC_ROWS.csv.gz](DIAGNOSTIC_ROWS.csv.gz), 출처·검산·비용은 [DIAGNOSTIC_SUMMARY_PUBLIC.json](DIAGNOSTIC_SUMMARY_PUBLIC.json)에 있다.",'',
        f"숫자를 생성한 diagnostics source SHA는 `{source_proof['original_numerical_source_sha256']}`이며 원본 snapshot을 보존했다. 그 뒤 추가한 원본 provenance 감사와 게시 코드에 대해 기존 모든 함수의 AST가 정확히 같음을 검사했다. 새 학습이나 자세 재계산 없이 source 확장 사실과 두 SHA를 공개 JSON에 구분했다.",'',
        '![자동/기하 역할/투영 코너 참조를 분리한 진단](figures/reference_replacement_diagnostics.png)','',
        '그림의 모든 패널은 같은235장이다. 위치 누적분포는0과 전체 극단값을 포함하도록 symmetric-log 축을 쓰며, 원수치는 CSV에 보존한다. 이 진단은 후보/관측/불확실성·고정 가시 집합·초기화·솔버의 병목을 함께 드러내며, 입력 교체의 개선을 자동 B/I 조건의 인과 기여나 달성 가능한 물리 상한으로 해석하지 않는다.','']
    destinations[0].write_text('\n'.join(text))
    return dict(status='PUBLIC_DIAGNOSTICS_COMPLETE',rows=705,files=[str(p) for p in destinations],
        source_extension_proof_sha256=B.sha(proof_path))


def run(args):
    started,cpu=time.monotonic(),time.process_time()
    root=Path(args.run_dir);output=Path(args.output)
    manifest,plan=sealed_gate(root,args.prediction_manifest_sha256,args.root_GO)
    if output.exists():raise FileExistsError('Every diagnostic attempt is preserved')
    output.mkdir(parents=True)
    torch.set_num_threads(1);cv2.setNumThreads(1)
    if torch.get_num_interop_threads()!=1:torch.set_num_interop_threads(1)
    model,norm,checkpoint=_load_representative(root,plan)
    summary=run_synthetic(root,Path(args.upstream_synthetic),Path(args.synthetic_gt),Path(args.bank_root),output,model,norm,plan)
    physical=physical_annotation_inventory(args.source_root) if args.source_root else dict(status='NOT_REQUESTED')
    B.write(output/'PHYSICAL_ANNOTATION_PROVENANCE.json',physical)
    physical_results=None
    if args.source_root and args.upstream_real and args.real_reference:
        physical_results=run_physical_points(root,Path(args.upstream_real),Path(args.real_reference),
            Path(args.source_root),output)
        physical['point_diagnostic_status']=physical_results['status']
        physical['status']='ORIGINAL_SOURCE_PROVENANCE_AUDITED'
        physical['physical_point_summary_sha256']=B.sha(output/'PHYSICAL_POINT_SUMMARY.json')
        B.write(output/'PHYSICAL_ANNOTATION_PROVENANCE.json',physical)
    receipt=dict(schema='BI_POSTSEAL_CONDITIONAL_DIAGNOSTIC_SEAL_V1',status='SYNTHETIC_COMPLETE_PHYSICAL_PROVENANCE_INVENTORIED',
        prediction_manifest_sha256=args.prediction_manifest_sha256,experiment_plan_sha256=B.sha(root/'EXPERIMENT_PLAN.json'),
        checkpoint=checkpoint,representative=REPRESENTATIVE,new_fits=0,new_backbone_forwards=0,
        real_numeric_annotation_read=bool(args.source_root),automatic_output_unchanged=B.sha(root/'PREDICTION_MANIFEST.json')==args.prediction_manifest_sha256,
        actual_pose_optimizer_calls=summary['actual_new_pose_optimizer_calls']+
            (physical_results['actual_new_pose_optimizer_calls'] if physical_results else 0),
        source_bindings={Path(p).name:B.sha(p) for p in (__file__,M.__file__,S.__file__,P.__file__)},
        command_arguments=vars(args),
        artifacts={p.name:B.sha(p) for p in output.iterdir() if p.is_file()},
        cost=dict(wall_seconds=time.monotonic()-started,CPU_seconds=time.process_time()-cpu,
                  peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss))
    B.write(output/'DIAGNOSTIC_SEAL.json',receipt)
    return receipt


if __name__=='__main__':
    import sys
    if sys.argv[1:]==['--self-test']:
        print(self_test());raise SystemExit(0)
    if sys.argv[1:2]==['--audit-legacy-ancestry']:
        audit_parser=argparse.ArgumentParser()
        audit_parser.add_argument('--audit-legacy-ancestry',action='store_true')
        audit_parser.add_argument('--source-root',required=True)
        audit_parser.add_argument('--diagnostic-output',required=True)
        audit_args=audit_parser.parse_args()
        audit=audit_legacy_ancestry(audit_args.source_root,audit_args.diagnostic_output)
        print(dict(status=audit['status'],counts=audit['counts']))
        raise SystemExit(0)
    if sys.argv[1:2]==['--publish']:
        publish_parser=argparse.ArgumentParser()
        publish_parser.add_argument('--publish',action='store_true')
        publish_parser.add_argument('--run-dir',required=True)
        publish_parser.add_argument('--public',required=True)
        publish_args=publish_parser.parse_args()
        print(publish_diagnostics(publish_args.run_dir,publish_args.public))
        raise SystemExit(0)
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('run-dir','upstream-synthetic','synthetic-gt','bank-root','output'):
        parser.add_argument('--'+name,required=True)
    parser.add_argument('--source-root')
    parser.add_argument('--upstream-real')
    parser.add_argument('--real-reference')
    parser.add_argument('--prediction-manifest-sha256',required=True)
    parser.add_argument('--root-GO',action='store_true')
    arguments=parser.parse_args()
    try:
        print(run(arguments))
    except BaseException as exc:
        import traceback
        failure=Path(arguments.output)/'FAILED_TECHNICAL.json'
        if not failure.exists():
            B.write(failure,dict(status='FAILED_TECHNICAL',error=repr(exc),traceback=traceback.format_exc(),
                command_arguments=vars(arguments),source_sha256=B.sha(__file__)))
        raise
