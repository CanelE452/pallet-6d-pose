"""Independent remaining143 binding/coverage audit and post-seal comparison.

Preparation hashes annotation bytes only through the fixed scorer PLAN; no
reference numeric values are decoded. Postseal requires the complete143 pose
seal AND the separately owned reference-score seal. No inference or fitting.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from itertools import product
from pathlib import Path
import shutil
from datetime import datetime, timezone

import cv2
import numpy as np
from scipy.spatial.distance import cdist
from scipy.spatial.transform import Rotation

from . import common as C
from . import statewise_evaluation_v4 as E
from . import observe_remainder_rgb_v4_14 as O
from . import score_remainder_derived_reference_v4_14 as REF
from .boundary_partial_observations_v4_4 import fixed_fit_measurements
from .uncertain_u_regions_v4_5 import build_u_regions
from .ot_domains_v2 import validate_domain_bindings

OUTPUT = O.REMAINDER / 'independent_pipeline_audit_v4_14'
POSE = O.REMAINDER / 'pose_evaluation_v4_14'
OLD = O.BOUNDARY / 'frozen_wd_partial_refinement_eval_v4_13e'
ARMS = ('INITIAL_N3_SUBPIX', 'OLD_G10', 'D_JOINT_RESELECT',
        'FROZEN_WD_FACE_AWARE_PARTIAL_BI')
# Traced transitive mathematical/producer dependencies omitted by the live
# top-level source set. Presence here means their functions supply the runtime
# path, not that every function/branch inside each file executed on every image.
TRACED = {
    'statewise_uncertain_solver_v4_bias.py': 'P.BASE: soft BI/prior seed and constraints',
    'statewise_u_region_solver_v4_5.py': 'P.U: U-only constraints and parent solver',
    'statewise_boundary_normal_solver_v4_4.py': 'P.N: line normalization/Jacobian',
    'statewise_global_outline_solver_v4_10.py': 'G10.G: numeric whole-outline objective and refinement',
    'statewise_global_outline_solver_v4_11.py': 'D.G.OLD: analytic outline primitives',
    'global_outline_analytic_factor_v4_11.py': 'D.G.OLD.outline_value_and_gradient',
    'statewise_partial_BI_u_solver_v4_13.py': 'D.G.P: typed BI likelihood revocation and scalar guards',
    'visible_bi_global_edge_correction_v4_9c.py': 'D.OBS.G: actual RGB branch sampling/guards',
}


def binding(path):
    return E._binding(path)


def verify_source_closure(folder=OUTPUT):
    """Read-only pre/post guard usable by the separately owned scorer."""
    folder=Path(folder); receipt=C.read(folder/'ADDITIONAL_EXECUTED_SOURCE_BINDINGS_V1.json')
    E._verify_sources(receipt['conservatively_bound_source_union_sha256'])
    for relative,digest in receipt['conservatively_bound_source_union_sha256'].items():
        if C.sha(folder/'code_snapshot'/relative)!=digest:
            raise ValueError('Conservative dependency snapshot changed: '+relative)
    for item in receipt['archived_original49_source_sha_matches']:
        if C.sha(C.ROOT/item['path'])!=item['archived_sha256']:
            raise ValueError('Original49 achieved source changed')
    for key in ('achieved49_protocol','remaining_pose_protocol','remaining_observation_seal'):
        E._verify_binding(receipt[key])
    E._verify_sources(receipt['protected_user_sha256'])
    return receipt


def prepare(output=OUTPUT):
    output=Path(output).resolve()
    if output.exists():raise FileExistsError('Preserve prior independent audit')
    pose_protocol=C.read(POSE/'PROTOCOL.json'); old_protocol=C.read(OLD/'PROTOCOL.json')
    old_seal=C.read(OLD/'PREDICTION_SEAL.json')
    if C.sha(OLD/'PROTOCOL.json')!=old_seal['protocol_sha256']:
        raise ValueError('Achieved49 protocol seal changed')
    union=dict(old_protocol['source_sha256'])
    for rel,digest in pose_protocol['source_sha256'].items():
        if rel in union and union[rel]!=digest:raise ValueError('Old/fresh source identity conflict')
        union[rel]=digest
    traced={}
    script_root=Path(__file__).parent
    for name,description in TRACED.items():
        path=script_root/name;relative=str(path.relative_to(C.ROOT));digest=C.sha(path)
        if relative not in old_protocol['source_sha256'] or old_protocol['source_sha256'][relative]!=digest:
            raise ValueError('Executed dependency not identical to achieved49 archive: '+name)
        union[relative]=digest;traced[relative]=dict(sha256=digest,code_path=description,
            was_in_live_protocol=relative in pose_protocol['source_sha256'])
    for values in (union,pose_protocol['input_sha256'],pose_protocol['protected_user_sha256']):
        E._verify_sources(values)
    output.mkdir(parents=True)
    for relative,digest in union.items():
        path=output/'code_snapshot'/relative;path.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(C.ROOT/relative,path)
        if C.sha(path)!=digest:raise ValueError('Snapshot source mismatch')
    own=output/'audit_code_snapshot'/Path(__file__).name
    own.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(__file__,own)
    receipt=dict(schema='ADDITIVE_REMAINDER143_SOURCE_BINDING_CLOSURE_V1',
        before_remaining_GT_read_UTC=datetime.now(timezone.utc).isoformat(),
        numeric_remaining_GT_values_decoded=False,live_evaluator_protocol_or_source_modified=False,
        conservatively_bound_source_union_sha256=union,traced_transitive_runtime_dependencies=traced,
        union_is_conservative_and_does_not_assert_all_files_executed=True,
        extras_not_claimed_as_executed=[rel for rel in union if rel not in pose_protocol['source_sha256'] and rel not in traced],
        archived_original49_source_sha_matches=[dict(path=rel,archived_sha256=digest)
            for rel,digest in old_protocol['source_sha256'].items()],
        achieved49_protocol=binding(OLD/'PROTOCOL.json'),remaining_pose_protocol=binding(POSE/'PROTOCOL.json'),
        remaining_observation_seal=binding(O.OUTPUT/'OBSERVATION_SEAL.json'),
        protected_user_sha256=pose_protocol['protected_user_sha256'],audit_source=binding(__file__))
    C.write(output/'ADDITIONAL_EXECUTED_SOURCE_BINDINGS_V1.json',receipt)
    verify_source_closure(output)

    inventory=C.read(O.REMAINDER/'inventory_v2/REMAINING_CLEAN_MODERATE143.json')['rows']
    native=list(C.rows(O.INPUT/'NATIVE_INPUTS143.jsonl.gz'))
    observations=list(C.rows(O.OUTPUT/'OBSERVATIONS.jsonl.gz'))
    ids=[r['id'] for r in inventory]
    if ids!=[r['id'] for r in native] or ids!=[r['id'] for r in observations] or ids!=pose_protocol['ids']:
        raise ValueError('All143 original inventory rows required in every stage')
    model=O.load_truncation_interval_model(O.INTERVAL_MODEL)
    Q90=np.array([[0.,0.,1.],[0.,1.,0.],[-1.,0.,0.]])
    checks=0; groups=defaultdict(list);phase=Counter();summary_rows=[]
    for meta,argument,row in zip(inventory,native,observations):
        frame=argument['frame'];source=row['_original_source'];g=source['initial_geometry']
        if meta['severity'] not in ('CLEAN','MODERATE_OCCLUSION'):raise ValueError('Severe frame entered143')
        checks+=1
        for k in ('image','image_sha256','session'):
            if row[k]!=frame[k]:raise ValueError('Native/fresh identity mismatch')
            checks+=1
        if g!=argument['initial_geometry']:raise ValueError('RGB changed coarse/native registration')
        checks+=1
        Q=np.asarray(g['physical_registration_Q']);R=np.asarray(g['R_cf']);xyz=np.asarray(g['native_xyz'])
        dims=frame['known_physical_dimensions_xyz_m'];body=C.cuboid(*dims)
        distances=cdist(xyz@Q,body)
        gap=max(float(distances.min(0).max()),float(distances.min(1).max()))
        if gap>1e-7 or not np.allclose(R@Q,g['R_physical'],atol=1e-7,rtol=0):
            raise ValueError('Known unordered physical size / native Q mismatch')
        phase_name='I' if np.allclose(Q,np.eye(3),atol=1e-7,rtol=0) else 'Ry90'
        if phase_name=='Ry90' and not np.allclose(Q,Q90,atol=1e-7,rtol=0):
            raise ValueError('DEV source=True/Rxpi or undocumented registration convention')
        checks+=2;phase[phase_name]+=1
        hidden={k for k,h in enumerate(g['initial_self_hidden'])if h}
        if hidden!={k for k,s in enumerate(row['states'])if s=='S'}:raise ValueError('Initial S changed')
        checks+=1
        raw=source['original_raw_points'];gated=argument['raw_points_initial_S_gated']
        if raw!=gated or any(raw[k]!=[-1.,-1.]for k in hidden):raise ValueError('Initial S input carrier leak')
        checks+=1
        if hidden.intersection(source['mask_audit']['prompt_native_ids']) or hidden.intersection(source['state_observation'].get('rgb_queried_ids',[])):
            raise ValueError('S image feature/prompt query')
        checks+=1
        if row['frozen_BI_fit']!=fixed_fit_measurements(source['calibrated_visible_measurements'],row['correction']):
            raise ValueError('Recovered BI changed original CAL effective Sigma')
        checks+=1
        if row['u_region_observation']!=build_u_regions(source,model):raise ValueError('U range changed')
        checks+=1
        image=cv2.imread(str(C.ROOT/row['image']))
        states=['V' if s in ('B','I')else s for s in row['states']]
        validate_domain_bindings(source['ot_domains'],image,raw,states,g['K'],g['native_xyz'],
            prediction_source='N3_SUBPIX',record_id=row['id'],self_hidden=g['initial_self_hidden'])
        checks+=1
        if row['correction']['measured_points'][8:]!=source['original_raw_points'][8:]:raise ValueError('Ninth metadata changed')
        checks+=1
        key=tuple(float(x)for x in dims);groups[(frame['session'],key)].append(row)
        summary_rows.append(dict(id=row['id'],session=frame['session'],severity=meta['severity'],
            known_dims_xyz_m=dims,CF_extents=g['cf_extents'],Q_convention=phase_name,
            unordered_registered_shape_gap_m=gap,states=row['states'],initial_S_ids=sorted(hidden),
            nonself_prediction_scale_px=source['ot_domains']['nonself_prediction_scale_px'],
            BI_count=len(row['frozen_BI_fit']),legacy_partial_corner_count=len(row['partial_boundary_observation']['boundary_observations']),
            O_regions=sum(d['state']=='O'for d in source['ot_domains']['domains']),
            T_regions=sum(d['state']=='T'for d in source['ot_domains']['domains']),
            U_regions=len(row['u_region_observation']['u_regions'])))
    group_report=[]
    for (session,dims),rows in groups.items():
        group_report.append(dict(session=session,known_dims_xyz_m=list(dims),frames=len(rows),
            state_counts=dict(Counter(s for r in rows for s in r['states'])),
            BI_counts=dict(Counter(len(r['frozen_BI_fit'])for r in rows)),
            partial_frames=sum(bool(r['partial_boundary_observation']['boundary_observations'])for r in rows)))
    C.write(output/'PRESEAL_PIPELINE_REVIEW_V1.json',dict(rows=143,checks=checks,failures=0,
        sessions_and_known_sizes=group_report,Q_conventions=dict(phase),records=summary_rows,
        source_convention='historical predictor source=False; CF/canonical +Y down; Q=I or Ry90; no DEV source=True Rxpi',
        CAL_models_refitted=False,frozen_state_and_OT_models_not_new143_coverage_calibration=True,
        U_and_partial_tangent_coverage_borrowed_CAL_not_guaranteed=True,
        all143_RGB_and_masks_fresh=True,pose_fits_or_model_forwards_by_auditor=0,numeric_GT_values_decoded=False,
        contact_sheet_selection='first/middle/last ordered frame in each of four sessions; no error/GT selection'))
    _contact_sheet(output,observations)
    policy=C.read(REF.OUTPUT/'PLAN.json')
    C.write(output/'COMPARISON_PLAN_V1.json',dict(arms=list(ARMS),rows=143,
        fixed_reference_policy=policy['reference_policy'],reference_plan=binding(REF.OUTPUT/'PLAN.json'),
        reference_preparation_seal=binding(REF.OUTPUT/'PREPARATION_SEAL.json'),reference_source=binding(REF.__file__),
        auditor_source=binding(__file__),native_input_seal=binding(O.INPUT/'INPUT_SEAL.json'),
        observation_seal=binding(O.OUTPUT/'OBSERVATION_SEAL.json'),
        source_closure=binding(output/'ADDITIONAL_EXECUTED_SOURCE_BINDINGS_V1.json'),
        comparison_metrics='same143 nominal centroid translation; fixed canonical candidate0+C2 rotation/ADD if valid; unresolved R retained',
        parameter_selection_or_new_inference=False,postseal_requires_complete143_and_owned_reference_score_seals=True,
        paired_statistics='all ordered 4-session cluster resamples, frame-weighted mean; exploratory descriptive intervals'))
    print('prepared source closure and independent143 pre-GT pipeline audit',flush=True)


def _contact_sheet(output,rows):
    by=defaultdict(list)
    for row in rows:by[row['session']].append(row)
    tiles=[]
    for session in sorted(by):
        group=by[session]
        for index in (0,len(group)//2,len(group)-1):
            row=group[index];image=cv2.imread(str(C.ROOT/row['image']));tile=cv2.resize(image,(320,240))
            with np.load(O.OUTPUT/'masks'/row['mask_evidence']['mask_npz'])as a:
                mask=(a['pallet_mask_clean']>.5).astype(np.uint8)
            contours,_=cv2.findContours(mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
            sx=320/image.shape[1];sy=240/image.shape[0]
            for contour in contours:
                xy=contour.astype(float);xy[:,:,0]*=sx;xy[:,:,1]*=sy
                cv2.polylines(tile,[np.rint(xy).astype(np.int32)],True,(40,230,60),1)
            P=np.asarray(row['runtime']['initial_geometry']['initial_projection_xy'])*[sx,sy]
            hull=cv2.convexHull(P.astype(np.float32)).astype(np.int32)
            cv2.polylines(tile,[hull],True,(255,210,40),1)
            header=np.zeros((36,320,3),np.uint8)
            cv2.putText(header,row['id'],(4,15),cv2.FONT_HERSHEY_SIMPLEX,.37,(255,255,255),1)
            cv2.putText(header,''.join(row['states'])+'  green:mask cyan:coarse model',(4,29),cv2.FONT_HERSHEY_SIMPLEX,.32,(255,255,255),1)
            tiles.append(np.vstack([header,tile]))
    sheet=np.vstack([np.hstack(tiles[k:k+3])for k in range(0,len(tiles),3)])
    cv2.imwrite(str(output/'PRE_GT_SESSION_MASK_CONTACT_SHEET.png'),sheet)


def independent_metric(prediction,reference):
    R=np.asarray(prediction['R_physical'],float);t=np.asarray(prediction['centroid'],float)
    translation=float(np.linalg.norm(t-np.asarray(reference['t']))*100)
    if reference['R']is None:
        return dict(translation_cm=translation,rotation_deg=None,ADDsym_cm=None,success_5cm_5deg=None)
    target=np.asarray(reference['R']);xyz=C.cuboid(*reference['xyz'])
    rotations=[];adds=[]
    for S in (np.eye(3),np.diag([-1.,1.,-1.])):
        truth=target@S
        rotations.append(float(np.degrees(Rotation.from_matrix(truth.T@R).magnitude())))
        adds.append(float(np.linalg.norm(xyz@R.T+t-(xyz@truth.T+reference['t']),axis=1).mean()*100))
    rotation=min(rotations)
    return dict(translation_cm=translation,rotation_deg=rotation,ADDsym_cm=min(adds),
                success_5cm_5deg=translation<5 and rotation<5)


def _distribution(values):
    a=np.asarray(values,float)
    return dict(n=len(a),mean=float(a.mean())if len(a)else None,median=float(np.median(a))if len(a)else None,
                P90=float(np.quantile(a,.9))if len(a)else None,max=float(a.max())if len(a)else None)


def postseal(output=OUTPUT):
    output=Path(output).resolve();plan=C.read(output/'COMPARISON_PLAN_V1.json')
    if (output/'POSTSEAL_COMPARISON_V1.json').exists():raise FileExistsError('Preserve prior143 comparative audit')
    verify_source_closure(output)
    for key in ('reference_plan','reference_preparation_seal','reference_source','auditor_source',
                'native_input_seal','observation_seal','source_closure'):E._verify_binding(plan[key])
    seal=C.read(POSE/'PREDICTION_SEAL.json');protocol=C.read(POSE/'PROTOCOL.json')
    if (seal['rows']!=143 or seal['severe']!=0 or not seal['before_remaining_GT_or_metric_read']
        or seal['ordered_ids']!=protocol['ids'] or C.sha(POSE/'PREDICTIONS.jsonl.gz')!=seal['predictions_sha256']
        or C.sha(POSE/'PROTOCOL.json')!=seal['protocol_sha256']):raise ValueError('Complete all143 prediction-seal gate not met')
    score_seal=C.read(REF.OUTPUT/'SCORE_SEAL.json')
    if score_seal['prediction_seal_sha256']!=C.sha(POSE/'PREDICTION_SEAL.json')or C.sha(REF.OUTPUT/'POSE_REFERENCE_SCORE.json')!=score_seal['score_sha256']:
        raise ValueError('Independently owned postseal reference score required')
    scored=C.read(REF.OUTPUT/'POSE_REFERENCE_SCORE.json');predictions=list(C.rows(POSE/'PREDICTIONS.jsonl.gz'))
    native={r['id']:r for r in C.rows(O.INPUT/'NATIVE_INPUTS143.jsonl.gz')}
    reference_plan=C.read(REF.OUTPUT/'PLAN.json'); annotations={r['id']:r for r in reference_plan['annotations']}
    owned={r['id']:r for r in scored['rows']}
    result=[];maxdiff=dict(translation_cm=0.,rotation_deg=0.,ADDsym_cm=0.)
    for row in predictions:
        annotation=annotations[row['id']];path=C.ROOT/annotation['annotation']
        if C.sha(path)!=annotation['annotation_sha256']:raise ValueError('Fixed reference document changed')
        # First numeric GT decode in this auditor follows BOTH full prediction
        # seal and owned postseal score checks above. Fixed reader has no pose-
        # error-dependent native phase, dimension or signed-axis choice.
        reference=REF.reference_from_document(C.read(path))
        poses={a:row['arms'][a]['actual_pose']for a in ARMS[1:]}
        poses[ARMS[0]]=native[row['id']]['frame']['initial_pose']
        metrics={a:independent_metric(poses[a],reference)for a in ARMS}
        for name in maxdiff:
            if metrics[ARMS[-1]][name]is not None:
                maxdiff[name]=max(maxdiff[name],abs(metrics[ARMS[-1]][name]-owned[row['id']]['metric'][name]))
        if metrics[ARMS[-1]]['success_5cm_5deg']!=owned[row['id']]['metric']['success_5cm_5deg']:
            raise ValueError('Independent final143 success mismatch')
        result.append(dict(id=row['id'],session=row['session'],metrics=metrics,states=row['states'],
            frozen_family=row['frozen_family'],d_reselected_family=row['d_reselected_family'],
            accepted={a:row['arms'][a]['accepted']for a in ARMS[1:]},
            material_ambiguity={a:row['arms'][a]['ambiguity_audit']['material_ambiguity']for a in ARMS[1:]},
            reference_audit=reference['audit'],reference_metadata=reference['metadata']))
    if maxdiff['translation_cm']>1e-9 or maxdiff['rotation_deg']>1e-5 or maxdiff['ADDsym_cm']>1e-8:
        raise ValueError('Independent reference metrics differ materially: '+str(maxdiff))
    summary={}
    for a in ARMS:
        summary[a]=dict(translation_cm=_distribution([r['metrics'][a]['translation_cm']for r in result]),
            rotation_deg=_distribution([r['metrics'][a]['rotation_deg']for r in result if r['metrics'][a]['rotation_deg']is not None]),
            translation_le5=sum(r['metrics'][a]['translation_cm']<=5 for r in result),
            rotation_eligible=sum(r['metrics'][a]['rotation_deg']is not None for r in result),
            successes=sum(bool(r['metrics'][a]['success_5cm_5deg'])for r in result),
            fallback_count=None if a==ARMS[0]else sum(not r['accepted'][a]for r in result),
            max_translation_id=max(result,key=lambda r:r['metrics'][a]['translation_cm'])['id'])
    paired={a:_paired(result,a,ARMS[-1])for a in ARMS[:-1]}
    verify_source_closure(output)
    report=dict(schema='INDEPENDENT_REMAINDER143_POSTSEAL_COMPARISON_V1',rows=143,
        summary=summary,paired_final_minus_comparator=paired,records=result,
        final_metric_recompute_max_abs_difference=maxdiff,all143_failures_fallbacks_and_axis_flags_preserved=True,
        reference_policy=plan['fixed_reference_policy'],independent_physical_GT_or_unseen_holdout_claim=False,
        no_parameter_choice_or_model_fit_or_inference=True,
        prediction_seal=binding(POSE/'PREDICTION_SEAL.json'),owned_score_seal=binding(REF.OUTPUT/'SCORE_SEAL.json'))
    C.write(output/'POSTSEAL_COMPARISON_V1.json',report)
    C.write(output/'POSTSEAL_AUDIT_SEAL_V1.json',dict(report_sha256=C.sha(output/'POSTSEAL_COMPARISON_V1.json'),
        plan_sha256=C.sha(output/'COMPARISON_PLAN_V1.json'),source_closure_sha256=C.sha(output/'ADDITIONAL_EXECUTED_SOURCE_BINDINGS_V1.json'),
        numeric_GT_decoded_after_full143_prediction_and_owned_score_seals=True,rows=143))
    print({a:summary[a]for a in ARMS},flush=True)


def _paired(rows,baseline,final):
    sessions=sorted({r['session']for r in rows});draws=np.asarray(list(product(range(len(sessions)),repeat=len(sessions))))
    chunks=[np.array([r['metrics'][final]['translation_cm']-r['metrics'][baseline]['translation_cm']
                      for r in rows if r['session']==s])for s in sessions]
    means=np.array([x.sum()for x in chunks])[draws].sum(1)/np.array([len(x)for x in chunks])[draws].sum(1)
    delta=np.concatenate(chunks)
    return dict(mean_translation_delta_cm=float(delta.mean()),improved=int((delta < -1e-9).sum()),
        worsened=int((delta>1e-9).sum()),unchanged=int((np.abs(delta)<=1e-9).sum()),
        ordered_session_resamples=len(draws),session_cluster_resample_CI95=np.quantile(means,[.025,.975]).tolist(),
        CI_is_exploratory_four_session_descriptive_not_independent_population_certificate=True,
        translation_over5_to_le5_ids=[r['id']for r in rows if r['metrics'][baseline]['translation_cm']>5>=r['metrics'][final]['translation_cm']],
        translation_le5_to_over5_ids=[r['id']for r in rows if r['metrics'][baseline]['translation_cm']<=5<r['metrics'][final]['translation_cm']],
        by_session={s:dict(frames=len(x),mean_translation_delta_cm=float(x.mean()))for s,x in zip(sessions,chunks)})


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=('prepare','postseal'))
    parser.add_argument('--output',type=Path,default=OUTPUT);args=parser.parse_args()
    (prepare if args.stage=='prepare'else postseal)(args.output)
