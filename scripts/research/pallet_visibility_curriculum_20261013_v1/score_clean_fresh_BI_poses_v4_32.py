"""Postseal V32 scores; fixed published references, all rows including no-pose.

The scorer does not open experiment inputs on import.
The root score gate, exact full238x5 prediction seal, independent audit and
complete ordered/e-exact scan precede every numeric reference decode. Native
translation is never mistaken for geometric center, nor native R for physical
R. Missing poses are explicit failures with undefined whole-population moments
and an infinite-failure maximum status; finite-only diagnostics are named.
"""
from collections import Counter
from copy import deepcopy
from pathlib import Path
from itertools import zip_longest

import numpy as np

from . import common as C
from . import evaluate_clean_fresh_BI_poses_v4_32_r2 as M
from . import scoring as METRIC
from . import score_remainder_derived_reference_v4_14 as REF

ARMS, PARTITIONS = M.ARMS, M.PARTITIONS
CONTROLLER_SHA256 = 'fa9ebe1d327972c585ee5f1f750a50f341151ec421c0e8ad5124a9a2e6b38187'
ANCHOR = M.B/'geometry_guided_edge_eval_v4_30/SCORE_SEAL.json'
ANCHOR_SHA256 = 'c2c85714f81dd90b78d46fd9e82541c9dc2ec1578e5e93c8f2209815ccf91d87'
PUBLIC_MANIFEST = C.DOC/'publication_followup_20261015_v4/COPIED_MANIFEST.json'
PUBLIC_MANIFEST_SHA256 = 'eb90dd261ec0c75c2e16d35422c4e9a2f380035baa741b13b09888b4a6f5df7f'
REFERENCE_PATHS = dict(CAL=C.DOC/'nonsevere_v1/OOF_pose_v1/ISOLATED_CAL_SCORING.json',
    TEST=C.DOC/'nonsevere_v1/test_reused_v1/ISOLATED_TEST_SCORING.json',REMAINDER=REF.OUTPUT/'PLAN.json')


def _rotation(value,name):
    return M.FACTORY.FRESH._rotation(value,name)


def prediction_pose(entry, *, arm, legacy_semantics=None):
    """Pure schema/physics conversion. A no-pose gate precedes all pose fields."""
    accepted=entry['accepted']
    if not isinstance(accepted,bool):raise ValueError('Explicit boolean accepted required')
    available=entry.get('pose_available',True) if arm==ARMS[0] else entry['pose_available']
    if not isinstance(available,bool):raise ValueError('Explicit boolean pose availability required')
    if not available:
        if accepted:raise ValueError('A no-pose row cannot be accepted')
        return None
    pose=entry['actual_pose']
    Rphysical=_rotation(pose['R_physical'],'Reported physical R')
    center=M.FACTORY.FRESH._array(pose['centroid'],(3,),'Reported geometric center')
    if arm==ARMS[0]:
        if (legacy_semantics is None or legacy_semantics.get('original_native_xyz_zero_mean_verified') is not True
                or legacy_semantics.get('comparison_only') is not True
                or legacy_semantics.get('numeric_worker_exposure') is not False
                or legacy_semantics.get('centroid_field')!='NATIVE_T_EQUAL_MODEL_CENTER_FOR_CENTERED_XYZ'):
            raise ValueError('Exact centered legacy-e comparison semantics required')
        return deepcopy(pose)  # Preserve its metric input; never re-estimate e.
    name=pose['WD_family']
    if name!=entry['WD_family'] or name not in M.FAMILIES:
        raise ValueError('Selected clean family must match actual pose')
    R=_rotation(pose['R_cf'],'Reported native R')
    Q=_rotation(pose['physical_registration_Q'],'Reported physical registration Q')
    t=M.FACTORY.FRESH._array(pose['native_t'],(3,),'Reported native translation')
    plan=entry['new_geometry_edge_cost_plans'][name]
    policy=next(p for p,arms in M.POLICY_ARMS.items()if arm in arms)
    if (plan['source_policy']!=policy or plan['observation_family']!=M.G.OBSERVATION_FAMILY
            or plan['population']!=M.G.POPULATION or plan['selection_profile']!=M.G.SELECTION_PROFILE):
        raise ValueError('Same typed V32 source/family cost plan required')
    geometry=plan['initial_model_ray_binding']['geometry']
    original_xyz=M.FACTORY.FRESH._array(geometry['native_xyz'],(8,3),'Known original XYZ')
    original_Q=_rotation(geometry['physical_registration_Q'],'Known original Q')
    # The trial WD is another native-channel correspondence. Only its native
    # origin/center basis is transported; no pervertex channel permutation is
    # invented and no reference or score chooses the family.
    trial_mean=Q@original_Q.T@original_xyz.mean(axis=0)
    expected_center=R@trial_mean+t
    if (not np.allclose(Rphysical,R@Q,atol=1e-10,rtol=0)
            or not np.allclose(center,expected_center,atol=1e-10,rtol=0)
            or not np.allclose(center,pose['model_center_camera_xyz'],atol=1e-12,rtol=0)
            or not np.allclose(R,entry['R'],atol=1e-12,rtol=0)
            or not np.allclose(t,entry['t'],atol=1e-12,rtol=0)
            or pose.get('centroid_semantics')!='GEOMETRIC_MODEL_CENTER_NOT_NATIVE_ORIGIN_TRANSLATION'):
        raise ValueError('Native t/geometric center/physical R-Q semantics changed')
    return deepcopy(pose)


def compact_predictions(predictions, comparisons, ordered_ids):
    """Scan every arm/row and exact parent e before any reference is read."""
    result=[]
    expected_parts=[part for part,count in PARTITIONS.items()for _ in range(count)]
    for i,(row,comparison)in enumerate(zip_longest(predictions,comparisons)):
        if (row is None or comparison is None or i>=len(ordered_ids)
                or row['id']!=ordered_ids[i] or comparison['id']!=row['id']
                or comparison['partition']!=row['partition'] or row['partition'] not in PARTITIONS
                or row['partition']!=expected_parts[i]
                or set(row['arms'])!=set(ARMS) or row['arms'][ARMS[0]]!=comparison['prediction']
                or row['legacy_e_pose_semantics']!=comparison['legacy_e_pose_semantics']):
            raise ValueError('All ordered238x5 and exact parent e comparison must survive before references')
        poses={arm:prediction_pose(row['arms'][arm],arm=arm,
            legacy_semantics=row['legacy_e_pose_semantics']) for arm in ARMS}
        result.append(dict(id=row['id'],partition=row['partition'],actual_poses=poses,
            accepted={a:row['arms'][a]['accepted']for a in ARMS},
            pose_available={a:poses[a]is not None for a in ARMS},
            WD_family={a:None if poses[a]is None else poses[a].get('WD_family')for a in ARMS}))
        del row,comparison
    if (len(result)!=238 or [r['id']for r in result]!=ordered_ids
            or len(set(ordered_ids))!=238 or Counter(r['partition']for r in result)!=Counter(PARTITIONS)):
        raise ValueError('Exact CAL46/TEST49/REMAINDER143; no missing/extra/duplicate rows')
    return result


def pose_metric(pose, reference, *, remainder=False):
    if pose is None:
        return dict(translation_cm=None,rotation_deg=None,ADDsym_cm=None,
            success_5cm_5deg=False,legacy_strict_success_5cm_5deg=False,
            translation_le5=False,pose_available=False,missing_pose_counts_as_failure=True)
    value=(REF.metric_for_reference if remainder else METRIC.pose_metric)(pose,reference)
    value['legacy_strict_success_5cm_5deg']=value['success_5cm_5deg']
    t,r=value['translation_cm'],value['rotation_deg']
    if not np.isfinite(t)or(r is not None and not np.isfinite(r)):
        raise ValueError('A reported finite pose must yield finite fixed-reference errors')
    value.update(success_5cm_5deg=None if r is None else bool(t<=5. and r<=5.),
        translation_le5=bool(t<=5.),pose_available=True,missing_pose_counts_as_failure=False)
    return value


def distribution(values, *, missing_pose_count=0, unresolved_reference_count=0):
    """All-row moments are not silently estimated from the finite subset."""
    finite=[float(x)for x in values if x is not None and np.isfinite(x)]
    stats=METRIC.distribution(finite)
    missing=len(values)-len(finite)
    if missing:
        total={k:None for k in('mean','median','P90','std','max')}
        total.update(n=len(values),status='UNDEFINED_OVER_FULL_DENOMINATOR',
            max_status='POSITIVE_INFINITY_NO_POSE_FAILURE'if missing_pose_count else'UNDEFINED_REFERENCE_ROTATION')
    else:total={**stats,'status':'FINITE_ALL_ROWS','max_status':'FINITE'if finite else'EMPTY'}
    return {**total,'finite_count':len(finite),'missing_or_undefined_count':missing,
        'missing_pose_count':missing_pose_count,'unresolved_reference_count':unresolved_reference_count,
        'finite_only_diagnostic':stats,'headline_uses_full_denominator':True}


def score_rows(compact, references):
    groups={p:[]for p in PARTITIONS}
    for row in compact:
        p=row['partition']
        groups[p].append(dict(id=row['id'],partition=p,accepted=row['accepted'],
            pose_available=row['pose_available'],WD_family=row['WD_family'],
            metrics={a:pose_metric(row['actual_poses'][a],references[p][row['id']],
                                  remainder=p=='REMAINDER')for a in ARMS}))
    groups['PRIMARY']=groups['TEST']+groups['REMAINDER']
    reports={}
    for p,count in((*PARTITIONS.items(),('PRIMARY',192))):
        rows=groups[p]
        if len(rows)!=count:raise ValueError('All fixed cohort rows must remain in scores')
        summary={}
        for arm in ARMS:
            values=[r['metrics'][arm]for r in rows]
            nopose=sum(not v['pose_available']for v in values)
            unresolved=sum(v['pose_available']and v['rotation_deg']is None for v in values)
            missingids=[r['id']for r in rows if not r['metrics'][arm]['pose_available']]
            finite=[r for r in rows if r['metrics'][arm]['translation_cm']is not None]
            finiteworst=max(finite,key=lambda r:r['metrics'][arm]['translation_cm'])if finite else None
            summary[arm]=dict(denominator=count,translation_cm=distribution([v['translation_cm']for v in values],missing_pose_count=nopose),
                rotation_deg=distribution([v['rotation_deg']for v in values],missing_pose_count=nopose,unresolved_reference_count=unresolved),
                translation_le5=sum(v['translation_le5']for v in values),
                successes_5cm_5deg=sum(bool(v['success_5cm_5deg'])for v in values),
                missing_pose_count=nopose,rotation_unresolved=nopose+unresolved,
                reference_rotation_unresolved_count=unresolved,
                fallback_count=sum(not r['accepted'][arm]and r['pose_available'][arm]for r in rows),
                unaccepted_count=sum(not r['accepted'][arm]for r in rows),
                max_translation_id=missingids[0]if missingids else finiteworst['id'],
                infinite_failure_max_ids=missingids,
                finite_only_max_translation_id=None if finiteworst is None else finiteworst['id'])
        reports[p]=dict(rows=rows,summary=summary,count=count,all_rows_kept=True,
            headline_joint_thresholds='translation_cm<=5 and rotation_deg<=5; legacy numeric errors unchanged',
            legacy_metric_boolean_was_strict_less_than=True,default='SAME_FROZEN_E unchanged',
            no_adoption_or_metrics_routing=True,TEST49_and_REMAINDER143_reused_exploratory=True,
            remaining_reference_is_derived_not_independent_metrology=True)
    return reports


def _reference_metadata(protocol, compact):
    if C.sha(ANCHOR)!=ANCHOR_SHA256 or C.sha(PUBLIC_MANIFEST)!=PUBLIC_MANIFEST_SHA256:
        raise ValueError('Exact published V30 fixed-reference anchor required')
    manifest=C.read(PUBLIC_MANIFEST)
    expected=[v for v in manifest['files']if v['path']==str(ANCHOR.relative_to(C.ROOT))]
    if len(expected)!=1 or expected[0]['sha256']!=ANCHOR_SHA256:
        raise ValueError('Published V4 manifest must pin the same V30 score seal')
    anchor=C.read(ANCHOR);bindings=anchor['reference_bindings']
    if set(bindings)!=set(PARTITIONS):raise ValueError('Same three declared reference roots required')
    inputs={str(ANCHOR.relative_to(C.ROOT)):ANCHOR_SHA256,
            str(PUBLIC_MANIFEST.relative_to(C.ROOT)):PUBLIC_MANIFEST_SHA256}
    for part,path in REFERENCE_PATHS.items():
        relative=str(path.relative_to(C.ROOT));binding=bindings[part]
        if binding['path']!=relative or not M._sha(binding['sha256'])or C.sha(path)!=binding['sha256']:
            raise ValueError('Fixed reference path/SHA changed: '+part)
        inputs[relative]=binding['sha256']
    refplan=C.read(REFERENCE_PATHS['REMAINDER'])  # Metadata inventory only.
    requested=[r['id']for r in compact if r['partition']=='REMAINDER']
    annotations=refplan['annotations']
    if (refplan['schema']!='DERIVED_FINAL143_REFERENCE_SCORE_PLAN_V4_14'
            or refplan['rows']!=143 or refplan['severe']!=0 or refplan['ids']!=requested
            or [a['id']for a in annotations]!=requested
            or refplan['annotation_coordinate_or_pose_values_decoded_at_preparation']is not False
            or refplan['no_reference_used_to_select_prediction_family_or_pose']is not True
            or refplan['all143_translation_rows_even_if_rotation_unresolved']is not True):
        raise ValueError('Exact existing143 derived reference inventory required')
    for a in annotations:
        path=Path(a['annotation'])
        if (path.is_absolute()or'..'in path.parts or not M._sha(a['annotation_sha256'])
                or refplan['input_sha256'].get(str(path))!=a['annotation_sha256']):
            raise ValueError('Exact original perannotation path/SHA required')
    inputs=M.OBS._union(protocol['input_sha256'],inputs,refplan['input_sha256'])
    M._verify(inputs,refplan['source_sha256'],refplan['protected_user_sha256'])
    return bindings,refplan,inputs


def _read_references(compact, refplan):
    result={}
    for part in PARTITIONS:
        requested=[r['id']for r in compact if r['partition']==part]
        if part=='REMAINDER':
            annotations={a['id']:a for a in refplan['annotations']}
            result[part]={ident:REF.reference_from_document(C.read(C.ROOT/annotations[ident]['annotation']))for ident in requested}
        else:
            labels=C.read(REFERENCE_PATHS[part])['rows']
            if any(ident not in labels for ident in requested):raise ValueError('Missing fixed reference; never drop a prediction row')
            result[part]={ident:labels[ident]['target']['truth']for ident in requested}
    return result


def _prediction_gate(output, expected_central, expected_prediction):
    seal=C.read(output/'PREDICTION_SEAL.json');protocol=C.read(output/'PROTOCOL.json')
    links={'predictions_sha256':'PREDICTIONS.jsonl.gz','protocol_sha256':'PROTOCOL.json',
        'input_seal_sha256':'INPUT_SEAL.json','execution_protocol_sha256':'PREDICTION_EXECUTION_PROTOCOL.json',
        'row_index_sha256':'PREDICTION_ROW_INDEX.json','separate_e_comparison_sha256':'E_COMPARISON.jsonl.gz'}
    if (C.sha(M.__file__)!=CONTROLLER_SHA256 or C.sha(output/'PREDICTION_SEAL.json')!=expected_prediction
            or seal['rows']!=238 or seal['arms']!=list(ARMS)or seal['arm_count']!=5
            or seal['all_declared_arms_complete']is not True or seal['no_population_drop']is not True
            or seal['before_fixed_reference_scoring']is not True
            or seal['default']!='SAME_FROZEN_E unchanged'
            or seal['ordered_ids']!=protocol['ordered_ids']or len(set(seal['ordered_ids']))!=238
            or protocol['central_plan_sha256']!=expected_central or C.sha(M.OBS.DECLARATION)!=expected_central
            or any(seal[k]!=C.sha(output/v)for k,v in links.items())):
        raise ValueError('Complete same-central sealed238x5 required before reference access')
    path=output/'independent_root_audit_v1/AUDIT.json';audit=C.read(path);aseal=C.read(path.with_name('AUDIT_SEAL.json'))
    if (audit['rows']!=238 or audit['failure_count']!=0 or audit['failures']
            or C.sha(path)!=aseal['receipt_sha256']
            or audit['prediction_seal']!=dict(path=str((output/'PREDICTION_SEAL.json').relative_to(C.ROOT)),sha256=expected_prediction)):
        raise ValueError('Current independent zero-failure238 audit must bind exact prediction seal')
    for key in('source_sha256','input_sha256','protected_user_sha256','original_CAL_file_sha256'):M._verify(protocol[key])
    return protocol


def score(*,root_score_GO=False,expected_central_plan_sha256=None,
          expected_prediction_seal_sha256=None,output=M.OUTPUT):
    if root_score_GO is not True:raise PermissionError('Root score GO before all IO/reference access')
    if not M._sha(expected_central_plan_sha256)or not M._sha(expected_prediction_seal_sha256):
        raise ValueError('Explicit central/prediction SHA required before IO')
    output=Path(output).resolve()
    names=['SCORING_PROTOCOL.json','SCORE_SEAL.json']+[p+'_SCORE.json'for p in(*PARTITIONS,'PRIMARY')]
    if any((output/name).exists()for name in names):raise FileExistsError('Preserve every score attempt')
    protocol=_prediction_gate(output,expected_central_plan_sha256,expected_prediction_seal_sha256)
    compact=compact_predictions(C.rows(output/'PREDICTIONS.jsonl.gz'),C.rows(output/'E_COMPARISON.jsonl.gz'),protocol['ordered_ids'])
    bindings,refplan,inputs=_reference_metadata(protocol,compact)
    stage_files=('PREDICTION_SEAL.json','PREDICTIONS.jsonl.gz','PROTOCOL.json','INPUT_SEAL.json',
        'PREDICTION_EXECUTION_PROTOCOL.json','PREDICTION_ROW_INDEX.json','E_COMPARISON.jsonl.gz',
        'independent_root_audit_v1/AUDIT.json','independent_root_audit_v1/AUDIT_SEAL.json')
    inputs=M.OBS._union(inputs,{str((output/name).relative_to(C.ROOT)):C.sha(output/name)for name in stage_files})
    sources=M.OBS._union(protocol['source_sha256'],refplan['source_sha256'],
        {str(Path(p).resolve().relative_to(C.ROOT)):C.sha(p)
         for p in(__file__,C.__file__,METRIC.__file__,REF.__file__,REF.PG.__file__,REF.SYM.__file__)})
    protected=M.OBS._union(protocol['protected_user_sha256'],refplan['protected_user_sha256'])
    M._verify(sources,inputs,protected)
    C.write(output/'SCORING_PROTOCOL.json',dict(schema='FIXED_REFERENCE_ALL_ROW_SCORING_PROTOCOL_V32',
        central_plan_sha256=expected_central_plan_sha256,prediction_seal_sha256=expected_prediction_seal_sha256,
        source_sha256=sources,input_sha256=inputs,protected_user_sha256=protected,
        reference_bindings=bindings,published_reference_anchor_sha256=ANCHOR_SHA256,
        all238_all5_and_exact_e_verified_before_numeric_references=True,
        headline_joint_rule='translation_cm<=5 and rotation_deg<=5',missing_pose_is_failure_not_dropped=True,
        numeric_reference_decodes_before_this_protocol=0))
    reports=score_rows(compact,_read_references(compact,refplan))
    for part,report in reports.items():
        report['reference_binding']=deepcopy(bindings if part=='PRIMARY'else bindings[part])
        C.write(output/(part+'_SCORE.json'),report)
    _prediction_gate(output,expected_central_plan_sha256,expected_prediction_seal_sha256)
    M._verify(sources,inputs,protected)
    C.write(output/'SCORE_SEAL.json',dict(schema='SEALED_FIXED_REFERENCE_CLEAN_FRESH_BI_SCORES_V32',
        prediction_seal=M._binding(output/'PREDICTION_SEAL.json'),scoring_protocol_sha256=C.sha(output/'SCORING_PROTOCOL.json'),
        score_sha256={p:C.sha(output/(p+'_SCORE.json'))for p in reports},reference_bindings=bindings,
        reference_anchor_sha256=ANCHOR_SHA256,all_numeric_references_decoded_only_after_all238x5_seal=True,
        all_rows_kept=True,default='SAME_FROZEN_E unchanged',new_parameter_or_default_selection_by_metrics=False))
    return M._binding(output/'SCORE_SEAL.json')


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root-score-GO',action='store_true')
    parser.add_argument('--central-plan-sha256',required=True)
    parser.add_argument('--prediction-seal-sha256',required=True)
    parser.add_argument('--output',type=Path,default=M.OUTPUT)
    args=parser.parse_args()
    print(score(root_score_GO=args.root_score_GO,expected_central_plan_sha256=args.central_plan_sha256,
                expected_prediction_seal_sha256=args.prediction_seal_sha256,output=args.output))
