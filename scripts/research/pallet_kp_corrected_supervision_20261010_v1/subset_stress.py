"""Fixed oracle mask stress on easy/medium Base and N3 coordinates only.

Two diagnostic families separate human self-occlusion from coordinate accuracy.
No detector, head, RGB generation, training or deployment-policy change.
"""
import argparse
from collections import Counter
import gzip
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time
from . import subset_downstream as S
from . import scoring_resume as R

sys.dont_write_bytecode=True
FAMILIES=('HUMAN_NOSELF','KNOWN_ACCURATE_OBSERVATIONS')
CONDITIONS=(('ANCHOR',0,0),('DROP_ONE_GOOD',1,0),('DROP_TWO_GOOD',2,0),
            ('RETAIN_ONE_BAD',0,1),('RETAIN_TWO_BAD',0,2),('BOTH_ONE_ONE',1,1),('BOTH_TWO_TWO',2,2))


def fixed(args):
    output=Path(args.downstream)
    return dict(code=S.binding(__file__),cohort=S.binding(args.cohort),
        original_inputs=S.binding(S.OLD/'INPUTS.json'),original_correspondence=S.binding(S.OLD/'REAL_CORRESPONDENCE_ROWS.jsonl.gz'),
        original_poses=S.binding(S.OLD/'POSE_DIAGNOSTICS.jsonl.gz'),original_controls=S.binding(S.OLD/'FIXED_CONTROLS.jsonl.gz'),
        original_solver=S.binding(S.CODE/'solver.py'),original_inference=S.binding(S.CODE/'inference.py'),original_scoring=S.binding(S.CODE/'evaluate.py'),
        scoring_resume=S.binding(R.__file__),reference_mapping=S.binding(S.DOC/'REFERENCE_ID_MAPPING.json'),
        corrected_geometry=S.binding(output/'LEARNED_GEOMETRY_SEALED.jsonl.gz'),completed_scoring=S.binding(output/'SUBSET_EVALUATE_ADAPTER_RECEIPT.json'))


def freeze(args):
    cohort=S.read(args.cohort);count=cohort['count'];S.prior_snapshot()
    S.write_new(args.stress_protocol,dict(schema='easy_medium_fixed_oracle_mask_stress_protocol_v1',status='FROZEN_NOT_RUN',
        fixed_inputs=fixed(args),cohort_frames=count,coordinates=['BASE','N3_SUBPIX'],families=list(FAMILIES),
        conditions=[dict(name=n,drop_good=d,retain_bad=k) for n,d,k in CONDITIONS],rows=count*2*2*7,
        unique_coordinate_banks=count*2,maximum_four_ID_subsets_per_dimension=70,maximum_dimension_hypothesis_generators_per_bank=140,
        returned_solution_count_may_exceed_generator_count=True,residual_threshold_px=8.0,maximum_refinement_starts=3,
        HUMAN_NOSELF='anchor excludes exactly human SELF; drops DIRECT_VISIBLE eligible matched/valid error<=8; retains eligible human SELF (observation-invalid) with coordinate error independently recorded, not automatically coordinate-wrong',
        KNOWN_ACCURATE_OBSERVATIONS='anchor allows only DIRECT_VISIBLE eligible matched/valid error<=8; bad retention adds eligible matched/valid error>8 from any human state; excludes all unknown accuracy from anchor and injection pool',
        trust='Existing frozen same-coordinate baseline native permutation and GEOMETRIC_PROXY references; threshold is original8px, no retuning.',
        selection='ascending SHA256(frame:coordinate:family:drop_or_keep:cornerID), same nested IDs for1/2; no outcome-based selection',
        nontransformable='If either requested candidate pool is insufficient, run the unchanged family anchor, tag nontransformable; never relabel this as a pose failure or remove the row.',
        hidden_output='Only human SELF intersect excluded IDs are reprojected if a new fit exists; never fit those excluded coordinates or reuse their projections.',
        compare='Corresponding BASE or N3_SUBPIX simple historical control; fixed N3_SUBPIX also retained for every row; all full operational and conditional new results kept.',
        geometry='Record remaining accurate IDs and2D/registry3D SVD layout; rank is local arrangement evidence and does not establish global uniqueness.',
        GT_use='ORACLE diagnostic masks/coordinate-accuracy perturbation and posthoc score only; solver receives points,K,dimensions and chosen masks, never target pose/coordinates.',
        deployment_result=False,new_detector_forwards=0,new_head_forwards=0,new_training_updates=0,new_RGB=0,
        wall_budget_seconds=3600,performance_tuning=False))
    print('REAL_STRESS_PROTOCOL_FROZEN',S.sha(args.stress_protocol),flush=True)


def shape(np,points):
    p=np.asarray(points,float)
    if len(p)==0:return dict(count=0,numerical_rank=0,singular_values=[],axis_span=[])
    singular=np.linalg.svd(p-p.mean(0),compute_uv=False);largest=max(float(singular[0]),1e-300)
    return dict(count=len(p),numerical_rank=int((singular>largest*1e-10).sum()),singular_values=singular.tolist(),axis_span=(p.max(0)-p.min(0)).tolist())


def selected(fid,arm,family,role,pool,count):
    return sorted(pool,key=lambda k:hashlib.sha256(f'{fid}:{arm}:{family}:{role}:{k}'.encode()).hexdigest())[:count]


def run(args):
    plan=S.read(args.stress_protocol);S.require(plan['fixed_inputs']==fixed(args),'frozen stress inputs changed')
    before=S.prior_snapshot();cohort=S.read(args.cohort);ids=cohort['ids'];chosen=set(ids)
    out=Path(args.output).resolve();S.require(not out.exists(),'preserve complete/interrupted stress directory')
    protected=[S.REPO.resolve(),Path(args.source_root).resolve(),Path(args.baseline_root).resolve(),
               *(Path('/dev/shm')/name for name in ('pallet-observation-private-20261009','pallet-kp-difficulty-private-20261010',
                   'pallet-kp-supervision-gate-private-20261010','pallet-kp-supervision-repair-private-20261010'))]
    S.require(all(not out.is_relative_to(root) and not root.is_relative_to(out) for root in protected),'stress output must be isolated')
    out.mkdir(parents=True);S.write_new(out/'REAL_STRESS_STARTED.json',dict(protocol=S.binding(args.stress_protocol),configured_rows=plan['rows']))
    os.environ['PALLET_SOURCE_ROOT']=str(Path(args.source_root).resolve());os.environ['PALLET_BASELINE_ROOT']=str(Path(args.baseline_root).resolve())
    import numpy as np
    import cv2
    from scripts.research.pallet_observation_refiner_20261009_v1 import common as C
    from scripts.research.pallet_observation_refiner_20261009_v1.solver import HypothesisBank
    from scripts.research.pallet_observation_refiner_20261009_v1.inference import finish
    from scripts.research.pallet_observation_refiner_20261009_v1.evaluate import packet,score
    C.source_modules();cv2.setNumThreads(1)
    scope=dict(ids=ids,count=len(ids),authority={f['id']:f for f in S.read(S.OLD/'INPUTS.json')['frames']})
    E,reference_frames,targets,reference=R.references(C,scope)
    initial={(r['id'],r['method'].removesuffix('_NO_MASK_STANDARD')):r['initial_pose'] for r in S.rows(S.OLD/'POSE_DIAGNOSTICS.jsonl.gz') if r['id'] in chosen and r['method'] in ('BASE_NO_MASK_STANDARD','N3_SUBPIX_NO_MASK_STANDARD')}
    audits={(r['id'],r['method'].removesuffix('_NO_MASK_STANDARD')):r for r in S.rows(S.OLD/'REAL_CORRESPONDENCE_ROWS.jsonl.gz') if r['id'] in chosen and r['method'] in ('BASE_NO_MASK_STANDARD','N3_SUBPIX_NO_MASK_STANDARD')}
    controls={(r['id'],r['method']):r for r in S.rows(S.OLD/'FIXED_CONTROLS.jsonl.gz') if r['id'] in chosen and r['method'] in ('BASE','N3_SUBPIX')}
    start=time.monotonic();ledger=Counter();summary={};bank_records=[];completed=0
    partial=out/'REAL_STRESS_ROWS.partial.jsonl.gz'
    with S.primitives(cv2) as (primitives,state),gzip.open(partial,'wt',compresslevel=6) as stream:
        for fid in ids:
            f=scope['authority'][fid]
            for arm in ('BASE','N3_SUBPIX'):
                q=np.asarray(f['points'][arm],float);init=initial[(fid,arm)];a=audits[(fid,arm)]
                bank=HypothesisBank(q,f['K'],f['xyz'],image_size=(f['raw_hw'][1],f['raw_hw'][0]));eligible=set(bank.eligible)
                states=a['human_states_native'];H={k for k,s in enumerate(states) if s=='SELF_OCCLUDED'}
                valid=set(a['reference_valid_native_ids']) if a['reference_matched'] else set();errors=a['reference_error_input_native_px']
                accurate={k for k in valid if errors[k] is not None and math.isfinite(errors[k]) and errors[k]<=8.0}
                bad_coordinate={k for k in valid if errors[k] is not None and math.isfinite(errors[k]) and errors[k]>8.0}
                direct={k for k,s in enumerate(states) if s=='DIRECT_VISIBLE'};good=eligible&accurate&direct
                for family in FAMILIES:
                    anchor_excluded=H if family=='HUMAN_NOSELF' else set(range(8))-good
                    bad_pool=eligible&H if family=='HUMAN_NOSELF' else eligible&bad_coordinate
                    good_pool=good-set(anchor_excluded)
                    for name,drop_count,keep_count in CONDITIONS:
                        possible=len(good_pool)>=drop_count and len(bad_pool)>=keep_count
                        drop=selected(fid,arm,family,'drop',good_pool,drop_count) if possible else []
                        keep=selected(fid,arm,family,'keep',bad_pool,keep_count) if possible else []
                        excluded=(set(anchor_excluded)|set(drop))-set(keep);hidden=sorted(H&excluded)
                        answer=finish(bank,q,init,excluded=sorted(excluded),hidden=hidden,robust=True)
                        answer.update(input_points=q,initial_pose=init,oracle=True,oracle_phase='ORACLE_MASK_AND_FIXED_NATIVE_PHASE',reprojection_source='human SELF excluded only')
                        row=score(E,reference_frames[fid],targets[fid],packet(f,f'{arm}::{family}::{name}',answer))
                        pool=set(answer['solver']['used']);inliers=set(answer['solver']['inliers']);correct=pool&accurate;wrong=pool&bad_coordinate;unknown=pool-valid
                        baseline=controls[(fid,arm)]['pose'];n3=controls[(fid,'N3_SUBPIX')]['pose'];metric=row['pose']
                        delta={k:metric[k]-baseline[k] if metric.get('available') and baseline.get('available') else None for k in ('translation_cm','rotation_deg','ADDsym_m')}
                        known={k for k,s in enumerate(states) if s!='UNANNOTATED'};mask_wrong=bool((excluded^H)&known)
                        improved=delta['translation_cm'] is not None and delta['translation_cm']<0 and delta['rotation_deg']<0
                        worsened=delta['translation_cm'] is not None and (delta['translation_cm']>0 or delta['rotation_deg']>0)
                        outcome='mask_wrong_pose_improved' if mask_wrong and improved else 'mask_correct_pose_worse' if not mask_wrong and worsened else 'other_or_mixed'
                        row.update(coordinate=arm,family=family,condition=name,transformation_possible=possible,requested_drop=drop_count,requested_retain=keep_count,
                            nontransformable_reason=None if possible else dict(good_candidates=len(good_pool),bad_candidates=len(bad_pool)),
                            anchor_excluded=sorted(anchor_excluded),drop_good_ids=drop,retain_bad_ids=keep,
                            retained_bad_reasons={str(k):dict(human_state=states[k],self_observation_invalid=k in H,coordinate_outlier_over8px=k in bad_coordinate,coordinate_accurate_at8px=k in accurate,reference_error_px=errors[k]) for k in keep},
                            human_states_native=states,permutation_native_to_canonical=a['permutation_native_to_canonical'],reference_error_input_native_px=errors,
                            reference_matched=a['reference_matched'],reference_valid_native_ids=sorted(valid),accurate_threshold_px=8.0,
                            correct_pool_ids=sorted(correct),wrong_pool_ids=sorted(wrong),unknown_pool_ids=sorted(unknown),
                            correct_pool_count=len(correct),remaining_direct_correct_ids=sorted(pool&good),correct_final_inlier_ids=sorted(inliers&accurate),
                            wrong_final_inlier_ids=sorted(inliers&bad_coordinate),unknown_final_inlier_ids=sorted(inliers-valid),
                            correct_pool_image_shape=shape(np,q[sorted(correct)]),correct_pool_registry_shape=shape(np,bank.models[0][sorted(correct)]),
                            corresponding_simple_pose=baseline,fixed_N3_SUBPIX_simple_pose=n3,delta_vs_corresponding_simple=delta,
                            mask_wrong_on_known=mask_wrong,mask_pose_outcome=outcome,oracle_diagnostic_not_deployment=True,
                            initial_coordinates_changed=False,GT_pose_or_coordinates_input_to_solver=False)
                        stream.write(json.dumps(C.finite(row),ensure_ascii=False,separators=(',',':'),allow_nan=False)+'\n');completed+=1
                        key=arm+'::'+family+'::'+name;summary.setdefault(key,Counter()).update(rows=1,new_pose=int(row['new_pose_estimated']),fallback=int(row['fallback_used']),failure=int(row['no_pose']),transformable=int(possible),nontransformable=int(not possible))
                        if time.monotonic()-start>3600:raise TimeoutError('Keep partial stress rows; fixed3600s budget exceeded')
                ledger.update(bank.ledger);bank_records.append(dict(id=fid,coordinate=arm,input_hash=bank.digest,counts=bank.ledger,eligible=list(bank.eligible)))
            if len(bank_records)%52==0:
                print('REAL_ORACLE_STRESS',len(bank_records)//2,len(ids),completed,round(time.monotonic()-start,2),flush=True)
    final=out/'REAL_STRESS_ROWS.jsonl.gz';partial.rename(final)
    S.require(completed==plan['rows']==6860,'stress output population differs')
    S.require(dict(primitives['real']).get('solvePnPGeneric',0)==ledger['generic_calls'] and dict(primitives['real']).get('solvePnPRefineLM',0)==ledger['lm_calls'],'actual solver entry counters differ from bank ledgers')
    S.require(S.prior_snapshot()==before,'protected331 prior files changed')
    S.write_new(out/'REAL_STRESS_EXECUTION.json',dict(schema='easy_medium_oracle_mask_stress_execution_v1',complete=True,
        frames=len(ids),rows=completed,banks=len(bank_records),protocol=S.binding(args.stress_protocol),raw_rows=S.binding(final),
        counts=dict(ledger),actual_OpenCV_counts={k:dict(v) for k,v in primitives.items()},bank_generation=bank_records,
        condition_yields={k:dict(v) for k,v in summary.items()},wall_seconds=time.monotonic()-start,
        new_detector_forwards=0,new_head_forwards=0,new_training_updates=0,new_RGB=0,oracle_diagnostic_not_deployment=True,
        reference_scope=reference,protected_prior_files_preserved=331,nontransformable_not_pose_failure=True))
    print('REAL_STRESS_COMPLETE',completed,'banks',len(bank_records),round(time.monotonic()-start,2),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('stage',choices=('freeze','run'))
    p.add_argument('--cohort',default=str(S.DOC/'COHORT.json'));p.add_argument('--stress-protocol',default=str(S.DOC/'REAL_STRESS_PROTOCOL.json'))
    p.add_argument('--downstream',default='/dev/shm/pallet-kp-corrected-supervision-private-20261010/downstream')
    p.add_argument('--output',default='/dev/shm/pallet-kp-corrected-supervision-private-20261010/stress')
    p.add_argument('--source-root',default=os.environ.get('PALLET_SOURCE_ROOT'));p.add_argument('--baseline-root',default=os.environ.get('PALLET_BASELINE_ROOT'))
    args=p.parse_args();freeze(args) if args.stage=='freeze' else run(args)
