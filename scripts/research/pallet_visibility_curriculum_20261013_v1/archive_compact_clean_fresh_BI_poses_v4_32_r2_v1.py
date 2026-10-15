"""One-pass compact publication archive of sealed final poses, never inputs.

Preserve actual_pose exactly as stored, the top-level WD family, acceptance,
availability, source flags and saved selected scalar fields. This is not a
seed/start/trajectory/full costplan archive. No reference, RGB or fit is read.
"""
from copy import deepcopy
from pathlib import Path
import gzip
import hashlib
import json

from . import common as C

B=C.DOC/'nonsevere_v1/statewise_joint_ot_20261014_v2/nonsevere_v4_fix/boundary_recovery_v1'
SOURCE=B/'clean_fresh_BI_pose_eval_v4_32_r2'
OUTPUT=B/'clean_fresh_BI_compact_pose_archive_v4_32_r2_v1'
PLAN=B/'clean_fresh_BI_compact_pose_archive_plan_v4_32_r2_v1/PREDECLARATION.json'
ARMS=('SAME_FROZEN_E','CLEAN_FRESH_BI_ALL_BI_NATIVE','CLEAN_FRESH_BI_ALL_BI_CONTAINMENT',
      'CLEAN_FRESH_BI_SOURCE_NATIVE','CLEAN_FRESH_BI_SOURCE_CONTAINMENT')
SCALARS=('objective','native_observation_objective','regularizer_objective','global_containment_cost')


def valid_sha(value):return isinstance(value,str)and len(value)==64 and all(c in'0123456789abcdef'for c in value)


def compact(row):
    if set(row['arms'])!=set(ARMS):raise ValueError('All five sealed arms required')
    result={}
    for arm in ARMS:
        entry=row['arms'][arm];accepted=entry['accepted']
        available=entry.get('pose_available',True)if arm==ARMS[0]else entry['pose_available']
        if not isinstance(accepted,bool)or not isinstance(available,bool):raise ValueError('Explicit saved status required')
        if not available and accepted:raise ValueError('No pose cannot be accepted')
        pose=deepcopy(entry['actual_pose'])if available else None
        if available and not isinstance(pose,dict):raise ValueError('Finite pose dict required for available row')
        solution=entry.get('solution',{})
        if not isinstance(solution,dict):raise ValueError('Saved solution mapping required')
        result[arm]=dict(actual_pose=pose,WD_family=entry.get('WD_family'),accepted=accepted,
            pose_available=available,source_flags={k:deepcopy(entry[k])for k in(
                'source_policy','selected_endpoint_origin','fallback_audit','new_noise_availability')if k in entry},
            comparison_only=arm==ARMS[0],selected_scalar={k:deepcopy(solution[k])for k in SCALARS if k in solution})
    return dict(id=row['id'],partition=row['partition'],states=deepcopy(row['states']),arms=result,
        legacy_e_pose_semantics=deepcopy(row['legacy_e_pose_semantics']),
        pose_semantics='actual_pose copied unchanged; new native_t/centroid/Rphysical explicit; legacy centered e attestation retained',
        full_costplan_seed_start_or_trajectory_archive=False)


class HashReader:
    def __init__(self,stream):self.stream=stream;self.digest=hashlib.sha256();self.byte_count=0
    def read(self,n=-1):
        value=self.stream.read(n);self.digest.update(value);self.byte_count+=len(value);return value
    def tell(self):return self.stream.tell()


def declare(*,root_metadata_GO=False,expected_prediction_sha256=None):
    if not root_metadata_GO:raise PermissionError('Root metadata GO before IO')
    if not valid_sha(expected_prediction_sha256):raise ValueError('Explicit original prediction-seal SHA before IO')
    if PLAN.exists():raise FileExistsError('Preserve archive plan')
    if C.sha(SOURCE/'PREDICTION_SEAL.json')!=expected_prediction_sha256:raise ValueError('Current exact PRED seal')
    seal=C.read(SOURCE/'PREDICTION_SEAL.json')
    if seal['rows']!=238 or seal['arms']!=list(ARMS)or not seal['all_declared_arms_complete']:raise ValueError('Full238x5 required')
    PLAN.parent.mkdir(parents=True)
    C.write(PLAN,dict(schema='COMPACT_FINAL_POSES_PUBLICATION_PLAN_V32_R2',source_sha256={str(Path(p).relative_to(C.ROOT)):C.sha(p)for p in(__file__,C.__file__)},
        original_prediction_seal_sha256=expected_prediction_sha256,original_prediction_journal_sha256=seal['predictions_sha256'],
        rows=238,arms=list(ARMS),ordered_ids=seal['ordered_ids'],
        rule='Every ordered row and all five final actual_pose records, no error-based filtering or family transport',
        preserve_fields=['actual_pose','WD_family','accepted','pose_available','source_flags','saved selected_scalar','legacy_e_pose_semantics'],
        omit=['all input observations','full costplans','seeds','native start packets','continuous trajectories','reference/score geometry'],
        PRED_decode_passes=1,compressed_source_SHA_computed_during_same_read=True,no_NEW_reference_RGB_PnP_fit_or_optimizer=True))
    C.write(PLAN.with_name('PLAN_SEAL.json'),dict(plan_sha256=C.sha(PLAN)))
    return C.sha(PLAN)


def run(*,root_actual_archive_GO=False):
    if not root_actual_archive_GO:raise PermissionError('Root sealed-PRED archive GO before IO')
    if OUTPUT.exists():raise FileExistsError('Preserve archive attempts')
    plan=C.read(PLAN)
    if C.sha(PLAN)!=C.read(PLAN.with_name('PLAN_SEAL.json'))['plan_sha256']:raise ValueError('Frozen plan')
    for rel,sha in plan['source_sha256'].items():
        if C.sha(C.ROOT/rel)!=sha:raise ValueError('Frozen source changed')
    if C.sha(SOURCE/'PREDICTION_SEAL.json')!=plan['original_prediction_seal_sha256']:raise ValueError('Frozen prediction seal changed')
    OUTPUT.mkdir();seen=[];counts={p:0 for p in('CAL','TEST','REMAINDER')};armcounts={a:dict(rows=0,accepted=0,pose_available=0)for a in ARMS}
    with (SOURCE/'PREDICTIONS.jsonl.gz').open('rb')as raw:
        hashed=HashReader(raw)
        with gzip.GzipFile(fileobj=hashed,mode='rb')as decoded,gzip.open(OUTPUT/'POSES.jsonl.gz','wt',compresslevel=5)as dst:
            for line in decoded:
                if not line.strip():raise ValueError('No untracked empty row')
                row=json.loads(line);index=len(seen)
                if index>=238 or row['id']!=plan['ordered_ids'][index]:raise ValueError('Exact full order')
                value=compact(row);seen.append(row['id']);counts[row['partition']]+=1
                for a in ARMS:
                    armcounts[a]['rows']+=1;armcounts[a]['accepted']+=value['arms'][a]['accepted'];armcounts[a]['pose_available']+=value['arms'][a]['pose_available']
                dst.write(json.dumps(value,ensure_ascii=False,separators=(',',':'),allow_nan=False)+'\n')
        original_sha=hashed.digest.hexdigest();original_bytes=hashed.byte_count
    if (seen!=plan['ordered_ids']or len(set(seen))!=238 or counts!=dict(CAL=46,TEST=49,REMAINDER=143)
            or original_sha!=plan['original_prediction_journal_sha256']):raise ValueError('Original one-pass journal SHA/count changed')
    C.write(OUTPUT/'SUMMARY.json',dict(rows=238,cohorts=counts,arms=armcounts,original_prediction_journal_sha256=original_sha,
        original_compressed_bytes_read=original_bytes,decode_passes=1,all_rows_retained=True,
        full_costplan_seed_start_or_trajectory_archive=False,actual_pose_not_recomputed_or_transported=True))
    C.write(OUTPUT/'SEAL.json',dict(schema='SEALED_COMPACT_FINAL_POSE_ARCHIVE_V32_R2',plan_sha256=C.sha(PLAN),
        original_prediction_seal_sha256=plan['original_prediction_seal_sha256'],original_prediction_journal_sha256=original_sha,
        source_sha256=plan['source_sha256'],assets={n:C.sha(OUTPUT/n)for n in('POSES.jsonl.gz','SUMMARY.json')},
        rows=238,arm_rows=1190,all_rows_retained=True,reference_or_RGB_reads=0,new_pose_or_score_computation=0))
    return C.sha(OUTPUT/'SEAL.json')


if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('stage',choices=('declare','run'))
    p.add_argument('--root-metadata-GO',action='store_true');p.add_argument('--root-actual-archive-GO',action='store_true');p.add_argument('--prediction-seal-sha256')
    a=p.parse_args();print(declare(root_metadata_GO=a.root_metadata_GO,expected_prediction_sha256=a.prediction_seal_sha256)if a.stage=='declare'else run(root_actual_archive_GO=a.root_actual_archive_GO))
