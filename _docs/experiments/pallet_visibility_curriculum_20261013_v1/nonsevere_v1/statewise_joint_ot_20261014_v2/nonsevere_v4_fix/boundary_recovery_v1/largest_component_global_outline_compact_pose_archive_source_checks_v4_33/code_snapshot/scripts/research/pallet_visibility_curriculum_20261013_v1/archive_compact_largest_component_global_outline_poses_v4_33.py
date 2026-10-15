"""One-pass V33 final-pose publication archive, with frozen compact fields.

No input/E/observation archive is opened. The same pure compact function and
compressed HashReader preserve all1190 saved poses/WD/status/scalars exactly.
This is a small final-output archive, not seeds/starts/plans or trajectories.
"""
from pathlib import Path
import gzip
import json

from . import archive_compact_clean_fresh_BI_poses_v4_32_r2_v1 as OLD


C=OLD.C
B=OLD.B
SOURCE=B/'largest_component_global_outline_pose_eval_v4_33'
OUTPUT=B/'largest_component_global_outline_compact_pose_archive_v4_33_v1'
PLAN=B/'largest_component_global_outline_compact_pose_archive_plan_v4_33_v1/PREDECLARATION.json'
CENTRAL=B/'largest_component_global_outline_method_plan_v4_33/PREDECLARATION.json'
CENTRAL_SHA256='44e8b7835be40246997d76603e96a8b28ec80826fc16afd3be72d57478d290de'
OLD_SHA256='762afcc84fdc2f8a454bcca6bb17d55fdf16199eea3215f525cf0b14b5ad0d66'
PROFILE='LCC8_MAX_AREA_LOWEST_LABEL_FROZEN_V13_OUTLINE_V33'
ARMS=OLD.ARMS
SCALARS=OLD.SCALARS
compact=OLD.compact
HashReader=OLD.HashReader
valid_sha=OLD.valid_sha


def _sources():
    paths=(Path(__file__),Path(__file__).with_name('test_archive_compact_largest_component_global_outline_poses_v4_33.py'),
        Path(OLD.__file__),Path(C.__file__))
    return {str(path.resolve().relative_to(C.ROOT)):C.sha(path)for path in paths}


def _closed(prediction_sha256,score_sha256,audit_sha256):
    if (C.sha(SOURCE/'PREDICTION_SEAL.json')!=prediction_sha256
            or C.sha(SOURCE/'SCORE_SEAL.json')!=score_sha256
            or C.sha(SOURCE/'independent_root_audit_v1/AUDIT.json')!=audit_sha256
            or C.sha(CENTRAL)!=CENTRAL_SHA256 or C.sha(OLD.__file__)!=OLD_SHA256):
        raise ValueError('Current fullprediction/score/audit/central and frozen pure source required')
    pred=C.read(SOURCE/'PREDICTION_SEAL.json');score=C.read(SOURCE/'SCORE_SEAL.json')
    audit=C.read(SOURCE/'independent_root_audit_v1/AUDIT.json')
    if (pred['rows']!=238 or pred['arms']!=list(ARMS)or pred['arm_count']!=5
            or len(pred['ordered_ids'])!=238 or len(set(pred['ordered_ids']))!=238
            or pred['all_declared_arms_complete']is not True or pred['no_population_drop']is not True
            or pred['outline_selection_profile']!=PROFILE
            or score['prediction_seal']['sha256']!=prediction_sha256 or score['all_rows_kept']is not True
            or audit['rows']!=238 or audit['failure_count']!=0 or audit['failures']
            or audit['prediction_seal']['sha256']!=prediction_sha256
            or C.read(SOURCE/'independent_root_audit_v1/AUDIT_SEAL.json')['receipt_sha256']!=audit_sha256):
        raise ValueError('All238x5/current zeroaudit/postscore closure required before archive')
    return pred


def declare(*,root_metadata_GO=False,expected_prediction_sha256=None,
            expected_score_sha256=None,expected_audit_sha256=None):
    if root_metadata_GO is not True:raise PermissionError('ROOT postscore archive metadata GO before IO')
    if not all(valid_sha(v)for v in (expected_prediction_sha256,expected_score_sha256,expected_audit_sha256)):
        raise ValueError('Explicit completed prediction/score/audit SHA before IO')
    if PLAN.exists():raise FileExistsError('Preserve every V33 compact plan')
    pred=_closed(expected_prediction_sha256,expected_score_sha256,expected_audit_sha256)
    PLAN.parent.mkdir(parents=True)
    C.write(PLAN,dict(schema='LCC8_COMPACT_FINAL_POSES_PUBLICATION_PLAN_V33',source_sha256=_sources(),
        original_prediction_seal_sha256=expected_prediction_sha256,score_seal_sha256=expected_score_sha256,
        independent_audit_sha256=expected_audit_sha256,central_plan_sha256=CENTRAL_SHA256,
        original_prediction_journal_sha256=pred['predictions_sha256'],ordered_ids=pred['ordered_ids'],rows=238,arms=list(ARMS),
        outline_selection_profile=PROFILE,pure_compact_function='exact frozenV32 compact; no field/numeric conversion',
        preserve_fields=['actual_pose','topWD_family','accepted','pose_available','source_flags','saved selected scalars','legacy_e_pose_semantics'],
        PRED_decode_passes=1,compressed_source_SHA_during_same_read=True,
        allrows_no_error_selection=True,input_E_OBS_mask_reference_model_or_seed_reads=0,
        full_costplan_seed_start_or_trajectory_archive=False,accuracy_or_adoption_claim=False))
    C.write(PLAN.with_name('PLAN_SEAL.json'),dict(plan_sha256=C.sha(PLAN)))
    return C.sha(PLAN)


def run(*,root_actual_archive_GO=False):
    if root_actual_archive_GO is not True:raise PermissionError('ROOT postscore one-pass archive GO before IO')
    if OUTPUT.exists():raise FileExistsError('Preserve V33 compact attempts')
    plan=C.read(PLAN)
    if C.sha(PLAN)!=C.read(PLAN.with_name('PLAN_SEAL.json'))['plan_sha256']:
        raise ValueError('Frozen compact plan required')
    for relative,digest in plan['source_sha256'].items():
        if C.sha(C.ROOT/relative)!=digest:raise ValueError('Frozen compact source changed')
    pred=_closed(plan['original_prediction_seal_sha256'],plan['score_seal_sha256'],plan['independent_audit_sha256'])
    if pred['ordered_ids']!=plan['ordered_ids']or pred['predictions_sha256']!=plan['original_prediction_journal_sha256']:
        raise ValueError('Exact original final journal metadata required')
    OUTPUT.mkdir();seen=[];counts=dict(CAL=0,TEST=0,REMAINDER=0)
    armcounts={arm:dict(rows=0,accepted=0,pose_available=0)for arm in ARMS}
    try:
        with (SOURCE/'PREDICTIONS.jsonl.gz').open('rb')as raw:
            hashed=HashReader(raw)
            with gzip.GzipFile(fileobj=hashed,mode='rb')as decoded,gzip.open(OUTPUT/'POSES.jsonl.gz','wt',encoding='utf-8',compresslevel=5)as dst:
                for line in decoded:
                    if not line.strip():raise ValueError('Untracked empty row')
                    row=json.loads(line);index=len(seen)
                    part='CAL'if index<46 else'TEST'if index<95 else'REMAINDER'
                    if index>=238 or row['id']!=plan['ordered_ids'][index]or row['partition']!=part:
                        raise ValueError('Every fixed ordered238/cohort row required')
                    value=compact(row);seen.append(row['id']);counts[part]+=1
                    for arm in ARMS:
                        armcounts[arm]['rows']+=1;armcounts[arm]['accepted']+=value['arms'][arm]['accepted']
                        armcounts[arm]['pose_available']+=value['arms'][arm]['pose_available']
                    dst.write(json.dumps(value,ensure_ascii=False,separators=(',',':'),allow_nan=False)+'\n')
            original_sha=hashed.digest.hexdigest();original_bytes=hashed.byte_count
        if (seen!=plan['ordered_ids']or len(set(seen))!=238 or counts!=dict(CAL=46,TEST=49,REMAINDER=143)
                or original_sha!=plan['original_prediction_journal_sha256']):
            raise ValueError('Original full one-pass SHA/order/count changed')
        C.write(OUTPUT/'SUMMARY.json',dict(rows=238,cohorts=counts,arms=armcounts,decode_passes=1,
            original_prediction_journal_sha256=original_sha,original_compressed_bytes_read=original_bytes,
            actual_pose_not_recomputed_or_transported=True,all_rows_retained=True,
            outline_selection_profile=PROFILE,full_costplan_seed_start_or_trajectory_archive=False))
        _closed(plan['original_prediction_seal_sha256'],plan['score_seal_sha256'],plan['independent_audit_sha256'])
        for relative,digest in plan['source_sha256'].items():
            if C.sha(C.ROOT/relative)!=digest:raise ValueError('Frozen compact source changed during archive')
        C.write(OUTPUT/'SEAL.json',dict(schema='SEALED_LCC8_COMPACT_FINAL_POSE_ARCHIVE_V33',plan_sha256=C.sha(PLAN),
            original_prediction_seal_sha256=plan['original_prediction_seal_sha256'],original_prediction_journal_sha256=original_sha,
            score_seal_sha256=plan['score_seal_sha256'],independent_audit_sha256=plan['independent_audit_sha256'],
            source_sha256=plan['source_sha256'],assets={name:C.sha(OUTPUT/name)for name in ('POSES.jsonl.gz','SUMMARY.json')},
            rows=238,arm_rows=1190,all_rows_retained=True,reference_or_RGB_reads=0,new_pose_or_score_computation=0))
    except BaseException as error:
        C.write(OUTPUT/'FAILURE_V1.json',dict(exception=type(error).__name__,reason=str(error),rows_completed=len(seen),
            attempt_preserved=True,complete_success_seal_created=False))
        raise
    return C.sha(OUTPUT/'SEAL.json')


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('stage',choices=('declare','run'))
    parser.add_argument('--root-metadata-GO',action='store_true');parser.add_argument('--root-actual-archive-GO',action='store_true')
    for name in ('prediction','score','audit'):parser.add_argument('--'+name+'-sha256')
    args=parser.parse_args()
    print(declare(root_metadata_GO=args.root_metadata_GO,expected_prediction_sha256=args.prediction_sha256,
        expected_score_sha256=args.score_sha256,expected_audit_sha256=args.audit_sha256)if args.stage=='declare'
        else run(root_actual_archive_GO=args.root_actual_archive_GO))
