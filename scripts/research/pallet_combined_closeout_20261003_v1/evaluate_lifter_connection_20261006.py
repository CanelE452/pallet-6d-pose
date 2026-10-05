"""Evaluate the existing twelve-frame batch only after genuine human submission.

This CPU wrapper uses frozen predictions and the original canonical evaluator.
It never submits annotations, fills missing coordinates, or starts inference.
The original 120/24 plan remains intact; twelve-frame results are exploratory.
"""
from __future__ import annotations

from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import sys
import time

ROOT = Path(__file__).resolve().parents[3]
LIFTER = ROOT / 'scripts/research/pallet_lifter_case_review_20261003_v1'
RAW = ROOT / 'data/pallet/results/pallet_lifter_case_review_20261003_v1'
REVIEW = RAW / 'review'
DOC = ROOT / '_docs/experiments/pallet_combined_closeout_20261003_v1/lifter_connection_20261006_v1'


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def reference_gate(reference, frame_ids):
    """Fail closed before generating any derived predictions or evaluation."""
    if reference.get('source_kind') != 'human_reviewed':
        return 'WAITING_HUMAN_REFERENCE_SUBMISSION'
    primary = [r for r in reference.get('records', []) if r.get('review_pass') == 'primary']
    indexed = {r.get('frame_id'): r for r in primary}
    if len(indexed) != len(primary):
        raise ValueError('Duplicate primary reference frame IDs')
    for fid in frame_ids:
        record = indexed.get(fid)
        if not record or record.get('status') not in ('reviewed', 'skipped'):
            return 'WAITING_HUMAN_REFERENCE_SUBMISSION'
        actor = record.get('reviewer', {})
        if (record.get('source_kind') != 'human_reviewed'
                or actor.get('entered_by') != 'human'
                or actor.get('confirmation') is not True):
            return 'WAITING_HUMAN_REFERENCE_SUBMISSION'
        for c in record.get('corners', []):
            if c.get('visibility') == 'direct_visible' and (c.get('x') is None or c.get('y') is None):
                return 'WAITING_VISIBLE_CORNER_COORDINATES'
    return 'READY_FOR_FROZEN_REFERENCE_VALIDATION'


def evaluate():
    batch = read(REVIEW / 'SMALL_BATCH_12_V1.json')
    reference_path = REVIEW / 'LIFTER_REFERENCE_REVIEWED.json'
    if not reference_path.is_file():
        return dict(status='WAITING_HUMAN_REFERENCE_SUBMISSION',
                    visible_corner_accuracy='x', required_new_model_forward_frames=0)
    gate = reference_gate(read(reference_path), batch['frame_ids'])
    if gate != 'READY_FOR_FROZEN_REFERENCE_VALIDATION':
        return dict(status=gate, visible_corner_accuracy='x', required_new_model_forward_frames=0)

    # prepare validates the actual submission, freezes its primary snapshot and
    # binds the queue to the same original prediction run; it creates no choices.
    sys.path.insert(0, str(LIFTER))
    from open_object_match_annotation import prepare
    ctx = prepare(REVIEW / 'SMALL_BATCH_12_V1.json')
    sidecar_path = ctx.store_path.parent / 'LIFTER_OBJECT_MATCH_REVIEWED.json'
    if not sidecar_path.is_file():
        return dict(status='WAITING_HUMAN_OBJECT_MATCH', visible_corner_accuracy='x',
                    match_queue=str(ctx.queue_path), required_new_model_forward_frames=0)
    sys.path.insert(0, str(LIFTER / 'combined_integration'))
    from bridge import apply_sidecar, validate_sidecar
    sidecar = read(sidecar_path)
    validate_sidecar(ctx.queue, sidecar)
    decisions = {r['frame_id']: r for r in sidecar['records']}
    if set(decisions) != set(batch['frame_ids']):
        return dict(status='WAITING_HUMAN_OBJECT_MATCH', visible_corner_accuracy='x',
                    reviewed_match_frames=len(decisions), required_new_model_forward_frames=0)
    if any(r['decision'] == 'undetermined' for r in decisions.values()):
        return dict(status='WAITING_UNDETERMINED_OBJECT_MATCH', visible_corner_accuracy='x',
                    decision_counts=dict(Counter(r['decision'] for r in decisions.values())),
                    required_new_model_forward_frames=0)

    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    output = ROOT / 'data/pallet/results/pallet_combined_closeout_20261003_v1/lifter_connection_20261006_v1' / stamp
    output.mkdir(parents=True, exist_ok=False)
    identity = ctx.predictions_path.parent / 'RUN_IDENTITY.json'
    shutil.copyfile(identity, output / 'RUN_IDENTITY.json')
    derived = output / 'ALL_STORED_FRAMES_EVALUATOR.jsonl'
    apply_sidecar(ctx.predictions_path, ctx.queue_path, sidecar_path, derived,
                  output / 'L4_EVALUATOR_DERIVATION.json')
    sys.path.insert(0, str(LIFTER / 'metrics'))
    from evaluate import run_evaluation
    report = run_evaluation(REVIEW / 'MANIFEST.json', RAW / 'LIFTER_EVALUATION_PLAN.json',
                            REVIEW / 'CORNER_CONTRACT.json', output,
                            reviewed_path=ctx.reviewed_path, predictions_path=derived)
    corners = report.get('fixed_sample_visible_corner_accuracy')
    if corners is None or corners.get('reviewed_frame_count') != len(batch['frame_ids']):
        raise ValueError('The frozen evaluator did not evaluate the complete twelve-frame batch')
    result = dict(status='EVALUATED_PARTIAL_12_FRAME_EXPLORATORY',
                  frozen_full_cohort_status=report['status'], full_120_24_completed=False,
                  primary_scope=getattr(ctx, 'native_scope_metadata', {}),
                  original_manifest_frame_count=corners['planned_frame_count'],
                  batch_frame_count=len(batch['frame_ids']),
                  pending_primary_frame_count=corners['pending_primary_frame_count'],
                  methods=corners['methods'], paired_difference=corners['paired_difference'],
                  sessions=corners['sessions'], independent_physical_TR='x',
                  stationary_noise='x', repeat_quality_not_completed=True,
                  output_dir=str(output), paper_inserted=False,
                  raw_predictions_sha256=sha(ctx.predictions_path),
                  reference_snapshot_sha256=sha(ctx.reviewed_path),
                  human_match_sidecar_sha256=sha(sidecar_path),
                  prediction_run_identity_sha256=sha(identity),
                  canonical_mode='canonical', new_model_forward_frames=0,
                  machine_human_review_created=False)
    (output / 'PARTIAL_BATCH_RESULT.json').write_text(json.dumps(result, ensure_ascii=False,
                                              indent=2, allow_nan=False) + '\n')
    return result


def main():
    started = time.perf_counter()
    result = evaluate()
    result.update(generated_at=datetime.now(timezone.utc).isoformat(),
                  cpu_wall_seconds=time.perf_counter() - started,
                  training_runs=0, optimizer_updates=0, hardware_control_calls=0,
                  script_sha256=sha(Path(__file__)))
    DOC.mkdir(parents=True, exist_ok=True)
    path = DOC / 'EVALUATION_GATE_STATUS.json'
    if path.exists():
        with path.with_suffix('.history.jsonl').open('a', encoding='utf-8') as handle:
            handle.write(json.dumps(read(path), ensure_ascii=False, allow_nan=False) + '\n')
    path.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    print(json.dumps(result, ensure_ascii=False, allow_nan=False))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
