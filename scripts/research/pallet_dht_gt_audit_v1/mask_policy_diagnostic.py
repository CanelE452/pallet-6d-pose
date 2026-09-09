"""Posthoc coordinate-footprint/mask audit; preserve all original metrics and GT."""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import json
from pathlib import Path

from .numeric_audit import ARMS, ROOT, sha, summarize


def run(run_dir):
    run_dir = Path(run_dir).resolve()
    numeric_path = run_dir/'NUMERIC_AUDIT.json'
    numeric = json.loads(numeric_path.read_text())
    assert numeric['complete'] and numeric['audit_integrity_PASS']
    csv_path = Path(numeric['csv']['path'])
    assert sha(csv_path) == numeric['csv']['sha256']
    bindings = {str(p): sha(p) for p in (numeric_path, csv_path, Path(__file__).resolve(), Path(__file__).with_name('numeric_audit.py').resolve())}
    rows = list(csv.DictReader(csv_path.open()))
    for r in rows:
        for key in ('width', 'height', 'visibility', 'point_id'):
            r[key] = int(r[key])
        for key in ('gt_x', 'gt_y'):
            r[key] = float(r[key])
        for key in ('gt_supervised', 'observed', 'in_frame', 'matched', 'predicted_valid'):
            assert r[key] in ('True', 'False')
            r[key] = r[key] == 'True'
        for arm in ARMS:
            key = f'error_{arm}_px'
            r[key] = float(r[key]) if r[key] else None
        r['computed_in_frame'] = 0 <= r['gt_x'] < r['width'] and 0 <= r['gt_y'] < r['height']
    assert len(rows) == 2871
    complete = summarize(rows)
    assert complete == numeric['metric_reconstruction']['summary']
    inside = [r for r in rows if r['computed_in_frame']]
    outside = [r for r in rows if not r['computed_in_frame']]
    cross = {}
    for vis in (0, 1, 2):
        for name, state in (('inside', True), ('outside', False)):
            cross[f'visibility{vis}_{name}'] = summarize([r for r in rows if r['visibility'] == vis and r['computed_in_frame'] == state])
    disagreements = [r for r in rows if r['computed_in_frame'] != r['in_frame']]
    p90 = complete['arms']['baseline']['p90_px']
    excerpt_keys = ('id', 'session_id', 'point_id', 'point_kind', 'visibility', 'reason', 'annotation_source',
        'width', 'height', 'gt_x', 'gt_y', 'gt_path', 'gt_sha256', 'in_frame', 'computed_in_frame',
        'gt_supervised', 'matched', 'observed', 'error_baseline_px', 'error_point_only_px', 'error_line_fusion_px')
    outside_observed = [r for r in outside if r['observed']]
    out = dict(schema='pallet_dht_gt_mask_policy_diagnostic_v1', complete=True, audit_integrity_PASS=True,
        generated_at=datetime.now(timezone.utc).isoformat(), input_sha256=bindings,
        scope='Posthoc descriptive stratification of frozen DEV319; no GT, evaluator, saved predictions or NUMERIC_AUDIT changed.',
        computed_inside_definition='0 <= GT.x < image.width and 0 <= GT.y < image.height, original-image coordinates, no tolerance or clipping.',
        original_scored_mask='visibility > 0 AND original finite predicted availability AND original baseline box IoU >= .5; no additional computed-in-frame restriction.',
        same_mask_within_each_group_for_all_three_arms=True,
        full_population=complete,
        by_computed_in_frame={'inside': summarize(inside), 'outside': summarize(outside)},
        by_stored_in_frame={str(v): summarize([r for r in rows if r['in_frame'] == v]) for v in (False, True)},
        visibility_cross_computed_inside=cross,
        outside_by_session={s: summarize([r for r in outside if r['session_id'] == s]) for s in sorted({r['session_id'] for r in outside})},
        outside_by_annotation_source={s: summarize([r for r in outside if r['annotation_source'] == s]) for s in sorted({r['annotation_source'] for r in outside})},
        stored_vs_computed_flag_disagreements=[{k: r[k] for k in excerpt_keys} for r in disagreements],
        outside_points=[{k: r[k] for k in excerpt_keys} for r in outside],
        tail_contribution=dict(full_baseline_p90_px=p90, n_all_observed_points_above_p90=sum(r['observed'] and r['error_baseline_px'] > p90 for r in rows),
            n_outside_observed_points=len(outside_observed),
            n_outside_observed_points_above_p90=sum(r['error_baseline_px'] > p90 for r in outside_observed)),
        limitations=['visibility0 outside points and visibility1 outside points have different score inclusion under the frozen visibility-only policy.',
            'This mask-policy difference does not establish that the underlying coordinate annotation is wrong.',
            'Inside-only P90 changes the denominator; it is not a corrected official metric or evidence that fusion improves the complete population.',
            'No model inference, GT adjustment, threshold choice, retraining or new held-out FINAL access.'])
    for path, digest in bindings.items():
        assert sha(path) == digest
    target = run_dir/'MASK_POLICY_DIAGNOSTIC.json'
    pending = target.with_suffix('.pending.json')
    pending.write_text(json.dumps(out, ensure_ascii=False, indent=2, allow_nan=False)+'\n')
    pending.replace(target)
    print(json.dumps(dict(path=str(target), sha256=sha(target), by_computed_in_frame=out['by_computed_in_frame'],
        visibility_cross_computed_inside=cross, n_flag_disagreements=len(disagreements), tail=out['tail_contribution']), indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-dir', default=str(ROOT/'data/pallet/results/pallet_dht_gt_audit_v1'))
    run(parser.parse_args().run_dir)
