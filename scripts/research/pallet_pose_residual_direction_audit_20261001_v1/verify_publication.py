"""Independent publication checks for the TRAIN-only direction input audit.

No new target, pose error, model score or policy is calculated. Pictures and
frozen numerical reports are accessed only by the explicitly invoked main CLI.
"""
from . import common as C
import argparse
import ast
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import re
import sys
from urllib.parse import unquote

import numpy as np
from PIL import Image

MODELS = ('R0', 'DIVERSE251_s1', 'DIVERSE251_s2', 'DIVERSE251_s3')
SCORERS = ('R0_ONLY', 'UNION_s1', 'UNION_s2', 'UNION_s3')
CHECKOUT = Path('/tmp/pallet-pose-github-review-20260930')
READS = []


def guard(event, args):
    if event != 'open' or not isinstance(args[0], (str, bytes, os.PathLike)):
        return
    path = Path(os.fsdecode(args[0])).resolve()
    if not path.is_relative_to(C.ROOT) or path.suffix in ('.py', '.pyc'):
        return
    name = str(path)
    forbidden = ('GEOMETRY_SIDETABLE', 'GEOMETRY_RESOLVED_POSE_GT', 'DIMENSION_SIDECAR',
                 'SOURCE_MANIFEST', 'SYNTH_RECORDS', 'SYNTH_LABELS', 'AXIS_REVIEW',
                 'TRUTH_FOR_DISPLAY', '/data/evaluation/', '/fits/', '/model_parameters/')
    assert not any(x in name for x in forbidden), ('PUBLIC_REFERENCE_OR_WEIGHT_DENIED', name)
    assert path.suffix not in ('.pt', '.pth'), ('NO_CHECKPOINT', name)
    READS.append(name)


def verify_bindings(value):
    if isinstance(value, dict):
        if 'path' in value and 'sha256' in value:
            C.verify(value)
        else:
            for x in value.values():
                verify_bindings(x)
    elif isinstance(value, list):
        for x in value:
            verify_bindings(x)


def read_csv(path):
    assert b'\r' not in path.read_bytes(), ('CSV_REQUIRES_LF', str(path))
    with path.open(newline='') as f:
        return list(csv.DictReader(f))


def csv_scalar(value):
    if value is None:
        return ''
    return str(value)


def assert_csv(path, expected):
    got = read_csv(path)
    assert len(got) == len(expected), (path.name, len(got), len(expected))
    for i, (row, raw) in enumerate(zip(got, expected)):
        assert set(row) == set(raw), (path.name, i, set(row) ^ set(raw))
        for key, value in raw.items():
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                assert float(row[key]) == value, (path.name, i, key)
            else:
                assert row[key] == csv_scalar(value), (path.name, i, key)
    return len(got)


def scalar_projection(pose, K):
    """Independent scalar camera-facing nine-point pinhole projection."""
    ext = np.asarray(pose['cf_extents'], np.float64)
    rot = np.asarray(pose['R_cf'], np.float64)
    t = np.asarray(pose['centroid'], np.float64)
    K = np.asarray(K, np.float64)
    assert ext.shape == (3,) and rot.shape == K.shape == (3, 3) and t.shape == (3,)
    assert all(np.isfinite(x).all() for x in (ext, rot, t, K))
    result = []
    signs = ((-1,-1,-1),(1,-1,-1),(1,1,-1),(-1,1,-1),
             (-1,-1,1),(1,-1,1),(1,1,1),(-1,1,1),(0,0,0))
    for sign in signs:
        point = [float(sign[j]) * float(ext[j]) / 2. for j in range(3)]
        camera = [sum(float(rot[a,b]) * point[b] for b in range(3)) + float(t[a]) for a in range(3)]
        homogeneous = [sum(float(K[a,b]) * camera[b] for b in range(3)) for a in range(3)]
        assert homogeneous[2] != 0
        result.append([homogeneous[0] / homogeneous[2], homogeneous[1] / homogeneous[2]])
    return np.array(result, np.float64)


def verify_scope(feature, representation, independent, audit_protocol, representation_protocol):
    for report in (feature, representation, independent):
        assert report['complete'] and report['PASS'] and report['source_TRAIN_only']
        assert report['frames'] == 2598 and not report['method_success'] and not report['goal_complete']
    assert feature['protocol'] == C.bind(C.DOC / 'AUDIT_PROTOCOL.json')
    assert representation['protocol'] == independent['protocol'] == C.bind(C.DOC / 'REPRESENTATION_PROTOCOL.json')
    assert representation_protocol['feature_protocol'] == independent['feature_protocol'] == feature['protocol']
    assert independent['audited_feature_receipt'] == C.bind(C.DOC / 'FEATURE_AUDIT.json')
    assert independent['audited_representation'] == C.bind(C.DOC / 'REPRESENTATION_AUDIT.json')
    assert feature['directions'] == C.bind(C.RAW / 'TRAIN_DIRECTIONS.npz')
    assert representation['arrays'] == C.bind(C.RAW / 'REPRESENTATION_ARRAYS.npz')
    assert set(feature['models']) == set(MODELS) and set(representation['models']) == set(SCORERS)
    assert set(independent['projection']) == set(MODELS) and set(independent['models']) == set(SCORERS)
    assert not feature['source_labels_read'] and not feature['VAL_quality_read'] and not feature['real_targets_read']
    for key in ('fits', 'image_forwards', 'PnP_calls', 'new_pose_estimates', 'argmin_calls', 'selected_policy_changes'):
        assert feature[key] == 0
    for key in ('optimizer_steps', 'new_fits', 'actual_data_weight_probes', 'actual_data_objective_probes',
                'new_policy_argmin', 'image_forwards', 'new_PnP_calls', 'raw_source_reference_reads', 'new_reference_metric_calculations'):
        assert representation[key] == 0
    assert not representation['prior_weights_read'] and not representation['VAL_quality_read'] and not representation['real_targets_read']
    for key in ('raw_reference_reads', 'VAL_quality_reads', 'real_reference_reads', 'new_model_fits',
                'model_weight_reads', 'new_objective_evaluations', 'new_policy_argmin', 'new_PnP_calls', 'image_forwards'):
        assert independent[key] == 0
    assert independent['all_projection_points'] == 4 * 5194 * 9
    assert representation['failed_rows_retained'] == independent['invalid_rows_retained'] == 1
    assert representation['available_anchor_rows'] == independent['available_anchor_rows'] == 2597
    assert audit_protocol['budgets'] == representation_protocol['budgets']
    assert all(value == 0 for value in audit_protocol['budgets'].values())
    assert audit_protocol['gallery'] == representation_protocol['gallery']
    assert audit_protocol['gallery']['display_arrow_scale'] == 20.
    for model in MODELS:
        assert feature['models'][model]['valid_candidates'] == 5194
        assert feature['models'][model]['allinvalid_rows'] == 1
        assert independent['projection'][model]['PASS']
    for model in SCORERS:
        span = representation['models'][model]['input_span']
        assert independent['models'][model]['PASS']
        assert independent['models'][model]['prior_six_hashes'] == representation['models'][model]['prior_six_hashes']
        assert independent['models'][model]['extra18_sha'] == representation['models'][model]['extra18_sha']
        assert independent['models'][model]['extended271_sha'] == representation['models'][model]['extended271_sha']
        assert not span['labels_used'] and not span['matrix_centering']
        assert not span['intercept_added'] and not span['frame_or_target_weights']
        assert span['threshold'] == 1e-4
    assert representation['novelty'] == dict(
        all_models_ratio_le1e_4=representation['novelty']['all_models_ratio_le1e_4'],
        status=representation['novelty']['status'], interpretation=representation['novelty']['interpretation'])
    assert independent['novelty']['status'] == representation['novelty']['status']
    assert independent['novelty']['performance_evidence'] is False
    forbidden = [C.RAW / 'fits', C.DOC / 'model_parameters', C.DOC / 'TRAINING_COMPLETE.json',
                 C.DOC / 'SOURCE_VAL_GATE.json', C.DOC / 'SOURCE_VAL_ROUTING_LOCK.json',
                 C.RAW / 'SOURCE_VAL_CHOICES.json', C.DOC / 'REAL_PROTOCOL.json',
                 C.DOC / 'REAL_ROUTING_LOCK.json', C.RAW / 'REAL_CHOICES.json',
                 C.DOC / 'REAL_RESULTS.json', C.RAW / 'POSE_METRICS.json']
    absence = {str(p.relative_to(C.ROOT)): not p.exists() for p in forbidden}
    assert all(absence.values()), absence
    return dict(source_TRAIN_only=True, frames=2598, original_invalid_rows_retained=1,
                all_projection_points=4*5194*9, new_fit_VAL_real_artifacts_absent=absence,
                new_fits=0, performance_evidence=False)


def verify_gallery(report, protocol):
    """Use only fixed input RGB, metadata, observations and frozen R0 poses."""
    gallery_binding = report['gallery']
    C.verify(gallery_binding)
    gallery = C.read(C.ROOT / gallery_binding['path'])
    rows = report['illustrations']
    assert gallery['illustrations'] == rows
    assert gallery['complete'] and gallery['status'] == 'FIXED_TRAIN_INPUT_ILLUSTRATIONS_ONLY'
    assert gallery['protocol'] == C.bind(C.DOC / 'AUDIT_PROTOCOL.json')
    assert gallery['metadata'] == protocol['inputs']['metadata']
    assert gallery['poses'] == protocol['inputs']['poses']
    assert gallery['predictions_lock'] == protocol['inputs']['source_predictions_lock']
    assert gallery['display_arrow_scale'] == 20. and gallery['additional_padding'] == 0
    assert gallery['actual_RGB_images'] == 6
    for key in ('new_fits', 'image_forwards', 'new_PnP_calls', 'new_pose_error_calculations', 'real_images_or_reference_reads'):
        assert gallery[key] == 0
    expected_ids = protocol['gallery']['ids']
    assert len(rows) == len(expected_ids) == len(set(expected_ids)) == 6
    assert [r['id'] for r in rows] == expected_ids
    assert gallery['selected_gallery_ids'] == expected_ids
    contract = C.read(C.ROOT / protocol['inputs']['source_contract']['path'])
    eligible = contract['fit_eligibility']['eligible_ids']['TRAIN']
    canonical = []
    for source in ('G38', 'P0', 'TEX'):
        canonical.extend(sorted(i for i in eligible if i.startswith(source + '__'))[:2])
    assert canonical == expected_ids
    metadata = C.read(C.ROOT / protocol['inputs']['metadata']['path'])
    index = {r['id']: i for i, r in enumerate(metadata)}
    by_id = {r['id']: r for r in metadata}
    with np.load(C.RAW / 'TRAIN_DIRECTIONS.npz', allow_pickle=False) as saved_directions:
        anchor_by_id = dict(zip(saved_directions['ids'].tolist(), saved_directions['anchor_index'].tolist()))
    poses = C.read(C.ROOT / protocol['inputs']['poses']['path'])
    predictions = C.read(C.ROOT / protocol['inputs']['source_predictions_lock']['path'])
    binding = predictions['receipts']['R0']; C.verify(binding)
    receipt = C.read(C.ROOT / binding['path'])
    assert receipt['complete'] and receipt['model'] == 'R0'
    C.verify(predictions['protocol'])
    source_protocol = C.read(C.ROOT / predictions['protocol']['path'])
    maximum_uv = maximum_arrow = 0.
    available_panels = 0
    images = []
    for row in rows:
        fid = row['id']; source = by_id[fid]
        assert source['split'] == 'TRAIN' and fid in eligible
        assert row['split'] == 'TRAIN' and row['source_family'] == fid.split('__')[0]
        assert row['source_index'] == index[fid]
        assert row['image'] == source['image'] and row['image_hw'] == source['hw']
        assert row['K'] == source['K'] and row['dimensions_m'] == source['dims']
        assert row['source_prediction'] == receipt['files'][index[fid]]
        assert row['source_prediction_receipt'] == binding
        C.verify(row['image']); C.verify(row['source_prediction'])
        with Image.open(C.ROOT / row['image']['path']) as im:
            assert [im.height, im.width] == source['hw']
            im.load()
            images.append(dict(id=fid, binding=row['image'], hw=source['hw'], format=im.format))
        saved = C.read(C.ROOT / row['source_prediction']['path'])
        assert saved['id'] == fid and saved['model'] == 'R0'
        assert saved['protocol_sha'] == predictions['protocol']['sha256']
        assert saved['checkpoint_sha'] == source_protocol['checkpoints']['R0']['sha256']
        prediction = saved['prediction']; selected = prediction['selected_index']
        assert row['prediction_selected_index'] == selected
        candidate = prediction['candidates'][selected] if selected is not None else None
        assert row['observed_q9'] == (candidate['keypoints_xy'] if candidate else None)
        assert row['bbox_xyxy'] == (candidate['box_xyxy'] if candidate else None)
        record = poses['records']['R0'][fid]
        assert row['anchor_hypothesis'] == record['GEO_name']
        assert row['anchor_pose'] == record['GEO_pose']
        assert row['anchor_index'] == anchor_by_id[fid]
        assert row['anchor_available'] == bool(anchor_by_id[fid] >= 0 and record['GEO_pose']['available'] and candidate is not None)
        assert row['display_arrow_scale'] == 20. and row['additional_padding'] == 0
        assert not row['GT_outline_shown'] and not row['physical_pose_errors_shown'] and not row['new_pose_estimate']
        assert not {'T_cm', 'R_deg', 'translation_cm', 'rotation_deg', 'reference_pose'} & set(row)
        if not row['anchor_available']:
            assert all(row[key] is None for key in ('projected_uv', 'residual_px', 'display_arrow_end_uv', 'bbox_diagonal'))
            continue
        available_panels += 1
        uv = scalar_projection(row['anchor_pose'], row['K'])
        q = np.asarray(row['observed_q9'], np.float64)
        residual = uv - q
        arrow = q + 20. * residual
        box = np.asarray(row['bbox_xyxy'], np.float64)
        diagonal = math.hypot(max(float(box[2]-box[0]), 1e-6), max(float(box[3]-box[1]), 1e-6))
        assert abs(row['bbox_diagonal'] - diagonal) <= 1e-12
        np.testing.assert_allclose(row['projected_uv'], uv, atol=1e-8, rtol=0)
        np.testing.assert_allclose(row['residual_px'], residual, atol=1e-8, rtol=0)
        np.testing.assert_allclose(row['display_arrow_end_uv'], arrow, atol=2e-7, rtol=0)
        maximum_uv = max(maximum_uv, float(np.max(abs(np.asarray(row['projected_uv']) - uv))))
        maximum_arrow = max(maximum_arrow, float(np.max(abs(np.asarray(row['display_arrow_end_uv']) - arrow))))
    return dict(PASS=True, images=images, raw_RGB_images=6, frozen_R0_pose_panels=available_panels,
                current_learned_pose_panels=0, projected_points=9*available_panels, arrow_scale=20.,
                added_padding_px=0, projection_max_absolute_px=maximum_uv,
                arrow_end_max_absolute_px=maximum_arrow,
                reference_pose_reads=0, new_T_R_calculations=0,
                selection='Protocol-fixed first two eligible TRAIN IDs per G38/P0/TEX source family.')


def verify_csv_exports(report, feature, representation, protocol):
    expected = {name: [] for name in ('FEATURE_PARITY.csv', 'REP_SUMMARY.csv',
                                     'EXTRA18_COLUMN_RESIDUAL.csv', 'TRAIN_ROW_MEMBERSHIP.csv')}
    for model in MODELS:
        row = feature['models'][model]
        values = dict(model=model, frames=row['frames'], valid_candidates=row['valid_candidates'],
                      allinvalid_rows=row['allinvalid_rows'])
        for key in ('raw94_px', 'raw94_bboxnorm', 'frozen_corner8_px'):
            values[key + '_max_absolute'] = row['parity_max_absolute'][key]
        for key in ('raw94_px', 'raw94_bboxnorm', 'frozen_corner8_px'):
            values[key + '_max_fraction_of_tolerance'] = row['parity_max_fraction_of_tolerance'][key]
        expected['FEATURE_PARITY.csv'].append(values)
    for model in SCORERS:
        data = representation['models'][model]; span = data['input_span']; collisions = data['collisions']
        old, new = collisions['old_all_valid'], collisions['extended_all_valid']
        row = dict(model=model, nonanchor_rows=span['rows'], old_rank=span['old']['rank'],
                   extended_rank=span['extended']['rank'], rank_gain=span['rank_gain'],
                   frobenius_residual_ratio=span['frobenius_residual_ratio'],
                   ratio_le1e_4=span['extra_information_below_fixed_ratio'],
                   old_repeated_groups=old['repeated_groups'],
                   old_anchor_nonanchor_groups=old['anchor_nonanchor_groups'])
        for kind in ('sign_conflict', 'strict_opposite'):
            for axis in ('T', 'R'):
                row[f'old_{axis}_{kind}_groups'] = old['axes'][axis][kind]['groups']
                row[f'new_{axis}_{kind}_groups'] = new['axes'][axis][kind]['groups']
        row['forced_anchor_rows'] = collisions['forced_anchor_rows']
        row['failed_rows_retained'] = collisions['all_invalid_frame_rows']
        expected['REP_SUMMARY.csv'].append(row)
        for j in range(18):
            expected['EXTRA18_COLUMN_RESIDUAL.csv'].append(dict(model=model, column=j,
                name=f'point{j//2}_{"x" if j%2 == 0 else "y"}',
                extra_column_norm=span['extra_column_norm'][j],
                residual_column_norm=span['residual_column_norm'][j],
                residual_ratio=span['per_column_residual_ratio'][j]))
    metadata = C.read(C.ROOT / protocol['inputs']['metadata']['path'])
    contract = C.read(C.ROOT / protocol['inputs']['source_contract']['path'])
    eligible = set(contract['fit_eligibility']['eligible_ids']['TRAIN'])
    with np.load(C.RAW / 'TRAIN_DIRECTIONS.npz', allow_pickle=False) as saved:
        ids, source_index, anchor = saved['ids'], saved['source_index'], saved['anchor_index']
        assert len(ids) == len(set(ids.tolist())) == 2598 and set(ids.tolist()) == eligible
        assert source_index.tolist() == [i for i, row in enumerate(metadata) if row['id'] in eligible]
        assert int(np.sum(anchor < 0)) == 1
        for j, fid in enumerate(ids.tolist()):
            row = metadata[int(source_index[j])]
            assert row['id'] == fid and row['split'] == 'TRAIN'
            value = dict(id=fid, source_index=int(source_index[j]), split='TRAIN',
                         source_family=fid.split('__')[0], anchor_index=int(anchor[j]),
                         anchor_available=bool(anchor[j] >= 0))
            value.update({m + '_valid_candidates': int(saved[m + '_valid'][j].sum()) for m in MODELS})
            expected['TRAIN_ROW_MEMBERSHIP.csv'].append(value)
    counts = {}
    assert set(report['exports']) == set(expected)
    for filename, rows in expected.items():
        path = C.DOC / filename
        counts[filename] = assert_csv(path, rows)
        manifest = report['exports'][filename]
        assert manifest['binding'] == C.bind(path) and manifest['rows'] == len(rows)
        with path.open(newline='') as f:
            columns = next(csv.reader(f))
        assert manifest['columns'] == columns and set(columns) == set(rows[0])
    assert counts == {'FEATURE_PARITY.csv': 4, 'REP_SUMMARY.csv': 4,
                      'EXTRA18_COLUMN_RESIDUAL.csv': 72, 'TRAIN_ROW_MEMBERSHIP.csv': 2598}
    assert report['csv_rows'] == counts
    return dict(rows=counts, all_cells_checked=True, line_endings='LF',
                no_training_label_array_values_read=True)


def verify_figures(report, representation):
    spans = [representation['models'][m]['input_span'] for m in SCORERS]
    expected = dict(rank_and_residual=dict(models=list(SCORERS),
        old_rank=[s['old']['rank'] for s in spans], extended_rank=[s['extended']['rank'] for s in spans],
        frobenius_residual_ratio=[s['frobenius_residual_ratio'] for s in spans], threshold=1e-4),
        column_residual_ratio=dict(models=list(SCORERS),
            columns=[f'P{j//2} {"x" if j%2 == 0 else "y"}' for j in range(18)],
            values=[s['per_column_residual_ratio'] for s in spans]))
    assert report['figure_values'] == expected
    paths = [C.DOC / 'figures' / name for name in ('input_rank_and_residual.png', 'extra18_column_residual.png',
        'train_input_residual_directions_1.jpg', 'train_input_residual_directions_2.jpg', 'train_input_residual_directions_3.jpg')]
    actual = sorted(p.resolve() for p in (C.DOC / 'figures').iterdir() if p.is_file())
    assert actual == sorted(p.resolve() for p in paths)
    assert report['figures'] == [C.bind(p) for p in paths]
    result = []
    for path in paths:
        with Image.open(path) as image:
            image.load()
            assert image.width > 800 and image.height > 500
            result.append(dict(binding=C.bind(path), width=image.width, height=image.height, format=image.format))
    return dict(summary_plots=2, gallery_pages=3, images_decodable=5,
                numeric_series=expected, image_inventory=result)


def verify_markdown(feature, representation, report):
    text = (C.DOC / 'REPORT_KO.md').read_text()
    expected = []
    for model in MODELS:
        row = feature['models'][model]; parity = row['parity_max_absolute']
        expected.append([model, str(row['frames']), str(row['valid_candidates']), str(row['allinvalid_rows']),
                         f"{parity['raw94_px']:.9g}", f"{parity['raw94_bboxnorm']:.9g}",
                         f"{parity['frozen_corner8_px']:.9g}"])
    for model in SCORERS:
        span = representation['models'][model]['input_span']
        expected.append([model, str(span['rows']), str(span['old']['rank']), str(span['extended']['rank']),
                         str(span['rank_gain']), f"{span['frobenius_residual_ratio']:.9g}",
                         str(span['extra_information_below_fixed_ratio'])])
    for model in SCORERS:
        collision = representation['models'][model]['collisions']
        old, new = collision['old_all_valid'], collision['extended_all_valid']
        row = [model, str(old['repeated_groups']), str(old['anchor_nonanchor_groups'])]
        for kind in ('sign_conflict', 'strict_opposite'):
            for axis in ('T', 'R'):
                row.append(f"{old['axes'][axis][kind]['groups']}→{new['axes'][axis][kind]['groups']}")
        expected.append(row)
    for row in report['illustrations']:
        h, w = row['image_hw']; dims = row['dimensions_m']
        expected.append([row['id'], f'{h}×{w}', *(f'{100*d:g}' for d in dims), str(row['anchor_available'])])
    identifiers = set(MODELS) | set(SCORERS) | {r['id'] for r in report['illustrations']}
    actual = []
    for line in text.splitlines():
        if line.startswith('|'):
            values = [part.strip() for part in line.strip().strip('|').split('|')]
            if values[0] in identifiers:
                actual.append(values)
    assert actual == expected and len(actual) == 18, ('MARKDOWN_NUMERIC_TABLE_MISMATCH', actual, expected)
    for exact in ('새 학습 0회', '새 VAL 성능 평가 0회', '실사 평가 0회',
                  'T/R 개선 효과는 측정하지 않았다', '원래 목표는 아직 달성되지 않았다',
                  '추가 padding은 0픽셀', '20배', '성능 gate가 아니다',
                  '정확한 충돌이 없더라도 표현이 충분', '실행 중 두 번의 파일 접근 중단',
                  representation['novelty']['status']):
        assert exact in text, ('MISSING_SCOPE_OR_HISTORY', exact)
    assert f"TRAIN **{representation['frames']:,}행**" in text
    assert f"유효 anchor {representation['available_anchor_rows']:,}행" in text
    assert f"전체 후보 실패 {representation['failed_rows_retained']}행" in text
    assert f"후보 {feature['normalization']['valid_candidate_count']:,}개" in text
    assert f"합계 {sum(feature['models'][m]['valid_candidates'] for m in MODELS):,}개" in text
    assert f"**{representation['novelty']['all_models_ratio_le1e_4']}**" in text
    pictures = re.findall(r'!\[[^\]]*\]\(([^)]+)\)', text)
    assert len(pictures) == len(set(pictures)) == 5
    assert sorted((C.DOC / name).resolve() for name in pictures) == sorted(
        (C.ROOT / b['path']).resolve() for b in report['figures'])
    return dict(numeric_rows=18, feature_rows=4, span_rows=4, collision_rows=4,
                physical_input_dimension_rows=6, embedded_figures=5,
                no_performance_claim_and_execution_history_checked=True)


def publication_files():
    excluded = {'PUBLIC_REVIEW.json', 'PUBLIC_REVIEW_KO.md', 'PUBLICATION_MANIFEST.json'}
    files = []
    for directory in (C.DOC, C.HERE):
        for path in sorted(directory.rglob('*')):
            if not path.is_file() or '__pycache__' in path.parts or path.name in excluded:
                continue
            assert path.stat().st_size < 50_000_000, ('PUBLIC_FILE_TOO_LARGE', str(path))
            assert path.suffix not in ('.pt', '.pth', '.npz', '.npy'), ('PRIVATE_ARRAY_OR_WEIGHT_IN_DOC', str(path))
            if path.suffix == '.py':
                ast.parse(path.read_text())
            files.append(C.bind(path))
    return files


def verify_metadata(report, feature, representation, independent):
    required_bindings = dict(protocol=C.DOC / 'AUDIT_PROTOCOL.json',
        representation_protocol=C.DOC / 'REPRESENTATION_PROTOCOL.json',
        feature_audit=C.DOC / 'FEATURE_AUDIT.json',
        representation_audit=C.DOC / 'REPRESENTATION_AUDIT.json',
        independent_verification=C.DOC / 'VERIFICATION.json',
        gallery=C.DOC / 'GALLERY_SELECTION.json')
    for key, path in required_bindings.items():
        assert report[key] == C.bind(path), key
    assert report['complete'] and not report['method_success'] and not report['goal_complete']
    assert report['actual_RGB_images'] == 6
    for key in ('new_fits', 'new_optimizer_steps', 'new_image_forwards', 'new_PnP_calls',
                'new_pose_error_calculations', 'VAL_quality_reads', 'real_reference_reads', 'real_routes'):
        assert report[key] == 0, key
    assert report['source_TRAIN_only']
    # These are hashes of existing frozen artifacts, not deserialization of
    # cached TRAIN target arrays or another execution of the mathematical audit.
    verify_bindings(report)
    verify_bindings(feature['bindings'])
    verify_bindings(representation['inputs'])
    verify_bindings(independent['bindings'])


def markdown_receipt(result):
    rows = result['csv']['rows']; gallery = result['gallery']
    lines = ['# 잔차 방향 입력 감사 공개물 독립 검산', '',
        '**공개물 검산 PASS. 새 학습 또는 T/R 개선 판정이 아니다.** 원래 TRAIN 2,598행과 invalid 1행을 유지한 입력 진단의 표·CSV·그림·사진 출처를 대조했다.', '',
        f"CSV 전체 {sum(rows.values()):,}행의 모든 셀을 동결 감사 JSON 및 TRAIN membership과 비교했다. 본문 수치표 {result['markdown']['numeric_rows']}행과 두 그래프의 원자료도 일치한다.", '',
        f"원본 TRAIN RGB 6개 SHA/픽셀 크기·치수·K·관측 q9·bbox·고정 R0 포즈를 확인했다. 독립 scalar 투영 최대 차이는 {gallery['projection_max_absolute_px']:.6g}px다. 화살표는 표시용 20배이고 추가 padding은0이다. GT outline 및 물리 T/R 성과 표시는 없다.", '',
        'PNG 2개와 JPG 3개를 디코딩했고, 실행자가 다섯 파일을 직접 시각 검토했다는 `--visual-reviewed` 확인을 같은 이미지 SHA에 결합했다. 전체 그림의 의미 판독을 해시 검사만으로 대신하지 않았다.', '',
        '새 모델 학습·weight 적용·선택 정책·PnP·참조 T/R 계산·VAL 품질·실사 평가는0이다. 공개 검증기는 기존 감사 결과를 확인했으며 새 열공간/타깃 회귀를 계산하지 않았다. 기존 TRAIN 타깃 NPZ의 바인딩 해시를 확인할 수 있지만 그 배열 값을 해석하지 않았다.', '',
        '이 영수증은 이후 수정하지 않는다. `PUBLICATION_MANIFEST.json`은 최종 공개 목록 생성 단계의 예정 링크이며, 생성 후 이 영수증을 다시 쓰지 않는다.', '',
        '[검산 JSON](PUBLIC_REVIEW.json) · [보고서](REPORT_KO.md) · [수학 검산](VERIFICATION_KO.md) · [공개 파일 목록](PUBLICATION_MANIFEST.json)', '']
    return '\n'.join(lines)


def main(visual_reviewed=False, dry_run=False):
    assert visual_reviewed, 'Five current rendered figures must be reviewed before invoking --visual-reviewed.'
    assert not (C.DOC / 'PUBLIC_REVIEW.json').exists() and not (C.DOC / 'PUBLIC_REVIEW_KO.md').exists(), 'Final receipt is immutable; do not repeat after freeze.'
    sys.addaudithook(guard)
    audit_protocol = C.protocol('AUDIT_PROTOCOL')
    C.verify(C.read(C.DOC / 'REPRESENTATION_PROTOCOL_SHA.json'))
    representation_protocol = C.read(C.DOC / 'REPRESENTATION_PROTOCOL.json')
    feature = C.read(C.DOC / 'FEATURE_AUDIT.json')
    representation = C.read(C.DOC / 'REPRESENTATION_AUDIT.json')
    independent = C.read(C.DOC / 'VERIFICATION.json')
    report = C.read(C.DOC / 'REPORT_DATA.json')
    scope = verify_scope(feature, representation, independent, audit_protocol, representation_protocol)
    verify_metadata(report, feature, representation, independent)
    csv_check = verify_csv_exports(report, feature, representation, audit_protocol)
    figures = verify_figures(report, representation)
    gallery = verify_gallery(report, audit_protocol)
    markdown = verify_markdown(feature, representation, report)
    links = check_links()
    files = publication_files()
    result = dict(complete=True, PASS=True, created_at=C.now(), status='PUBLIC_INPUT_AUDIT_VERIFIED',
        source_TRAIN_only=True, scope=scope, csv=csv_check, figures=figures, gallery=gallery,
        markdown=markdown, links=links, artifacts=files,
        visual_review=dict(operator_attested=True, mechanism='Explicit --visual-reviewed CLI argument',
                           image_bindings=report['figures'], images=5),
        report=C.bind(C.DOC / 'REPORT_KO.md'), report_data=C.bind(C.DOC / 'REPORT_DATA.json'),
        report_code=C.bind(C.HERE / 'report.py'), code=C.bind(__file__),
        mathematical_verification=C.bind(C.DOC / 'VERIFICATION.json'),
        input_protocol=C.bind(C.DOC / 'AUDIT_PROTOCOL.json'),
        representation_protocol=C.bind(C.DOC / 'REPRESENTATION_PROTOCOL.json'),
        new_model_fits=0, new_policy_argmin=0, new_PnP_calls=0,
        new_pose_error_calculations=0, cached_TRAIN_target_array_values_read=False,
        VAL_quality_values_read=False, real_reference_values_read=False,
        method_success=False, goal_complete=False, read_paths=sorted(set(READS)))
    if not dry_run:
        C.save(C.DOC / 'PUBLIC_REVIEW_KO.md', markdown_receipt(result))
        C.save(C.DOC / 'PUBLIC_REVIEW.json', result)
    print(json.dumps(dict(PASS=True, dry_run=dry_run, csv_rows=csv_check['rows'],
        markdown_numeric_rows=markdown['numeric_rows'], images=5,
        raw_RGB_images=6, projection_max_absolute_px=gallery['projection_max_absolute_px']), ensure_ascii=False), flush=True)
    return result


def check_links():
    pending = {'PUBLIC_REVIEW.json', 'PUBLIC_REVIEW_KO.md', 'PUBLICATION_MANIFEST.json'}
    checked, future = [], []
    for path in C.DOC.rglob('*.md'):
        for link in re.findall(r'\]\(([^)]+)\)', path.read_text()):
            if link.startswith(('http://', 'https://', '#')):
                continue
            destination = (path.parent / unquote(link.split('#')[0])).resolve()
            assert destination.is_relative_to(C.ROOT), (path, link)
            if destination.parent == C.DOC and destination.name in pending:
                if not destination.exists():
                    future.append(str(destination.relative_to(C.ROOT)))
            elif destination.is_relative_to(C.DOC) or destination.is_relative_to(C.HERE):
                assert destination.exists(), (path, link)
            else:
                assert (CHECKOUT / destination.relative_to(C.ROOT)).exists(), ('UNPUBLISHED_LINK', path, link)
            checked.append(dict(markdown=str(path.relative_to(C.ROOT)), target=str(destination.relative_to(C.ROOT))))
    return dict(checked=len(checked), links=checked,
                prospective_self_receipts=[p for p in sorted(set(future)) if Path(p).name.startswith('PUBLIC_REVIEW')],
                pending_publication_manifest=[p for p in sorted(set(future)) if Path(p).name == 'PUBLICATION_MANIFEST.json'])


def selfcheck():
    pose = dict(cf_extents=[1.2, .14, .8], R_cf=np.eye(3).tolist(), centroid=[.1, -.2, 4.])
    K = np.array([[600., 1., 320.], [0., 620., 240.], [0., 0., 1.]])
    projected = scalar_projection(pose, K)
    np.testing.assert_allclose(projected[8], [(600.*.1-.2)/4.+320., 620.*-.2/4.+240.], atol=1e-12, rtol=0)
    assert projected.shape == (9, 2) and np.isfinite(projected).all()
    assert csv_scalar(None) == '' and csv_scalar(False) == 'False'
    print('PUBLICATION_SELFCHECK_PASS invented_only=True artifact_reads=0', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--selfcheck', action='store_true')
    parser.add_argument('--visual-reviewed', action='store_true')
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()
    selfcheck() if args.selfcheck else main(args.visual_reviewed, args.dry_run)
