"""Independent publication checks for an input audit, never a pose-quality test."""
from pathlib import Path
import argparse
import ast
import hashlib
import json
import os
import re
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
NAME = 'pallet_pose_dino_input_audit_20261001_v1'
HERE = Path(__file__).resolve().parent
DOC = ROOT / '_docs/experiments' / NAME
RAW = ROOT / 'data/pallet/results' / NAME
CHECKOUT = Path('/tmp/pallet-pose-github-review-20260930')
MODELS = ('R0', 'DIVERSE251_s1', 'DIVERSE251_s2', 'DIVERSE251_s3')
HYP = ('long-face-front', 'short-face-front')
PENDING = {'PUBLIC_REVIEW.json', 'PUBLIC_REVIEW_KO.md', 'PUBLICATION_MANIFEST.json'}


def bind(path):
    path = Path(path).resolve()
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(chunk)
    return dict(path=str(path.relative_to(ROOT)), sha256=h.hexdigest(), bytes=path.stat().st_size)


def read(path):
    return json.loads(Path(path).read_text())


def bound_read(binding):
    path = ROOT / binding['path']
    assert bind(path) == binding, binding['path']
    return read(path)


def save(path, value):
    with path.open('x') as f:
        if isinstance(value, str):
            f.write(value)
        else:
            json.dump(value, f, ensure_ascii=False, indent=2, allow_nan=False)
            f.write('\n')


def guard():
    outputs = {DOC / 'PUBLIC_REVIEW.json', DOC / 'PUBLIC_REVIEW_KO.md'}
    def audit(event, args):
        if event != 'open' or not isinstance(args[0], (str, bytes, os.PathLike)):
            return
        path = Path(os.fsdecode(args[0])).resolve()
        mode, flags = args[1:3]
        writing = (isinstance(mode, str) and any(x in mode for x in 'wax+')) or (
            isinstance(flags, int) and bool(flags & (os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_TRUNC | os.O_APPEND)))
        if writing and path.is_relative_to(ROOT):
            assert path in outputs, ('PUBLIC_REVIEW_WRITE_SCOPE', str(path))
        if not writing:
            forbidden = ('SOURCE_TRAIN_LABELS', 'GEOMETRY_RESOLVED_POSE_GT', 'GEOMETRY_SIDETABLE',
                         'DIMENSION_SIDECAR', 'SYNTH_LABELS', 'SYNTH_RECORDS', 'TRUTH_FOR_DISPLAY')
            assert not any(x in str(path) for x in forbidden), ('NO_TARGET_READ', str(path))
            assert path.suffix not in ('.pt', '.pth'), ('NO_MODEL_LOAD', str(path))
    sys.addaudithook(audit)


def scalar_projection(pose, K):
    """Separate scalar camera equations; no production projector or PnP."""
    ext, rot, t = pose['cf_extents'], pose['R_cf'], pose['centroid']
    signs = ((-1,-1,-1),(1,-1,-1),(1,1,-1),(-1,1,-1),
             (-1,-1,1),(1,-1,1),(1,1,1),(-1,1,1))
    result = []
    for s in signs:
        x = [s[j] * ext[j] / 2 for j in range(3)]
        camera = [sum(rot[i][j] * x[j] for j in range(3)) + t[i] for i in range(3)]
        if camera[2] <= 0:
            result.append([0., 0.])
            continue
        q = [sum(K[i][j] * camera[j] for j in range(3)) for i in range(3)]
        result.append([q[0] / q[2], q[1] / q[2]])
    return np.array(result, np.float64)


def public_paths():
    paths = [ROOT / 'readme.md', ROOT / '.gitignore', *HERE.glob('*.py'),
             *DOC.glob('*.md'), *DOC.glob('*.json'),
             *(DOC / 'figures').glob('*.png'), *(DOC / 'figures').glob('*.jpg')]
    paths = sorted(p for p in paths if p.name not in PENDING)
    assert len(paths) == len(set(paths)) and all(p.stat().st_size < 20_000_000 for p in paths)
    return paths


def check_links(paths):
    count = 0
    for path in paths:
        if path.suffix != '.md':
            continue
        for target in re.findall(r'!?\[[^\]]*\]\(([^)]+)\)', path.read_text()):
            target = target.strip().strip('<>')
            if re.match(r'^[a-z]+://', target) or target.startswith('#'):
                continue
            target = target.split('#', 1)[0]
            if not target:
                continue
            dest = (path.parent / target).resolve()
            assert dest.is_relative_to(ROOT), ('LINK_OUTSIDE_REPOSITORY', target)
            if dest.name in PENDING and dest.parent == DOC:
                continue
            assert dest.exists() or (CHECKOUT / dest.relative_to(ROOT)).exists(), ('MISSING_LINK', str(path), target)
            count += 1
    return count


def verify_cache(cache):
    assert cache['complete'] and cache['PASS'] and not cache['current_VAL_quality_read'] and not cache['real_data_read']
    for key in ('new_forwards', 'new_fits', 'new_argmin', 'new_pose_errors', 'GT_NPZ_member_reads', 'legacy_GT_array_reads'):
        assert cache[key] == 0, key
    mapping = cache['identity_mapping']
    assert len(mapping) == 5120 and len({r['id'] for r in mapping}) == 5120
    definitions = (('all5120', mapping, 5120, 710),
                   ('eligibleTRAIN2598', [r for r in mapping if r['eligible_TRAIN']], 2598, 376),
                   ('VAL1024', [r for r in mapping if r['split'] == 'VAL'], 1024, 140))
    for name, rows, total, matched in definitions:
        assert len(rows) == total
        assert sum(r['wide_input_cache_present'] for r in rows) == matched
        assert not any(r['strict_wide_reusable'] for r in rows)
        s = cache['summary'][name]
        assert s['rows'] == total and s['wide_image_identity_matches'] == matched and s['strict_reusable'] == 0
    assert cache['inspected_source_images'] == 710 and cache['verified_unique_token_caches'] == 850
    diversity = cache['phases']['dino_source_diversity']
    assert diversity['cache_rows'] == 8505 and diversity['source_rows'] == 8192 + 64 and diversity['real_rows'] == 249
    return cache['summary']


def verify_report(report, receipt, verification, cache, protocol):
    assert report['complete'] and receipt['complete'] and receipt['input_construction_pass']
    assert verification['complete'] and verification['PASS'] and verification['verification_PASS']
    assert receipt['protocol'] == verification['protocol'] == bind(DOC / 'INPUT_PROTOCOL.json')
    assert verification['tokens'] == receipt['tokens'] and verification['descriptors'] == receipt['descriptors']
    assert verification['full_token_file_sha_checked'] and verification['normalization_exact']
    assert verification['code'] == bind(HERE / 'verify_train_inputs.py')
    assert cache['code'] == bind(HERE / 'cache_contract.py')
    for binding in protocol['codes']:
        assert bind(ROOT / binding['path']) == binding
    original = bind(RAW / 'TRAIN_APPEARANCE_INPUTS.json')
    assert verification['input_receipt'] == original
    public = bind(DOC / 'TRAIN_APPEARANCE_INPUTS.json')
    assert (original['sha256'], original['bytes']) == (public['sha256'], public['bytes'])
    assert receipt['frames'] == verification['frames'] == 2598
    assert receipt['image_forwards'] == verification['token_frame_hashes_checked'] == 2597
    assert verification['valid_candidate_descriptors'] == 20776
    assert verification['all_invalid_rows'] == receipt['original_allinvalid_rows'] == 1
    assert receipt['normalization']['count'] == verification['normalization']['valid_candidates'] == 5194
    assert receipt['normalization']['array_sha'] == verification['normalization']['array_sha']
    assert protocol['output_dim'] == 385 and protocol['tokens']['shape'] == [384, 56, 42]
    assert protocol['tokens']['dtype'] == 'float16' and protocol['tokens']['inference_dtype'] == 'float32'
    assert not protocol['old_cache_reuse'] and protocol['train_only']
    assert all(receipt['precision'][k] is False for k in ('matmul_allow_tf32', 'cudnn_allow_tf32', 'cudnn_benchmark'))
    for k in ('source_label_values_read', 'source_VAL_features_extracted', 'real_features_extracted',
              'stable_joint_improvement_achieved'):
        assert receipt[k] is False, k
    assert receipt['fits_executed'] == receipt['new_PnP_solves'] == 0
    for k in ('new_fits', 'optimizer_steps', 'policy_selections', 'new_PnP_calls', 'new_image_forwards',
              'image_files_read', 'weight_files_read', 'target_values_read', 'VAL_quality_reads',
              'real_reference_reads', 'real_routes'):
        assert verification[k] == 0, k
    for k in ('method_success', 'goal_complete', 'performance_improvement_measured',
              'backbone_forward_recomputed', 'backbone_inference_correctness_independently_verified'):
        assert verification[k] is False, k
    expected_core = dict(frames=2598, image_forwards=2597, original_allinvalid_rows=1,
        valid_candidates_total=20776, valid_candidates_per_model=5194, descriptor_dim=385,
        normalization_count=5194, new_fits=0, new_PnP_solves=0, new_T_R_evaluations=0,
        source_VAL_features_extracted=False, real_features_extracted=False,
        stable_joint_improvement_achieved=False)
    assert report['core'] == expected_core
    assert report['code'] == bind(HERE / 'report.py')
    assert report['report'] == bind(DOC / 'REPORT_KO.md')
    for key in ('new_fits', 'new_PnP_solves', 'report_image_forwards', 'new_T_R_evaluations', 'policy_selections'):
        assert report[key] == 0, key
    for key in ('method_success', 'goal_complete', 'performance_improvement_measured',
                'source_VAL_features_extracted', 'real_features_extracted', 'report_reads_retained_tokens',
                'report_reads_model_weights', 'report_reads_target_values'):
        assert report[key] is False, key
    assert report['source_TRAIN_only'] and report['original_candidate_validity_preserved']
    assert report['current_descriptor_ready_for_training'] is False
    assert report['padding_contract_correction_before_fit'] is True
    assert report['synthetic_RGB_images'] == report['actual_RGB_images'] == 6
    old = bound_read(report['inputs']['previous_Q_source_gate'])
    assert old['complete'] and old['checks_total'] == 45 and old['checks_passed'] == 43
    assert old['PASS'] is False and old['real_routing_authorized'] is False
    assert report['previous_Q_source_gate'] == dict(checks_total=45, checks_passed=43, PASS=False, unchanged=True)
    assert report['private_reproduction'] == dict(tokens=receipt['tokens'], descriptors=receipt['descriptors'], public_copy=False)
    expected_hist, differences = {}, {}
    for m in MODELS:
        a, v = receipt['models'][m], verification['models'][m]
        assert a['valid_candidates'] == v['comparison']['checked_candidates'] == 5194
        assert a['supported_corner_histogram'] == v['support_histogram']
        assert set(v['support_histogram']) == {str(i) for i in range(9)}
        assert sum(v['support_histogram'].values()) == 5194
        for key in ('descriptor_sha', 'valid_sha', 'support_sha'):
            assert a[key] == v[key]
        assert 0 <= v['comparison']['max_tolerance_fraction'] <= 1
        stats = v['input_statistics']
        assert not stats['performance_metric'] and not stats['targets_used'] and stats['policy_selections'] == 0
        expected_hist[m] = v['support_histogram']
        differences[m] = stats
    assert report['support_histograms'] == expected_hist
    assert report['descriptor_differences'] == differences
    assert report['cache_coverage'] == verify_cache(cache)
    assert len(receipt['frame_receipts']) == 2598
    assert sum(r['image_forward'] for r in receipt['frame_receipts']) == 2597
    for b in report['inputs'].values():
        assert bind(ROOT / b['path']) == b, b['path']
    for forbidden in ('TRAINING_COMPLETE.json', 'SOURCE_VAL_GATE.json', 'REAL_ROUTING_LOCK.json', 'REAL_RESULTS.json'):
        assert not (DOC / forbidden).exists(), ('UNEXPECTED_METHOD_EXECUTION', forbidden)
    assert not (RAW / 'fits').exists()
    text = (DOC / 'REPORT_KO.md').read_text()
    for phrase in ('새 모델 학습 0회', '새 T/R 성능 평가 0회', '실사 실행 0회',
                   '43/45 조건 통과, 전체 FAIL', '반사 padding', 'descriptor·지원 mask·정규화는 수정하지 않았다',
                   'backbone 순전파와 RGB 전처리 전체를 독립적으로 다시 실행한 검산은 아니다'):
        assert phrase in text, ('REQUIRED_SCOPE_DISCLOSURE', phrase)
    assert f"{receipt['tokens']['bytes']:,} bytes" in text
    return dict(core=expected_core, support_histograms=expected_hist, cache_coverage=report['cache_coverage'])


def verify_gallery(report, receipt, protocol):
    """Fixed first-six input pictures and old R0 anchor; no reference errors."""
    import cv2
    metadata = bound_read(protocol['inputs']['metadata'])
    poses = bound_read(protocol['inputs']['poses'])
    descriptor = ROOT / receipt['descriptors']['path']
    assert bind(descriptor) == receipt['descriptors']
    with np.load(descriptor, allow_pickle=False) as z:
        ids = z['ids'].tolist()
        indices = z['source_index'].copy()
        anchors = z['anchor_index'].copy()
        matrices = z['crop_matrices'].copy()
        valid = z['R0_valid'].copy()
        support = z['R0_support8'].copy()
    rows = report['gallery']
    assert len(rows) == 6 and [r['id'] for r in rows] == ids[:6]
    assert poses['ids'] == [r['id'] for r in metadata]
    max_error = 0.
    for i, r in enumerate(rows):
        source = metadata[int(indices[i])]
        assert source['id'] == r['id'] and source['split'] == 'TRAIN'
        assert r['source_index'] == int(indices[i]) and r['image'] == source['image']
        assert r['K'] == source['K'] and r['image_hw'] == source['hw']
        assert r['dimensions_m'] == source['dims']
        np.testing.assert_array_equal(r['dims_cm'], np.asarray(source['dims'], np.float64) * 100)
        assert r['anchor_index'] == int(anchors[i]) and r['anchor_name'] == HYP[int(anchors[i])]
        assert valid[i, anchors[i]]
        hypotheses = poses['records']['R0'][r['id']]['hypotheses']
        h = next(x for x in hypotheses if x['name'] == r['anchor_name'])
        assert r['anchor_pose'] == h['pose'] and h['pose']['available']
        expected = scalar_projection(h['pose'], source['K'])
        np.testing.assert_allclose(r['projected8'], expected, atol=1e-9, rtol=0)
        max_error = max(max_error, float(np.max(np.abs(np.asarray(r['projected8']) - expected))))
        np.testing.assert_array_equal(r['crop_matrix'], matrices[i])
        np.testing.assert_array_equal(r['support8'], support[i, anchors[i]])
        assert r['pad'] == source['pad']
        expected_bounds = [source['pad'], source['pad'], source['hw'][1] - source['pad'], source['hw'][0] - source['pad']]
        np.testing.assert_array_equal(r['original_image_bounds_xyxy'], expected_bounds)
        native = ((expected[:, 0] >= expected_bounds[0]) & (expected[:, 0] < expected_bounds[2]) &
                  (expected[:, 1] >= expected_bounds[1]) & (expected[:, 1] < expected_bounds[3]))
        np.testing.assert_array_equal(r['padding_supported8'], support[i, anchors[i]] & ~native)
        signs = [[-1,-1,-1],[1,-1,-1],[1,1,-1],[-1,1,-1],[-1,-1,1],[1,-1,1],[1,1,1],[-1,1,1]]
        edges = [[0,1],[1,2],[2,3],[3,0],[4,5],[5,6],[6,7],[7,4],[0,4],[1,5],[2,6],[3,7]]
        assert r['corner_signs'] == signs and r['edges'] == edges
        assert r['additional_padding'] == 0 and r['source'] == 'synthetic TRAIN RGB'
        assert r['GT_overlay'] is False and r['new_pose_estimate'] is False
        image_path = ROOT / r['image']['path']
        actual_image = bind(image_path)
        # The frozen metadata's logical dataset path can be a symlink to a
        # stage_a/legacy image. Logical identity was checked above against the
        # exact metadata binding; verify the referenced bytes independently.
        assert actual_image['sha256'] == r['image']['sha256']
        if 'bytes' in r['image']:
            assert actual_image['bytes'] == r['image']['bytes']
        im = cv2.imdecode(np.frombuffer(image_path.read_bytes(), np.uint8), cv2.IMREAD_COLOR)
        assert im is not None and list(im.shape[:2]) == source['hw']
    return dict(images=6, first_six_ids=ids[:6], geometry_reference_used=False,
                independent_scalar_projection_max_abs_px=max_error, dimensions_m_to_cm_verified=True,
                original_prepared_RGB_shape_verified=True, new_image_model_forwards=0)


def verify_padding(report, receipt, protocol):
    padding = read(DOC / 'PADDING_SUPPORT_AUDIT.json')
    assert padding['complete'] and padding['PASS']
    assert padding['code'] == bind(HERE / 'padding_audit.py')
    assert padding['projection_operator'] == bind(HERE / 'verify_train_inputs.py')
    assert padding['protocol'] == bind(DOC / 'INPUT_PROTOCOL.json')
    assert padding['descriptors'] == receipt['descriptors']
    assert padding['metadata'] == protocol['inputs']['metadata'] and padding['poses'] == protocol['inputs']['poses']
    assert padding['frames'] == 2598 and padding['original_valid_candidates'] == 20776
    assert padding['padding_values'] == [100]
    for key in ('original_support_mask_changed', 'descriptor_values_changed', 'normalization_changed', 'performance_improvement_measured'):
        assert padding[key] is False
    for key in ('new_fits', 'new_forwards', 'new_pose_solves', 'target_values_read'):
        assert padding[key] == 0
    expected = [(41282, 3618, 2002, 1049), (41512, 3390, 1890, 983),
                (41505, 3418, 1893, 986), (41513, 3395, 1882, 979)]
    for m, counts in zip(MODELS, expected):
        d = padding['models'][m]
        assert d['valid_candidates'] == 5194
        assert tuple(d[k] for k in ('supported_points', 'padding_supported_points',
                                   'candidates_with_padding_support', 'frames_with_padding_support')) == counts
        assert d['supported_points'] == sum(int(k) * v for k, v in report['support_histograms'][m].items())
        assert d['frames_with_padding_support'] <= d['candidates_with_padding_support'] <= 2*d['frames_with_padding_support']
        assert d['candidates_with_padding_support'] <= d['padding_supported_points'] <= 8*d['candidates_with_padding_support']
    assert report['padding_support_audit'] == bind(DOC / 'PADDING_SUPPORT_AUDIT.json')
    assert report['padding_support_summary'] == padding['models']
    return dict(models=padding['models'], gallery_original_bounds_checked=True,
                full_projection_recalculated=False, sealed_support_unchanged=True)


def verify_tables(report, verification):
    """Exact numeric Markdown rows, independently formatted from receipts."""
    tables, current = [], []
    for line in (DOC / 'REPORT_KO.md').read_text().splitlines() + ['']:
        if line.startswith('|'):
            current.append([x.strip() for x in line.strip().strip('|').split('|')])
        elif current:
            tables.append(current); current = []
    expected = {}
    expected['현재 집합'] = [[label, f"{report['cache_coverage'][key]['rows']:,}",
        f"{report['cache_coverage'][key]['wide_image_identity_matches']:,}", '0']
        for key, label in [('all5120','고정 source 전체'), ('eligibleTRAIN2598','적격 TRAIN'), ('VAL1024','source VAL')]]
    expected['Expert|유효 후보'] = []
    expected['Expert|지원점 전체'] = []
    expected['Expert|서로 다른 W/D 가설 입력 행 / 2,597'] = []
    for m in MODELS:
        h = report['support_histograms'][m]
        expected['Expert|유효 후보'].append([m, '5,194', str(sum(h[str(k)] for k in range(6))), h['6'], h['7'], f"{h['8']:,}"])
        p = report['padding_support_summary'][m]
        expected['Expert|지원점 전체'].append([m, *[f'{p[k]:,}' for k in (
            'supported_points', 'padding_supported_points', 'candidates_with_padding_support')]])
        s = verification['models'][m]['input_statistics']
        expected['Expert|서로 다른 W/D 가설 입력 행 / 2,597'].append([m,
            f"{s['hypothesis_descriptor_different_rows']:,}", f"{s['hypothesis_raw_L2']['median']:.9g}",
            f"{s['hypothesis_normalized_L2']['median']:.9g}", f"{verification['models'][m]['comparison']['max_absolute']:.9g}"])
    expected['TRAIN ID'] = [[r['id'], str(r['source_index']), *[f'{x:.2f}' for x in r['dims_cm']]]
                            for r in report['gallery']]
    checked, unknown = 0, []
    for table in tables:
        assert len(table) >= 2 and all(re.fullmatch(r':?-+:?', x) for x in table[1])
        key = table[0][0]
        if key == 'Expert':
            key += '|' + table[0][1]
        if key not in expected:
            unknown.append(table); continue
        wanted = [[str(x) for x in row] for row in expected.pop(key)]
        assert table[2:] == wanted, ('MARKDOWN_NUMERIC_TABLE', key, table[2:], wanted)
        checked += len(wanted)
    assert not expected, ('MISSING_TABLE', list(expected))
    assert not unknown, ('UNVERIFIED_NUMERIC_TABLE', unknown)
    assert checked == 21
    return dict(verified_numeric_rows=checked, tables=len(tables), remaining_tables=unknown)


def verify_figures(report):
    from PIL import Image
    figures = report['figures']
    assert len(figures) == 4
    assert sum(Path(b['path']).suffix == '.png' for b in figures) == 1
    assert sum(Path(b['path']).suffix == '.jpg' for b in figures) == 3
    out = []
    for b in figures:
        p = ROOT / b['path']
        assert p.parent == DOC / 'figures' and bind(p) == b
        with Image.open(p) as im:
            shape = list(im.size)
            assert min(shape) > 200
            im.verify()
        out.append(dict(**b, width_height=shape))
    return out


def selfcheck():
    pose = dict(cf_extents=[2., 4., 6.], R_cf=np.eye(3).tolist(), centroid=[.2, -.4, 10.])
    K = [[300., 3., 120.], [0., 310., 160.], [0., 0., 1.]]
    q = scalar_projection(pose, K)
    expected_first = [(300*(-.8)+3*(-2.4)+120*7)/7, (310*(-2.4)+160*7)/7]
    np.testing.assert_allclose(q[0], expected_first, rtol=0, atol=1e-13)
    assert q.shape == (8, 2) and np.isfinite(q).all()
    return dict(PASS=True, invented_only=True, actual_inputs_read=0, actual_images_read=0,
                model_forwards=0, fits=0, physical_error_evaluations=0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--selfcheck', action='store_true')
    parser.add_argument('--visual-reviewed', action='store_true')
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--write', action='store_true', help='Explicit alias for the default final receipt write.')
    args = parser.parse_args()
    if args.selfcheck:
        print(json.dumps(selfcheck(), ensure_ascii=False)); return
    assert not (args.write and args.dry_run)
    assert args.visual_reviewed, 'Root must inspect all four actual figures before attestation.'
    assert not (DOC / 'PUBLIC_REVIEW.json').exists() and not (DOC / 'PUBLIC_REVIEW_KO.md').exists()
    guard()
    report = read(DOC / 'REPORT_DATA.json')
    receipt = read(DOC / 'TRAIN_APPEARANCE_INPUTS.json')
    verification = read(DOC / 'INPUT_VERIFICATION.json')
    cache = read(DOC / 'CACHE_CONTRACT.json')
    protocol = read(DOC / 'INPUT_PROTOCOL.json')
    pure = read(DOC / 'PREPROCESSING_PURE_CHECK.json')
    assert pure['complete'] and pure['PASS'] and pure['actual_backbone_forwards'] == pure['weights_loaded'] == 0
    assert pure['extractor_file_sha256_at_check'] == bind(HERE / 'freeze_train.py')['sha256']
    assert (DOC / 'PREPROCESSING_PURE_CHECK.json').read_bytes() == (RAW / 'PREPROCESSING_PURE_CHECK.json').read_bytes()
    assert (DOC / 'CACHE_CONTRACT.json').read_bytes() == (RAW / 'cache_contract.json').read_bytes()
    for function in pure['functions']:
        tree = ast.parse((ROOT / function['module_path']).read_text())
        node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == function['function'])
        assert hashlib.sha256(ast.dump(node, include_attributes=False).encode()).hexdigest() == function['ast_sha256']
    metrics = verify_report(report, receipt, verification, cache, protocol)
    gallery = verify_gallery(report, receipt, protocol)
    padding = verify_padding(report, receipt, protocol)
    tables = verify_tables(report, verification)
    figures = verify_figures(report)
    paths = public_paths()
    links = check_links(paths)
    review = dict(schema='pallet_pose_dino_input_public_review_v1', complete=True, PASS=True,
        stable_joint_improvement_achieved=False, method_success=False, goal_complete=False,
        verification_scope='Published input-audit quantities, cached whole-pose illustrations and file provenance; no T/R performance evaluation',
        report=bind(DOC / 'REPORT_DATA.json'), input_verification=bind(DOC / 'INPUT_VERIFICATION.json'),
        input_receipt=bind(DOC / 'TRAIN_APPEARANCE_INPUTS.json'), cache_contract=bind(DOC / 'CACHE_CONTRACT.json'),
        preprocessing=bind(DOC / 'PREPROCESSING_PURE_CHECK.json'), checks=metrics, gallery=gallery, figures=figures,
        padding_support=padding, markdown_numeric_tables=tables,
        root_visual_review_completed=True, markdown_links_checked=links,
        new_fits=0, new_backbone_forwards=0, new_PnP_solves=0, new_candidate_selections=0,
        new_T_R_evaluations=0, source_VAL_quality_reads=0, real_reference_reads=0,
        execution_history=[dict(session=13940, exit_code=1, receipt_written=False,
            failed_verifier_sha256='75a8399b24137e2779c808eec53f904760aba074ad49ffb67d96642fc8fa0fbd',
            reason='Image bindings contain path and sha256 with optional bytes; exact dictionary equality wrongly required a bytes key. Actual image SHA matched. Verifier now checks bytes when present; producer/report/images unchanged.'),
            dict(session=38909, exit_code=1, receipt_written=False,
            failed_verifier_sha256='1183d3c7629b45cf5a2c29ec615e82f38fb8730d1fb4b1b395fa3a6610aabee3',
            reason='Logical merged-dataset image bindings point through symlinks to stage_a/legacy files. Resolved path equality was incorrect; all six SHA values matched. Exact logical metadata identity and referenced-byte SHA checks are retained; producer/report/images unchanged.')],
        reviewed_artifacts=[bind(p) for p in paths],
        limitations='Independent descriptor verifier reuses retained tokens and does not rerun backbone; this publication check is not a learned method success.')
    if args.dry_run:
        print('PUBLIC_REVIEW_DRY_RUN_PASS', len(paths), len(figures), links); return
    save(DOC / 'PUBLIC_REVIEW.json', review)
    save(DOC / 'PUBLIC_REVIEW_KO.md', '# 입력 감사 공개 기록 검산\n\n'
         '공개 기록 검산 PASS입니다. TRAIN 2,598행, 실제 영상 특징 추출 2,597회, 유효 후보 20,776개와 '
         '캐시 입력 일치 710/376/140 및 정확 재사용 0을 대조했습니다. 원래 무효 1행을 유지했습니다.\n\n'
         '지원 코너 분포·입력 차이 통계, 첫 TRAIN RGB 6장의 치수와 기존 R0 anchor 투영, 그림 4개 및 상대 링크를 확인했습니다. '
         '이번 공개 검산의 학습·backbone forward·T/R 평가·VAL quality·실사 참조 조회는 모두 0입니다. '
         '저장된 token 이후의 독립 입력 검산과 backbone 추론 자체의 독립 재현은 구분합니다. 전체 안정적 개선 목표는 미달성입니다.\n\n'
         '[검산 JSON](PUBLIC_REVIEW.json) · [상세 보고서](REPORT_KO.md)\n')
    print('PUBLIC_REVIEW_PASS', len(paths), len(figures), links, bind(DOC / 'PUBLIC_REVIEW.json'))


if __name__ == '__main__':
    main()
