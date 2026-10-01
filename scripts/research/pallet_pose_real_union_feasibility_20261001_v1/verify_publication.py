"""Independently verify public cached diagnostics; never fit or route a model.

Run only after the root has generated the report. --freeze writes the two
review receipts once; their manifest-link status is never updated afterwards.
"""
import argparse
import ast
from collections import Counter
import csv
from datetime import datetime, timezone
import hashlib
import itertools
import json
import os
from pathlib import Path
import re
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
DOC = ROOT / '_docs/experiments' / HERE.name
BASE = Path('/tmp/pallet-pose-github-review-20260930')
DIAGNOSTICS = ('AXIS_LOWER_BOUND', 'FIXED_TRAIN_COST_ORACLE')
POPS = ('NATURAL99', 'CLEAN29', 'WOOD45')
KEYS = ('T_median_cm', 'R_median_deg', 'T_P90_cm', 'R_P90_deg')
FIGURES = ('natural_candidate_feasibility.png', 'diagnostic_gates.png',
           'real_rgb_dimensions_1.jpg', 'real_rgb_dimensions_2.jpg', 'real_rgb_dimensions_3.jpg')


def read(path):
    return json.loads(Path(path).read_text())


def bind(path):
    path = Path(path).resolve()
    return dict(path=str(path.relative_to(ROOT)), sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                bytes=path.stat().st_size)


def verify(binding):
    actual = bind(ROOT / binding['path'])
    assert actual['sha256'] == binding['sha256'], binding['path']
    assert 'bytes' not in binding or actual['bytes'] == binding['bytes']


def install_guard():
    allowed = {DOC / 'PUBLIC_REVIEW.json', DOC / 'PUBLIC_REVIEW_KO.md'}

    def hook(event, args):
        if event != 'open' or not isinstance(args[0], (str, bytes, os.PathLike)):
            return
        path = Path(os.fsdecode(args[0])).resolve()
        if not path.is_relative_to(ROOT):
            return
        mode = args[1]
        if isinstance(mode, str) and any(k in mode for k in 'wax+'):
            assert path in allowed, ('PUBLIC_REVIEW_WRITE_SCOPE', str(path))
            return
        assert not any(token in str(path) for token in ('/annotations/', '/real_gt_v2/',
            'GEOMETRY_RESOLVED_POSE_GT', 'AXIS_REVIEW_MANIFEST', 'TRUTH_FOR_DISPLAY',
            'GEOMETRY_SIDETABLE', 'SOURCE_TRAIN_LABELS', 'SOURCE_VAL_METRICS')), str(path)
        assert path.suffix not in ('.pt', '.pth', '.onnx'), str(path)

    sys.addaudithook(hook)


def quantile(values, q):
    """Independent interpolation of already published finite scalar errors."""
    values = sorted(float(v) for v in values)
    position = (len(values) - 1) * q
    lower, upper = int(np.floor(position)), int(np.ceil(position))
    return values[lower] + (position - lower) * (values[upper] - values[lower])


def statistics(array):
    array = np.asarray(array, np.float64)
    assert array.ndim == 3 and array.shape[-1] == 2 and np.isfinite(array).all()
    return dict(zip(KEYS, [float(np.mean([quantile(seed[:, axis], q) for seed in array]))
        for axis, q in ((0, .5), (1, .5), (0, .9), (1, .9))]))


def gallery_trace(report, metadata, metrics, lookup, groups, candidates, expected):
    """Capture the existing gallery's plot calls without drawing/saving images.

    Compare its projected lines with an independent OpenCV projection and its
    RGB arrays with decoded, SHA-bound source images. No PnP or GT is involved.
    """
    import cv2
    import matplotlib.pyplot as plt
    from PIL import Image
    captured = []

    class Axis:
        def __init__(self):
            self.lines = []
            self.rgb = None

        def imshow(self, value):
            self.rgb = np.asarray(value)

        def plot(self, x, y, **kwargs):
            self.lines.append((np.asarray(x), np.asarray(y), kwargs))

        def set_xlim(self, *value):
            self.xlim = value

        def set_ylim(self, *value):
            self.ylim = value

        def set_title(self, value, **kwargs):
            self.title = value

        def axis(self, value):
            assert value == 'off'

    class Figure:
        def suptitle(self, value, **kwargs):
            assert 'diagnosis only' in value and 'no ground-truth outline' in value

        def savefig(self, path, **kwargs):
            assert Path(path).name == f'real_rgb_dimensions_{len(captured)}.jpg'
            assert kwargs == {'dpi': 125}

    def subplots(rows, columns, **kwargs):
        assert (rows, columns) == (2, 2) and kwargs['figsize'] == (14, 10.5)
        axes = np.array([[Axis() for _ in range(columns)] for _ in range(rows)], object)
        captured.append(axes)
        return Figure(), axes

    original_subplots, original_close = plt.subplots, plt.close
    plt.subplots, plt.close = subplots, lambda figure: None
    try:
        _, details = report.gallery(metadata, metrics, lookup, groups, candidates)
    finally:
        plt.subplots, plt.close = original_subplots, original_close
    assert details == expected and len(captured) == 3
    metadata = {r['id']: r for r in metadata}
    signs = np.asarray(list(itertools.product((-1., 1.), repeat=3)))
    edges = [(a, b) for a in range(8) for b in range(a + 1, 8)
             if sum(signs[a] != signs[b]) == 1]
    lines_checked, projection_gap, source_images = 0, 0., []
    for index, detail in enumerate(details):
        row = metadata[detail['id']]
        verify(row['image'])
        with Image.open(ROOT / row['image']['path']) as image:
            rgb = np.asarray(image.convert('RGB'))
        assert list(rgb.shape[:2]) == row['hw']
        assert detail['dimensions_m'] == row['xyz'] and len(row['xyz']) == 3
        source_images.append(dict(id=row['id'], recording=row['recording'], binding=row['image'],
            pixels=[rgb.shape[1], rgb.shape[0]], dimensions_m=row['xyz'],
            decoded_RGB_sha256=hashlib.sha256(rgb.tobytes()).hexdigest()))
        for column, panel in enumerate(detail['panels']):
            axis = captured[index // 2][index % 2, column]
            assert np.array_equal(axis.rgb, rgb)
            pose = next(h['pose'] for h in candidates[panel['model']][row['id']]['hypotheses']
                        if h['name'] == panel['hypothesis'])
            corners = signs * np.asarray(pose['cf_extents']) / 2
            rotation = np.asarray(pose['R_cf'], np.float64)
            center = np.asarray(pose['centroid'], np.float64)
            vector, _ = cv2.Rodrigues(rotation)
            expected_uv, _ = cv2.projectPoints(corners, vector, center,
                np.asarray(row['K'], np.float64), np.zeros(5, np.float64))
            expected_uv = expected_uv.reshape(8, 2)
            depths = (corners @ rotation.T + center)[:, 2]
            visible_edges = [(a, b) for a, b in edges if depths[a] > 0 and depths[b] > 0]
            assert len(axis.lines) == panel['drawn_edges'] == len(visible_edges)
            for line, (a, b) in zip(axis.lines, visible_edges):
                actual = np.stack(line[:2], axis=1)
                gap = float(np.abs(actual - expected_uv[[a, b]]).max())
                assert gap < 1e-5, (row['id'], panel['model'], gap)
                projection_gap = max(gap, projection_gap)
                assert line[2]['color'] == ('#ef4444' if column == 0 else '#a855f7')
                lines_checked += 1
            assert axis.xlim == (-.5, rgb.shape[1] - .5)
            assert axis.ylim == (rgb.shape[0] - .5, -.5)
            dims = ' x '.join(f'{100 * d:g}' for d in row['xyz'])
            for token in (row['recording'], f"T={panel['T_cm']:.2f} cm",
                          f"R={panel['R_deg']:.2f} deg", dims + ' cm', panel['hypothesis']):
                assert token in axis.title
    return dict(source_images=source_images, rendered_panels=12, line_segments_checked=lines_checked,
        independent_cv2_projection_max_absolute_pixel_difference=projection_gap,
        direct_projection_only=True, new_PnP_calls=0, new_output_images_written=0)


def run(freeze=False, visually_reviewed=()):
    sys.dont_write_bytecode = True
    install_guard()
    from . import report
    from PIL import Image
    protocol = read(DOC / 'PROTOCOL.json')
    verify(read(DOC / 'PROTOCOL_SHA.json'))
    for binding in [*protocol['inputs'].values(), *protocol['codes']]:
        verify(binding)
    results, data = read(DOC / 'RESULTS.json'), read(DOC / 'REPORT_DATA.json')
    independent = read(DOC / 'VERIFICATION.json')
    assert independent['complete'] and independent['PASS'] and independent['frame_rows_checked'] == 1038
    assert independent['all_gate_booleans_exact'] and not independent['method_success']
    for key in ('results', 'frame_rows', 'protocol', 'verifier'):
        verify(independent[key])
    assert results['complete'] and data['complete']
    for output in (results, data):
        assert output['diagnostic_only'] and not output['method_success'] and not output['goal_complete']
    for key in ('protocol', 'results', 'report_code'):
        verify(data[key])
    for binding in [*results['artifacts'], *data['artifacts']]:
        verify(binding)
    inputs = protocol['inputs']
    metadata = read(ROOT / inputs['eval_metadata']['path'])
    groups = read(ROOT / inputs['eval_groups']['path'])
    metrics = read(ROOT / inputs['operational_metrics']['path'])
    candidates = read(ROOT / inputs['stable_pose_candidates']['path'])
    ids = [r['id'] for r in metadata]
    meta = {r['id']: r for r in metadata}
    csv_path = DOC / 'FRAME_RESULTS.csv'
    assert b'\r' not in csv_path.read_bytes()
    with csv_path.open(newline='') as handle:
        rows = list(csv.DictReader(handle))
    lookup = {(r['diagnostic'], int(r['seed']), r['id']): r for r in rows}
    assert len(rows) == len(lookup) == data['CSV_rows'] == results['full_frame_rows'] == 1038
    assert set(lookup) == {(d, s, i) for d in DIAGNOSTICS for s in (1, 2, 3) for i in ids}
    with (ROOT / inputs['candidate_rows']['path']).open(newline='') as handle:
        old_rows = list(csv.DictReader(handle))
    old = {}
    for r in old_rows:
        key = r['model'], r['id'], r['selected_whole_pose']
        pair = (float(r['T_cm']), float(r['R_deg']))
        assert key not in old or old[key] == pair
        old[key] = pair
    csv_error_checks = 0
    for (diagnostic, seed, fid), r in lookup.items():
        assert r['available'] == r['GT_derived_diagnostic_only'] == 'True'
        assert r['recording'] == meta[fid]['recording']
        assert 1 <= int(r['nondominated_union_candidates']) <= int(r['deduplicated_union_candidates']) <= 4
        for axis, column in [('T', 'T_cm'), ('R', 'R_deg')]:
            model, hypothesis = r[f'{axis}_source_model'], r[f'{axis}_source_hypothesis']
            assert model in ('R0', f'DIVERSE251_s{seed}')
            assert float(r[column]) == old[model, fid, hypothesis][axis == 'R']
            csv_error_checks += 1
        if diagnostic == 'FIXED_TRAIN_COST_ORACLE':
            assert r['physical_complete_pose'] == 'True'
            assert r['selected_model'] == r['T_source_model'] == r['R_source_model']
            assert r['selected_hypothesis'] == r['T_source_hypothesis'] == r['R_source_hypothesis']
            assert float(r['fixed_TRAIN_cost']) == max(float(r['T_cm']) / protocol['scale'][0],
                                                     float(r['R_deg']) / protocol['scale'][1])
        else:
            assert r['physical_complete_pose'] == 'False'
            assert r['selected_model'] == r['selected_hypothesis'] == r['fixed_TRAIN_cost'] == ''
    table_checks = 0
    for population, summary in data['summary'].items():
        assert population in POPS
        for name, recorded in summary.items():
            diagnostic = next((d for d in DIAGNOSTICS if name.startswith(d)), None)
            if diagnostic:
                seeds = [int(name[-1])] if name[-3:-1] == '_s' else [1, 2, 3]
                array = [[[float(lookup[diagnostic, seed, fid][key]) for key in ('T_cm', 'R_deg')]
                          for fid in groups[population]] for seed in seeds]
            else:
                names = [f'{name}_s{s}' for s in (1, 2, 3)] if name in ('SINGLE251', 'DIVERSE251') else [name]
                array = [[[metrics[m][fid][key] for key in ('translation_cm', 'rotation_deg')]
                          for fid in groups[population]] for m in names]
            expected = statistics(array)
            for key in KEYS:
                assert abs(expected[key] - recorded[key]) <= 1e-12, (population, name, key)
                table_checks += 1
            assert recorded['frames_per_seed'] == len(groups[population]) and recorded['seeds'] == len(array)
            assert recorded['failures'] == 0
            if diagnostic:
                ds = results['diagnostics'][diagnostic]
                if len(array) == 3:
                    rs = ds['summaries'][population]['DIVERSE251']
                else:
                    rs = ds['per_seed_summaries'][population][f'DIVERSE251_s{seeds[0]}']
                for key, statistic, metric in [('T_median_cm', 'median', 'translation_cm'),
                    ('R_median_deg', 'median', 'rotation_deg'), ('T_P90_cm', 'P90', 'translation_cm'),
                    ('R_P90_deg', 'P90', 'rotation_deg')]:
                    assert abs(recorded[key] - rs['full_population'][statistic][metric]['value']) <= 1e-12
    body = (DOC / 'REPORT_KO.md').read_text()
    labels = {'SINGLE251': 'SINGLE251 (3seed 평균)', 'DIVERSE251': 'DIVERSE251 운영 (3seed 평균)',
              'AXIS_LOWER_BOUND': '축별 하한 (진단, 3seed 평균)',
              'FIXED_TRAIN_COST_ORACLE': '전체 pose oracle (진단, 3seed 평균)'}
    markdown_rows = 0
    for population in POPS:
        section = body.split(f'### {population}\n', 1)[1].split('\n### ', 1)[0]
        for name, values in data['summary'][population].items():
            label = labels.get(name, name.replace('AXIS_LOWER_BOUND_', '축별 하한 ').replace('FIXED_TRAIN_COST_ORACLE_', '전체 pose oracle '))
            line = '| ' + label + ' | ' + ' | '.join(f'{values[k]:.6f}' for k in KEYS) + ' | 0 |'
            assert line in section, (population, name)
            markdown_rows += 1
    for phrase in ('아직 달성하지 못했습니다', '실제 추론 선택기가 아님', '3seed를 합쳐 중앙값 하나로 바꾸지 않았습니다',
                   'R0_ONLY', '독립 장비 실측 정답이나 새 촬영 일반화 검증이 아닙니다', '수동 코너38개/이미지9장'):
        assert phrase in body
    for diagnostic, label in [('AXIS_LOWER_BOUND', '축별 낙관적 하한'),
                              ('FIXED_TRAIN_COST_ORACLE', '고정 TRAIN 비용 전체 pose oracle')]:
        status = 'PASS' if results['diagnostics'][diagnostic]['stability']['PASS'] else 'FAIL'
        assert f'{label}: **{status}**' in body
    oracle = results['diagnostics']['FIXED_TRAIN_COST_ORACLE']
    assert results['diagnostics']['AXIS_LOWER_BOUND']['stability']['PASS']
    assert [k for k, v in oracle['stability']['gates'].items() if not v['PASS']] == ['natural_tails']
    tails = oracle['stability']['gates']['natural_tails']['comparisons']
    failed_scalar_guards = [(reference, criterion) for reference, block in tails.items()
                           for criterion, check in block['checks'].items() if not check['pass_guard']]
    assert failed_scalar_guards == [('R0', 'P90:translation_cm')]
    tail = tails['R0']['checks']['P90:translation_cm']
    for label, key in [('R0', 'before'), ('원래 허용 한계 (R0×1.05)', 'limit'),
                       ('고정 비용 전체 pose oracle', 'after')]:
        assert f"| {label} | {tail[key]['value']:.6f} |" in body
    for baseline in ('SINGLE251', 'R0'):
        ci = oracle['hierarchy']['NATURAL99'][f'DIVERSE251-minus-{baseline}']['hierarchical_bootstrap']['metrics']
        cells = [f"[{ci[m]['CI95'][0]:.6f}, {ci[m]['CI95'][1]:.6f}]" for m in ('translation_cm', 'rotation_deg')]
        assert f'| {baseline} | ' + ' | '.join(cells) + ' |' in body
        assert all(ci[m]['CI95'][1] < 0 for m in ('translation_cm', 'rotation_deg'))
    assert '다른 전체 pose 선택 조합까지 불가능하다고 단정할 수 없습니다' in body
    chosen = [min((fid for fid in groups['NATURAL99'] if meta[fid]['recording'] == recording),
                  key=lambda fid: (-metrics['R0'][fid]['translation_cm'], fid))
              for recording in sorted({meta[i]['recording'] for i in groups['NATURAL99']})]
    assert len(chosen) == len(set(chosen)) == data['actual_RGB_images'] == 6
    assert [v['id'] for v in data['illustrations']] == chosen
    for detail in data['illustrations']:
        fid = detail['id']
        assert detail['oracle_seed'] == 1 and len(detail['panels']) == 2
        first, second = detail['panels']
        assert first['model'] == 'R0' and first['hypothesis'] == candidates['R0'][fid]['GEO_name']
        for key, old_key in [('T_cm', 'translation_cm'), ('R_deg', 'rotation_deg')]:
            assert first[key] == metrics['R0'][fid][old_key]
        row = lookup['FIXED_TRAIN_COST_ORACLE', 1, fid]
        assert second['model'] == row['selected_model'] and second['hypothesis'] == row['selected_hypothesis']
        assert second['T_cm'] == float(row['T_cm']) and second['R_deg'] == float(row['R_deg'])
    projection = gallery_trace(report, metadata, metrics, lookup, groups, candidates, data['illustrations'])
    pictures = []
    assert {Path(b['path']).name for b in data['artifacts']} == set(FIGURES)
    for name in FIGURES:
        path = DOC / 'figures' / name
        with Image.open(path) as image:
            image.verify()
        with Image.open(path) as image:
            size = list(image.size)
        assert size == ([1960, 1260] if name == FIGURES[0] else [1820, 518] if name == FIGURES[1] else [1750, 1312])
        pictures.append(dict(binding=bind(path), pixels=size, decoded=True,
            visually_reviewed=name in visually_reviewed))
    links, pending = [], []
    for path in sorted(DOC.glob('*.md')):
        if path.name.startswith('PUBLIC_REVIEW'):
            continue
        for href in re.findall(r'\]\(([^)]+)\)', path.read_text()):
            if href.startswith(('http://', 'https://', '#')):
                continue
            target = (path.parent / href.split('#')[0]).resolve()
            if target == DOC / 'PUBLICATION_MANIFEST.json':
                pending.append(dict(source=path.name, target=str(target.relative_to(ROOT)),
                    status='Root-generated publication manifest; deliberately excluded from immutable review timing.'))
                continue
            assert target.exists(), (path.name, href)
            rel = target.relative_to(ROOT)
            assert target.is_relative_to(DOC) or target.is_relative_to(HERE) or (BASE / rel).exists(), ('UNPUBLISHED_LINK', str(rel))
            links.append(str(rel))
    scripts = []
    for path in sorted(HERE.glob('*.py')):
        ast.parse(path.read_text())
        scripts.append(bind(path))
    result = dict(complete=True, PASS=True, created_at=datetime.now(timezone.utc).isoformat(),
        scope='Independent public report/CSV/actual-RGB projection parity; no new model, oracle routing, GT scoring or gates.',
        code=bind(Path(__file__)), protocol=bind(DOC / 'PROTOCOL.json'),
        results=bind(DOC / 'RESULTS.json'), report_data=bind(DOC / 'REPORT_DATA.json'),
        report=bind(DOC / 'REPORT_KO.md'), report_code=bind(HERE / 'report.py'),
        independent_numeric_verification=bind(DOC / 'VERIFICATION.json'),
        reviewed_markdown=[bind(p) for p in sorted(DOC.glob('*.md')) if not p.name.startswith('PUBLIC_REVIEW')],
        frame_CSV=bind(csv_path), frame_rows=1038, unique_diagnostic_seed_ID_rows=1038,
        source_cached_error_cells_exact=csv_error_checks, whole_pose_source_pair_parity=True,
        summary_numeric_values_checked=table_checks, markdown_numeric_rows_checked=markdown_rows,
        additional_tail_guard_values_checked=3, bootstrap_CI_endpoint_values_checked=8,
        sole_failed_scalar_guard=failed_scalar_guards,
        declared_diagnostic_verdicts={d:results['diagnostics'][d]['stability']['PASS'] for d in DIAGNOSTICS},
        gallery_selection='One largest operational R0 translation error per natural recording; exact ID ties. Fixed seed1 oracle, no best-seed selection.',
        projection=projection, figures=pictures, markdown_links_checked=len(links),
        pending_publication_manifest_links=pending, broken_links=[], new_python_AST=scripts,
        immutable_after_freeze=True, fits=0, image_forwards=0, new_PnP=0, raw_GT_reads=0,
        new_reference_error_calls=0, new_learned_routing=0, diagnostic_rules_reselected=0,
        method_success=False, goal_complete=False)
    if freeze:
        assert set(visually_reviewed) == set(FIGURES), 'Open all five actual artifact images before final freeze.'
        assert not (DOC / 'PUBLIC_REVIEW.json').exists() and not (DOC / 'PUBLIC_REVIEW_KO.md').exists()
        markdown = f'''# 공개 보고서 독립 검증

**PASS. 공개 숫자와 이미지의 출처가 저장된 진단 결과와 일치한다. 실제 모델의 T/R 개선 성공 판정은 아니다.**

FRAME CSV **1,038개 고유 행**과 기존 후보 오류 **{csv_error_checks:,}개 셀**, 요약 **{table_checks}개 수치**, Markdown **{markdown_rows}개 숫자 행**을 대조했다. 추가된 T P90 제한표3개 값과 bootstrap CI8개 끝점도 저장 결과와 일치한다. 전체 pose oracle의 T와 R는 같은 model·hypothesis에서 왔으며 축별 하한은 물리 pose로 표시하지 않았다. 축별 하한은5조건을 통과하고 전체 pose oracle은 R0 대비 자연 T P90 한 항목만 실패한다는 설명이 맞다.

실제 RGB **6장**의 파일 SHA·디코딩 크기·입력 치수·선택 ID를 확인했다. 각 자연 recording에서 R0 T 오류가 최대인 한 장을 ID 동률 규칙으로 고르고, 오른쪽 oracle은 모두 미리 고정한 seed1이다. 사진 **12개 panel**, 투영 선분 **{projection['line_segments_checked']}개**를 실제 gallery 코드의 저장 없는 plot trace와 독립 OpenCV 투영으로 비교했다. 최대 좌표 차이는 **{projection['independent_cv2_projection_max_absolute_pixel_difference']:.3g}px**다. raw GT, 새 PnP, 모델 추론은 사용하지 않았다.

그래프2개·실제 RGB montage3개인 **5개 출력 이미지**를 모두 열어 확인했다. 크기·SHA는 [PUBLIC_REVIEW.json](PUBLIC_REVIEW.json)에 기록했다. 그래프는 운영 모델과 참조 기반 진단을 구분하며 montage는 정답 외곽선이 아닌 기존 후보 pose의 투영이다.

Markdown 상대 링크 **{len(links)}개**는 현재 새 공개 범위 또는 기존 publication checkout에서 확인했다. root가 마지막에 생성하는 `PUBLICATION_MANIFEST.json` 링크 **{len(pending)}개**는 별도 생성예정 항목이며, manifest 생성 후 이 영수증을 다시 쓰지 않는다. 새 Python 파일은 AST 검사에 통과했다.

이번 검증은 저장 결과의 표시·연결·투영을 확인한 것이다. 재사용 DEV와 2D 주석 기반 참조의 한계, 미실행 learned R0_ONLY 실사 비교, 수동 교사 계보는 보고서에 유지돼 있다. 새 학습·이미지 추론·참조 오차 계산·learned routing은0회이고 전체 목표 성공은 여전히 주장하지 않는다.
'''
        with (DOC / 'PUBLIC_REVIEW_KO.md').open('x') as handle:
            handle.write(markdown)
        result['review_note'] = bind(DOC / 'PUBLIC_REVIEW_KO.md')
        with (DOC / 'PUBLIC_REVIEW.json').open('x') as handle:
            handle.write(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps(dict(PASS=True, frozen=freeze, rows=1038, summary_checks=table_checks,
        markdown_rows=markdown_rows, pictures=5, RGB_images=6, projection=projection,
        pending_manifest_links=len(pending)), ensure_ascii=False))
    if freeze:
        print(json.dumps(dict(receipt=bind(DOC / 'PUBLIC_REVIEW.json'),
            note=bind(DOC / 'PUBLIC_REVIEW_KO.md')), ensure_ascii=False))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--freeze', action='store_true')
    parser.add_argument('--visually-reviewed', nargs='*', default=[])
    arguments = parser.parse_args()
    run(arguments.freeze, arguments.visually_reviewed)
