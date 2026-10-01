"""Independent public export audit against frozen fit, route, and metric files."""
import argparse
import ast
import csv
import json
from pathlib import Path
import re
import subprocess
from urllib.parse import unquote
import numpy as np
from PIL import Image
from . import common as C
from .verify_gallery import verify as verify_gallery


def read_csv(path):
    assert b'\r' not in path.read_bytes()
    with path.open(newline='') as f:
        return list(csv.DictReader(f))


def binding(value):
    C.verify(value)
    return C.read(C.ROOT / value['path'])


def all_bindings(value):
    if isinstance(value, dict):
        if {'path', 'sha256'} <= value.keys():
            C.verify(value)
        for child in value.values():
            all_bindings(child)
    elif isinstance(value, list):
        for child in value:
            all_bindings(child)


def scalar_equal(text, expected):
    if isinstance(expected, (int, float)) and not isinstance(expected, bool):
        return float(text) == expected
    return text == str(expected)


def verify_tables_and_figures(report, markdown, source, real, fits):
    training = C.read(C.DOC / 'TRAIN_CONVERGENCE.json')
    previous = binding(report['previous_source_gate'])
    lines = markdown.splitlines()
    table_count = row_count = 0

    def table(header, expected, occurrence=0):
        nonlocal table_count, row_count
        starts = [i for i, line in enumerate(lines) if line == header]
        assert len(starts) > occurrence, header
        start = starts[occurrence] + 2
        found = []
        while start < len(lines) and lines[start].startswith('|'):
            found.append(lines[start]); start += 1
        assert found == expected, (header, occurrence, found, expected)
        table_count += 1; row_count += len(expected)

    expected = []
    for model in C.MODEL_NAMES:
        info = fits[model]
        gap = training['models'][model]['independent_recompute']['certified_gap_upper_bound']
        expected.append(f"| {model} | {info['iterations']} | {info['objective_calls']} | {gap:.3e} | PASS |")
    table('| 모델 | 반복 | objective 호출 | gap 상한 | 인증 |', expected)
    expected = []
    risk_rows = []
    for model in C.MODEL_NAMES:
        current = training['models'][model]
        before = current['previous_converged_same_target']
        objective = current['independent_recompute']
        expected.append(f"| {model} | {before['objective']['objective']:.9f} → {objective['objective']:.9f} | {before['native_unadjusted_objective']['CE']:.9f} → {objective['unadjusted_CE']:.9f} | {100*before['statistics']['target_accuracy_full_population']:.3f}% → {100*current['statistics']['target_accuracy_full_population']:.3f}% |")
        cells = [f"{before['statistics']['anchor_violations'][axis]['count']} → {current['statistics']['anchor_violations'][axis]['count']}" for axis in ('T', 'R', 'either')]
        cells += [f"{before['risk_statistics']['normalized_max_excess_all_available']['maximum']:.6f} → {current['risk_statistics']['normalized_max_excess_all_available']['maximum']:.6f}"]
        risk_rows.append('| ' + model + ' | ' + ' | '.join(cells) + ' |')
    table('| 모델 | 같은 margin objective 이전→현재 | 일반 CE 이전→현재 | 정답 선택률 이전→현재 |', expected)
    table('| 모델 | T 위반 이전→현재 | R 위반 이전→현재 | 한 축 이상 위반 이전→현재 | 최대 정규화 초과 이전→현재 |', risk_rows)
    for index, summaries in enumerate([source['summaries']] + [real['summaries'][p] for p in ('NATURAL99', 'CLEAN29', 'WOOD45')]):
        expected = []
        for model, summary in summaries.items():
            v = summary['full_population']
            expected.append(f"| {model} | {v['translation_cm']['median']:.6f} | {v['rotation_deg']['median']:.6f} | {v['translation_cm']['P90']:.6f} | {v['rotation_deg']['P90']:.6f} | {summary['failed_pose']} |")
        table('| 모델 | T 중앙값 cm | R 중앙값 ° | T P90 cm | R P90 ° | 실패 |', expected, index)
    expected = []
    for name, gate in real['stability']['gates'].items():
        values = [gate['matched_intervention']['PASS'], gate['original_goal']['PASS'], gate['PASS']]
        expected.append('| ' + name + ' | ' + ' | '.join('PASS' if v else 'FAIL' for v in values) + ' |')
    table('| 판정 범주 | matched 대조 | 원래 SINGLE251 포함 목표 | 최종 AND |', expected)
    expected = []
    for metric, value in real['hierarchy']['NATURAL99']['R0']['hierarchical_bootstrap']['metrics'].items():
        expected.append(f"| {metric} | {value['point_estimate']['value']:.9f} | [{value['CI95'][0]:.9f}, {value['CI95'][1]:.9f}] |")
    table('| R0 대비 자연99 | 평균 seed 중앙값 차이 | recording×seed bootstrap95% CI |', expected)
    values = report['figure_values']
    for metric in ('translation_cm', 'rotation_deg'):
        for quantile in ('median', 'P90'):
            v = values[f'source_{metric}_{quantile}']
            assert v['previous_context189'] == [previous['summaries'][m]['full_population'][metric][quantile] for m in C.MODEL_NAMES]
            assert v['current_margin189'] == [source['summaries'][m]['full_population'][metric][quantile] for m in C.MODEL_NAMES]
            assert v['fixed_R0_GEO'] == source['summaries']['R0_GEO']['full_population'][metric][quantile]
            for pop in ('NATURAL99', 'CLEAN29', 'WOOD45'):
                v = values[f'real_{pop}_{metric}_{quantile}']
                assert v['models'] == real['models']
                assert v['values'] == [real['summaries'][pop][m]['full_population'][metric][quantile] for m in real['models']]
    for kind in ('previous_context189', 'current_margin189'):
        data = [training['models'][m]['previous_converged_same_target'] if kind.startswith('previous') else training['models'][m] for m in C.MODEL_NAMES]
        assert values['train_target_accuracy'][kind] == [d['statistics']['target_accuracy_full_population']*100 for d in data]
        assert values['train_unsafe_count'][kind] == [d['statistics']['anchor_violations']['either']['count'] for d in data]
        assert values['train_max_risk'][kind] == [d['risk_statistics']['normalized_max_excess_all_available']['maximum'] for d in data]
    gates = values['real_gate_matrix']
    assert gates['rows'] == list(real['stability']['gates'])
    assert gates['columns'] == ['matched', 'original_SINGLE251', 'combined_AND']
    assert gates['values'] == [[int(g['matched_intervention']['PASS']), int(g['original_goal']['PASS']), int(g['PASS'])] for g in real['stability']['gates'].values()]
    return dict(markdown_tables=table_count, exact_table_rows=row_count, figure_value_groups=len(values))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--visual-reviewed', action='store_true', required=True)
    args = parser.parse_args()
    assert args.visual_reviewed
    doc = C.DOC
    report = C.read(doc / 'REPORT_DATA.json')
    md = (doc / 'REPORT_KO.md').read_text()
    all_bindings(report)
    for name in ('PREFIT_REVIEW.json', 'TRAIN_CONVERGENCE.json',
                 'SOURCE_VAL_VERIFICATION.json', 'REAL_VERIFICATION.json'):
        item = C.read(doc / name)
        assert item['complete'] and item['PASS'], name
    source = C.read(doc / 'SOURCE_VAL_GATE.json')
    real = C.read(doc / 'REAL_RESULTS.json')
    complete = C.read(doc / 'TRAINING_COMPLETE.json')
    assert source['PASS'] and source['checks_total'] == source['checks_passed'] == 45
    assert real['complete'] and real['full_frame_rows'] == 2249
    assert real['baseline_metric_parity_checks'] == 1557
    assert not real['stability']['PASS'] and not real['stability']['goal_complete']
    assert [g['PASS'] for g in real['stability']['gates'].values()] == [False, False, False, True, True]
    assert real['real_stability_contract'] == 'matched_intervention_AND_original_SINGLE251_stability'
    for name, group in real['stability']['gates'].items():
        assert group['PASS'] == (group['matched_intervention']['PASS'] and group['original_goal']['PASS']), name
    assert complete['fit_count'] == 4 and complete['total_objective_calls'] == 1453
    # Source CSV carries every row, including routing identity and failure state.
    C.verify(source['metrics'])
    choices = binding(C.read(doc / 'SOURCE_VAL_ROUTING_LOCK.json')['choices'])
    source_rows = read_csv(doc / 'SOURCE_VAL_FRAME_RESULTS.csv')
    source_index = {(r['model'], r['id']): r for r in source_rows}
    assert len(source_rows) == len(source_index) == 8192
    with np.load(C.ROOT / source['metrics']['path'], allow_pickle=False) as z:
        for model in z['models'].tolist():
            for i, fid in enumerate(z['ids'].tolist()):
                row = source_index[model, fid]
                assert float(row['T_cm']) == z[model][i, 0]
                assert float(row['R_deg']) == z[model][i, 1]
                assert row['available'] == 'True' and row['split'] == 'VAL'
                pick = choices['records'].get(model, {}).get(fid, {})
                assert row['candidate'] == pick.get('candidate_name', 'fixed_GEO')
                assert row['fallback'] == str(pick.get('fallback', False))
    checks = read_csv(doc / 'SOURCE_VAL_CHECKS.csv')
    expected_checks = [dict(model=m, baseline=b, criterion=k, PASS=v)
        for m, group in source['comparisons'].items() for b, comp in group.items()
        for k, v in comp['checks'].items()]
    assert len(checks) == len(expected_checks) == 45
    for row, expected in zip(checks, expected_checks):
        assert all(row[k] == str(v) for k, v in expected.items())
    # Independently verified real metric export remains byte-bound by evaluator.
    for artifact in real['artifacts'].values() if isinstance(real['artifacts'], dict) else real['artifacts']:
        C.verify(artifact)
    real_rows = read_csv(doc / 'REAL_FRAME_RESULTS.csv')
    assert len(real_rows) == len({(r['model'], r['id']) for r in real_rows}) == 2249
    assert all(r['pose_available'] == 'True' and r['full_population_error_status'] == 'FINITE' for r in real_rows)
    groups = binding(C.read(doc / 'REAL_PROTOCOL.json')['inputs']['groups'])
    for population in ('NATURAL99', 'CLEAN29', 'WOOD45'):
        # Membership comes from fixed population list, never from metric values.
        ids = set(groups[population])
        assert len(ids) == real['populations'][population]
        for model in real['models']:
            subset = [r for r in real_rows if r['model'] == model and r['id'] in ids]
            assert len(subset) == len(ids)
            for metric in ('translation_cm', 'rotation_deg'):
                vals = np.array([float(r[metric]) for r in subset])
                summary = real['summaries'][population][model]['full_population'][metric]
                assert np.isclose(float(np.median(vals)), summary['median'], rtol=0, atol=1e-12)
                assert np.isclose(float(np.quantile(vals, .9)), summary['P90'], rtol=0, atol=1e-12)
    # Parameter JSONs must be the four actually optimized terminal models.
    fits = {}
    trace_records = []
    for model in C.MODEL_NAMES:
        fit = C.read(doc / f'FIT_{model}.json')
        for k in ('START', 'checkpoint', 'trace', 'protocol'):
            C.verify(fit[k])
        ck = binding(fit['checkpoint'])
        assert ck['schema'] == 'pallet_pose_anchor_risk_linear189_v1'
        assert len(ck['weight']) == 189 and ck['runtime_uses_margin'] is False
        exports = [p for p in (doc / 'model_parameters').glob('*.json') if p.read_bytes() == (C.ROOT / fit['checkpoint']['path']).read_bytes()]
        assert len(exports) == 1, (model, exports)
        logs = [json.loads(s) for s in (C.ROOT / fit['trace']['path']).read_text().splitlines()]
        calls = [r for r in logs if r['event'] == 'objective']
        assert [r['call'] for r in calls] == list(range(1, fit['objective_calls'] + 1))
        assert calls[-1]['weight_sha'] == fit['final_weight_sha']
        assert fit['certificate']['PASS']
        trace_records += [dict(model=model, **r) for r in calls]
        fits[model] = dict(checkpoint=C.bind(exports[0]), objective_calls=fit['objective_calls'], iterations=fit['iterations'])
    assert sum(v['iterations'] for v in fits.values()) == 1313
    trace_rows = read_csv(doc / 'TRAINING_OBJECTIVE_LOG.csv')
    assert len(trace_rows) == len(trace_records) == 1453
    for row, raw in zip(trace_rows, trace_records):
        assert {'model', 'call', 'objective', 'CE', 'unadjusted_CE'} <= row.keys()
        for key, value in row.items():
            assert key in raw and scalar_equal(value, raw[key]), (key, value, raw.get(key))
    table_review = verify_tables_and_figures(report, md, source, real, fits)
    images = []
    for path in sorted((doc / 'figures').glob('*')):
        if path.suffix in ('.png', '.jpg'):
            with Image.open(path) as im:
                width, height = im.size
                im.verify()
            assert width > 500 and height > 300
            assert 'figures/' + path.name in md, path
            images.append(dict(binding=C.bind(path), width=width, height=height))
    assert len([v for v in images if v['binding']['path'].endswith('.jpg')]) == 3
    gallery = verify_gallery(report)
    # Every local Markdown link must exist here or already in the publication.
    base = Path('/tmp/pallet-pose-github-review-20260930')
    pending = {'PUBLIC_REVIEW.json', 'PUBLIC_REVIEW_KO.md', 'PUBLICATION_MANIFEST.json'}
    links = 0
    for path in doc.glob('*.md'):
        for url in re.findall(r'\]\(([^)]+)\)', path.read_text()):
            if url.startswith(('http://', 'https://', '#')):
                continue
            target = (path.parent / unquote(url.split('#')[0])).resolve()
            assert target.is_relative_to(C.ROOT), (path, url)
            if target.parent == doc and target.name in pending:
                pass
            elif target.is_relative_to(doc) or target.is_relative_to(C.HERE):
                assert target.exists(), (path, url)
            else:
                assert (base / target.relative_to(C.ROOT)).exists(), (path, url)
            links += 1
    files = [C.ROOT / '.gitignore', C.ROOT / 'readme.md']
    for folder in (doc, C.HERE):
        files += [p for p in folder.rglob('*') if p.is_file() and '__pycache__' not in p.parts
                  and p.name not in ('PUBLIC_REVIEW.json', 'PUBLIC_REVIEW_KO.md', 'PUBLICATION_MANIFEST.json')]
    for p in files:
        if p.suffix == '.py':
            ast.parse(p.read_text(), filename=str(p))
    review = dict(complete=True, PASS=True, created_at=C.now(), code=C.bind(Path(__file__)),
        source_csv_rows=len(source_rows), source_checks=45, real_csv_rows=len(real_rows),
        training_objective_rows=len(trace_rows), model_parameters=fits,
        actual_RGB_frames=6, visual_review_completed=True, images=images, gallery=gallery,
        markdown_links_checked=links, table_review=table_review, source_gate_PASS=True, real_stability_PASS=False,
        real_stability_checks_passed=2, goal_complete=False,
        reviewed_artifacts=[C.bind(p) for p in sorted(set(files))],
        raw_reference_metric_recalculation=0, new_fits=0,
        limits='Independent TRAIN/source/real verification supplies numeric correctness; this audit checks public exports, original/combined verdict, parameters, links and image presentation.')
    C.save(doc / 'PUBLIC_REVIEW.json', review)
    C.save(doc / 'PUBLIC_REVIEW_KO.md', '# 공개 결과 검산\n\n공개 수치·파라미터·링크·이미지 검산 PASS입니다. 방법의 실사 성공을 뜻하지 않습니다.\n\n'
        '- 합성 전체 8,192행, 조건 45개, 실사 전체 2,249행을 보존했습니다.\n'
        '- 최종 모델 4개의 공개 파라미터가 실제 학습 종료 파일과 동일합니다.\n'
        '- 학습 목적함수 기록 1,453행과 실제 반복 1,313회를 확인했습니다.\n'
        '- 실제 RGB 6장의 R0 및 세 seed 투영 이미지와 치수를 직접 검토했습니다.\n'
        '- 합성 45/45 통과, 실사 2/5 통과·안정적 동시 개선 미달성 판정을 유지했습니다.\n\n'
        '[검산 상세와 파일 해시](PUBLIC_REVIEW.json) · [실사 독립 검산](REAL_VERIFICATION_KO.md) · [결과 보고서](REPORT_KO.md)\n')
    print('PUBLIC_REVIEW_PASS', len(files), len(images), links)


if __name__ == '__main__':
    main()
