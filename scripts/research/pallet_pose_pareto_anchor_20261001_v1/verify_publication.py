"""Verify published exports, not new model performance or new references.

Run with --freeze only after the report and independent numeric reviews are final.
No metric helper, source label array, optimizer, or inference module is imported.
"""
import argparse
import csv
import itertools
import json
from pathlib import Path
import re
import subprocess
from urllib.parse import unquote

import numpy as np
from PIL import Image
from . import common as C


def csv_rows(path):
    data = Path(path).read_bytes()
    if Path(path).resolve().is_relative_to(C.DOC):
        assert b'\r' not in data, path
    with Path(path).open(newline='') as f:
        return list(csv.DictReader(f))


def value_from(binding):
    C.verify(binding)
    return C.read(C.ROOT / binding['path'])


def equal_csv(rows, expected):
    assert len(rows) == len(expected)
    cells = 0
    for row, ref in zip(rows, expected):
        assert row.keys() == ref.keys()
        for key, val in ref.items():
            if isinstance(val, (int, float)) and not isinstance(val, bool):
                assert float(row[key]) == val, (key, row[key], val)
            else:
                assert row[key] == str(val), (key, row[key], val)
            cells += 1
    return cells


def links(review_text, base):
    checked, pending = [], []
    sources = {p: p.read_text() for p in C.DOC.glob('*.md')}
    sources[C.DOC / 'PUBLIC_REVIEW_KO.md'] = review_text
    manifest = C.DOC / 'PUBLICATION_MANIFEST.json'
    for source, content in sources.items():
        for link in re.findall(r'!?\[[^\]]*\]\(([^)]+)\)', content):
            link = unquote(link.strip().strip('<>'))
            if re.match(r'https?://|mailto:|#', link):
                continue
            target = (source.parent / link.split('#', 1)[0]).resolve()
            assert target.is_relative_to(C.ROOT), (source, link)
            if target == manifest and not target.exists():
                pending.append(str(target.relative_to(C.ROOT)))
            elif target.is_relative_to(C.DOC) or target.is_relative_to(C.HERE):
                assert target.exists() or target == C.DOC / 'PUBLIC_REVIEW_KO.md', (source, link)
            else:
                # Publication checkout, not unrelated unpublished workspace files.
                assert (base / target.relative_to(C.ROOT)).exists(), (source, link)
            checked.append(dict(source=str(source.relative_to(C.ROOT)), target=link))
    assert set(pending) <= {str(manifest.relative_to(C.ROOT))}
    return dict(checked=checked, checked_count=len(checked), allowed_pending_unique=sorted(set(pending)))


def verify(base):
    report = C.read(C.DOC / 'REPORT_DATA.json')
    md = (C.DOC / 'REPORT_KO.md').read_text()
    for key in ('code', 'real_protocol', 'train_protocol', 'real_feasibility', 'source_feasibility',
                'source_gate', 'training_verification', 'real_not_run'):
        C.verify(report[key])
    rp = value_from(report['real_protocol'])
    sg = value_from(report['source_gate'])
    real = value_from(report['real_feasibility'])
    train = value_from(report['training_verification'])
    sf = value_from(report['source_feasibility'])
    nr = value_from(report['real_not_run'])
    sv = C.read(C.DOC / 'SOURCE_VAL_VERIFICATION.json')
    rv = C.read(C.DOC / 'REAL_VERIFICATION.json')
    assert sv['PASS'] and rv['PASS'] and train['PASS'] and sf['PASS']
    for obj, key in ((sv, 'source_gate'), (rv, 'results'), (rv, 'rows')):
        C.verify(obj[key])
    assert sv['source_gate'] == report['source_gate']
    assert rv['results'] == report['real_feasibility']
    assert sg['complete'] and not sg['PASS'] and sg['checks_passed'] == 38 and sg['checks_total'] == 45
    assert sv['checks_passed'] == 38 and sv['checks_total'] == 45 and sv['all_gate_booleans_exact']
    assert real['PASS'] and rv['diagnostic_gate_PASS'] and not real['method_success'] and not real['goal_complete']
    assert not report['learned_real_evaluated'] and not report['stable_joint_improvement_achieved'] and not report['goal_complete']
    for key in ('learned_real_routes', 'real_metric_calls', 'raw_real_reference_reads', 'image_forwards', 'new_fits_by_real_evaluator'):
        assert nr[key] == 0
    for path, absent in nr['absence_checks'].items():
        assert absent and not (C.ROOT / path).exists(), path

    # Every exported source metric, candidate identity, and fallback flag.
    C.verify(sg['metrics'])
    choices = value_from(sv['choices'])
    with np.load(C.ROOT / sg['metrics']['path'], allow_pickle=False) as z:
        ids, model_names = z['ids'].tolist(), z['models'].tolist()
        errors = {m: z[m].copy() for m in model_names}
    expected = []
    for model in model_names:
        assert errors[model].shape == (1024, 2) and np.isfinite(errors[model]).all()
        for j, fid in enumerate(ids):
            pick = choices['records'].get(model, {}).get(fid, {})
            expected.append(dict(model=model, id=fid, split='VAL', available=True,
                                 T_cm=float(errors[model][j, 0]), R_deg=float(errors[model][j, 1]),
                                 candidate=pick.get('candidate_name', 'fixed_GEO'), fallback=pick.get('fallback', False)))
    source_cells = equal_csv(csv_rows(C.DOC / 'SOURCE_VAL_FRAME_RESULTS.csv'), expected)
    assert len(expected) == 8192
    checks = [dict(model=m, baseline=b, criterion=k, PASS=v) for m, cc in sg['comparisons'].items()
              for b, group in cc.items() for k, v in group['checks'].items()]
    check_cells = equal_csv(csv_rows(C.DOC / 'SOURCE_VAL_CHECKS.csv'), checks)
    assert len(checks) == 45 and sum(c['PASS'] for c in checks) == 38
    failures = ['/'.join((r['model'], r['baseline'], r['criterion'])) for r in checks if not r['PASS']]
    assert failures == sg['failed_checks'] == nr['failed_checks'] == sv['failed_checks']
    assert sum(s.endswith('rotation_deg_P90_guard') for s in failures) == 6
    assert sum(s.endswith('rotation_deg_median_strict') for s in failures) == 1

    # Every objective-log field and exact published checkpoint bytes.
    traces, fits = [], {}
    for model in C.MODEL_NAMES:
        fit = C.read(C.DOC / f'FIT_{model}.json')
        for key in ('START', 'trace', 'checkpoint', 'protocol'):
            C.verify(fit[key])
        assert fit['protocol'] == report['train_protocol'] and fit['fits_executed'] == 1
        log = [json.loads(line) for line in (C.ROOT / fit['trace']['path']).read_text().splitlines()]
        obj = [r for r in log if r['event'] == 'objective']
        iterations = [r for r in log if r['event'] == 'iteration']
        assert [r['call'] for r in obj] == list(range(1, fit['objective_calls'] + 1))
        assert [r['iteration'] for r in iterations] == list(range(1, fit['iterations'] + 1))
        assert fit['certificate']['PASS'] and fit['certificate']['optimizer_success']
        assert fit['objective_calls'] <= 2000 and fit['iterations'] <= 1000
        assert obj[-1]['objective'] == fit['certificate']['objective_value']
        assert obj[-1]['weight_sha'] == fit['final_weight_sha']
        assert train['models'][model]['objective_calls'] == fit['objective_calls']
        assert train['models'][model]['iterations'] == fit['iterations']
        export = report['exports'][model]
        assert export['local'] == fit['checkpoint']
        C.verify(export['published'])
        assert (C.ROOT / export['local']['path']).read_bytes() == (C.ROOT / export['published']['path']).read_bytes()
        ck = value_from(fit['checkpoint'])
        assert ck['target_rule'] == fit['target_rule'] == 'R0_GEO_PARETO_NONREGRESSION_FIXED_TRAIN_MINMAX'
        traces.extend(dict(model=model, **r) for r in obj)
        fits[model] = dict(iterations=fit['iterations'], objective_calls=fit['objective_calls'],
                          certified_gap_upper_bound=fit['certificate']['gradient_l2_squared_over_2lambda'],
                          checkpoint=export['published'], byte_identical=True)
    trace_cells = equal_csv(csv_rows(C.DOC / 'TRAINING_OBJECTIVE_LOG.csv'), traces)
    assert len(traces) == report['objective_calls'] == 1143
    assert sum(r['iterations'] for r in fits.values()) == report['iterations'] == 1027

    # The existing independent real review already reproduced all selection/gate
    # computations. Here check exports against its bound caches, without scoring.
    meta = value_from(rp['inputs']['eval_metadata'])
    operational = value_from(rp['inputs']['operational_metrics'])
    candidates = value_from(rp['inputs']['stable_pose_candidates'])
    groups = value_from(rp['inputs']['eval_groups'])
    C.verify(rp['inputs']['candidate_rows'])
    cache = {}
    for row in csv_rows(C.ROOT / rp['inputs']['candidate_rows']['path']):
        key = (row['model'], row['id'], row['selected_whole_pose'])
        vals = (float(row['T_cm']), float(row['R_deg']))
        if key in cache:
            assert cache[key] == vals
        cache[key] = vals
    for fid, rec in operational['R0'].items():
        key = ('R0', fid, candidates['R0'][fid]['GEO_name'])
        vals = (rec['translation_cm'], rec['rotation_deg'])
        if key in cache:
            assert cache[key] == vals
        cache[key] = vals
    rows = csv_rows(C.DOC / 'REAL_FEASIBILITY_ROWS.csv')
    assert len(rows) == rv['frame_seed_rows_checked'] == report['real_diagnostic_csv_rows'] == 519
    lookup = {(int(r['seed']), r['id']): r for r in rows}
    assert len(lookup) == 519 and set(k[0] for k in lookup) == {1, 2, 3}
    anchor_returned = 0
    for row in rows:
        fid, model, hyp = row['id'], row['selected_model'], row['selected_hypothesis']
        assert row['diagnostic'] == 'PARETO_ANCHOR_ORACLE'
        assert row['available'] == row['physical_complete_pose'] == row['GT_derived_diagnostic_only'] == 'True'
        assert row['T_source_model'] == row['R_source_model'] == model
        assert row['T_source_hypothesis'] == row['R_source_hypothesis'] == hyp
        pair = (float(row['T_cm']), float(row['R_deg']))
        assert pair == cache[(model, fid, hyp)]
        anchor = operational['R0'][fid]
        assert float(row['anchor_T_cm']) == anchor['translation_cm']
        assert float(row['anchor_R_deg']) == anchor['rotation_deg']
        assert all(v <= a for v, a in zip(pair, (anchor['translation_cm'], anchor['rotation_deg'])))
        assert row['T_nonincrease'] == row['R_nonincrease'] == 'True'
        assert float(row['fixed_TRAIN_cost']) == max(v / s for v, s in zip(pair, rp['scale']))
        anchor_returned += row['anchor_returned'] == 'True'
    assert anchor_returned == real['fallback_counts']['anchor_returned'] == 291

    # Independently reconstruct all numeric Markdown table rows from bound data.
    required_lines = []
    previous = value_from(rp['inputs']['previous_feasibility'])
    for label, status, summary in [
        ('직전 비용 oracle', '4/5', previous['diagnostics']['FIXED_TRAIN_COST_ORACLE']['summaries']['NATURAL99']['DIVERSE251']['conditional']),
        ('R0 양축 보존 oracle', '5/5', real['summaries']['NATURAL99']['DIVERSE251']['conditional'])]:
        nums = [summary[q][axis]['value'] for axis, q in [('translation_cm', 'median'), ('rotation_deg', 'median'), ('translation_cm', 'P90'), ('rotation_deg', 'P90')]]
        required_lines.append('| ' + label + ' | ' + status + ' | ' + ' | '.join(f'{v:.6f}' for v in nums) + ' |')
    for model, fit in fits.items():
        required_lines.append(f"| {model} | {fit['iterations']} | {fit['objective_calls']} | {fit['certified_gap_upper_bound']:.3e} |")
    for model in C.MODEL_NAMES[1:]:
        r = train['models'][model]
        old, new = r['previous_converged_same_target']['statistics'], r['statistics']
        assert old['available_anchor_rows'] == new['available_anchor_rows'] == 2597
        required_lines.append(f"| {model} | {old['target_accuracy_available_rows']*100:.2f}% → {new['target_accuracy_available_rows']*100:.2f}% | {old['anchor_violations']['either']['count']} → {new['anchor_violations']['either']['count']} / 2597 |")
    for model, summary in sg['summaries'].items():
        nums = [summary['full_population'][axis][q] for axis, q in [('translation_cm', 'median'), ('rotation_deg', 'median'), ('translation_cm', 'P90'), ('rotation_deg', 'P90')]]
        required_lines.append('| ' + model + ' | ' + ' | '.join(f'{v:.6f}' for v in nums) + f" | {summary['failed_pose']} |")
    required_lines.extend('| ' + ' | '.join(f.split('/')) + ' |' for f in failures)
    for line in required_lines:
        assert line in md, line
    for claim in ('안정적인 T·R 동시 개선은 아직 달성하지 못했습니다', '38/45 통과·7개 실패',
                  '학습 정답 생성에만 적용', 'learned 실사 routing·실사 오차 계산은0회',
                  'T 꼬리를 보존하는 대신 직전 비용 oracle보다 R P90은 증가',
                  '반복 사용했습니다', '네 개 독립 반복 실험', '새 실사 정답을 사용하지 않았습니다'):
        assert claim in md, claim

    # Validate the six real RGB headers/dimensions and all twelve displayed
    # cuboid projections using OpenCV's independent projection implementation.
    import cv2
    meta_by_id = {m['id']: m for m in meta}
    expected_ids = [min((fid for fid in groups['NATURAL99'] if meta_by_id[fid]['recording'] == rec),
                        key=lambda fid: (-operational['R0'][fid]['translation_cm'], fid))
                    for rec in sorted({meta_by_id[fid]['recording'] for fid in groups['NATURAL99']})]
    assert [d['id'] for d in report['illustrations']] == expected_ids
    signs = np.asarray(list(itertools.product((-1., 1.), repeat=3)))
    edges = [(i, j) for i in range(8) for j in range(i+1, 8) if np.count_nonzero(signs[i] != signs[j]) == 1]
    illustrations, projection_max = [], 0.
    for detail in report['illustrations']:
        fid = detail['id']; m = meta_by_id[fid]
        assert detail['image'] == m['image'] and detail['dimensions_m'] == m['xyz']
        C.verify(m['image'])
        with Image.open(C.ROOT / m['image']['path']) as im:
            assert list(im.size[::-1]) == m['hw']
            image_size = list(im.size)
        assert detail['oracle_seed'] == 1 and len(detail['panels']) == 2
        oracle = lookup[(1, fid)]
        for j, panel in enumerate(detail['panels']):
            if j == 0:
                assert panel['model'] == 'R0' and panel['hypothesis'] == candidates['R0'][fid]['GEO_name']
            else:
                assert panel['model'] == oracle['selected_model'] and panel['hypothesis'] == oracle['selected_hypothesis']
            assert (panel['T_cm'], panel['R_deg']) == cache[(panel['model'], fid, panel['hypothesis'])]
            pose = next(h['pose'] for h in candidates[panel['model']][fid]['hypotheses'] if h['name'] == panel['hypothesis'])
            corners = signs * np.asarray(pose['cf_extents']) / 2
            rotation, center, camera = np.asarray(pose['R_cf']), np.asarray(pose['centroid']), np.asarray(m['K'])
            xyz = corners @ rotation.T + center
            homog = xyz @ camera.T
            direct = homog[:, :2] / homog[:, 2:3]
            rvect = cv2.Rodrigues(rotation)[0]
            other = cv2.projectPoints(corners, rvect, center, camera, np.zeros(5))[0].reshape(-1, 2)
            difference = float(np.max(np.abs(direct - other)))
            assert difference < 1e-7, (fid, difference)
            projection_max = max(projection_max, difference)
            assert panel['drawn_edges'] == sum(xyz[[i, k], 2].min() > 0 for i, k in edges) == 12
        illustrations.append(dict(id=fid, recording=m['recording'], source_image=m['image'],
                                  width=image_size[0], height=image_size[1], dimensions_m=m['xyz'], panels_checked=2))
    assert len(illustrations) == report['actual_RGB_images'] == 6
    for artifact in report['artifacts']:
        C.verify(artifact)
        with Image.open(C.ROOT / artifact['path']) as im:
            im.verify()
    assert len(report['artifacts']) == 6

    # Report plots are explicitly sourced from the same bound numeric objects.
    # Human visual inspection checks axis units/zero bars and diagnostic labels.
    old_source = C.read(C.CONV_DOC / 'SOURCE_VAL_GATE.json')
    figure_values = {}
    for axis, q in [('translation_cm', 'median'), ('rotation_deg', 'median'), ('translation_cm', 'P90'), ('rotation_deg', 'P90')]:
        k = axis + '_' + q
        figure_values[k] = dict(real=[real['summaries']['NATURAL99']['R0']['conditional'][q][axis]['value'],
                                     previous['diagnostics']['FIXED_TRAIN_COST_ORACLE']['summaries']['NATURAL99']['DIVERSE251']['conditional'][q][axis]['value'],
                                     real['summaries']['NATURAL99']['DIVERSE251']['conditional'][q][axis]['value']],
                              source_old=[old_source['summaries'][m]['full_population'][axis][q] for m in C.MODEL_NAMES],
                              source_new=[sg['summaries'][m]['full_population'][axis][q] for m in C.MODEL_NAMES])
    review_text = '\n'.join([
        '# 공개 보고서 독립 검증', '',
        '**공개 산출물 검증 PASS입니다. 실제 T·R 동시 개선 달성 판정은 FAIL 상태입니다.**', '',
        '[본 보고서](REPORT_KO.md)의 실사 oracle 5/5 통과는 참조를 이용하는 가능성 진단입니다. 실제 학습한 네 선택기는 source VAL에서 38/45만 통과했고, 실패한 7개 조건을 모두 공개했습니다. learned 실사 선택·채점은 실행하지 않았으며 9개 관련 산출물의 부재를 재확인했습니다.', '',
        '- VAL CSV 8,192행의 모든 열을 동결된 오차 NPZ·선택 기록과 일치 확인했습니다. 45개 판정 CSV도 모든 열을 원본 판정과 대조했습니다.',
        '- 학습 CSV 1,143행 전체를 네 원본 trace의 목적함수·CE·L2·gradient·weight SHA에 대조했습니다. optimizer iteration은 총 1,027회이며 네 수렴 인증은 모두 통과했습니다.',
        '- 공개 checkpoint JSON 네 개는 각 fit이 저장한 원본 파일과 byte 단위로 같습니다. 수렴 인증은 CE+ridge에 대한 것이며 성능 보장은 아닙니다.',
        '- 실사 진단 519행의 선택한 전체 pose identity·T/R·고정 TRAIN 비용·R0 양축 비증가를 기존 동결 캐시에 대조했습니다. 원래 5개 기준의 독립 재현은 [실사 검산](REAL_VERIFICATION_KO.md)에 연결했습니다.',
        '- 본문의 숫자 표 24행과 실패 조건을 원본 결과에서 다시 구성해 대조했습니다. TRAIN 정확도와 R0 악화 선택 수는 유효 anchor 2,597장 분모이며, 원래 실패 1장은 전체 2,598장 집계에 남아 있습니다.',
        '- 실제 RGB 6장의 원본 SHA·가로/세로·입력 치수를 확인했습니다. 표시된 12개 cuboid는 기존 pose의 직접 투영이며 OpenCV 별도 투영과 1e−7 pixel 이내로 일치합니다. 새 PnP나 참조 윤곽 생성은 없습니다.',
        '- 그림 6개를 시각 검토했습니다. 숫자 그래프의 단위·축·모든 모델 표시, oracle 진단 문구와 source 실패 문구가 구분됩니다. 갤러리 제목 겹침을 수정한 최종 저장본을 다시 확인했습니다. RGB 사례는 각 recording의 가장 큰 기존 R0 T 오차로 정한 진단 예시입니다.', '',
        '[source 독립 검산](SOURCE_VAL_VERIFICATION_KO.md)과 [TRAIN 독립 검산](TRAIN_CONVERGENCE_KO.md)의 최종 결과를 SHA로 연결했습니다. 공개 수치 검증은 새 fit·새 성능 채점·새 정답 읽기 없이 저장 결과만 대조했습니다.', '',
        '검증의 범위는 공개 자료의 정합성입니다. 반복 사용한 DEV/source VAL의 독립 일반화나 실제 개선 성공을 증명하지 않습니다. 모든 Markdown 링크를 현재 공개 예정 디렉터리와 기존 publication checkout에서 확인했으며, 이후 생성되는 PUBLICATION_MANIFEST.json 한 파일만 미생성 예외입니다.', '',
        '세부 원본 SHA, 행·열 검사 수, 투영 최대차이, 링크 목록은 PUBLIC_REVIEW.json에 기록했습니다.', ''])
    # 2 oracle +4 fits +3 train +8 VAL +7 failure rows.
    assert len(required_lines) == 24
    link_report = links(review_text, base)
    bindings = {name: C.bind(C.DOC / name) for name in [
        'REPORT_KO.md', 'REPORT_DATA.json', 'SOURCE_VAL_GATE.json', 'SOURCE_VAL_VERIFICATION.json',
        'SOURCE_VAL_VERIFICATION_KO.md', 'REAL_VERIFICATION.json', 'REAL_VERIFICATION_KO.md',
        'TRAIN_CONVERGENCE.json', 'TRAIN_CONVERGENCE_KO.md', 'REAL_EVALUATION_NOT_RUN.json',
        'SOURCE_VAL_FRAME_RESULTS.csv', 'SOURCE_VAL_CHECKS.csv', 'TRAINING_OBJECTIVE_LOG.csv', 'REAL_FEASIBILITY_ROWS.csv']}
    base_head = subprocess.check_output(['git', '-C', str(base), 'rev-parse', 'HEAD'], text=True).strip()
    receipt = dict(complete=True, PASS=True, created_at=C.now(), verifier=C.bind(Path(__file__)),
                   scope='Publication consistency only; no new fit, inference, selection, or reference-metric scoring.',
                   report_code=report['code'], bindings=bindings,
                   prior_source_figure_reference=C.bind(C.CONV_DOC / 'SOURCE_VAL_GATE.json'),
                   CSV=dict(source_rows=8192, source_cells_checked=source_cells, check_rows=45,
                            check_cells_checked=check_cells, trace_rows=1143, trace_cells_checked=trace_cells,
                            real_diagnostic_rows=519, all_csv_LF=True),
                   fits=fits, new_fits_reported=4, objective_calls=1143, optimizer_iterations=1027,
                   markdown_numeric_rows_checked=len(required_lines), actual_source_checks_passed=38,
                   actual_source_checks_total=45, failed_checks=failures, real_oracle_diagnostic_gate_PASS=True,
                   learned_real_evaluated=False, real_absence_checks=nr['absence_checks'],
                   figures=report['artifacts'], figure_values=figure_values, visual_review_all6=True,
                   gallery_layout_fixed_and_rereviewed=True, illustrations=illustrations,
                   projection_max_absolute_difference_px=projection_max,
                   projection_check='Existing cuboid/K matrix projection versus cv2.projectPoints; no PnP or metric scoring.',
                   links=link_report, publication_base_head=base_head,
                   new_fits_by_verifier=0, new_inference=0, new_GT_reads=0, new_metrics=0,
                   method_success=False, goal_complete=False)
    return receipt, review_text


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--freeze', action='store_true')
    parser.add_argument('--publication-base', type=Path, default=Path('/tmp/pallet-pose-github-review-20260930'))
    args = parser.parse_args()
    receipt, text = verify(args.publication_base)
    if args.freeze:
        assert not (C.DOC / 'PUBLIC_REVIEW.json').exists() and not (C.DOC / 'PUBLIC_REVIEW_KO.md').exists()
        C.save(C.DOC / 'PUBLIC_REVIEW_KO.md', text)
        receipt['review_markdown'] = C.bind(C.DOC / 'PUBLIC_REVIEW_KO.md')
        C.save(C.DOC / 'PUBLIC_REVIEW.json', receipt)
    print(json.dumps(dict(PASS=True, frozen=args.freeze, csv=receipt['CSV'], links=receipt['links']['checked_count'],
                          projection_max_absolute_difference_px=receipt['projection_max_absolute_difference_px']), ensure_ascii=False))


if __name__ == '__main__':
    main()
