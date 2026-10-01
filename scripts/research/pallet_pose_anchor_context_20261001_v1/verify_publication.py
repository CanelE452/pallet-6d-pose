"""Audit public exports against frozen results; never rescore model/reference data.

The previous real gallery is reused byte for byte and checked against its
already published review. No new real oracle, pose projection, or GT is read.
Freeze only once, after the report is final and all three new plots are viewed.
"""
import argparse
import ast
import csv
import hashlib
import json
from pathlib import Path
import re
import subprocess
from urllib.parse import unquote

import numpy as np
from PIL import Image
from . import common as C


def rows(path):
    assert b'\r' not in Path(path).read_bytes(), path
    with Path(path).open(newline='') as handle:
        return list(csv.DictReader(handle))


def value(binding):
    C.verify(binding)
    return C.read(C.ROOT / binding['path'])


def csv_equal(actual, expected):
    assert len(actual) == len(expected)
    count = 0
    for row, target in zip(actual, expected):
        assert row.keys() == target.keys()
        for key, number in target.items():
            if isinstance(number, (int, float)) and not isinstance(number, bool):
                assert float(row[key]) == number, (key, row[key], number)
            else:
                assert row[key] == str(number), (key, row[key], number)
            count += 1
    return count


def public_links(review, base):
    tracked = set(subprocess.check_output(['git','-C',str(base),'ls-tree','-r','--name-only','HEAD'],text=True).splitlines())
    sources = {p: p.read_text() for p in C.DOC.glob('*.md')}
    sources[C.DOC/'PUBLIC_REVIEW_KO.md'] = review
    expected_outputs = {C.DOC/'PUBLIC_REVIEW_KO.md', C.DOC/'PUBLIC_REVIEW.json'}
    manifest = C.DOC/'PUBLICATION_MANIFEST.json'
    checked, pending = [], set()
    for source, content in sources.items():
        for link in re.findall(r'!?\[[^\]]*\]\(([^)]+)\)', content):
            link = unquote(link.strip().strip('<>'))
            if re.match(r'https?://|mailto:|#', link):
                continue
            target = (source.parent/link.split('#', 1)[0]).resolve()
            assert target.is_relative_to(C.ROOT), (source, link)
            if target == manifest and not target.exists():
                pending.add(str(target.relative_to(C.ROOT)))
            elif target.is_relative_to(C.DOC) or target.is_relative_to(C.HERE):
                assert target.exists() or target in expected_outputs, (source, link)
            else:
                relative = str(target.relative_to(C.ROOT))
                assert (base/relative).exists(), ('UNPUBLISHED_LINK', source, link)
                assert relative in tracked or any(p.startswith(relative.rstrip('/')+'/') for p in tracked), ('NOT_IN_PUBLICATION_HEAD',source,link)
            checked.append(dict(source=str(source.relative_to(C.ROOT)), target=link))
    assert pending <= {str(manifest.relative_to(C.ROOT))}
    return dict(checked_count=len(checked), checked=checked, allowed_pending_unique=sorted(pending))


def published_copy(binding, base):
    C.verify(binding)
    local, remote = C.ROOT/binding['path'], base/binding['path']
    assert remote.is_file(), binding['path']
    assert local.read_bytes() == remote.read_bytes(), ('PUBLISHED_BYTE_DRIFT', binding['path'])


def verify_gallery(report, base, markdown):
    old_review_path = C.ANCHOR_DOC/'PUBLIC_REVIEW.json'
    old_report_path = C.ANCHOR_DOC/'REPORT_DATA.json'
    published_copy(C.bind(old_review_path), base)
    prior = C.read(old_review_path)
    assert prior['complete'] and prior['PASS'] and prior['visual_review_all6']
    assert prior['new_metrics'] == prior['new_GT_reads'] == 0
    assert prior['bindings']['REPORT_DATA.json'] == C.bind(old_report_path)
    published_copy(prior['bindings']['REPORT_DATA.json'], base)
    old_report = C.read(old_report_path)
    assert report['reused_image_report'] == prior['bindings']['REPORT_DATA.json']
    assert report['reused_actual_RGB_images'] == old_report['actual_RGB_images'] == 6
    old_images = [b for b in old_report['artifacts'] if b['path'].endswith('.jpg')]
    assert len(old_images) == 3
    assert report['reused_figures'] == old_images
    reused = []
    for binding in old_images:
        published_copy(binding, base)
        target = '../pallet_pose_pareto_anchor_20261001_v1/figures/' + Path(binding['path']).name
        assert target in markdown, ('MISSING_REUSED_GALLERY_EMBED', target)
        with Image.open(C.ROOT/binding['path']) as im:
            size = list(im.size); im.verify()
        reused.append(dict(binding=binding, dimensions_px=size, previously_published_byte_identical=True))
    prior_inputs = {r['id']: r for r in prior['illustrations']}
    assert len(prior_inputs) == 6
    input_evidence = []
    for detail in old_report['illustrations']:
        before = prior_inputs[detail['id']]
        assert before['source_image'] == detail['image']
        assert before['dimensions_m'] == detail['dimensions_m'] and detail['oracle_seed'] == 1
        assert len(detail['panels']) == before['panels_checked'] == 2
        assert all(p['drawn_edges'] == 12 for p in detail['panels'])
        # Verify old input image integrity only; there is no model/pose operation.
        C.verify(detail['image'])
        with Image.open(C.ROOT/detail['image']['path']) as im:
            assert list(im.size) == [before['width'], before['height']]
            im.verify()
        input_evidence.append(dict(id=detail['id'], image=detail['image'],
            dimensions_m=detail['dimensions_m'], width=before['width'], height=before['height']))
    return dict(previous_public_review=C.bind(old_review_path), previous_report=prior['bindings']['REPORT_DATA.json'],
        reused_montages=reused, source_RGB_inputs=input_evidence, source_RGB_count=6,
        new_real_metric_calculations=0, new_pose_projections=0,
        display_rule='Previously published seed1-only R0-largest-T frame per natural recording; unchanged selection and pixels.',
        previous_projection_verification_max_px=prior['projection_max_absolute_difference_px'])


def verify(base, visual):
    initial_report = C.bind(C.DOC/'REPORT_DATA.json')
    initial_md = C.bind(C.DOC/'REPORT_KO.md')
    report, md = C.read(C.DOC/'REPORT_DATA.json'), (C.DOC/'REPORT_KO.md').read_text()
    for key in ('code', 'train_protocol', 'source_gate', 'training_verification', 'real_not_run',
                'source_verification', 'prefit_verification', 'previous_source_gate', 'training_risk_diagnostic'):
        C.verify(report[key])
    sg = value(report['source_gate'])
    train = value(report['training_verification'])
    nr = value(report['real_not_run'])
    protocol = value(report['train_protocol'])
    sf = value(protocol['inputs']['source_feasibility'])
    sv = C.read(C.DOC/'SOURCE_VAL_VERIFICATION.json')
    prefit = C.read(C.DOC/'PREFIT_REVIEW.json')
    risk = value(report['training_risk_diagnostic'])
    assert risk['complete'] and risk['PASS'] and risk['source_TRAIN_only']
    assert not risk['VAL_quality_read'] and not risk['real_targets_read']
    assert risk['new_fits'] == risk['optimizer_steps'] == risk['alternative_policy_probes'] == 0
    assert report['complete'] and report['actual_new_fits'] == 4
    assert not report['learned_real_evaluated'] and not report['stable_joint_improvement_achieved'] and not report['goal_complete']
    assert sf['PASS'] and train['PASS'] and sv['PASS'] and prefit['PASS']
    assert report['source_verification'] == C.bind(C.DOC/'SOURCE_VAL_VERIFICATION.json')
    assert report['prefit_verification'] == C.bind(C.DOC/'PREFIT_REVIEW.json')
    assert sg['complete'] and not sg['PASS'] and sg['checks_total'] == 45 and sg['checks_passed'] == 43
    assert sv['source_gate'] == report['source_gate'] and sv['checks_passed'] == 43 and sv['all_gate_booleans_exact']
    assert not sv['source_gate_PASS'] and not sg['real_routing_authorized']
    for key in ('learned_real_routes', 'real_metric_calls', 'raw_real_reference_reads', 'image_forwards', 'new_fits_by_real_evaluator'):
        assert nr[key] == 0
    assert nr['source_gate'] == report['source_gate'] and nr['status'] == 'NOT_RUN_SOURCE_GATE_FAILED'
    for path, absent in nr['absence_checks'].items():
        assert absent and not (C.ROOT/path).exists(), path
    choices = value(sv['choices'])
    C.verify(sg['metrics'])
    with np.load(C.ROOT/sg['metrics']['path'], allow_pickle=False) as z:
        ids, models = z['ids'].tolist(), z['models'].tolist()
        errors = {m: z[m] for m in models}
    previous_source = value(report['previous_source_gate'])
    assert previous_source['checks_passed'] == report['previous_source_VAL_checks_passed'] == 38
    C.verify(previous_source['metrics'])
    fixed_parity = {}
    with np.load(C.ROOT/previous_source['metrics']['path'], allow_pickle=False) as z:
        assert z['ids'].tolist() == ids
        for model in ('R0_GEO','DIVERSE251_s1_GEO','DIVERSE251_s2_GEO','DIVERSE251_s3_GEO'):
            assert z[model].tobytes() == errors[model].tobytes()
            assert previous_source['summaries'][model] == sg['summaries'][model]
            fixed_parity[model] = True
        changed_control = z['R0_ONLY'].tobytes() != errors['R0_ONLY'].tobytes()
        assert changed_control, 'The retrained R0_ONLY control must not be described as a fixed baseline.'
    expected = []
    for model in models:
        assert errors[model].shape == (1024, 2) and np.isfinite(errors[model]).all()
        for j, fid in enumerate(ids):
            pick = choices['records'].get(model, {}).get(fid, {})
            expected.append(dict(model=model, id=fid, split='VAL', available=True,
                T_cm=float(errors[model][j, 0]), R_deg=float(errors[model][j, 1]),
                candidate=pick.get('candidate_name', 'fixed_GEO'), fallback=pick.get('fallback', False)))
    source_cells = csv_equal(rows(C.DOC/'SOURCE_VAL_FRAME_RESULTS.csv'), expected)
    assert len(expected) == 8192 and len({(r['model'], r['id']) for r in expected}) == 8192
    check_rows = [dict(model=m, baseline=b, criterion=k, PASS=v) for m, cc in sg['comparisons'].items()
                  for b, group in cc.items() for k, v in group['checks'].items()]
    check_cells = csv_equal(rows(C.DOC/'SOURCE_VAL_CHECKS.csv'), check_rows)
    failures = ['/'.join((r['model'], r['baseline'], r['criterion'])) for r in check_rows if not r['PASS']]
    assert len(check_rows) == 45 and sum(r['PASS'] for r in check_rows) == 43
    assert failures == sg['failed_checks'] == sv['failed_checks'] == nr['failed_checks'] == [
        'UNION_s3/R0_ONLY/translation_cm_P90_guard', 'UNION_s3/R0_GEO/translation_cm_P90_guard']
    fits, traces, walls = {}, [], []
    for model in C.MODEL_NAMES:
        fit = C.read(C.DOC/f'FIT_{model}.json')
        for key in ('START', 'trace', 'checkpoint', 'protocol'):
            C.verify(fit[key])
        assert fit['protocol'] == report['train_protocol'] and fit['fits_executed'] == 1
        log = [json.loads(s) for s in (C.ROOT/fit['trace']['path']).read_text().splitlines()]
        objective = [r for r in log if r['event'] == 'objective']
        iterations = [r for r in log if r['event'] == 'iteration']
        assert [r['call'] for r in objective] == list(range(1, fit['objective_calls']+1))
        assert [r['iteration'] for r in iterations] == list(range(1, fit['iterations']+1))
        assert objective[-1]['weight_sha'] == fit['final_weight_sha']
        assert objective[-1]['objective'] == fit['certificate']['objective_value']
        assert fit['certificate']['PASS'] and fit['certificate']['gradient_l2_squared_over_2lambda'] <= 1e-6
        assert train['models'][model]['objective_calls'] == fit['objective_calls']
        assert train['models'][model]['iterations'] == fit['iterations']
        export = report['exports'][model]
        assert export['local'] == fit['checkpoint']
        C.verify(export['published'])
        assert (C.ROOT/export['local']['path']).read_bytes() == (C.ROOT/export['published']['path']).read_bytes()
        ck = value(fit['checkpoint'])
        assert ck['schema'] == 'pallet_pose_anchor_context_linear189_v1'
        assert ck['feature_map'] == 'normalized94_abs_anchor_delta94_identity1' and len(ck['weight']) == 189
        assert ck['target_rule'] == 'R0_GEO_PARETO_NONREGRESSION_FIXED_TRAIN_MINMAX'
        assert ck['runtime_safe_mask'] is False
        traces.extend(dict(model=model, **r) for r in objective)
        fits[model] = dict(iterations=fit['iterations'], objective_calls=fit['objective_calls'],
            certified_gap_upper_bound=fit['certificate']['gradient_l2_squared_over_2lambda'],
            checkpoint=export['published'], byte_identical=True)
        walls.append(fit['wall_seconds'])
    trace_cells = csv_equal(rows(C.DOC/'TRAINING_OBJECTIVE_LOG.csv'), traces)
    assert len(traces) == report['objective_calls'] == 1495
    assert sum(f['iterations'] for f in fits.values()) == report['iterations'] == 1350
    required_lines = []
    for model, fit in fits.items():
        required_lines.append(f"| {model} | {fit['iterations']} | {fit['objective_calls']} | {fit['certified_gap_upper_bound']:.3e} |")
    for model, summary in sg['summaries'].items():
        nums = [summary['full_population'][axis][q] for axis,q in [('translation_cm','median'),('rotation_deg','median'),('translation_cm','P90'),('rotation_deg','P90')]]
        required_lines.append('| '+model+' | '+' | '.join(f'{v:.6f}' for v in nums)+f" | {summary['failed_pose']} |")
    for model in C.MODEL_NAMES:
        entry = train['models'][model]
        old, new = entry['previous_converged_same_target']['statistics'], entry['statistics']
        assert old['frames'] == new['frames'] == 2598
        assert old['available_anchor_rows'] == new['available_anchor_rows'] == 2597
        assert old['failed_rows'] == new['failed_rows'] == 1
        required_lines.append(f"| {model} | {old['target_accuracy_full_population']*100:.2f}% → {new['target_accuracy_full_population']*100:.2f}% | {old['anchor_violations']['either']['count']} → {new['anchor_violations']['either']['count']} |")
    failure_values = report['seed3_T_P90_failure']
    seed3_p90 = sg['summaries']['UNION_s3']['full_population']['translation_cm']['P90']
    baseline_p90 = sg['summaries']['R0_GEO']['full_population']['translation_cm']['P90']
    assert failure_values['value_cm'] == seed3_p90
    assert failure_values['R0_GEO_cm'] == failure_values['R0_ONLY_cm'] == baseline_p90
    assert failure_values['limit_cm'] == 1.05*baseline_p90
    assert failure_values['ratio_to_baseline'] == seed3_p90/baseline_p90
    assert failure_values['percent_above_baseline'] == 100*(seed3_p90/baseline_p90-1)
    for failure in failures:
        model, baseline, criterion = failure.split('/')
        cap = sg['summaries'][baseline]['full_population']['translation_cm']['P90']*1.05
        required_lines.append(f'| {model} | {baseline} | {criterion} | {seed3_p90:.6f} | {cap:.9f} |')
    for model in C.MODEL_NAMES:
        old = previous_source['summaries'][model]['full_population']
        new = sg['summaries'][model]['full_population']
        comparisons = [f'{old[axis][q]:.6f} → {new[axis][q]:.6f}' for axis,q in
                       [('translation_cm','median'),('rotation_deg','median'),('translation_cm','P90'),('rotation_deg','P90')]]
        required_lines.append('| '+model+' | '+' | '.join(comparisons)+' |')
    risk_lines = []
    for model in C.MODEL_NAMES:
        r = risk['models'][model]
        chosen = r['selected']['classes']; targets = r['target']['classes']
        assert targets['unsafe']['count'] == 0 and chosen['failed']['count'] == 1
        assert sum(v['count'] for v in chosen.values()) == 2598
        assert chosen['unsafe']['count'] == train['models'][model]['statistics']['anchor_violations']['either']['count']
        safe_target = targets['safe_improvement']['count'] + targets['safe_equal']['count']
        if model != 'R0_ONLY':
            maximum=r['selected']['violations']['either']['normalized_max_excess']['maximum']
            required_lines.append(f"| {model} | {safe_target} | {chosen['anchor']['count']} | {chosen['safe_improvement']['count']} | {chosen['unsafe']['count']} | {maximum:.2f} |")
        risk_lines.append('| '+model+' | '+' | '.join(str(v) for v in [targets['anchor']['count'],safe_target,
            chosen['anchor']['count'],chosen['safe_improvement']['count'],chosen['safe_equal']['count'],chosen['unsafe']['count'],chosen['failed']['count']])+' |')
        for axis in ('T','R','either'):
            v=r['selected']['violations'][axis];q=v['normalized_max_excess']
            risk_lines.append(f"| {model} | {axis} | {v['count']} | {q['median']:.6f} | {q['P90']:.6f} | {q['maximum']:.6f} |")
    risk_md=(C.DOC/'TRAIN_RISK_DIAGNOSTIC_KO.md').read_text()
    for line in risk_lines:
        assert line in risk_md, ('RISK_TABLE_MISMATCH',line)
    for line in required_lines:
        assert line in md, ('MISSING_NUMERIC_ROW', line)
    assert len(required_lines) == 25 and len(risk_lines) == 16
    for claim in ('안정적인 T·R 동시 개선은 아직 달성하지 못했습니다', '43/45 통과·2개 실패',
                  'R0_ONLY도 이번 표현으로 다시 학습', '고정 R0_GEO 비교',
                  'learned 실사 routing·오차 계산은0회', '이번 새 모델의 실사 결과가 아닙니다',
                  '비율의 분모는 전체2598장', '네 독립 반복 실험으로 해석하지 않습니다',
                  'source VAL과 실사 DEV는 여러 방법에서 반복 사용했습니다'):
        assert claim in md, ('MISSING_SCOPE_CLAIM', claim)
    assert f'{failure_values["percent_above_baseline"]:.2f}% 증가' in md
    figure_values = {}
    for axis,q in [('translation_cm','median'),('rotation_deg','median'),('translation_cm','P90'),('rotation_deg','P90')]:
        key=axis+'_'+q
        figure_values[key]=dict(previous94=[previous_source['summaries'][m]['full_population'][axis][q] for m in C.MODEL_NAMES],
                               current189=[sg['summaries'][m]['full_population'][axis][q] for m in C.MODEL_NAMES])
    oldstats=[train['models'][m]['previous_converged_same_target']['statistics'] for m in C.MODEL_NAMES]
    newstats=[train['models'][m]['statistics'] for m in C.MODEL_NAMES]
    figure_values['train_target_accuracy_percent']=dict(previous94=[s['target_accuracy_full_population']*100 for s in oldstats],
        current189=[s['target_accuracy_full_population']*100 for s in newstats],denominator=2598)
    figure_values['train_any_axis_anchor_violation_count']=dict(previous94=[s['anchor_violations']['either']['count'] for s in oldstats],
        current189=[s['anchor_violations']['either']['count'] for s in newstats],frames=2598,available_anchor_rows=2597,failed_rows=1)
    assert report['figure_values'] == figure_values
    for m in C.MODEL_NAMES[1:]:
        old=previous_source['summaries'][m]['full_population'];new=sg['summaries'][m]['full_population']
        assert new['rotation_deg']['P90'] < old['rotation_deg']['P90']
        if m == 'UNION_s3':
            assert new['translation_cm']['median'] == old['translation_cm']['median']
        else:
            assert new['translation_cm']['median'] > old['translation_cm']['median']
        old=train['models'][m]['previous_converged_same_target']['statistics']['selected_error_summary']['full_population']
        new=train['models'][m]['statistics']['selected_error_summary']['full_population']
        assert all(new[axis]['median'] > old[axis]['median'] for axis in ('translation_cm','rotation_deg'))
    figures = []
    for artifact in report['artifacts']:
        C.verify(artifact)
        with Image.open(C.ROOT/artifact['path']) as im:
            size = list(im.size); im.verify()
        if artifact['path'].startswith(str(C.DOC.relative_to(C.ROOT))+'/'):
            assert artifact['path'].endswith('.png')
            figures.append(dict(binding=artifact, dimensions_px=size))
    assert len(figures) == 3
    gallery = verify_gallery(report, base, md)
    review = '\n'.join([
        '# 공개 보고서 독립 검증', '',
        '**공개 자료 정합성 검산 PASS. 학습한 방법은 source VAL 43/45로 기준 미달이며 실제 T·R 동시 개선 목표는 미달성이다.**', '',
        '[본 보고서](REPORT_KO.md)의 VAL CSV 8,192행·45개 판정 CSV·학습 목적함수 CSV 1,495행 모든 열을 각각 동결된 원본 배열·선택 기록·trace에 대조했다. 네 fit의 optimizer iteration은 총 1,350회이며 공개 parameter JSON 네 개는 원본 final checkpoint와 byte 단위로 같다.', '',
        '학습 전 특징 검산, TRAIN 수렴 검산, source VAL 물리 지표 검산은 각각 PASS지만 학습 모델의 gate는 FAIL이다. 실패한 두 조건은 UNION_s3의 T P90을 R0_ONLY 및 R0_GEO와 비교한 보호 조건이다. 실제 이미지의 learned routing과 성능 계산은 실행하지 않았고 관련 9개 산출물의 부재를 재확인했다.', '',
        '새 숫자 그래프 3개를 시각 검토했다. source VAL T/R median·P90, 실제 목적함수 trace, TRAIN 정확도·anchor 악화 선택 수가 서로 다른 집계임을 확인했다. 본문의 숫자 표 25행과 별도 TRAIN 위험 진단 표 16행을 원본 JSON에 대조했다. TRAIN 정확도 분모는 전체 2,598개이며 악화 선택 수는 유효 anchor 2,597개 중 센 값이다. 실패 한 행은 그대로 남는다. 위험 진단의 제안은 아직 실행한 학습 방법이 아니다.', '',
        '직전 38/45와 현재 43/45는 R0_ONLY도 각각 재학습한 비교다. control의 저장 오류 배열이 실제로 달라짐을 확인했다. 고정 R0_GEO와 세 DIVERSE_GEO의 8,192개 오류 값은 직전과 byte 단위로 같다. 통과 수 증가를 모든 T/R 지표의 개선으로 해석하지 않는 본문 문구를 확인했다.', '',
        '기존 실사 RGB 6장과 montage JPG 3장은 직전 공개 보고서·검산 영수증의 SHA 및 publication checkout byte와 연결했다. 입력 이미지 SHA·크기·치수, seed1 고정·기존 recording별 사례 선택은 바꾸지 않았다. 새 실사 oracle, pose 투영 또는 참조 지표를 계산하지 않았다. 이 이미지는 이전 GT 기반 가능성 진단이며 이번 모델의 실사 예측 결과가 아니다.', '',
        '모든 Markdown 링크는 이번 공개 디렉터리 또는 기존 publication checkout에서 확인했다. 이후 생성할 PUBLICATION_MANIFEST.json 한 파일만 미생성 예외다. 이 검산은 반복 사용 source VAL의 일반화나 성능 성공을 보장하지 않는다.', '',
        '[검산 JSON](PUBLIC_REVIEW.json), [source 독립 검산](SOURCE_VAL_VERIFICATION_KO.md), [TRAIN 독립 검산](TRAIN_CONVERGENCE_KO.md), [실사 미실행](REAL_EVALUATION_NOT_RUN_KO.md)', ''])
    link_result = public_links(review, base)
    names = ['REPORT_DATA.json','REPORT_KO.md','SOURCE_VAL_GATE.json','SOURCE_VAL_VERIFICATION.json','SOURCE_VAL_VERIFICATION_KO.md',
             'TRAIN_CONVERGENCE.json','TRAIN_CONVERGENCE_KO.md','PREFIT_REVIEW.json','PREFIT_REVIEW_KO.md','REAL_EVALUATION_NOT_RUN.json',
             'SOURCE_VAL_FRAME_RESULTS.csv','SOURCE_VAL_CHECKS.csv','TRAINING_OBJECTIVE_LOG.csv',
             'TRAIN_RISK_DIAGNOSTIC.json','TRAIN_RISK_DIAGNOSTIC_KO.md']
    for path in C.HERE.glob('*.py'):
        ast.parse(path.read_text())
    assert initial_report == C.bind(C.DOC/'REPORT_DATA.json') and initial_md == C.bind(C.DOC/'REPORT_KO.md'), 'REPORT_CHANGED_DURING_REVIEW'
    receipt = dict(complete=True, PASS=True, created_at=C.now(), verifier=C.bind(__file__), report_code=report['code'],
        scope='Publication consistency only; no new fit, inference, reference metric, target or oracle computation.',
        bindings={name:C.bind(C.DOC/name) for name in names},
        CSV=dict(source_rows=8192,source_cells_checked=source_cells,check_rows=45,check_cells_checked=check_cells,
                 trace_rows=1495,trace_cells_checked=trace_cells,all_csv_LF=True),
        fits=fits, objective_calls=1495, optimizer_iterations=1350, actual_new_fits=4,
        fixed_GEO_byte_parity=fixed_parity, R0_ONLY_control_retrained_and_arrays_changed=changed_control,
        figure_values=figure_values,
        markdown_numeric_rows_checked=len(required_lines), risk_markdown_numeric_rows_checked=len(risk_lines),
        source_checks_total=45, source_checks_passed=43, failed_checks=failures,
        learned_real_evaluated=False, real_absence_checks=nr['absence_checks'], figures=figures,
        visual_review_new3=visual, old_gallery_reuse=gallery, links=link_result,
        publication_base_head=subprocess.check_output(['git','-C',str(base),'rev-parse','HEAD'],text=True).strip(),
        new_fits_by_verifier=0,new_inference=0,new_GT_reads=0,new_metrics=0,new_real_oracle=0,
        method_success=False,goal_complete=False)
    return receipt, review


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--freeze',action='store_true')
    parser.add_argument('--visual-reviewed',action='store_true')
    parser.add_argument('--publication-base',type=Path,default=Path('/tmp/pallet-pose-github-review-20260930'))
    args=parser.parse_args()
    assert not args.freeze or args.visual_reviewed, 'View all3 new figure files before freezing.'
    receipt, review=verify(args.publication_base,args.visual_reviewed)
    if args.freeze:
        assert not (C.DOC/'PUBLIC_REVIEW.json').exists() and not (C.DOC/'PUBLIC_REVIEW_KO.md').exists()
        C.save(C.DOC/'PUBLIC_REVIEW_KO.md',review)
        receipt['review_markdown']=C.bind(C.DOC/'PUBLIC_REVIEW_KO.md')
        C.save(C.DOC/'PUBLIC_REVIEW.json',receipt)
    print(json.dumps(dict(PASS=True,frozen=args.freeze,CSV=receipt['CSV'],links=receipt['links']['checked_count'])))


if __name__=='__main__':
    main()
