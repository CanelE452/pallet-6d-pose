"""최종33개 구성·반복·예산·공개 산출물 감사. 미완료 입력은 PASS가 아니다."""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime
from pathlib import Path
import re
import subprocess
import xml.etree.ElementTree as ET

import numpy as np

from . import common as C
from . import audit_controls as A

REQUIRED = ('FINAL_RESULTS.json', 'FINAL_ROBUSTNESS.json', 'FINAL_DECISION.json',
    'FINAL_CONFIGURATION.json', 'REPORT_BUILD.json', 'REPORT_KO.md', 'REPRODUCE.md',
    'CONTROL_AUDIT.json', 'CALIBRATION_INTEGRITY_AUDIT.json', 'CLEAR43_EVALUATION_AUDIT.json',
    'HISTORICAL_RAW_LOCK.json', 'HISTORICAL_RAW_RESULTS.json', 'CASES_FINAL.json',
    'CANDIDATE_TAIL_CEILING.json', 'NEXT_HYPOTHESIS_AUDIT.md', 'NEXT_HYPOTHESIS_DECISION_KO.md',
    'TRAIN_CROSS_INPUT_RESULTS.json', 'TRAIN_CROSS_INPUT_PROTOCOL.json',
    'TRAIN_CROSS_INPUT_PREDICTIONS_LOCK.json', 'TEST_FINAL.xml')


class Verifier(A.Verifier):
    """과거 시점의 비용 해시는 당시 바이트가 보존된 별도 스냅샷으로 검증한다."""
    def __init__(self):
        super().__init__()
        self.historical_ledger_snapshots = set()

    def binding(self, row):
        if row['path'] == str((C.DOC/'RESOURCE_LEDGER.json').relative_to(C.ROOT)):
            snapshot = C.DOC/'RESOURCE_LEDGER_BEFORE_SELECTOR.json'
            if snapshot.is_file() and row['sha256'] == C.sha(snapshot):
                if 'bytes' in row:
                    assert snapshot.stat().st_size == row['bytes']
                self.seen.add((row['path'], row['sha256']))
                self.historical_ledger_snapshots.add(str(snapshot.relative_to(C.ROOT)))
                return
        super().binding(row)


def missing_inputs(root):
    return [name for name in REQUIRED if not (root/name).is_file()]


def test_count(path):
    root = ET.parse(path).getroot()
    suites = [root] if root.tag == 'testsuite' else list(root.findall('testsuite'))
    assert suites
    counts = {key: sum(int(s.attrib.get(key, 0)) for s in suites) for key in ('tests', 'failures', 'errors', 'skipped')}
    assert counts['tests'] > 0 and counts['failures'] == counts['errors'] == 0
    return dict(counts, passed=counts['tests']-counts['skipped'])


def bootstrap(before, after, rows, repeats=2000, seed=20260929):
    """집계 코드를 재호출하지 않고 recording 복원추출 구간을 재계산한다."""
    names = sorted({row['recording'] for row in rows})
    blocks = [[r['id'] for r in rows if r['recording'] == name] for name in names]
    rng = np.random.default_rng(seed)
    samples = {key: [] for key in ('translation_cm', 'rotation_deg')}
    undefined = dict.fromkeys(samples, 0)
    for _ in range(repeats):
        ids = [fid for i in rng.integers(0, len(blocks), len(blocks)) for fid in blocks[i]]
        for key in samples:
            medians = [np.median([data[fid][key] if data[fid]['available'] else np.inf for fid in ids]) for data in (before, after)]
            if not np.isfinite(medians).all():
                undefined[key] += 1
            else:
                samples[key].append(medians[1]-medians[0])
    return dict(recording_count=len(names), recording_sizes={name: len(block) for name, block in zip(names, blocks)},
        intervals={key: dict(percentile95=np.quantile(values, [.025, .975]).tolist() if values else None,
            finite_resamples=len(values), undefined_resamples=undefined[key]) for key, values in samples.items()})


def privacy(value):
    """실사 좌표/카메라 행렬/pose array는 금지하되 해당 파일 해시 연결은 허용한다."""
    denied = {'keypoints', 'coordinates', 'R_physical', 'R_cf', 'centroid', 'K',
              'image_base64', 'image_data', 'predictions', 'candidate_metrics'}
    if isinstance(value, dict):
        for key, child in value.items():
            if key in denied:
                assert isinstance(child, dict) and {'path', 'sha256'} <= set(child), key
            privacy(child)
    elif isinstance(value, list):
        for child in value:
            privacy(child)
    elif isinstance(value, str):
        assert not value.startswith('data:image/')


def current_cost(verifier):
    ledger = verifier.read(C.DOC/'RESOURCE_LEDGER.json')
    verifier.nested(ledger['inherited_ledger'])
    historical = verifier.read(C.ROOT/ledger['inherited_ledger']['path'])
    A.close(ledger['inherited_totals'], historical['totals'])
    assert historical['totals']['student_fits'] == 6
    keys = ('student_fits', 'optimizer_updates', 'selector_fits', 'GPU_training_seconds')
    summed = {key: ledger['inherited_totals'][key]+sum(e[key] for e in ledger['new_events']) for key in keys}
    A.close(ledger['totals'], summed)
    assert summed['student_fits'] == 8 and summed['optimizer_updates'] == 2560 and summed['selector_fits'] == 1
    assert summed['GPU_training_seconds'] <= 21600
    assert len(ledger['new_events']) == len({e['event'] for e in ledger['new_events']}) == 3
    assert ledger['caps'] == dict(student_fits=10, selector_fits=1, GPU_training_seconds=21600)
    fit = verifier.read(C.DOC/'SELECTOR_CALIBRATION_FIT.json')
    selector = next(e for e in ledger['new_events'] if e['selector_fits'])
    assert selector['details']['fit'] == C.bind(C.DOC/'SELECTOR_CALIBRATION_FIT.json')
    assert selector['details']['selector_optimizer_steps'] == fit['optimizer_steps'] == 416
    assert selector['optimizer_updates'] == 0
    A.close(selector['GPU_training_seconds'], fit['seconds'])
    for target in ('RAW', 'REF'):
        path = C.DOC/f'training/CLEAR_S43/FIT_CLEAN_{target}_CLEAR_S43.json'
        row = verifier.read(path); verifier.nested(row)
        assert row['complete'] and row['optimizer_steps'] == 320 and row['seed'] == 43
        event = next(e for e in ledger['new_events'] if e['event'] == f'CLEAR_S43::FIT_CLEAN_{target}_CLEAR_S43')
        assert event['student_fits'] == 1 and event['optimizer_updates'] == 320
        A.close(event['GPU_training_seconds'], row['seconds'])
    return ledger


def markdown_links(path, allowed_missing=()):
    text = path.read_text()
    targets = re.findall(r'!?\[[^\]]*\]\(([^)]+)\)', text)
    checked = 0
    for target in targets:
        target = target.strip('<>').split('#')[0]
        if not target or target.startswith(('https://', 'http://', 'mailto:')):
            continue
        item = (path.parent/target).resolve()
        if item.name in allowed_missing:
            continue
        assert item.is_file(), (path, target)
        checked += 1
    return checked


def run():
    missing = missing_inputs(C.DOC)
    if missing:
        print('FINAL_AUDIT_PENDING', missing, flush=True)
        return dict(status='PENDING', passed=False, missing=missing)
    output_path = C.DOC/'FINAL_AUDIT.json'
    if output_path.exists():
        result = C.read(output_path)
        for binding in result['sources']:
            C.verify(binding)
        assert result['passed']; print('FINAL_AUDIT_ALREADY_PASSED', flush=True); return result
    from scripts.research.pallet_clean_to_pose_transfer_v1 import eval_student as E
    from scripts.research.pallet_clean_to_pose_transfer_v1.old_reference import scalar_pose_recheck
    from . import historical_raw as H
    verifier = Verifier()
    result = verifier.read(C.DOC/'FINAL_RESULTS.json')
    robust = verifier.read(C.DOC/'FINAL_ROBUSTNESS.json')
    verifier.nested(result); verifier.nested(robust)
    metrics = verifier.read(C.RAW/'FINAL_FRAME_METRICS_PRIVATE.json')
    poses = verifier.read(C.RAW/'FINAL_POSES_PRIVATE.json')
    association = verifier.read(C.RAW/'FINAL_CASE_BINDINGS_PRIVATE.json')
    rows = verifier.read(C.OLD.RAW/'evaluation/S42/METADATA.json')
    groups = E.group_ids(rows)
    prefixes = ('R0', 'OLD_RAW', 'OLD_REF') + tuple(f'{t}_{c}_S{s}' for t in ('RAW', 'REF') for c in ('CLEAR', 'OCC') for s in (42, 43))
    arms = {f'{p}_{selector}' for p in prefixes for selector in ('D9', 'GEO', 'NEWGEO')}
    assert set(metrics) == set(poses) == set(association) == set(result['available_models']) == arms
    assert len(arms) == 33 and all(set(v) == set(groups['FULL128']) for v in metrics.values())
    assert 'same R0 weights' in result['repeat_scope'] and 'not independent initializations' in result['repeat_scope']
    assert result['no_oracle_selection'] and result['no_best_seed_or_selector_mix'] and result['same_evaluation']
    for name in ('CONTROL_AUDIT.json', 'CALIBRATION_INTEGRITY_AUDIT.json', 'CLEAR43_EVALUATION_AUDIT.json'):
        sub = verifier.read(C.DOC/name); assert sub['passed']; verifier.nested(sub)
    calibration = verifier.read(C.DOC/'CALIBRATION_INTEGRITY_AUDIT.json')
    assert (calibration['new_path_inherited_manual_images'], calibration['new_path_inherited_manual_corners']) == (9, 38)
    assert (calibration['cumulative_research_manual_images'], calibration['cumulative_research_manual_corners']) == (19, 86)
    old = verifier.read(C.RAW/'CONTROL_FRAME_METRICS_PRIVATE.json')
    current = verifier.read(C.RAW/'CURRENT_GEO_FRAME_METRICS_PRIVATE.json')
    clear = verifier.read(C.RAW/'CLEAR43_FRAME_METRICS_PRIVATE.json')
    for source in (old, current, clear):
        assert all(metrics[a] == v for a, v in source.items())
    checked_candidates = 0
    verifier.nested(association)
    for arm, bindings in association.items():
        assert verifier.read(C.ROOT/bindings['metadata']['path']) == rows
        candidate_path = C.ROOT/bindings['candidates']['path']
        candidates = verifier.read(candidate_path)[bindings['candidate_arm']]
        if bindings.get('decisions'):
            choices = verifier.read(C.ROOT/bindings['decisions']['path'])
            if bindings.get('decisions_arm'):
                choices = choices[bindings['decisions_arm']]
        else:
            choices = None
        if (candidate_path.parent/'ORACLE_METRICS_SELECTIONS_PRIVATE.json').is_file():
            cached = verifier.read(candidate_path.parent/'ORACLE_METRICS_SELECTIONS_PRIVATE.json')['candidate_metrics'][bindings['candidate_arm']]
        else:
            assert candidate_path.parent.name == 'evaluation_clear_S43'
            cached = verifier.read(C.RAW/'CLEAR43_CANDIDATE_METRICS_PRIVATE.json')[bindings['candidate_arm']]
        for fid in groups['FULL128']:
            record = candidates[fid]
            name = choices[fid]['selected'] if choices else record['selected_name']
            expected_pose = A.selected_pose(record, name) if choices else record['current']
            A.close(poses[arm][fid], expected_pose)
            found = [v['metric'] for v in cached[fid] if v['name'] == name]
            assert len(found) == 1
            A.close(metrics[arm][fid], found[0])
            checked_candidates += 1
    # 이전 RAW의 두 선택기는 기존30팔 감사가 다루지 않았으므로 여기서 확인한다.
    import torch
    from scripts.research.pallet_selector_recovery_v1 import models as SM, common as SC
    historical_lock = H.verify(); verifier.nested(historical_lock)
    assert historical_lock['read_guard_active']
    assert not any(any(token in path for token in A.FORBIDDEN_READS) for path in historical_lock['read_paths'])
    history_features = verifier.read(H.PRIVATE/'FEATURES.json')
    decisions = verifier.read(H.PRIVATE/'DECISIONS.json')
    torch.set_num_threads(2)
    historical_scores = 0
    for selector, binding in historical_lock['scorers'].items():
        checkpoint = torch.load(C.ROOT/binding['path'], map_location='cpu', weights_only=False)
        for fid, feature in history_features.items():
            if not feature['valid']:
                continue
            scores = SM.scores(checkpoint, np.asarray(feature['features'], np.float32)[None])[0]
            row = decisions['OLD_RAW_'+selector][fid]
            np.testing.assert_allclose(scores, row['scores'], atol=1e-7, rtol=1e-7)
            assert row['selected'] == SC.HYP[min(range(len(scores)), key=lambda i: (float(scores[i]), SC.HYP[i]))]
            historical_scores += 1
    from scripts.research.pallet_clean19_pose_mismatch_v1 import diagnose as D
    _, truth = D.Pose.metadata('REAL_DEV')
    for arm in arms:
        for fid in groups['FULL128']:
            scalar_pose_recheck(metrics[arm][fid], poses[arm][fid], truth[fid])
    blocks = pairs = 0
    for group, ids in groups.items():
        for arm in arms:
            A.close(A.summary(metrics[arm][fid] for fid in ids), result['groups'][group][arm]); blocks += 1
        for name, value in result['paired'][group].items():
            after, before = name.split('-minus-')
            A.close(A.paired(metrics[before], metrics[after], ids), value); pairs += 1
    natural = [r for r in rows if r['severity'] != 'CLEAN']
    cluster_checks = 0
    for name, comparison in robust['comparisons'].items():
        before, after = comparison['before'], comparison['after']
        assert name == after+'-minus-'+before
        A.close(comparison['NATURAL99'], result['paired']['NATURAL99'][name])
        expected = bootstrap(metrics[before], metrics[after], natural,
            repeats=comparison['cluster_bootstrap']['repeats'], seed=comparison['cluster_bootstrap']['seed'])
        for key, value in expected.items():
            A.close(value, comparison['cluster_bootstrap'][key])
        for rec, value in comparison['leave_one_recording_out'].items():
            ids = [r['id'] for r in natural if r['recording'] != rec]
            A.close(A.paired(metrics[before], metrics[after], ids), value)
        cluster_checks += 1
    start = verifier.read(C.DOC/'START.json')
    verifier.nested(start['immutable_inputs']); verifier.binding(start['user_changes_preserved'])
    changed = subprocess.check_output(['git', 'diff', '--name-only', 'HEAD', '--',
        str(C.OLD.DOC.relative_to(C.ROOT)), str(Path(E.__file__).parent.relative_to(C.ROOT))], cwd=C.ROOT, text=True).strip()
    assert not changed
    ledger = current_cost(verifier)
    protocol = verifier.read(C.DOC/'training/CLEAR_S43/PRIMARY_PROTOCOL.json')
    verifier.nested(protocol)
    pair = verifier.read(C.DOC/'training/CLEAR_S43/PAIR_INTEGRITY_S43.json'); verifier.nested(pair)
    assert pair['passed'] and pair['batches'] == 320 and pair['coordinate_different_batches'] > 0
    for arm in ('CLEAN_RAW_CLEAR', 'CLEAN_REF_CLEAR'):
        assert pair['states'][arm]['protected_exact'] == 747 and pair['states'][arm]['checkpoint_states'] == 879
        assert all(value > 0 for value in pair['actual_seed42_vs43'][arm].values())
        assert pair['same_seed_clear_occ'][arm]['passed'] and pair['same_seed_clear_occ'][arm]['batches'] == 320
    snapshot = verifier.read(C.ROOT/protocol['resource_snapshot']['path'])
    assert snapshot['totals']['student_fits'] == 6 and snapshot['totals']['selector_fits'] == 1
    technical = verifier.read(C.DOC/'CLEAR43_INFERENCE_TECHNICAL_CORRECTION.json'); verifier.nested(technical)
    assert technical['new_student_fits'] == technical['new_selector_fits'] == technical['optimizer_updates'] == technical['failed_prediction_outputs'] == 0
    cross = verifier.read(C.DOC/'TRAIN_CROSS_INPUT_RESULTS.json'); verifier.nested(cross)
    cross_protocol = verifier.read(C.DOC/'TRAIN_CROSS_INPUT_PROTOCOL.json'); verifier.nested(cross_protocol)
    cross_lock = verifier.read(C.DOC/'TRAIN_CROSS_INPUT_PREDICTIONS_LOCK.json'); verifier.nested(cross_lock)
    assert cross['status'] == 'COMPLETE' and len(cross['groups']) == 8
    assert cross['new_fits'] == cross['optimizer_updates'] == 0
    assert cross_protocol['protocol_locked_before_new_outputs'] and cross_protocol['common_RGB_targets_support_exact']
    assert cross_protocol['evaluation_refs_read'] is False and cross_lock['evaluation_refs_read'] is False
    assert cross_protocol['K_batches'] == 8 and cross_protocol['input_seed'] == 42
    assert cross_lock['model_contexts'] == 16 and cross_lock['new_fits'] == cross_lock['optimizer_updates'] == 0
    assert cross_lock['inputs_same_across_models'] and cross_lock['detector_parity_all_models']
    assert cross_lock['optimizer_constructed'] is False and cross_lock['model_weights_not_written']
    assert all(s['state_unchanged'] and s['gradients_absent'] for s in cross_lock['states'].values())
    assert datetime.fromisoformat(cross_protocol['created_at']) < datetime.fromisoformat(cross_lock['created_at']) < datetime.fromisoformat(cross['created_at'])
    assert (cross['real_occurrences'], cross['real_unique'], cross['common_supervised_points'], cross['canonical_covered']) == (62, 45, 476, 21)
    for row in cross['own_target_comparison']:
        for name, count in (('ALL_SUPERVISED', 476), ('CANONICAL_REF_PLANNED_COVERED', 21), ('CANONICAL_REF_UNMASKED', 455)):
            paired = row['paired'][name]
            assert paired['point_occurrences'] == count
            assert paired['improved'] + paired['worsened'] + paired['equal'] == paired['common_observed']
            A.close(paired['full_missing_penalty_delta_mean_px'],
                row['OCCtrained'][name]['L2_missing_diagonal_penalty_mean_px'] - row['CLEARtrained'][name]['L2_missing_diagonal_penalty_mean_px'])
    cases = verifier.read(C.DOC/'CASES_FINAL.json'); verifier.nested(cases)
    from . import cases as CASES
    approved = CASES.approved_bindings(C.read(CASES.PAPER), C.read(CASES.SPLIT)['heldout'], C.read(CASES.RETAINED))
    for figure in cases['figures']:
        CASES.check_public_image(figure['id'], figure['image'], approved)
    build = verifier.read(C.DOC/'REPORT_BUILD.json'); verifier.nested(build)
    decision = verifier.read(C.DOC/'FINAL_DECISION.json')
    assert decision['resource_totals'] == ledger['totals']
    for before, after in decision['key_pairs']:
        assert after+'-minus-'+before in result['paired']['NATURAL99']
    public_count = 0
    for name in REQUIRED:
        path = C.DOC/name
        if path.suffix == '.json':
            privacy(verifier.read(path)); public_count += 1
    links = sum(markdown_links(C.DOC/name, ('FINAL_AUDIT_KO.md',)) for name in ('REPORT_KO.md', 'CASES_FINAL.md', 'REPRODUCE.md'))
    tests = test_count(C.DOC/'TEST_FINAL.xml')
    checklist = dict(same33_configs_full128_and_natural99=True, selected_candidates_and_cached_metric_parity=True,
        matched217_RAW_REF_control_present=True, all_summary_and_paired_rows_recomputed=True,
        recording_bootstrap_not_independent_frames=True, actual_CLEAR_OCC42_43_scope=True,
        same_R0_not_independent_initialization=True, current_and_old_frozen_selector_shared=True,
        new8_cumulative_student_fits_and_one_selector_budget=True, historical_final_results_preserved=True,
        user_dirty_annotation_progress_unchanged=True, technical_inference_retry_not_fit_or_score_retry=True,
        historical19_86_supervision_scope_preserved=True, train_cross_input_locked_no_fit_diagnostic=True,
        originalRGB_and_coordinates_not_published=True,
        public_examples_restricted_to_approved_hashes=True, report_links_and_tests_pass=True)
    sources = [C.bind(C.DOC/name) for name in REQUIRED]
    sources += [C.bind(p) for p in (C.DOC/'RESOURCE_LEDGER.json', C.DOC/'training/CLEAR_S43/PAIR_INTEGRITY_S43.json',
        C.DOC/'CLEAR43_INFERENCE_TECHNICAL_CORRECTION.json', Path(__file__))]
    output = dict(status='PASS', passed=True, created_at=C.now(), checklist=checklist,
        configuration_count=33, selected_pose_and_metric_checks=checked_candidates,
        direct_T_C2_rotation_yaw_recomputed=checked_candidates, historical_scorer_CPU_checks=historical_scores,
        summary_blocks=blocks, paired_blocks=pairs, bootstrap_comparisons=cluster_checks,
        input_bindings_checked=len(verifier.seen), public_payloads_checked=public_count,
        historical_ledger_verified_via_immutable_snapshots=sorted(verifier.historical_ledger_snapshots),
        previously_approved_public_figures=len(cases['figures']), local_links_checked=links,
        resource_totals=ledger['totals'], tests=tests, sources=sources,
        new_audit_fits=0, new_audit_GPU_seconds=0,
        limits=['초기879 tensor 동일성은 실제 runtime assertion/원checkpoint 해시 근거이며 존재하지 않는 초기 스냅샷을 만들지 않았다.',
                '전체320 입력 parity가 존재하는 것은 이번 clean78 반복이다. 과거217 first-batch/설정 감사를 전체 tensor 전수 일치라고 쓰지 않는다.',
                'selector416 optimizer step은 코드의32 batch×13epoch 검산이며 독립 optimizer hook 추적이 아니다.',
                '33개 결과는 reusedDEV/geometry-derived6D이다. 독립 초기화·독립 TEST·physical6D 검증 완료가 아니다.',
                'T/R/yaw 원식과 후보 metric/요약을 검산했다. 모든 개별 IoU3D 교차 기하를 다시 계산한 것은 아니다.'])
    C.save(output_path, output, True)
    lines = ['# 최종 독립 감사', '', '**PASS**', '',
        f'- 33개 고정 구성 × 128장 = {checked_candidates:,}개 선택 자세·metric 및 T/C2 R/yaw 원식 확인.',
        f'- summary {blocks}개, paired {pairs}개, recording-bootstrap {cluster_checks}비교를 독립 재계산.',
        '- CLEAR/OCC × RAW/REF의 실제42/43 입력 반복이 존재한다. 초기 checkpoint는 같은R0이며 독립 초기화가 아니다.',
        '- 이전217 RAW 대조, 두 frozen 선택기, 과거 결과, 사용자 수정 파일 SHA를 모두 확인했다.',
        f'- 누적 학생8회/2,560update·선택기1회(별도416step)·GPU 학습{ledger["totals"]["GPU_training_seconds"]:.3f}초. 예산10회/1회/6시간 이내.',
        f'- tests {tests["passed"]} passed, {tests["skipped"]} skipped; 승인 이미지 {len(cases["figures"])}개와 문서 링크 {links}개 확인.', '',
        '## 감사가 의미하지 않는 것', '', *['- '+v for v in output['limits']], '',
        '[체크리스트·출처·상세 결과](FINAL_AUDIT.json)', '']
    C.save(C.DOC/'FINAL_AUDIT_KO.md', '\n'.join(lines), True)
    print('FINAL_AUDIT_PASS', checked_candidates, blocks, pairs, cluster_checks, tests, flush=True)
    return output


if __name__ == '__main__':
    run()
