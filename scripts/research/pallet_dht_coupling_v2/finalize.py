"""Verify completed comparison, visibly open its report, deliver requested notice."""
from __future__ import annotations

import argparse
from pathlib import Path
import re
import shutil
import subprocess
import time

from .prepare import read, write, sha, check
from .driver import audit_training, verify_bindings
from scripts.research.pallet_dht_joint_v1.driver import now

TITLE = 'Pallet DHT Coupling · 점·선 결합 학습 검증'


def show_page(page):
    launcher = next((shutil.which(name) for name in
        ('google-chrome', 'google-chrome-stable', 'chromium', 'chromium-browser', 'firefox')
        if shutil.which(name)), None)
    receipt = dict(html=str(page), html_sha256=sha(page), window_visibility_confirmed=False)
    if launcher:
        with (page.parent / 'GALLERY_OPEN.log').open('a') as stream:
            child = subprocess.Popen([launcher, '-new-window' if Path(launcher).name == 'firefox' else '--new-window', page.as_uri()],
                stdin=subprocess.DEVNULL, stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
        receipt.update(status='launch_requested', pid=child.pid, launcher=launcher)
        for _ in range(12):
            try:
                windows = subprocess.run(['xprop', '-root', '_NET_CLIENT_LIST_STACKING'], capture_output=True, text=True, timeout=5).stdout
                for window in re.findall(r'0x[0-9a-fA-F]+', windows):
                    title = subprocess.run(['xprop', '-id', window, '_NET_WM_NAME'], capture_output=True, text=True, timeout=5).stdout
                    if TITLE not in title:
                        continue
                    info = subprocess.run(['xwininfo', '-id', window], capture_output=True, text=True, timeout=5).stdout
                    state = subprocess.run(['xprop', '-id', window, '_NET_WM_STATE'], capture_output=True, text=True, timeout=5).stdout
                    if 'Map State: IsViewable' in info and '_NET_WM_STATE_HIDDEN' not in state:
                        receipt.update(status='window_confirmed', window_visibility_confirmed=True,
                                       window_id=window, title=TITLE)
                        break
            except (OSError, subprocess.TimeoutExpired):
                break
            if receipt['window_visibility_confirmed']:
                break
            time.sleep(.5)
    else:
        receipt['status'] = 'browser_not_found'
    write(page.parent / 'GALLERY_OPEN.json', receipt)
    return receipt


def marker(path, runtime=False):
    value = read(path)
    recorded_failure = (runtime and Path(path).name == 'RUNTIME.json'
        and value.get('PASS') is False and value.get('parity_PASS') is False
        and value.get('timing_collection_complete') is True
        and value.get('status') == 'COMPLETE_WITH_STRICT_PARITY_FAILURE'
        and len(value.get('strict_failures', [])) > 0
        and value.get('parity_policy') == dict(atol=1e-4, rtol=0, criterion_changed=False))
    check(value.get('complete') is True and (value.get('PASS', True) is True or recorded_failure),
          f'Incomplete artifact: {path}')
    verify_bindings(value, Path(path).parent)
    for filename, digest in value.get('source_results', {}).items():
        check(sha(filename) == digest, f'Changed diagnostic result: {filename}')
    return value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path, required=True)
    root = parser.parse_args().run_dir.resolve()
    protocol = read(root / 'TRAIN_PROTOCOL.json')
    audit_training(root, protocol)
    names = list(protocol['preflight_required']) + [
        'COMPLETION_CHAIN_BINDING.json', 'TRAINING_AUDIT.json', 'MATCHED_CONTROL_AUDIT.json',
        'RUNTIME.json', 'SUMMARY.json', 'VERDICT.json', 'AGGREGATE_COMPLETE.json',
        'REPORT_RENDER.json', 'INDEPENDENT_FINAL_AUDIT.json', 'ACTUAL_VISUAL_QA.json', 'VISUAL_QA.json', 'VISUAL_REVIEW.json']
    for name in names:
        marker(root / name, runtime=name == 'RUNTIME.json')
    runtime = read(root / 'RUNTIME.json')
    independent = read(root / 'INDEPENDENT_FINAL_AUDIT.json')['runtime_audit']
    check(runtime['n_models'] == 21 and runtime['n_timing_observations'] == 1638,
          'Runtime model or observation count differs')
    check(runtime['timing_collection_complete'] and independent['observed_timing_count'] == 1638
          and independent['strict_parity_PASS'] is runtime['parity_PASS']
          and independent['strict_failure_count'] == len(runtime['strict_failures'])
          and independent['criterion_changed'] is False,
          'Runtime status lacks independent verification')
    evaluations = []
    for seed in protocol['training']['seeds']:
        for arm in protocol['arms']:
            relative = f'evaluation/{arm}_seed{seed}/COMPLETION.json'
            evaluated = marker(root / relative)
            check(evaluated['actual_positive_forwards'] == 319 and evaluated['actual_negative_forwards'] == 2689
                  and evaluated['baseline_candidate_copying'] is False, 'Actual new inference count differs')
            trained = marker(root / 'runs' / f'{arm}_seed{seed}' / 'COMPLETION.json')
            check(evaluated['checkpoint_sha256'] == trained['checkpoint_sha256'], 'Wrong evaluation checkpoint')
            evaluations.append(dict(arm=arm, seed=seed, checkpoint_sha256=evaluated['checkpoint_sha256']))
            names.append(relative)
    check(len(evaluations) == 12, 'Expected all twelve new cells')
    references = read(root / 'REUSED_CONTROLS.json')
    check(len(references['controls']) == references['n_control_cells'] == 9, 'Expected nine verified historical controls')
    page = root / 'index.html'
    render = read(root / 'REPORT_RENDER.json')
    qa = read(root / 'ACTUAL_VISUAL_QA.json')
    for value in (render, qa):
        check(value.get('experiment_complete') and value.get('n_completed_evaluations') == 12
              and value.get('n_verified_reference_evaluations') == 9
              and value.get('n_gallery_model_runs') == 21 and value.get('n_gallery_frames') == 319
              and value['html_sha256'] == sha(page), 'Incomplete or changed actual report')
    check(TITLE in page.read_text(), 'Unexpected report title')
    static_qa = read(root / 'VISUAL_QA.json')
    check(static_qa.get('html_sha256') == sha(page) and static_qa.get('broken_local_images') == 0,
          'Static HTML QA differs')
    review = read(root / 'VISUAL_REVIEW.json')
    check(review.get('html_sha256') == sha(page) and review.get('actual_screenshots_inspected') is True,
          'Actual screenshot review must bind the current report')
    verdict = read(root / 'VERDICT.json')
    check(verdict.get('selected_winning_arm') is None and verdict.get('real_model_selection_performed') is False,
          'Undeclared model selection')
    opened = show_page(page)
    check(opened['window_visibility_confirmed'], 'Report finished, but browser window was not visibly confirmed')
    from scripts.research.discord_notify import notify
    message = '\n'.join([
        '팔레트 점·Deep Hough 결합의 gradient 진단과 비교 실험이 완료됐습니다.',
        verdict['headline_ko'], *verdict.get('discord_lines_ko', []),
        '손실 비중 조정 / PCGrad / 두 방법 결합 / 점–선 관계 손실: 4방법×3seed를 합성55,980장으로 각각2epochs 학습했습니다.',
        '동일 예산의 이전9모델을 무결성 검증 후 대조군으로 재사용했습니다. 신규12모델은 각각 실사319장과 negative2689장을 새로 추론했습니다.',
        '재사용 DEV 평가이며 독립 FINAL 검증은 아닙니다.',
        ('속도 측정의 예측 일치 검사(atol1e-4)는 통과했습니다.' if runtime['parity_PASS'] else
         f"속도는 참고 실측값입니다. 원래 예측 일치 검사(atol1e-4)는 {len(runtime['strict_failures'])}/1638회 실패했으며 그대로 보고했습니다."),
        f'시각화 HTML(작업 PC): {page}', '브라우저 보고서 창 표시를 확인했습니다.'])
    delivered = notify(message, root, evidence=root / 'VERDICT.json')
    check(delivered['status'] == 'sent' and delivered.get('http_status') in (200, 204),
          'Research finished; Discord delivery not confirmed')
    names.extend(['GALLERY_OPEN.json', 'DISCORD_NOTIFICATION.json'])
    done = dict(complete=True, PASS=True, completed_at_utc=now(), runs=evaluations,
        n_new_trained_and_evaluated_models=12, n_reused_control_models=9,
        scope='Completed research execution and honest reporting; PASS does not imply an accuracy gain or every numerical check passing.',
        all_checks_passed=runtime['parity_PASS'], runtime_strict_parity_PASS=runtime['parity_PASS'],
        runtime_strict_failure_count=len(runtime['strict_failures']),
        headline_ko=verdict['headline_ko'], overall_accuracy_improved=verdict['overall_accuracy_improved'],
        report=str(page), html_sha256=sha(page), browser_window_confirmed=True,
        discord_status=delivered['status'], discord_http_status=delivered['http_status'],
        artifact_sha256={name:sha(root / name) for name in names},
        source_sha256={str(Path(__file__).resolve()):sha(__file__)})
    write(root / 'COMPLETION.json', done)
    print(verdict['headline_ko'], flush=True)


if __name__ == '__main__':
    main()
