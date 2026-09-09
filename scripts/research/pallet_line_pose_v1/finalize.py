"""Audit final artifacts, show the HTML and send the authorized Discord verdict."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time

from driver import read, sha, write

HERE = Path(__file__).resolve().parent
TITLE = 'Pallet Line Pose · YOLO26n 검증'


def confirmed_window():
    """Inspect only the requested title; keep unrelated desktop titles private."""
    for _ in range(12):
        try:
            windows = subprocess.run(['xprop', '-root', '_NET_CLIENT_LIST_STACKING'],
                capture_output=True, text=True, timeout=5).stdout
            for window in re.findall(r'0x[0-9a-fA-F]+', windows):
                title = subprocess.run(['xprop', '-id', window, '_NET_WM_NAME'],
                    capture_output=True, text=True, timeout=5).stdout
                if TITLE not in title:
                    continue
                info = subprocess.run(['xwininfo', '-id', window], capture_output=True,
                    text=True, timeout=5).stdout
                state = subprocess.run(['xprop', '-id', window, '_NET_WM_STATE'],
                    capture_output=True, text=True, timeout=5).stdout
                if 'Map State: IsViewable' in info and '_NET_WM_STATE_HIDDEN' not in state:
                    return window
        except (OSError, subprocess.TimeoutExpired):
            return None
        time.sleep(.5)
    return None


def show_page(page):
    launcher = next((shutil.which(name) for name in ('google-chrome', 'google-chrome-stable',
                    'chromium', 'chromium-browser', 'firefox') if shutil.which(name)), None)
    receipt = dict(html=str(page), html_sha256=sha(page), window_visibility_confirmed=False)
    if launcher:
        flag = '-new-window' if Path(launcher).name == 'firefox' else '--new-window'
        with (page.parent/'GALLERY_OPEN.log').open('a') as handle:
            process = subprocess.Popen([launcher, flag, page.as_uri()], stdin=subprocess.DEVNULL,
                stdout=handle, stderr=subprocess.STDOUT, start_new_session=True)
        receipt.update(launcher=launcher, pid=process.pid, status='launch_requested')
        window = confirmed_window()
        if window:
            receipt.update(status='window_confirmed', window_id=window,
                           title=TITLE, window_visibility_confirmed=True)
    else:
        receipt['status'] = 'browser_not_found'
    write(page.parent/'GALLERY_OPEN.json', receipt)
    return receipt


def verify_marker(path):
    data = read(path)
    if data.get('complete') is not True or data.get('PASS', True) is not True:
        raise ValueError(f'Incomplete experiment artifact: {path}')
    for key in ('source_sha256', 'input_sha256', 'output_sha256'):
        for file, expected in data.get(key, {}).items():
            location = Path(file)
            if not location.is_absolute():
                location = path.parent/location
            if sha(location) != expected:
                raise ValueError(f'Artifact source/output mismatch: {location}')
    return data


def finalize(run_dir):
    run_dir = Path(run_dir).resolve()
    protocol = verify_marker(run_dir/'TRAIN_PROTOCOL.json')
    selection = verify_marker(run_dir/'SELECTION.json')
    if not selection.get('no_real_selection') or selection.get('selection_population') != 'synth_val':
        raise ValueError('Selection was not synthetic-only')
    markers = ['DECISION_PROTOCOL.json', 'cache/CACHE_COMPLETE.json', 'LOGITS_MANIFEST.json', 'SELECTION.json',
               'SYNTHETIC_EVALUATION.json', 'REAL_EVALUATION_COMPLETE.json',
               'SUMMARY.json', 'VERDICT.json', 'REPORT_RENDER.json']
    for name in markers:
        verify_marker(run_dir/name)
    real = read(run_dir/'REAL_EVALUATION_COMPLETE.json')
    if (real['selection_sha256'] != sha(run_dir/'SELECTION.json')
            or real['baseline_cache_sha256'] != sha(run_dir/'baseline/FULL_CANDIDATES.json')
            or real['expected_runs'] != 9 or len(real['runs']) != 9
            or real['real_selection_performed'] or not real['negative_outputs_preserved']):
        raise ValueError('Real evaluation does not bind the frozen selection and population')
    runtime_path = Path(real['runtime']['path'])
    if not runtime_path.is_absolute():
        runtime_path = run_dir/runtime_path
    runtime = verify_marker(runtime_path)
    if (sha(runtime_path) != real['runtime']['sha256'] or not runtime['parity_PASS']
            or runtime['bindings']['selection_sha256'] != sha(run_dir/'SELECTION.json')
            or runtime['bindings']['baseline_cache_sha256'] != real['baseline_cache_sha256']):
        raise ValueError('Actual runtime result or accuracy parity changed')
    markers.append(str(runtime_path.relative_to(run_dir)))
    runs = []
    for seed in protocol['seeds']:
        for arm in protocol['arms']:
            folder = run_dir/'runs'/f'{arm}_seed{seed}'
            completed = verify_marker(folder/'COMPLETION.json')
            if completed['smoke'] or completed['step'] != protocol['steps']:
                raise ValueError('A smoke/partial model entered the final comparison')
            if sha(folder/'last.pt') != completed['checkpoint_sha256']:
                raise ValueError('A final model changed after training')
            evaluation_path = run_dir/'evaluation'/f'{arm}_seed{seed}'/'COMPLETION.json'
            verify_marker(evaluation_path)
            corresponding = [r for r in real['runs'] if (r['arm'], r['seed']) == (arm, seed)]
            if (len(corresponding) != 1
                    or corresponding[0]['completion_sha256'] != sha(evaluation_path)
                    or corresponding[0]['checkpoint_sha256'] != completed['checkpoint_sha256']):
                raise ValueError('A real evaluation is not bound to its completed model')
            runs.append(dict(arm=arm, seed=seed, step=completed['step'],
                             checkpoint_sha256=completed['checkpoint_sha256']))
            markers.extend([f'runs/{arm}_seed{seed}/COMPLETION.json',
                            f'evaluation/{arm}_seed{seed}/COMPLETION.json'])
    render = read(run_dir/'REPORT_RENDER.json')
    if render.get('experiment_complete') is not True or render.get('n_completed_evaluations') != 9:
        raise ValueError('The report still contains an incomplete experiment scaffold')
    visual = read(run_dir/'VISUAL_QA.json')
    if (visual.get('PASS') is not True or visual.get('experiment_complete') is not True
            or visual.get('html_sha256') != render['html_sha256']
            or visual.get('broken_local_images') != 0):
        raise ValueError('The completed report failed automatic document checks')
    markers.append('VISUAL_QA.json')
    page = Path(render.get('html', run_dir/'index.html'))
    if not page.is_absolute():
        page = run_dir/page
    if sha(page) != render['html_sha256']:
        raise ValueError('HTML differs from the verified renderer output')
    if TITLE not in page.read_text():
        raise ValueError('Unexpected report title')
    opened = show_page(page)
    verdict = read(run_dir/'VERDICT.json')
    message = ['팔레트 점·선 아키텍처의 학습·논문 기준 비교 검증이 완료됐습니다.',
               verdict['headline_ko']]
    message += verdict.get('discord_lines_ko', [])
    message += ['실사319장·negative2689장, 3개 구조×3개 seed를 비교했습니다. '
                '반복 사용한 DEV 평가이며 독립 최종 테스트는 아닙니다.',
                f'시각화 HTML(작업 PC): {page}',
                'Chrome 보고서 창 표시를 확인했습니다.' if opened['window_visibility_confirmed']
                else '브라우저 창 표시 여부는 확인되지 않았습니다.']
    sys.path.insert(0, str(HERE.parent))
    from discord_notify import notify
    delivered = notify('\n'.join(message), run_dir, evidence=run_dir/'VERDICT.json')
    completion = dict(complete=True, PASS=True, completed_at_utc=datetime.now(timezone.utc).isoformat(),
        scope='Completed training/evaluation/artifact verification, not automatically a claim of accuracy improvement.',
        runs=runs, real_dev_frames=319, negative_frames=2689,
        overall_accuracy_improved=verdict['overall_accuracy_improved'],
        headline_ko=verdict['headline_ko'], report=str(page), html_sha256=sha(page),
        browser_window_confirmed=opened['window_visibility_confirmed'],
        discord_status=delivered['status'], discord_http_status=delivered.get('http_status'),
        artifact_sha256={name:sha(run_dir/name) for name in markers})
    write(run_dir/'COMPLETION.json', completion)
    if delivered['status'] != 'sent':
        raise RuntimeError('Experiment complete; Discord delivery not yet confirmed')
    print(json.dumps(dict(complete=True, PASS=True, verdict=verdict['headline_ko'],
                         browser_visible=opened['window_visibility_confirmed'], report=str(page)), ensure_ascii=False))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path, required=True)
    finalize(parser.parse_args().run_dir)
