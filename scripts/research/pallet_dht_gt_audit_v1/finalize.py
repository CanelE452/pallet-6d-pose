"""Verify the completed read-only audit, visibly open it, and notify the owner."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import shutil
import subprocess
import time


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def write(path, data):
    Path(path).write_text(json.dumps(data, ensure_ascii=False, indent=2) + '\n')


def open_report(page, title):
    launcher = next((shutil.which(x) for x in
                     ('google-chrome', 'google-chrome-stable', 'chromium', 'firefox')
                     if shutil.which(x)), None)
    assert launcher, 'No desktop browser launcher available'
    with (page.parent / 'GALLERY_OPEN.log').open('a') as log:
        process = subprocess.Popen(
            [launcher, '-new-window' if Path(launcher).name == 'firefox' else '--new-window', page.as_uri()],
            stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    receipt = dict(html=str(page), html_sha256=sha(page), title=title,
                   launcher=launcher, pid=process.pid, window_visibility_confirmed=False)
    for _ in range(20):
        windows = subprocess.run(['xprop', '-root', '_NET_CLIENT_LIST_STACKING'],
                                 capture_output=True, text=True, timeout=5).stdout
        for window in re.findall(r'0x[0-9a-fA-F]+', windows):
            name = subprocess.run(['xprop', '-id', window, '_NET_WM_NAME'],
                                  capture_output=True, text=True, timeout=5).stdout
            if title not in name:
                continue
            info = subprocess.run(['xwininfo', '-id', window],
                                  capture_output=True, text=True, timeout=5).stdout
            state = subprocess.run(['xprop', '-id', window, '_NET_WM_STATE'],
                                   capture_output=True, text=True, timeout=5).stdout
            if 'Map State: IsViewable' in info and '_NET_WM_STATE_HIDDEN' not in state:
                receipt.update(window_visibility_confirmed=True, window_id=window,
                               status='window_confirmed')
                break
        if receipt['window_visibility_confirmed']:
            break
        time.sleep(.5)
    write(page.parent / 'GALLERY_OPEN.json', receipt)
    assert receipt['window_visibility_confirmed'], 'Visible report window not confirmed'
    return receipt


def finalize(root):
    root = root.resolve()
    assert not (root / 'COMPLETION.json').exists(), 'Audit already delivered'
    snapshot = read(root / 'INPUT_SNAPSHOT.json')
    for path, digest in snapshot['sha256'].items():
        assert sha(path) == digest, f'Original input changed: {path}'
    numeric = read(root / 'NUMERIC_AUDIT.json')
    assert numeric['complete'] and numeric['audit_integrity_PASS']
    assert numeric['metric_reconstruction']['summary']['n_observed_points'] == 2738
    for path, digest in numeric['input_sha256'].items():
        assert sha(path) == digest, f'Numerical audit input changed: {path}'
    for name in ('INDEPENDENT_AUDIT.json', 'specialist_gt_review.json',
                 'VISUAL_QA.json', 'VISUAL_REVIEW.json', 'REPORT_RENDER.json'):
        value = read(root / name)
        assert value['complete'] and value['PASS'], f'Incomplete {name}'
    independent = read(root / 'INDEPENDENT_AUDIT.json')
    assert independent['numeric_audit_sha256'] == sha(root / 'NUMERIC_AUDIT.json')
    for path, digest in independent['input_sha256'].items():
        assert sha(path) == digest, f'Independent audit input changed: {path}'
    assert read(root / 'ROOT_GT_VISUAL_REVIEW.json')['complete']
    page = root / 'gt_review.html'
    report = read(root / 'REPORT_RENDER.json')
    for path, digest in report['input_sha256'].items():
        assert sha(path) == digest, f'Report input changed: {path}'
    for name in ('REPORT_RENDER.json', 'VISUAL_QA.json', 'VISUAL_REVIEW.json'):
        assert read(root / name)['html_sha256'] == sha(page), f'Stale {name}'
    visual = read(root / 'VISUAL_REVIEW.json')
    assert visual['actual_screenshots_inspected']
    for path, digest in visual['screenshot_sha256'].items():
        assert sha(path) == digest, f'Screenshot changed: {path}'
    conclusion = read(root / 'AUDIT_CONCLUSION.json')
    assert conclusion['gt_corrections_applied'] == 0 and not conclusion['gt_fully_certified']
    open_report(page, report['title'])
    from scripts.research.discord_notify import notify
    message = '\n'.join([
        '팔레트 점·Hough 결합 실험의 수동 GT 감사를 완료했습니다.',
        conclusion['headline_ko'],
        '같은319장 중309장·2738점에서 P90 41.49px를 정확 재현했습니다. 신규 학습·추론·GT 수정은 없습니다.',
        '보이는 점 표시 subset P90 37.11px, 직접클릭 기록 subset28.81px. 이는 GT 수정 전후 비교가 아닙니다.',
        '아래 끝점 위치와 화면 밖인데 visible로 표시된 GT 등 재검토 후보를 표시했습니다. 큰 예측 오류도 남습니다.',
        f'GT/raw/예측 비교 HTML(작업 PC): {page}',
        '실제 브라우저 창 표시를 확인했습니다. 전체 GT의 픽셀 정확성을 인증한 결과는 아닙니다.'
    ])
    delivery = notify(message, root, evidence=root / 'AUDIT_CONCLUSION.json')
    assert delivery['status'] == 'sent' and delivery.get('http_status') in (200, 204)
    names = ['INPUT_SNAPSHOT.json', 'NUMERIC_AUDIT.json', 'NUMERIC_AUDIT.csv',
             'MASK_POLICY_DIAGNOSTIC.json', 'ROOT_GT_VISUAL_REVIEW.json',
             'specialist_gt_review.json', 'specialist_gt_review.md', 'INDEPENDENT_AUDIT.json',
             'AUDIT_CONCLUSION.json', 'GT_REVIEW_QUEUE.json', 'GT_REVIEW_DATA.json',
             'REPORT_RENDER.json', 'VISUAL_QA.json', 'VISUAL_REVIEW.json',
             'GALLERY_OPEN.json', 'DISCORD_NOTIFICATION.json', 'CONTACT_SHEETS.json',
             'root_review_details/INDEX.json', 'root_review_details/ENDPOINT_CROP.json']
    write(root / 'COMPLETION.json', dict(
        complete=True, PASS=True, completed_at_utc=datetime.now(timezone.utc).isoformat(),
        scope='Numerical/qualitative audit and delivery complete; not certification of all GT or model improvement.',
        gt_fully_certified=False, gt_corrections_applied=0, new_model_forwards=0, new_training_updates=0,
        original_inputs_preserved=len(snapshot['sha256']), report=str(page), html_sha256=sha(page),
        headline_ko=conclusion['headline_ko'], browser_window_confirmed=True,
        discord_http_status=delivery['http_status'],
        artifact_sha256={name: sha(root / name) for name in names},
        source_sha256={str(path): sha(path) for path in Path(__file__).resolve().parent.glob('*.py')}))
    print(conclusion['headline_ko'], flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path,
                        default=Path('data/pallet/results/pallet_dht_gt_audit_v1'))
    finalize(parser.parse_args().run_dir)
