"""Open the verified pilot report visibly and send the user's requested Discord notice."""
import argparse
from pathlib import Path
import re
import shutil
import subprocess
import time

from scripts.research.pallet_dht_coupling_v2.prepare import read, write, sha, check
from .freeze import verify
from .driver import now

TITLE = 'Point–Line Decoder Probe · 점·선 후보 결합'


def show_page(page):
    launcher = next((shutil.which(name) for name in
        ('google-chrome', 'google-chrome-stable', 'chromium', 'chromium-browser', 'firefox')
        if shutil.which(name)), None)
    receipt = dict(html=str(page), html_sha256=sha(page), window_visibility_confirmed=False)
    check(launcher is not None, 'No browser launcher available')
    with (page.parent / 'GALLERY_OPEN.log').open('a') as stream:
        child = subprocess.Popen([launcher, '-new-window' if Path(launcher).name == 'firefox' else '--new-window', page.as_uri()],
            stdin=subprocess.DEVNULL, stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
    receipt.update(launcher=launcher, pid=child.pid)
    for _ in range(20):
        windows = subprocess.run(['xprop', '-root', '_NET_CLIENT_LIST_STACKING'], capture_output=True, text=True, timeout=5).stdout
        for window in re.findall(r'0x[0-9a-fA-F]+', windows):
            title = subprocess.run(['xprop', '-id', window, '_NET_WM_NAME'], capture_output=True, text=True, timeout=5).stdout
            if TITLE not in title:
                continue
            info = subprocess.run(['xwininfo', '-id', window], capture_output=True, text=True, timeout=5).stdout
            state = subprocess.run(['xprop', '-id', window, '_NET_WM_STATE'], capture_output=True, text=True, timeout=5).stdout
            if 'Map State: IsViewable' in info and '_NET_WM_STATE_HIDDEN' not in state:
                receipt.update(status='window_confirmed', window_visibility_confirmed=True, window_id=window, title=TITLE)
                break
        if receipt['window_visibility_confirmed']:
            break
        time.sleep(.5)
    write(page.parent / 'GALLERY_OPEN.json', receipt)
    check(receipt['window_visibility_confirmed'], 'Report exists, but visible browser window not confirmed')
    return receipt


def finalize(root):
    root = Path(root).resolve()
    check(not (root / 'COMPLETION.json').exists(), 'Already delivered')
    verify(root)
    names = ['TRAIN_PROTOCOL.json', 'SOURCE_FREEZE.json', 'EVALUATION_SOURCE_FREEZE.json', 'PREFLIGHT.json', 'DIAGNOSTIC.json',
             'CACHE_COMPLETION.json', 'TRAINING_COMPLETION.json', 'EVALUATION_COMPLETION.json',
             'RESULTS.json', 'PREDICTIONS.json', 'REPORT_RENDER.json', 'ACTUAL_VISUAL_QA.json',
             'VISUAL_REVIEW.json', 'INDEPENDENT_AUDIT.json']
    for name in names:
        value = read(root / name)
        if name not in ('TRAIN_PROTOCOL.json', 'PREDICTIONS.json', 'RESULTS.json'):
            check(value.get('complete') is True and value.get('PASS', True) is True, f'Incomplete {name}')
    amendment_path = root / 'RENDER_VISIBILITY_AMENDMENT.json'
    amendment = read(amendment_path) if amendment_path.exists() else None
    for path, digest in read(root / 'EVALUATION_SOURCE_FREEZE.json')['source_sha256'].items():
        if amendment and path == amendment['source_path']:
            check(Path(path).name == 'report.py' and amendment['render_only'] is True
                  and amendment['original_source_sha256'] == digest
                  and amendment['revised_source_sha256'] == sha(path), 'Undeclared rendering change')
            for artifact, expected in amendment['preserved_result_sha256'].items():
                check(sha(artifact) == expected, f'Render-only fix changed measured results: {artifact}')
            for artifact, expected in amendment['archived_before_sha256'].items():
                check(sha(artifact) == expected, f'Original rendering archive differs: {artifact}')
        else:
            check(sha(path) == digest, f'Evaluation/report/audit source changed: {path}')
    if amendment:
        names.append('RENDER_VISIBILITY_AMENDMENT.json')
    for relative in ('provenance/model_smoke/RESULTS.json',
                     'provenance/additional_report_screenshots/CAPTURE_RECEIPT.json',
                     'provenance/docs_interpretation/RESULTS.json',
                     'provenance/docs_interpretation/gate_fixed_one/PROTOCOL.json',
                     'provenance/docs_interpretation/gate_fixed_one/RESULTS.json'):
        check((root / relative).is_file(), f'Missing completed diagnostic: {relative}')
        names.append(relative)
    page = root / 'index.html'
    check(TITLE in page.read_text(), 'Unexpected HTML title')
    for name in ('REPORT_RENDER.json', 'ACTUAL_VISUAL_QA.json', 'VISUAL_REVIEW.json'):
        check(read(root / name)['html_sha256'] == sha(page), f'{name} does not bind actual HTML')
    check(read(root / 'VISUAL_REVIEW.json')['actual_screenshots_inspected'], 'Screenshot inspection missing')
    audit = read(root / 'INDEPENDENT_AUDIT.json')
    check(audit['protocol_sha256'] == sha(root / 'TRAIN_PROTOCOL.json'), 'Independent audit protocol differs')
    results = read(root / 'RESULTS.json')
    headline = results.get('headline_ko', results.get('verdict', {}).get('headline_ko'))
    check(isinstance(headline, str) and headline, 'A measured result headline is required')
    show_page(page)
    from scripts.research.discord_notify import notify
    message = '\n'.join(['팔레트 점·Hough 선 후보를 직접 결합하는 빠른 실험을 완료했습니다.', headline,
        *results.get('discord_lines_ko', []),
        '기존3seed×고정14장의 무학습 진단과, 고정 backbone에서 동일 크기 결합부2개×1000updates의 합성 학습을 완료했습니다.',
        '합성2048train/512val, 실사319 DEV. 새로운 학습seed는1개이며 2D 점 평가입니다. 안정적 일반화·6D 향상의 확증은 아닙니다.',
        f'결과 HTML(작업 PC): {page}', '보고서의 실제 브라우저 창 표시를 확인했습니다.'])
    delivery = notify(message, root, evidence=root / 'RESULTS.json')
    check(delivery['status'] == 'sent' and delivery.get('http_status') in (200, 204), 'Discord delivery not confirmed')
    names.extend(['GALLERY_OPEN.json', 'DISCORD_NOTIFICATION.json'])
    receipt = dict(complete=True, PASS=True, completed_at_utc=now(), headline_ko=headline,
        scope='Execution, audit and delivery completed; PASS does not imply an accuracy improvement.',
        report=str(page), html_sha256=sha(page), protocol_sha256=sha(root / 'TRAIN_PROTOCOL.json'),
        browser_window_confirmed=True, discord_status=delivery['status'], discord_http_status=delivery['http_status'],
        artifact_sha256={name: sha(root / name) for name in names},
        source_sha256={str(p): sha(p) for p in Path(__file__).resolve().parent.glob('*.py')})
    write(root / 'COMPLETION.json', receipt)
    write(root / 'STATUS.json', dict(stage='complete', complete=True, updated_at_utc=now(), headline_ko=headline))
    print(headline, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path, required=True)
    finalize(parser.parse_args().run_dir)
