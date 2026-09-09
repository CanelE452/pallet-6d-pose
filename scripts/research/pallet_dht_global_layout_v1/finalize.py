"""Verify the completed pilot, show its actual window, and notify its owner."""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
from pathlib import Path
from .preflight import REPO, read, sha, write
from scripts.research.pallet_dht_gt_audit_v1.finalize import open_report

def finalize(root):
    root = root.resolve()
    assert not (root / 'COMPLETION.json').exists(), 'Pilot already delivered'
    snapshot = read(root / 'INPUT_SNAPSHOT.json')
    assert snapshot['complete'] and snapshot['PASS']
    for path, digest in snapshot['sha256'].items():
        assert sha(path) == digest, f'Original input changed: {path}'
    for name in ('INDEPENDENT_AUDIT.json', 'REPORT_RENDER.json', 'ACTUAL_VISUAL_QA.json',
                 'ROOT_VISUAL_REVIEW.json'):
        receipt = read(root / name)
        assert receipt['complete'] and receipt['PASS'], f'Incomplete {name}'
        for key in ('input_sha256', 'source_sha256', 'screenshot_sha256', 'output_sha256'):
            for path, digest in receipt.get(key, {}).items():
                candidate = Path(path)
                if not candidate.is_absolute():
                    candidate = root / candidate
                assert sha(candidate) == digest, f'Stale {name}: {path}'
    freeze = read(root / 'SOURCE_FREEZE.json')
    for path, digest in freeze['sha256'].items():
        assert sha(path) == digest, f'Frozen inference source changed: {path}'
    assert sha(root / 'PROTOCOL.json') == snapshot['protocol_sha256']
    page = root / 'index.html'
    for name in ('REPORT_RENDER.json', 'ACTUAL_VISUAL_QA.json', 'ROOT_VISUAL_REVIEW.json'):
        assert read(root / name)['html_sha256'] == sha(page), f'Stale rendering receipt {name}'
    review = read(root / 'ROOT_VISUAL_REVIEW.json')
    assert review['actual_screenshots_inspected']
    conclusion = read(root / 'CONCLUSION.json')
    assert conclusion['complete'] and conclusion['gt_modifications'] == 0
    assert conclusion['results_sha256'] == sha(root / 'RESULTS.json')
    report = read(root / 'REPORT_RENDER.json')
    open_report(page, report['title'])
    from scripts.research.discord_notify import notify
    message = '\n'.join([
        '팔레트 8점·12선 전체 배치 선택 실험을 완료했습니다.',
        conclusion['headline_ko'], *conclusion['summary_lines_ko'],
        f'비교 HTML(작업 PC): {page}',
        '실제 브라우저 창 표시를 확인했습니다. GT 수정·새 CNN 학습은 없습니다.'
    ])
    delivery = notify(message, root, evidence=root / 'CONCLUSION.json')
    assert delivery['status'] == 'sent' and delivery.get('http_status') in (200, 204)
    artifacts = sorted(p for p in root.glob('*.json') if p.name not in ('COMPLETION.json', 'STATUS.json'))
    write(root / 'COMPLETION.json', dict(
        complete=True, PASS=True, completed_at_utc=datetime.now(timezone.utc).isoformat(),
        scope='Execution, independent audit and delivery complete; accuracy verdict is separate.',
        scientific_proceed=conclusion['scientific_proceed'],
        headline_ko=conclusion['headline_ko'], gt_fully_certified=False,
        gt_modifications=0, new_cnn_training_updates=0, new_cnn_forwards=0,
        report=str(page), html_sha256=sha(page), original_inputs_preserved=len(snapshot['sha256']),
        browser_window_confirmed=True, discord_http_status=delivery['http_status'],
        artifact_sha256={p.name: sha(p) for p in artifacts},
        source_sha256={str(p): sha(p) for p in Path(__file__).resolve().parent.glob('*.py')},
        opener_source_sha256=sha(REPO / 'scripts/research/pallet_dht_gt_audit_v1/finalize.py')))
    print(conclusion['headline_ko'], flush=True)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path, default=REPO / 'data/pallet/results/pallet_dht_global_layout_v1')
    finalize(parser.parse_args().run_dir)
