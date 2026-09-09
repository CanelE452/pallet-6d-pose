"""Verify uncertainty experiment artifacts, open the report and notify Discord."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(HERE.parent))
from discord_notify import notify


def read(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def browser_window(page):
    """Confirm only the requested report window; do not expose other titles."""
    title = 'YOLO σ + DHT uncertainty · Fusion verification'
    for _ in range(10):
        try:
            listing = subprocess.run(['xprop', '-root', '_NET_CLIENT_LIST_STACKING'],
                                     capture_output=True, text=True, timeout=5)
            for window in re.findall(r'0x[0-9a-fA-F]+', listing.stdout):
                name = subprocess.run(['xprop', '-id', window, '_NET_WM_NAME'],
                                      capture_output=True, text=True, timeout=5).stdout
                if title not in name:
                    continue
                info = subprocess.run(['xwininfo', '-id', window], capture_output=True,
                                      text=True, timeout=5).stdout
                state = subprocess.run(['xprop', '-id', window, '_NET_WM_DESKTOP', '_NET_WM_STATE'],
                                       capture_output=True, text=True, timeout=5).stdout
                visible = 'Map State: IsViewable' in info and '_NET_WM_STATE_HIDDEN' not in state
                if visible:
                    receipt_path = page.parent/'GALLERY_OPEN.json'
                    receipt = read(receipt_path) if receipt_path.exists() else {}
                    receipt.update(status='window_confirmed', window_id=window,
                                   window_title=title, window_visibility_confirmed=True,
                                   html_sha256=sha(page), checked_at_utc=datetime.now(timezone.utc).isoformat())
                    receipt_path.write_text(json.dumps(receipt, ensure_ascii=False, indent=2)+'\n')
                    return True
        except (OSError, subprocess.TimeoutExpired):
            return False
        time.sleep(.5)
    return False


def main(live, no_open=False):
    out = live/'uncertainty_fusion_v1'
    protocol = read(out/'PROTOCOL.json')
    for name, digest in protocol['source_sha256'].items():
        assert sha(live/name) == digest, name
    for name in ['EXTRACTION_AUDIT_YOLO.json', 'EXTRACTION_AUDIT_DHT.json',
                 'EVALUATION_AUDIT.json', 'INDEPENDENT_AUDIT.json', 'STATISTICAL_AUDIT.json']:
        assert read(out/name)['PASS'], name
    for name in ['INDEPENDENT_AUDIT.json', 'STATISTICAL_AUDIT.json']:
        for path, digest in read(out/name)['input_sha256'].items():
            assert sha(Path(path)) == digest, path
    assert read(out/'STATISTICAL_AUDIT.json')['independent_audit_sha256'] == sha(out/'INDEPENDENT_AUDIT.json')
    audit = read(out/'EVALUATION_AUDIT.json')
    for name, digest in audit['output_sha256'].items():
        assert sha(out/name) == digest, name
    for path, digest in audit['source_sha256'].items():
        assert sha(Path(path)) == digest, path
    assert read(out/'EXTRACTION_AUDIT_YOLO.json')['prediction_sha256'] == sha(out/'YOLO_UNCERTAINTY.json')
    assert read(out/'EXTRACTION_AUDIT_DHT.json')['output_sha256'] == sha(out/'DHT_UNCERTAINTY.json')
    render, visual = read(out/'REPORT_RENDER.json'), read(out/'VISUAL_QA.json')
    page = out/'uncertainty_report.html'
    assert render['complete'] and render['html_sha256'] == sha(page)
    assert visual['PASS'] and visual['html_sha256'] == sha(page)
    for name, digest in render['input_sha256'].items():
        assert sha(out/name) == digest, name
    result = read(out/'RESULTS.json')
    assert result['complete'] and result['n_frames'] == 692
    metrics = {}
    for arm in ['baseline', 'previous_joint_gate', 'unconditional_lambda1',
                'point_sigma_only', 'line_uncertainty_only', 'point_line_uncertainty']:
        row = next(r for r in result['seed_summary'] if
                   (r['arm'], r['population'], r['group']) == (arm, 'real_dev', 'all'))
        metrics[arm] = {k: row['metrics'][k]['mean'] for k in ['corner_mean_px', 'pck_10px']}
    if not no_open:
        sys.path.insert(0, str(HERE.parent/'deep_hough_side_v1'))
        from visualize import open_gallery
        open_gallery(page)
    window_confirmed = browser_window(page)
    message = ['YOLO RLE sigma + DHT 선 불확실성 결합 검증이 다 끝났습니다.',
               '기존 692장 / 실사 DEV52 / DHT 3개 seed. 합성128장 보정, 별도128장 선택 후 실사 평가했습니다.']
    for arm, label in [('baseline', '기본 YOLO'), ('previous_joint_gate', '이전 선택적 결합'),
                       ('point_sigma_only', '점 sigma만'), ('line_uncertainty_only', '선 불확실성만'),
                       ('point_line_uncertainty', '점+선 불확실성')]:
        m = metrics[arm]
        message.append(f'{label}: 평균 {m["corner_mean_px"]:.2f}px, PCK@10 {100*m["pck_10px"]:.2f}%.')
    base, joint = metrics['baseline'], metrics['point_line_uncertainty']
    delta = joint['corner_mean_px'] - base['corner_mean_px']
    if abs(delta) < 1e-9:
        message.append('점·선의 불확실성을 모두 쓰는 방식은 λ0(결합하지 않음)가 선택되어 기본 YOLO를 유지했습니다.')
    elif delta > 0:
        message.append('점+선 결합은 기본 YOLO 대비 평균오차가 악화되었습니다.')
    else:
        message.append('점+선 결합의 실사 평균오차가 감소했습니다. 통계 구간은 HTML에서 확인할 수 있습니다.')
    line_delta = metrics['line_uncertainty_only']['corner_mean_px']-base['corner_mean_px']
    line_ci = next(r for r in read(out/'STATISTICAL_AUDIT.json')['paired_bootstrap']
                   if r['arm'] == 'line_uncertainty_only' and r['reference'] == 'baseline')['mean_error_delta_ci95']
    message.append(f'선 불확실성만 반영한 약한 결합의 평균오차 변화 {line_delta:+.3f}px '
                   f'(프레임 bootstrap 95% 구간 {line_ci[0]:+.3f}~{line_ci[1]:+.3f}).')
    cross_before = next(r for r in result['seed_summary'] if
                       (r['arm'], r['population'], r['group']) == ('baseline', 'cross_v4', 'all'))
    cross_after = next(r for r in result['seed_summary'] if
                      (r['arm'], r['population'], r['group']) == ('line_uncertainty_only', 'cross_v4', 'all'))
    message.append(f'다른 합성 분포 cross_v4에서는 평균 '
                   f'{cross_before["metrics"]["corner_mean_px"]["mean"]:.2f}→'
                   f'{cross_after["metrics"]["corner_mean_px"]["mean"]:.2f}px로 악화되어 일반적 개선으로 단정하지 않습니다.')
    message += ['기존 DEV52의 반복 평가이며 독립 실사 일반화 검증은 아닙니다.',
                f'시각화 HTML(작업 PC): {page}',
                'Chrome 보고서 창 표시를 확인했습니다.' if window_confirmed else '브라우저 창 표시 확인은 되지 않았습니다.']
    names = ['PROTOCOL.json', 'YOLO_UNCERTAINTY.json', 'DHT_UNCERTAINTY.json',
             'EXTRACTION_AUDIT_YOLO.json', 'EXTRACTION_AUDIT_DHT.json', 'CALIBRATION.json',
             'SELECTION.json', 'RESULTS.json', 'RELIABILITY.json', 'EVALUATION_AUDIT.json',
             'INDEPENDENT_AUDIT.json', 'STATISTICAL_AUDIT.json', 'REPORT_RENDER.json',
             'uncertainty_report.html', 'VISUAL_QA.json', 'DIAGNOSIS.json', 'PEAK_REPRODUCIBILITY_DHT.json']
    if (out/'NUMERICAL_STABILITY_DHT.json').exists():
        names.append('NUMERICAL_STABILITY_DHT.json')
    completion = dict(PASS=True, completed_at_utc=datetime.now(timezone.utc).isoformat(),
                      scope='Execution and artifact verification; not a claim of accuracy gain.',
                      n_frames=692, real_dev_frames=52, dht_seeds=[1, 2, 3], newly_trained_models=0,
                      calibration_frames=128, selection_frames=128, real_tuning=False,
                      visual_qa_verified=True, browser_window_confirmed=window_confirmed,
                      real_metrics=metrics, artifact_sha256={name: sha(out/name) for name in names})
    receipt = notify('\n'.join(message), out, evidence=out/'RESULTS.json')
    completion.update(discord_status=receipt['status'], discord_http_status=receipt.get('http_status'),
                      discord_receipt_sha256=sha(out/'DISCORD_NOTIFICATION.json'))
    (out/'COMPLETION.json').write_text(json.dumps(completion, ensure_ascii=False, indent=2)+'\n')
    if receipt['status'] != 'sent':
        raise SystemExit('Verification finished; Discord delivery not confirmed. See DISCORD_NOTIFICATION.json.')
    print(json.dumps(dict(PASS=True, browser_window_confirmed=window_confirmed,
                          discord_status=receipt['status'], report=str(page)), ensure_ascii=False))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path, default=ROOT/'data/pallet/results/dht_pose_integration_v1/live')
    parser.add_argument('--no-open', action='store_true')
    args = parser.parse_args()
    main(args.run_dir.resolve(), args.no_open)
