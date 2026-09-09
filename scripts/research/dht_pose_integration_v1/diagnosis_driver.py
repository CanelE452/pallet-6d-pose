"""Run cached point/line diagnosis through evaluation, HTML and Discord notification."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(HERE.parent))
from discord_notify import notify


def read(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def finalize(live, no_open=False):
    out = live/'fusion_diagnosis_v1'
    for name, digest in read(out/'PROTOCOL.json')['input_sha256'].items():
        assert sha(live/name) == digest, name
    assert read(out/'GEOMETRY_AUDIT.json')['status'] == 'PASS'
    audit = read(out/'GATE_AUDIT.json')
    assert audit['PASS'] and read(out/'STATISTICAL_AUDIT.json')['PASS']
    for name, digest in audit['output_sha256'].items():
        assert sha(out/name) == digest, name
    render = read(out/'REPORT_RENDER.json')
    page = out/'fusion_diagnosis.html'
    assert render['complete'] and sha(page) == render['html_sha256']
    for name, digest in render['input_sha256'].items():
        assert sha(out/name) == digest, name
    result = read(out/'GATE_RESULTS.json')
    assert result['complete']
    summaries = []
    message = ['점·선 결합 악화 원인 분석과 선택적 결합 실험이 다 끝났습니다.',
               '기존 692장(실사 DEV52), DHT 3개 seed로 검증했습니다.',
               '원인: 일부 잘못된 선이 정확한 점까지 크게 이동시킵니다.']
    for model, label in [('DOPE_exact_resize', 'DOPE(좌표 보정)'), ('YOLO', 'YOLO')]:
        metrics = {}
        for arm in ['baseline', 'unconditional_lambda1', 'confidence_line_move']:
            row = next(r for r in result['seed_summary'] if
                       (r['model'], r['arm'], r['population'], r['group']) ==
                       (model, arm, 'real_dev', 'all'))
            metrics[arm] = {k: row['metrics'][k]['mean'] for k in ['corner_mean_px', 'pck_10px']}
        base, fixed, gated = [metrics[k] for k in ['baseline', 'unconditional_lambda1', 'confidence_line_move']]
        summaries.append(dict(model=model, metrics=metrics))
        message.append(f'{label} 평균오차: 기본 {base["corner_mean_px"]:.2f} → '
                       f'무조건 결합 {fixed["corner_mean_px"]:.2f} → 선택적 결합 {gated["corner_mean_px"]:.2f}px. '
                       f'PCK@10: {100*base["pck_10px"]:.1f} → {100*gated["pck_10px"]:.1f}%.')
    message += ['선택 기준은 합성 검증에서만 정했습니다. 일부 지표 개선은 있지만 평균오차의 전체 개선은 확인되지 않았습니다.',
                f'비교 HTML(작업 PC): {page}']
    if not no_open:
        sys.path.insert(0, str(HERE.parent/'deep_hough_side_v1'))
        from visualize import open_gallery
        open_gallery(page)
    names = ['PROTOCOL.json', 'CONFIG_GATE.json', 'GATE_SELECTION.json', 'GATE_RESULTS.json',
             'GEOMETRY_AUDIT.json', 'GATE_AUDIT.json', 'STATISTICAL_AUDIT.json',
             'DIAGNOSIS.json', 'REPORT_RENDER.json', 'fusion_diagnosis.html']
    visual = out/'VISUAL_QA.json'
    visual_verified = False
    if visual.exists():
        visual_result = read(visual)
        visual_verified = bool(visual_result.get('PASS') and visual_result.get('html_sha256') == sha(page))
        names.append(visual.name)
    completion = dict(PASS=True, scope='Analysis execution and artifact verification; not a claim of overall accuracy gain.',
                      n_frames=692, real_dev_frames=52, dht_seeds=[1, 2, 3], newly_trained_models=0,
                      gate_selected_only_on_synthetic_validation=True, visual_qa_verified=visual_verified,
                      real_metrics=summaries, artifact_sha256={name: sha(out/name) for name in names})
    (out/'COMPLETION.json').write_text(json.dumps(completion, ensure_ascii=False, indent=2)+'\n')
    return notify('\n'.join(message), out, evidence=out/'GATE_RESULTS.json')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path, default=ROOT/'data/pallet/results/dht_pose_integration_v1/live')
    parser.add_argument('--finalize-only', action='store_true', help='Verify completed artifacts, open HTML and notify without recomputing')
    parser.add_argument('--no-open', action='store_true')
    args = parser.parse_args()
    live = args.run_dir.resolve()
    out = live/'fusion_diagnosis_v1'
    if not (out/'PURPOSE.md').is_file():
        raise ValueError('Prepare diagnosis PURPOSE.md and PROTOCOL.json first')
    if not args.finalize_only:
        for script in ['diagnose_fusion.py', 'selective_fusion.py']:
            subprocess.run([sys.executable, str(HERE/script), '--run-dir', str(live)], cwd=out, check=True)
        subprocess.run([sys.executable, str(HERE/'audit_selective_fusion.py'), '--output-dir', str(out)], cwd=out, check=True)
        subprocess.run([sys.executable, str(HERE/'diagnosis_report.py'), '--run-dir', str(live), '--no-open'], cwd=out, check=True)
    receipt = finalize(live, args.no_open)
    if receipt['status'] != 'sent':
        raise SystemExit('Analysis completed, but Discord delivery was not confirmed; see DISCORD_NOTIFICATION.json')
