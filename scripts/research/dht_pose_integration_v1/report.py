"""Offline, interactive report of frozen DOPE/YOLO plus DHT inference.

Only saved predictions are visualized. Rendering never runs a neural network.
The default launcher opens Chrome directly; --no-open renders without opening.
"""
from __future__ import annotations

import argparse
import hashlib
import html
import json
from pathlib import Path
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
SIDE_EDGES = [[1, 2], [3, 0], [5, 6], [7, 4], [0, 4], [1, 5], [2, 6], [3, 7]]
WIDTH_EDGES = [[0, 1], [3, 2], [4, 5], [7, 6]]
POPULATIONS = ['real_dev', 'synth_test', 'cross_v4', 'synth_val']


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def number(value, digits=2):
    return f'{float(value):.{digits}f}' if value is not None and np.isfinite(value) else '—'


def table(headers, rows):
    head = ''.join(f'<th>{html.escape(str(x))}</th>' for x in headers)
    body = ''.join('<tr>' + ''.join(f'<td>{html.escape(str(x))}</td>' for x in row)
                   + '</tr>' for row in rows)
    return f'<div class="tablewrap"><table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table></div>'


def visualization_data(run_dir, manifest, selection, fused, dht):
    """Join strictly by identical ordered IDs, never by nearest GT geometry."""
    expected = [record['id'] for record in manifest['records']]
    if [r['id'] for r in dht['records']] != expected:
        raise ValueError('DHT prediction order differs from the manifest')
    model_names = ['DOPE', 'DOPE_exact_resize', 'YOLO']
    for model in model_names:
        if [r['id'] for r in fused['models'][model]['records']] != expected:
            raise ValueError(f'{model} fused prediction order differs from the manifest')
    output = []
    for index, record in enumerate(manifest['records']):
        row = dict(id=record['id'], population=record['population'],
                   group=record['group'], width=record['width'], height=record['height'],
                   image=Path(record['image']).resolve().as_uri(),
                   gt=record['gt_points'], gt_valid=record['gt_valid'],
                   baselines={}, fused={}, lines={})
        for model in model_names:
            pred = fused['models'][model]['records'][index]
            row['baselines'][model] = dict(kps=pred['baseline']['kps'],
                                            valid=pred['baseline']['kp_valid'])
            row['fused'][model] = {}
            for seed in ('1', '2', '3'):
                row['fused'][model][seed] = {}
                for old, new in [('selected', 'selected'), ('fixed1', 'diagnostic')]:
                    value = pred['seeds'][seed][old]
                    if value['kp_valid'] != pred['baseline']['kp_valid']:
                        raise ValueError('Fusion unexpectedly changed point validity')
                    row['fused'][model][seed][new] = {'kps': value['kps']}
        for seed in ('1', '2', '3'):
            row['lines'][seed] = dht['records'][index]['seeds'][seed]['lines']
        output.append(row)
    return dict(records=output, populations=POPULATIONS, edges=SIDE_EDGES,
                width_edges=WIDTH_EDGES,
                selected_lambda={m: selection['models'][m]['lambda'] for m in model_names})


def validate_inputs(run_dir, payloads):
    expected_config, expected_manifest = sha(run_dir / 'CONFIG.json'), sha(run_dir / 'manifest.json')
    checked = {}
    for name, payload in payloads.items():
        if payload.get('config_sha256') != expected_config:
            raise ValueError(f'{name}: config hash mismatch')
        if payload.get('manifest_sha256') != expected_manifest:
            raise ValueError(f'{name}: manifest hash mismatch')
        for file, digest in payload.get('source_sha256', {}).items():
            source = Path(file)
            if not source.is_absolute():
                source = ROOT / source
            if sha(source) != digest:
                raise ValueError(f'{name}: changed source {source}')
        checked[name] = sha(run_dir / name)
    return checked


MODEL_LABELS = {'YOLO': 'YOLO', 'DOPE_exact_resize': 'DOPE · resize 보정',
                'DOPE': 'DOPE · 기존 역변환'}


def metric_cell(result, key, percent=False, interval=True):
    value = result['metrics'][key]
    if value is None:
        return '—'
    factor, suffix = (100, '%') if percent else (1, '')
    center = number(value['mean'] * factor) + suffix
    if interval and value['max'] - value['min'] > 1e-10:
        center += f" [{number(value['min'] * factor)}, {number(value['max'] * factor)}]"
    return center


def find_summary(results, model, arm, population, group='all'):
    return next(r for r in results['seed_summary'] if
                (r['model'], r['arm'], r['population'], r['group']) ==
                (model, arm, population, group))


def result_tables(results, selection):
    sections = []
    headers = ['모델 / 설정', 'N', 'Coverage', 'PCK@10px', '조건부 median px',
               '조건부 P90 px', 'All8@10px']
    for population in POPULATIONS:
        rows = []
        for model in ('YOLO', 'DOPE_exact_resize', 'DOPE'):
            for arm in ('baseline', 'selected', 'fixed1'):
                summary = find_summary(results, model, arm, population)
                raw = next(r for r in results['summaries'] if
                           (r['model'], r['arm'], r['population'], r['group']) ==
                           (model, arm, population, 'all'))
                label = ('기준' if arm == 'baseline' else
                         f"+ DHT · 선택 λ={selection['models'][model]['lambda']}" if arm == 'selected'
                         else '+ DHT · 고정 λ=1 진단')
                rows.append([MODEL_LABELS[model] + ' / ' + label, raw['n_frames'],
                             number(100 * raw['corner_coverage']) + '%',
                             metric_cell(summary, 'pck_10px', True),
                             metric_cell(summary, 'corner_median_px'),
                             metric_cell(summary, 'corner_p90_px'),
                             metric_cell(summary, 'all8_10px_success', True)])
        content = table(headers, rows)
        if population == 'real_dev':
            sections.append('<h3>실제 개발 이미지 · real_dev 52장</h3>' + content)
        else:
            sections.append(f'<details><summary>{population} 결과</summary>{content}</details>')
    sections.append('<p class="small">결합 수치는 seed 1·2·3에서 각각 계산한 통계의 평균 [최솟값, 최댓값]입니다. 같은 이미지를 세 번 합쳐 분모를 늘리지 않았습니다. 기준 모델은 한 번만 계산했습니다. 선택 λ=0이면 기준 예측과 정확히 같습니다.</p>')
    group_rows = []
    groups = sorted({r['group'] for r in results['summaries'] if r['population'] == 'real_dev' and r['group'] != 'all'})
    for group in groups:
        for model in ('YOLO', 'DOPE_exact_resize', 'DOPE'):
            before = find_summary(results, model, 'baseline', 'real_dev', group)
            after = find_summary(results, model, 'selected', 'real_dev', group)
            raw = next(r for r in results['summaries'] if
                       (r['model'], r['arm'], r['population'], r['group']) ==
                       (model, 'baseline', 'real_dev', group))
            group_rows.append([group, MODEL_LABELS[model], raw['n_frames'],
                               metric_cell(before, 'pck_10px', True, False) + ' → ' + metric_cell(after, 'pck_10px', True, False),
                               metric_cell(before, 'corner_p90_px', interval=False) + ' → ' + metric_cell(after, 'corner_p90_px', interval=False)])
    sections.append('<details open><summary>실제 촬영 그룹별: 기준 → 선택된 결합</summary>' +
                    table(['그룹', '모델', 'N', 'PCK@10px', '조건부 P90 px'], group_rows) + '</details>')
    line_rows = []
    for model in ('YOLO', 'DOPE_exact_resize', 'DOPE'):
        for arm in ('baseline', 'selected', 'fixed1'):
            summary = find_summary(results, model, arm, 'real_dev')
            line_rows.append([MODEL_LABELS[model], arm,
                              metric_cell(summary, 'line_distance_median_px'),
                              metric_cell(summary, 'line_distance_p90_px'),
                              metric_cell(summary, 'line_angle_median_deg'),
                              metric_cell(summary, 'line_success_fraction', True)])
    sections.append('<details><summary>실제 이미지의 8개 측면 선 지표</summary><p class="small">기준/결합의 코너를 연결해 얻은 선입니다. GT support는 평가 분모에만 사용하며 결합 입력을 고르는 데 사용하지 않습니다. 선 성공은 각도 ≤5°와 거리 ≤8px를 동시에 만족한 비율입니다.</p>' +
                    table(['모델', '설정', '거리 median px', '거리 P90 px', '각도 median °', '선 성공'], line_rows) + '</details>')
    return ''.join(sections)


def selection_tables(results, selection):
    rows = []
    for model in ('YOLO', 'DOPE_exact_resize', 'DOPE'):
        for trial in selection['models'][model]['grid']:
            lam = trial['lambda_value']
            rows.append([MODEL_LABELS[model], lam,
                         number(trial['mean_score'], 6),
                         ' / '.join(number(x, 6) for x in trial['per_seed']),
                         '선택' if lam == selection['models'][model]['lambda'] else ''])
    output = '<p class="small">선택 목적함수: 각 코너의 오차를 원본 대각선으로 나눈 뒤 1로 cap하고 프레임 평균을 계산합니다. 누락 코너는 1입니다. 낮을수록 좋으며 동률이면 작은 λ를 선택합니다.</p>'
    output += table(['모델', 'λ', '3 seed 평균', 'seed1 / seed2 / seed3', '선택'], rows)
    seed_rows = []
    for model in ('YOLO', 'DOPE_exact_resize', 'DOPE'):
        for arm in ('selected', 'fixed1'):
            for seed in (1, 2, 3):
                summary = next(r for r in results['summaries'] if
                               (r['model'], r['arm'], r['seed'], r['population'], r['group']) ==
                               (model, arm, seed, 'real_dev', 'all'))
                paired = next(r for r in results['paired'] if
                              (r['model'], r['arm'], r['seed'], r['population'], r['group']) ==
                              (model, arm, seed, 'real_dev', 'all'))
                seed_rows.append([MODEL_LABELS[model], arm, seed,
                                  number(summary['pck_10px'] * 100) + '%',
                                  number(summary['corner_p90_px']), number(paired['mean_delta_px']),
                                  f"{paired['improved_frames']} / {paired['worsened_frames']} / {paired['unchanged_frames']}"])
    output += '<details><summary>실제 이미지 seed별 상세와 같은 프레임의 변화</summary><p class="small">Δ는 결합−기준의 프레임 평균 코너 오차입니다. 누락은 이미지 대각선만큼 부과합니다. 음수는 감소, 양수는 증가입니다. 개선/악화/동일은 이 프레임 오차 기준입니다.</p>'
    output += table(['모델', '설정', 'seed', 'PCK@10px', 'P90 px', '평균 Δ px', '개선 / 악화 / 동일'], seed_rows) + '</details>'
    return output


def timing_tables(run_dir, runtime):
    rows = []
    for model in ('YOLO', 'DOPE'):
        baseline = read(run_dir / f'BASELINE_{model}.json')
        for population in ('real_dev', 'synth_test'):
            values = [r['inference_ms'] for r in baseline['records'] if r['population'] == population]
            rows.append([model + ' 개별 실행', population, len(values), number(np.median(values)), number(np.percentile(values, 90))])
    output = '<h3>개별 모델 · 전체 평가 이미지에서 측정</h3>' + table(['모델', '데이터', 'N', 'median ms', 'P90 ms'], rows)
    if runtime is None:
        return output + '<p class="note warning">결합 파이프라인의 실제 지연시간은 아직 측정 완료되지 않았습니다. 개별 시간의 합으로 대체하지 않았습니다.</p>'
    if not runtime.get('complete') or not runtime.get('parity_PASS'):
        return output + '<p class="note warning">결합 실행시간 기록의 동일 예측 검증이 아직 통과하지 않아 해당 수치를 제외했습니다. 정확도 표는 검증된 저장 예측을 사용합니다.</p>'
    rows = []
    for model, data in runtime['by_model'].items():
        for population in ('all', 'real_dev', 'synth_test'):
            group = data['by_population'][population]
            for arm in ('baseline', 'selected', 'fixed_lambda1'):
                values = group['arms'][arm]
                rows.append([MODEL_LABELS[model], population, arm, values['n'],
                             number(values['median_ms']), number(values['p90_ms'])])
    output += '<h3 style="margin-top:20px">결합 파이프라인 · 사전 고정 24장 × 3회 실제 실행</h3>'
    output += '<p class="small">동일한 24장은 real_dev와 synth_test에서 프레임 ID의 SHA 순서로 고정했습니다. λ=0인 선택 설정은 DHT 실행을 생략하여 기준 모델과 같은 연산을 수행합니다. 이 둘의 측정 시간 차이는 실행 변동이며 속도 향상이 아닙니다. 고정 λ=1은 실제로 DHT와 코너 결합을 실행합니다.</p>'
    output += table(['모델', '데이터', '설정', '측정 횟수', 'median ms', 'P90 ms'], rows)
    return output


def verdict_text(results, selection):
    yolo_lambda = selection['models']['YOLO']['lambda']
    baseline = find_summary(results, 'DOPE_exact_resize', 'baseline', 'real_dev')
    fused = find_summary(results, 'DOPE_exact_resize', 'selected', 'real_dev')
    legacy = find_summary(results, 'DOPE', 'baseline', 'real_dev')
    legacy_fused = find_summary(results, 'DOPE', 'selected', 'real_dev')
    return (f"<strong>YOLO의 합성 검증 선택은 λ={yolo_lambda}입니다" +
            (' — 기준 모델을 유지합니다.' if yolo_lambda == 0 else '.') + '</strong> '
            f"DOPE resize 보정 대조군은 선택 λ={selection['models']['DOPE_exact_resize']['lambda']}에서 "
            f"실제 PCK@10px {metric_cell(baseline, 'pck_10px', True, False)} → "
            f"{metric_cell(fused, 'pck_10px', True, False)}, P90 "
            f"{metric_cell(baseline, 'corner_p90_px', interval=False)} → "
            f"{metric_cell(fused, 'corner_p90_px', interval=False)}px입니다. "
            f"기존 DOPE의 PCK@10px는 {metric_cell(legacy, 'pck_10px', True, False)} → "
            f"{metric_cell(legacy_fused, 'pck_10px', True, False)}입니다. "
            '각 지표와 실패 사례를 함께 확인해야 하며, 일부 PCK 변화만으로 전반적인 개선을 판단할 수 없습니다.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--no-open', action='store_true')
    args = parser.parse_args()
    run_dir = args.run_dir.resolve()
    manifest = read(run_dir / 'manifest.json')
    names = ['BASELINE_DOPE.json', 'BASELINE_YOLO.json', 'DHT_LINES.json',
             'SELECTION.json', 'FUSED_PREDICTIONS.json', 'RESULTS.json']
    payloads = {name: read(run_dir / name) for name in names}
    verified = validate_inputs(run_dir, payloads)
    results, selection = payloads['RESULTS.json'], payloads['SELECTION.json']
    if not results.get('complete') or results['selection_sha256'] != sha(run_dir / 'SELECTION.json'):
        raise ValueError('Final metrics are incomplete or selection changed')
    if payloads['FUSED_PREDICTIONS.json']['selection_sha256'] != results['selection_sha256']:
        raise ValueError('Visualization predictions use a different lambda selection')
    data = visualization_data(run_dir, manifest, selection,
                              payloads['FUSED_PREDICTIONS.json'], payloads['DHT_LINES.json'])
    runtime = read(run_dir / 'RUNTIME.json') if (run_dir / 'RUNTIME.json').exists() else None
    if runtime is not None and runtime.get('complete') and runtime.get('parity_PASS'):
        verified.update(validate_inputs(run_dir, {'RUNTIME.json': runtime}))
        for name, digest in runtime['input_sha256'].items():
            if sha(run_dir / name) != digest:
                raise ValueError(f'Combined benchmark input changed: {name}')
    facts = '<div class="facts">' + ''.join(
        f'<div class="fact"><strong>{big}</strong><span>{label}</span></div>' for big, label in
        [('692', '동일한 평가 이미지'), ('52', '실제 개발 이미지 · 3개 그룹'),
         ('3', '합성 학습 DHT seed'), ('0', '이번 비교에서 새로 학습한 모델')]) + '</div>'
    link_names = ['RESULTS.json', 'SELECTION.json', 'RUNTIME.json', 'EVALUATION_COMPLETION.json',
                  'DOPE_RESIZE_AUDIT.json', 'CONFIG.json', 'AUXILIARY_PROTOCOL.json',
                  'BASELINE_DOPE.json', 'BASELINE_YOLO.json', 'DHT_LINES.json', 'FUSED_PREDICTIONS.json']
    links = '<p>' + ' · '.join(f'<a href="{name}">{name}</a>' for name in link_names if (run_dir / name).exists()) + '</p>'
    old_gallery = ROOT / 'data/pallet/results/deep_hough_side_v1/deep_hough_gallery.html'
    links += f'<p><a href="{old_gallery.as_uri()}">이전 DHT feature sensitivity / Hough 확률 시각화</a></p>'
    scope_note = ('<p class="note">이 보고서는 각 원본 이미지에서 batch 1로 DHT feature를 새로 계산한 최종 경로입니다. '
                  '기존 batch 12 feature 캐시와 실제 실행에서 일부 Hough 최댓값 bin이 달라져 전체 692장·3 seed를 다시 추론했습니다. '
                  '같은 합성 validation 선택 규칙을 다시 적용했고, 하이퍼파라미터는 바꾸지 않았습니다.</p>') if run_dir.name == 'live' else (
                  '<p class="note warning">기존 batch 12 feature 캐시를 사용한 재현 진단입니다. 최종 실제 이미지 batch 1 경로와 구분합니다.</p>')
    if run_dir.name == 'live':
        links += '<p><a href="../integration_report.html">기존 feature 캐시 재현 진단</a>'
        if (run_dir.parent / 'DOPE_RESIZE_AUDIT.json').exists():
            links += ' · <a href="../DOPE_RESIZE_AUDIT.json">DOPE resize 대조군 감사</a>'
        links += '</p>'
    substitutions = dict(FACTS=facts, SCOPE_NOTE=scope_note, VERDICT=verdict_text(results, selection),
                         RESULT_TABLES=result_tables(results, selection),
                         SELECTION=selection_tables(results, selection),
                         TIMING=timing_tables(run_dir, runtime), LINKS=links,
                         DATA=json.dumps(data, ensure_ascii=False, separators=(',', ':'),
                                         allow_nan=False).replace('<', '\\u003c'))
    page = PAGE
    for token, content in substitutions.items():
        page = page.replace(f'__{token}__', content)
    target = run_dir / 'integration_report.html'
    target.write_text(page, encoding='utf-8')
    qa = dict(render_complete=True, n_frames=len(data['records']),
              n_models=3, seeds=[1, 2, 3], default_population='real_dev',
              input_sha256=verified, report_code_sha256=sha(Path(__file__)),
              html_sha256=sha(target), runtime_available=runtime is not None)
    (run_dir / 'REPORT_RENDER.json').write_text(json.dumps(qa, indent=2) + '\n')
    print(f'Rendered {target} ({len(data["records"])} images)', flush=True)
    if not args.no_open:
        sys.path.insert(0, str(HERE.parent / 'deep_hough_side_v1'))
        from visualize import open_gallery
        open_gallery(target)
    return 0


PAGE = r'''<!doctype html>
<html lang="ko"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>DOPE / YOLO + DHT 실제 비교</title>
<style>
:root{font-family:Arial,"Noto Sans CJK KR",sans-serif;color:#eaf0f9;background:#101722;color-scheme:dark}
*{box-sizing:border-box}body{margin:0}main{max-width:1500px;margin:auto;padding:30px 28px 60px}
h1{font-size:30px;letter-spacing:-.8px;margin:10px 0}h2{font-size:22px;margin:0 0 14px}h3{font-size:16px;margin:0 0 10px}
p{line-height:1.65;margin:8px 0;color:#b8c5d6}a{color:#7ccfff}section{border:1px solid #314056;border-radius:14px;background:#172130;padding:22px;margin-top:22px}
.eyebrow{font-size:12px;letter-spacing:1.5px;color:#7ccfff}.badge{display:inline-block;font-size:12px;padding:5px 9px;background:#253a43;color:#a9efd1;border-radius:5px}
.facts{display:flex;flex-wrap:wrap;gap:14px;margin:20px 0}.fact{flex:1;min-width:180px;padding:16px 20px;background:#172130;border:1px solid #314056;border-radius:12px}.fact strong{display:block;font-size:28px;margin-bottom:4px}.fact span{font-size:13px;color:#b8c5d6}
.note{background:#1b3040;border-left:3px solid #62bef0;padding:12px 16px;border-radius:5px;color:#c3d9e8}.warning{background:#322a1e;border-color:#e5b86a;color:#e8d2ac}
.tablewrap{overflow-x:auto}table{border-collapse:collapse;width:100%;font-size:13px;white-space:nowrap}th,td{text-align:right;border-bottom:1px solid #334259;padding:10px 12px}th{color:#90a6bf;font-weight:500}th:first-child,td:first-child{text-align:left}tbody tr:hover{background:#223249}
.controls{display:flex;gap:12px;flex-wrap:wrap;margin:16px 0}.controls label{display:flex;flex-direction:column;gap:6px;font-size:12px;color:#b8c5d6}select,button{font:inherit;color:#eaf0f9;background:#233349;border:1px solid #455873;border-radius:6px;padding:9px 11px;cursor:pointer}.framecontrol{flex:1;min-width:290px}select{max-width:100%}.legend{display:flex;gap:18px;flex-wrap:wrap;font-size:13px;margin:14px 0}.legend span:before{content:"";display:inline-block;width:20px;height:3px;margin-right:7px;vertical-align:middle;background:var(--c)}
.panels{display:grid;grid-template-columns:1fr 1fr;gap:14px}.panel{padding:12px;background:#101823;border:1px solid #34445c;border-radius:10px;min-width:0}.panel h3{font-size:14px;color:#cbd8e8;font-weight:500}.panel canvas{display:block;width:100%;height:auto;background:#0b1018}.panel p{font-size:12px;min-height:22px;margin:7px 0 0}.small{font-size:12px}.grid2{display:grid;grid-template-columns:1fr 1fr;gap:18px}details{margin-top:14px}summary{cursor:pointer;color:#91d3ff;padding:10px 0}.chips{display:flex;gap:8px;flex-wrap:wrap}.chips span{border:1px solid #415474;border-radius:5px;padding:5px 9px;font-size:12px}.loading{color:#f0c777}
@media(max-width:850px){main{padding:18px 12px}.panels,.grid2{grid-template-columns:1fr}section{padding:16px}h1{font-size:25px}.fact{min-width:130px}}
</style><main>
<div class="eyebrow">FROZEN MODEL · PAIRED INFERENCE VERIFICATION</div>
<h1>DOPE / YOLO에 DHT 선을 더하면 좋아지는가?</h1>
<p>기존 꼭짓점 예측을 DHT의 측면 선 쪽으로 조정한 결과를 같은 이미지에서 비교합니다. 새 모델 학습 없이 기존 모델을 사용했습니다.</p>
__FACTS__
__SCOPE_NOTE__
<div class="note">__VERDICT__</div>
<section><h2>정량 결과</h2><p>코너 번호를 고정하여 비교했습니다. 누락된 예측도 PCK의 실패로 포함합니다. 조건부 픽셀 오차는 검출된 코너에서만 계산되므로 coverage와 함께 읽어야 합니다.</p>__RESULT_TABLES__</section>
<section id="visuals"><h2>같은 이미지에서 전후 비교</h2>
<p>전체 692장을 탐색할 수 있습니다. 기본 선택은 real_dev의 사전 고정 첫 이미지입니다. 아래 시각화는 저장된 예측이며 학습 attention을 나타내지 않습니다.</p>
<div class="controls"><label>데이터<select id="population"></select></label><label class="framecontrol">이미지<select id="frame"></select></label><label>기준 모델<select id="model"><option>YOLO</option><option value="DOPE_exact_resize">DOPE · resize 역변환 보정</option><option value="DOPE">DOPE · 기존 역변환</option></select></label><label>DHT 학습 seed<select id="seed"><option>1</option><option>2</option><option>3</option></select></label><label>결합 설정<select id="mode"><option value="selected">합성 val로 선택한 λ</option><option value="diagnostic">고정 λ=1 진단</option></select></label><label>표시할 측면 선<select id="role"><option value="all">전체 8개</option></select></label></div>
<div class="legend"><span style="--c:#43ff81">정답 GT</span><span style="--c:#00d4ff">기준 모델</span><span style="--c:#ff59dd">기준 모델 + DHT</span><span style="--c:#ffc266">DHT 예측 선</span></div>
<div id="frameinfo" class="chips"></div>
<p class="small">번호 0–7은 동일한 의미의 코너입니다. GT의 점선 4개는 이번 DHT가 학습하지 않은 폭 방향입니다. 회색 여백은 원본 이미지 바깥이며, 선은 무한 직선의 화면 내 부분을 표시합니다. DOPE resize 보정은 기존 좌표의 가로 1% 복원 오차를 실제 입력 크기로 고친 대조군입니다. GT를 사용하지 않습니다.</p>
<div class="panels"><div class="panel"><h3>원본 + 정답</h3><canvas id="truth"></canvas><p id="truthnote"></p></div><div class="panel"><h3 id="baseheading">기준 모델 + 정답</h3><canvas id="base"></canvas><p id="basenote"></p></div><div class="panel"><h3 id="fuseheading">DHT로 조정 + 정답</h3><canvas id="fused"></canvas><p id="fusenote"></p></div><div class="panel"><h3>DHT의 측면 직선 + 정답</h3><canvas id="lines"></canvas><p id="linesnote"></p></div></div>
<p id="loadstatus" class="small loading">이미지를 불러오는 중…</p>
</section>
<section><h2>λ 선택과 seed별 결과</h2><p>λ는 synth_val 256장에서 3개 DHT seed의 평균 오차로 선택했습니다. 실제 이미지의 점수로 λ, resize, checkpoint를 선택하지 않았습니다. λ=0은 기준 모델을 그대로 유지합니다.</p>__SELECTION__</section>
<section id="timing"><h2>실제 측정한 처리 시간</h2>__TIMING__<p class="small">배치 1, 같은 GPU/환경, 고정 warmup 후 동기화한 wall time입니다. 디스크 읽기와 모델 로딩은 제외합니다. 개별 실행 시간의 합은 실제 결합 지연시간으로 표기하지 않습니다.</p></section>
<section><h2>해석 범위와 재현 자료</h2>
<div class="grid2"><div><h3>결과를 읽는 범위</h3><p>real_dev는 canonical 개발 이미지 52장입니다. 최종 봉인 테스트를 사용하지 않았습니다. 기존 backbone 사전학습과 합성 평가 이미지의 중복 여부는 확인되지 않았습니다.</p><p>DHT는 실제 보이는 물리 edge만 골라내는 모델이 아니라, 가려진 부분을 포함할 수 있는 구조적 측면 직선을 예측합니다. 이번 평가는 2D 코너와 선이며 6D pose 향상을 입증하지 않습니다.</p></div><div><h3>결합 방식과 대조군</h3><p>각 코너의 기존 예측을 기준으로 유지하면서, 그 코너와 연결된 두 DHT 직선까지의 거리를 줄입니다. 누락 코너를 생성하지 않으며 centroid는 유지합니다. 두 모델을 별도로 실행하는 추론 후 결합입니다.</p><p>DOPE는 기존 stage0 전처리와 채널별 peak decoder를 사용합니다. affinity 기반 다중 인스턴스 묶음은 사용하지 않습니다. 가로 좌표의 1% 복원 오차가 남는 기존 결과와 실제 resize 크기로 역변환한 대조군을 모두 보고합니다. DHT가 단순 좌표 오차를 보완한 효과와 구분하기 위한 대조군입니다.</p></div></div>
__LINKS__</section>
</main>
<script id="data" type="application/json">__DATA__</script>
<script>
'use strict';
const DATA=JSON.parse(document.getElementById('data').textContent), E=DATA.edges, WEDGES=DATA.width_edges;
const $=id=>document.getElementById(id), colors={gt:'#43ff81',base:'#00d4ff',fused:'#ff59dd',dht:'#ffc266'};
let generation=0,loadedImage=null,currentRecord=null;
function options(node,entries){node.replaceChildren(...entries.map(([v,t])=>{let o=document.createElement('option');o.value=v;o.textContent=t;return o;}));}
options($('population'),DATA.populations.map(p=>[p,`${p} (${DATA.records.filter(r=>r.population===p).length})`]));
E.forEach((e,i)=>{let o=document.createElement('option');o.value=i;o.textContent=`${i}: ${i<4?'높이':'깊이'} ${e[0]}–${e[1]}`;$('role').append(o);});
function updateFrames(){options($('frame'),DATA.records.filter(r=>r.population===$('population').value).map(r=>[r.id,`${r.group} · ${r.id}`]));render();}
function validPoint(p){return p&&p.length===2&&p.every(Number.isFinite);}
function stroke(ctx,a,b,color,width=2,dash=[]){if(!validPoint(a)||!validPoint(b))return;ctx.strokeStyle=color;ctx.lineWidth=width;ctx.setLineDash(dash);ctx.beginPath();ctx.moveTo(...a);ctx.lineTo(...b);ctx.stroke();ctx.setLineDash([]);}
function graph(ctx,points,valid,color,gt=false){const selected=$('role').value;let edges=selected==='all'?E:[E[Number(selected)]];if(gt&&selected==='all')WEDGES.forEach(([a,b])=>{if(valid[a]&&valid[b])stroke(ctx,points[a],points[b],color,1.4,[5,5]);});edges.forEach(([a,b])=>{if(valid[a]&&valid[b])stroke(ctx,points[a],points[b],color,gt?1.8:2.2);});let relevant=selected==='all'?new Set([0,1,2,3,4,5,6,7]):new Set(E[Number(selected)]);points.slice(0,8).forEach((p,i)=>{if(!valid[i]||!validPoint(p)||!relevant.has(i))return;ctx.beginPath();ctx.arc(p[0],p[1],gt?3:4,0,Math.PI*2);ctx.strokeStyle=color;ctx.lineWidth=2;ctx.stroke();ctx.font='bold 12px Arial';ctx.lineWidth=3;ctx.strokeStyle='#101722';ctx.strokeText(String(i),p[0]+5,p[1]-5);ctx.fillStyle=color;ctx.fillText(String(i),p[0]+5,p[1]-5);});}
function setup(id,r){let c=$(id),margin=60;c.width=r.width+margin*2;c.height=r.height+margin*2;let ctx=c.getContext('2d');ctx.fillStyle='#252d38';ctx.fillRect(0,0,c.width,c.height);ctx.translate(margin,margin);ctx.drawImage(loadedImage,0,0,r.width,r.height);ctx.strokeStyle='#5c6878';ctx.lineWidth=1;ctx.strokeRect(0,0,r.width,r.height);graph(ctx,r.gt,r.gt_valid,colors.gt,true);return ctx;}
function infinite(ctx,line,r){if(!line||!line.every(validPoint))return;let a=line[0],b=line[1],dx=b[0]-a[0],dy=b[1]-a[1],norm=Math.hypot(dx,dy);if(norm<1e-9)return;let length=2*Math.hypot(r.width+120,r.height+120);stroke(ctx,[a[0]-dx/norm*length,a[1]-dy/norm*length],[a[0]+dx/norm*length,a[1]+dy/norm*length],colors.dht,1.8);}
function stats(points,valid,r){let errors=[],ngt=0;for(let i=0;i<8;i++){if(!r.gt_valid[i])continue;ngt++;if(valid[i]&&validPoint(points[i]))errors.push(Math.hypot(points[i][0]-r.gt[i][0],points[i][1]-r.gt[i][1]));}let sorted=[...errors].sort((a,b)=>a-b),med=sorted.length?(sorted[Math.floor((sorted.length-1)/2)]+sorted[Math.ceil((sorted.length-1)/2)])/2:null;return `예측 ${errors.length}/${ngt} 코너 · PCK@10px ${ngt?(100*errors.filter(v=>v<=10).length/ngt).toFixed(1):'—'}% · 조건부 median ${med===null?'—':med.toFixed(2)}px`;}
function draw(r){let model=$('model').value,seed=$('seed').value,mode=$('mode').value,b=r.baselines[model],f=r.fused[model][seed][mode],lambda=mode==='selected'?DATA.selected_lambda[model]:1;
 setup('truth',r);graph(setup('base',r),b.kps,b.valid,colors.base);graph(setup('fused',r),f.kps,b.valid,colors.fused);let ctx=setup('lines',r);r.lines[seed].forEach((line,i)=>{if($('role').value==='all'||i===Number($('role').value))infinite(ctx,line,r);});
 let modelLabel=$('model').selectedOptions[0].textContent;$('baseheading').textContent=`${modelLabel} 기준 모델 + 정답`;$('fuseheading').textContent=`${modelLabel} + DHT · λ=${lambda} · seed${seed}`;$('truthnote').textContent=`원본 ${r.width}×${r.height} · ${r.id}`;$('basenote').textContent=stats(b.kps,b.valid,r);$('fusenote').textContent=stats(f.kps,b.valid,r);$('linesnote').textContent=`8개 구조적 supporting line · seed${seed} · GT로 선을 선택하지 않음`;
 $('frameinfo').replaceChildren(...[r.population,r.group,`현재 설정 λ=${lambda}`,'GT 코너 번호 고정'].map(t=>{let s=document.createElement('span');s.textContent=t;return s;}));$('loadstatus').textContent='저장된 예측 표시 완료';$('loadstatus').className='small';window.reportReady=true;window.currentReportFrame=r.id;}
function render(){let r=DATA.records.find(r=>r.id===$('frame').value);if(!r)return;let token=++generation;window.reportReady=false;$('loadstatus').textContent='이미지를 불러오는 중…';if(currentRecord&&currentRecord.id===r.id&&loadedImage){draw(r);return;}let img=new Image();img.onload=()=>{if(token!==generation)return;loadedImage=img;currentRecord=r;draw(r);};img.onerror=()=>{$('loadstatus').textContent='이미지 읽기 실패: '+r.image;window.integrationReportError=r.image;};img.src=r.image;}
$('population').addEventListener('change',updateFrames);['frame','model','seed','mode','role'].forEach(id=>$(id).addEventListener('change',render));updateFrames();
</script></html>'''


if __name__ == '__main__':
    raise SystemExit(main())
