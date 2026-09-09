"""Render saved fusion diagnostics and deployable gate comparisons offline.

No inference, gate fitting, or prediction changes occur in this renderer.
Use --run-dir PATH_TO_LIVE --no-open during development. The only new report
is written below live/fusion_diagnosis_v1; the frozen integration report stays
unchanged. Without --no-open, the existing browser helper opens the new HTML.
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
MODELS = ['YOLO', 'DOPE_exact_resize']
MODEL_LABELS = {'YOLO': 'YOLO', 'DOPE_exact_resize': 'DOPE / corrected resize',
                'DOPE': 'DOPE / legacy resize'}
DIFFICULTIES = ['easy_le10', 'moderate_10_20', 'hard_gt20', 'missing']
DIFFICULTY_LABELS = {'easy_le10': 'Easy: point error ≤10px',
                     'moderate_10_20': 'Moderate: 10–20px',
                     'hard_gt20': 'Hard: point error >20px',
                     'missing': 'Missing point'}
POPULATIONS = ['real_dev', 'synth_test', 'cross_v4', 'synth_val']
ARMS = ['baseline', 'unconditional_lambda1', 'confidence_only', 'confidence_line_move']
ARM_LABELS = {'baseline': 'Points only', 'unconditional_lambda1': 'Always fuse / λ=1',
              'confidence_only': 'Point confidence gate',
              'confidence_line_move': 'Point + line + movement gate'}
EDGES = [[1, 2], [3, 0], [5, 6], [7, 4], [0, 4], [1, 5], [2, 6], [3, 7]]


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for part in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(part)
    return digest.hexdigest()


def esc(value):
    return html.escape(str(value))


def num(value, digits=2):
    return f'{float(value):.{digits}f}' if value is not None and np.isfinite(value) else '—'


def table(headers, rows):
    return ('<div class="tablewrap"><table><thead><tr>' +
            ''.join(f'<th>{esc(v)}</th>' for v in headers) + '</tr></thead><tbody>' +
            ''.join('<tr>' + ''.join(f'<td>{esc(v)}</td>' for v in row) + '</tr>' for row in rows) +
            '</tbody></table></div>')


def validate_payload(path, payload, live):
    for field, source in [('config_sha256', live / 'CONFIG.json'),
                          ('manifest_sha256', live / 'manifest.json')]:
        if field in payload and payload[field] != sha(source):
            raise ValueError(f'{path.name}: mismatched {field}')
    for field in ('source_sha256',):
        for name, digest in payload.get(field, {}).items():
            source = Path(name)
            if not source.is_absolute():
                source = (live if len(source.parts) == 1 else ROOT) / source
            if sha(source) != digest:
                raise ValueError(f'{path.name}: changed source {source}')


def build_visuals(manifest, lines, gated, corners):
    expected = [r['id'] for r in manifest['records']]
    if [r['id'] for r in lines['records']] != expected:
        raise ValueError('DHT line order differs from the frozen manifest')
    for model in MODELS:
        if [r['id'] for r in gated['models'][model]['records']] != expected:
            raise ValueError(f'{model}: gate prediction order differs from the manifest')
    index = {(r['model'], r['id'], str(r['seed']), r['corner']): r for r in corners['records']}
    rows = []
    for position, record in enumerate(manifest['records']):
        row = dict(id=record['id'], population=record['population'], group=record['group'],
                   image=Path(record['image']).resolve().as_uri(), width=record['width'],
                   height=record['height'], gt=record['gt_points'], gt_valid=record['gt_valid'],
                   models={}, lines={s: lines['records'][position]['seeds'][s]['lines'] for s in ('1', '2', '3')})
        for model in MODELS:
            prediction = gated['models'][model]['records'][position]
            value = dict(baseline=prediction['baseline'], seeds=prediction['seeds'],
                         signals=prediction['signals'], corners={})
            for seed in ('1', '2', '3'):
                value['corners'][seed] = []
                for corner in range(8):
                    diagnostic = index[(model, record['id'], seed, corner)]
                    value['corners'][seed].append({key: diagnostic[key] for key in
                        ('difficulty', 'baseline_error_px', 'incident_roles',
                         'incident_line_gt_distance_px', 'incident_baseline_residual_px',
                         'incident_normal_error_px', 'incident_tangent_error_px',
                         'incident_sin_angle', 'fixed1')})
            row['models'][model] = value
        rows.append(row)
    return dict(records=rows, edges=EDGES, populations=POPULATIONS,
                model_labels=MODEL_LABELS, arm_labels=ARM_LABELS,
                difficulty_labels=DIFFICULTY_LABELS)


def stat(row, key, percent=False, interval=False, digits=2):
    value = row['metrics'].get(key)
    if value is None or value.get('mean') is None:
        return '—'
    factor = 100 if percent else 1
    text = num(value['mean'] * factor, digits) + ('%' if percent else '')
    if interval and value['max'] - value['min'] > 1e-10:
        text += f" [{num(value['min'] * factor, digits)}, {num(value['max'] * factor, digits)}]"
    return text


def diagnosis_row(diagnosis, model, population='real_dev', difficulty='all', condition='fixed1', group='all'):
    return next(r for r in diagnosis['corner_summary'] if
                (r['model'], r['population'], r['difficulty'], r['condition'], r['group']) ==
                (model, population, difficulty, condition, group))


def mechanism_tables(diagnosis):
    rows = []
    for model in MODELS:
        summary = diagnosis_row(diagnosis, model)
        rows.append([MODEL_LABELS[model], stat(summary, 'movement_mean_px'),
                     stat(summary, 'wrong_direction_fraction_of_worsened', True),
                     stat(summary, 'overshoot_fraction_of_worsened', True)])
    result = table(['Real DEV52 / unconditional λ=1', 'Mean movement px',
                    'Wrong direction / worsened', 'Overshoot / worsened'], rows)
    result += '<p class="small">Wrong direction: movement initially points away from GT. Overshoot: movement initially points toward GT but is too large, so final error increases. Fractions use worsened observed corners as their denominator, averaged over three seeds.</p>'
    rows = []
    for model in MODELS:
        for family in ('height', 'depth'):
            row = next(r for r in diagnosis['constraint_summary'] if
                       (r['model'], r['population'], r['group'], r['difficulty'], r['role_family']) ==
                       (model, 'real_dev', 'all', 'all', family))
            rows.append([MODEL_LABELS[model], family,
                         stat(row, 'baseline_abs_normal_error_mean_px'),
                         stat(row, 'line_gt_distance_mean_px'),
                         stat(row, 'line_closer_than_baseline_in_normal_fraction', True)])
    result += '<details><summary>Is the line actually more accurate along the direction it constrains?</summary><p>A line constrains the point perpendicular to the line. The table compares line-to-GT distance with the baseline point error in that same normal direction. A single line does not determine the tangent coordinate.</p>'
    result += table(['Model', 'Incident line', 'Point normal error px', 'Line normal bias px', 'Line closer than point'], rows) + '</details>'
    quant = next(r for r in diagnosis['quantization_reference'] if
                 (r['population'], r['group'], r['role_family']) == ('real_dev', 'all', 'all'))
    result += (f'<p class="note">Grid resolution alone does not explain the observed tail. Rounding GT lines to the existing Hough grid gives '
               f"mean {num(quant['quantized_gt']['distance_mean_px'])}px / P90 {num(quant['quantized_gt']['distance_p90_px'])}px, "
               f"while learned DHT lines give mean {stat(quant['dht_learned'], 'distance_mean_px')}px / "
               f"P90 {stat(quant['dht_learned'], 'distance_p90_px')}px on the same 414 supported real side lines. "
               'The rounded-GT result is a diagnostic reference, not an attainable guarantee or a causal percentage decomposition.</p>')
    return result


def difficulty_tables(diagnosis):
    output = '<p><strong>More improved corners can still mean worse average error.</strong> Many small improvements can be outweighed by a few large failures; read the improvement fraction together with the before/after error.</p>'
    for population in POPULATIONS:
        rows = []
        for model in MODELS:
            for difficulty in DIFFICULTIES:
                try:
                    summary = diagnosis_row(diagnosis, model, population, difficulty)
                except StopIteration:
                    rows.append([MODEL_LABELS[model], DIFFICULTY_LABELS[difficulty],
                                 'No observations', '—', '—', '—', '—'])
                    continue
                rows.append([MODEL_LABELS[model], DIFFICULTY_LABELS[difficulty],
                             stat(summary, 'n_frames', digits=0),
                             stat(summary, 'n_gt_corners', digits=0),
                             stat(summary, 'baseline_mean_px') + ' → ' + stat(summary, 'fused_mean_px'),
                             stat(summary, 'delta_mean_px', interval=True),
                             stat(summary, 'improved_fraction', True)])
        content = table(['Model', 'GT-defined difficulty', 'Frames', 'Corners',
                         'Mean point error → fused px', 'Mean Δ px [seed range]', 'Improved corners'], rows)
        output += ('<h3>Real development / 52 unique frames</h3>' + content) if population == 'real_dev' else (
            f'<details><summary>{esc(population)} difficulty breakdown</summary>{content}</details>')
    output += '<p class="small">These are unconditional λ=1 results. Counts are not multiplied by the three DHT seeds. A frame can contribute corners to multiple difficulty bins, so frame counts across bins must not be added. Pixel errors condition on observed points; missing predictions remain missing.</p>'
    rows = []
    for model in MODELS:
        value = next(r for r in diagnosis['frame_paired_summary'] if
                     (r['model'], r['population'], r['group'], r['condition']) ==
                     (model, 'real_dev', 'all', 'fixed1'))
        rows.append([MODEL_LABELS[model], value['n_independent_frames'], num(value['mean_delta_px']),
                     ' to '.join(num(v) for v in value['mean_delta_frame_bootstrap95_px']),
                     f"{value['improved_frames']} / {value['worsened_frames']} / {value['unchanged_frames']}"])
    output += '<details><summary>Frame-level uncertainty for unconditional fusion</summary><p>Seeds are averaged within a frame before bootstrapping frames. The 52 real images remain 52 observations. Δ is fused minus baseline per-frame mean error; unchanged missing-diagonal penalties cancel.</p>'
    output += table(['Model', 'Unique frames', 'Mean Δ px', 'Frame bootstrap 95% interval', 'Better / worse / same'], rows) + '</details>'
    return output


def gate_row(results, model, arm, population='real_dev', group='all'):
    return next(r for r in results['seed_summary'] if
                (r['model'], r['arm'], r['population'], r['group']) == (model, arm, population, group))


def gate_tables(results):
    output = '<p class="note">Visibility confidence is not localization uncertainty. YOLO keypoint confidence is trained from label presence/visibility, not from the probability of a ≤10px coordinate error. Its synthetic-validation confidence values are near 0.999. The existing RLE sigma head was not evaluated here. Hough peak values are also uncalibrated geometric-accuracy proxies.</p>'
    for population in POPULATIONS:
        rows = []
        for model in MODELS:
            for arm in ARMS:
                aggregate = gate_row(results, model, arm, population)
                raw = next(r for r in results['summaries'] if
                           (r['model'], r['arm'], r['population'], r['group']) ==
                           (model, arm, population, 'all'))
                same = [r for r in results['summaries'] if
                        (r['model'], r['arm'], r['population'], r['group']) ==
                        (model, arm, population, 'all')]
                adjusted = np.mean([r['n_adjusted_corners'] for r in same])
                rows.append([MODEL_LABELS[model], ARM_LABELS[arm], raw['n_frames'],
                             stat(aggregate, 'corner_mean_px'), stat(aggregate, 'corner_p90_px'),
                             stat(aggregate, 'pck_10px', True),
                             num(raw['corner_coverage'] * 100) + '%', num(adjusted, 1)])
        content = table(['Model', 'Rule', 'Frames', 'Mean error px', 'P90 px', 'PCK10', 'Point coverage', 'Corners adjusted / seed'], rows)
        output += '<h3>Real development / same 52 frames</h3>' + content if population == 'real_dev' else (
            f'<details><summary>{esc(population)} gate results</summary>{content}</details>')
    output += '<p class="small">Gate summaries first compute each seed statistic, then average the three seeds. PCK includes missing predictions as failures; mean and P90 pixel errors condition on observed points. Gate-false points and all missing points retain their baseline values exactly. A selected baseline fallback performs no fusion.</p>'
    rows = []
    groups = sorted({r['group'] for r in results['summaries'] if r['population'] == 'real_dev' and r['group'] != 'all'})
    for group in groups:
        for model in MODELS:
            base = gate_row(results, model, 'baseline', group=group)
            fused = gate_row(results, model, 'confidence_line_move', group=group)
            rows.append([group, MODEL_LABELS[model],
                         stat(base, 'corner_mean_px') + ' → ' + stat(fused, 'corner_mean_px'),
                         stat(base, 'corner_p90_px') + ' → ' + stat(fused, 'corner_p90_px'),
                         stat(base, 'pck_10px', True) + ' → ' + stat(fused, 'pck_10px', True)])
    output += '<details><summary>Real capture groups / point baseline → point + line + movement gate</summary>' + table(
        ['Capture group', 'Model', 'Mean error px', 'P90 px', 'PCK10'], rows) + '</details>'
    return output


def rule_tables(selection):
    selected_models = selection.get('models', selection)
    rows = []
    for model in MODELS:
        for family in ('confidence_only', 'confidence_line_move'):
            choice = selected_models[model]['families'][family]
            rule = choice['selected_rule']
            if rule.get('kind') == 'baseline' or rule.get('gate_enabled') is False:
                description = 'Keep every baseline point / no fusion'
            else:
                constraints = [f"point confidence ≤ {num(rule['confidence_upper'], 6)}"]
                if family == 'confidence_line_move':
                    constraints += [f"min incident Hough peak ≥ {num(rule['line_quality_lower'], 6)}",
                                    f"proposed move / diagonal ≤ {num(rule['move_normalized_upper'], 6)}"]
                description = ' AND '.join(constraints)
            rows.append([MODEL_LABELS[model], ARM_LABELS[family], rule['id'], description,
                         num(choice['validation_score'], 6)])
    return ('<p>Threshold candidates come from prediction-only Q25/Q50/Q75/Q100 values on synthetic validation. '
            'Selection uses the predeclared capped diagonal-normalized point error, averaged over frames and seeds. '
            'The same λ=1 displacement proposal is either accepted or rejected. Real GT does not choose a threshold. '
            'The angle between lines is shown for diagnosis only and is not part of these selected gate rules.</p>' +
            table(['Model', 'Gate family', 'Selected rule', 'Prediction-only condition', 'Validation score'], rows))


def oracle_tables(diagnosis):
    rows = []
    for model in MODELS:
        for condition, label in [('perfect_lines_fixed1', 'GT-perfect lines / same λ=1'),
                                 ('oracle_min_baseline_fixed1', 'GT chooses point or fused per corner')]:
            row = diagnosis_row(diagnosis, model, condition=condition)
            rows.append([MODEL_LABELS[model], label, stat(row, 'baseline_mean_px'),
                         stat(row, 'fused_mean_px'), stat(row, 'improved_fraction', True)])
    return table(['Model', 'GT-dependent control', 'Point mean px', 'Oracle mean px', 'Improved corners'], rows)


def verdict(diagnosis, gates):
    easy = diagnosis_row(diagnosis, 'YOLO', difficulty='easy_le10')
    parts = [f"<strong>Good points are not safe under unconditional fusion.</strong> "
             f"YOLO's real corners initially within 10px move from mean {stat(easy, 'baseline_mean_px')}px "
             f"to {stat(easy, 'fused_mean_px')}px under λ=1. "]
    for model in MODELS:
        before = gate_row(gates, model, 'baseline')
        after = gate_row(gates, model, 'confidence_line_move')
        parts.append(f"{esc(MODEL_LABELS[model])}: point-only mean {stat(before, 'corner_mean_px')}px → "
                     f"validation-selected joint gate {stat(after, 'corner_mean_px')}px on real DEV52. ")
    parts.append('The tested gates reduce the damage from blind fusion but do not improve observed point-only mean error here. These are diagnostics on reused development images, not a new held-out result.')
    return ''.join(parts)


def confidence_intervals(audit):
    if not audit.get('PASS'):
        raise ValueError('Statistical audit has not passed')
    rows = []
    for row in audit['paired_bootstrap']:
        rows.append([MODEL_LABELS[row['model']], ARM_LABELS[row['arm']],
                     num(row['mean_observed_corner_error_delta_px']),
                     ' to '.join(num(v) for v in row['mean_error_delta_ci95']),
                     num(row['pck10_delta_percentage_points']),
                     ' to '.join(num(v) for v in row['pck10_delta_ci95'])])
    return ('<details open><summary>Paired frame bootstrap / gate minus point baseline</summary>' +
            table(['Model', 'Gate', 'Mean error Δ px', '95% interval', 'PCK10 Δ pp', '95% interval'], rows) +
            '<p class="small">5,000 paired image resamples within the three real capture groups; three DHT seeds are averaged inside each image. '
            'These intervals describe these 52 reused DEV images. Correlated frames/scenes and model uncertainty are not fully represented, '
            'and the intervals do not establish independent generalization. PCK and mean error can move in different directions.</p></details>')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path, required=True, help='The frozen live integration directory')
    parser.add_argument('--no-open', action='store_true')
    args = parser.parse_args()
    live = args.run_dir.resolve()
    destination = live / 'fusion_diagnosis_v1'
    names = ['DIAGNOSIS.json', 'CORNER_DIAGNOSTICS.json', 'GEOMETRY_AUDIT.json',
             'GATE_RESULTS.json', 'GATED_PREDICTIONS.json', 'CONFIG_GATE.json',
             'GATE_AUDIT.json', 'STATISTICAL_AUDIT.json', 'CONFIDENCE_SEMANTICS.json']
    frozen_before = {str(p): sha(p) for p in live.iterdir() if p.is_file() and p.suffix in ('.json', '.html')}
    artifacts = {name: read(destination / name) for name in names}
    for name, payload in artifacts.items():
        validate_payload(destination / name, payload, live)
    diagnosis, gates = artifacts['DIAGNOSIS.json'], artifacts['GATE_RESULTS.json']
    if not diagnosis.get('complete') or not gates.get('complete'):
        raise ValueError('Diagnosis/gate evaluation must be complete before rendering')
    if artifacts['GEOMETRY_AUDIT.json'].get('status') != 'PASS':
        raise ValueError('The saved geometry audit has not passed')
    if not artifacts['GATE_AUDIT.json'].get('PASS'):
        raise ValueError('The saved gate audit has not passed')
    manifest, lines = read(live / 'manifest.json'), read(live / 'DHT_LINES.json')
    data = build_visuals(manifest, lines, artifacts['GATED_PREDICTIONS.json'], artifacts['CORNER_DIAGNOSTICS.json'])
    substitutions = dict(VERDICT=verdict(diagnosis, gates), MECHANISM_TABLES=mechanism_tables(diagnosis),
                         DIFFICULTY_TABLES=difficulty_tables(diagnosis),
                         GATE_TABLES=gate_tables(gates) + confidence_intervals(artifacts['STATISTICAL_AUDIT.json']),
                         RULES=rule_tables(gates['selection']), ORACLE=oracle_tables(diagnosis),
                         DATA=json.dumps(data, separators=(',', ':'), allow_nan=False).replace('<', '\\u003c'))
    links = names + ['PROTOCOL.json', 'CONFIDENCE_SEMANTICS.json', 'GATE_SELECTION.json', 'GATE_AUDIT.json']
    substitutions['LINKS'] = '<p>' + ' · '.join(f'<a href="{name}">{name}</a>' for name in links if (destination / name).exists()) + '</p>'
    substitutions['LINKS'] += '<p><a href="../integration_report.html">Frozen point + DHT integration report</a></p>'
    page = PAGE
    for key, value in substitutions.items():
        page = page.replace(f'__{key}__', value)
    path = destination / 'fusion_diagnosis.html'
    path.write_text(page, encoding='utf-8')
    if any(sha(Path(p)) != digest for p, digest in frozen_before.items()):
        raise ValueError('An original frozen artifact changed while rendering')
    report = dict(complete=True, n_frames=len(data['records']), models=MODELS, seeds=[1, 2, 3],
                  html_sha256=sha(path), report_code_sha256=sha(Path(__file__)),
                  input_sha256={name: sha(destination / name) for name in names},
                  frozen_original_artifacts_unchanged=True,
                  example_selection='Default manifest order; optional GT-ranked unconditional error changes labeled posthoc')
    (destination / 'REPORT_RENDER.json').write_text(json.dumps(report, indent=2) + '\n')
    print(f'Rendered {path} with {len(data["records"])} frames', flush=True)
    if not args.no_open:
        sys.path.insert(0, str(HERE.parent / 'deep_hough_side_v1'))
        from visualize import open_gallery
        open_gallery(path)
    return 0


PAGE = r'''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Why point–line fusion fails · Selective DHT diagnosis</title>
<style>
:root{font-family:Arial,Helvetica,sans-serif;background:#101722;color:#e6edf7;color-scheme:dark}*{box-sizing:border-box}body{margin:0}main{max-width:1560px;margin:auto;padding:30px 28px 65px}h1{font-size:32px;letter-spacing:-.6px;margin:10px 0 12px}h2{font-size:23px;margin:0 0 12px}h3{font-size:15px;margin:0 0 9px}p{color:#b7c5d8;line-height:1.6;margin:9px 0}a{color:#7acfff}.eyebrow{font-size:12px;letter-spacing:1.6px;color:#89bfee}section{padding:22px;background:#172130;border:1px solid #304058;border-radius:13px;margin-top:22px}.flow{display:grid;grid-template-columns:repeat(4,1fr);gap:12px;margin:22px 0}.step{background:#192536;border:1px solid #344660;border-radius:10px;padding:17px}.step span{color:#7994b6;font-size:12px}.step strong{display:block;font-size:17px;margin:8px 0}.step p{font-size:13px;margin:0}.note{padding:13px 17px;border-left:3px solid #78bde5;background:#1d3040;border-radius:5px}.posthoc{border-color:#e7bd73;background:#332c21}.verdict{font-size:16px}.small{font-size:12px}.tablewrap{overflow:auto}table{border-collapse:collapse;width:100%;font-size:13px;white-space:nowrap}th,td{padding:10px 11px;border-bottom:1px solid #31425c;text-align:right}th{font-weight:400;color:#92a8c3}th:first-child,td:first-child{text-align:left}tbody tr:hover{background:#203148}.grid2{display:grid;grid-template-columns:1fr 1fr;gap:18px}details{margin:15px 0}summary{cursor:pointer;color:#94d4ff;padding:8px 0}code{font-family:monospace;color:#cde9fc}.controls{display:flex;gap:10px;flex-wrap:wrap;margin:18px 0}.controls label{display:flex;flex-direction:column;gap:6px;font-size:11px;color:#a8bed7}.controls select{max-width:100%}.framecontrol{flex:1;min-width:285px}select{background:#25364c;color:#e6edf7;border:1px solid #475f7d;border-radius:6px;padding:9px 10px;font:inherit;cursor:pointer}.legend{display:flex;gap:18px;flex-wrap:wrap;font-size:12px;margin:12px 0}.legend span:before{content:"";display:inline-block;width:22px;height:3px;vertical-align:middle;margin-right:6px;background:var(--c)}.panels{display:grid;grid-template-columns:repeat(6,1fr);gap:12px}.panel{grid-column:span 2;background:#101923;padding:11px;border:1px solid #34455d;border-radius:9px;min-width:0}.panel:nth-child(n+4){grid-column:span 3}.panel canvas{display:block;width:100%;height:auto}.panel h3{font-weight:400;font-size:13px;color:#d1ddef}.panel p{font-size:11px;margin:7px 0 0;min-height:28px}.chips{display:flex;gap:7px;flex-wrap:wrap;margin:12px 0}.chips span{border:1px solid #425978;border-radius:5px;font-size:11px;padding:5px 8px}.metricbar{height:7px;background:#334257;border-radius:4px;overflow:hidden;margin-top:6px}.metricbar i{display:block;height:100%;background:#ad8aff}.formula{font-family:monospace;font-size:15px;color:#d1e6fa;padding:15px;background:#101923;border-radius:6px}.oracle{border:1px dashed #b1996e;padding:14px}.explanation{max-width:1050px}
@media(max-width:1000px){.flow{grid-template-columns:1fr 1fr}.panels{grid-template-columns:1fr 1fr}.panel,.panel:nth-child(n+4){grid-column:auto}.panel:last-child{grid-column:1/-1}.grid2{grid-template-columns:1fr}}@media(max-width:650px){main{padding:20px 12px}.flow,.panels{grid-template-columns:1fr}.panel:last-child{grid-column:auto}h1{font-size:27px}section{padding:15px}}
</style><main>
<div class="eyebrow">FROZEN PREDICTIONS · POSTHOC DIAGNOSIS · VALIDATION-ONLY GATE SELECTION</div>
<h1>When should a line move a predicted corner?</h1>
<p class="explanation">A line can help a weak point, but it can also move a good point away from the target. This report separates what ground truth reveals after the fact from what a deployable gate can decide using predictions alone.</p>
<div class="flow"><div class="step"><span>01 / MECHANISM</span><strong>Two lines affect every point</strong><p>Unconditional fusion uses the two incident DHT lines even when the point is already accurate.</p></div><div class="step"><span>02 / DIAGNOSIS ONLY</span><strong>Split by point error</strong><p>GT-based easy, moderate, hard and missing bins reveal which corners help or harm the aggregate.</p></div><div class="step"><span>03 / DEPLOYABLE RULE</span><strong>Accept or keep the point</strong><p>Prediction-only confidence, line quality and proposed movement determine whether to fuse.</p></div><div class="step"><span>04 / SAME-FRAME EVIDENCE</span><strong>Inspect the displacement</strong><p>Compare points, unconditional fusion and the selected gate on exactly the same image.</p></div></div>
<div class="note verdict">__VERDICT__</div>
<section><h2>1. Why unconditional fusion can make a correct point worse</h2>
<div class="grid2"><div><div class="formula">q = argmin ‖q − p‖² + λ ∑ (nᵢ · q − ρᵢ)²</div><p>Each observed corner has two incident lines: one height line and one depth line. The original fusion uses the same λ for every observed corner. It does not ask whether the point is already correct or whether the line is more reliable.</p></div><div><div class="formula">Δ squared error = 2 (p − GT) · (q − p) + ‖q − p‖²</div><p>A large movement is useful only when its direction and size correct the existing error. Agreement between two predictions is not proof that either is right. Missing corners remain missing.</p></div></div>
__MECHANISM_TABLES__</section>
<section><h2>2. Difficulty bins explain the failure pattern</h2><p class="note posthoc">These bins use the baseline corner's measured GT error: easy ≤10px, moderate 10–20px, hard &gt;20px, or missing. They are posthoc diagnostic labels and are never inputs to the gate.</p>__DIFFICULTY_TABLES__</section>
<section><h2>3. Can a prediction-only gate choose better?</h2><p>The candidate rules and thresholds are selected on synthetic validation only. The actual-image batch-1 predictions, semantic corner IDs and confidence masks are frozen. Confidence-only gating is shown separately from adding line quality and a movement bound.</p>__GATE_TABLES__
<details><summary>Selected rules and synthetic-validation selection record</summary>__RULES__</details>
<details><summary>Oracle comparison — uses GT, cannot be deployed</summary><div class="oracle"><p>These controls deliberately use ground truth to illustrate available headroom. Their performance does not establish a working gate and must not be compared as a deployable method.</p>__ORACLE__</div></details></section>
<section id="examples"><h2>4. Same-frame evidence</h2><p>Explore all 692 frames. The default is the first real-development frame in manifest order. The optional “most harmful/helpful” ranking uses GT error changes under unconditional λ=1 and is a posthoc demonstration, not a model-selection rule.</p>
<div class="controls"><label>Population<select id="population"></select></label><label>Model<select id="model"><option>YOLO</option><option value="DOPE_exact_resize">DOPE / corrected resize</option></select></label><label>DHT seed<select id="seed"><option>1</option><option>2</option><option>3</option></select></label><label>Gate family<select id="family"><option value="confidence_line_move">Point + line + movement</option><option value="confidence_only">Point confidence only</option></select></label><label>Corner<select id="corner"><option value="all">All 8 corners</option></select></label><label>GT difficulty (posthoc)<select id="difficulty"><option value="all">All difficulties</option></select></label><label>Example order (posthoc)<select id="ranking"><option value="manifest">Manifest order</option><option value="harm">Most harmful unconditional move</option><option value="help">Most helpful unconditional move</option></select></label><label>View<select id="zoom"><option value="full">Full image</option><option value="focus">GT crop / diagnostic only</option></select></label><label class="framecontrol">Frame<select id="frame"></select></label></div>
<div class="legend"><span style="--c:#44f286">GT / structural corners</span><span style="--c:#57c7ff">Point baseline</span><span style="--c:#ff6da8">Unconditional λ=1</span><span style="--c:#b09aff">Selected gate</span><span style="--c:#ffd166">DHT supporting lines</span></div>
<div id="frameinfo" class="chips"></div><p id="filterinfo" class="small"></p>
<div class="panels"><div class="panel"><h3>Original image + GT</h3><canvas id="truth"></canvas><p id="truthnote"></p></div><div class="panel"><h3>Point baseline + GT</h3><canvas id="baseline"></canvas><p id="basenote"></p></div><div class="panel"><h3>Always fuse / λ=1 + GT</h3><canvas id="unconditional"></canvas><p id="uncondnote"></p></div><div class="panel"><h3 id="gateheading">Validation-selected gate + GT</h3><canvas id="gated"></canvas><p id="gatenote"></p></div><div class="panel"><h3>DHT lines + displacement arrows</h3><canvas id="movement"></canvas><p>Pink: baseline → unconditional. Purple: baseline → gated. Lines are structural predictions, not attention.</p></div></div>
<p id="loadstatus" class="small">Loading image…</p><div id="cornertable"></div>
<p class="small">The displayed metrics cover the highlighted corners. A difficulty filter keeps frames with at least one matching corner. A GT crop only changes the view; GT never alters the predictions or gate decisions. Gray canvas is outside the original image. DHT does not provide physical visible-edge labels.</p></section>
<section><h2>Scope and reproducibility</h2><p>Real development data: 52 canonical images in three capture groups; no sealed final-test data. No network was retrained. Synthetic pretraining overlap with evaluation remains unverified. An oracle or a GT-defined difficulty bin is not a deployable signal. These are 2D corner/line measurements and do not establish improved 6D pose.</p>__LINKS__</section>
</main><script id="payload" type="application/json">__DATA__</script>
<script>
'use strict';
const D=JSON.parse(document.getElementById('payload').textContent),$=id=>document.getElementById(id),C={gt:'#44f286',base:'#57c7ff',uncond:'#ff6da8',gate:'#b09aff',line:'#ffd166'};
let image=null,currentId=null,generation=0;
function options(node,rows){node.replaceChildren(...rows.map(([value,label])=>{let option=document.createElement('option');option.value=value;option.textContent=label;return option;}));}
options($('population'),D.populations.map(p=>[p,`${p} (${D.records.filter(r=>r.population===p).length})`]));
for(let i=0;i<8;i++){$('corner').append(new Option(String(i),String(i)));}Object.entries(D.difficulty_labels).forEach(([v,t])=>$('difficulty').append(new Option(t,v)));
function valid(p){return p&&p.length===2&&p.every(Number.isFinite);}
function diagnostics(r){return r.models[$('model').value].corners[$('seed').value];}
function selectedCorners(r){let x=diagnostics(r);return [...Array(8).keys()].filter(i=>($('corner').value==='all'||Number($('corner').value)===i)&&($('difficulty').value==='all'||x[i].difficulty===$('difficulty').value));}
function score(r){let a=selectedCorners(r).map(i=>diagnostics(r)[i].fixed1.delta_error_px).filter(Number.isFinite);return a.length?a.reduce((x,y)=>x+y,0)/a.length:null;}
function updateFrames(){let old=$('frame').value,rows=D.records.filter(r=>r.population===$('population').value&&selectedCorners(r).length);let ranking=$('ranking').value;if(ranking!=='manifest'){rows=rows.filter(r=>score(r)!==null).sort((a,b)=>(ranking==='harm'?-1:1)*(score(a)-score(b)));}options($('frame'),rows.map(r=>[r.id,`${r.group} · ${r.id}${ranking==='manifest'?'':` · Δ ${score(r).toFixed(2)}px`}`]));if(ranking==='manifest'&&rows.some(r=>r.id===old))$('frame').value=old;render();}
function stroke(ctx,a,b,color,width=2,dash=[]){if(!valid(a)||!valid(b))return;ctx.strokeStyle=color;ctx.lineWidth=width;ctx.setLineDash(dash);ctx.beginPath();ctx.moveTo(...a);ctx.lineTo(...b);ctx.stroke();ctx.setLineDash([]);}
function graph(ctx,p,mask,color,r){let selected=new Set(selectedCorners(r));D.edges.forEach(([a,b])=>{if(selected.has(a)&&selected.has(b)&&mask[a]&&mask[b])stroke(ctx,p[a],p[b],color,1.8);});p.slice(0,8).forEach((point,i)=>{if(!selected.has(i)||!mask[i]||!valid(point))return;ctx.beginPath();ctx.arc(...point,3,0,Math.PI*2);ctx.lineWidth=1.7;ctx.strokeStyle=color;ctx.stroke();ctx.font='bold 11px Arial';ctx.strokeStyle='#101722';ctx.lineWidth=3;ctx.strokeText(String(i),point[0]+4,point[1]-4);ctx.fillStyle=color;ctx.fillText(String(i),point[0]+4,point[1]-4);});}
function crop(r){if($('zoom').value==='full')return[-55,-55,r.width+110,r.height+110];let points=selectedCorners(r).map(i=>r.gt[i]).filter(valid);if(!points.length)return[-55,-55,r.width+110,r.height+110];let xs=points.map(p=>p[0]),ys=points.map(p=>p[1]),cx=(Math.min(...xs)+Math.max(...xs))/2,cy=(Math.min(...ys)+Math.max(...ys))/2;let width=Math.max(160,Math.max(...xs)-Math.min(...xs)+80),height=Math.max(120,Math.max(...ys)-Math.min(...ys)+80);width=Math.max(width,height*4/3);height=width*3/4;return[cx-width/2,cy-height/2,width,height];}
function setup(id,r){let canvas=$(id);canvas.width=800;canvas.height=600;let ctx=canvas.getContext('2d');ctx.fillStyle='#26313e';ctx.fillRect(0,0,800,600);let [x,y,w,h]=crop(r),scale=Math.min(800/w,600/h);ctx.translate((800-w*scale)/2,(600-h*scale)/2);ctx.scale(scale,scale);ctx.translate(-x,-y);ctx.drawImage(image,0,0,r.width,r.height);ctx.strokeStyle='#506074';ctx.lineWidth=1;ctx.strokeRect(0,0,r.width,r.height);graph(ctx,r.gt,r.gt_valid,C.gt,r);return ctx;}
function arrow(ctx,a,b,color){if(!valid(a)||!valid(b))return;let dx=b[0]-a[0],dy=b[1]-a[1],n=Math.hypot(dx,dy);if(n<.05)return;stroke(ctx,a,b,color,2.2);let ux=dx/n,uy=dy/n,len=Math.min(6,n*.5);stroke(ctx,b,[b[0]-len*ux+len*.5*uy,b[1]-len*uy-len*.5*ux],color,2.2);stroke(ctx,b,[b[0]-len*ux-len*.5*uy,b[1]-len*uy+len*.5*ux],color,2.2);}
function infinite(ctx,line,r){if(!line||!line.every(valid))return;let [a,b]=line,dx=b[0]-a[0],dy=b[1]-a[1],n=Math.hypot(dx,dy),length=4*Math.hypot(r.width,r.height);if(n<1e-9)return;stroke(ctx,[a[0]-length*dx/n,a[1]-length*dy/n],[a[0]+length*dx/n,a[1]+length*dy/n],C.line,1.4);}
function errors(r,p,mask){return selectedCorners(r).filter(i=>r.gt_valid[i]).map(i=>mask[i]&&valid(p[i])?Math.hypot(p[i][0]-r.gt[i][0],p[i][1]-r.gt[i][1]):null);}
function stats(r,p,mask){let all=errors(r,p,mask),finite=all.filter(Number.isFinite),mean=finite.length?finite.reduce((a,b)=>a+b,0)/finite.length:null;return `${finite.length}/${all.length} points · PCK10 ${all.length?(100*finite.filter(e=>e<=10).length/all.length).toFixed(1):'—'}% · mean ${mean===null?'—':mean.toFixed(2)}px`;}
function format(v,d=2){return Number.isFinite(v)?v.toFixed(d):'—';}
function draw(r){let model=$('model').value,seed=$('seed').value,family=$('family').value,p=r.models[model],base=p.baseline,u=p.seeds[seed].unconditional_lambda1,g=p.seeds[seed][family],chosen=selectedCorners(r);setup('truth',r);graph(setup('baseline',r),base.kps,base.kp_valid,C.base,r);graph(setup('unconditional',r),u.kps,u.kp_valid,C.uncond,r);graph(setup('gated',r),g.kps,g.kp_valid,C.gate,r);let ctx=setup('movement',r);let roleSet=new Set(chosen.flatMap(i=>p.corners[seed][i].incident_roles));r.lines[seed].forEach((line,i)=>{if(roleSet.has(i))infinite(ctx,line,r);});graph(ctx,base.kps,base.kp_valid,C.base,r);chosen.forEach(i=>{if(base.kp_valid[i]){arrow(ctx,base.kps[i],u.kps[i],C.uncond);arrow(ctx,base.kps[i],g.kps[i],C.gate);}});
 $('truthnote').textContent=`${r.width}×${r.height} · original-image coordinates`;$('basenote').textContent=stats(r,base.kps,base.kp_valid);$('uncondnote').textContent=stats(r,u.kps,u.kp_valid);$('gatenote').textContent=stats(r,g.kps,g.kp_valid)+` · ${chosen.filter(i=>g.gate[i]).length} gate pass / ${chosen.filter(i=>g.adjusted[i]).length} moved`;$('gateheading').textContent=D.arm_labels[family]+' + GT';
 $('frameinfo').replaceChildren(...[r.population,r.group,`seed ${seed}`,`${chosen.length} highlighted corners`,'GT labels are diagnostic only'].map(t=>{let el=document.createElement('span');el.textContent=t;return el;}));$('filterinfo').textContent=$('ranking').value==='manifest'?'Example order: frozen manifest.':`Posthoc example rank: ${$('ranking').selectedOptions[0].textContent}; mean unconditional Δ over highlighted observed corners = ${format(score(r))}px.`;
 let header=['Corner','GT difficulty','Point error px','Uncond. Δ px','Gate pass','Gated Δ px','Point confidence','Line quality','Proposed move / diagonal','sin(angle)'];let rows=chosen.map(i=>{let c=p.corners[seed][i],signal=p.signals[seed],ge=base.kp_valid[i]&&valid(g.kps[i])&&r.gt_valid[i]?Math.hypot(g.kps[i][0]-r.gt[i][0],g.kps[i][1]-r.gt[i][1])-c.baseline_error_px:null;return[i,D.difficulty_labels[c.difficulty]||c.difficulty,format(c.baseline_error_px),format(c.fixed1.delta_error_px),g.gate[i]?'yes':'no',format(ge),format(signal.point_confidence[i],4),format(signal.line_quality[i],4),format(signal.proposed_move_normalized[i],4),format(signal.sin_incident_angle[i],4)];});let table=document.createElement('table'),thead=document.createElement('thead'),tr=document.createElement('tr');header.forEach(v=>{let cell=document.createElement('th');cell.textContent=v;tr.append(cell);});thead.append(tr);table.append(thead);let tbody=document.createElement('tbody');rows.forEach(row=>{let tr=document.createElement('tr');row.forEach(v=>{let cell=document.createElement('td');cell.textContent=v;tr.append(cell);});tbody.append(tr);});table.append(tbody);$('cornertable').className='tablewrap';$('cornertable').replaceChildren(table);$('loadstatus').textContent='Saved predictions displayed. No inference or GT-based gating occurs in this viewer.';window.diagnosisReady=true;window.currentDiagnosisFrame=r.id;}
function render(){let r=D.records.find(r=>r.id===$('frame').value),token=++generation;window.diagnosisReady=false;if(!r){document.querySelectorAll('canvas').forEach(c=>{let ctx=c.getContext('2d');ctx.fillStyle='#26313e';ctx.fillRect(0,0,c.width,c.height);});$('cornertable').replaceChildren();$('frameinfo').replaceChildren();['truthnote','basenote','uncondnote','gatenote'].forEach(id=>$(id).textContent='');$('loadstatus').textContent='No frames in this diagnostic subset.';window.diagnosisReady=true;window.currentDiagnosisFrame=null;return;}$('loadstatus').textContent='Loading image…';if(currentId===r.id&&image){draw(r);return;}let next=new Image();next.onload=()=>{if(token!==generation)return;image=next;currentId=r.id;draw(r);};next.onerror=()=>{$('loadstatus').textContent='Image load failed: '+r.image;window.diagnosisError=r.image;};next.src=r.image;}
['population','model','seed','corner','difficulty','ranking'].forEach(id=>$(id).addEventListener('change',updateFrames));['frame','family','zoom'].forEach(id=>$(id).addEventListener('change',render));updateFrames();
</script></html>'''


if __name__ == '__main__':
    raise SystemExit(main())
