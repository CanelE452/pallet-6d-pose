"""Render frozen YOLO/RLE and DHT uncertainty fusion results without inference.

Use --run-dir PATH_TO_LIVE --no-open while developing or when a driver owns
the final browser opening. Only uncertainty_fusion_v1 report files are written.
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
POPS = ['real_dev', 'synth_test', 'cross_v4', 'synth_val']
ARMS = ['baseline', 'previous_joint_gate', 'unconditional_lambda1',
        'point_sigma_only', 'line_uncertainty_only', 'point_line_uncertainty']
LABELS = {'baseline': 'YOLO points', 'previous_joint_gate': 'Previous joint gate',
          'unconditional_lambda1': 'Unconditional / λ=1',
          'point_sigma_only': 'Point σ only', 'line_uncertainty_only': 'Line uncertainty only',
          'point_line_uncertainty': 'Point + line uncertainty'}
EDGES = [[1, 2], [3, 0], [5, 6], [7, 4], [0, 4], [1, 5], [2, 6], [3, 7]]
INCIDENT = [[r for r, edge in enumerate(EDGES) if corner in edge] for corner in range(8)]
# This is fusion.INCIDENT_ROLES order: height (0..3), then depth (4..7).


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda: handle.read(1048576), b''):
            digest.update(chunk)
    return digest.hexdigest()


def esc(value):
    return html.escape(str(value))


def number(value, digits=2):
    return f'{float(value):.{digits}f}' if value is not None and np.isfinite(value) else '—'


def table(headers, rows):
    return ('<div class="tablewrap"><table><thead><tr>' + ''.join(f'<th>{esc(v)}</th>' for v in headers)
            + '</tr></thead><tbody>' + ''.join('<tr>' + ''.join(f'<td>{esc(v)}</td>' for v in row)
                                            + '</tr>' for row in rows) + '</tbody></table></div>')


def metric(row, key):
    value = row.get('metrics', row).get(key)
    return value.get('mean') if isinstance(value, dict) else value


def aggregate(results, arm, population='real_dev', group='all'):
    return next(row for row in results['seed_summary'] if
                (row['arm'], row['population'], row['group']) == (arm, population, group))


def result_tables(results):
    output = ''
    for population in POPS:
        rows = []
        for arm in ARMS:
            row = aggregate(results, arm, population)
            raw = next(r for r in results['summaries'] if
                       (r['arm'], r['population'], r['group']) == (arm, population, 'all'))
            rows.append([LABELS[arm], raw['n_frames'], number(metric(row, 'corner_mean_px')),
                         number(metric(row, 'corner_p90_px')),
                         number(100 * metric(row, 'pck_10px')) + '%',
                         number(100 * metric(row, 'corner_coverage')) + '%'])
        content = table(['Method', 'Frames', 'Mean error px', 'P90 px', 'PCK10', 'Point coverage'], rows)
        output += ('<h3>Real DEV / 52 reused images</h3>' + content) if population == 'real_dev' else (
            f'<details><summary>{esc(population)}</summary>{content}</details>')
    output += '<p class="small">Each DHT seed is evaluated separately; displayed statistics average the three seeds. Mean/P90 errors condition on observed corners. PCK10 counts missing predictions as failures. The original point-validity mask and missing predictions are preserved.</p>'
    rows = []
    for group in sorted({r['group'] for r in results['summaries']
                         if r['population'] == 'real_dev' and r['group'] != 'all'}):
        before, after = (aggregate(results, arm, group=group)
                         for arm in ('baseline', 'line_uncertainty_only'))
        rows.append([group, number(metric(before, 'corner_mean_px')) + ' → ' + number(metric(after, 'corner_mean_px')),
                     number(metric(before, 'corner_p90_px')) + ' → ' + number(metric(after, 'corner_p90_px')),
                     number(100 * metric(before, 'pck_10px')) + '% → ' + number(100 * metric(after, 'pck_10px')) + '%'])
    output += '<details><summary>Real capture groups / baseline → line uncertainty only</summary>' + table(
        ['Capture group', 'Mean error px', 'P90 px', 'PCK10'], rows) + '</details>'
    return output


def verdict(results):
    a, b, c = (aggregate(results, arm) for arm in ('baseline', 'point_sigma_only', 'line_uncertainty_only'))
    joint_rule = results['selection']['families']['point_line_uncertainty']['selected_rule']
    joint = ('The combined point + line family selected λ=0 and keeps the baseline.'
             if joint_rule['lambda'] == 0 else f'The combined family selected λ={number(joint_rule["lambda"], 4)}.')
    shifts = []
    for pop in ('synth_test', 'cross_v4'):
        before, after = (aggregate(results, arm, pop) for arm in ('baseline', 'line_uncertainty_only'))
        shifts.append(f'{pop} {number(metric(before, "corner_mean_px"))} → {number(metric(after, "corner_mean_px"))}px')
    return (f'<strong>Line uncertainty alone gives the clearest improvement on reused real DEV52.</strong> '
            f'Mean error: baseline {number(metric(a, "corner_mean_px"))}px, point-σ-only '
            f'{number(metric(b, "corner_mean_px"))}px, line-only {number(metric(c, "corner_mean_px"))}px. '
            f'Line-only PCK10: {number(100 * metric(a, "pck_10px"))}% → {number(100 * metric(c, "pck_10px"))}%. '
            f'{joint} Line-only mean error elsewhere: {"; ".join(shifts)}. '
            'The cross_v4 result worsens; this is not a general improvement across domains. All calibration and fusion choices use synthetic validation.')


def selection_tables(selection, calibration):
    rows = []
    for arm in ('point_sigma_only', 'line_uncertainty_only', 'point_line_uncertainty'):
        choice = selection['families'][arm]
        rule = choice['selected_rule']
        cap = rule['max_move_diagonal_fraction']
        rows.append([LABELS[arm], number(rule['lambda'], 4),
                     'None' if cap is None else number(100 * cap) + '% image diagonal',
                     number(choice['selection_score'], 7),
                     'Baseline fallback / no movement' if rule['lambda'] == 0 else 'Weighted solve'])
    output = '<p>Calibration uses 128 synthetic-validation images; disjoint selection uses the other 128. The score is capped diagonal-normalized point error averaged over frames and seeds. The λ grid includes zero, so selection can retain the baseline. The optional displacement cap clips the solved movement; it does not alter the original point validity.</p>'
    output += table(['Family', 'Selected λ', 'Movement cap', 'Selection score', 'Behavior'], rows)
    rows = []
    for axis, alpha, constant in zip(('x', 'y'), calibration['point_alpha_xy'], calibration['constant_point_variance_xy']):
        rows.append(['Point / ' + axis, number(alpha, 6), number(constant)])
    for family, beta, constant in zip(('height', 'depth'), calibration['line_beta_height_depth'], calibration['constant_line_variance_height_depth']):
        rows.append(['Line / ' + family, number(beta, 6), number(constant)])
    output += table(['Calibrated quantity', 'Variance multiplier', 'Constant variance for ablation / px²'], rows)
    output += '<p class="small">Working point variance = max(α × max(raw σ², 1), 1) per axis. Working line variance = max(β × max(zᵀMz, 1), 1) per role family, at the baseline point z=(x,y,1). All variance floors are 1px². M is the full valid-bin distribution’s second moment about the argmax line, with line-normal signs aligned; it includes other modes. These are squared-error working scales, not certified Gaussian variances.</p>'
    output += '<details><summary>Exact frozen selection record</summary><pre>' + esc(json.dumps(selection, indent=2)) + '</pre></details>'
    return output


def reliability_data(reliability):
    """Normalize the evaluator's diagnostic rows for offline tables and plots."""
    rows = reliability.get('rows', reliability.get('summaries', reliability.get('records', [])))
    if not isinstance(rows, list):
        raise ValueError('Expected a list of saved reliability rows')
    return rows


def confidence_intervals(audit):
    if not audit.get('PASS'):
        raise ValueError('Statistical audit has not passed')
    rows = []
    for row in audit['paired_bootstrap']:
        rows.append([LABELS[row['arm']], LABELS[row['reference']], row['n_unique_frames'],
                     number(row['mean_observed_corner_error_delta_px']),
                     ' to '.join(number(v) for v in row['mean_error_delta_ci95']),
                     number(row['pck10_delta_percentage_points']),
                     ' to '.join(number(v) for v in row['pck10_delta_ci95'])])
    return ('<details open><summary>Paired real-frame differences / method minus reference</summary>' +
            table(['Method', 'Reference', 'Frames', 'Mean error Δ px', '95% interval',
                   'PCK10 Δ pp', '95% interval'], rows) +
            '<p class="small">5,000 paired resamples within real capture groups; the three seeds are averaged inside each image. '
            'Negative mean-error Δ favors the method; positive PCK10 Δ favors the method. These intervals condition on the reused 52 DEV images '
            'and do not capture independent domain generalization. Separate intervals against a shared reference are not a direct statistical test of the two methods.</p></details>')


def reliability_intervals(audit):
    value = audit.get('point_uncertainty_reliability')
    if value is None:
        return ''
    rows = [[row['signal'], number(row['auroc'], 3), ' to '.join(number(v, 3) for v in row['auroc_ci95'])]
            for row in value['signals']]
    diff = value['difference']
    rows.append(['Raw σ AUROC − low-visibility AUROC', number(diff['auroc_delta'], 3),
                 ' to '.join(number(v, 3) for v in diff['auroc_delta_ci95'])])
    return ('<details><summary>Real DEV / paired uncertainty-ranking comparison</summary>' +
            table(['Uncertainty score', 'AUROC or difference', 'Frame bootstrap 95% interval'], rows) +
            '<p class="small">The same 5,000 paired frame draws compare raw RLE scale with 1 − keypoint visibility confidence. '
            'Corners are not resampled as independent observations. This checks error ranking above 10px on the reused real set; it does not certify uncertainty calibration.</p></details>')


def difficulty_table(diagnosis):
    labels = {'easy_le10': 'Easy / ≤10px', 'moderate_10_20': 'Moderate / 10–20px',
              'moderate_10to20': 'Moderate / 10–20px',
              'hard_gt20': 'Hard / >20px'}
    rows = []
    for row in diagnosis['rows']:
        if row['population'] != 'real_dev' or row['group'] != 'all':
            continue
        rows.append([LABELS[row['arm']], labels.get(row['difficulty'], row['difficulty']),
                     row['n_frames'], row['n_corners'],
                     number(row['baseline_error_px']) + ' → ' + number(row['fused_error_px']),
                     number(row['delta_mean_px']), number(100 * row['improved_fraction']) + '%',
                     number(100 * row['worsened_fraction']) + '%', number(row['mean_move_px'])])
    return ('<details><summary>Posthoc GT difficulty / which real corners improve or worsen?</summary>'
            '<p class="small">Easy, moderate and hard use the baseline corner’s GT error after the experiment. '
            'These are explanatory groups, never inputs to fusion. Each corner averages the three seeds; counts are not tripled. '
            'One image can appear in multiple difficulty groups, so frame counts must not be summed.</p>' +
            table(['Method', 'GT difficulty', 'Frames', 'Corners', 'Baseline → fused error px',
                   'Mean Δ px', 'Improved', 'Worsened', 'Movement px'], rows) + '</details>')


def build_visuals(manifest, predictions, lines):
    expected = [r['id'] for r in manifest['records']]
    records = predictions['records']
    if [r['id'] for r in records] != expected or [r['id'] for r in lines['records']] != expected:
        raise ValueError('Prediction/line order does not match the frozen manifest')
    data = []
    for i, original in enumerate(manifest['records']):
        prediction = records[i]
        data.append(dict(id=original['id'], population=original['population'], group=original['group'],
                         image=Path(original['image']).resolve().as_uri(), width=original['width'],
                         height=original['height'], gt=original['gt_points'], gt_valid=original['gt_valid'],
                         baseline=prediction['baseline'], seeds=prediction['seeds'],
                         lines={seed: lines['records'][i]['seeds'][seed]['lines'] for seed in ('1', '2', '3')}))
    return data


def validate_sources(payload, live, destination, artifact_name):
    for key, path in [('config_sha256', live / 'CONFIG.json'), ('manifest_sha256', live / 'manifest.json')]:
        if key in payload and payload[key] != sha(path):
            raise ValueError(f'Changed {key}')
    for name, digest in payload.get('source_sha256', {}).items():
        path = Path(name)
        if not path.is_absolute():
            if artifact_name == 'PROTOCOL.json':
                # The preregistration hashes its original live inputs, including
                # live/RESULTS.json, not this experiment's later RESULTS.json.
                path = live / path
            else:
                candidates = [destination / path, live / path, ROOT / path]
                path = next((p for p in candidates if p.exists()), candidates[0])
        if sha(path) != digest:
            raise ValueError(f'Changed source: {path}')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path, required=True, help='Frozen live integration directory')
    parser.add_argument('--no-open', action='store_true')
    args = parser.parse_args()
    live = args.run_dir.resolve()
    destination = live / 'uncertainty_fusion_v1'
    frozen = {p: sha(p) for p in live.iterdir() if p.is_file() and p.suffix in ('.json', '.html')}
    diagnosis = live / 'fusion_diagnosis_v1'
    frozen.update({p: sha(p) for p in diagnosis.iterdir() if p.is_file() and p.suffix in ('.json', '.html')})
    names = ['RESULTS.json', 'PREDICTIONS.json', 'SELECTION.json', 'RELIABILITY.json',
             'CALIBRATION.json', 'DHT_UNCERTAINTY.json', 'YOLO_UNCERTAINTY.json', 'PROTOCOL.json']
    for optional in ('STATISTICAL_AUDIT.json', 'DIAGNOSIS.json'):
        if (destination / optional).exists():
            names.append(optional)
    artifacts = {name: read(destination / name) for name in names}
    if not artifacts['RESULTS.json'].get('complete'):
        raise ValueError('Uncertainty evaluation must be complete before rendering')
    for name, payload in artifacts.items():
        validate_sources(payload, live, destination, name)
    manifest = read(live / 'manifest.json')
    data = dict(records=build_visuals(manifest, artifacts['PREDICTIONS.json'], artifacts['DHT_UNCERTAINTY.json']),
                labels=LABELS, populations=POPS, edges=EDGES, incident=INCIDENT,
                selected_rules={arm: value['selected_rule'] for arm, value in artifacts['SELECTION.json']['families'].items()},
                reliability=reliability_data(artifacts['RELIABILITY.json']),
                reliability_thresholds=artifacts['RELIABILITY.json']['thresholds'])
    links = sorted(p.name for p in destination.iterdir() if p.suffix in ('.json', '.md') and
                   p.name not in ('REPORT_RENDER.json', 'VISUAL_QA.json', 'GALLERY_OPEN.json'))
    substitutions = dict(VERDICT=verdict(artifacts['RESULTS.json']), RESULTS=result_tables(artifacts['RESULTS.json']),
                         SELECTION=selection_tables(artifacts['SELECTION.json'], artifacts['RESULTS.json']['calibration']),
                         RELIABILITY_CI='',
                         LINKS=' · '.join(f'<a href="{esc(name)}">{esc(name)}</a>' for name in links),
                         DATA=json.dumps(data, separators=(',', ':'), allow_nan=False).replace('<', '\\u003c'))
    if 'STATISTICAL_AUDIT.json' in artifacts:
        substitutions['RESULTS'] += confidence_intervals(artifacts['STATISTICAL_AUDIT.json'])
        substitutions['RELIABILITY_CI'] = reliability_intervals(artifacts['STATISTICAL_AUDIT.json'])
    if 'DIAGNOSIS.json' in artifacts:
        substitutions['RESULTS'] += difficulty_table(artifacts['DIAGNOSIS.json'])
    page = PAGE
    for key, value in substitutions.items():
        page = page.replace(f'__{key}__', value)
    target = destination / 'uncertainty_report.html'
    target.write_text(page, encoding='utf-8')
    if any(sha(path) != digest for path, digest in frozen.items()):
        raise ValueError('A frozen original artifact changed during rendering')
    audit = dict(complete=True, n_frames=len(data['records']), models=['YOLO'], seeds=[1, 2, 3],
                 html_sha256=sha(target), report_code_sha256=sha(Path(__file__)),
                 input_sha256={name: sha(destination / name) for name in names},
                 frozen_original_artifacts_unchanged=True,
                 default_example_selection='First real_dev frame in frozen manifest order')
    (destination / 'REPORT_RENDER.json').write_text(json.dumps(audit, indent=2) + '\n')
    print(f'Rendered {target} with {len(data["records"])} frames', flush=True)
    if not args.no_open:
        sys.path.insert(0, str(HERE.parent / 'deep_hough_side_v1'))
        from visualize import open_gallery
        open_gallery(target)
    return 0


PAGE = r'''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>YOLO σ + DHT uncertainty · Fusion verification</title>
<style>
:root{font-family:Arial,Helvetica,sans-serif;background:#101722;color:#e6edf7;color-scheme:dark}*{box-sizing:border-box}body{margin:0}main{max-width:1560px;margin:auto;padding:28px 26px 60px}h1{font-size:32px;margin:10px 0 12px;letter-spacing:-.5px}h2{font-size:23px;margin:0 0 12px}h3{font-size:15px;margin:8px 0}p{color:#b8c7da;line-height:1.55}a,summary{color:#92d5ff}.eyebrow{color:#8bbbeb;font-size:12px;letter-spacing:1.6px}section{background:#172130;border:1px solid #30435d;border-radius:12px;padding:22px;margin:22px 0}.note{padding:14px 16px;background:#203243;border-left:3px solid #7bcdf3;border-radius:5px}.small{font-size:12px}.tablewrap{overflow:auto}table{border-collapse:collapse;width:100%;font-size:13px;white-space:nowrap}th,td{padding:10px;border-bottom:1px solid #30445f;text-align:right}th{font-weight:400;color:#98b1cd}th:first-child,td:first-child{text-align:left}tbody tr:hover{background:#203149}details{margin:13px 0}summary{cursor:pointer;padding:8px 0}pre{font-size:12px;max-height:440px;overflow:auto;background:#101923;padding:14px;border-radius:6px}.grid2{display:grid;grid-template-columns:1fr 1fr;gap:16px}.controls{display:flex;gap:10px;flex-wrap:wrap;margin:15px 0}.controls label{font-size:11px;color:#9db5d0;display:flex;flex-direction:column;gap:5px}.framecontrol{flex:1;min-width:300px}select{font:inherit;color:#e6edf7;background:#25374d;border:1px solid #47627e;border-radius:6px;padding:9px;max-width:100%;cursor:pointer}.panel{background:#101923;border:1px solid #354b66;border-radius:8px;padding:11px;min-width:0}.panel h3{font-size:14px;font-weight:400}.panel p{font-size:11px;min-height:28px;margin:7px 0 0}.panel canvas{display:block;width:100%;height:auto}.legend{display:flex;gap:16px;flex-wrap:wrap;font-size:12px;margin:12px 0}.legend span:before{content:"";display:inline-block;width:22px;height:3px;background:var(--c);vertical-align:middle;margin-right:7px}.chips{font-size:12px;color:#a7bed7;margin:12px 0}.plot svg{width:100%;height:auto;display:block;background:#101923;border:1px solid #30435d;border-radius:7px}.formula{font-family:monospace;padding:13px;background:#101923;border-radius:5px;font-size:14px}.intro{max-width:1140px}.empty{padding:24px;color:#a1b2c7}.plotnote{min-height:38px;font-size:12px}
@media(max-width:800px){main{padding:20px 12px}section{padding:15px}.grid2{grid-template-columns:1fr}h1{font-size:27px}}
</style><main>
<div class="eyebrow">FROZEN YOLO · RLE SCALE HEAD · DHT DISTRIBUTION · SYNTHETIC-ONLY CALIBRATION</div>
<h1>Does uncertainty make point–line fusion more reliable?</h1>
<p class="intro">The earlier gate used keypoint visibility scores. This experiment extracts the existing YOLO RLE scale head and the full DHT parameter distribution, then checks their relation to actual error and their value in weighted fusion. No network is retrained.</p>
<p class="note">__VERDICT__</p>
<section id="reliability"><h2>1. Does predicted uncertainty track actual error?</h2>
<p>The rank correlations and binned errors below diagnose uncertainty quality. They are not probabilities that a keypoint or line is correct. Four bins use prediction-only Q25/Q50/Q75 boundaries from calibration128; a real-data bin can contain few or zero observations.</p>
<div class="controls"><label>Reliability population<select id="relpopulation"></select></label><label>Point signal<select id="pointsignal"></select></label><label>Line signal<select id="linesignal"></select></label></div>
<div class="grid2"><div class="plot"><h3 id="pointtitle">Point scale versus point error</h3><div id="pointplot"></div><p class="plotnote" id="pointplotnote"></p></div><div class="plot"><h3 id="linetitle">Line spread versus normal error</h3><div id="lineplot"></div><p class="plotnote" id="lineplotnote"></p></div></div>
<div id="reliabilitytable"></div><p class="small">AUROC ranks errors above 10px for points and above 8px for lines; higher predicted uncertainty is the positive score. Point observations are unique corners. Line observations include three dependent DHT seeds, so they must not be treated as independent images. “Norm” denotes a vector length in original pixels, not division by the image diagonal. These are association checks, not calibrated reliability probabilities.</p>
__RELIABILITY_CI__
<p class="small">The RLE paper predicts a location and a scale for a learned residual distribution. Here, raw σ is an uncalibrated model scale; the working variance is a separate synthetic-validation calibration. Neither an ellipse nor a narrow DHT peak guarantees real-world coverage. <a href="https://arxiv.org/html/2107.11291v2#S3.SS2">RLE paper, reparameterization</a>.</p>
</section>
<section id="results"><h2>2. Frozen synthetic selection → real-image check</h2>
<div class="formula">q = argmin (q − p)ᵀ Σₚ⁻¹ (q − p) + λ ∑ (nᵢ · q + cᵢ)² / vᵢ</div>
<p>Point and line variances change how strongly each measurement constrains the corner. The point-σ-only and line-only arms isolate each signal. The DHT argmax lines remain the geometric targets; uncertainty changes their weights.</p>
__RESULTS__<p class="small">The previous joint gate is a historical reference selected using all 256 synthetic-validation frames. The new families use a 128/128 calibration/selection split, so this is not a comparison with identical validation budgets. The synth_val table includes those calibration and selection images.</p><details open><summary>Calibration split, frozen parameters and validation selection</summary>__SELECTION__</details>
</section>
<section id="examples"><h2>3. The same image, four ways to use its predictions</h2>
<p>All 692 images are available. The default is the first real DEV image in frozen manifest order. GT marks and the optional GT crop are diagnostic display aids; they never change predictions, variances or fusion decisions.</p>
<div class="controls"><label>Population<select id="population"></select></label><label>DHT seed<select id="seed"><option>1</option><option>2</option><option>3</option></select></label><label>Corner<select id="corner"><option value="all">All 8 corners</option></select></label><label>Third panel<select id="comparison"></select></label><label>Final panel<select id="finalarm"><option value="line_uncertainty_only">Line uncertainty only</option><option value="point_line_uncertainty">Point + line uncertainty</option></select></label><label>Ellipse radii<select id="ellipse"><option value="working">Calibrated RLE √variance / px</option><option value="raw">Raw RLE σ / px</option><option value="off">Off</option></select></label><label>View<select id="zoom"><option value="full">Full image</option><option value="focus">GT crop / diagnostic only</option></select></label><label class="framecontrol">Frame<select id="frame"></select></label></div>
<div class="legend"><span style="--c:#44f286">GT</span><span style="--c:#57c7ff">YOLO baseline / scale ellipse</span><span style="--c:#ff9c70">Previous joint gate</span><span style="--c:#e39dff">Third-panel fusion</span><span style="--c:#b0a0ff">Final-panel fusion / displacement</span><span style="--c:#ffd166">DHT line spread</span></div>
<div class="chips" id="frameinfo"></div>
<div class="grid2"><div class="panel"><h3>YOLO baseline + RLE scale ellipse + GT</h3><canvas id="baseline"></canvas><p id="basenote"></p></div><div class="panel"><h3>Previous joint gate + GT</h3><canvas id="previous"></canvas><p id="previousnote"></p></div><div class="panel"><h3 id="comparisonheading">Point σ only + GT</h3><canvas id="pointonly"></canvas><p id="pointnote"></p></div><div class="panel"><h3 id="finalheading">Line uncertainty only · lines and displacement</h3><canvas id="combined"></canvas><p id="combinednote"></p></div></div>
<p id="loadstatus" class="small">Loading image…</p><p class="small">Ellipses show the baseline point’s RLE signal at 1× scale, using raw σ or its synthetic-calibrated variance. They are not calibrated coverage contours. The line-only ablation uses constant point variance instead, shown in the table. The gold line band shows ±√working line variance perpendicular to each argmax line at the selected baseline corner; with all corners shown, each role uses the mean endpoint variance. Wider/brighter bands mean larger variance, not stronger evidence. Purple arrows connect the baseline point to the saved fused point. All overlays use original-image pixels. Gray canvas lies outside the image.</p><div id="cornertable"></div>
</section>
<section><h2>Scope and source records</h2><p>YOLO only: the current DOPE baseline has no corresponding RLE head. The same 52 real images have already been inspected in earlier experiments. Calibration and model selection use disjoint halves of synthetic validation; this does not turn reused real development data into an independent test. DHT lines describe structural targets, including amodal boundaries, and are not physical visible-edge labels or attention. These 2D checks do not establish improved 6D pose.</p><p>__LINKS__</p><p><a href="../fusion_diagnosis_v1/fusion_diagnosis.html">Earlier selective fusion diagnosis</a> · <a href="../integration_report.html">Frozen integration report</a></p></section>
</main><script id="payload" type="application/json">__DATA__</script><script>
'use strict';
const D=JSON.parse(document.getElementById('payload').textContent),$=id=>document.getElementById(id),COLORS={gt:'#44f286',base:'#57c7ff',previous:'#ff9c70',point:'#e39dff',both:'#b0a0ff'};
let loadedImage=null,loadedId=null,generation=0;
function options(node,rows){node.replaceChildren(...rows.map(([v,t])=>{let o=document.createElement('option');o.value=v;o.textContent=t;return o;}));}
function fmt(v,n=2){return Number.isFinite(v)?v.toFixed(n):'—';}
function rootVariance(v){return Number.isFinite(v)&&v>=0?Math.sqrt(v):NaN;}
function valid(p){return Array.isArray(p)&&p.length===2&&p.every(Number.isFinite);}
function corners(){return $('corner').value==='all'?[0,1,2,3,4,5,6,7]:[Number($('corner').value)];}
function htmlTable(target,headers,rows){let table=document.createElement('table'),head=document.createElement('thead'),body=document.createElement('tbody');for(let [parent,values,tag] of [[head,headers,'th'],...rows.map(r=>[body,r,'td'])]){let tr=document.createElement('tr');values.forEach(v=>{let c=document.createElement(tag);c.textContent=v;tr.append(c);});parent.append(tr);}table.append(head,body);$(target).className='tablewrap';$(target).replaceChildren(table);}
options($('population'),D.populations.map(p=>[p,`${p} (${D.records.filter(r=>r.population===p).length})`]));options($('relpopulation'),D.populations.map(p=>[p,p]));for(let i=0;i<8;i++)$('corner').append(new Option(String(i),String(i)));options($('comparison'),['point_sigma_only','line_uncertainty_only','unconditional_lambda1'].map(a=>[a,D.labels[a]]));
function prediction(r,arm){return arm==='baseline'?r.baseline:r.seeds[$('seed').value][arm];}
function updateFrames(){let old=$('frame').value,rows=D.records.filter(r=>r.population===$('population').value);options($('frame'),rows.map(r=>[r.id,`${r.group} · ${r.id}`]));if(rows.some(r=>r.id===old))$('frame').value=old;render();}
function crop(r){if($('zoom').value==='full')return[-35,-35,r.width+70,r.height+70];let points=corners().map(i=>r.gt[i]).filter(valid);if(!points.length)return[-35,-35,r.width+70,r.height+70];let x=points.map(p=>p[0]),y=points.map(p=>p[1]),cx=(Math.min(...x)+Math.max(...x))/2,cy=(Math.min(...y)+Math.max(...y))/2;let width=Math.max(160,Math.max(...x)-Math.min(...x)+90,(Math.max(...y)-Math.min(...y)+90)*4/3);return[cx-width/2,cy-width*3/8,width,width*3/4];}
function line(ctx,a,b,color,width=1.7){if(!valid(a)||!valid(b))return;ctx.strokeStyle=color;ctx.lineWidth=width;ctx.beginPath();ctx.moveTo(...a);ctx.lineTo(...b);ctx.stroke();}
function graph(ctx,kps,mask,color){let chosen=new Set(corners());D.edges.forEach(([a,b])=>{if(chosen.has(a)&&chosen.has(b)&&mask[a]&&mask[b])line(ctx,kps[a],kps[b],color);});corners().forEach(i=>{let p=kps[i];if(!mask[i]||!valid(p))return;ctx.beginPath();ctx.arc(...p,2.8,0,2*Math.PI);ctx.strokeStyle=color;ctx.lineWidth=1.5;ctx.stroke();ctx.font='bold 10px Arial';ctx.strokeStyle='#101722';ctx.lineWidth=3;ctx.strokeText(String(i),p[0]+4,p[1]-4);ctx.fillStyle=color;ctx.fillText(String(i),p[0]+4,p[1]-4);});}
function setup(id,r){let canvas=$(id);canvas.width=800;canvas.height=600;let ctx=canvas.getContext('2d');ctx.fillStyle='#26313e';ctx.fillRect(0,0,800,600);let[x,y,w,h]=crop(r),scale=Math.min(800/w,600/h);ctx.translate((800-w*scale)/2,(600-h*scale)/2);ctx.scale(scale,scale);ctx.translate(-x,-y);ctx.drawImage(loadedImage,0,0,r.width,r.height);graph(ctx,r.gt,r.gt_valid,COLORS.gt);return ctx;}
function pointVariance(r,p){return p.point_variance_px2||r.baseline.point_variance_px2;}
function ellipses(ctx,r,p){if($('ellipse').value==='off')return;let raw=$('ellipse').value==='raw',values=raw?r.baseline.sigma_original_px:r.baseline.point_variance_px2;if(!values)return;corners().forEach(i=>{let a=r.baseline.kps[i],v=values[i];if(!r.baseline.kp_valid[i]||!valid(a)||!valid(v)||v.some(x=>x<0))return;let rx=raw?v[0]:Math.sqrt(v[0]),ry=raw?v[1]:Math.sqrt(v[1]);ctx.beginPath();ctx.ellipse(a[0],a[1],rx,ry,0,0,2*Math.PI);ctx.fillStyle='#57c7ff20';ctx.strokeStyle=COLORS.base;ctx.lineWidth=1.1;ctx.fill();ctx.stroke();});}
function arrow(ctx,a,b){if(!valid(a)||!valid(b))return;let dx=b[0]-a[0],dy=b[1]-a[1],n=Math.hypot(dx,dy);if(n<.02)return;line(ctx,a,b,COLORS.both,2);let ux=dx/n,uy=dy/n,l=Math.min(6,n*.5);line(ctx,b,[b[0]-l*ux+l*.5*uy,b[1]-l*uy-l*.5*ux],COLORS.both,2);line(ctx,b,[b[0]-l*ux-l*.5*uy,b[1]-l*uy+l*.5*ux],COLORS.both,2);}
function lineBands(ctx,r,both){let incident=both.incident_line_variance_px2;if(!incident)return;let rolevars={};corners().forEach(i=>{D.incident[i].forEach((role,j)=>{let v=incident[i]&&incident[i][j];if(Number.isFinite(v)&&v>=0)(rolevars[role]??=[]).push(v);});});for(let[role,vars]of Object.entries(rolevars)){let pair=r.lines[$('seed').value][role];if(!pair||!pair.every(valid))continue;let[a,b]=pair,dx=b[0]-a[0],dy=b[1]-a[1],n=Math.hypot(dx,dy);if(n<1e-9)continue;let rms=Math.sqrt(vars.reduce((x,y)=>x+y,0)/vars.length),ux=dx/n,uy=dy/n,nx=-uy,ny=ux,L=4*Math.hypot(r.width,r.height),ends=[[a[0]-L*ux,a[1]-L*uy],[a[0]+L*ux,a[1]+L*uy]],light=50+20*Math.min(1,Math.log1p(rms)/Math.log(101));ctx.fillStyle=`hsla(39,95%,${light}%,0.10)`;ctx.beginPath();ctx.moveTo(ends[0][0]+rms*nx,ends[0][1]+rms*ny);ctx.lineTo(ends[1][0]+rms*nx,ends[1][1]+rms*ny);ctx.lineTo(ends[1][0]-rms*nx,ends[1][1]-rms*ny);ctx.lineTo(ends[0][0]-rms*nx,ends[0][1]-rms*ny);ctx.closePath();ctx.fill();line(ctx,...ends,`hsl(39,95%,${light}%)`,1.1);}}
function pointError(r,p,i){return r.gt_valid[i]&&p.kp_valid[i]&&valid(p.kps[i])?Math.hypot(p.kps[i][0]-r.gt[i][0],p.kps[i][1]-r.gt[i][1]):null;}
function stats(r,p){let ids=corners().filter(i=>r.gt_valid[i]),e=ids.map(i=>pointError(r,p,i)).filter(Number.isFinite);return `${e.length}/${ids.length} points · mean ${e.length?fmt(e.reduce((x,y)=>x+y,0)/e.length):'—'}px · PCK10 ${ids.length?fmt(100*e.filter(v=>v<=10).length/ids.length,1):'—'}%`;}
function draw(r){let base=r.baseline,previous=prediction(r,'previous_joint_gate'),comparison=prediction(r,$('comparison').value),both=prediction(r,$('finalarm').value);let ctx=setup('baseline',r);ellipses(ctx,r,both);graph(ctx,base.kps,base.kp_valid,COLORS.base);graph(setup('previous',r),previous.kps,previous.kp_valid,COLORS.previous);graph(setup('pointonly',r),comparison.kps,comparison.kp_valid,COLORS.point);ctx=setup('combined',r);lineBands(ctx,r,both);graph(ctx,base.kps,base.kp_valid,COLORS.base);corners().forEach(i=>{if(base.kp_valid[i]&&both.kp_valid[i])arrow(ctx,base.kps[i],both.kps[i]);});graph(ctx,both.kps,both.kp_valid,COLORS.both);$('basenote').textContent=stats(r,base)+` · ${$('ellipse').selectedOptions[0].textContent}`;$('previousnote').textContent=stats(r,previous);$('pointnote').textContent=stats(r,comparison);$('combinednote').textContent=stats(r,both)+` · selected λ=${D.selected_rules[$('finalarm').value].lambda}`;$('finalheading').textContent=D.labels[$('finalarm').value]+' · lines and displacement';$('comparisonheading').textContent=D.labels[$('comparison').value]+' + GT';$('frameinfo').textContent=`YOLO · ${r.population} · ${r.group} · seed ${$('seed').value} · ${r.width}×${r.height} original pixels · ${corners().length} highlighted corners`;let vars=pointVariance(r,both)||[];let rows=corners().map(i=>{let v=vars[i],lv=both.incident_line_variance_px2?.[i],sigma=base.sigma_original_px?.[i];return[i,fmt(pointError(r,base,i)),fmt(pointError(r,both,i)),sigma?`${fmt(sigma[0])} / ${fmt(sigma[1])}`:'—',v?`${fmt(rootVariance(v[0]))} / ${fmt(rootVariance(v[1]))}`:'—',lv?lv.map(x=>fmt(rootVariance(x))).join(' / '):'—',fmt(both.move_px?.[i])];});htmlTable('cornertable',['Corner','Point error px','Final-panel error px','Raw σx / σy px','Final working √var x / y px','Incident line RMS px','Movement px'],rows);$('loadstatus').textContent='Saved original-coordinate predictions. No inference or recalibration occurs in this viewer.';window.uncertaintyReady=true;window.currentUncertaintyFrame=r.id;}
function render(){let r=D.records.find(r=>r.id===$('frame').value),token=++generation;window.uncertaintyReady=false;if(!r){$('loadstatus').textContent='No frames selected.';return;}if(loadedId===r.id&&loadedImage){draw(r);return;}$('loadstatus').textContent='Loading image…';let next=new Image();next.onload=()=>{if(token!==generation)return;loadedId=r.id;loadedImage=next;draw(r);};next.onerror=()=>{$('loadstatus').textContent='Image load failed.';window.uncertaintyError=r.image;};next.src=r.image;}
function signalRows(){return D.reliability.filter(r=>r.population===$('relpopulation').value&&(r.group===undefined||r.group==='all'));}
function signalName(row){return row.signal||row.name;}
function plot(target,row,color){
 let bins=row?.bins||[],values=bins.map(b=>b.mean_error_px),finite=values.filter(Number.isFinite);
 if(!finite.length){$(target).innerHTML='<div class="empty">No saved observations for this signal.</div>';return;}
 let w=680,h=310,left=64,right=20,top=24,bottom=77,max=Math.max(...finite)*1.15||1;
 let px=i=>left+(w-left-right)*(i+.5)/bins.length,py=v=>h-bottom-(h-top-bottom)*v/max;
 let svg=`<svg viewBox="0 0 ${w} ${h}" role="img" aria-label="Observed mean error by fixed uncertainty bin">`;
 for(let i=0;i<=4;i++){let v=max*i/4,y=py(v);svg+=`<line x1="${left}" y1="${y}" x2="${w-right}" y2="${y}" stroke="#283d56"/><text x="${left-9}" y="${y+4}" text-anchor="end" fill="#9eb4ce" font-size="12">${fmt(v,1)}</text>`;}
 bins.forEach((b,i)=>{let v=values[i];if(Number.isFinite(v)){
  if(i>0&&Number.isFinite(values[i-1]))svg+=`<line x1="${px(i-1)}" y1="${py(values[i-1])}" x2="${px(i)}" y2="${py(v)}" stroke="${color}" stroke-width="2.5"/>`;
  svg+=`<circle cx="${px(i)}" cy="${py(v)}" r="4" fill="${color}"/><text x="${px(i)}" y="${py(v)-10}" text-anchor="middle" fill="#dbe8f6" font-size="12">${fmt(v)}</text>`;
 }else{svg+=`<text x="${px(i)}" y="${h-bottom-12}" text-anchor="middle" fill="#97a6ba" font-size="12">Empty</text>`;}
 svg+=`<text x="${px(i)}" y="${h-bottom+21}" text-anchor="middle" fill="#aabed5" font-size="12">Bin ${i+1}</text><text x="${px(i)}" y="${h-bottom+39}" text-anchor="middle" fill="#91a6bf" font-size="11">n=${b.n_observations}</text>`;});
 svg+=`<text x="${(left+w-right)/2}" y="${h-8}" text-anchor="middle" fill="#aabed5" font-size="13">Predicted uncertainty: lower → higher</text><text transform="translate(17 ${(top+h-bottom)/2}) rotate(-90)" text-anchor="middle" fill="#aabed5" font-size="13">Observed mean error / px</text></svg>`;
 $(target).innerHTML=svg;
}
function drawReliability(){
 let rows=signalRows();
 for(let[select,target,note,color]of [['pointsignal','pointplot','pointplotnote','#57c7ff'],['linesignal','lineplot','lineplotnote','#ffd166']]){
  let row=rows.find(r=>signalName(r)===$(select).value);plot(target,row,color);
  let thresholds=row?D.reliability_thresholds[signalName(row)]:[];
  $(note).textContent=row?`${signalName(row)} · Spearman ρ ${fmt(row.spearman,3)} · ${row.n_frames} frames / ${row.n_observations} observations. Bin cuts: ${thresholds.map(v=>fmt(v,4)).join(', ')}.`:'No observations.';
 }
 htmlTable('reliabilitytable',['Signal','Frames','Observations','Spearman ρ','Error > px','AUROC','Mean error px','P90 error px'],rows.map(r=>[signalName(r),r.n_frames,r.n_observations,fmt(r.spearman,3),r.error_threshold_px,fmt(r.auroc,3),fmt(r.mean_error_px),fmt(r.p90_error_px)]));
}
function updateReliability(){
 let rows=signalRows(),names=[...new Set(rows.map(signalName))],point=names.filter(n=>n&&(n.includes('point')||n.includes('conf')||n.includes('sigma'))),lines=names.filter(n=>!point.includes(n));
 options($('pointsignal'),point.map(n=>[n,n]));options($('linesignal'),lines.map(n=>[n,n]));
 if(point.includes('point_sigma_norm'))$('pointsignal').value='point_sigma_norm';if(lines.includes('line_posterior_rms'))$('linesignal').value='line_posterior_rms';drawReliability();
}
$('population').addEventListener('change',updateFrames);['seed','corner','comparison','finalarm','ellipse','zoom','frame'].forEach(id=>$(id).addEventListener('change',render));$('relpopulation').addEventListener('change',updateReliability);['pointsignal','linesignal'].forEach(id=>$(id).addEventListener('change',drawReliability));updateReliability();updateFrames();
</script></html>'''


if __name__ == '__main__':
    raise SystemExit(main())
