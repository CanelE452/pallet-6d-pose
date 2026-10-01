"""Insert frozen CPU evaluation summaries; never run inference/training or infer missing results.

--pending writes an explicit unmeasured draft. --results-dir requires completed
DEV, paired, training, selection and heldout receipts from the new experiment.
Third-estimator status belongs exclusively to update_resnet_results.py.
"""
from pathlib import Path
import argparse
import hashlib
import json
import math

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
TABLES = HERE / 'generated_tables'
EXPERIMENT = ROOT / '_docs/experiments/pallet_dope_refiner_20261001_v1'
ARMS = ('DOPE', 'D1', 'D2', 'D3', 'P1', 'P2', 'P3')


def read(path):
    return json.loads(Path(path).read_text())


def bind(path):
    path = Path(path).resolve()
    return dict(path=str(path.relative_to(ROOT)),
                sha256=hashlib.sha256(path.read_bytes()).hexdigest(), bytes=path.stat().st_size)


def verify(value):
    path = ROOT / value['path']
    assert path.resolve().is_relative_to(ROOT)
    observed = bind(path)
    assert observed['sha256'] == value['sha256'], ('Input changed', path)
    assert 'bytes' not in value or observed['bytes'] == value['bytes'], path
    return path


def number(value, digits=3, scale=1):
    if value is None:
        return '--'
    assert math.isfinite(float(value)), value
    return f'{float(value)*scale:.{digits}f}'


def interval(value):
    if value.get('status') != 'COMPLETE':
        return 'unavailable'
    return f"{number(value['delta'])} [{number(value['low'])}, {number(value['high'])}]"


def table(headers, rows, caption, label, wide=False):
    env = 'table*' if wide else 'table'
    lines = [rf'\begin{{{env}}}[t]\centering\small',
             rf'\caption{{{caption}}}\label{{{label}}}',
             r'\begin{tabular}{l' + 'r'*(len(headers)-1) + r'}\toprule',
             ' & '.join(headers) + r'\\\midrule']
    for row in rows:
        assert len(row) == len(headers)
        lines.append(' & '.join(map(str, row)) + r'\\')
    lines.extend([r'\bottomrule\end{tabular}', rf'\end{{{env}}}'])
    return '\n'.join(lines) + '\n'


THIRD = ('The third application is specified as a full-image, nine-landmark '
         'SimpleBaseline-derived ResNet18 with ImageNet initialization. Its baseline '
         'training, frozen P/D head fits and measured outcomes remain pending in this '
         'revision. Completion of DOPE alone does not satisfy the three-estimator '
         'evidence requirement.')


def status_file(abstract, main, conclusion):
    return ('% Generated evidence status: not a manuscript acceptance/submission flag.\n'
            + '\n'.join(rf'\newcommand{{\{name}}}{{{text}}}' for name, text in (
                ('DopeAbstractStatus', abstract), ('DopeMainStatus', main),
                ('DopeConclusionStatus', conclusion))) + '\n')


def pending():
    reason = ('DOPE source training/evaluation results have not been inserted. '
              'No DOPE accuracy, coverage, pose benefit, parameter count or measured latency '
              'is claimed in this draft.')
    return {
        'dope_status.tex': status_file(
            'DOPE results remain pending in this revision.', reason,
            'The DOPE application remains unresolved; the historical YOLO result is '
            'not evidence of its outcome.'),
        'dope_results.tex': r'\noindent DOPE measurement status: pending. '
            + 'The reserved result table will contain baseline DOPE and three-seed D/P '
            'statistics, with matched counts, all-GT PCK and pose coverage. No zero or '
            'historical YOLO value represents an unmeasured DOPE result.\n',
        'dope_paired.tex': r'\noindent DOPE paired intervals: pending. '
            + 'The frozen session-bootstrap outputs will be inserted without dropping '
            'support mismatches or failed resamples.\n',
        'dope_source.tex': r'\noindent DOPE source evidence: pending. '
            + 'Actual usable rows, head parameter counts, completed updates, calibration '
            'temperatures, selected movement rules and heldout results require their own receipts.\n',
        'dope_runtime.tex': r'\noindent DOPE cost measurements: pending. '
            + 'Historical YOLO latency is not an estimate of this path. The prescribed '
            'seed-1 timing panel must finish before reporting DOPE overhead.\n',
        'dope_full.tex': r'\noindent DOPE per-seed table: pending. '
            + 'Seven rows (DOPE, D1--D3, P1--P3) are reserved; their complete denominator '
            'is 319 images and 2818 supervised landmarks.\n',
    }


def validate(results, paired, training, selection, heldout):
    assert all(v.get('complete') is True for v in (results, paired, training, selection, heldout))
    assert results['schema'] == 'dope_refiner_dev_results_v1' and results['role'] == 'REUSED_DEV'
    assert (results['positive_frames'], results['sessions'], results['gt_denominator']) == (319, 13, 2818)
    assert set(results['methods']) == set(ARMS) and set(results['seed_mean']) == {'D', 'P'}
    assert results['same_conditional_support'] is True and results['DOPE_AP_claim'] is False
    assert results['geometry_derived_reference_not_independent_physical_metrology'] is True
    assert paired['strict_support_no_silent_intersection'] is True
    assert paired['predictions'] == results['predictions']
    assert selection['real_selection'] is False
    assert (training['fits'], training['steps_per_fit'], training['total_head_updates']) == (6, 6000, 36000)
    assert len(training['seeds']) == 3 and {s['seed'] for s in training['seeds']} == {1, 2, 3}
    for arm, row in results['methods'].items():
        assert row['gt_denominator'] == 2818 and row['full_frame_denominator'] == 319
        assert row['pose']['denominator'] == 319
        assert row['pose']['n'] + row['pose']['failure_count'] == 319
        assert row['matched_frames'] + row['failed_or_excluded_frames'] == 319
        # Missing predicted points and whole-frame conditional exclusions differ.
        # A finite but unmatched frame excludes GT without adding missing points.
        assert 0 <= row['supervised_points'] <= 2818
        assert 0 <= row['missing_supervised_keypoints'] <= 2818 - row['supervised_points']
    assert set(heldout['results']) == set(ARMS)


def render(results, paired, training, selection, heldout):
    validate(results, paired, training, selection, heldout)
    output = pending()
    base = results['methods']['DOPE']; p = results['seed_mean']['P']
    observed = (f"For frozen DOPE on reused DEV, the conditional keypoint median is "
                f"{number(base['median_px'])} pixels before refinement and "
                f"{number(p['median_px'])} pixels for the mean of three P seed statistics; "
                f"full-GT PCK10 is {number(base['ALL_GT_PCK']['10'], scale=100)} and "
                f"{number(p['ALL_GT_PCK']['10'], scale=100)} percent, respectively. ")
    output['dope_status.tex'] = status_file(observed, observed + 'The tables retain missing predictions and unfavorable outcomes.',
        'The DOPE measurements describe one additional fixed application; they do not '
        'establish arbitrary-backbone transfer, independent TEST generalization or stable physical T/R improvement.')
    rows = []
    for name, value in [('DOPE', base), ('DOPE+D mean', results['seed_mean']['D']), ('DOPE+P mean', p)]:
        pose = value['pose']
        rows.append([name, number(value['median_px']), number(value['p90_px']),
            number(value['ALL_GT_PCK']['10'], scale=100), number(value['matched_frames'], 1),
            number(pose['translation_median_cm']), number(pose['rotation_median_deg']),
            number(pose['coverage'], scale=100)])
    output['dope_results.tex'] = table(
        ['Arm', 'Med. px', 'P90 px', 'PCK10 \\%', 'Match/319', 'T cm', 'R deg', 'Pose \\%'], rows,
        'New fixed DOPE application on reused DEV. D/P rows average three per-seed '
        'statistics, not predictions. Keypoint medians are conditional; PCK retains '
        '2818 GT points. Pose medians are geometry-reference errors on available '
        'fits; coverage uses319 frames. No cross-estimator equality of matched support is assumed.',
        'tab:dope', True)
    rows = []
    for key, title in [('P_minus_DOPE', 'P--DOPE'), ('D_minus_DOPE', 'D--DOPE'), ('P_minus_D', 'P--D')]:
        value = paired['results'][key]
        rows.append([title, interval(value['conditional_keypoint_median']['session']),
                     interval(value['pose']['translation_error_cm']), interval(value['pose']['rotation_error_deg'])])
    output['dope_paired.tex'] = table(['Contrast', 'Keypoint px [95\\% CI]', 'T cm [95\\% CI]', 'R deg [95\\% CI]'],
        rows, 'DOPE paired session-bootstrap differences: first arm minus second. '
        'Negative values favor the first arm for these errors. An unavailable interval '
        'denotes incompatible or empty support, not a zero effect. These reused-DEV '
        'secondary intervals are unadjusted and exploratory.', 'tab:dopepaired', True)
    rows = []
    for name in ARMS:
        value = results['methods'][name]; pose = value['pose']
        rows.append([name, number(value['median_px']), number(value['p90_px']),
            value['matched_frames'], value['supervised_points'], 2818-value['supervised_points'],
            number(value['ALL_GT_PCK']['10'], scale=100), pose['n'],
            number(pose['translation_median_cm']), number(pose['rotation_median_deg'])])
    output['dope_full.tex'] = table(
        ['Arm', 'Med.', 'P90', 'Match', 'Used GT', 'Excl. GT', 'PCK10\\%', 'Pose n', 'T cm', 'R deg'], rows,
        'DOPE per-seed values; keypoint errors are pixels. Matched frames and pose fits '
        'have denominator319; used/excluded GT sum to2818. Excluded GT includes '
        'finite predictions on unmatched or incomplete frames, not only absent points. '
        'No failed population row is removed.',
        'tab:dopefull')
    seeds = sorted(training['seeds'], key=lambda v: v['seed'])
    for s in seeds:
        assert s['real_training'] == 0 and s['final_checkpoint_only'] is True and s['backbone_retrained'] is False
        assert s['updates_per_arm'] == 6000 and s['exposures_per_arm'] == 96000
    assert len({s['usable_training_rows'] for s in seeds}) == 1
    assert len({tuple(sorted(s['parameters'].items())) for s in seeds}) == 1
    source_text = (f"The completed DOPE fits use {seeds[0]['usable_training_rows']} usable training rows, "
        f"{seeds[0]['parameters']['P']} learned parameters for P and "
        f"{seeds[0]['parameters']['D']} for D. Six final fits each complete6000 updates "
        'with96000 nominal source exposures; no real images train these heads. ')
    for arm in ('P', 'D'):
        rule = selection['rules'][arm]
        cap = 'none' if rule['max_move_image_diagonal_fraction'] is None else number(rule['max_move_image_diagonal_fraction'], 4)
        source_text += f"{arm} selects $\\lambda={number(rule['lam'],4)}$ and original-image-diagonal cap {cap}. "
    source_text += 'P temperatures for seeds1--3 are ' + ', '.join(number(selection['temperatures'][f'P{s}'], 2) for s in (1,2,3)) + '.\n'
    rows = []
    for arm in ARMS:
        value = heldout['results'][arm]
        assert value['frames'] == 1985 and value['gt9'] == value['observed9'] + value['missing9']
        rows.append([arm, number(value['median_px']), number(value['p90_px']),
            number(value['pck10_all_gt'], scale=100), value['observed9'], value['missing9']])
    output['dope_source.tex'] = source_text + table(['Arm', 'Med. px', 'P90 px', 'PCK10\\%', 'Used GT', 'Miss GT'],
        rows, 'DOPE source heldout partition after wrapper freezing (1985 records). '
        'This source diagnostic uses its declared partial-point support and is not '
        'the complete-nine-point DEV statistic or independent TEST.', 'tab:dopesource', True)
    return output


def render_runtime(runtime):
    assert runtime['schema'] == 'dope_refiner_runtime_v1'
    assert runtime['complete'] is True and runtime['PASS'] is True
    assert (runtime['frames'], runtime['sessions'], runtime['repeats'], runtime['batch']) == (26,13,5,1)
    assert runtime['measured_calls'] == 390 and runtime['warmup_calls'] == 60
    assert runtime['all_cache_replays_PASS'] is True and runtime['no_fastest_selection'] is True
    assert runtime['partial_or_failed_rows_discarded'] is False
    assert runtime['historical_YOLO_latency_is_separate'] is True
    assert set(runtime['results']) == {'DOPE','D1','P1'}
    assert len(runtime['measurements']) == 390
    rows = []
    for arm, title in [('DOPE','DOPE'),('D1','DOPE+D1'),('P1','DOPE+P1')]:
        values = runtime['results'][arm]
        assert values['failures_included'] is True
        assert sum(values['pose_status_counts'].values()) == 130
        assert sum(row['arm'] == arm for row in runtime['measurements']) == 130
        for key in ('keypoints_ms','pose_ms','full_ms'):
            assert values[key]['n'] == 130
        rows.append([title, number(values['keypoints_ms']['median']),
            number(values['pose_ms']['median']), number(values['full_ms']['median']),
            number(values['full_ms']['p90']), values['pose_status_counts'].get('OK',0)])
    text = table(['Arm','2D med. ms','PnP med. ms','Full med. ms','Full P90 ms','Pose OK/130'], rows,
        'Measured DOPE seed-1 desktop panel, 26 prespecified images with five repeats '
        'per arm after20 warmups. Decoded RGB through real preprocessing/inference '
        'and canonical PnP; disk reads, loading and replay checks excluded. Each column '
        'is its own statistic: medians must not be added. All failure timings remain. '
        'This is a separate session from historical YOLO timing.', 'tab:doperuntime', True)
    return text + ('The completed panel contains 390 measured calls; coordinate replay '
        f"passed at the declared absolute tolerance of {number(runtime['point_atol_px'],12)} pixels. "
        'Earlier failed attempts, if any, remain separately recorded and are not mixed '
        'into this completed summary. This runtime result does not establish accuracy or '
        'embedded-device throughput.\n')


def generate(folder, runtime_path=None):
    if folder is None:
        output = pending(); sources = []; state = 'DOPE_RESULTS_PENDING'
    else:
        folder = Path(folder).resolve()
        assert folder == EXPERIMENT.resolve(), 'Only the declared DOPE experiment can populate this revision'
        names = ['DEV_RESULTS.json', 'DEV_PAIRED_RESULTS.json', 'TRAINING_COMPLETE.json',
                 'SELECTION.json', 'SYNTHETIC_HELDOUT.json']
        data = [read(folder / n) for n in names]
        result, paired, training, selection, heldout = data
        for key in ('code', 'evaluation_lock', 'predictions', 'frame_results', 'full_precision_errors', 'paired_results'):
            verify(result[key])
        assert result['paired_results'] == bind(folder / 'DEV_PAIRED_RESULTS.json')
        assert heldout['selection']['sha256'] == bind(folder/'SELECTION.json')['sha256']
        verify(selection['protocol']); verify(selection['validation'])
        for b in result['source_bindings']:
            verify(b)
        output = render(*data); sources = [bind(folder / n) for n in names]
        state = 'DOPE_RESULTS_INSERTED_RUNTIME_PENDING'
        if runtime_path is not None:
            runtime_path = Path(runtime_path).resolve()
            assert runtime_path == folder/'RUNTIME.json'
            runtime = read(runtime_path)
            runtime_plan = read(verify(runtime['plan']))
            verify(runtime['code']); verify(runtime['raw_rows'])
            assert runtime_plan['predictions']['sha256'] == result['predictions']['sha256']
            assert runtime_plan['protocol']['sha256'] == selection['protocol']['sha256']
            assert runtime_plan['training']['sha256'] == bind(folder/'TRAINING_COMPLETE.json')['sha256']
            for value in runtime_plan['codes']:
                verify(value)
            output['dope_runtime.tex'] = render_runtime(runtime)
            sources += [bind(runtime_path), runtime['plan']]
            state = 'DOPE_RESULTS_AND_RUNTIME_INSERTED'
    for name, content in output.items():
        assert name.startswith('dope_') and name.endswith('.tex')
        (TABLES / name).write_text(content)
    receipt = dict(schema='sensors_dope_manuscript_insert_v1', status=state,
        draft=True, not_submitted=True, third_estimator_status_owner='RESNET_RESULTS_BINDINGS.json',
        independent_TEST_available=False, source_numbers_recomputed=False,
        new_training_or_inference=0, runtime_receipt_adapter_pending=False,
        runtime_results_inserted=runtime_path is not None,
        code=bind(__file__), evidence=sources, generated_tables=[bind(TABLES/n) for n in output])
    (HERE/'DOPE_RESULTS_BINDINGS.json').write_text(json.dumps(receipt, indent=2)+'\n')
    print(json.dumps(dict(status=state, generated_tables=len(output))))


def selfcheck():
    assert number(None) == '--' and number(0) == '0.000'
    assert interval(dict(status='SUPPORT_MISMATCH_NO_PAIRED_CONDITIONAL_CI')) == 'unavailable'
    assert interval(dict(status='COMPLETE', delta=-1., low=-2., high=.5)) == '-1.000 [-2.000, 0.500]'
    assert 'ThirdEstimatorStatus' not in pending()['dope_status.tex']
    assert all('pending' in v.lower() for v in pending().values())
    try:
        number(float('nan'))
    except AssertionError:
        pass
    else:
        raise AssertionError('Nonfinite measurements must not be printed')
    try:
        validate(dict(complete=False), {}, {}, {}, {})
    except AssertionError:
        pass
    else:
        raise AssertionError('Incomplete results must be rejected')
    fixture = dict(schema='dope_refiner_runtime_v1',complete=True,PASS=True,
        frames=26,sessions=13,repeats=5,batch=1,measured_calls=390,warmup_calls=60,
        all_cache_replays_PASS=True,no_fastest_selection=True,partial_or_failed_rows_discarded=False,
        historical_YOLO_latency_is_separate=True,point_atol_px=1e-10,
        measurements=[dict(arm=a) for a in ('DOPE','D1','P1') for _ in range(130)],
        results={a:dict(failures_included=True,pose_status_counts={'OK':100,'NO_DETECTION':30},
          **{k:dict(n=130,median=v,p90=v+1) for k,v in [('keypoints_ms',4.),('pose_ms',2.),('full_ms',7.)]})
          for a in ('DOPE','D1','P1')})
    text = render_runtime(fixture)
    assert '7.000' in text and 'DOPE+P1' in text and '100' in text
    fixture['partial_or_failed_rows_discarded'] = True
    try:
        render_runtime(fixture)
    except AssertionError:
        pass
    else:
        raise AssertionError('Failure-dropping runtime must be rejected')
    print('PURE_DRAFT_INSERT_SELFCHECK_PASS; no actual DOPE result read')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--pending', action='store_true')
    group.add_argument('--results-dir', type=Path)
    group.add_argument('--selfcheck', action='store_true')
    parser.add_argument('--runtime',type=Path,help='Completed DOPE RUNTIME.json; requires --results-dir')
    args = parser.parse_args()
    if args.runtime and not args.results_dir:
        parser.error('--runtime requires --results-dir')
    selfcheck() if args.selfcheck else generate(args.results_dir,args.runtime)
