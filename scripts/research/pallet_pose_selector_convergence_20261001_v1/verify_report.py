"""Verify published frozen tables and weights without reading raw GT."""
from . import common as C
READS = C.U.source_guard(allow_source_targets=False)
import csv
import json
from pathlib import Path
import numpy as np

AUDIT = C.ROOT / '_docs/experiments/pallet_pose_selector_objective_audit_20261001_v1'


def read_csv(path):
    with path.open(newline='') as handle:
        return list(csv.DictReader(handle))


def csv_text(value):
    return '' if value is None else str(value)


def main():
    report = (C.DOC / 'REPORT_KO.md').read_text()
    audit_report = (AUDIT / 'REPORT_KO.md').read_text()
    data = C.read(C.DOC / 'REPORT_DATA.json')
    gate = C.read(C.DOC / 'SOURCE_VAL_GATE.json')
    complete = C.read(C.DOC / 'TRAINING_COMPLETE.json')
    assert gate['complete'] and not gate['PASS'] and not gate['real_routing_authorized']
    assert complete['complete'] and complete['all_certified']
    C.verify(gate['metrics'])
    C.verify(gate['routing_lock'])
    routing = C.read(C.ROOT / gate['routing_lock']['path'])
    C.verify(routing['choices'])
    choices = C.read(C.ROOT / routing['choices']['path'])['records']
    assert data['gate'] == C.bind(C.DOC / 'SOURCE_VAL_GATE.json')
    assert data['code'] == C.bind(C.HERE / 'report.py')
    expected_logs, exported, fits = [], {}, {}
    for model in C.MODEL_NAMES:
        receipt = C.read(C.DOC / f'FIT_{model}.json')
        fits[model] = receipt
        for key in ('checkpoint', 'trace'):
            C.verify(receipt[key])
        raw = C.ROOT / receipt['checkpoint']['path']
        public = C.DOC / 'model_parameters' / f'{model}.json'
        assert raw.read_bytes() == public.read_bytes()
        assert data['exports'][model] == dict(original=receipt['checkpoint'], published=C.bind(public))
        exported[model] = dict(byte_identical=True, sha256=C.sha(public))
        trace = [json.loads(row) for row in (C.ROOT / receipt['trace']['path']).read_text().splitlines()]
        expected_logs += [dict(model=model, **row) for row in trace if row['event'] == 'objective']
        cert = receipt['certificate']
        line = f"| {model} | {receipt['iterations']} | {receipt['objective_calls']} | {cert['CE']:.9f} | {cert['gradient_l2']:.3e} | {cert['gradient_l2_squared_over_2lambda']:.3e} | PASS |"
        assert line in report
    actual = read_csv(C.DOC / 'TRAINING_OBJECTIVE_LOG.csv')
    assert len(actual) == len(expected_logs) == 1332
    for a, e in zip(actual, expected_logs):
        assert set(a) == set(e)
        for key in a:
            assert a[key] == csv_text(e[key]), (key, a[key], e[key])
    assert sum(f['iterations'] for f in fits.values()) == data['optimizer_iterations'] == 1208
    assert sum(f['wall_seconds'] for f in fits.values()) == data['cpu_fit_seconds']
    assert f"{data['cpu_fit_seconds']:.3f}초" in report
    assert 'fit 내부 CPU 시간의 합' not in report
    expected_frames = []
    with np.load(C.ROOT / gate['metrics']['path'], allow_pickle=False) as z:
        ids, models = z['ids'].tolist(), z['models'].tolist()
        assert len(ids) == 1024 and len(set(ids)) == 1024 and len(models) == 8
        for model in models:
            for fid, error in zip(ids, z[model]):
                route = choices.get(model, {}).get(fid, {})
                expected_frames.append(dict(id=fid, model=model, split='VAL', T_cm=float(error[0]),
                    R_deg=float(error[1]), available=bool(np.isfinite(error).all()),
                    candidate=route.get('candidate_name', 'fixed_GEO'), fallback=route.get('fallback', False)))
    actual = read_csv(C.DOC / 'SOURCE_VAL_FRAME_RESULTS.csv')
    assert len(actual) == len(expected_frames) == 8192
    for a, e in zip(actual, expected_frames):
        assert set(a) == set(e)
        for key in a:
            assert a[key] == csv_text(e[key]), (key, a[key], e[key])
    expected_checks = [dict(model=model, baseline=base, criterion=k, PASS=v)
        for model, comp in gate['comparisons'].items() for base, c in comp.items() for k, v in c['checks'].items()]
    actual = read_csv(C.DOC / 'SOURCE_VAL_CHECKS.csv')
    assert len(actual) == len(expected_checks) == 45
    for a, e in zip(actual, expected_checks):
        assert a == {k: csv_text(v) for k, v in e.items()}
    failed = [f"{c['model']}/{c['baseline']}/{c['criterion']}" for c in expected_checks if not c['PASS']]
    assert failed == gate['failed_checks'] and len(failed) == 4
    for model, s in gate['summaries'].items():
        p = s['full_population']
        values = [p[a][q] for a, q in [('translation_cm', 'median'), ('rotation_deg', 'median'), ('translation_cm', 'P90'), ('rotation_deg', 'P90')]]
        assert '| ' + model + ' | ' + ' | '.join(f'{v:.6f}' for v in values) + f" | {s['failed_pose']} |" in report
    r3 = gate['summaries']['UNION_s3']['full_population']
    for base in ('R0_ONLY', 'R0_GEO'):
        old = gate['summaries'][base]['full_population']
        assert f"| T 중앙값 vs {base} | {r3['translation_cm']['median']:.6f}cm | {old['translation_cm']['median']:.6f}cm |" in report
        assert f"| R P90 vs {base} | {r3['rotation_deg']['P90']:.6f}° | {old['rotation_deg']['P90']:.6f}° | 기준×1.05 이하({1.05*old['rotation_deg']['P90']:.6f}°) |" in report
    relative = (r3['rotation_deg']['P90'] / gate['summaries']['R0_ONLY']['full_population']['rotation_deg']['P90'] - 1) * 100
    assert f'{relative:.2f}%' in report
    stable_path = C.ROOT / '_docs/experiments/pallet_pose_stable_improvement_20261001_v1/RESULTS.json'
    stable = C.read(stable_path)['summaries']['NATURAL99']
    for model in ('R0', 'PRIOR1', 'FULL125'):
        p = stable[model]['full_population']
        assert f"| {model} | {p['translation_cm']['median']:.6f} | {p['rotation_deg']['median']:.6f} | {p['translation_cm']['P90']:.6f} |" in report
    avg = [np.mean([stable[f'DIVERSE251_s{s}']['full_population'][axis][q] for s in (1, 2, 3)]) for axis, q in
           [('translation_cm', 'median'), ('rotation_deg', 'median'), ('translation_cm', 'P90')]]
    assert '| DIVERSE251 세 seed 통계의 평균 | ' + ' | '.join(f'{v:.6f}' for v in avg) + ' |' in report
    oracle = C.read(AUDIT / 'VAL_ORACLE.json')
    C.verify(oracle['rows'])
    rows = read_csv(AUDIT / 'VAL_ORACLE_ROWS.csv')
    assert len(rows) == 3072
    with np.load(C.PARENT_RAW / 'SOURCE_VAL_METRICS.npz', allow_pickle=False) as z:
        position = {fid: j for j, fid in enumerate(z['ids'].tolist())}
        for seed in (1, 2, 3):
            selected = [row for row in rows if int(row['seed']) == seed]
            assert len(selected) == 1024 and set(r['id'] for r in selected) == set(position)
            for row in selected:
                np.testing.assert_array_equal([float(row['learned_T_cm']), float(row['learned_R_deg'])], z[f'UNION_s{seed}'][position[row['id']]])
                for label in ('learned', 'oracle'):
                    calculated = max(float(row[label + '_T_cm']) / oracle['scale'][0], float(row[label + '_R_deg']) / oracle['scale'][1])
                    assert calculated == float(row[label + '_cost'])
                assert float(row['regret']) == float(row['learned_cost']) - float(row['oracle_cost'])
                assert row['diagnostic_only'] == 'True'
            for what, label in [('learned', '기존 학습 선택'), ('oracle', 'GT를 아는 oracle')]:
                p = oracle['seeds'][f'UNION_s{seed}'][what]['full_population']
                values = [p[a][q] for a, q in [('translation_cm', 'median'), ('rotation_deg', 'median'), ('translation_cm', 'P90'), ('rotation_deg', 'P90')]]
                line = f'| s{seed} {label} | ' + ' | '.join(f'{v:.6f}' for v in values) + ' |'
                assert line in audit_report
                err = np.array([[float(r[what+'_T_cm']), float(r[what+'_R_deg'])] for r in selected])
                for j, axis in enumerate(('translation_cm', 'rotation_deg')):
                    for name, q in [('median', .5), ('P90', .9)]:
                        np.testing.assert_allclose(np.quantile(err[:, j], q), p[axis][name], rtol=1e-14, atol=1e-12)
    detection = C.read(AUDIT / 'DETECTION_SOURCE_FEASIBILITY.json')
    d = detection['all']
    assert d['frames'] == 5120 and d['candidates'] == 7173
    assert list(d['categories'].values()) == [6785, 275, 50, 63]
    assert d['IoU_0.5']['selected_match'] == 5113 and d['IoU_0.5']['selected_miss_with_matching_alternative'] == 6 and d['no_detection'] == 1
    low = [v for k, v in d['by_category_pose_validity'].items() if k != 'MATCH_IOU_GE_0_5']
    assert sum(v['candidates'] for v in low) == sum(v['any_pose_all_corners_positive_depth'] for v in low) == 388
    disjoint = detection['by_split']['VAL']['by_category_pose_validity']['TARGET_BOX_DISJOINT_NOT_SEMANTIC_NEGATIVE']
    assert disjoint['candidates'] == 42 and disjoint['score_ge_0_1'] == 0
    assert sum(r['count'] for r in detection['negative9k']['metadata_counts'] if r['provenance'] == 'legacy_unavailable') == 4385
    train = C.read(AUDIT / 'TRAIN_DIAGNOSTIC.json')
    assert train['frames'] == 2598
    figure_data = {}
    for seed in (1, 2, 3):
        t = train['fits'][f'UNION_s{seed}']
        assert t['regret']['n'] == 2597 and t['summary']['failures'] == 1
        high = t['top10percent_regret_frames']
        assert high['n'] == 260 and high['regret_share'] > .95
        figure_data[str(seed)] = dict(regret_share=high['regret_share'], CE_share=high['CE_share'])
    no_real = [C.DOC / 'REAL_ROUTING_LOCK.json', C.DOC / 'REAL_RESULTS.json', C.RAW / 'REAL_CHOICES.json']
    assert not any(p.exists() for p in no_real)
    reviewed_files = [C.DOC / p for p in ('REPORT_KO.md', 'REPORT_DATA.json', 'SOURCE_VAL_GATE.json',
        'TRAINING_OBJECTIVE_LOG.csv', 'SOURCE_VAL_FRAME_RESULTS.csv', 'SOURCE_VAL_CHECKS.csv',
        'figures/objective_convergence.png', 'figures/source_val_results.png')]
    reviewed_files += [AUDIT / p for p in ('REPORT_KO.md', 'VAL_ORACLE.json', 'VAL_ORACLE_ROWS.csv',
        'TRAIN_DIAGNOSTIC.json', 'DETECTION_SOURCE_FEASIBILITY.json', 'figures/source_val_oracle_gap.png',
        'figures/train_error_cost_vs_ce.png', 'figures/source_candidate_contract.png')]
    reviewed_files += [C.HERE / 'report.py', C.ROOT / 'scripts/research/pallet_pose_selector_objective_audit_20261001_v1/report.py', Path(__file__), stable_path]
    result = dict(complete=True, PASS=True, created_at=C.now(), scope='READ_ONLY_PUBLICATION_REVIEW',
        numeric_checks=dict(training_csv_rows=1332, training_csv_every_column_exact=True,
            frame_csv_rows=8192, frame_csv_every_column_exact=True, frame_numeric_T_R_values=16384,
            gate_csv_rows=45, gate_csv_every_column_exact=True, gate_passed=41, gate_failed=4,
            learned_model_checkpoints_byte_identical=4, main_fit_rows_exact_to_display_precision=4,
            main_source_summary_rows_exact_to_display_precision=8, main_failed_checks_rows=4,
            old_real_context_rows=4, audit_oracle_table_rows=6, audit_oracle_csv_rows=3072,
            audit_oracle_learned_errors_exact_to_old_npz=True, oracle_cost_regret_and_summary_consistency=True),
        exported_weights=exported, inspected_figures=5,
        visual_review='All five PNGs viewed directly. Units, candidate/seed categories, numerical bar labels and 0-based bar axes match JSON/code. Convergence figure x-axis is objective evaluations, not epochs. Source chart shows R0 GEO/shared control/three UNION models; full eight-model table retained.',
        train_share_plot_values=figure_data,
        interpretation_review='PASS: convergence is not joint T/R success; one-hot CE+explicit ridge scope only; original AdamW is distinguished; single RGB+dimensions retains calibrated K; oracle is diagnostic; no seed dropped; 41/45 gate failure stops all new real routing.',
        resolved_wording=['wall_seconds described as elapsed CPU-run fit time, not process CPU time',
            'source chart alt text no longer claims all eight models are plotted',
            'immutability claim scoped to original training/evaluation artifacts'],
        limitations=['No raw source geometry or real GT reopened. Oracle per-frame references were not recomputed; learned errors matched the frozen NPZ and oracle cost/regret/quantile consistency was checked.',
            'Manifest and virtual publication-tree link/AST checks are root-owned and run after this review.',
            'Unused real evaluator execution is not certified by this publication review; source gate failed and no new real routes exist.'],
        bindings=[C.bind(path) for path in reviewed_files], new_fits=0, new_GT_reads=0,
        read_paths=sorted(set(READS)))
    C.save(C.DOC / 'PUBLIC_REVIEW.json', result)
    print('PUBLICATION_REVIEW_PASS: 1332 objective rows,8192 frame rows,45 gates,4 byte-identical checkpoints,2 reports,5 figures.', flush=True)


if __name__ == '__main__':
    main()
