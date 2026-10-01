"""Independent source VAL route, physical metric, and 45-gate verification.

Run verify only after root reports the four-model freeze and score complete.
No evaluation helper is imported. Before all 4096 saved routes are verified,
the audit hook denies source reference and VAL quality containers. Afterwards
only the locked 1024 VAL reference indices are scored; real GT stays denied.
"""
import argparse
from collections import Counter
import hashlib
import json
import math
import os
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
DOC = ROOT / '_docs/experiments' / HERE.name
RAW = ROOT / 'data/pallet/results' / HERE.name
PARENT_DOC = ROOT / '_docs/experiments/pallet_pose_union_selection_20261001_v1'
PARENT_RAW = ROOT / 'data/pallet/results/pallet_pose_union_selection_20261001_v1'
SOURCE = ROOT / 'data/pallet/results/pallet_selector_recovery_v1/stage2_synth_scorer'
HYP = ('long-face-front', 'short-face-front')
LEARNED = ('R0_ONLY', 'UNION_s1', 'UNION_s2', 'UNION_s3')
BASELINES = ('R0_GEO', 'DIVERSE251_s1_GEO', 'DIVERSE251_s2_GEO', 'DIVERSE251_s3_GEO')
METRICS = ('translation_cm', 'rotation_deg')
STATE = {'references_allowed': False, 'reads': set()}


def read(path):
    return json.loads(Path(path).read_text())


def bind(path):
    path = Path(path).resolve()
    return dict(path=str(path.relative_to(ROOT)), sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                bytes=path.stat().st_size)


def verify(binding):
    actual = bind(ROOT / binding['path'])
    assert actual['sha256'] == binding['sha256'], binding['path']
    assert 'bytes' not in binding or actual['bytes'] == binding['bytes']


def bindings(value):
    if isinstance(value, dict):
        if isinstance(value.get('path'), str) and isinstance(value.get('sha256'), str):
            yield value
        else:
            for child in value.values():
                yield from bindings(child)
    elif isinstance(value, list):
        for child in value:
            yield from bindings(child)


def array_sha(value):
    value = np.ascontiguousarray(value)
    digest = hashlib.sha256(str(value.dtype).encode())
    digest.update(json.dumps(list(value.shape)).encode())
    digest.update(value.tobytes())
    return digest.hexdigest()


def install_guard():
    allowed_writes = {DOC / 'SOURCE_VAL_VERIFICATION.json', DOC / 'SOURCE_VAL_VERIFICATION_KO.md'}
    def hook(event, args):
        if event != 'open' or not isinstance(args[0], (str, bytes, os.PathLike)):
            return
        path = Path(os.fsdecode(args[0])).resolve()
        if not path.is_relative_to(ROOT):
            return
        text = str(path)
        mode = args[1]
        if isinstance(mode, str) and any(letter in mode for letter in 'wax+'):
            assert path in allowed_writes, ('VERIFIER_WRITE_SCOPE', text)
            return
        assert not any(token in text for token in ('/data/evaluation/', '/real_gt_v2/',
            '/annotations/', 'GEOMETRY_RESOLVED_POSE_GT', 'AXIS_REVIEW_MANIFEST', 'TRUTH_FOR_DISPLAY'))
        assert path.suffix.lower() not in ('.png', '.jpg', '.jpeg', '.pt', '.pth', '.onnx'), text
        if not STATE['references_allowed']:
            assert not any(token in text for token in ('GEOMETRY_SIDETABLE.npz', '/SYNTH_RECORDS.json',
                '/SOURCE_MANIFEST.json', '/DIMENSION_SIDECAR.json', '/SYNTH_LABELS.npz',
                'SOURCE_VAL_METRICS.npz', 'SOURCE_VAL_GATE.json', 'VAL_ORACLE.json')), ('BEFORE_ROUTE_VERIFICATION', text)
        STATE['reads'].add(str(path.relative_to(ROOT)))
    sys.addaudithook(hook)


def model_parts(model):
    return ['R0'] + ([] if model == 'R0_ONLY' else [f'DIVERSE251_s{model[-1]}'])


def names_for(model):
    return [parent + ':' + hypothesis for parent in model_parts(model) for hypothesis in HYP]


def tie_key(name):
    parent, hypothesis = name.split(':')
    return parent != 'R0', hypothesis, parent


def selected_pose(choice, records, fid):
    if choice['fallback']:
        assert choice['candidate_index'] == -1 and choice['candidate_name'] is None
        return records['R0'][fid]['GEO_pose']
    parent, hypothesis = choice['candidate_name'].split(':')
    assert parent == choice['parent'] and hypothesis == choice['hypothesis']
    pool = [h['pose'] for h in records[parent][fid]['hypotheses'] if h['name'] == hypothesis]
    assert len(pool) == 1 and pool[0]['available']
    return pool[0]


def independent_scores(checkpoint, features, valid):
    mean, std = np.asarray(checkpoint['mean'], np.float32), np.asarray(checkpoint['std'], np.float32)
    assert features.dtype == np.float32 and valid.dtype == bool
    normalized = np.zeros_like(features)
    normalized[valid] = (features[valid] - mean) / std
    values = np.sum(normalized.astype(np.float64) * np.asarray(checkpoint['weight'], np.float64), axis=2)
    return np.where(valid, values, np.inf)


def verify_routes():
    from . import common as C
    from . import convex_train as T
    assert T.OLD.READS is None, 'Do not import a fitting or source-score process into this verifier.'
    protocol = C.protocol('TRAIN_PROTOCOL')
    assert protocol['models'] == list(LEARNED)
    assert protocol['target_rule'] == 'R0_GEO_PARETO_NONREGRESSION_FIXED_TRAIN_MINMAX'
    assert protocol['normalization'] == 'old_float32_then_float64'
    complete = read(DOC / 'TRAINING_COMPLETE.json')
    assert complete['complete'] and complete['all_certified'] and complete['fit_count'] == 4
    assert complete['models'] == list(LEARNED) and complete['protocol'] == bind(DOC / 'TRAIN_PROTOCOL.json')
    assert complete['source_TRAIN_only'] and not complete['VAL_quality_read'] and not complete['real_targets_read']
    lock = read(DOC / 'SOURCE_VAL_ROUTING_LOCK.json')
    assert lock['complete'] and lock['frames'] == 1024 and lock['models'] == list(LEARNED)
    assert lock['baselines'] == list(BASELINES)
    assert lock['protocol'] == bind(DOC / 'TRAIN_PROTOCOL.json')
    assert lock['training_complete'] == bind(DOC / 'TRAINING_COMPLETE.json')
    assert not lock['source_labels_read'] and not lock['source_VAL_label_values_read']
    assert not lock['real_references_read'] and not lock['VAL_quality_scored']
    for binding in bindings(lock):
        verify(binding)
    contract = read(ROOT / lock['source_contract']['path'])
    assert contract['complete'] and contract['status'] == 'PASS'
    assert protocol['inputs']['source_contract'] == lock['source_contract']
    input_rows = read(ROOT / lock['metadata']['path'])
    positions = np.flatnonzero([row['split'] == 'VAL' for row in input_rows])
    rows = [input_rows[j] for j in positions]
    ids = [row['id'] for row in rows]
    assert len(ids) == len(set(ids)) == 1024
    assert set(ids) == set(contract['fit_eligibility']['eligible_ids']['VAL'])
    assert set(ids).isdisjoint(contract['fit_eligibility']['eligible_ids']['TRAIN'])
    feature_lock = read(ROOT / lock['feature_lock']['path'])
    verify(feature_lock['features'])
    assert feature_lock['features'] == protocol['inputs']['features']
    assert feature_lock['poses'] == lock['poses'] and feature_lock['metadata'] == lock['metadata']
    poses = read(ROOT / lock['poses']['path'])
    assert poses['ids'] == [row['id'] for row in input_rows]
    assert not poses['source_targets_read'] and not poses['real_targets_read']
    choices = read(ROOT / lock['choices']['path'])
    assert choices['ids'] == ids and choices['models'] == list(LEARNED)
    assert not choices['source_labels_read'] and not choices['real_references_read']
    with np.load(ROOT / feature_lock['features']['path'], allow_pickle=False) as stored:
        assert stored['ids'].tolist() == poses['ids']
        assert stored['split'].tolist() == [row['split'] for row in input_rows]
        assert stored['hypothesis_names'].tolist() == list(HYP)
        features = {m: stored[m + '_geo'][positions] for m in ('R0',) + tuple(f'DIVERSE251_s{s}' for s in (1, 2, 3))}
        valid = {m: stored[m + '_valid'][positions] for m in features}
    model_checks, fit_models, score_gap = {}, set(), 0.
    for receipt_binding in complete['fits']:
        verify(receipt_binding)
        receipt = read(ROOT / receipt_binding['path'])
        model = receipt['model']
        assert model in LEARNED and model not in fit_models
        fit_models.add(model)
        assert receipt['complete'] and receipt['protocol'] == lock['protocol']
        assert receipt_binding == lock['fits'][model]['receipt']
        assert receipt['checkpoint'] == lock['fits'][model]['checkpoint']
        for key in ('checkpoint', 'START', 'trace'):
            verify(receipt[key])
        certificate = receipt['certificate']
        assert certificate['PASS'] and certificate['optimizer_success']
        assert certificate['lambda_l2'] == 1e-4 and certificate['gradient_l2_squared_over_2lambda'] <= 1e-6
        assert 0 < certificate['objective_calls'] <= 2000 and 0 <= certificate['iterations'] <= 1000
        checkpoint = read(ROOT / receipt['checkpoint']['path'])
        assert checkpoint['protocol'] == lock['protocol'] and checkpoint['model'] == model
        assert checkpoint['certificate'] == certificate and checkpoint['names'] == names_for(model)
        assert checkpoint['target_rule'] == protocol['target_rule']
        assert checkpoint['normalization'] == 'old_float32_then_float64'
        mean, std = np.asarray(checkpoint['mean'], np.float32), np.asarray(checkpoint['std'], np.float32)
        assert (std >= np.float32(1e-6)).all()
        assert array_sha(np.stack([mean, std])) == receipt['normalization_sha'] == checkpoint['normalization_sha']
        assert array_sha(np.asarray(checkpoint['weight'], np.float64)) == receipt['final_weight_sha']
        x = np.concatenate([features[m] for m in model_parts(model)], axis=1)
        mask = np.concatenate([valid[m] for m in model_parts(model)], axis=1)
        names = names_for(model)
        scores = T.score_candidates(checkpoint, x, mask)
        manual = independent_scores(checkpoint, x, mask)
        local_gap = float(np.abs(scores[mask] - manual[mask]).max()) if mask.any() else 0.
        assert local_gap <= 1e-10
        score_gap = max(score_gap, local_gap)
        assert set(choices['records'][model]) == set(ids)
        counts = Counter()
        for j, fid in enumerate(ids):
            record = choices['records'][model][fid]
            assert record['candidate_valid'] == mask[j].tolist(), 'Runtime must use original geometric validity, never a reference-safe mask.'
            available = np.flatnonzero(mask[j]).tolist()
            selected = min(available, key=lambda k: (float(scores[j, k]), tie_key(names[k]))) if available else -1
            manual_selected = min(available, key=lambda k: (float(manual[j, k]), tie_key(names[k]))) if available else -1
            assert selected == manual_selected == record['candidate_index']
            assert record['scores'] == [float(s) if flag else None for s, flag in zip(scores[j], mask[j])]
            assert record['candidate_name'] == (names[selected] if selected >= 0 else None)
            assert record['fallback'] == (selected < 0)
            pose = selected_pose(record, poses['records'], fid)
            assert record['pose_available'] == bool(pose['available'])
            expected_status = 'SELECTED' if selected >= 0 else 'R0_GEO_FALLBACK' if pose['available'] else 'FAILED'
            assert record['status'] == expected_status
            counts[expected_status] += 1
        model_checks[model] = dict(rows=1024, original_geometric_mask_exact=True,
            selected_indices_and_names_exact=True, shared_score_cells_exact=int(mask.sum()),
            independent_score_max_absolute_error=local_gap, status_counts=dict(counts),
            checkpoint=receipt['checkpoint'], fit_receipt=receipt_binding)
    assert fit_models == set(LEARNED)
    assert not STATE['references_allowed']
    return protocol, lock, contract, rows, poses, choices, model_checks, score_gap


def physical_errors(pose, renderer_rotation, renderer_translation):
    """Canonical SO(3) trace metric, cross-checked by Frobenius inner product."""
    if not pose['available']:
        return np.array([np.inf, np.inf], np.float64), 0.
    rotation = np.asarray(pose['R_physical'], np.float64)
    translation = np.asarray(pose['centroid'], np.float64)
    reference = np.asarray(renderer_rotation, np.float64) @ np.diag([1., -1., -1.])
    assert rotation.shape == reference.shape == (3, 3) and translation.shape == (3,)
    assert np.isfinite(rotation).all() and np.isfinite(translation).all()
    angles, alternate = [], []
    for symmetry in (np.eye(3), np.diag([-1., 1., -1.])):
        equivalent = reference @ symmetry
        relative = equivalent.T @ rotation
        cosine = np.clip((np.trace(relative) - 1.) / 2., -1., 1.)
        angles.append(float(np.degrees(np.arccos(cosine))))
        inner = float(np.sum(equivalent * rotation))
        alternate.append(float(np.degrees(np.arccos(np.clip((inner - 1.) / 2., -1., 1.)))))
    error = np.array([float(np.linalg.norm(translation - renderer_translation) * 100.), min(angles)])
    assert np.isfinite(error).all() and (error >= 0).all()
    gap = abs(error[1] - min(alternate))
    assert gap <= 1e-6, ('EQUIVALENT_C2_FORMULA_GAP', gap)
    return error, gap


def quantile(values, fraction):
    values = sorted(float(v) for v in values)
    assert not any(math.isnan(v) or v < 0 for v in values)
    if not values:
        return None
    rank = (len(values)-1) * fraction
    lower, upper = math.floor(rank), math.ceil(rank)
    if lower == upper:
        return values[lower]
    if math.isinf(values[upper]):
        return float('inf')
    return values[lower] + (rank-lower) * (values[upper]-values[lower])


def summary(errors):
    available = np.isfinite(errors).all(1)
    assert np.array_equal(available, np.isfinite(errors).any(1))
    value = dict(frames=len(errors), valid_pose=int(available.sum()), failed_pose=int((~available).sum()))
    for mode, rows in [('full_population', errors), ('conditional', errors[available])]:
        value[mode] = {}
        for j, metric in enumerate(METRICS):
            cell = {}
            for label, q in [('median', .5), ('P90', .9)]:
                result = quantile(rows[:, j], q)
                cell[label] = result if result is not None and math.isfinite(result) else None
                cell[label + '_status'] = 'NA_EMPTY' if result is None else 'FINITE' if math.isfinite(result) else 'POSITIVE_INFINITY'
            value[mode][metric] = cell
    return value


def checks(before, after):
    output = {'failure_no_increase': int((~np.isfinite(after).all(1)).sum()) <= int((~np.isfinite(before).all(1)).sum())}
    for j, metric in enumerate(METRICS):
        output[metric + '_median_strict'] = quantile(after[:, j], .5) < quantile(before[:, j], .5)
        output[metric + '_P90_guard'] = quantile(after[:, j], .9) <= 1.05 * quantile(before[:, j], .9)
    return output


def reference_chain(contract):
    history = ROOT / '_docs/experiments/pallet_selector_recovery_v1/stage2_synth_scorer/SYNTHETIC_SPLIT_LOCK.json'
    geometry = ROOT / 'challenge/yolo_pose_one_model/pallet_translation_loss_v1/GEOMETRY_SIDETABLE.npz'
    record_path = SOURCE / 'SYNTH_RECORDS.json'
    expected = {}
    for path in (history, geometry, record_path):
        rel = str(path.relative_to(ROOT))
        matches = [b for b in bindings(contract) if b['path'] == rel]
        assert matches and all(b == matches[0] for b in matches)
        expected[rel] = matches[0]
    history_binding = expected[str(history.relative_to(ROOT))]
    verify(history_binding)
    historical = read(history)
    for path in (geometry, record_path):
        rel = str(path.relative_to(ROOT))
        matches = [b for b in bindings(historical) if b['path'] == rel]
        assert matches == [expected[rel]]
        verify(expected[rel])
    return history_binding, expected, geometry, record_path


def run():
    sys.dont_write_bytecode = True
    assert not (DOC / 'SOURCE_VAL_VERIFICATION.json').exists()
    assert not (DOC / 'SOURCE_VAL_VERIFICATION_KO.md').exists()
    install_guard()
    protocol, lock, contract, rows, poses, choices, route_checks, score_gap = verify_routes()
    # The complete GT-free route replay and its original validity masks passed.
    STATE['references_allowed'] = True
    gate = read(DOC / 'SOURCE_VAL_GATE.json')
    assert gate['complete'] and gate['routing_lock'] == bind(DOC / 'SOURCE_VAL_ROUTING_LOCK.json')
    assert gate['protocol'] == bind(DOC / 'TRAIN_PROTOCOL.json')
    assert gate['training_complete'] == bind(DOC / 'TRAINING_COMPLETE.json')
    assert gate['frames'] == 1024 and gate['models'] == list(LEARNED) and gate['baselines'] == list(BASELINES)
    assert not gate['real_reference_accessed']
    history_binding, references, geometry, record_path = reference_chain(contract)
    assert gate['source_history_binding'] == history_binding
    assert all(b == references[b['path']] for b in gate['source_reference_bindings'])
    ids = [row['id'] for row in rows]
    wanted = set(ids)
    metadata = {r['id']: r for r in read(record_path) if r['id'] in wanted}
    assert set(metadata) == wanted and all(r['split'] == 'VAL' for r in metadata.values())
    index = np.array([metadata[fid]['table_index'] for fid in ids], np.int64)
    assert len(set(index.tolist())) == 1024
    with np.load(geometry, allow_pickle=False) as table:
        assert table['stems'][index].tolist() == ids
        rotations, translations = table['R'][index], table['t'][index]
        intrinsics, dimensions = table['K'][index], table['dims'][index]
    for j, row in enumerate(rows):
        fx, fy, cx, cy = intrinsics[j]
        np.testing.assert_allclose(row['K'], [[fx, 0, cx], [0, fy, cy], [0, 0, 1]], rtol=0, atol=1e-7)
        np.testing.assert_allclose(row['dims'], dimensions[j], rtol=0, atol=1e-7)
    recomputed = {m: np.full((1024, 2), np.inf, np.float64) for m in LEARNED + BASELINES}
    frobenius_gap = 0.
    for j, fid in enumerate(ids):
        for model in LEARNED + BASELINES:
            pose = selected_pose(choices['records'][model][fid], poses['records'], fid) if model in LEARNED else poses['records'][model[:-4]][fid]['GEO_pose']
            value, difference = physical_errors(pose, rotations[j], translations[j])
            recomputed[model][j] = value
            frobenius_gap = max(frobenius_gap, difference)
    verify(gate['metrics'])
    physical_max_gap = 0.
    with np.load(ROOT / gate['metrics']['path'], allow_pickle=False) as stored:
        assert stored['ids'].tolist() == ids and stored['models'].tolist() == list(LEARNED + BASELINES)
        assert stored['metric_names'].tolist() == list(METRICS)
        np.testing.assert_array_equal(stored['source_table_index'], index)
        saved = {m: stored[m] for m in LEARNED + BASELINES}
    for model, errors in recomputed.items():
        np.testing.assert_array_equal(errors, saved[model])
        finite = np.isfinite(errors)
        physical_max_gap = max(physical_max_gap, float(np.abs(errors[finite] - saved[model][finite]).max()) if finite.any() else 0.)
        assert summary(errors) == gate['summaries'][model]
    old_gate_path = ROOT / '_docs/experiments/pallet_pose_selector_convergence_20261001_v1/SOURCE_VAL_GATE.json'
    old_gate = read(old_gate_path)
    assert old_gate['complete'] and old_gate['frames'] == 1024
    assert old_gate['source_contract'] == gate['source_contract']
    assert old_gate['source_history_binding'] == history_binding and old_gate['source_reference_bindings'] == gate['source_reference_bindings']
    verify(old_gate['metrics'])
    baseline_parity = {}
    with np.load(ROOT / old_gate['metrics']['path'], allow_pickle=False) as previous:
        assert previous['ids'].tolist() == ids
        for model in BASELINES:
            assert previous[model].dtype == saved[model].dtype == np.float64
            assert previous[model].shape == saved[model].shape == (1024, 2)
            assert previous[model].tobytes() == saved[model].tobytes()
            baseline_parity[model] = dict(error_cells=2048, bit_exact=True, array_sha=array_sha(saved[model]))
    failed, count = [], 0
    for seed in (1, 2, 3):
        model = f'UNION_s{seed}'
        for baseline in ('R0_ONLY', 'R0_GEO', f'DIVERSE251_s{seed}_GEO'):
            original = gate['comparisons'][model][baseline]
            actual = checks(recomputed[baseline], recomputed[model])
            assert actual == original['checks']
            assert original['PASS'] == all(actual.values())
            assert original['before'] == summary(recomputed[baseline]) and original['after'] == summary(recomputed[model])
            count += len(actual)
            failed.extend(f'{model}/{baseline}/{criterion}' for criterion, passed in actual.items() if not passed)
    assert count == gate['checks_total'] == 45 and failed == gate['failed_checks']
    assert 45 - len(failed) == gate['checks_passed'] and gate['PASS'] == (not failed)
    assert gate['real_routing_authorized'] == gate['PASS']
    verdict = 'PASS' if gate['PASS'] else 'FAIL'
    from datetime import datetime, timezone
    result = dict(complete=True, PASS=True, created_at=datetime.now(timezone.utc).isoformat(),
        independent_verification=True, implementation='GT-free shared scorer replay plus independent float32-normalized multiply-sum/tie selection; direct cached-pose C2 trace and Frobenius checks; independent sorted interpolation and all45 gates. No evaluate_source/source_features metric helper imported.',
        verifier=bind(Path(__file__)), source_gate=bind(DOC / 'SOURCE_VAL_GATE.json'),
        protocol=bind(DOC / 'TRAIN_PROTOCOL.json'), training_complete=bind(DOC / 'TRAINING_COMPLETE.json'),
        routing_lock=bind(DOC / 'SOURCE_VAL_ROUTING_LOCK.json'), choices=bind(ROOT / lock['choices']['path']),
        reference_chain=list(references.values()), VAL_table_index_sha=array_sha(index),
        route_rows_reproduced=4096, route_models=route_checks,
        original_runtime_valid_masks_preserved=True, GT_safe_mask_used_in_runtime=False,
        independent_score_max_absolute_difference=score_gap, source_VAL_reference_rows_scored=1024,
        source_TRAIN_reference_rows_scored=0, source_TEST_reference_rows_scored=0, real_reference_rows_scored=0,
        recomputed_pose_errors=8192, recomputed_error_cells_exact=16384,
        physical_metric_max_absolute_difference=physical_max_gap,
        alternate_Frobenius_C2_max_difference_deg=frobenius_gap,
        summary_models_exact=8, full_population_frames_per_model=1024,
        fixed_GEO_baseline_bit_parity=baseline_parity, previous_baseline_metrics=old_gate['metrics'],
        previous_baseline_gate=bind(old_gate_path), checks_total=45, checks_passed=gate['checks_passed'],
        failed_checks=failed, source_gate_PASS=gate['PASS'], all_gate_booleans_exact=True,
        label_container_disclosure='The historically bound geometry NPZ contains broader source rows; only the1024 verified VAL indices were selected and scored. No TRAIN/TEST/real reference errors were computed.',
        source_reference_read_only_after_all_routes_verified=True, read_paths=sorted(STATE['reads']),
        new_fits=0, image_forwards=0, new_PnP_calls=0, new_route_artifacts=0, threshold_changes=0,
        real_GT_reads=0, method_success=False, goal_complete=False)
    markdown = f'''# Source VAL 독립 검산

**검산 PASS. 학습된 방법의 source gate는 {verdict} ({gate['checks_passed']}/45)**이다. 검산 통과와 방법 성공은 별개다.

GT 접근을 막은 상태에서 최종4개 checkpoint의 **4,096개 선택**을 원래 cached feature로 재현했다. float32 정규화→float64 scorer와 별도 곱셈·합산/동률 선택을 비교했으며 모든 index·후보 이름·fallback·원래 geometric-valid mask가 일치했다. runtime에서 참조 기반 safe mask를 사용하지 않았다. 독립 score 최대 차이는 {score_gap:.3g}이다.

전체 routing lock과 모든 최종 fit 바인딩 확인 후에만 기존 source 참조의 **VAL1,024개 index**를 읽었다. renderer R에 `diag(1,-1,-1)`를 적용한 물리 기준과 C2 대칭 회전, 중심 이동 cm를 직접 계산했다. 평가 helper의 metric/gate 함수는 import하지 않았다. **8모델×1,024 pose = 8,192개**, T/R **16,384개 오류 값**이 저장 배열과 정확히 일치한다. 동일 회전각의 Frobenius 식도 확인했으며 최대 차이는 {frobenius_gap:.3g}°다.

이전 convergence 결과의 R0 GEO 및 DIVERSE3 GEO **8,192개 오류 값**은 현재 fixed baseline과 byte 단위로 같다. 기존 결과를 새 모델 성능처럼 다시 생성하거나 baseline을 바꾸지 않았다.

전체1,024분모·실패/+∞·median/P90을 독립 정렬 보간으로 확인하고, 각 UNION을 R0_ONLY/R0_GEO/paired DIVERSE_GEO와 비교하는 **45개 Boolean**을 모두 재현했다. 임계값·seed·checkpoint를 바꾸지 않았다. 실패 목록은 JSON에 전부 남긴다.

원본 geometry 컨테이너에는 다른 source 행도 들어 있지만 검산 대상은 잠금된 VAL1,024 index뿐이다. TRAIN/TEST/실사 참조 오류 계산, 새 fit, 이미지 forward, PnP 및 새 routing 산출물은0회다. 이 VAL은 반복 사용 개발 자료이며 실제 실사 개선이나 전체 목표 달성의 증거가 아니다.

[검산 JSON](SOURCE_VAL_VERIFICATION.json), [source gate](SOURCE_VAL_GATE.json), [선택 동결](SOURCE_VAL_ROUTING_LOCK.json)
'''
    with (DOC / 'SOURCE_VAL_VERIFICATION_KO.md').open('x') as handle:
        handle.write(markdown)
    result['note'] = bind(DOC / 'SOURCE_VAL_VERIFICATION_KO.md')
    with (DOC / 'SOURCE_VAL_VERIFICATION.json').open('x') as handle:
        handle.write(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    print('SOURCE_VAL_INDEPENDENT_VERIFICATION_PASS', dict(source_gate=verdict,
        checks_passed=gate['checks_passed'], route_rows=4096, error_cells=16384,
        baseline_bit_exact_cells=8192, receipt=bind(DOC / 'SOURCE_VAL_VERIFICATION.json')), flush=True)


def selfcheck():
    from . import convex_train as T
    assert T.OLD.READS is None
    pose = dict(available=True, R_physical=np.eye(3).tolist(), centroid=[0., 0., 1.])
    basis = np.diag([1., -1., -1.])
    np.testing.assert_array_equal(physical_errors(pose, basis, np.array([0., 0., 1.]))[0], [0., 0.])
    np.testing.assert_array_equal(physical_errors(pose, np.diag([-1., -1., 1.]), np.array([.01, 0., 1.]))[0], [1., 0.])
    assert np.isposinf(physical_errors(dict(available=False), basis, np.zeros(3))[0]).all()
    equal = np.array([[1., 2.], [3., 4.]])
    assert not checks(equal, equal)['translation_cm_median_strict']
    assert quantile([1., np.inf], .5) == np.inf
    assert summary(np.array([[1., 2.], [np.inf, np.inf]]))['failed_pose'] == 1
    checkpoint = dict(schema='pallet_pose_convex_linear94_v1', bias=0., lambda_l2=1e-4,
        weight=np.arange(94, dtype=float).tolist(), mean=np.zeros(94).tolist(), std=np.ones(94).tolist())
    x = np.arange(2*4*94, dtype=np.float32).reshape(2, 4, 94) / np.float32(100)
    valid = np.array([[True, False, True, True], [False]*4])
    np.testing.assert_allclose(independent_scores(checkpoint, x, valid),
                               T.score_candidates(checkpoint, x, valid), rtol=1e-15, atol=1e-10)
    assert not STATE['references_allowed'] and not STATE['reads']
    print('VERIFY_SOURCE_SELFCHECK_PASS: invented C2/basis, exact median ties, failures and independent scorer; no data artifacts opened.', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('stage', choices=('selfcheck', 'verify'))
    arguments = parser.parse_args()
    (selfcheck if arguments.stage == 'selfcheck' else run)()
