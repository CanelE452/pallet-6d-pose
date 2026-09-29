"""동일 GEO 무학습 대조를 독립 집계·선택·출처 검사한다. 기존 파일 쓰기 없음."""
from __future__ import annotations

from collections import Counter
from datetime import datetime
import math
from pathlib import Path
import subprocess

import numpy as np

from . import common as C

FIELDS = ('translation_cm', 'rotation_deg', 'yaw_deg')
MODELS = ('R0', 'OLD_REF', 'RAW_CLEAR_S42', 'REF_CLEAR_S42',
          'RAW_OCC_S42', 'REF_OCC_S42', 'RAW_OCC_S43', 'REF_OCC_S43')
FORBIDDEN_READS = ('ORACLE_METRICS', 'POSE_METRICS.json', 'FRAME_METRICS.json',
    'GEOMETRY_RESOLVED', 'TRUTH_FOR_DISPLAY', 'VERIFIED_LABELS', 'EVAL_RESULTS',
    'CANDIDATE_ORACLE', '/data/evaluation/', 'AXIS_REVIEW')


def close(a, b):
    if isinstance(a, dict):
        assert isinstance(b, dict) and set(a) == set(b)
        for key in a:
            close(a[key], b[key])
    elif isinstance(a, (list, tuple)):
        assert isinstance(b, (list, tuple)) and len(a) == len(b)
        for x, y in zip(a, b):
            close(x, y)
    elif isinstance(a, (int, float)) and not isinstance(a, bool):
        assert np.isclose(a, b, rtol=1e-7, atol=1e-8), (a, b)
    else:
        assert a == b, (a, b)


class Verifier:
    def __init__(self):
        self.seen = set()
        self.cache = {}

    def binding(self, row):
        key = row['path'], row['sha256']
        if key not in self.seen:
            C.verify(row)
            if 'bytes' in row:
                assert (C.ROOT/row['path']).stat().st_size == row['bytes']
            self.seen.add(key)

    def nested(self, value):
        if isinstance(value, dict):
            if {'path', 'sha256'} <= set(value):
                self.binding(value)
            for child in value.values():
                self.nested(child)
        elif isinstance(value, list):
            for child in value:
                self.nested(child)

    def read(self, path):
        path = Path(path)
        if path not in self.cache:
            self.cache[path] = C.read(path)
        return self.cache[path]


def distribution(values):
    """유효값 조건부와 실패 +inf 전체 분모를 구별하는 독립 순위 계산."""
    values = np.sort(np.asarray(list(values), dtype=float))
    result = {}
    for name, q in (('q10', .1), ('q25', .25), ('median', .5), ('q75', .75),
                    ('P90', .9), ('P95', .95), ('P99', .99), ('max', 1.)):
        if not len(values):
            value, status = None, 'NA_EMPTY'
        else:
            index = (len(values)-1)*q
            lo, hi = math.floor(index), math.ceil(index)
            value = values[lo] if lo == hi else values[lo] + (index-lo)*(values[hi]-values[lo]) if np.isfinite(values[hi]) else np.inf
            status = 'FINITE' if np.isfinite(value) else 'POSITIVE_INFINITY'
            value = float(value) if np.isfinite(value) else None
        result[name], result[name+'_status'] = value, status
    return result


def summary(records):
    """원 집계함수를 호출하지 않고 모든 공개 summary 필드를 다시 계산한다."""
    from scripts.paper.pose_metric_closure_v1.symmetry_aware_pose_metrics import AUC_MAX_FRACTION, AUC_INTEGRATION_POINTS
    rows = list(records)
    valid = [row for row in rows if row['available']]
    if rows:
        errors = np.asarray([row['ADDsym_normalized'] if row['available'] else np.inf for row in rows])
        limits = np.linspace(0., AUC_MAX_FRACTION, AUC_INTEGRATION_POINTS)
        accuracy = (errors[:, None] <= limits[None, :]).mean(0)
        auc = float(np.trapz(accuracy, limits)/AUC_MAX_FRACTION)
    else:
        auc = None
    out = dict(frames=len(rows), valid_pose=len(valid), failed_pose=len(rows)-len(valid),
        coverage=len(valid)/len(rows) if rows else None, status='OK' if rows else 'NA_EMPTY_POPULATION',
        conditional={k: distribution(row[k] for row in valid) for k in FIELDS},
        full_population={k: distribution(row[k] if row['available'] else np.inf for row in rows) for k in FIELDS},
        axis_mismatch_count=sum(row.get('axis_correct') is False for row in valid),
        axis_available_count=sum('axis_correct' in row for row in valid),
        ADDsym_AUC=auc, IoU3D=distribution(row['IoU3D'] for row in valid))
    for axis in ('x', 'z'):
        values = [row['camera_'+axis+'_signed_cm'] for row in valid if 'camera_'+axis+'_signed_cm' in row]
        out['camera_'+axis] = dict(available=len(values), absolute_cm=distribution(map(abs, values)),
            signed_bias_mean_cm=float(np.mean(values)) if values else None,
            signed_bias_median_cm=float(np.median(values)) if values else None)
    return out


def paired(before, after, ids):
    """프레임 차이의 중앙값과 각 집계 중앙값의 차이를 별도로 독립 계산한다."""
    common = [fid for fid in ids if before[fid]['available'] and after[fid]['available']]
    counts = Counter()
    deltas = {key: [after[fid][key]-before[fid][key] for fid in common] for key in FIELDS}
    def sign(value):
        return 'IMPROVE' if value < 0 else 'WORSEN' if value > 0 else 'TIE'
    for t, r in zip(deltas['translation_cm'], deltas['rotation_deg']):
        counts[f'T_{sign(t)}__R_{sign(r)}'] += 1
    old, new = [summary(data[fid] for fid in ids) for data in (before, after)]
    return dict(frames=len(ids), common_valid_frames=len(common),
        before_valid=old['valid_pose'], after_valid=new['valid_pose'],
        available_to_failed=sum(before[f]['available'] and not after[f]['available'] for f in ids),
        failed_to_available=sum(not before[f]['available'] and after[f]['available'] for f in ids),
        difference_of_conditional_medians={key: new['conditional'][key]['median']-old['conditional'][key]['median']
            if new['valid_pose'] and old['valid_pose'] else None for key in FIELDS},
        median_of_common_frame_differences={key: float(np.median(values)) if values else None for key, values in deltas.items()},
        common_frame_delta_distributions={key: distribution(values) for key, values in deltas.items()},
        paired_direction_counts={f'T_{t}__R_{r}': counts[f'T_{t}__R_{r}'] for t in ('IMPROVE', 'TIE', 'WORSEN') for r in ('IMPROVE', 'TIE', 'WORSEN')},
        common_valid_before=summary(before[f] for f in common), common_valid_after=summary(after[f] for f in common),
        axis_correct_to_wrong=sum(before[f].get('axis_correct') is True and after[f].get('axis_correct') is False for f in common),
        axis_wrong_to_correct=sum(before[f].get('axis_correct') is False and after[f].get('axis_correct') is True for f in common))


def validate_decision_reads(lock):
    assert lock['reference_or_metrics_read_before_decisions'] is False
    assert lock['read_guard_active'] and lock['candidate_order_swap_test']
    assert lock['final_poses_from_cached_candidates'] and lock['new_selector_fits'] == 0
    assert lock['read_paths']
    offending = [path for path in lock['read_paths'] if any(token in path for token in FORBIDDEN_READS)]
    assert not offending, offending
    return len(lock['read_paths'])


def selected_pose(record, selected):
    matches = [row['pose'] for row in record['hypotheses'] if row['name'] == selected]
    assert len(matches) <= 1
    return matches[0] if matches else record['current']


def public_privacy(value):
    """공개 JSON은 수치 요약·출처 해시만; 픽셀/점/자세 좌표 배열을 포함하지 않는다."""
    denied = {'keypoints', 'coordinates', 'R_physical', 'R_cf', 'centroid', 'K', 'xyz',
              'image_base64', 'image_data', 'predictions', 'candidate_metrics'}
    if isinstance(value, dict):
        assert not (set(value) & denied), set(value) & denied
        for child in value.values():
            public_privacy(child)
    elif isinstance(value, list):
        for child in value:
            public_privacy(child)
    elif isinstance(value, str):
        assert not value.startswith('data:image/')


def run():
    import torch
    from scripts.research.pallet_clean_to_pose_transfer_v1 import eval_student as E
    from scripts.research.pallet_clean_to_pose_transfer_v1.old_reference import scalar_pose_recheck
    from scripts.research.pallet_selector_recovery_v1 import models as SM, common as SC
    verifier = Verifier()
    result_path = C.DOC/'CONTROL_RESULTS.json'
    lock_path = C.DOC/'SAME_GEO_CONTROLS_LOCK.json'
    result, lock = [verifier.read(path) for path in (result_path, lock_path)]
    verifier.nested(result)
    verifier.nested(lock)
    assert lock['decision_guard_active'] and lock['new_control_selection_locked_before_scoring']
    assert not lock['new_controls_gt_or_metric_reads']
    assert result['no_missing_seed_fabrication'] and result['old_results_preserved']
    assert result['missing_models'] == ['RAW_CLEAR_S43', 'REF_CLEAR_S43']
    expected = {model+'_'+selector for model in MODELS for selector in ('D9', 'GEO')}
    assert set(result['available_models']) == expected
    for document in (result, lock):
        assert document['new_student_fits'] == document['new_selector_fits'] == document['optimizer_updates'] == document['GPU_seconds'] == 0
        public_privacy(document)
    association = verifier.read(C.RAW/'CONTROL_CASE_BINDINGS_PRIVATE.json')
    poses = verifier.read(C.RAW/'CONTROL_POSES_PRIVATE.json')
    metrics = verifier.read(C.RAW/'CONTROL_FRAME_METRICS_PRIVATE.json')
    assert set(association) == set(poses) == set(metrics) == expected
    verifier.nested(association)
    rows = verifier.read(C.ROOT/association['R0_D9']['metadata']['path'])
    groups = E.group_ids(rows)
    assert {k: len(groups[k]) for k in ('FULL128', 'CLEAN29', 'MOD21', 'SEV78', 'NATURAL99')} == dict(FULL128=128, CLEAN29=29, MOD21=21, SEV78=78, NATURAL99=99)
    assert set(result['groups']) == set(result['paired']) == set(groups)
    childlocks = {}
    for model in MODELS:
        if model in lock['children']:
            child_path = C.ROOT/lock['children'][model]['lock']['path']
        else:
            target, _, seed = model.split('_')
            pair = verifier.read(C.OLD.DOC/f'SELECTOR_PAIR_LOCK_{seed}.json')
            verifier.nested(pair)
            assert pair['scorer'] == lock['same_frozen_scorer']
            child_path = C.ROOT/pair['arms'][target]['lock']['path']
        child = verifier.read(child_path)
        verifier.nested(child)
        assert child['old_scorer'] == lock['same_frozen_scorer']
        childlocks[model] = child
        validate_decision_reads(child)
        public_privacy(child)
        if model in lock['children']:
            child_result = verifier.read(C.DOC/f'CONTROL_{model}_RESULTS.json')
            assert datetime.fromisoformat(child['created_at']) < datetime.fromisoformat(child_result['created_at'])
            assert datetime.fromisoformat(lock['created_at']) < datetime.fromisoformat(child_result['created_at'])
            assert child_path.stat().st_mtime_ns <= (C.DOC/f'CONTROL_{model}_RESULTS.json').stat().st_mtime_ns
            verifier.nested(child_result)
            public_privacy(child_result)
    scorer = torch.load(C.ROOT/lock['same_frozen_scorer']['path'], map_location='cpu', weights_only=False)
    assert scorer['d'] == 94 and scorer['variant'] == 'GEO_LINEAR'
    torch.set_num_threads(2)
    matches, score_checks, selection_changes = 0, 0, {}
    for model in MODELS:
        child = childlocks[model]
        fpath = next(row for row in child['files'] if row['path'].endswith('/FEATURES.json'))
        features = verifier.read(C.ROOT/fpath['path'])
        changes = 0
        for selector in ('D9', 'GEO'):
            arm = model+'_'+selector
            binding = association[arm]
            assert verifier.read(C.ROOT/binding['metadata']['path']) == rows
            candidate_path = C.ROOT/binding['candidates']['path']
            source_arm = binding['candidate_arm']
            candidates = verifier.read(candidate_path)[source_arm]
            cached_choices = verifier.read(candidate_path.parent/'ORACLE_METRICS_SELECTIONS_PRIVATE.json')['candidate_metrics'][source_arm]
            original_metrics = verifier.read(candidate_path.parent/'POSE_METRICS.json')[source_arm]
            assert set(poses[arm]) == set(metrics[arm]) == set(candidates) == set(groups['FULL128'])
            decisions = verifier.read(C.ROOT/binding['decisions']['path']) if selector == 'GEO' else None
            if decisions:
                assert set(decisions) == set(features) == set(groups['FULL128'])
            for fid in groups['FULL128']:
                record = candidates[fid]
                assert not record['reference_coordinates_read']
                selected = record['selected_name'] if decisions is None else decisions[fid]['selected']
                expected_pose = record['current'] if decisions is None else selected_pose(record, selected)
                close(poses[arm][fid], expected_pose)
                options = [row['metric'] for row in cached_choices[fid] if row['name'] == selected]
                assert len(options) <= 1
                expected_metric = options[0] if options else original_metrics[fid]
                close(metrics[arm][fid], expected_metric)
                if decisions:
                    d, f = decisions[fid], features[fid]
                    assert d['D9_selected'] == record['selected_name']
                    assert d['changed'] == (selected != record['selected_name'])
                    changes += d['changed']
                    if f['valid']:
                        scores = SM.scores(scorer, np.asarray(f['features'], np.float32)[None])[0]
                        np.testing.assert_allclose(scores, d['scores'], rtol=1e-7, atol=1e-7)
                        chosen = min(range(len(scores)), key=lambda i: (float(scores[i]), SC.HYP[i]))
                        assert selected == SC.HYP[chosen]
                        reverse = min(range(len(scores)), key=lambda i: (float(scores[::-1][i]), SC.HYP[::-1][i]))
                        assert chosen == 1-reverse
                        score_checks += 1
                    else:
                        assert selected == record['selected_name'] and d['fallback'] is not None
                matches += 1
        assert changes == child['counts']['changed']
        selection_changes[model] = changes
    # 모든 선택과 잠금의 검증을 끝낸 뒤에만, 사후 감사용 참조를 연다.
    from scripts.research.pallet_clean19_pose_mismatch_v1 import diagnose as D
    _, truth = D.Pose.metadata('REAL_DEV')
    for arm in expected:
        for fid in groups['FULL128']:
            scalar_pose_recheck(metrics[arm][fid], poses[arm][fid], truth[fid])
    summaries_checked = pairs_checked = loto_checked = 0
    for group, ids in groups.items():
        assert set(result['groups'][group]) == expected
        for arm in expected:
            close(summary(metrics[arm][fid] for fid in ids), result['groups'][group][arm])
            summaries_checked += 1
        for name, recorded in result['paired'][group].items():
            after, before = name.split('-minus-')
            assert after in expected and before in expected
            close(paired(metrics[before], metrics[after], ids), recorded)
            pairs_checked += 1
    for recording, pairs in result['leave_one_recording_out_NATURAL99'].items():
        ids = [row['id'] for row in rows if row['severity'] != 'CLEAN' and row['recording'] != recording]
        for name, recorded in pairs.items():
            after, before = name.split('-minus-')
            close(paired(metrics[before], metrics[after], ids), recorded)
            loto_checked += 1
    natural = result['groups']['NATURAL99']
    pareto = []
    for arm in sorted(a for a in expected if a.endswith('_GEO')):
        t, r = [natural[arm]['full_population'][key]['median'] for key in FIELDS[:2]]
        others = [(natural[a]['full_population']['translation_cm']['median'], natural[a]['full_population']['rotation_deg']['median']) for a in expected if a.endswith('_GEO') and a != arm]
        if not any(ot <= t and orot <= r and (ot < t or orot < r) for ot, orot in others):
            pareto.append(dict(card_id=arm, translation_cm=t, rotation_deg=r))
    close(pareto, result['pareto_GEO_by_T_R_median_only'])
    start = verifier.read(C.DOC/'START.json')
    verifier.nested(start['immutable_inputs'])
    ledger = verifier.read(C.DOC/'RESOURCE_LEDGER.json')
    verifier.nested(ledger['inherited_ledger'])
    old_ledger = verifier.read(C.ROOT/ledger['inherited_ledger']['path'])
    assert ledger['inherited_totals'] == old_ledger['totals']
    assert ledger['totals'] == ledger['inherited_totals'] and ledger['new_events'] == []
    assert ledger['totals']['student_fits'] == 6 and ledger['totals']['optimizer_updates'] == 1920
    assert ledger['totals']['selector_fits'] == 0
    changed = subprocess.check_output(['git', 'diff', '--name-only', 'HEAD', '--',
        str(C.OLD.DOC.relative_to(C.ROOT)), str(Path(E.__file__).parent.relative_to(C.ROOT))], cwd=C.ROOT, text=True).strip()
    assert not changed, ('기존 완료 namespace 변경', changed)
    output = dict(status='PASS', passed=True, created_at=C.now(),
        scope='고정 후보/선택/프레임 오차의 독립 CPU 감사; 새 평가 선택/fit/신경망 pose 추론 없음',
        models=8, selector_configurations=16, frames_per_configuration=128,
        selected_pose_and_metric_matches=matches, frozen_scorer_CPU_score_checks=score_checks,
        scalar_centroid_C2_rotation_yaw_recomputed=matches, selection_changes=selection_changes,
        summary_blocks_independently_recomputed=summaries_checked,
        paired_contrast_blocks_independently_recomputed=pairs_checked,
        leave_one_recording_out_blocks_independently_recomputed=loto_checked,
        unique_source_bindings_verified=len(verifier.seen), old_artifacts_unchanged=True,
        reference_access_guard_paths_checked=True, selection_lock_precedes_new_scoring=True,
        no_raw_RGB_or_coordinate_arrays_in_public_controls=True,
        missing_models_preserved=result['missing_models'], cumulative_cost_at_audit=ledger['totals'],
        new_student_fits=0, new_selector_fits=0, optimizer_updates=0, GPU_seconds=0,
        limits=['별도 OS 수준 파일접근 감사를 했다는 뜻은 아니다. 저장된 Python audit read-path 기록과 고정 코드/해시/시각을 검증했다.',
                'IoU3D 및 ADD 개별 프레임 원식은 다시 풀지 않았다. 선택 후보의 기존 전체 metric 행 일치와 AUC/IoU 집계를 검증했다.',
                '두 seed OCC만 존재한다. CLEAR seed43와 독립 TEST를 만들어낸 것이 아니다.',
                '자연99는 프레임 오차를 직접 pooling했다. 난도 중앙값 평균이 아니다.'],
        sources=[C.bind(p) for p in (result_path, lock_path, C.DOC/'START.json',
            C.RAW/'CONTROL_FRAME_METRICS_PRIVATE.json', C.RAW/'CONTROL_POSES_PRIVATE.json',
            C.RAW/'CONTROL_CASE_BINDINGS_PRIVATE.json', Path(__file__))])
    C.save(C.DOC/'CONTROL_AUDIT.json', output, True)
    lines = ['# 동일 GEO 무학습 대조 독립 감사', '', '**PASS**', '',
        f'- 8개 기존 모델 × D9/GEO = 16개 설정, 각 128장을 검사했다. 선택 후보 자세와 metric 행 {matches:,}개가 일치했다.',
        f'- 같은 frozen GEO의 CPU 점수·이름 tie-break를 {score_checks:,}프레임에서 다시 계산했다. 정답 오차를 선택에 사용하지 않았다.',
        f'- centroid 위치·C2 전체 회전·yaw {matches:,}개를 참조에서 독립 재계산했다.',
        f'- 전체 난도/촬영 기록 summary {summaries_checked}개, paired 비교 {pairs_checked}개, recording 제외 비교 {loto_checked}개를 별도 구현으로 재집계했다.',
        '- RAW/REF CLEAR seed43는 없는 것으로 유지했다. 기존 완료 namespace의 변경도 없었다.',
        '- 누적 비용은 학생 6회, 1,920 update, GPU 학습 417.781초, 새 선택기 학습 0회 그대로다.',
        '- 공개 대조 JSON에는 원본 RGB나 원 좌표 배열이 없다. 비공개 결과는 경로/해시만 연결했다.', '',
        '## 감사 범위의 한계', '', *['- '+v for v in output['limits']], '', '[상세 JSON](CONTROL_AUDIT.json)', '']
    C.save(C.DOC/'CONTROL_AUDIT.md', '\n'.join(lines), True)
    print('CONTROL_AUDIT_PASS', matches, score_checks, summaries_checked, pairs_checked, loto_checked, flush=True)
    return output


if __name__ == '__main__':
    run()
