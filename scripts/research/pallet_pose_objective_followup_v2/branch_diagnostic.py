"""Post-lock CPU branch counterfactuals; never changes a deployed prediction.

Reference coordinates are opened only in score(), after candidate freeze().
This is an order-dependent diagnostic decomposition, not causal identification.
"""
from collections import Counter
from pathlib import Path
import time

import numpy as np

from . import common as C
from . import eval_student as E
from . import metric_baseline as M
from scripts.research.pallet_oracle_mechanism_followup_v1 import pose_oracle as O

CASES = [('C_EXPOSURE', 'PLASTIC'), ('WOOD_APPLICABILITY', 'WOOD')]
ARMS = ('OLD_REF', 'NEW_RAW', 'NEW_REF')


def decompose(base, held, final):
    within = held - base
    switch = final - held
    total = final - base
    assert np.isclose(within + switch, total, atol=1e-10, rtol=1e-10)
    return dict(within_baseline_branch=within, branch_switch=switch, total=total)


def freeze():
    lock_path = C.RAW / 'branch_diagnostic/CANDIDATES_LOCK.json'
    if lock_path.exists():
        for b in C.read(lock_path)['files'] + C.read(lock_path)['sources']:
            C.verify(b)
        return lock_path
    start = time.perf_counter()
    files, sources, parity = [], [], 0
    for cycle, material in CASES:
        p = E.paths(cycle, material, 42)
        E.inference_binding_checks(p)
        rows, predictions, poses = [C.read(p[k]) for k in ('metadata', 'predictions', 'poses')]
        output = {a: {} for a in ARMS}
        for row in rows:
            fid = row['id']
            for arm in ARMS:
                record = O.candidate_record(predictions[arm][fid], row)
                O.D.close(record['current'], poses[arm][fid])
                output[arm][fid] = record
                parity += 1
        path = C.RAW / f'branch_diagnostic/{material}_CANDIDATES.json'
        C.save(path, output, freeze=True)
        files.append(C.bind(path))
        sources.extend(C.bind(p[k]) for k in ('metadata', 'predictions', 'poses', 'lock'))
    sources.extend(C.bind(Path(m.__file__)) for m in (O, O.D, O.D.Pose))
    sources.append(C.bind(Path(__file__)))
    C.save(lock_path, dict(files=files, sources=sources, created_at=C.now(),
        exact_existing_selected_pose_parity=parity,
        no_reference_coordinates_read=True, no_new_candidates_or_selector=True,
        wall_seconds=time.perf_counter()-start), freeze=True)
    return lock_path


def score(lock_path):
    destination = C.DOC / 'BRANCH_DIAGNOSTIC.json'
    if destination.exists():
        C.verify(C.read(destination)['candidate_lock'])
        return
    start, cpu = time.perf_counter(), time.process_time()
    for b in C.read(lock_path)['files'] + C.read(lock_path)['sources']:
        C.verify(b)
    _, truth = O.D.Pose.metadata('REAL_DEV')
    public, private = {}, {}
    for cycle, material in CASES:
        p = E.paths(cycle, material, 42)
        candidates = C.read(C.RAW / f'branch_diagnostic/{material}_CANDIDATES.json')
        two = C.read(p['raw'] / f'FRAME_METRICS_{p["tag"]}.json')
        rows = C.read(p['metadata'])
        groups = M.group_ids(rows, material)
        metrics = {a: {} for a in ARMS}
        options = {a: {} for a in ARMS}
        for arm in ARMS:
            for fid, record in candidates[arm].items():
                metrics[arm][fid] = O.D.metric(fid, record['current'], truth[fid])
                options[arm][fid] = [dict(name=h['name'], metric=O.D.metric(fid, h['pose'], truth[fid]))
                                     for h in record['hypotheses']]
        material_result, private[material] = {}, {}
        for group, ids in groups.items():
            comparisons = {}
            for before in ('OLD_REF', 'NEW_RAW'):
                counts = Counter()
                deltas = {k: [] for k in ('translation_cm', 'rotation_deg')}
                per_frame = []
                for fid in ids:
                    a, b = candidates[before][fid], candidates['NEW_REF'][fid]
                    switched = a['selected_name'] != b['selected_name']
                    counts['branch_switch' if switched else 'same_branch'] += 1
                    at, bt = two[before][fid], two['NEW_REF'][fid]
                    counts['detection_or_match_transition'] += int((at['detected'], at['matched']) != (bt['detected'], bt['matched']))
                    held = next((h['metric'] for h in options['NEW_REF'][fid] if h['name'] == a['selected_name']), None)
                    am, bm = metrics[before][fid], metrics['NEW_REF'][fid]
                    if not all(r and r['available'] for r in (am, bm, held)):
                        counts['unavailable_counterfactual'] += 1
                        continue
                    detail = {k: decompose(am[k], held[k], bm[k]) for k in deltas}
                    for k in deltas:
                        deltas[k].append(detail[k])
                        if not switched:
                            assert abs(detail[k]['branch_switch']) < 1e-7
                    counts['counterfactual_valid'] += 1
                    if at['evaluable'] and bt['evaluable']:
                        gain = sum(e <= 10 for e in bt['errors']) > sum(e <= 10 for e in at['errors'])
                        both_bad = bm['translation_cm'] > am['translation_cm'] and bm['rotation_deg'] > am['rotation_deg']
                        counts['frame_PCK10_count_gain'] += int(gain)
                        counts['frame_PCK10_count_gain_both_pose_worse'] += int(gain and both_bad)
                    per_frame.append(dict(id=fid, switched=switched, decomposition=detail))
                summary = {k: {part: dict(mean=float(np.mean([v[part] for v in values])),
                                            median=float(np.median([v[part] for v in values]))) if values else dict(mean=None, median=None)
                               for part in ('within_baseline_branch', 'branch_switch', 'total')}
                           for k, values in deltas.items()}
                for k in deltas:
                    if deltas[k]:
                        assert np.isclose(summary[k]['within_baseline_branch']['mean'] + summary[k]['branch_switch']['mean'], summary[k]['total']['mean'])
                comparisons['NEW_REF-minus-'+before] = dict(frames=len(ids), counts=dict(counts), delta_summary=summary)
                private[material][group+'__'+before] = per_frame
            oracles = {arm: M.oracle_for(ids, options[arm], metrics[arm])[0] for arm in ARMS}
            material_result[group] = dict(comparisons=comparisons, fixed_candidates_T_R_oracle=oracles)
        public[material] = material_result
    private_path = C.RAW / 'branch_diagnostic/FRAME_DECOMPOSITION_PRIVATE.json'
    C.save(private_path, private, freeze=True)
    C.save(destination, dict(created_at=C.now(), materials=public, candidate_lock=C.bind(lock_path),
        private_rows=C.bind(private_path), reference_scoring_only=True,
        meanings=dict(within_baseline_branch='new keypoints solved using baseline selected W/D branch minus baseline selected pose error',
                      branch_switch='new selected pose minus same-new-keypoints baseline-branch counterfactual error',
                      total='new selected pose error minus baseline selected pose error'),
        limitations=['Order-dependent frozen counterfactual, not experimental causal identification',
                     'Means telescope; medians and differences of aggregate medians do not add',
                     'Candidate metrics/oracles use legacy geometry reference, not independent physical pose',
                     'No oracle choices, GT scores, or held-branch poses used by training/deployment',
                     'No evaluation detector/match transition does not prove target assignment during training is correct'],
        new_fits=0, optimizer_updates=0, GPU_seconds=0,
        wall_seconds=time.perf_counter()-start, CPU_seconds=time.process_time()-cpu), freeze=True)
    lines = ['# 新 학생의 2D–6D 불일치: 고정 D9 branch 분해'.replace('新', '새'), '',
             '새 가설/학습/선택기를 추가하지 않은 사후 CPU 진단이다. 같은 새 키포인트에서 기존 모델이 선택했던 W/D branch를 고정한 pose를 중간값으로 둔다. frame별 변화는 branch 내부 변화와 branch 전환 변화로 나뉘지만, **중앙값을 더해 전체 중앙값 변화라고 할 수 없다.** 아래는 합이 보존되는 평균 분해다. 원고 주 지표 중앙값을 대체하지 않는다.', '',
             '| 집합 / 비교 | branch 전환 / N | 평균 ΔT 내부 / 전환 / 전체 (cm) | 평균 ΔR 내부 / 전환 / 전체 (deg) | PCK10 증가인데 T·R 모두 악화한 frame |',
             '|---|---:|---:|---:|---:|']
    for material, group in [('PLASTIC', M.PRIMARY), ('WOOD', 'ALL')]:
        for contrast, value in public[material][group]['comparisons'].items():
            ds = value['delta_summary']
            formatted = [' / '.join(f"{ds[k][part]['mean']:+.6f}" for part in ('within_baseline_branch', 'branch_switch', 'total')) for k in ds]
            counts = value['counts']
            lines.append(f"| {material}/{contrast} | {counts.get('branch_switch', 0)}/{value['frames']} | {' | '.join(formatted)} | {counts.get('frame_PCK10_count_gain_both_pose_worse', 0)}/{counts.get('frame_PCK10_count_gain', 0)} |")
    lines += ['', '새 예측의 T-optimal/R-optimal 고정 후보 oracle과 후보 하나가 두 축을 동시에 개선할 수 있는 frame 수는 JSON에 각각 보존한다. 두 oracle의 최솟값은 하나의 pose가 아니다. 검출 박스·점수·선택 인덱스 parity는 기존 예측 lock에서 확인했으며, 검출/매칭 전이도 별도로 센다. 이는 학습 target assignment가 항상 올바르다는 증거는 아니다.', '',
              '참조는 기존 geometry-derived pose다. reference/물리 축 오류와 실제 키포인트 오류의 독립 분리는 아직 미확정이다. 모든 후보와 oracle 선택은 비공개 scoring 산출물이며 학습·배포에 반영하지 않았다.', '']
    C.save(C.DOC / 'BRANCH_DIAGNOSTIC.md', '\n'.join(lines), freeze=True)


if __name__ == '__main__':
    score(freeze())
    print('BRANCH_DIAGNOSTIC_COMPLETE', flush=True)
