"""Phase 3: 과거 고정 자세를 동일128장/현재 T·R 정의로 CPU 재집계한다.

신경망 추론, fit, optimizer, pose 재추정, 이전 파일 수정은 하지 않는다.
프레임 ID·개별 오차는 private 결과에만 보존한다.
"""
from __future__ import annotations

from collections import Counter
import hashlib
import json
from pathlib import Path

import numpy as np

from scripts.research.pallet_pose_objective_followup_v2 import metric_baseline as M

ROOT = Path(__file__).resolve().parents[3]
NAME = 'pallet_clean_to_pose_transfer_v1'
DOC = ROOT / '_docs/experiments' / NAME
RAW = ROOT / 'data/pallet/results' / NAME / 'old_reference'
OLD_DOC = ROOT / '_docs/experiments/pallet_clean19_structured_easyhard_v1'
OLD_RAW = ROOT / 'data/pallet/results/pallet_clean19_structured_easyhard_v1'
REC_DOC = ROOT / '_docs/experiments/pallet_recording_disjoint_transfer_v1'
REC_RAW = ROOT / 'data/pallet/results/pallet_recording_disjoint_transfer_v1'
MAIN_RAW = ROOT / 'data/pallet/results/pallet_pose_objective_followup_v2/metric_baseline'
FIELDS = ('translation_cm', 'rotation_deg', 'yaw_deg')


def read(path):
    return json.loads(Path(path).read_text())


def bind(path):
    path = Path(path).resolve()
    return dict(path=str(path.relative_to(ROOT)), sha256=hashlib.sha256(path.read_bytes()).hexdigest(), bytes=path.stat().st_size)


def save(path, value):
    path = Path(path).resolve()
    assert path.is_relative_to(DOC) or path.is_relative_to(RAW)
    text = value if isinstance(value, str) else json.dumps(M.clean(value), ensure_ascii=False, indent=2, allow_nan=False) + '\n'
    if path.exists():
        assert path.read_text() == text, f'고정 결과를 덮어쓰지 않음: {path}'
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(text)
    temporary.replace(path)


def verify(binding):
    actual = bind(ROOT / binding['path'])
    assert all(actual[k] == binding[k] for k in ('path', 'sha256', 'bytes'))


def summary(rows):
    rows = list(rows)
    valid = [r for r in rows if r['available']]
    return dict(frames=len(rows), valid_pose=len(valid), failed_pose=len(rows)-len(valid),
        coverage=len(valid)/len(rows) if rows else None,
        conditional={k:dict(median=float(np.median([r[k] for r in valid])),
            P90=float(np.quantile([r[k] for r in valid], .9))) if valid else dict(median=None, P90=None) for k in FIELDS},
        full_population={k:M.distribution([r[k] if r['available'] else float('inf') for r in rows]) for k in FIELDS},
        axis_correct=sum(r.get('axis_correct') is True for r in valid))


def pair(before, after, ids):
    counts = Counter()
    rows = []
    for fid in ids:
        a, b = before[fid], after[fid]
        if not (a['available'] and b['available']):
            continue
        delta = {k:b[k]-a[k] for k in FIELDS}
        sign = lambda x:'IMPROVE' if x < 0 else 'WORSEN' if x > 0 else 'TIE'
        counts[f'T_{sign(delta["translation_cm"])}__R_{sign(delta["rotation_deg"])}'] += 1
        rows.append(dict(id=fid, delta=delta))
    directions = {f'T_{t}__R_{r}':counts[f'T_{t}__R_{r}'] for t in ('IMPROVE','TIE','WORSEN') for r in ('IMPROVE','TIE','WORSEN')}
    old, new = [summary(data[i] for i in ids) for data in (before, after)]
    out = dict(frames=len(ids), common_valid=len(rows),
        both_improve=counts['T_IMPROVE__R_IMPROVE'],
        T_only=counts['T_IMPROVE__R_WORSEN'], R_only=counts['T_WORSEN__R_IMPROVE'],
        both_worsen=counts['T_WORSEN__R_WORSEN'],
        any_tie=sum(n for k,n in directions.items() if 'TIE' in k),
        direction_counts=directions,
        available_to_failed=sum(before[i]['available'] and not after[i]['available'] for i in ids),
        failed_to_available=sum(not before[i]['available'] and after[i]['available'] for i in ids),
        difference_of_medians={k:new['conditional'][k]['median']-old['conditional'][k]['median']
            if old['valid_pose'] and new['valid_pose'] else None for k in FIELDS},
        median_of_paired_differences={k:float(np.median([r['delta'][k] for r in rows])) if rows else None for k in FIELDS},
        delta_distributions={k:M.distribution(r['delta'][k] for r in rows) for k in FIELDS})
    assert sum(directions.values()) == len(rows)
    return out, rows


def scalar_pose_recheck(row, pose, truth):
    """기존 centroid·C2 full rotation·yaw 식만 독립 재계산; IoU/PnP를 재실행하지 않음."""
    assert row['available'] == pose['available']
    if not row['available']:
        return
    assert truth['order'] == 2
    rotation, yaw = [], []
    for theta in (0., np.pi):
        Q = np.array([[np.cos(theta),0,np.sin(theta)],[0,1,0],[-np.sin(theta),0,np.cos(theta)]])
        relative = (np.asarray(truth['R']) @ Q).T @ np.asarray(pose['R_physical'])
        rotation.append(float(np.degrees(np.arccos(np.clip((np.trace(relative)-1)/2, -1, 1)))))
        yaw.append(abs(float((np.degrees(np.arctan2(relative[0,2],relative[2,2]))+180)%360-180)))
    actual = dict(translation_cm=float(np.linalg.norm(np.asarray(pose['centroid'])-truth['t'])*100),
        rotation_deg=min(rotation), yaw_deg=min(yaw))
    for key, value in actual.items():
        np.testing.assert_allclose(value, row[key], rtol=1e-7, atol=1e-7)


def close_summary(actual, historical):
    assert actual['frames'] == historical['frames']
    assert actual['valid_pose'] == historical['available']
    for key in FIELDS:
        for stat in ('median', 'P90'):
            np.testing.assert_allclose(actual['conditional'][key][stat], historical[key][stat], rtol=1e-7, atol=1e-7)


def main():
    split_path = ROOT / '_docs/experiments/pallet_existing_data_transfer_v1/SPLIT_LOCK.json'
    split = read(split_path)
    records = split['heldout']
    ids = [r['id'] for r in records]
    assert len(ids) == len(set(ids)) == 128
    assert Counter(r['severity'] for r in records) == dict(CLEAN=29, MODERATE_OCCLUSION=21, SEVERE_OCCLUSION=78)
    groups = {'ALL128':ids, 'CLEAN29':[r['id'] for r in records if r['severity']=='CLEAN'],
        'MODERATE21':[r['id'] for r in records if r['severity']=='MODERATE_OCCLUSION'],
        'SEVERE78':[r['id'] for r in records if r['severity']=='SEVERE_OCCLUSION'],
        'MODERATE_PLUS_SEVERE99':[r['id'] for r in records if r['severity']!='CLEAN']}
    groups.update({rec:[r['id'] for r in records if r['recording_group']==rec] for rec in sorted({r['recording_group'] for r in records})})
    verify(read(OLD_DOC/'POSE_PREDICTIONS_LOCK.json'))
    rec_lock = read(REC_DOC/'POSE_PREDICTIONS_LOCK.json')
    for key in ('file','selector','solver','metadata'):
        verify(rec_lock[key])
    checkpoint_bindings = {}
    for arm in ('S0','S1','S2'):
        fit = read(OLD_DOC/f'FIT_PLASTIC_{arm}.json')
        assert fit['complete'] and fit['steps'] == 320
        checkpoint_bindings[arm] = fit['checkpoint']
        verify(fit['checkpoint'])
    old = read(OLD_RAW/'POSE_METRICS.json')
    old_pose = read(OLD_RAW/'POSE_PREDICTIONS.json')
    recorded = read(REC_RAW/'POSE_METRICS.json')
    main_rows = read(MAIN_RAW/'FRAME_METRICS_PRIVATE.json')['PLASTIC']
    main_members = read(MAIN_RAW/'POPULATION_LOCK_PRIVATE.json')['PLASTIC']
    assert ids == main_members['groups']['ALL']
    assert groups['MODERATE_PLUS_SEVERE99'] == main_members['groups'][M.PRIMARY]
    metrics = {a:{fid:old[a][fid] for fid in ids} for a in ('S0','S1','S2')}
    metrics.update({a:{fid:main_rows[a][fid] for fid in ids} for a in ('R0','OLD_REF')})
    for arm in ('R0','S0','S1'):
        for fid in ids:
            assert old[arm][fid]['available'] == recorded[arm][fid]['current']['available']
            for key in FIELDS:
                np.testing.assert_allclose(old[arm][fid][key], recorded[arm][fid]['current'][key], rtol=1e-7, atol=1e-7)
                np.testing.assert_allclose(metrics[arm][fid][key], old[arm][fid][key], rtol=1e-7, atol=1e-7)
    # 참조는 기존 예측 lock 검증 뒤 scoring 전용으로 연다. 새 fit/추론은 없다.
    from scripts.research.pallet_clean19_pose_mismatch_v1 import diagnose as D
    _, truth = D.Pose.metadata('REAL_DEV')
    for arm in ('S0','S1','S2'):
        for fid in ids:
            scalar_pose_recheck(metrics[arm][fid], old_pose[arm][fid], truth[fid])
    result = {g:{a:summary(mm[i] for i in ii) for a,mm in metrics.items()} for g,ii in groups.items()}
    recorded_result = read(REC_DOC/'RESULTS.json')['groups']
    for new, previous in [('ALL128','ALL'),('CLEAN29','CLEAN'),('MODERATE21','MODERATE'),('SEVERE78','SEVERE')]:
        for arm in ('R0','S0','S1'):
            close_summary(result[new][arm], recorded_result[previous][arm]['current'])
    historical = read(OLD_DOC/'RESULTS.json')['groups']['PLASTIC_SEVERE_OCCLUSION']
    full_eval = read(ROOT/'_docs/experiments/pallet_replay_clean19_v1/SPLIT.json')['evaluation']
    old_severe = [r['id'] for r in full_eval if r['object_type'].upper()=='PLASTIC' and r['severity']=='SEVERE_OCCLUSION']
    assert len(old_severe) == 81 and set(groups['SEVERE78']) < set(old_severe)
    for arm in ('S0','S1','S2'):
        close_summary(summary(old[arm][fid] for fid in old_severe), historical[arm]['sixD'])
    contrasts, private_pairs = {}, {}
    for group, subset in groups.items():
        contrasts[group], private_pairs[group] = {}, {}
        for after in ('S1','S2'):
            key = after+'-minus-S0'
            contrasts[group][key], private_pairs[group][key] = pair(metrics['S0'], metrics[after], subset)
    inputs = [split_path,OLD_RAW/'POSE_METRICS.json',OLD_RAW/'POSE_PREDICTIONS.json',OLD_DOC/'POSE_PREDICTIONS_LOCK.json',
        OLD_DOC/'RESULTS.json',REC_RAW/'POSE_METRICS.json',REC_DOC/'POSE_PREDICTIONS_LOCK.json',REC_DOC/'RESULTS.json',
        MAIN_RAW/'FRAME_METRICS_PRIVATE.json',MAIN_RAW/'POPULATION_LOCK_PRIVATE.json',
        ROOT/'_docs/experiments/pallet_pose_objective_followup_v2/METRIC_AND_SELECTION_LOCK.json',
        ROOT/'_docs/experiments/pallet_replay_clean19_v1/SPLIT.json',
        ROOT/'data/pallet/results/paper_pose_metric_closure_v1/GEOMETRY_RESOLVED_POSE_GT.json',
        ROOT/'data/pallet/results/paper_pose_metric_closure_v1/AXIS_REVIEW_MANIFEST.json',
        ROOT/'scripts/research/pallet_dim_conditioned_p_v1/pose.py',
        ROOT/'scripts/research/pallet_pose_objective_followup_v2/metric_baseline.py']
    inputs += [OLD_DOC/f'FIT_PLASTIC_{a}.json' for a in ('S0','S1','S2')]
    private_path = RAW/'FRAME_METRICS_AND_PAIRED_PRIVATE.json'
    save(private_path, dict(records=[{k:r[k] for k in ('id','severity','recording_group')} for r in records],
        groups=groups, metrics=metrics, paired=private_pairs, removed_historical_severe_ids=sorted(set(old_severe)-set(groups['SEVERE78']))))
    output = dict(status='CPU_FROZEN_CACHE_REAGGREGATION_COMPLETE', new_fits=0, optimizer_updates=0,
        new_neural_inference=0, new_pose_solves=0, GPU_seconds=0,
        population=dict(frames=128, clean=29, moderate=21, severe=78, primary=99,
            recordings=len({r['recording_group'] for r in records}), role='재사용 recording-disjoint DEV; 독립 TEST 아님'),
        groups=result, contrasts=contrasts,
        old_severe81_reference={a:historical[a]['sixD'] for a in ('S0','S1','S2')},
        checks=dict(historical_S0_S1_R0_frame_scalar_parity=384, historical_group_parity=12,
            old_S0_S1_S2_current_T_R_yaw_recomputed=384, same_R0_old_and_current_128=True,
            same_population_order_as_current=True, old_severe81_reproduced=True, checkpoint_hashes_verified=True),
        contracts=dict(selector='기존 deployable D9; GEO_LINEAR/oracle/재선택 없음',
            solver='corner0..7 SQPnP/LM; selector는 기존9점 계약 유지',
            translation='100 × norm(predicted centroid - geometry-derived reference centroid), OpenCV camera cm',
            rotation='C2(I,Ry180) 중 최소 full rotation geodesic degree; 90도 W/D swap 동치 아님',
            yaw='동일 C2 아래 기존 atan2(relative[0,2], relative[2,2]) 절대 wrap 최솟값',
            reference='기존 annotation + known dimensions로 재구성; 독립 측정 physical6D 아님',
            aggregation='실제 프레임 오차를 합쳐 median/P90; 난도별 median을 평균하지 않음',
            paired='after-before. T-only/R-only는 반대 지표가 엄밀히 악화하는 경우. 한 지표 동률은 별도9분류.',
            failure='조건부 valid-pose median과 전체 분모 병기; invalid는 full-population에서 +infinity, JSON null+status.',
            baseline='현재 R0/OLD_REF는 동일 population 참고값; 과거 S0/S1/S2와 matched training causal pair 아님'),
        inputs=[bind(p) for p in inputs], checkpoint_bindings=checkpoint_bindings,
        private_artifact=bind(private_path), implementation=bind(Path(__file__)))
    save(DOC/'OLD_CLEAN19_POSE_REFERENCE.json', output)
    save(DOC/'OLD_CLEAN19_POSE_REFERENCE.md', report(output))
    print(json.dumps({g:{a:{k:v['conditional'][k]['median'] for k in FIELDS[:2]} for a,v in result[g].items()}
        for g in ('CLEAN29','MODERATE21','SEVERE78','MODERATE_PLUS_SEVERE99')}, indent=2))


def report(result):
    lines = ['# 과거 Clean19 S0/S1/S2: 동일128장 T/R 고정 재검산', '',
        'Phase 3 완료: 새 학습0, 신경망 추론0, PnP 재계산0. 기존 frozen pose와 per-frame cache만 CPU 재집계했다.', '',
        '## 비교 계약', '',
        '- 현재 main과 같은 Plastic128: Clean29 / Moderate21 / Severe78, 자연 가림 주집합99.',
        '- 모든 팔에 기존 D9 선택을 유지했다. GEO_LINEAR·oracle·GT 기반 후보 교체는 없다.',
        '- centroid translation(cm), C2 대칭을 적용한 full rotation(°), yaw의 기존 정의를 보존했다. 기하학적으로 재구성한6D 참조이며 독립 물리 측정 정답은 아니다.',
        '- S0/S1/S2의384개 pose에서 현재 수식으로 T/R/yaw를 다시 계산해 cache와 확인했다. R0의 과거/현재128개 개별 오차도 일치한다.',
        '- 과거 Severe81의14.93/8.01 등의 수치는81장 결과다. 이번 Severe78과 분모가 다르므로 직접 하나의 개선율로 섞지 않는다.',
        '- 자연99의 median은99개 원오차를 합쳐 계산한다. 난도별 median의 평균이 아니다. 따라서 R0 Severe78의 Rmedian63.30°와 자연99의6.04°가 동시에 성립한다.',
        '- 모든 집합은 재사용 DEV다. S0/S1/S2 내부 비교와 현재 R0/OLD_REF의 참고 비교를 구분한다.', '',
        '## 동일 population 결과', '',
        '| 집합 | 모델 | valid/전체 | T median/P90 cm | full R median/P90 ° | yaw median/P90 ° |',
        '|---|---|---:|---:|---:|---:|']
    for group in ('CLEAN29','MODERATE21','SEVERE78','MODERATE_PLUS_SEVERE99','ALL128'):
        for arm in ('R0','OLD_REF','S0','S1','S2'):
            row=result['groups'][group][arm]
            vals=[f'{row["conditional"][k]["median"]:.4f} / {row["conditional"][k]["P90"]:.4f}' for k in FIELDS]
            lines.append(f'| {group} | {arm} | {row["valid_pose"]}/{row["frames"]} | '+ ' | '.join(vals)+' |')
    lines += ['', '## 같은 프레임의 변화', '',
        'after−before가 음수면 개선이다. T-only/R-only는 다른 지표가 악화한 경우이며, 정확한 동률은 별도로 분리했다. 통계적/실무적 유의성 문턱을 뜻하지 않는다.', '',
        '| 집합 | 비교 | 둘 다 개선 | T만 개선 | R만 개선 | 둘 다 악화 | 동률 포함 | ΔT median | ΔR median |',
        '|---|---|---:|---:|---:|---:|---:|---:|---:|']
    for group, pairs in result['contrasts'].items():
        for comparison,row in pairs.items():
            lines.append(f'| {group} | {comparison} | {row["both_improve"]} | {row["T_only"]} | {row["R_only"]} | {row["both_worsen"]} | {row["any_tie"]} | {row["difference_of_medians"]["translation_cm"]:+.4f} | {row["difference_of_medians"]["rotation_deg"]:+.4f} |')
    lines += ['', '위 Δmedian은 집계 median의 차이다. 개별 프레임 차이의 median·분포·전체9분류·recording별 원집계는 JSON에 별도로 보존했다.', '',
        '## Recording별 결과', '', '| Recording | 모델 | valid/전체 | T median/P90 cm | full R median/P90 ° | yaw median/P90 ° |',
        '|---|---|---:|---:|---:|---:|']
    for group, arms in result['groups'].items():
        if not group.startswith('REC_'):continue
        for arm in ('R0','OLD_REF','S0','S1','S2'):
            row=arms[arm]
            vals=[f'{row["conditional"][k]["median"]:.4f} / {row["conditional"][k]["P90"]:.4f}' for k in FIELDS]
            lines.append(f'| {group} | {arm} | {row["valid_pose"]}/{row["frames"]} | '+ ' | '.join(vals)+' |')
    lines += ['', '## 해석 한계', '',
        '과거 S1/S2가 S0보다 개선된 것은 해당 teacher·Clean10 Plastic 이미지·감독·학습범위·LR 및 가림 계약에서의 양성 신호다. 현재 Replay9/38 +217 main으로 자동 일반화하거나, 계약 차이 중 특정 하나가 원인이라고 결론내릴 수 없다. 이 분석은 다음 clean-only 짝지은 실험의 근거이며 새 성능 향상을 만든 실험이 아니다.', '',
        '[집계·9분류·입력 해시 JSON](OLD_CLEAN19_POSE_REFERENCE.json)', '',
        '재현: `MPLCONFIGDIR=/tmp/pallet-clean-transfer-mpl OMP_NUM_THREADS=2 python -m scripts.research.pallet_clean_to_pose_transfer_v1.old_reference`', '']
    return '\n'.join(lines)


if __name__ == '__main__':
    main()
