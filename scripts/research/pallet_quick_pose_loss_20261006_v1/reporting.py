"""Read-only saved-row quick loss comparison; writes only this new namespace.

No model forwards, candidate generation, pose solves, manuscript I/O or changes
to inherited reporting globals. All displayed experiments must have completed.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from .common import ROOT, DOC, HARD_DOC, OLD_DOC, read, sha
from scripts.research.pallet_pose_target_6d_20261006_v1 import reporting as R

NEW = ('SOFT6D', 'EXPECT6D')
CONTROLS = ('N3', 'HARD6D_GEO', 'SOFT2D_GEO')
ALIASES = dict(RAW='RAW', N3='N3_seed1', PoseFix='PoseFix_seed1',
               SOFT2D_GEO='FIT_GEO_J_seed1', HARD6D_GEO='POSE_TARGET_GEO_J_seed1',
               GEO_ORACLE='GEO_6D_ORACLE')
TOL = dict(R.TOLERANCES)
PIXEL_TOL = 1e-9
RATE_TOL = 1e-9


def binding(path):
    path = Path(path)
    return dict(path=str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path),
                sha256=sha(path), bytes=path.stat().st_size)


def write_json(path, value):
    path = Path(path)
    if not path.resolve().is_relative_to(DOC.resolve()):
        raise ValueError('quick reporting writes must remain in its new namespace')
    path.parent.mkdir(parents=True, exist_ok=True)
    pending = path.with_suffix(path.suffix + '.pending')
    pending.write_text(json.dumps(R.clean(value), ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    pending.replace(path)


def write_text(path, value):
    path = Path(path)
    if not path.resolve().is_relative_to(DOC.resolve()):
        raise ValueError('quick reporting writes must remain in its new namespace')
    pending = path.with_suffix(path.suffix + '.pending')
    pending.write_text(value)
    pending.replace(path)


def load_split(split, training):
    """Reaggregate exactly seed1 controls and verify new final-checkpoint rows."""
    ids, inherited, oracle, inputs = R.load_methods(split, [1])
    methods = {alias: inherited[name] for alias, name in ALIASES.items()}
    baseline = read(OLD_DOC / f'results/A_{split}_BASELINES.json')
    if split == 'SYNTH_HELDOUT':
        bank_sha = next(x['sha256'] for x in read(OLD_DOC / 'A_manifest.json')['cache_files']
                        if x['name'] == 'source_banks.npy')
    else:
        bank_sha = next(x['sha256'] for x in read(OLD_DOC / 'results/A_SAMPLING_MASK_RECEIPT.json')['external_mask_files']
                        if Path(x['path']).name == 'REAL_DEV_generated_banks.npz')
    for method in NEW:
        path = DOC / f'results/{split}_{method}_seed1.jsonl'
        execution = DOC / f'results/{split}_{method}_seed1_EXECUTION.json'
        result = read(execution)
        assert result['status'] == 'DONE' and result['complete'], 'BLOCKED_DATA: evaluation incomplete'
        assert result['full_denominator'] == len(ids) and result['GT_inference_access'] is False
        assert result['readout'] in ('unchanged J', 'hardJ', 'hard J'), 'changed inference readout'
        assert result['rowfile_sha256'] == sha(path), 'prediction row file changed'
        assert result['code_sha256'] == sha(Path(__file__).with_name('evaluation.py'))
        assert result['protocol_sha256'] == sha(DOC / 'PROTOCOL.json')
        assert result['setup_sha256'] == sha(DOC / 'LOSS_SETUP.json')
        assert result['checkpoint_sha256'] == training['methods'][method]['checkpoint_sha256']
        assert result['target_sha256'] == baseline['target_sha256'], 'evaluation target changed'
        assert result['bank_binding'] == baseline['bank_binding'], 'native candidate bank changed'
        assert result['bank_artifact']['sha256'] == bank_sha, 'native candidate bytes differ'
        rows = R.ordered([json.loads(line) for line in path.read_text().splitlines() if line.strip()], ids)
        for row, bound, raw in zip(rows, oracle, methods['RAW']):
            assert row['session'] == raw['session']
            assert type(row['selected_index']) is int and 0 <= row['selected_index'] < bound['actions']
            assert row['action_count'] == bound['actions']
            assert row['oracle_selected_index'] == bound['index']
            assert row['F_attempt'] and row['F_complete'], 'quietly excluded incomplete F attempt'
            assert row['F_available'] == row['pose']['available']
            assert row['corner']['evaluable'] == raw['corner']['evaluable']
            if row['corner']['evaluable']:
                assert row['corner']['canonical_valid'] == raw['corner']['canonical_valid']
            if bound['oracle']['available']:
                assert abs(row['oracle_ADDsym_m'] - bound['oracle']['ADDsym_m']) <= TOL['ADDsym_m']
        methods[method] = rows
        inputs.extend([binding(path), binding(execution)])
    methods = {name: methods[name] for name in ('RAW', 'N3', 'PoseFix', 'SOFT2D_GEO', 'HARD6D_GEO', *NEW, 'GEO_ORACLE')}
    return ids, methods, inputs


def flag_nonincrease(new, base, tolerance=0):
    return None if new is None or base is None else bool(new <= base + tolerance)


def preservation_flags(new_summary, base_summary, raw_new_damage, raw_base_damage):
    """C checks use full population denominators; RAW damage has canonical GT IDs."""
    n, b = new_summary['corner'], base_summary['corner']
    flags = dict(
        pose_failures_nonincrease=flag_nonincrease(new_summary['pose']['failures'], base_summary['pose']['failures']),
        RAW_good5_to_bad10_nonincrease=flag_nonincrease(raw_new_damage['good5_to_bad10'], raw_base_damage['good5_to_bad10']),
        gross20_nonincrease=flag_nonincrease(n['gross20'], b['gross20'], RATE_TOL),
        full_PCK10_non_decrease=None if n['PCK']['10'] is None or b['PCK']['10'] is None else bool(n['PCK']['10'] >= b['PCK']['10'] - RATE_TOL),
        matched_2D_median_nonincrease=flag_nonincrease(n['matched_pooled_corner8_median_px'], b['matched_pooled_corner8_median_px'], PIXEL_TOL),
        matched_2D_P90_nonincrease=flag_nonincrease(n['matched_pooled_corner8_P90_px'], b['matched_pooled_corner8_P90_px'], PIXEL_TOL))
    return dict(flags=flags, all_preserved=all(v is True for v in flags.values()),
                missing=[k for k, v in flags.items() if v is None],
                violated=[k for k, v in flags.items() if v is False],
                RAW_new_damage=raw_new_damage, RAW_base_damage=raw_base_damage,
                scope='fixed RAW-to-method canonical good<5px to bad>10px counts; full-reference PCK/gross20; matched conditional pooled 2D quantiles; pose P90 is a separate B tail flag')


def axes(comparison, new_summary, base_summary, primary):
    result = dict(primary_bootstrap=primary, A={}, B={})
    for key in R.METRICS:
        stat = comparison['statistics'][key][primary]
        ci, value = stat['CI95'], stat['mean_paired_difference']
        result['A'][key] = dict(paired_mean_delta=value, tolerance=TOL[key],
            improving_direction=value is not None and value < -TOL[key],
            CI95=ci, gain_signal=value is not None and value < -TOL[key] and ci is not None and ci[1] < 0,
            interval_includes_zero=ci is not None and ci[0] <= 0 <= ci[1])
        ns, bs = new_summary['pose'][key], base_summary['pose'][key]
        result['B'][key] = {}
        for statistic in ('mean', 'median', 'P90'):
            a, b = ns[statistic], bs[statistic]
            delta = None if a is None or b is None else a-b
            result['B'][key][statistic] = dict(delta=delta,
                nonworse=delta is not None and delta <= TOL[key],
                strictly_lower=delta is not None and delta < -TOL[key])
    result['B']['both_T_R_medians_strictly_lower'] = all(
        result['B'][k]['median']['strictly_lower'] for k in ('translation_cm', 'rotation_deg'))
    result['B']['pose_P90_tail_preservation'] = {
        k: result['B'][k]['P90']['nonworse'] for k in R.METRICS}
    return result


def decision_label(comparisons, preservation):
    """Fixed N3-first labels. Never turns an OLD-only gain into a N3 win.

    comparisons[split][control] is axes(); preservation has the same keys.
    REAL session-primary is the decision interval; SYNTH frame-primary supplies
    its fixed opposing-direction control. No additional seed or fit is opened.
    """
    real = comparisons['REAL_DEV']['N3']
    synth = comparisons['SYNTH_HELDOUT']['N3']
    add_signal = real['A']['ADDsym_m']['gain_signal']
    tr = all(real['B'][k]['median']['nonworse'] for k in ('translation_cm', 'rotation_deg'))
    syn_nonworse = synth['A']['ADDsym_m']['paired_mean_delta'] is not None and synth['A']['ADDsym_m']['paired_mean_delta'] <= TOL['ADDsym_m']
    failure = all(preservation[s]['N3']['flags']['pose_failures_nonincrease'] is True for s in ('SYNTH_HELDOUT', 'REAL_DEV'))
    c_preserved = all(preservation[s]['N3']['all_preserved'] for s in ('SYNTH_HELDOUT', 'REAL_DEV'))
    conditions = dict(REAL_ADD_session_gain_signal=add_signal, REAL_T_R_medians_nonworse=tr,
                      SYNTH_ADD_frame_mean_nonworse=syn_nonworse, pose_failures_both_splits_nonincrease=failure,
                      C_all_both_splits_preserved=c_preserved)
    six_d = add_signal and tr and syn_nonworse and failure
    old_gains = {base: {split: comparisons[split][base]['A']['ADDsym_m']['gain_signal']
                        for split in ('SYNTH_HELDOUT', 'REAL_DEV')}
                 for base in ('HARD6D_GEO', 'SOFT2D_GEO')}
    if six_d:
        label = 'N3_BEAT_SIGNAL' if c_preserved else 'N3_BEAT_WITH_TRADEOFF'
    elif add_signal and (not tr or not syn_nonworse):
        label = 'POSE_TRADEOFF_SIGNAL'
    elif any(any(x.values()) for x in old_gains.values()):
        label = 'OLD_ONLY_GAIN'
    else:
        label = 'NO_CLEAR_GAIN'
    return dict(label=label, N3_6D_conditions=conditions, OLD_ADD_gain_signals=old_gains,
        both_REAL_T_R_medians_strictly_lower=real['B']['both_T_R_medians_strictly_lower'],
        training_seeds=[1], extra_fits_authorized=False,
        scope='fixed seed1 exploratory repeat-use DEV evidence; no independent confirmation, loss-family impossibility claim, or decision based only on an OLD improvement')


def oracle_analysis(raw, new, oracle):
    inherited = R.recovery(raw, new, new, oracle)
    output = dict(full_frames=inherited['full_frames'], common_success=inherited['common_success'],
        excluded_missing_or_F_failure=inherited['excluded_missing_or_F_failure'],
        ratio_eligible_frames=inherited['ratio_eligible_frames'],
        excluded_headroom_le_tolerance=inherited['excluded_headroom_le_tolerance'], tolerance_m=1e-7,
        distributions={name.replace('new_', ''): dist for name, dist in inherited['distributions'].items() if not name.startswith('old_')},
        ratio=inherited['ratios']['new'], exact_oracle_action=inherited['exact_oracle_action']['new'],
        rows=[{k.replace('new_', ''): v for k, v in row.items() if not k.startswith('old_')} for row in inherited['rows']],
        scope='same native GEO bank and hardJ final F; rho=(RAW ADD-new ADD)/(RAW ADD-oracle ADD); exclude headroom<=1e-7m; preserve negative and >1 values')
    if output['ratio']['oracle_bound_violations_beyond_tolerance']:
        raise AssertionError('BLOCKED_INTEGRITY: new native hardJ cost beats its exact bank oracle beyond numeric tolerance')
    return output


def wd_analysis(raw, new, oracle):
    groups = {}
    for state in ('NO_SWITCH', 'SWITCH', 'UNAVAILABLE'):
        pairs = [(a, n, o) for a, n, o in zip(raw, new, oracle) if R.face_state(a, o) == state]
        eligible = [(a, n, o) for a, n, o in pairs if n['pose']['available'] and o['pose']['available']]
        recovered = sum(n.get('final_hypothesis') == o.get('final_hypothesis') for _, n, o in eligible)
        groups[state] = dict(frames=len(pairs), common_F_available=len(eligible),
            final_hypothesis_recovered=recovered, final_hypothesis_not_recovered=len(eligible)-recovered,
            final_hypothesis_recovery_rate=recovered/len(eligible) if eligible else None,
            exact_oracle_action=sum(o['pose']['available'] and n['selected_index'] == o['selected_index'] for _, n, o in pairs),
            exact_oracle_action_denominator=sum(o['pose']['available'] for _, _, o in pairs),
            NEW_switches_from_RAW=sum(R.face_state(a, n) == 'SWITCH' for a, n, _ in pairs),
            ids=[a['id'] for a, _, _ in pairs])
    return dict(groups=groups, rows=[dict(id=a['id'], RAW=a.get('final_hypothesis'), NEW=n.get('final_hypothesis'),
        ORACLE=o.get('final_hypothesis'), RAW_to_oracle_group=R.face_state(a, o)) for a, n, o in zip(raw, new, oracle)],
        scope='RAW-to-oracle actual final F W/D hypothesis switch; recovery compares actual final hypothesis, never bank generating-hypothesis label')


def analyze_split(split, training):
    ids, methods, inputs = load_split(split, training)
    raw = methods['RAW']
    group_name = 'scenario' if split == 'SYNTH_HELDOUT' else 'session'
    bootstraps = dict(frame=R.SharedBootstrap(ids))
    bootstraps[group_name] = R.SharedBootstrap(ids, [r['session'] for r in raw])
    primary = 'frame' if split == 'SYNTH_HELDOUT' else 'session'
    summaries = {name: R.summarize(rows) for name, rows in methods.items()}
    for summary in summaries.values():
        assert summary['corner']['corners'] == summaries['RAW']['corner']['corners'], 'full-reference corner denominator changed'
    damages = {name: R.M.damage([r['corner'] for r in raw], [r['corner'] for r in rows])
               for name, rows in methods.items()}
    comparisons, analysis_axes, preserves, recovery, wd = {}, {}, {}, {}, {}
    for method in NEW:
        comparisons[method], analysis_axes[method], preserves[method] = {}, {}, {}
        for base in CONTROLS:
            comparison = R.compare(methods[method], methods[base], bootstraps)
            comparisons[method][base] = comparison
            analysis_axes[method][base] = axes(comparison, summaries[method], summaries[base], primary)
            preserves[method][base] = preservation_flags(summaries[method], summaries[base], damages[method], damages[base])
        recovery[method] = oracle_analysis(raw, methods[method], methods['GEO_ORACLE'])
        wd[method] = wd_analysis(raw, methods[method], methods['GEO_ORACLE'])
    return dict(full_frames=len(ids), original_ID_sha256=R.hashlib.sha256('\n'.join(ids).encode()).hexdigest(),
        primary_bootstrap=primary, secondary_bootstrap=group_name if primary == 'frame' else 'frame',
        bootstrap={name: bootstrap.meta() for name, bootstrap in bootstraps.items()},
        summaries=summaries, RAW_canonical_damage=damages, paired_comparisons=comparisons,
        axes=analysis_axes, preservation=preserves, oracle_recovery=recovery, W_D_hypothesis_recovery=wd,
        source_bindings=inputs, scope='seed1 only; full original IDs retained, pose failures separate from common-success paired metrics; corner conditional and full-reference denominators separate')


def fmt(value, digits=4):
    return 'NA' if value is None else f'{value:.{digits}f}'


def interval(stat):
    ci = stat['CI95']
    return fmt(stat['mean_paired_difference'], 6) + (' [NA]' if ci is None else f' [{fmt(ci[0],6)}, {fmt(ci[1],6)}]')


def probe_summary(rows):
    """Final fixed TRAIN subset, distinct from repeated pre-update exposures."""
    eligible = [r for r in rows if r['oracle_index'] >= 0]
    finite = [r for r in rows if r['selected_F_available']]
    expected_finite = [r for r in rows if not r['expected_ADDsym_infinite']]
    return dict(frames=len(rows), oracle_eligible=len(eligible),
        exact_oracle_action=sum(r['oracle_exact'] for r in eligible),
        exact_oracle_action_rate=sum(r['oracle_exact'] for r in eligible)/len(eligible) if eligible else None,
        NoOp=sum(r['NoOp'] for r in rows), selected_F_failures=len(rows)-len(finite),
        selected_finite_cached_ADDsym_m=R.distribution([r['selected_ADDsym_m'] for r in finite]),
        selected_finite_oracle_gap_m=R.distribution([r['oracle_gap_m'] for r in finite if r['oracle_gap_m'] is not None]),
        expected_infinite_frames=len(rows)-len(expected_finite),
        expected_exact_finite_cached_ADDsym_m=R.distribution([r['expected_ADDsym_m'] for r in expected_finite]),
        expected_penalty_approximation_m=R.distribution([r['expected_finite_proxy_ADDsym_m'] for r in rows]),
        failed_action_probability_mass=R.distribution([r['failed_action_probability_mass'] for r in rows]),
        scope='final checkpoint on fixed first256 TRAIN calibration IDs; cached costs only; no complete TRAIN accuracy or independent generalization claim')


def render(result):
    verdict = result['decisions']
    lines = ['# 고정 seed1 SOFT6D / EXPECT6D 비교', '',
        '[확인] SOFT6D와 EXPECT6D를 각각 원래 seed1 초기값부터 6,000 update씩 실행하고 마지막 checkpoint만 평가했다. 두 실행은 첫 방법의 성능에 따라 바뀌지 않았다.', '',
        f"[확인] 판정: SOFT6D **{verdict['SOFT6D']['label']}**, EXPECT6D **{verdict['EXPECT6D']['label']}**. N3 대비 판정과 OLD 대비 개선 신호를 구분한다.", '',
        '[확인] 모든 표는 seed1이다. T는 cm, R은 degree, ADDsym는 m이다. pose 통계는 실제 F 성공 프레임 조건부이며 실패 수는 전체 등록 프레임으로 따로 표시한다. 2D median/P90은 매칭된 관측 코너를 모은 조건부 값이고 PCK10/gross20은 GT 코너 전체에 누락 페널티를 적용한 값이다.', '']
    for split, data in result['splits'].items():
        lines.extend([f'## {split}', '',
            f"[확인] 전체 {data['full_frames']}프레임. 주 구간은 {data['primary_bootstrap']}, 보조 구간은 {data['secondary_bootstrap']} 재표집이다. 10,000회, seed 20260917, 동일 원 ID/단위 추출을 모든 방법·지표에 공유했다.", '',
            '| 방법 | pose 성공/전체 | T mean / med / P90 | R mean / med / P90 | ADD mean / med / P90 |',
            '|---|---:|---:|---:|---:|'])
        for method, summary in data['summaries'].items():
            p = summary['pose']
            values = [' / '.join(fmt(p[key][s], 6 if key == 'ADDsym_m' else 4) for s in ('mean', 'median', 'P90')) for key in R.METRICS]
            lines.append(f"| {method} | {p['available']}/{p['total_frames']} | " + ' | '.join(values) + ' |')
        lines.extend(['', '| 방법 | 2D med / P90 px | PCK10 / gross20 비율 | GT / 관측 코너 | RAW good5→bad10 / bad20→good10 | NoOp |',
                      '|---|---:|---:|---:|---:|---:|'])
        for method, summary in data['summaries'].items():
            c, damage = summary['corner'], data['RAW_canonical_damage'][method]
            lines.append(f"| {method} | {fmt(c['matched_pooled_corner8_median_px'])} / {fmt(c['matched_pooled_corner8_P90_px'])} | {fmt(c['PCK']['10'],6)} / {fmt(c['gross20'],6)} | {c['corners']} / {c['observed_corners']} | {damage['good5_to_bad10']} / {damage['bad20_to_good10']} | {summary['NoOp'] if summary['NoOp'] is not None else 'NA'} |")
        reference = data['summaries']['RAW']['corner']
        lines.extend(['', f"[확인] 2D evaluable {reference['evaluable_frames']}, matched {reference['matched']}, detected {reference['detected']}프레임. good5→bad10은 프레임 수가 아니라 동일 canonical GT 코너 수다.", '',
            '| NEW − 대조군 | T paired mean [95% CI] cm | R paired mean [95% CI] degree | ADD paired mean [95% CI] m | C 손상 보존 실패 |',
            '|---|---:|---:|---:|---|'])
        for method in NEW:
            for base in CONTROLS:
                comparison = data['paired_comparisons'][method][base]
                values = [interval(comparison['statistics'][key][data['primary_bootstrap']]) for key in R.METRICS]
                failed = ', '.join(data['preservation'][method][base]['violated']) or '없음'
                lines.append(f'| {method} − {base} | ' + ' | '.join(values) + f' | {failed} |')
        lines.extend(['', '[확인] 음의 paired mean은 낮은 오차 방향이며 CI가 0을 포함하면 확정 개선 신호로 부르지 않는다. paired median과 보조 재표집 CI, T/R/ADD median·P90 차이 및 각각의 보존 flag는 SUMMARY.json에 모두 있다.', ''])
        for method in NEW:
            rec, wd = data['oracle_recovery'][method], data['W_D_hypothesis_recovery'][method]['groups']['SWITCH']
            ratio = rec['ratio']
            lines.append(f"[확인] {method}: oracle action 적중 {rec['exact_oracle_action']['count']}/{rec['exact_oracle_action']['denominator']}; headroom>1e-7m {rec['ratio_eligible_frames']}프레임에서 rho 중앙값 {fmt(ratio['distribution']['median'])}, 음수 {ratio['negative']}, >1 {ratio['over_one']}, 허용오차 초과 oracle bound 위반 {ratio['oracle_bound_violations_beyond_tolerance']}. RAW→oracle 최종 W/D 전환 {wd['frames']}프레임 중 실제 최종 가설 회수 {wd['final_hypothesis_recovered']}/{wd['common_F_available']}.")
        lines.append('')
    lines.extend(['## 손실 설정과 학습 관측', '',
        f"[확인] TRAIN 전역 scale s={fmt(result['loss_setup']['scale_m'],9)}m, SOFT6D teacher tau={fmt(result['loss_setup']['tau'],9)}. tau는 미리 고정한 1,024 TRAIN ID의 원래 2D teacher 정규화 entropy를 맞춰 정했으며 heldout/REAL 성능을 사용하지 않았다. 모델 softmax 온도는 1이다.", '',
        '[확인] EXPECT6D는 고정 은행 전체에 대한 확률 가중 비용 합이다. 후보 좌표/F를 미분하거나 후보를 샘플링하지 않는다. 실패 후보의 유한 비용은 (최악 유효 regret/s)+1이라는 새 설계 근사이며 원 cost cache의 +inf를 바꾸지 않았다.', ''])
    for method in NEW:
        train = result['training']['methods'][method]
        stats, last = train['training_stats'], train['last100_updates']
        lines.append(f"[확인] {method}: {train['updates']} update / {train['exposures']} 반복 exposure; 누적 oracle action 정확도 {fmt(stats['exact_oracle_index_accuracy'],6)}, 마지막 100 update 정확도 {fmt(last['exact_oracle_index_accuracy'],6)}, 마지막 hardJ cached ADD 평균 {fmt(last['hard_selected_finite_cached_ADDsym_mean_m'],6)}m, 마지막 oracle gap 평균 {fmt(last['hard_selected_finite_oracle_gap_mean_m'],6)}m. 학습 기록의 변하는 checkpoint·pre-update 반복 exposure 통계이며 최종 whole-TRAIN 정확도가 아니다.")
        probe = result['fixed_TRAIN_probe_summaries'][method]
        lines.append(f"[확인] {method} 최종 checkpoint의 동일 고정 256 TRAIN ID probe: oracle action {probe['exact_oracle_action']}/{probe['oracle_eligible']} ({fmt(probe['exact_oracle_action_rate'],6)}), NoOp {probe['NoOp']}, 선택 F 실패 {probe['selected_F_failures']}, 유효 선택 cached ADD 평균 {fmt(probe['selected_finite_cached_ADDsym_m']['mean'],6)}m, gap 평균 {fmt(probe['selected_finite_oracle_gap_m']['mean'],6)}m, 실제 expected ADD가 무한인 프레임 {probe['expected_infinite_frames']}, 별도 유한 페널티 근사 평균 {fmt(probe['expected_penalty_approximation_m']['mean'],6)}m. probe는 F 추가 호출 없이 기존 비용을 조회하며 성능 선택에 쓰지 않았다.")
    lines.extend(['', '## 해석 범위', '',
        '[확인] 한 training seed와 이미 여러 번 본 DEV의 진단이다. 합성 frame 구간은 원래 singleton/pair scenario 의존성을 모두 해결하지 않으므로 scenario 보조 구간을 함께 제공한다. REAL session 구간은 13개 세션의 재표집 불확실성이며 새로운 데이터나 training seed 변동의 구간이 아니다. 다중 비교 보정은 하지 않았다.', '',
        '[확인] REAL 319장은 2D/치수 기반 재구성 pose reference를 사용한다. 독립 물리 계측은 0쌍이고 BLOCKED_DATA 상태이므로 실제 물리 자세의 개선을 확인한 결과가 아니다.', '',
        '[확인] 같은 후보 은행·scorer·초기값·순서·optimizer·6000 update에서 손실만 바꿨다. soft2D→hard6D는 목적 정렬과 target hardness가 함께 바뀌어 두 원인을 독립 식별하지 못한다. 이번 SOFT6D/EXPECT6D의 tau·scale·실패 페널티도 신규 고정 설계이다. 현재 결과로 전체 6D 손실군의 불가능, 모델 크기나 최적화 실패의 원인을 단정하지 않는다.', '',
        '[확인] 추가 seed/손실/규모 확대·subgroup·그림·runtime benchmark·원고 작업은 실행하지 않았다. 실행 시간은 실제 작업 wall time이며 모델 latency 비교가 아니다. private 데이터·feature·가중치와 외부 원본 cache가 필요하므로 공개 checkout만으로 완전 재현된다고 주장하지 않는다.', ''])
    return '\n'.join(lines)


def report():
    began = time.monotonic()
    required = [DOC / name for name in ('PROTOCOL.json', 'LOSS_SETUP.json', 'TRAIN_RECEIPTS.json')]
    if any(not path.exists() for path in required):
        raise FileNotFoundError('BLOCKED_DATA: actual protocol/setup/completed training receipts required')
    protocol, setup, training = [read(path) for path in required]
    assert setup['status'] == 'PASS'
    assert set(training['methods']) == set(NEW), 'both fixed methods must execute'
    old_fit = read(OLD_DOC / 'A_fits/GEO_seed1.json')
    for method in NEW:
        fit = training['methods'][method]
        assert fit['status'] == 'DONE' and fit['complete'] and fit['updates'] == 6000 and fit['exposures'] == 96000
        assert fit['params'] == 20259 and fit['final_checkpoint_only']
        assert fit['order_sha256'] == old_fit['order_sha256']
        assert fit['initial_state_sha256'] == old_fit['first_step']['initial_state_sha256']
        assert sha(Path(fit['checkpoint_path'])) == fit['checkpoint_sha256']
    splits = {split: analyze_split(split, training) for split in ('SYNTH_HELDOUT', 'REAL_DEV')}
    decisions = {method: decision_label({s: splits[s]['axes'][method] for s in splits},
                                      {s: splits[s]['preservation'][method] for s in splits}) for method in NEW}
    probes = {}
    for method in NEW:
        path = DOC / f'TRAIN_PROBE_{method}_seed1.json'
        probe = read(path)
        assert probe['status'] == 'DONE' and probe['checkpoint_sha256'] == training['methods'][method]['checkpoint_sha256']
        assert probe['complete'] and probe['execution']['new_F_calls'] == 0 and len(probe['rows']) == 256
        assert probe['GT_inference_access'] is False
        assert probe['code_sha256'] == sha(Path(__file__).with_name('evaluation.py'))
        assert [r['id'] for r in probe['rows']] == setup['calibration']['ids'][:256]
        assert [r['source_cache_row'] for r in probe['rows']] == setup['calibration']['rows'][:256]
        probes[method] = probe
        required.append(path)
    assert [r['id'] for r in probes['SOFT6D']['rows']] == [r['id'] for r in probes['EXPECT6D']['rows']]
    result = dict(schema='pallet_quick_pose_loss_saved_rows_v1', status='DONE', completed_methods=list(NEW), seeds=[1],
        splits=splits, decisions=decisions, loss_setup=setup, training=training, train_probes=probes,
        fixed_TRAIN_probe_summaries={method: probe_summary(probes[method]['rows']) for method in NEW},
        source_bindings=[binding(path) for path in required],
        code_binding=binding(Path(__file__)), protocol_sha256=sha(DOC / 'PROTOCOL.json'),
        secondary_analyses=['same-bank oracle headroom/recovery', 'RAW-to-oracle final W/D hypothesis switch recovery'],
        inference_GT_access=False, additional_training_seeds=False,
        execution=dict(new_model_forwards=0, new_final_F_calls=0, new_PnP_calls=0, optimizer_updates=0,
                       bootstrap_resamples_per_shared_draw_set=10000, bootstrap_seed=20260917),
        verification=dict(exact_original_ID_join=True, completed_final_checkpoint_only=True,
            original_seed1_initialization_and_order=True, same_target_and_native_bank_bytes=True,
            oracle_bound_checked=True, control_statistics_reaggregated_from_original_seed1_rows=True,
            frozen_old_namespace_unchanged_by_module=True, no_global_monkeypatch=True,
            no_manuscript_IO=True, no_new_inference_or_fit=True))
    result['execution']['report_wall_seconds'] = time.monotonic() - began
    write_json(DOC / 'SUMMARY.json', result)
    write_text(DOC / 'RESULT_KO.md', render(result))
    readme = '\n'.join(['# 고정 seed1 quick 6D loss 비교', '',
        f"[확인] SOFT6D: **{decisions['SOFT6D']['label']}**; EXPECT6D: **{decisions['EXPECT6D']['label']}**. 두 방법 각 6000 update를 실제 실행했다. 추가 seed/대규모 sweep은 없다.", '',
        '[RESULT_KO.md](RESULT_KO.md)의 단위·분모·불확실성 설명과 [SUMMARY.json](SUMMARY.json)의 원행 재집계/공유 bootstrap/보존 flag를 함께 읽는다. [PROTOCOL.json](PROTOCOL.json), [LOSS_SETUP.json](LOSS_SETUP.json), [TRAIN_RECEIPTS.json](TRAIN_RECEIPTS.json), [VERIFICATION.json](VERIFICATION.json)이 실제 설정·학습·검증 근거다.', '',
        '[확인] report 시점에 setup PASS, 두 train DONE, 두 방법의 SYNTH/REAL evaluate 및 TRAIN probe DONE을 확인했다. 최초 setup은 아래 direct setup CLI로 실행했다. 나머지는 고정 경로의 단계·완료 결과 재사용 CLI이다. verification은 보고서 생성 뒤 별도 실행하며 완료 근거는 VERIFICATION.json으로 확인한다.', '',
        '```bash', 'cd /home/minjae/Documents/github/pallet-pose',
        'export PALLET_BASELINE_ROOT=/home/minjae/Documents/github/pallet-pose-handoff-20261006',
        'PALLET_QUICK_PYTHON=/home/minjae/anaconda3/envs/pallet-yolo26/bin/python',
        '$PALLET_QUICK_PYTHON -B -m scripts.research.pallet_quick_pose_loss_20261006_v1.run --help',
        '$PALLET_QUICK_PYTHON -B -u -m scripts.research.pallet_quick_pose_loss_20261006_v1.setup --source-root /home/minjae/Documents/github/pallet-pose --bank-cache /tmp/pallet-joint-action-cache --cost-cache /tmp/pallet-pose-target-6d-cache --output-cache /tmp/pallet-quick-pose-loss-20261006-cache',
        '$PALLET_QUICK_PYTHON -B -m scripts.research.pallet_quick_pose_loss_20261006_v1.run --stage train',
        '$PALLET_QUICK_PYTHON -B -m scripts.research.pallet_quick_pose_loss_20261006_v1.run --stage evaluate',
        '$PALLET_QUICK_PYTHON -B -m scripts.research.pallet_quick_pose_loss_20261006_v1.run --stage report',
        '$PALLET_QUICK_PYTHON -B -m scripts.research.pallet_quick_pose_loss_20261006_v1.verification', '```', '',
        '| 읽기 전용 의존성 / 새 외부 출력 | 위치 |', '|---|---|',
        f"| source / 공유 detector feature·치수·seed1 순서 | {protocol['source_root']} |",
        f"| native GEO bank | {protocol['candidate_bank_cache']} |",
        f"| 완성한 TRAIN 6D 비용 은행 | {protocol['pose_cost_cache']} |",
        f"| 불변 a22 원 구현·기존 대조군 | {OLD_DOC.parents[2]} (`PALLET_BASELINE_ROOT`로 지정; 원 코드/근거 SHA 검증) |",
        f"| 신규 target·checkpoint·실행 ledger | {protocol['output_cache']} |", '',
        '| 새 private 최종 가중치 | SHA256 | bytes |', '|---|---|---:|',
        *[f"| {training['methods'][m]['checkpoint_path']} | {training['methods'][m]['checkpoint_sha256']} | {training['methods'][m]['checkpoint_bytes']} |" for m in NEW], '',
        '[확인] 저장소 출력은 `_docs/experiments/pallet_quick_pose_loss_20261006_v1/`의 PROTOCOL.json, LOSS_SETUP.json, TRAIN_RECEIPTS.json, SUMMARY.json, RESULT_KO.md, README_KO.md, VERIFICATION.json 및 `results/{SYNTH_HELDOUT,REAL_DEV}_{SOFT6D,EXPECT6D}_seed1.jsonl`/각 EXECUTION.json, `TRAIN_PROBE_{SOFT6D,EXPECT6D}_seed1.json`이다. 가중치는 `/tmp/pallet-quick-pose-loss-20261006-cache/fits/{SOFT6D,EXPECT6D}_seed1/last.pt`에 남기고 commit하지 않는다.', '',
        '[확인] 완료된 학습·평가 영수증 재사용은 `run --stage train` 또는 `run --stage all`의 검증된 CLI만 사용한다. 완료 receipt 없는 기존 ATTEMPT_STATE는 RUN_FAILED로 닫히며 자동 재학습·quiet replay가 없다. 중단된 학습을 low-level training.py로 재개하지 않는다.', '',
        '[확인] reporting은 완료된 저장 원행만 읽고 새 namespace에 보고서만 쓴다. F/NN/optimizer 추가 호출 0. private feature/weight/cache와 불변 a22 baseline은 별도 의존성이며 공개 저장소만의 완전 재현을 주장하지 않는다. REAL 독립 물리 계측은 0쌍/BLOCKED_DATA. 원고·참고문헌·PDF는 수정하지 않았다.', ''])
    write_text(DOC / 'README_KO.md', readme)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args(argv)
    result = report()
    print(json.dumps(dict(status=result['status'], decisions={k: v['label'] for k, v in result['decisions'].items()},
                          report_wall_seconds=result['execution']['report_wall_seconds']), ensure_ascii=False))


if __name__ == '__main__':
    main()
