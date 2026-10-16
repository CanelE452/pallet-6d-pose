"""Generate reviewable Korean reports from sealed saved results only.

No report IO occurs without root_report_GO. This module never opens an image,
original reference/annotation, checkpoint, or optimizer. It does not select a
winning cap/arm or change defaults. A report with missing evidence says so.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from . import analysis as A


METHOD_LABELS = {
    'R0': 'RAW YOLO + F', 'R1': 'RAW YOLO + 최소 B/I',
    'N0': 'N3 + SubPix + F', 'N1': 'N3 + SubPix + 최소 B/I',
    'E0': '기존 전체 e', 'E1': 'e 뒤 재관측 B/I 재선택',
    'POSEFIX_RAW': 'PoseFix RAW PASS1 + F', 'N3_ONLY': 'N3 단독 + F',
    'SUBPIX_ONLY': 'SubPix 단독 + F',
    'RA': 'RAW + guard A', 'RB': 'RAW + guard B',
    'NA': 'N3/SubPix + guard A', 'NB': 'N3/SubPix + guard B',
    'EA': 'e + guard A', 'EB': 'e + guard B',
    'REOBS_MULTI': '재관측 / 기존 다중 시작', 'REOBS_SINGLE': '재관측 / E0 단일 시작',
    'TYPED_MULTI': '원 typed 관측 / 기존 다중 시작', 'TYPED_SINGLE': '원 typed 관측 / E0 단일 시작'}


def number(value, digits=3):
    return 'NA' if value is None else f'{value:.{digits}f}'


def mean_sd(distribution):
    return number(distribution['mean'])+' ± '+number(distribution['sample_std'])


def ci(value, scale=1.):
    estimate = value['mean_difference']
    bounds = value.get('CI95')
    return (number(None if estimate is None else estimate*scale)
            +' ['+('NA' if bounds is None else ', '.join(number(x*scale) for x in bounds))+']')


def flag(value):
    known = value['true_n']+value['false_n']
    if not known:
        return 'NA'
    return f"{value['true_n']}/{known}"+(f" (NA {value['unknown_n']})" if value['unknown_n'] else '')


def summary_table(population, *, arms=A.ARMS):
    lines = [
        '| 경로 | 공동 성공/N | T/R available N | 위치 평균±SD cm | 위치 분산 cm² | 위치 최대 cm | 회전 평균±SD ° | 회전 분산 °² | 회전 최대 ° | >50/>100cm | pose 없음 | 수락 | fallback | solver 거부 | 기술실패 | timeout | 모호·수락 |',
        '|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    for arm in arms:
        s = population['arms'][arm]
        t, r, status = s['errors']['translation_cm'], s['errors']['rotation_deg'], s['status_counts']
        tails = s['translation_tails']
        label = arm+' · '+METHOD_LABELS.get(arm, arm)
        lines.append(f"| {label} | {s['successes_le5']}/{s['denominator']} | {t['n']}/{r['n']} | {mean_sd(t)} | "
            f"{number(t['sample_variance'])} | {number(t['max'])} | {mean_sd(r)} | "
            f"{number(r['sample_variance'])} | {number(r['max'])} | {tails['50']['count']}/{tails['100']['count']} | "
            f"{s['no_pose_n']} | {flag(status['accepted'])} | {flag(status['fallback'])} | "
            f"{flag(status['solver_rejected'])} | {flag(status['technical_failure'])} | {flag(status['timeout'])} | "
            f"{flag(status['ambiguous_but_accepted'])} |")
    return '\n'.join(lines)


def transition_table(populations):
    lines = ['| 분모 | 주대비: 새−기존 | 양쪽 성공 | 회복 | 손상 | 양쪽 실패 | 참조 미확정 | 성공률 Δpp [95% CI] | 위치 Δcm [95% CI] | 회전 Δ° [95% CI] |',
             '|---|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for pop in ('PRIMARY', 'TEST', 'REMAINDER', 'CAL'):
        for new, before in A.PRIMARY_CONTRASTS:
            key = new+'-'+before
            p = populations[pop]['comparisons'][key]
            counts = p['transition']['counts']
            lines.append(f"| {pop}{populations[pop]['rows']} | {key} | {counts['both_success']} | "
                f"{counts['failure_to_success']} | {counts['success_to_failure']} | {counts['both_failure']} | "
                f"{counts['reference_unresolved']} | {ci(p['success_rate'], 100)} | "
                f"{ci(p['continuous']['translation_cm'])} | {ci(p['continuous']['rotation_deg'])} |")
    return '\n'.join(lines)


def failure_contribution_table(primary):
    lines = ['| 주대비 | 양쪽 실패 N | 위치 Δ총합 cm | 전체 공통분모 평균에 대한 기여 cm | 회전 Δ총합 ° | 평균 기여 ° |',
             '|---|---:|---:|---:|---:|---:|']
    for new, before in A.PRIMARY_CONTRASTS:
        key = new+'-'+before
        c = primary['comparisons'][key]
        t = c['continuous']['translation_cm']['group_contributions']['both_failure']
        r = c['continuous']['rotation_deg']['group_contributions']['both_failure']
        lines.append(f"| {key} | {c['transition']['counts']['both_failure']} | {number(t['signed_delta_sum'])} | "
                     f"{number(t['contribution_to_common_available_mean'])} | {number(r['signed_delta_sum'])} | "
                     f"{number(r['contribution_to_common_available_mean'])} |")
    return '\n'.join(lines)


def pf_common_table(common):
    lines = ['| 공통분모 | 경로 | 공동 성공/N | 위치 평균±SD cm | 위치 최대 cm | 회전 평균±SD ° | 회전 최대 ° |',
             '|---|---|---:|---:|---:|---:|---:|']
    for pop in ('CAL', 'TEST', 'COMMON95'):
        for arm in common['arms']:
            s = common['populations'][pop]['arms'][arm]
            t, r = s['errors']['translation_cm'], s['errors']['rotation_deg']
            lines.append(f"| {pop} | {arm} | {s['successes_le5']}/{s['denominator']} | {mean_sd(t)} | "
                         f"{number(t['max'])} | {mean_sd(r)} | {number(r['max'])} |")
    return '\n'.join(lines)


def diagnostics_text(diag):
    lines = ['## CAL46의 잔차·상태·물리 대응과 좋은 입력 손상', '',
        '아래 px 오차는 기존 기하 재구성 참조의 투영에 대한 잔차다. 독립적으로 클릭한 영상점 정답이나 semantic owner 정답이 아니다. '
        '영상별 초기점 묶음에 한 번 정한 C2 phase만 분석에 쓰고, phase0/180 잔차를 모두 보존한다. 점별 가까운 정답 재배정이나 90° 동등성은 추가하지 않았다.', '',
        '| CAL 단계 | 상태 | 관측 가능한 점 N | px 평균±SD | px 최대 |',
        '|---|---|---:|---:|---:|']
    for stage, states in diag['CAL_point_stage_state_errors_px'].items():
        for state, d in states.items():
            lines.append(f"| {stage} | {state} | {d['n']}/{d['full_denominator']} | {mean_sd(d)} | {number(d['max'])} |")
    p = diag['PRIMARY_PoseFix_input_R0_good_point']
    e = diag['CAL_original_e_complete_good_input']
    lines += ['', f"PRIMARY192의 R0→PoseFix 고정 입력 owner 비교: 좋은 입력(<5px) {p['good_input_lt5_n']}점 중 "
              f"출력>10px 손상 {p['damaged_good_to_gt10_n']}점. 입력>20px {p['large_input_gt20_n']}점 중 "
              f"≤10px 회복 {p['recovered_large_to_le10_n']}점. 좋은 입력의 이동량은 {mean_sd(p['good_input_movement_px'])}px다.", '',
        f"CAL 원 e의 활성 complete {e['rows']}점에서는 N3 입력<5px {e['good_input_lt5_n']}점 중 "
        f"출력>10px {e['damaged_good_to_gt10_n']}점, 입력>20px {e['large_input_gt20_n']}점 중 "
        f"출력≤10px {e['recovered_large_to_le10_n']}점이었다.", '',
        '이 점 단위 손상 수를 pose 성공→실패 이미지 수와 합치지 않는다. PF 상태별 표는 원 N3 상태를 모델의 물리 꼭짓점으로 옮긴 문맥이며 PF 자체 가시성 정답이 아니다. '
        'S/O/T/U는 실제 B/I 점으로 승격하지 않았고, partial 선의 잔차는 별도의 1차원 값이다.', '',
        '| PRIMARY192 PoseFix/e 패턴 | 이미지 N |', '|---|---:|']
    for category, value in diag['PRIMARY_PoseFix_vs_E0_patterns'].items():
        lines.append(f"| {category} | {value['n']} |")
    lines += ['', '| 경로 | 기존 e 성공 입력 N | 그중 pose 손상 N | 위치 평균 짝차 cm | 회전 평균 짝차 ° |',
              '|---|---:|---:|---:|---:|']
    for arm, value in diag['PRIMARY_E0_good_pose_diagnostic'].items():
        lines.append(f"| {arm} | {value['E0_good_input_n']} | {len(value['damaged_ids'])} | "
                     f"{number(value['delta_translation_cm']['mean'])} | {number(value['delta_rotation_deg']['mean'])} |")
    lines += ['', '현재 학습 오류지원 판정: **'+diag['error_support_status']+'**. '
              '과거의 5–15% bbox대각선 위치 교란과 현재 잔차 크기를 비교하는 것은 의미 대응·상태·오류빈도의 학습지원 검증이 아니다. 새 학습은 실행하지 않았다.', '',
              '[전체 CAL 점 잔차](diagnostics/CAL_POINT_RESIDUALS.csv), '
              '[PF 좋은 입력 이동·손상 전량](diagnostics/PRIMARY_POSEFIX_GOOD_INPUT_MOVEMENTS.csv), '
              '[상태·owner 감사](diagnostics/CAL_OWNER_AND_STATE_AUDIT.json), '
              '[전체 진단](diagnostics/DIAGNOSTICS.json).']
    return '\n'.join(lines)


def cost_text(cost, ledger, score_seal, diagnostic_seal):
    lines = ['## 실제 실행량·시간', '',
        '| 계측 | 실제 저장값 |', '|---|---:|',
        f"| 본 분석 새 학습 update | {score_seal['new_training_updates']} |",
        f"| 본 분석 새 neural forward | {score_seal['new_model_forwards']} |",
        f"| 본 분석 새 pose solve | {score_seal['new_pose_solves']} |",
        f"| 분석 프로세스 elapsed s | {number(score_seal.get('elapsed_seconds'))} |",
        f"| 분석 process CPU s | {number(score_seal.get('process_cpu_seconds'))} |",
        f"| 잔차 진단 elapsed s | {number(diagnostic_seal.get('elapsed_seconds'))} |",
        f"| 잔차 진단 process CPU s | {number(diagnostic_seal.get('process_cpu_seconds'))} |"]
    for key in ('cpu_solver_analysis_stage_elapsed_seconds', 'gpu_process_elapsed_seconds',
                'whole_task_wall_seconds', 'worker_cpu_seconds', 'top_level_new_pose_slots',
                'unique_slot_ID_arm_pairs', 'pilot_slots_reused_by_full_run',
                'additional_original_e_logging_slots', 'original_e_logging_shared_with_TYPED_MULTI'):
        lines.append(f"| 실행 장부 {key} | {number(cost.get(key))} |")
    lines += ['', '위 분석 프로세스의 solve0은 네 진단 경로의 자세 재계산0이라는 뜻이 아니다. '
              '재계산 슬롯·내부 후보 기록·GPU/CPU 누적량은 별도 실행 장부를 기준으로 한다. '
              '겹쳐 실행된 단계 시간을 임의로 더해 전체 wall time이나 core-time으로 바꾸지 않았다. '
              '미기록 값은 NA이며, 저장 pose 재집계로 full-RGB FPS를 추정하지 않는다. '
              '범위는 캐시된 관측 이후의 pose replay·저장 결과 분석이며 원래 E0 상위 pipeline 비용은 추가다. '
              '과거 소스 읽기·첫 실패 추출의 CPU 정밀 계측은 복원하지 않았다. '
              '입력 materialization은 whole-task wall에 들어가지만 runner 단계 타이머에서는 빠져 있다. '
              'pilot24는 전체952에 재사용된 것으로 추가952+24로 세지 않는다. '
              '원 e 로깅238도 TYPED_MULTI와 공유하며 별도 pose 슬롯을 중복 추가하지 않는다.', '',
              '| 경로 | 새 상위 슬롯 N | 슬롯 elapsed 합 s | 실제 worker CPU s | 슬롯 중앙/최대 s | SLSQP 후보 기록 N | seed least-squares 시도 N |',
              '|---|---:|---:|---:|---:|---:|---:|']
    for arm, row in cost.get('per_arm', {}).items():
        lines.append(f"| {arm} | {row['new_slots']} | {number(row['sum_slot_elapsed_seconds'])} | "
                     f"{number(row['worker_cpu_seconds'])} | {number(row['median_slot_elapsed_seconds'])}/{number(row['max_slot_elapsed_seconds'])} | "
                     f"{number(row['SLSQP_candidate_records'], 0)} | {number(row['seed_least_squares_attempts'], 0)} |")
    lines += ['', '후보 기록·seed 시도는 optimizer function evaluation 수나 전체 내부 solve 수와 같지 않다. '
              'function evaluation의 미계측을0으로 채우지 않았으며 TYPED 경로의 해당 분모는 NA다.', '',
              '| 저장 자원 장부 단계 | 종류 | elapsed s |', '|---|---|---:|']
    for row in ledger:
        lines.append(f"| {row.get('stage', 'UNRECORDED')} | {row.get('kind', 'UNRECORDED')} | {number(row.get('elapsed_seconds'))} |")
    lines += ['', '[실제 비용과 정의](evidence/COST.json), [자원·실행 증거](evidence/RESOURCE_LEDGER.jsonl), [분석 봉인](metrics/SCORE_SEAL.json), '
              '[진단 봉인](diagnostics/DIAGNOSTIC_SEAL.json).']
    return '\n'.join(lines)


def oracle_text(oracle):
    if oracle is None:
        return ('**CANDIDATE_RECORD_INCOMPLETE / oracle 결과 파일 미확인**. '
                'generated·pruning·selection 전량 상한을 완료로 주장하지 않는다.')
    incomplete = oracle['original_bank_incomplete']
    lines = [('**원 e가 실제 읽은 consumer 후보 bank에도 기록 공백이 있다.** '
              '그 공백을 성공 후보 부재나 관측 한계로 판정하지 않는다.') if incomplete else
             '**원 e가 실제 읽은 consumer 후보 bank는 전238장에서 확인됐다.** '
             '이는 원 caller가 읽는 accepted winner·전체 feasible·명시 초기값 범위의 완전성이다.', '',
             '원 caller가 읽지 않은 과거 W/D·P1 rejected 전량은 consumer 후보 상한의 분모가 아니다. '
             '역사적 모든 생성 시도의 기록 완전성과 실제 consumer bank의 완전성은 별도로 보존했다. '
             '고정 consumer에서 좋은 후보가 없다는 결과를 전역 물리 자세 불가능성으로 확대하지 않는다. '
             'after_support는 계측 후보의 사후 원 support 검사이며 새 순차 pruning 정책이 아니다.', '',
             '| 분모 | consumer 생성 완전 이미지 N | 미확정 이미지 N | 해당 consumer bank 완전 |',
             '|---|---:|---:|---|']
    for pop in ('PRIMARY', 'TEST', 'REMAINDER', 'CAL'):
        value = oracle['populations'][pop]
        lines.append(f"| {pop}{value['denominator']} | {value['original_consumer_generated_complete_frames']} | "
                     f"{value['original_consumer_generated_unknown_frames']} | {value['original_consumer_generated_bank_complete']} |")
    lines += ['',
             '| 분모 | 저장 단계 | 같은 후보의 joint 성공 가능 N | 후보가 저장된 이미지 N | 후보 없음 N | support 미확정 이미지 N |',
             '|---|---|---:|---:|---:|---:|']
    for pop in ('PRIMARY', 'TEST', 'REMAINDER', 'CAL'):
        value = oracle['populations'][pop]
        for stage, s in value['stages'].items():
            lines.append(f"| {pop}{value['denominator']} | {stage} | {s['joint_success_oracle']} | "
                         f"{s['candidate_available_frames']} | {s['no_candidates_frames']} | "
                         f"{s['after_support_unknown_frames'] if s['after_support_unknown_frames'] is not None else 'NA'} |")
    lines += ['', '| 분모 | 원 저장 패키지 | 단계 | 같은 후보의 joint 성공 가능 N | 후보가 저장된 이미지 N |',
              '|---|---|---|---:|---:|']
    for pop in ('PRIMARY', 'TEST', 'REMAINDER', 'CAL'):
        value = oracle['populations'][pop]
        for package, stages in value['original_stage_oracles'].items():
            for stage, s in stages.items():
                lines.append(f"| {pop}{value['denominator']} | {package} | {stage} | "
                             f"{s['joint_success_oracle']} | {s['candidate_available_frames']} |")
    primary = oracle['populations']['PRIMARY']
    lines += ['', '후보·coverage flag는 전체 이미지에 기록되므로 최종 성공 영상에도 존재할 수 있다. '
              '최종 실패의 원인 수와 혼동하지 않도록 아래 두 분모를 나눈다.', '',
              '| PRIMARY 전체192 coverage/후보 flag | 이미지 N |', '|---|---:|']
    for label, n in oracle['populations']['PRIMARY']['failure_flags'].items():
        lines.append(f'| {label} | {n} |')
    lines += ['', f"최종 실패 {primary['final_failure_n']}장, 성공 {primary['final_success_n']}장, "
              f"회전 참조 미확정 {primary['final_reference_unresolved_n']}장.", '',
              '| PRIMARY 최종 실패 영상 안의 flag | 이미지 N |', '|---|---:|']
    for label, n in primary['failure_flags_final_failure_only'].items():
        lines.append(f'| {label} | {n} |')
    lines += ['', 'flag는 중복될 수 있다. G10 family 선택과 G13 내부 refinement의 관측 패키지가 달라 '
              '두 단계를 한 목적함수의 단일 후보 사다리로 해석하지 않는다. '
              '좋은 후보가 없는 이유가 관측/제안 한계인지 단순 누락인지 기록의 범위 안에서만 나눈다. '
              'partial 선 잔차 파일은 실제 선 관측이 있을 때 생성하며 '
              '[진단 봉인 파일 목록](diagnostics/DIAGNOSTIC_SEAL.json)을 기준으로 확인한다. '
              '[SINGLE 초기값·fallback 감사](oracle/SINGLE_FALLBACK_AUDIT.json), '
              '[고정6사례 단계별 진단](oracle/DIAGNOSTIC_CASE_STAGES.json)을 함께 보존한다.']
    return '\n'.join(lines)


def independent_text(receipt, summary=None):
    if receipt is None:
        return '**독립 검산 영수증 미확인**. 보고서 생성은 독립 검산 완료를 뜻하지 않는다.'
    if 'arm_rows' in receipt and 'discrepancies' in receipt:
        lines = [f"독립 식(SciPy 회전벡터·별도 중심 norm)으로 {receipt['arm_rows']}개 행별 오차를 재집계했고 "
                 f"저장 discrepancy는 {len(receipt['discrepancies'])}개, 공동 성공 전량 일치는 {receipt['success_all_match']}다. "
                 f"허용치는 위치{receipt['tolerance']['translation_cm']}cm/회전{receipt['tolerance']['rotation_deg']}°다. "
                 '실제 기록은 [독립 행 검산](evidence/INDEPENDENT_REAGGREGATION.json)에 있다.']
        if summary is not None:
            lines.append(f"4모집단×19경로={summary['arm_cohort_rows']}개 집계의 전체N·성공·availableN·평균·표본분산 "
                         f"검산 all_pass는 {summary['all_pass']}다. "
                         '[독립 집계 확인](evidence/INDEPENDENT_SUMMARY_CHECK.json)을 별도로 보존한다. '
                         '이는 bootstrap CI나 의미 owner 정답까지 독립 검증했다는 뜻이 아니다.')
        return '\n\n'.join(lines)
    checks = receipt.get('checks', receipt.get('check_count', receipt.get('checks_run')))
    failures = receipt.get('failure_count', receipt.get('failures'))
    if isinstance(checks, (list, dict)):
        checks = len(checks)
    if isinstance(failures, (list, dict)):
        failures = len(failures)
    return (f"독립 영수증에 기록된 검사 수 {checks if checks is not None else 'NA'}, "
            f"실패 수 {failures if failures is not None else 'NA'}. "
            '검산 범위·SciPy와 trace/acos의 수치 차이·참조 바인딩·상태 분모는 '
            '[독립 영수증](evidence/INDEPENDENT_REAGGREGATION.json)에서 확인한다. 숫자가 미기록인 항목을 0으로 채우지 않았다.')


def observation_headline(folder):
    """Read sealed predictor-only lineage counts and verify full238 owner rows."""
    folder = Path(folder)/'e_trace_v2'
    seal = A.read(folder/'EXTRACTION_SEAL.json')
    path = folder/'OBSERVATION_TYPE_AND_OWNER_AUDIT.csv'
    if A.sha(path) != seal['files'][path.name]['sha256']:
        raise ValueError('Sealed observation/owner audit changed')
    with path.open(newline='') as stream:
        rows = list(csv.DictReader(stream))
    if (len(rows) != 238*8 or seal['rows'] != 238
            or set(r['id'] for r in rows) != set(seal['ordered_ids'])):
        raise ValueError('All238 eight-corner observation lineage required')
    changed = [r for r in rows if r['N3_to_e_same_channel_physical_owner_changed'] == 'True']
    if len(changed) != seal['totals']['N3_to_e_same_channel_physical_owner_changed']:
        raise ValueError('Owner mapping count differs from extraction seal')
    return dict(rows=seal['rows'], totals=seal['totals'],
        partial_to_post_full_sources=seal['partial_to_post_full_sources'],
        nonidentity_frames=len({r['id'] for r in changed}), nonidentity_vertices=len(changed),
        semantic_owner_status=seal['semantic_owner_status'], extraction_seal_sha256=A.sha(folder/'EXTRACTION_SEAL.json'))


def observation_text(audit):
    t, s = audit['totals'], audit['partial_to_post_full_sources']
    return (f"실제238장 관측 계보에서 B/I {t['visible_archives']}개는 원 e의 활성 full {t['complete']}개, "
        f"typed partial {t['typed_partial']}개, legacy partial {t['legacy_partial']}개로 나뉜다. "
        f"그중 원 e owner의 partial이 E 후단에서 full로 연결된 것은 {t['e_owner_partial_to_post_full']}개"
        f"(typed {t['e_owner_typed_to_post_full']}+legacy {t['e_owner_legacy_to_post_full']})다. "
        f"출처는 예측 fallback {s.get('VISIBLE_BACKBONE_FALLBACK', 0)}개, 배경 경계선 {s.get('B_TARGET_BACKGROUND_LINES', 0)}개, "
        f"내부 접합 {s.get('I_INTERNAL_RGB_JUNCTION', 0)}개, 경계 복구 {s.get('B_ACTUAL_BOUNDARY_RECOVERY', 0)}개다. "
        '예측 fallback을 새 영상 교점 측정으로 읽으면 안 되며, partial 원관측과 새 full 패키지는 같은 관측이 아니다.\n\n'
        f"N3 채널과 e owner 사이에는 비항등 모델 vertex 매핑이 {audit['nonidentity_frames']}장/"
        f"{audit['nonidentity_vertices']}꼭짓점에서 필요했다. 모델 Q·등록 bijection은 검산했지만 영상에서 그 점·선이 "
        f"실제로 어느 물리 코너인지의 정답은 **{audit['semantic_owner_status']}**다. "
        f"이는 후단 패키지 변경과 손상을 연결하는 진단이며 {t['e_owner_partial_to_post_full']}개 각각이 실패를 일으켰다는 "
        '순수 인과증명이나 재번호 하나의 효과가 아니다. '
        '[봉인된 전238 관측·owner 감사](evidence/OBSERVATION_TYPE_AND_OWNER_AUDIT.csv), '
        '[추출 집계·봉인](evidence/EXTRACTION_SEAL.json)을 먼저 확인한다.')


def typed_numeric_audit(scored_rows):
    """Report raw floating differences, retaining the existing replay tolerance."""
    primary = [r for r in scored_rows if r['partition'] != 'CAL']
    if len(primary) != 192:
        raise ValueError('Same PRIMARY192 required for floating-difference audit')
    result = {}
    for key in A.METRICS:
        values = [r['metrics']['TYPED_SINGLE'][key]-r['metrics']['E0'][key] for r in primary
                  if all(r['metrics'][a][key] is not None for a in ('TYPED_SINGLE', 'E0'))]
        result[key] = dict(common_available_n=len(values), raw_improved=sum(v < 0 for v in values),
            raw_worsened=sum(v > 0 for v in values), raw_tied=sum(v == 0 for v in values),
            max_absolute_difference=max(map(abs, values)) if values else None,
            beyond_existing_1e_9_tolerance=sum(abs(v) > 1e-9 for v in values))
    return result


def tradeoff_text(primary, numeric):
    t, r = numeric['translation_cm'], numeric['rotation_deg']
    lines = ['## 복원·재관측의 손익과 수치적 동등성', '',
        f"TYPED_SINGLE−E0의 PRIMARY 위치는 {t['raw_tied']}/{t['common_available_n']}장에서 정확히 같다. "
        f"회전의 원시 비교에는 개선 {r['raw_improved']}장·악화 {r['raw_worsened']}장·동률 {r['raw_tied']}장이 남지만 "
        f"최대 차이는 {r['max_absolute_difference']:.12g}°다. 기존 replay 허용치1e-9를 넘은 경우는 "
        f"{r['beyond_existing_1e_9_tolerance']}장이며, 이 부동소수 차이를 실질적인 정확도 이득으로 세지 않는다. "
        '원시 비교 수와 값을 지우거나 새 임곗값으로 성공 여부를 바꾸지 않았다.', '',
        '| PRIMARY192 재관측−e | 회복 | 손상 | 양쪽 실패 | 위치 평균 짝차 cm | 회전 평균 짝차 ° |',
        '|---|---:|---:|---:|---:|---:|']
    for arm in ('REOBS_MULTI', 'REOBS_SINGLE'):
        pair = primary['comparisons'][arm+'-E0']
        c = pair['transition']['counts']
        lines.append(f"| {arm} | {c['failure_to_success']} | {c['success_to_failure']} | {c['both_failure']} | "
                     f"{number(pair['continuous']['translation_cm']['mean_difference'])} | "
                     f"{number(pair['continuous']['rotation_deg']['mean_difference'])} |")
    a = primary['comparisons']['REOBS_MULTI-E0']['transition']['ids']['success_to_failure']
    b = primary['comparisons']['REOBS_SINGLE-E0']['transition']['ids']['success_to_failure']
    lines += ['', f"두 재관측 경로의 손상 ID 집합은 실제로 {'같다' if set(a) == set(b) else '다르다'}. "
              '단순히 개수가 같다는 이유로 같은 사례라고 간주하지 않았으며 ID 전량은 전이표에 보존했다.', '',
              '| PRIMARY192 경로 | 위치 최대 cm | >50cm N | >100cm N |', '|---|---:|---:|---:|']
    for arm in ('E0', 'TYPED_SINGLE', 'REOBS_MULTI', 'REOBS_SINGLE'):
        s = primary['arms'][arm]
        lines.append(f"| {arm} | {number(s['errors']['translation_cm']['max'])} | "
                     f"{s['translation_tails']['50']['count']} | {s['translation_tails']['100']['count']} |")
    lines += ['', 'typed는 기존 e의 최대오차와 실패를 함께 복원한다. 재관측이 최대 위치오차를 더 낮추는 이득도 남으므로 '
              'typed가 모든 지표에서 우월하다고 주장하지 않는다. 평균·공동 성공·tail 빈도·최대값을 함께 해석한다. '
              '선택손실이 모두 큰 W/D 뒤집힘이라는 결론도 아니다. family 선택과 동일 family 내부의 작은 임계값 손실을 구분한다.']
    return '\n'.join(lines)


def recommendation_text(primary, test, oracle):
    damages = [primary['comparisons'][a+'-E0']['transition']['counts']['success_to_failure']
               for a in ('REOBS_MULTI', 'REOBS_SINGLE')]
    test_damages = [test['comparisons'][a+'-E0']['transition']['counts']['success_to_failure']
                    for a in ('REOBS_MULTI', 'REOBS_SINGLE')]
    typed = primary['arms']['TYPED_SINGLE']['successes_le5']
    typed_test = test['arms']['TYPED_SINGLE']['successes_le5']
    flags = None if oracle is None else oracle['populations']['PRIMARY']['failure_flags_final_failure_only']
    selection = 'NA' if flags is None else flags.get('SELECTION_LOSS', 0)
    limit = 'NA' if flags is None else flags.get('OBSERVATION_OR_PROPOSAL_LIMIT', 0)
    return '\n'.join(['| 항목 | 권고 | 근거·한계 |', '|---|---|---|',
        '| N3/SubPix | 유지 | 기존 고정 비교 결과와 함께 유지한다. 이번 실험은 새 제거 실험이 아니며 typed 복원에서 이 초기 경로를 버릴 근거를 얻지 못함 |',
        f"| E 후 B/I 재관측 | 운영 채택 보류 | PRIMARY 기존 성공 손상 MULTI/SINGLE {damages[0]}/{damages[1]}장·"
        f"TEST {test_damages[0]}/{test_damages[1]}장. 일부 회복·더 낮은 최대값을 포함해도 무손상 개선이 아님 |",
        f"| typed SINGLE/MULTI | 복원 대조군으로 유지 | e와 같은 {typed}/192·{typed_test}/49이며 새 정확도 향상이 아님. "
        'SINGLE 시작정책의 순수 인과분리는 NOT_ISOLATED |',
        f"| 다음 판단 | 선택손실과 관측·제안 제한 분리 | 고정 consumer의 선택손실{selection}와 관측·제안 제한{limit}을 "
        '각 근거 범위 안에서 다룬다. 선택손실이 모두 큰 W/D 오류라는 결론이나 사후 GT 선택기를 만들지 않음 |'])


def render(results, common, diag, oracle, cost, ledger, metadata):
    """Pure renderer, permitting generated-fixture checks without data IO."""
    pops = results['populations']
    primary = pops['PRIMARY']
    if (list(results['arms']) != list(A.ARMS) or {p: pops[p]['rows'] for p in A.PARTITIONS} != A.PARTITIONS
            or primary['rows'] != 192 or common['PRIMARY192_established'] is not False):
        raise ValueError('Final full denominators and comparator contract required')
    p = primary['comparisons']['TYPED_SINGLE-E0']
    counts = p['transition']['counts']
    lines = ['# 상태별 관측 전달·대응·자세 선택 진단 결과', '',
        observation_text(metadata['observation_headline']), '',
        f"PRIMARY192의 사전고정 주대비 TYPED_SINGLE−E0는 회복 {counts['failure_to_success']}장, 손상 "
        f"{counts['success_to_failure']}장이다. 위치 평균 짝차는 {ci(p['continuous']['translation_cm'])}cm, "
        f"회전 평균 짝차는 {ci(p['continuous']['rotation_deg'])}°다. 연속 오차는 낮을수록 좋다. "
        '이 수치와 관측/목적함수 동등성·후보 bank 범위를 함께 해석해야 한다.', '',
        '## 네 우선순위의 완료 범위', '',
        '| 우선순위 | 이 보고서의 근거 | 판정 범위 |', '|---|---|---|',
        '| 1 관측 전달·owner | 전체238 계보·실제 consumer 활성항·CAL 잔차 | 모델 좌표 등록 검산과 semantic owner 정답을 분리. semantic owner는 미검증 |',
        '| 2 SINGLE 대 MULTI | 고정4경로 결과와 parity·패킷 봉인 | N3/SubPix 유지, e 후 재관측 운영 채택 보류. typed는 복원 대조군이며 추가 이득 아님. 순수 시작정책 비교는 NOT_ISOLATED 한계 적용 |',
        '| 3 후보 상한·선택 | 아래 consumer coverage와 단계별 기록 | 원 caller가 실제 읽은 후보의 전량성과 역사적 모든 생성 시도의 기록을 분리. consumer 공백은 상한 미확정으로 보존 |',
        '| 4 전체 손익·PoseFix | 192/49/143/46 전량, 회복·손상·tail, 별도 PF+SubPix95 | 반복 개발 자료. 서로 다른 머리/초기화/후단 패키지의 비교이며 독립 test나 순수 보정기 비교가 아님 |', '',
        '## 모집단·지표·기술 상태', '',
        'PRIMARY192=TEST49+REMAINDER143, CAL46은 별도다. severe·새 데이터·새 정답·새 학습·새 seed는 추가하지 않았다. '
        '성공은 같은 pose 하나의 중심 위치≤5cm AND physical/C2 회전≤5°다. 직사각형의 90° W/D 오류는 C2 동등성으로 지우지 않는다. '
        '분산은 cm²/°², SD는 cm/°이며 표본분산·SD(ddof1)를 사용한다.', '',
        '연속 통계는 관측 가능 N에 대한 값이며 pose 없음도 이진 실패의 전체 분모에 남긴다. '
        '짝차는 두 경로 공통 available만 사용하고 그 N·CI·전체 N을 원자료에 함께 저장한다. '
        '상태 열은 true/기록된 N이다. NA는0이 아니다. 수락·fallback·ambiguity는 정확도 성공과 다르다. '
        'SINGLE의 local rank 기반 ambiguity와 MULTI의 후보 탐색은 검출 범위가 달라 scope를 별도로 보존했다. '
        'E0 초기값을 사용하는 경로는 REOBS_MULTI·REOBS_SINGLE·TYPED_SINGLE이며, TYPED_MULTI는 원 e의 계층 pipeline을 재현한다. '
        'TS의 audit.E0_start_only와 SINGLE 범위는 봉인 예측을 바꾸지 않고 분석 메타데이터에서 운반했다. '
        'E0의 수락/fallback은 원래 공통 캐시의 NA와 별도로 봉인된 원 E 입력의 명시 상태를 복원했다. '
        '정확히 같은 저장 pose와 source SHA를 검산하고 historical_common 및 restored_original_status를 함께 보존했으며, '
        '성공 여부나 joint-feasible 값으로 상태를 추측하지 않았다.', '']
    for pop in ('PRIMARY', 'TEST', 'REMAINDER', 'CAL'):
        lines += [f"## {pop} {pops[pop]['rows']}장 전체19경로", '', summary_table(pops[pop]), '']
    lines += [tradeoff_text(primary, metadata['typed_numeric_audit']), '',
        '## 사전고정 주대비·양방향 전이', '', transition_table(pops), '',
        '세 주대비는 결과 전에 고정한 TYPED_SINGLE−E0, TYPED_SINGLE−TYPED_MULTI, TYPED_SINGLE−REOBS_SINGLE다. '
        'TYPED_SINGLE−TYPED_MULTI의 시작정책 인과분리는 **NOT_ISOLATED**다. MULTI의 G10 full/legacy family 선택 뒤 G13와 '
        'SINGLE의 이미 선택된 family 고정·별도 B/I seed fit 생략이 함께 달라, 단일 시작 효과만으로 귀속하지 않는다. '
        '마지막은 관측 패키지 전체의 차이이며 재번호 하나의 순수 인과효과가 아니다. '
        'CI는 해당 ID 집합의 실제 원녹화 cluster를 whole-recording 단위로 10,000회, seed20260917로 새로 재표집한 '
        'frame-weighted 짝차다. 새 독립 평가나 다중비교 확증적 유의성·동등성 검증으로 쓰지 않는다.', '',
        '## 양쪽 실패 집단의 오차 기여', '', failure_contribution_table(primary), '',
        '회복−손상은 성공 수 차이와 같고, 네 전이군의 연속 짝차 합은 같은 공통 available 분모의 평균 차이를 재현한다. '
        '양쪽 실패의 큰 악화도 포함했다. 전체 실패 ID·>50/>100cm ID·최대값과 모든 보조 대비는 '
        '[전체 결과](metrics/RESULTS.json), [행별 오차](metrics/ROW_METRICS.csv), '
        '[전량 전이표](metrics/PAIR_TRANSITIONS.csv)에 있다.', '',
        '## PoseFix RAW와 두 기존 SubPix cap의 공통95', '',
        '전체192의 PF comparator는 synthetic-only PRIOR1 last6000의 uncapped RAW PASS1+F다. '
        '실사 replay last300이나 synthetic selection의 1% wrapper와 다르다. Gaussian 입력·좌표 loss는 이미 구현되어 있다. '
        '아래 두 SubPix cap은 기존 CAL46/TEST49의 고정 결과를 모두 직접 재채점했으며 승자를 선택하지 않았다.', '',
        pf_common_table(common), '',
        '**POSEFIX_SUBPIX_FULL_COHORT_NOT_ESTABLISHED**: 전체192에 사용할 단일 기존 규칙이 확립되지 않았으므로 '
        '이 두 cap의 공통95를 RAW192에 합치거나 더 좋은 cap으로143장을 채우지 않았다. '
        'RAW와의 비교만으로 기존 Table II 최상 경쟁군보다 우수하다고 일반화하지 않는다. '
        '[두 cap의 전량 결과·전이·CI](metrics/POSEFIX_SUBPIX_COMMON95.json), '
        '[95장 행별 수치](metrics/POSEFIX_SUBPIX_COMMON95.csv), [원문·구현·과거 실패 감사](POSEFIX_TRANSFER_AUDIT.md).', '',
        diagnostics_text(diag), '',
        '## 후보 상한·실패 단계와 남는 공백', '',
        metadata['oracle_summary_text'], '',
        '다른 후보의 좋은 위치와 회전을 합쳐 공동 성공을 만들지 않았다. oracle은 고정된 기하 재구성 참조 기준의 '
        '저장 후보 도달 가능성이지 물리 참자세의 절대 상한이 아니다. 후보 누락은 실패0건이 아니라 판정 불가로 보존한다. '
        '[후보 범위·진단](ORACLE_LIMITS.md), [oracle 전체 결과](oracle/ORACLE_SUMMARY.json).', '',
        '## 구현 유지·보류와 다음 판단', '',
        recommendation_text(primary, pops['TEST'], oracle), '',
        '운영 기본값·논문·가중치는 교체하지 않았다. TYPED_MULTI가 원 e와 같은 관측·후보·선택을 복원했다면 '
        '동등한 복원 대조군으로 해석하며, E0로 복귀한 것을 e보다 향상된 새 방법으로 세지 않는다. '
        '전체 손상·원49 보존·평균과 tail을 함께 확인하고 유리한 영상별 routing이나 새 비열등성 margin을 만들지 않는다.', '',
        '후속 판단은 좋은 후보 부재, 중간 pruning, 최종 선택, refinement 이동을 기록이 허용하는 범위에서 나누는 것이다. '
        '큰 실패가 남았다는 사실만으로 상태별 처리의 기여를 폐기하거나 같은 Gaussian·좌표 loss·교란/replay 학습을 새 제안으로 반복하지 않는다. '
        '현재 학습 오류지원은 미확인이며 새 학습안은 채택하지 않았다.', '',
        cost_text(cost, ledger, metadata['score_seal'], metadata['diagnostic_seal']), '',
        '## 재현·독립 검산·게시 상태', '',
        metadata['independent_summary_text'], '',
        metadata.get('metadata_bridge_text', ''), '',
        '이 보고서는 저장 결과에서 생성했다. [재현 안내](REPRODUCE.md), [실제 단계별 갤러리](gallery/index.html), '
        '[선행·기여 범위](CONTRIBUTION_EVIDENCE.md), [문헌·소스 접근 장부](LITERATURE_SOURCE_AUDIT.json)를 연결한다. '
        '실제 push SHA와 원격 게시 확인은 별도의 게시 영수증으로 확인하며, 이 생성기가 게시 완료를 선언하지 않는다.', '',
        '| 봉인·소스 | SHA256 |', '|---|---|']
    for key, value in metadata['hashes'].items():
        lines.append(f'| {key} | `{value}` |')
    lines += ['', '## 요구 산출물 확인', '', '| 산출물 | 저장 증거 상태 |', '|---|---|']
    for key, value in metadata['artifact_status'].items():
        lines.append(f'| {key} | {value} |')
    lines += ['', '파일 존재·소스 감사·수치 검산과 방법 효과 입증은 별개다. 누락/불완전 항목을 완료로 승격하지 않았다. '
        '원 RGB·대형 저널·기존 실패 시도·원참조·사용자 수정은 보존하며, 원 참조 좌표 파일은 분석 전용 로컬 증거다.', '']
    return '\n'.join(lines)


def _verify_assets(folder, seal_name, key='artifacts'):
    seal = A.read(Path(folder)/seal_name)
    for name, expected in seal[key].items():
        if A.sha(Path(folder)/name) != expected:
            raise ValueError('Saved report input seal changed: '+name)
    return seal


def generate(folder, output, *, score_folder=None, diagnostic_folder=None, oracle_folder=None,
             independent_receipt=None, cost_path=None, root_report_GO=False):
    if root_report_GO is not True:
        raise PermissionError('Root report GO required before result IO')
    folder, output = Path(folder), Path(output)
    score_folder = Path(score_folder) if score_folder is not None else folder/'score_v2'
    diagnostic_folder = Path(diagnostic_folder) if diagnostic_folder is not None else folder/'diagnostics_v1'
    oracle_folder = Path(oracle_folder) if oracle_folder is not None else folder/'oracle_v1'
    cost_path = Path(cost_path) if cost_path is not None else folder/'COST.json'
    if any((output/n).exists() for n in ('REPORT_KO.md', 'REPRODUCE.md')):
        raise FileExistsError('Preserve previous generated report attempts')
    ss = _verify_assets(score_folder, 'SCORE_SEAL.json')
    ds = _verify_assets(diagnostic_folder, 'DIAGNOSTIC_SEAL.json')
    if (A.sha(folder/'PREDICTION_SEAL.json') != ss['prediction_seal_sha256']
            or A.sha(folder/'CANDIDATE_SEAL.json') != ss['candidate_seal_sha256']):
        raise ValueError('Same complete postseal experiment required')
    bridge_path = folder/'SCORE_V1_V2_METADATA_BRIDGE.json'
    bridge = None
    if ds['score_seal_sha256'] != A.sha(score_folder/'SCORE_SEAL.json'):
        bridge = A.read(bridge_path)
        if (bridge['old_score_seal_sha256'] != ds['score_seal_sha256']
                or bridge['new_score_seal_sha256'] != A.sha(score_folder/'SCORE_SEAL.json')
                or bridge['all_4522_metric_dicts_exact_equal'] is not True
                or bridge['all_76_numeric_arm_aggregates_exact_equal'] is not True
                or bridge['all_paired_comparisons_and_CIs_exact_equal'] is not True
                or bridge['prediction_seal_sha256'] != ss['prediction_seal_sha256']
                or bridge['candidate_seal_sha256'] != ss['candidate_seal_sha256']):
            raise ValueError('Metadata-only bridge required for prior diagnostic and independent evidence')
    results, common = A.read(score_folder/'RESULTS.json'), A.read(score_folder/'POSEFIX_SUBPIX_COMMON95.json')
    observation = observation_headline(folder)
    numeric = typed_numeric_audit(A.read(score_folder/'SCORED_ROWS.json')['rows'])
    diag = A.read(diagnostic_folder/'DIAGNOSTICS.json')
    oracle_path = oracle_folder/'ORACLE_SUMMARY.json'
    oracle = None
    if oracle_path.exists():
        oracle_seal = _verify_assets(oracle_folder, 'ORACLE_SEAL.json')
        if (oracle_seal['prediction_seal_sha256'] != ss['prediction_seal_sha256']
                or oracle_seal['candidate_seal_sha256'] != ss['candidate_seal_sha256']):
            raise ValueError('Oracle belongs to different sealed predictions')
        oracle = A.read(oracle_path)
    independent_path = Path(independent_receipt) if independent_receipt is not None else folder/'INDEPENDENT_REAGGREGATION.json'
    independent = A.read(independent_path) if independent_path.exists() else None
    independent_summary = A.read(folder/'INDEPENDENT_SUMMARY_CHECK.json') if (folder/'INDEPENDENT_SUMMARY_CHECK.json').exists() else None
    cost = A.read(cost_path) if cost_path.exists() else {}
    ledger_path = folder/'RESOURCE_LEDGER.jsonl'
    ledger = [json.loads(line) for line in ledger_path.read_text().splitlines() if line.strip()] if ledger_path.exists() else []
    metadata = dict(score_seal=ss, diagnostic_seal=ds, oracle_summary_text=oracle_text(oracle), typed_numeric_audit=numeric,
        observation_headline=observation,
        independent_summary_text=independent_text(independent, independent_summary),
        metadata_bridge_text=('SCORE v2는 TS의 E0 초기화·SINGLE ambiguity 범위만 바로잡은 메타데이터 수정이다. '
            '4,522개 metric dict·76개 수치 집계·모든 짝대비/CI의 v1/v2 exact 일치를 확인했다. '
            '기존 diagnostics v1·gallery·독립 metric 영수증은 '
            '[동등성 연결 영수증](evidence/SCORE_V1_V2_METADATA_BRIDGE.json)으로 연결하며 재추론하지 않았다.' if bridge else ''),
        hashes={'PREDICTION_SEAL': ss['prediction_seal_sha256'], 'CANDIDATE_SEAL': ss['candidate_seal_sha256'],
                'SCORE_SEAL': A.sha(score_folder/'SCORE_SEAL.json'), 'DIAGNOSTIC_SEAL': A.sha(diagnostic_folder/'DIAGNOSTIC_SEAL.json'),
                'EXTRACTION_SEAL': observation['extraction_seal_sha256'],
                'analysis.py': ss['source_sha256']['analysis.py'], 'report.py': A.sha(__file__)},
        artifact_status={'전체19경로4모집단': '실제 봉인 score에서 생성', 'PoseFix+SubPix 공통95 두 cap': '둘 다 보고; 전체192 미확립',
                         'CAL 잔차·good 입력 진단': '실제 진단 봉인 확인',
                         '후보 oracle': '결과 파일 존재; coverage 제한 확인 필요' if oracle else '미완료/미확인',
                         '독립 검산': '영수증 존재; 실제 실패·범위 확인 필요' if independent else '영수증 미확인',
                         '통합 COST': '명시 COST 기록 존재' if cost else '통합 기록 미확인; 단계 ledger 및 개별 seal만 보고'})
    if bridge is not None:
        metadata['hashes']['SCORE_V1_V2_METADATA_BRIDGE'] = A.sha(bridge_path)
    body = render(results, common, diag, oracle, cost, ledger, metadata)
    output.mkdir(parents=True, exist_ok=True)
    with (output/'REPORT_KO.md').open('x') as stream:
        stream.write(body)
    reproduce = '''# 재현 안내

입력은 기존 고정238(CAL46/TEST49/추가143), 동일 가중치·seed·참조·등록 Q다. 새 fit·가중치 업데이트·데이터·정답 수정은 없다. 원문의 전체 실행 범위와 source hashes는 REPLAY_PROTOCOL·SOURCE_AND_INPUT_MANIFEST·실제 실행 영수증을 우선한다.

`python -m scripts.research.pallet_statewise_correspondence_diagnosis_v1.analysis --help`

`python -m scripts.research.pallet_statewise_correspondence_diagnosis_v1.diagnose --help`

`python -m scripts.research.pallet_statewise_correspondence_diagnosis_v1.report --help`

analysis는 현재 REPLAY_PREDICTIONS와 후보 bank의 두 봉인 SHA, 기존 single/guarded 저장 결과, 고정 reference manifest를 요구한다. 명시적 root scoring GO 전에는 자료 IO를 거부한다. 모든 예측·후보가 봉인된 뒤 별도 프로세스에서만 참조를 읽는다. 기존 결과나 참조 파일을 출력 경로로 쓰지 않는다.

diagnose는 같은 두 봉인, 분석 SCORE_SEAL, e_trace_v2 EXTRACTION_SEAL, 원 component 입력 봉인을 검산한 뒤 CAL 투영잔차·원 owner/state와 PRIMARY 좋은 입력 손상을 계산한다. 이 잔차는 기하 재구성 참조이며 독립 2D 클릭 정답이 아니다. phase0/180을 둘 다 남기고 영상별 전역 C2 phase만 분석한다.

출력 디렉터리는 매 시도 새 경로를 사용한다. 실패 시도도 지우지 않는다. 두 기존 PoseFix+SubPix cap은 공통95에만 각각 보고하며 결과로 선택하지 않는다. 후보의 생성 누락·원49 손상·noPose·연속 오차 NA를 제거하지 않는다. CI는 실제 원녹화 cluster,10,000회,seed20260917을 사용한다.

순수 수학/임시 봉인 테스트: `python -m unittest scripts.research.pallet_statewise_correspondence_diagnosis_v1.test_analysis scripts.research.pallet_statewise_correspondence_diagnosis_v1.test_diagnose`.

공개 작은 결과의 해시는 SCORE_SEAL·DIAGNOSTIC_SEAL·독립 검산/게시 영수증을 확인한다. 원 RGB·대형 저널·checkpoint·원참조는 로컬 바이트 바인딩으로 보존한다. 공개 보고서의 존재만으로 원자료 다운로드 없이 전 단계 추론을 재현할 수 있다고 주장하지 않는다.
'''
    with (output/'REPRODUCE.md').open('x') as stream:
        stream.write(reproduce)
    return {'REPORT_KO.md': A.sha(output/'REPORT_KO.md'), 'REPRODUCE.md': A.sha(output/'REPRODUCE.md')}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--folder', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    for key in ('score-folder', 'diagnostic-folder', 'oracle-folder', 'independent-receipt', 'cost-path'):
        p.add_argument('--'+key, type=Path)
    p.add_argument('--root-report-go', action='store_true')
    a = p.parse_args()
    result = generate(a.folder, a.output, score_folder=a.score_folder, diagnostic_folder=a.diagnostic_folder,
        oracle_folder=a.oracle_folder, independent_receipt=a.independent_receipt, cost_path=a.cost_path,
        root_report_GO=a.root_report_go)
    print(json.dumps(result, ensure_ascii=False))


if __name__ == '__main__':
    main()
