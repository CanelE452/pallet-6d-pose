"""Exclusive reviewer summary by transcription of checked V19 saved JSON.

ROOT alone freezes/runs this source. No scientific gzip is decoded, no metric,
moment, CI, model, image, GT, fit, projection, Jacobian or SVD is recomputed.
The immutable detailed report remains a separate FULL_AUDIT_KO.md document.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import traceback

PRIVATE = Path('/tmp/pallet-finite-support-orthogonal-private-20261011-v19')
FAMILY = 'pallet_finite_support_orthogonal_audit_20261011_v19'
DOC = Path('_docs/experiments') / FAMILY
CODE = Path('scripts/research') / FAMILY
METHODS = ('BASE', 'N3_SUBPIX', 'HYBRID_STANDARD_PNP', 'NATIVE_STANDARD_PNP',
    'HYBRID_POINT_CONSENSUS_RESTORED', 'HYBRID_OBSERVED_FRAME_FACTOR_SUBSET_CONSENSUS', 'HYBRID_OBSERVED_FRAME_POINT_ONLY')
LABELS = dict(zip(METHODS, ('Base', 'N3→cornerSubPix', 'Hybrid ordinary', 'Native ordinary', 'V18 복원 점', 'V19 관측선', 'V19 같은 q 점')))
NEW = METHODS[-2:]
FIELDS = ('translation_cm', 'rotation_deg', 'ADDsym_cm')
TIMES = ('full', 'detector', 'N3_correction', 'initial_pose', 'boundary_observation',
    'endpoint_validation_and_filtered_local', 'final_pose_reprojection_and_metadata', 'return')
FILES = ('SUMMARY_PROTOCOL.json', 'SUMMARY_STARTED.json', 'SUMMARY_CHECKS.json', 'RESULT_KO.md.ready')
LIMITS = dict(saved_JSON_transcription_only=True, new_scientific_raw_decode_metric_moment_CI_GT_model_fit_RGB_J_SVD_calls=0,
    original_full_report_preserved=True, original_runtime_cleanup_failed_preserved=True,
    official_clean_runtime_or_physical_truth_or_ownership_certified=False,
    historical_full_V16_564_FAIL_remains_FAIL=True, publication_or_portable_reader_PASS_not_claimed=True)


def require(value, message):
    if not value: raise ValueError(message)


def read(path): return json.loads(Path(path).read_bytes())


def binding(path):
    path = Path(path).absolute(); require(path.is_file() and not any(p.is_symlink() for p in (path, *path.parents)), 'missing/symlink ' + str(path))
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1048576), b''): h.update(block)
    return dict(path=str(path), bytes=path.stat().st_size, sha256=h.hexdigest())


def same(a, b): return isinstance(a, dict) and isinstance(b, dict) and all(a.get(k) == b.get(k) for k in ('bytes', 'sha256'))


def write(path, value):
    with Path(path).open('x', encoding='utf-8') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False); stream.write('\n'); stream.flush(); os.fsync(stream.fileno())


def paths(a):
    f = dict(full_report=a.report_input / 'RESULT_KO.md.ready', full_checks=a.report_input / 'REPORT_RESULTS_CHECKS.json',
        full_protocol=a.report_input / 'REPORT_RESULTS_PROTOCOL.json', full_started=a.report_input / 'REPORT_RESULTS_STARTED.json', checker=Path(__file__))
    old = read(f['full_checks'])
    aliases = ('e:METRICS.json', 'd:CORRESPONDENCE_SUMMARY.json', 'd:CORRESPONDENCE_CHECKS.json',
        't:RUNTIME_RECEIPT.json', 'c:RUNTIME_CHECKS.json', 'f:FIGURE_BINDINGS.json', 'f:ROOT_VISUAL_REVIEW.json',
        'g:GEOMETRY_RECEIPT.json', 'v:VALIDATION_CHECKS.json', 's:STATISTICS_CHECKS.json', 'toy:PNG',
        'first_geometry_receipt', 'serialization_ready_proof', 'publication_checkpoint')
    f.update({k: Path(old['inputs'][k]['path']) for k in aliases})
    f.update({'PNG:' + str(i): Path(old['inputs']['PNG:' + str(i)]['path']) for i in range(9)})
    return f


def gate(f, b):
    old, proto, started = [read(f[k]) for k in ('full_checks', 'full_protocol', 'full_started')]
    require(old['schema'] == 'saved_observed_frame_report_checks_v19' and old['complete'] is old['passed'] is True
        and old['error'] is None and old['new_production_calls'] == started['new_production_calls'] == 0
        and old['inputs'] == proto['inputs'] == started['inputs'] and same(old['report'], b['full_report'])
        and same(old['own_protocol'], b['full_protocol']) and same(started['own_protocol'], b['full_protocol']), 'immutable completed detailed transcription')
    counts = old['transcription_counts']
    require(all(counts[k] == n for k, n in dict(status_rows=21, marginal_distributions=189, matched_distributions=486,
        core_moment_fields=5400, paired_CI_rows=162, runtime_summary_slots=56, PNG_images=9, fixed_case_panels=18).items())
        and counts['original_runtime_cleanup_failed'] is True and counts['runtime_statistics_official'] is False
        and counts['saved_runtime_rows_scientific_passed'] is True and counts['original_runtime_contract_passed'] is False, 'full audit and runtime scopes')
    for key in f:
        if key not in ('full_report', 'full_checks', 'full_protocol', 'full_started', 'checker'):
            require(same(b[key], old['inputs'][key]), 'same frozen detailed source ' + key)
    m, d, r, c, fig, visual, g, v, s = [read(f[k]) for k in ('e:METRICS.json', 'd:CORRESPONDENCE_SUMMARY.json',
        't:RUNTIME_RECEIPT.json', 'c:RUNTIME_CHECKS.json', 'f:FIGURE_BINDINGS.json', 'f:ROOT_VISUAL_REVIEW.json',
        'g:GEOMETRY_RECEIPT.json', 'v:VALIDATION_CHECKS.json', 's:STATISTICS_CHECKS.json')]
    require(m['complete'] is True and m['methods'] == list(METHODS) and m['all245_preserved_no_performance_filter'] is True
        and m['V16_full_scalar_validation_passed'] is False, 'seven-method known DEV evaluation')
    for name, n in (('combined', 245), ('easy', 153), ('medium', 92)):
        require(m['strata'][name]['frames'] == d['strata'][name]['frames'] == n, 'retained stratum ' + name)
        require(all(m['strata'][name]['methods'][method]['denominator'] == n for method in METHODS), 'retained method population ' + name)
    require(d['complete'] is True and d['methods'] == list(NEW), 'same correspondence membership summary')
    require(r['complete'] is False and r['status'] == 'FAILED_RUNTIME_CLEANUP' and r['statistics_official'] is False
        and r['summaries'] is None and r['serialized_rows'] == r['execution']['calls_complete'] == 150,
        'original full150 recorded runtime remains failed and unofficial')
    require(c['schema'] == 'independent_observed_frame_runtime_rows_checks_v19' and c['complete'] is True and c['passed'] is False
        and c['scientific_rows_passed'] is True and c['operational_harness_cleanup_passed'] is False and c['failure_count'] == 0
        and c['exception'] is None and c['expected_historical_cleanup_failure_count'] == c['overall_contract_failure_count'] == 1
        and c['rows'] == 150 and c['measured'] == 130 and c['summary_slots'] == 56
        and c['summaries_unofficial_due_to_historical_cleanup_failure'] is True, 'saved row audit versus original runtime cleanup FAIL')
    require(fig['complete'] is visual['complete'] is visual['passed'] is True and len(fig['figures']) == 9
        and fig['actual_counts']['case_panels'] == visual['case_panels'] == 18 and visual['actual_images_viewed'] == 9, 'actual nine-PNG review')
    viewed = {item['path']: item for item in visual['figures']}
    for i, item in enumerate(fig['figures']):
        require(same(item, b['PNG:' + str(i)]) and item['path'] in viewed and same(item, viewed[item['path']]), 'rendered/viewed same PNG')
    require(g['complete'] is v['complete'] is v['passed'] is s['complete'] is s['passed'] is True
        and v['failure_count'] == 0 and v['old_full_V16_validation_passed'] is False, 'geometry and saved-statistics scope')
    return m, d, r, c, fig, g, v, s


def cell(value):
    if value is None: return 'None'
    if isinstance(value, (dict, list)): return json.dumps(value, ensure_ascii=False, separators=(',', ':')).replace('|', '\\|')
    return str(value).replace('|', '\\|').replace('\n', ' ')


def table(headers, rows):
    return '\n'.join(['| ' + ' | '.join(headers) + ' |', '| ' + ' | '.join('---' for _ in headers) + ' |',
        *('| ' + ' | '.join(cell(v) for v in row) + ' |' for row in rows)])


def label(path, a):
    path = Path(path).resolve()
    if path.is_relative_to(a.private):
        rel = path.relative_to(a.private); target = CODE / rel if len(rel.parts) == 1 and path.suffix == '.py' else DOC / rel
    else:
        require(path.is_relative_to(a.repo), 'outside public roots'); target = path.relative_to(a.repo)
    return Path(os.path.relpath(target, DOC)).as_posix()


def report(a, f, evidence):
    m, d, r, c, fig, g, v, s = evidence; combined = m['strata']['combined']; primary = NEW[0]
    contrast_rows = []
    for comparator in ('N3_SUBPIX', METHODS[3], METHODS[2], NEW[1], METHODS[4]):
        pair = combined['contrasts'][primary + '_minus_' + comparator]['common_operational']
        contrast_rows.append([LABELS[comparator], pair['common_frames'], *[pair['metrics'][k]['mean_delta'] for k in FIELDS],
            pair['metrics'][FIELDS[0]]['CI95'], pair['metrics'][FIELDS[1]]['CI95']])
    answer = ('동시 개선을 확인했다.' if m['primary_both_T_R_means_better_than_N3_full245'] else '동시 개선을 확인하지 못했다.')
    statuses, moments, masks, named, corner = [], [], [], [], []
    for stratum, block in m['strata'].items():
        for method, value in block['methods'].items():
            statuses.append([stratum, LABELS[method], value['denominator'], value['operational'], value['new_pose'], value['fallback'], value['no_pose']])
            for metric in FIELDS:
                moment = value['metrics']['operational'][metric]
                moments.append([stratum, LABELS[method], metric, *[moment[k] for k in ('n', 'mean', 'population_std', 'sample_std', 'median', 'P90', 'max', 'unit')]])
    for method in NEW:
        diag = combined['diagnostics'][method]
        for key in ('WRONG_ON_KNOWN__BOTH_BETTER', 'WRONG_ON_KNOWN__BOTH_WORSE', 'WRONG_ON_KNOWN__MIXED_OR_UNCHANGED',
                    'MATCHES_ON_KNOWN__BOTH_BETTER', 'MATCHES_ON_KNOWN__BOTH_WORSE', 'MATCHES_ON_KNOWN__MIXED_OR_UNCHANGED'):
            value = diag['mask_and_pose_groups_vs_fixed_N3'][key]; masks.append([LABELS[method], key, value['n'], value['new_pose'], value['fallback']])
        saved = d['strata']['combined']['methods'][method]
        for key in ('accepted_fit_at_least4_proxy_correct_but_pose_both_worse', 'accepted_fit_at_least4_directly_visible_proxy_correct_but_pose_both_worse',
                    'wrong_coordinate_final_inlier_remaining', 'wrong_selected_head_q_final_inlier_remaining', 'changed_H_set_with_approved_NEW'):
            named.append([LABELS[method], key, saved['named_groups'][key]['n']])
        for group in ('DIRECT_VISIBLE', 'SELF_OCCLUDED', 'ACTUALLY_REPROJECTED'):
            value = diag[group]
            for stage in ('N3', 'output'):
                moment = value['errors'][stage]; corner.append([LABELS[method], group, stage, value['corners'], value.get('damage_5px_to_over10px'),
                    *[moment[k] for k in ('n', 'mean', 'median', 'P90')]])
    lines = [
        '# V19 관측 선분 보정 결과 — 검토용 요약',
        '> 가림 판단이 일부 틀려도 다른 유효 대응점으로 자세를 구하고 자기 가림 코너를 재투영했을 때, 기존 단순 대조보다 실제 위치·회전이 좋아졌는가?',
        '**현재 245장 평가 참조 기준의 N3 대비 운용 평균에서 위치·회전 ' + answer + '** 아래 표는 N3·native ordinary·hybrid ordinary·같은 q 점 전용 대조와의 실제 공통 245장을 따로 보여 준다. 참조는 기존 주석·K·등록 cuboid에서 재구성한 알려진 DEV GEOMETRIC_PROXY다. 독립 실측 자세·물리 edge/corner ownership·미관측 데이터 성능을 인증하지 않는다.',
        table(['대조(새 선−대조)', '공통 n', 'ΔT cm', 'ΔR deg', 'ΔADDsym cm', 'T 95% CI', 'R 95% CI'], contrast_rows),
        '음수는 새 선 경로에 유리하다. 같은 q 점 경로와는 위치·회전의 방향이 다를 수 있으므로 한 축 개선을 두 축의 성공으로 쓰지 않는다. 전체 mean·분산(ddof0/1)·SD·median·P90·max, NEW/fallback 공통 집합, 모든 162 CI 및 frame ID는 [FULL_AUDIT_KO.md](FULL_AUDIT_KO.md)와 [METRICS.json](evaluation/METRICS.json)에 있다.',
        '## 무엇을 고쳤고 무엇이 남았는가',
        '확정한 오류는 부분선 경로의 longitudinal hinge가 모델 투영 tangent를 써서 매우 짧고 관측선과 직교한 선분에서도 긴 support를 inside로 볼 수 있었던 점이다. 고정 관측 tangent의 interval objective로 바꿨다. 구조 fixture4/미분30 및 저장21context·29line 중 기존 inlier23개가 새 outlier임을 확인했다. 실제 nearest-segment 거리 자체나 물리 ownership을 인증하는 식은 아니다. accepted context5/approved NEW0이므로 이 허점을 V18 점 전용의 나쁜 자세 원인으로 단정하지 않는다.',
        '과거 새 head의 2,164 query 수치 실패→no_match 감독은 wire/depth와 POSITIVE/NONE/IGNORE로 이미 고쳤으며 main N3→cornerSubPix 오류가 아니다. 최초 읽기 전용 인수인계 당시 재학습 없음과, 이후 승인된 수정 감독 3×3,000=9,000 업데이트 완료를 구분한다. 수정 전 IMAGE_ROLE 0/319를 현재 결과로 쓰지 않는다. 이번 새 학습·RGB·seed·설정 sweep은 0이다. [감독 수정](../pallet_kp_supervision_repair_20261010_v1/RESULT_KO.md) · [수정 학습](../pallet_kp_corrected_supervision_20261010_v1/TRAINING_COMPLETION.json).',
        'corner q는 임의 한 ray 대응을 3D corner에 붙인 값이 아니라 incident한 서로 다른 TLS 경계 두 선의 교점이다. 네 hybrid 후단만 같은 H·q를 공유한다. native ordinary는 원 N3 q, fixed Base/N3의 H metadata는 unrecorded다. 코너 LOO는 H∪{k}, unused 선 endpoint LOO는 H∪{a,b}를 제외한다. 충분한 나머지 대응이 있으면 마스크 일부 오판에도 합의로 계산한다. 승인 NEW에서 H 초기 XY를 fit/score에 넣지 않고 최종 R,t로 한 번 교체하며 재투영을 새 관측으로 다시 fit하지 않는다. fallback은 원 N3 전체 출력이다.',
        '기존 V18의 추가 cost veto161개 복원과 이번 선 objective 수정은 다른 변경이다. V16 전체 198,134,007검사 중564 FAIL은 그대로 보존한다. 처음 기하 실행은 실제2solve·46optimizer 후 JSON ndarray 직렬화에서 실패해0행이었다. 별도 driver의 common.finite 두 JSON 호출만 수정해 실행했고 수학·관측·설정은 바꾸지 않았다. 다음 V21 후보 보강의 결과는 이 V19 보고서에 포함되지 않는다.',
        '## 전체 산출과 운용 오차',
        'Clean153·Moderate92, 전체245장을 유지했다. Severe74는 사용자 변경 요청으로 제외했다. Base/N3는 고정 출력이므로 NEW0/fallback0이고 산출은 전체에 존재한다. 큰 오차를 실패로 빼지 않았다. 표본 SD는 ddof1, 모집단 SD는 ddof0이며 None은 빈 집합이다.',
        table(['층', '방법', '전체', '산출', 'NEW', 'fallback', '완전 실패'], statuses),
        table(['층', '방법', '지표', 'n', '평균', '모집단 SD', '표본 SD', '중앙값', 'P90', '최대', '단위'], moments),
        '## 마스크 오류와 남은 대응',
        '아래는 저장된 GEOMETRIC_PROXY 품질과 알려진 가시성의 membership/count 진단이다. UNKNOWN/UNANNOTATED는 의미상 known 비교에서 제외하고 UNKNOWN을 NONE으로 바꾸지 않았다. proxy 8px와 DIRECT_VISIBLE proxy 8px도 물리적으로 맞는 대응의 독립 인증은 아니다.',
        table(['방법', '마스크×N3 대비 T/R', 'n', 'NEW', 'fallback'], masks),
        table(['방법', '관측/배치 진단 집합', 'n'], named),
        'eligible/solver_used/accepted_fit/final_inlier별 올바른 proxy 점 수·잘못 제외1/2·잘못 남김1/2·동시 오류·fit ID/선 ID·기존 cuboid 부호 배치·저장 rank/spectrum·모든 frame ID는 [FULL_AUDIT_KO.md](FULL_AUDIT_KO.md), [CORRESPONDENCE_SUMMARY.json](correspondence/CORRESPONDENCE_SUMMARY.json), [CORRESPONDENCE_ROWS.jsonl.gz](correspondence/CORRESPONDENCE_ROWS.jsonl.gz)에 있다. rank6나 충분한 proxy 점 수만으로 좋은 conditioning·올바른 ownership·유일해를 인증하지 않는다. 같은 선의 q와 선을 중복 관측으로 세지 않았다.',
        '## 직접 가시점 손상과 숨은 점 재투영',
        '직접 손상 flag는 N3≤5px→output>10px다. hidden 집합의 None은 이 flag가 정의되지 않는다는 뜻이다. 숨은 점 오차나 median만 낮아졌다고 R,t가 개선됐다고 말하지 않는다. ACTUALLY_REPROJECTED는 승인 NEW에서 실제로 교체한 H만 포함한다.',
        table(['방법', '집합', '단계', '코너 수', '직접 손상 flag', 'n', '평균 px', '중앙값 px', 'P90 px'], corner),
        '## 실제 호출 시간: cleanup 실패로 비공식',
        '**원 runtime은 FAILED_RUNTIME_CLEANUP / complete=False / statistics_official=False / summaries=None이다.** 26영상·13session, warmup20+measured130=150회가 모두 반환·직렬화됐지만 환경 복원이 실패했다. detector151/N3150에는 첫 predict 내부 준비1회가 포함되고 로딩 forward는0이다. 별도 저장 행 검산은 scientific_rows_passed=True/scalar FAIL0이나 overall passed=False/cleanup=False/역사 cleanup 실패1을 유지한다. 재측정하지 않았다.',
        '검출기→N3/cornerSubPix→초기 N3·Base feature 자세→고정 IMAGE_ROLE/CAL→corner LOO/강건 점 PnP→native endpoint LOO→새 선 consensus/refit→H 재투영·metadata의 실제 전체 경로 tick이다. cache replay나 기존 단계 시간의 합산이 아니다. 아래는 같은 저장 tick에서 원 describe 식으로 계산한 비공식·탐색적 통계이며 clean 배포 경로 또는 현재 process의 환경·namespace 복원 인증이 아니다. [실패 원 receipt](runtime/RUNTIME_RECEIPT.json) · [저장 행 검산](runtime_checks_ready/RUNTIME_CHECKS.json).',
        table(['단계', 'n', '평균 ms', '표본 SD ms', '중앙값 ms', 'P90 ms', '최대 ms'], [[k, *[c['summaries'][k][q] for q in ('n', 'mean', 'sample_std', 'median', 'P90', 'maximum')]] for k in TIMES]),
        '## 실제 검산과 실행량',
        table(['권위', '저장된 확인 범위'], [['완료 기하', g['actual_execution']], ['전체 기하 scalar', dict(checks=v['checks'], failures=v['failure_count'])],
            ['저장 자세 통계', s['actual_counts']], ['비공식 runtime 행', dict(checks=c['checks'], rows=c['rows'], measured=c['measured'], scalar_failures=c['failure_count'], overall_passed=c['passed'])]]),
        '원 CV/SciPy/SVD entry·return·exception/callback, 모델·head 횟수와 cleanup의 전체 counter는 [FULL_AUDIT_KO.md](FULL_AUDIT_KO.md)에 있다. 기하 replay elapsed는 초기 RGB 경로 latency가 아니다. 전체 기하 PASS는 식·산술·fit 계약, correspondence PASS는 membership/count, 시각 PASS는 가독성에 한정한다. 물리 GT·전역 유일해·성능 성공으로 확대하지 않는다.',
        '## 고정 그림: 동일한 6사례',
        '결과에 따라 사례를 바꾸지 않았다. 9PNG/18panel을 ROOT가 실제 열어 확인했다. N3의 H(pred)=[] 또는 unrecorded metadata는 실제 가림 없음이나 빈 fit 인증이 아니다. green+=입력q, cyan=출력, orange=NEW H 재투영, white=proxy, yellow=실제 선 support, magenta=virtual model segment다. [실제 시각 검토](render_ready/ROOT_VISUAL_REVIEW.json).',
    ]
    lines.extend('![' + Path(item['path']).name + '](' + label(item['path'], a) + ')' for item in fig['figures'])
    lines.extend(['![별도 유한 support 구조 진단](' + label(f['toy:PNG'], a) + ')',
        '## 공개 원행·한계',
        '원 주석·K·등록 cuboid의 proxy R,t는 N3 예측을 정답으로 되먹인 값이 아니다. N3는 코너 번호 정렬 기준이다. 원 annotation에는 unknown/manual_click/pnp_projected가 섞여 전점 독립 클릭·물리 실측 보장이 없다. CAL128/348 corner의 virtual/native 평균차3.0674443457516407e−05px도 실사 ownership 인증은 아니다. P0의 외부 가림·비가림 방해물·저대비 네 통제 변형은 별도 검증되지 않았다.',
        '상세 원 report는 수정 없이 [FULL_AUDIT_KO.md](FULL_AUDIT_KO.md)로 함께 게시한다. 40MiB 초과 원 gzip은 [ARCHIVE_MANIFEST.json](ARCHIVE_MANIFEST.json)의 ordered parts/원 SHA로 확인하고 FULL_AUDIT의 clone restore-freeze|restore-run 절차로 복원한다. 원 SHA·13문제 query의 source provenance는 물리 ID 승인과 다르다. main 변경·force push·새 촬영·annotation·학습은 수행했다고 주장하지 않는다.',
        '이 요약 생성은 push나 copied portable reader 완료를 뜻하지 않는다. V19 게시 commit/원격 SHA와 공개 복원·replay는 별도 actual receipt가 확인해야 한다. 이전 게시 checkpoint는 [PUSH_V16_V18_RECEIPT.json](../pallet_observation_refiner_status_20261011/PUSH_V16_V18_RECEIPT.json)이다.',
    ])
    return '\n\n'.join(lines) + '\n', dict(status_rows=len(statuses), operational_moment_rows=len(moments), common_contrasts=len(contrast_rows),
        mask_rows=len(masks), observation_groups=len(named), corner_rows=len(corner), runtime_rows=8, PNGs=9, structural_PNGs=1,
        full_audit_preserved=True, runtime_official=False, independent_metric_recomputation=False)


def main():
    p = argparse.ArgumentParser(description=__doc__); p.add_argument('stage', choices=('freeze', 'run')); p.add_argument('--repo', type=Path, required=True)
    p.add_argument('--private', type=Path, default=PRIVATE); p.add_argument('--report-input', type=Path, default=PRIVATE / 'report_ready')
    p.add_argument('--output', type=Path, default=PRIVATE / 'report_summary'); a = p.parse_args()
    for k in ('repo', 'private', 'report_input', 'output'): setattr(a, k, getattr(a, k).resolve())
    require(a.output.is_relative_to(a.private) and a.output != a.private and not a.output.is_relative_to(a.report_input)
        and not a.report_input.is_relative_to(a.output), 'new disjoint private summary output')
    a.output.mkdir(parents=True, exist_ok=True)
    for name in FILES if a.stage == 'freeze' else FILES[1:]: require(not (a.output / name).exists(), 'preserve ' + name)
    f = paths(a); b = {k: binding(v) for k, v in f.items()}; evidence = gate(f, b)
    if a.stage == 'freeze':
        write(a.output / FILES[0], dict(schema='saved_observed_frame_summary_protocol_v19', inputs=b, checker=b['checker'], limits=LIMITS, frozen_before_transcription=True)); return
    own = read(a.output / FILES[0]); require(own['schema'] == 'saved_observed_frame_summary_protocol_v19' and own['inputs'] == b and own['limits'] == LIMITS, 'frozen summary inputs unchanged')
    ob = binding(a.output / FILES[0]); write(a.output / FILES[1], dict(inputs=b, own_protocol=ob, checker=b['checker'], new_production_calls=0)); error = None; counts = {}
    try:
        text, counts = report(a, f, evidence); require(len(text.splitlines()) <= 250, 'reviewer summary exceeds 250 lines')
        require(b == {k: binding(v) for k, v in f.items()}, 'summary source changed')
        with (a.output / FILES[3]).open('x', encoding='utf-8') as stream: stream.write(text); stream.flush(); os.fsync(stream.fileno())
    except BaseException as exc: error = dict(type=type(exc).__name__, message=str(exc), traceback=traceback.format_exc())
    write(a.output / FILES[2], dict(schema='saved_observed_frame_summary_checks_v19', complete=error is None, passed=error is None, inputs=b, own_protocol=ob,
        checker=b['checker'], error=error, transcription_counts=counts, report=binding(a.output / FILES[3]) if (a.output / FILES[3]).exists() else None,
        limits=LIMITS, new_production_calls=0, original_runtime_contract_passed=False, publication_complete=False))
    if error: raise SystemExit(1)


if __name__ == '__main__': main()
