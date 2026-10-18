"""Prepare descriptive reports from sealed outputs; never fit, predict, or solve.

Private locations are explicit CLI inputs. Public derivatives carry exact
source/transform hashes for the separate publication checker.
"""
from __future__ import annotations
import argparse
from collections import Counter
import csv
import gzip
import hashlib
import io
import json
from pathlib import Path
from scripts.research.pallet_risk_information_v1 import publish_pack as OLD

NS = 'pallet_observation_contract_h1_v1'
SEEDS = (20261018, 20261019, 20261020)


def put(path, data, history):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(data, str): data = data.encode()
    if path.exists() and path.read_bytes() != data:
        old = history / (path.name + '.' + OLD.sha(path))
        old.parent.mkdir(parents=True, exist_ok=True)
        if not old.exists(): old.write_bytes(path.read_bytes())
    path.write_bytes(data)


def encoded(data): return OLD.encoded(data)


def table(headers, rows):
    def cell(x): return str(x).replace('|', '\\|').replace('\n', ' ')
    return '\n'.join(['| ' + ' | '.join(map(cell, headers)) + ' |',
                      '| ' + ' | '.join(['---'] * len(headers)) + ' |'] +
                     ['| ' + ' | '.join(map(cell, r)) + ' |' for r in rows]) + '\n'


def fmt(x): return 'NA' if x is None else f'{x:.3f}'


def flatten(data, prefix=''):
    out = {}
    for k, v in data.items():
        name = prefix + str(k)
        if isinstance(v, dict): out.update(flatten(v, name + '/'))
        elif isinstance(v, list): out[name] = json.dumps(v, ensure_ascii=False, separators=(',', ':'))
        else: out[name] = v
    return out


def prepare(args):
    run, repo, public = args.run_dir, args.repo, args.public_root
    history = run / 'report_artifact_history'
    derivatives = []

    def copy(src, target=None, mode='JSONsafe'):
        src = Path(src); target = target or src.name
        if mode == 'byte_exact': data = src.read_bytes()
        else: data, _ = OLD.derivative_bytes(src, 'summary' if mode == 'JSONsafe' else 'csv')
        dest = public / target; put(dest, data, history)
        derivatives.append(dict(original_path=str(src.resolve()), original_sha256=OLD.sha(src),
            public_path=str(target), public_sha256=OLD.sha(dest), transform=mode))

    for src in sorted((run / 'evaluation_v1').iterdir()):
        if str(src).endswith(('.csv', '.csv.gz')) or src.name == 'SELECTED_POSES.jsonl.gz':
            copy(src, mode='byte_exact')
    for name in ('EVALUATION.json', 'EVALUATION_SEAL.json', 'REFERENCE_OPEN_RECEIPT.json'):
        copy(run / 'evaluation_v1' / name)
    roots = ('INPUT_MANIFEST.json', 'EXPERIMENT_PLAN.json', 'B_FEATURE_DEPENDENCY.json',
        'NO_B_TRAINING_MANIFEST.json', 'NO_B_TRAINING_CLOSURE.json',
        'NO_B_TRAINING_COST_AND_EXECUTION.json', 'NO_B_LOGGER_TECHNICAL_FAILURE_V1.json',
        'NO_B_WRITER_TECHNICAL_FAILURE_V2.json', 'NO_B_OPERATIONAL_REPAIR_V3.json',
        'NO_B_FULL_MOCK_FIT_CONTRACT_V3.json', 'POLICY_SPLIT_LOCK_v1.json',
        'POLICY_SPLIT_LOCK_v2.json', 'POLICY_SPLIT_TECHNICAL_AMENDMENT.json',
        'POLICY_CALIBRATION.json', 'POLICY_CONFIRMATION.json', 'AUGMENTATION_GATE.json',
        'P4_AUGMENTATION_GATE.json', 'PREDICTION_MANIFEST.json', 'PREDICTION_GUARD.json',
        'FIGURE_DATA_INDEPENDENT_REVIEW.json', 'B_P2_PRESENTATION_REPAIR_PARITY.json',
        'B_MASK_SEMANTIC_SOURCE_REVIEW.json', 'FIGURE_PRESENTATION_REPAIR_V2.json',
        'EXTREME_PRESENTATION_PLAN.json', 'EXTREME_PRESENTATION_RECEIPT.json',
        'FINAL_DECISION_REVIEW.json',
        'INSTRUCTION_COMPLETION_AUDIT.json', 'INSTRUCTION_COMPLETION_AUDIT_POSTREPAIR.json',
        'ARCHIVED_H1_FALLBACK_TECHNICAL_AMENDMENT.json',
        'ARCHIVED_H1_FALLBACK_PUBLIC_SOURCE_BINDING.json',
        'PUBLIC_PACK_STREAMING_AMENDMENT_V2.json', 'PUBLIC_PACK_REFERENCE_BINDING_AMENDMENT_V3.json',
        'PUBLIC_PACK_BASELINE_POINTER_AMENDMENT_V4.json', 'PUBLIC_ROOT_README_POINTER_SCOPE_PROOF.json',
        'PUBLIC_PACK_SCALAR_COUNT_AMENDMENT_V5.json',
        'PUBLIC_PACK_COVERAGE_FIXTURE_INITIAL_FAILURE.json',
        'REMOTE_PUBLICATION_HELPER_STATIC_REVIEW_B_V2.json')
    optional = {'FINAL_DECISION_REVIEW.json', 'ORIGINAL_PRESERVATION_REVIEW.json',
        'INSTRUCTION_COMPLETION_AUDIT.json', 'INSTRUCTION_COMPLETION_AUDIT_POSTREPAIR.json',
        'PUBLIC_PACK_BASELINE_POINTER_AMENDMENT_V4.json'}
    for name in roots:
        if (run / name).exists(): copy(run / name)
        elif name not in optional: raise FileNotFoundError(name)
    preservation_source = run / 'ORIGINAL_PRESERVATION_REVIEW.json'
    if preservation_source.exists():
        preservation = OLD.read(preservation_source)
        for key in ('original', 'new_worktree'):
            preservation[key].pop('path', None)
        preservation['original_full_private_receipt_sha256'] = OLD.sha(preservation_source)
        preservation['public_projection'] = 'Only two machine-specific root path strings omitted; all checks and failure evidence retained. Pre-pointer snapshot, final publication scope checked separately.'
        projected = run / 'PUBLISHABLE_ORIGINAL_PRESERVATION_REVIEW.json'
        put(projected, encoded(preservation), history)
        copy(projected, 'ORIGINAL_PRESERVATION_REVIEW.json')
    for seed in SEEDS:
        for name in ('HISTORY.json', 'RUN_RECEIPT.json'):
            copy(run / 'training' / f'H1_NO_B_FEATURE_s{seed}' / name,
                 f'training/s{seed}_{name}')
    for src in sorted((run / 'failed_attempts').glob('*/FAILED_TECHNICAL.json')):
        copy(src, 'training/' + src.parent.name + '_FAILED_TECHNICAL.json')
    for directory, target in (
        ('independent_no_b_v1', 'INDEPENDENT_NO_B.json'),
        ('independent_training_v1', 'INDEPENDENT_TRAINING.json'),
        ('independent_policy_v1', 'INDEPENDENT_POLICY.json'),
        ('independent_b_query_v1', 'INDEPENDENT_B_QUERY.json'),
        ('independent_evaluation_v1', 'INDEPENDENT_EVALUATION.json')):
        copy(run / directory / 'VERIFICATION.json', target)
    for directory, name in (
        ('b_query_factorial_v1', 'B_QUERY_FACTORIAL.csv.gz'),
        ('b_original_masks_v1', 'B_MASK_QUERY_ROWS.csv.gz'),
        ('b_original_mask_bounds_v1', 'B_MASK_TRANSITION_BOUNDS.csv.gz')):
        copy(run / directory / name, mode='byte_exact')
    for directory, name in (
        ('synthetic_coverage_v3', 'SYNTHETIC_COVERAGE.json'),
        ('procedural_shape_control_v1', 'PROCEDURAL_SHAPE_CONTROL.json'),
        ('procedural_shape_control_v1', 'PROCEDURAL_RENDER_SEAL.json'),
        ('procedural_shape_plan_v1', 'CONFIG.json')):
        copy(run / directory / name, 'PROCEDURAL_RENDER_CONFIG.json' if name == 'CONFIG.json' else name)

    # Flatten already-sealed descriptive rows. This is no generation/resampling.
    rows = [flatten(r) for r in OLD.rows(run / 'synthetic_coverage_v3/SYNTHETIC_COVERAGE_ROWS.jsonl.gz')]
    fields = sorted(set().union(*(set(r) for r in rows)))
    output = io.StringIO(newline=''); writer = csv.DictWriter(output, fieldnames=fields)
    writer.writeheader(); writer.writerows(rows)
    src = run / 'SYNTHETIC_COVERAGE.csv'; put(src, output.getvalue(), history)
    copy(src, mode='byte_exact')

    reviews = []
    review_sources = []
    for name, key, pathkey in (
        ('ROOT_PIXEL_REVIEW.json', 'records', 'path'),
        ('FIGURE_PIXEL_REVIEW.json', 'records', 'file'),
        ('PRIVACY_PIXEL_REVIEW.json', 'reviewed_images', 'relative_public_file'),
        ('B_P2_PIXEL_REVIEW.json', 'reviewed', 'file')):
        src = run / name; data = OLD.read(src)
        review_sources.append(dict(basename=name, sha256=OLD.sha(src), status=data['status']))
        for record in data[key]:
            rel = record[pathkey]
            prefix = f'_docs/experiments/{NS}/'
            if rel.startswith(prefix): rel = rel[len(prefix):]
            if OLD.sha(public / rel) != record['sha256']: raise ValueError('Reviewed raster changed: ' + rel)
            reviews.append(dict(file=rel, sha256=record['sha256'], actual_pixels_viewed=True,
                                review_source=name, reviewer_kind='model_visual_review_not_human'))
        copy(src, 'reviews/' + name)
    if len(reviews) != 36 or len({r['file'] for r in reviews}) != 36: raise ValueError('Actual review scope changed')
    rasters = [p for p in public.rglob('*') if p.suffix.lower() in ('.png', '.jpg', '.jpeg')]
    pixel = dict(schema='OBSERVATION_H1_ACTUAL_PIXEL_REVIEW_V1', status='PASS_FOR_RECORDED_SCOPE',
        public_raster_count=len(rasters), actually_opened_public_raster_count=len(reviews),
        reviewed_all_public_rasters=False, human_review=False, physical_correctness_certified=False,
        full_privacy_certification=False, whole_visual_review_wall_seconds=None,
        timing_note='Receipt/hash join timings do not measure the full visual review.',
        sources=review_sources, records=reviews,
        private_synthetic_source_views='Separate private receipt; no original synthetic RGB published.')
    put(run / 'PIXEL_REVIEW.json', encoded(pixel), history); copy(run / 'PIXEL_REVIEW.json')
    rights = dict(schema='OBSERVATION_H1_PUBLIC_RIGHTS_V1',
        primary192_derived_RGB_authorized=True,
        authorization='Inherited explicit owner confirmation: self-photographed evaluation images; result panels may be public.',
        raw_RGB_publish=False, raw_synthetic_RGB_publish=False, measurement_changed=False,
        previous_public_rights_sha256=OLD.sha(repo / '_docs/experiments/pallet_risk_information_v1/RIGHTS_AND_PRIVACY.json'),
        gallery_manifest_sha256=OLD.sha(public / 'gallery/GALLERY_MANIFEST.json'),
        privacy_rectangles_current_frame_level=26,
        original_plan_top_level_count=22,
        privacy_note='Four existing frame-level supplements are retained; original plan is not rewritten.',
        independent_join_receipt_sha256=OLD.sha(run / 'PRIVACY_PIXEL_REVIEW.json'),
        procedural_figures='Own primitive geometry/software renders, no external RGB asset.',
        synthetic_asset_rights='Not universally established. Original synthetic RGB remains private.')
    put(run / 'RIGHTS_AND_PRIVACY.json', encoded(rights), history); copy(run / 'RIGHTS_AND_PRIVACY.json')

    execution = []
    for src in sorted((run / 'execution').glob('*/RECEIPT.json')):
        receipt = OLD.read(src)
        execution.append(dict(name=src.parent.name, receipt_sha256=OLD.sha(src),
            status=receipt.get('status'), exit_code=receipt.get('actual_exit_code'),
            wall_seconds=receipt.get('wall_seconds'),
            CPU_user_seconds=receipt.get('child_CPU_user_seconds'),
            CPU_system_seconds=receipt.get('child_CPU_system_seconds'),
            peak_rss_kib=receipt.get('child_peak_rss_kib'),
            scope='Whole recorded child CLI; nested diagnostics must not be added again.'))
    cost = dict(schema='OBSERVATION_H1_COST_EXECUTION_V1', measured_child_CLIs=execution,
        optimizer_attempts=5, successful_fits=3, failed_attempts=2,
        previous_D1_producer_reused=True, previous_18_fits_reused=True,
        previous_producer_incremental_rerun_seconds=0,
        previous_producer_original_cost_seconds=None,
        primary_new_pose_solves=0, primary_new_candidate_generation=0,
        new_backbone_or_SAM_forwards=0, new_primitive_control_renders=4,
        GPU_training_jobs=0, uninstrumented_source_editing_seconds=None,
        full_visual_review_seconds=None,
        interpretation='CPU whole-process wall is not GPU inference latency or Jetson FPS. No combined nested-cost sum.')
    put(run / 'COST_AND_EXECUTION.json', encoded(cost), history); copy(run / 'COST_AND_EXECUTION.json', 'EXECUTION_AND_COST.json')
    portable = []
    replacements = sorted(((str(public), '${PUBLIC_ROOT}'), (str(repo), '${REPO_ROOT}'),
        (str(run), '${RUN_DIR}'),
        (str(args.risk_run), '${RISK_RUN}'), (str(args.d1_run), '${D1_RUN}')),
        key=lambda pair: -len(pair[0]))
    for src in sorted((run / 'execution').glob('*/RECEIPT.json')):
        receipt = OLD.read(src); argv = list(receipt.get('command', []))
        for i, value in enumerate(argv):
            if i == 0 and Path(value).name.startswith('python'): value = '${PYTHON}'
            else:
                for original, variable in replacements:
                    if value == original or value.startswith(original + '/'):
                        value = variable + value[len(original):]; break
            argv[i] = value
        portable.append(dict(name=src.parent.name, raw_private_receipt_sha256=OLD.sha(src),
            argv=argv, exit_code=receipt.get('actual_exit_code'), wall_seconds=receipt.get('wall_seconds')))
    commands = dict(schema='PORTABLE_RECORDED_COMMAND_PROJECTION_V1',
        interpretation='Actual recorded argv with machine roots replaced by named variables; failed attempts are preserved. Other private input paths are sanitized and require manifest matching.',
        commands=portable)
    put(run / 'RECORDED_COMMANDS_PUBLIC_INPUT.json', encoded(commands), history)
    copy(run / 'RECORDED_COMMANDS_PUBLIC_INPUT.json', 'RECORDED_COMMANDS.json')

    audits = []
    for path in public.glob('INDEPENDENT_*.json'):
        data = OLD.read(path)
        audits.append(dict(file=path.name, sha256=OLD.sha(path), status=data.get('status'),
                           evidence=data))
    verification = dict(schema='OBSERVATION_H1_VERIFICATION_LEVELS_V1', numeric_audits=audits,
        figure_data_audit_sha256=OLD.sha(run / 'FIGURE_DATA_INDEPENDENT_REVIEW.json'),
        actual_pixel_review_sha256=OLD.sha(public / 'PIXEL_REVIEW.json'),
        physical_target_edge_ownership_certified=False,
        automatic_typed_B_supplier_implemented=False,
        independent_new_real_test_n=0,
        remote_publication='Not inferred from these checks. A separate receipt is written only after actual remote verification.')
    put(run / 'VERIFICATION.json', encoded(verification), history); copy(run / 'VERIFICATION.json')

    evaluate = OLD.read(run / 'evaluation_v1/EVALUATION.json')
    comparisons = evaluate['populations']
    main_arms = ['N0', 'E0', 'POSEFIX_SUBPIX_PF_ANCHOR', 'POSEFIX_SUBPIX_YOLO_ANCHOR']
    for prefix in ('D1_H1_MLP', 'H1_NO_B_FEATURE', 'R_CONT_F0', 'H1_GUARDED_SWITCH'):
        main_arms.extend(f'{prefix}_s{s}' for s in SEEDS)
    header = ['경로', '성공/전체', '위치 평균±SD cm', '위치 분산 cm²', '회전 평균±SD °', '회전 분산 °²', 'T>50/T>100/R>45', '최대 cm/°']

    def metric_table(pop, methods):
        result = []
        for name in methods:
            m = comparisons[pop][name]; t, r = m['translation_cm'], m['rotation_deg']
            result.append([name, f"{m['successes']}/{m['denominator']}",
                fmt(t['mean']) + '±' + fmt(t['sample_std']), fmt(t['sample_variance']),
                fmt(r['mean']) + '±' + fmt(r['sample_std']), fmt(r['sample_variance']),
                f"{m['T_gt50cm']}/{m['T_gt100cm']}/{m['R_gt45deg']}", fmt(t['max']) + '/' + fmt(r['max'])])
        return table(header, result)

    full = ['# 전체 결과 — 고정 26경로 × 네 모집단\n',
        '제외·절삭 0장. 성공은 동일 자세에서 위치 ≤5cm AND C2 회전 ≤5°. SD·분산은 표본값(ddof=1)이다. '
        '누락 자세도 분모에 남기며 이번 출력의 no-pose는 0이다. P90/P95/median과 정확한 원수치는 '
        '[CSV](COMPARISON_ALL_ARMS.csv)에 있다. CAL46은 별도 기술 분석이며 선택에 쓰지 않았다.\n']
    for pop in comparisons:
        full += [f'## {pop}\n', metric_table(pop, evaluate['arms'])]
    put(public / 'FULL_RESULTS.md', '\n'.join(full), history)

    paired_rows = []
    for pop in comparisons:
        for prefix in ('H1_NO_B_FEATURE', 'H1_GUARDED_SWITCH', 'H1_GUARD_CALIBRATED_TRIAL'):
            for seed in SEEDS:
                key = f'D1_H1_MLP_s{seed}__TO__{prefix}_s{seed}'
                p = evaluate['paired'][pop][key]
                d = p['deltas']; cont = p['both_failure_continuous']
                ci = d['success_rate']['recording_cluster_CI']['CI95']
                paired_rows.append([pop, f'{prefix}_s{seed}', p['denominator'], p['counts']['RECOVERY'],
                    p['counts']['DAMAGE'], p['success_difference_count'], fmt(d['translation_cm']['mean']),
                    fmt(d['rotation_deg']['mean']), cont['denominator'],
                    fmt(cont['translation_cm']['mean']), fmt(cont['rotation_deg']['mean']),
                    '[' + ', '.join(fmt(x * 100) for x in ci) + ']'])
    put(public / 'PAIRED_RESULTS.md', '# 같은 seed H1 대비 짝지은 변화\n\n' +
        '변화는 after−before이며 음의 오차 변화가 개선이다. 마지막 열은 원 촬영 단위 cluster bootstrap 성공률 차이의 95% CI(%p)다. '
        'PRIMARY의 원 촬영 6단위 및 이름 기반 8세션 민감도 분석을 [EVALUATION](EVALUATION.json)에 모두 보존했다. '
        '3seed를 576장 독립 표본으로 세지 않았다. trial은 내부 확인에 실패한 미채택 정책까지 보존한 진단이다.\n\n' +
        table(['모집단', 'after (before=같은 seed H1)', 'N', '회복', '손상', '성공 증감', 'ΔT cm', 'ΔR °',
               '공통 실패 N', '공통 실패 ΔT', '공통 실패 ΔR', '성공률 CI %p'], paired_rows), history)

    report = '''# Observation contract / H1 guarded switch — 실행 결과

B 유래 특징 제거와 이번 제한적 전환 정책은 채택하지 않는다. 기존 H1, 상태별 처리, N3·SubPix, 과거 비교 결과·운영 기본값·논문은 보존했다. 성공을 잃으면서 일부 큰 오차만 낮추는 교환이 발생했으며, 나쁜 seed도 남겼다.

[전체 104행 결과](FULL_RESULTS.md) · [짝지은 비교](PAIRED_RESULTS.md) · [192장 병렬 갤러리](gallery/index.html) · [모든 새 seed 최대 오차](gallery/extremes/index.html) · [관측 계약 상세](OBSERVATION_CONTRACT.md) · [재현 안내](REPRODUCE.md)

## 실행과 입력 계약

기준 origin/main f68988e9bfabb9437da7e71e1c16b39bf6308a1d에서 별도 worktree로 실행했다. 원 사용자 저장소의 dirty 파일·index·local main을 건드리지 않는다. 이전 D1 전체 생산, 18회 fit, 고정 후보/관측/참조를 SHA로 결합했다. 이전146,135/2,142 head replay의 최대 logit 차이0 증거도 연결했다. 이전 캐시 재사용은 원 생산 비용0이나 새 독립 검증 세트라는 뜻이 아니다. [입력](INPUT_MANIFEST.json), [사전 계획](EXPERIMENT_PLAN.json), [원 출력 봉인](PREDICTION_MANIFEST.json)을 보존한다.

PRIMARY192·TEST49·REMAINDER143·별도CAL46 전부 평가했다. 새 실사 PnP·후보 생성·정답 수정·새 backbone/SAM forward는 0회다. 26경로×238장 예측 출력을 먼저 봉인한 후 실사 참조를 열었다. C2(I, Yπ) 대칭을 적용했고 C4나 GT 정규직교화는 사용하지 않았다. 성공은 위치≤5cm와 회전≤5°를 같은 자세에서 동시에 만족해야 한다. outlier를 제외하지 않았다. 독립 새 실사 test는0장이다.

## P1: B 특징 제거

원래188→64→64→1, 16,321파라미터 selector에서 B 유래 scalar12개와 대응 validity를 local 단계에서 0으로 만들고 mean/max pooling 정보도 차단했다. 최종188차원 중72개가0, 나머지116개는 bitwise 일치한다. 행동 bank, eligibility, alias, 순서, dedup, N0 표시는 그대로다. A/full/partial/outline의 다른 정보는 남기므로 ‘B producer 전체 삭제’ 실험이 아니다.

원래 seed20261018/19/20, 768/235 합성 분할·정규화·손실·optimizer를 유지해 3개 fit을 완료했다. logger reserved-key 오류(0step), 저장 API 인자 오류(24step)가 발생했다. 두 실패와 부분 checkpoint를 private에 보존했고 운영 코드만 고쳤다. 실제 optimizer 시도는5회(완료3+실패2), 성공1128step+실패24step이다. 고정 과학 설정을 재탐색하지 않았다. 두 기술 실패 후 최종3fit 재개 전에 full training 흐름을 MockOptimizer로 검사했다. 원 학습의 archival 초기 tensor/order bytes는 미보관이라 원 소스 recipe를 재구성한 동등성까지 확인했으며 역사적 byte 동일성으로 과장하지 않는다.

같은 seed H1의 성공120/117/121이 no-B113/113/116으로 감소했다. 회복/손상은3/10,3/7,3/8이다. 위치 평균은 두 seed에서 좋아지고 마지막은 나빠진다. 회전도 모든 seed에서 개선되지 않는다. seed19에서는 wood_night_01:031426가176.54→208.22cm로 악화했다. B 관측에는 오류가 있지만 고정 bank 내 성공 후보를 구별하는 정보도 있어, 단순 제거가 성공 보존과 위험 감소를 함께 달성하지 못한다. 이 결과로 B 관측 전체의 필수성이나 정보누출 부재를 증명하지는 않는다.

## P2: 관측이 무엇을 지지하는가

앵커/방향2×2는 GT 진단용이다. 원1,003장 중 B 존재673/부재330, 두 대응 family와4질의를 전부 보존했다. ORIGINAL_WD의 어느 전체 phase라도 유한 지지를 갖는 frame은 Q00 377/673, GT앵커만 Q10 456(회복91/손상12), GT방향만 Q01 370(6/13), 둘다 Q11 643(266/0)이었다. 원 profile·along·관측 energy를 새 RGB 위치로 운반하거나 바꾸지 않았다. 자동 자세 정확도 개선치가 아니다. [전체139,976행](B_QUERY_FACTORIAL.csv.gz)과 [다층 집계](B_QUERY_AGGREGATES.csv)를 제공한다.

원 visible/amodal/SAM mask와 canvas 연결은1,003장 전부 확인했다. 원 renderer PNG의 넓은 배경에도 값1이 있어 중간값을 단순 AA 또는 인증된 target 확률로 해석할 수 없다. 0/255를 배경/target으로 해석한다는 미인증 계약 아래 Q00의17,497개 선 중4,634개는 최소 두 전환이 필요하고12,863개는 미결정이다. 물리 소속·encoding/compositor 복구는 미완료다. 원 complete scene/generator/asset library가 없어 exact scene replay는0회다.

자체 primitive 4장면은 같은 외곽 cuboid에서도 구멍·다른 물체 가림에 따라 전환 범위가 [3,3]/[2,4]/[1,1]/[2,2]가 될 수 있음을 보인다. 원 mesh 재현이나 실제 자동 성능 증거는 아니다. typed evaluator는 실루엣, 유한 edge 지지, 알려진 projection edge를 구분하며 candidate에 고정된 edge binding이 없으면 제약을 주지 않는다. 이 binding도 독립적인 물리 소속 인증은 아니다. 실제 typed 공급기와 실사 보정은 아직 구현하지 않았다.

## P3: H1 제한적 전환

H1 argmax 행동 h와 같은 seed R_CONT minimax 행동 r을 그대로 사용한다. 서로 다른 자세이며 연속 위험 이득>0이고 H1 score gap≤τ일 때만 r로 바꾼다. 같은 자세이면 H1을 유지한다. 현재 실사 입력은 모두 유한하다. GT·실패집단·영상 ID는 전환 입력에 쓰지 않았다.

최종 감사에서 원 adapter가 누락·비유한 score를 처리할 때 정책 함수의 archived H1 index를 전달하지 않는 공백을 발견했다. 봉인 후 기술 수리로 실제 보관 H1 자세를 동일 bank에 exact binding해 연결했고, 알려진 H1이 있으면 누락·잘못된 score/대안에도 그 자세를 유지한다. H1 자체가 없으면 없음을 보존하며 N0를 만들어내지 않고, 유효 score와 원 행동/자세가 모순되면 계속 거부한다. 기존 원 실행 source를 별도로 게시하고 238장×3seed×active/trial의1,428개 whole 반환 dict가 원 adapter·수리 adapter 모두 봉인 출력과 완전히 같음을 확인했다. 원 예측/평가·정책·문턱 bytes를 변경하지 않았고 새 forward/GT열람/자세풀이도 없다. [수리·동등성 근거](ARCHIVED_H1_FALLBACK_TECHNICAL_AMENDMENT.json)를 따로 남긴다.

원 합성235장을 생성 run으로 묶어 label-free hash순서에서153개 development/82개 confirmation으로 나눴다(12/11 run, asset1개 공유). 초기 hashmod split203/32는 label-free 상태에서 수정했고 원 lock과 amendment를 보존했다. development의 유한 gap 전량과 -∞만 검사하고, 성공 비감소·양 평균/큰오차 비증가 조건을 통과하는 τ 중 사전 순서로 선택했다. confirmation은 한 번만 평가하고 실패하면 τ=-∞, 즉 H1으로 복귀한다. 독립 certification은 아니다.

seed18 τ=0.6228189468만 내부 확인을 통과했다. seed19/20은 회전 평균 비증가 조건을 실패해 H1 identity로 복귀했다. seed18 실사192장에서는120→116성공(회복2/손상6), 위치평균12.073→11.385cm, 회전11.549→10.246°다. seed19 실패 trial이 실사에서117→119여도 실사 결과로 되살리지 않았다. seed20 trial도121→120이었다. identity 복귀 두 개를 새로운 안전 성능 향상으로 세지 않는다.

## P4: 조건부 증강 미실행

원1,003장과40,000원 provenance를 연결해768train/235validation coverage 및 N0/E0/전체 실패의 연속 Pareto 관계를 집계했다. train 전체 실패416/그중 Pareto215, validation171/104이며 단순 all-failure라는 이유로 관계가 모두 사라지는 것은 아니다. validation의 선언된 주변 visibility/elevation/azimuth/scene/occluder bin은 train에 존재했다. 인과적인 생성 가능 누락 조건을 확인하지 못했으며 조합 coverage가 완전함을 증명한 것은 아니다.

원 scene/generator 부재와 함께, 실제 fit 시도5회로 최대9회 중4회만 남아 동일량 단순추가3seed+구조추가3seed의6회를 충족하지 못한다. 새 증강frame0/fit0, 비대칭 부분 비교나 좋은 결과를 위한 resampling을 하지 않았다. [게이트](AUGMENTATION_GATE.json)에 미실행 사유를 모두 보존했다.

## PRIMARY192 요약

'''
    report += metric_table('PRIMARY192', main_arms)
    report += '''
전체 bank에는 성공 가능151장/모든 후보 실패41장, N0/E0 pair에는133/59장이 있었다. 후보 부재41장에서는 selector 변경만으로 성공을 만들 수 없다. 공통 실패의 연속 오차와 새로운 큰오차 발생/해소도 [짝지은 표](PAIRED_RESULTS.md), [고정 실패집단 위험](ALL_FAILURE_RISK.csv), [모든 tail 전환](TAIL_TRANSITIONS.csv.gz)에 남겼다. seed18 no-B 공통실패69장의 평균ΔT=-2.612cm/ΔR=-2.458°여도 개별 최대 악화ΔT=34.119cm/ΔR=80.518°다. guarded18 공통실패70장은 평균ΔT=-1.642cm/ΔR=-3.664°이며 평균 개선을 모든 영상의 안전성으로 해석하지 않는다.

## 검산·픽셀·물리 대응·비용

독립 입력/가중치 mutation 검산13,653, 학습이력130, 정책27,472, B 질의1,439,190, 전체 수치3,609,607, 그림 원수치57,020개 검사에서 실패0이었다. 원238×26 selected whole pose와12개 CSV를 독립 재집계했다. 부동소수점 안정식 차이를 따로 보고하고 성공 경계·원 strict delta 부호는 바꾸지 않았다. [검산](VERIFICATION.json)은 수치, [픽셀 검토](PIXEL_REVIEW.json)는 실제 열어 본36/216개의 모델 시각 검토, 물리 edge 대응은 미인증으로 구분한다. 자동 image decode/링크 점검은 픽셀을 보았다는 뜻이 아니다. 사람의 검토·독립적인 실제 소속 인증·완전한 개인정보 인증을 주장하지 않는다. HTML은 생성/링크 점검 대상이며 브라우저에서 열렸다고 주장하지 않는다.

3개 학습 CLI의 중복 없는 전체 wall은160.779초(실패 포함), peak RSS720,132KiB였다. 새 후보를 푸는 실험이 아니어서 Jetson FPS 개선이나 전체 RGB 파이프라인 비용을 측정한 것으로 해석하지 않는다. 원 D1 재실행 비용0은 원 생산 비용0이 아니다. B2×2 75.921초, 원 mask43.212초, 조건부 DP94.374초, 정책 score7.009초·선택1.107초, 예측20.732초, 실사 평가13.681초, 주 갤러리98.467초를 별도 whole CLI로 기록했다. 도구 밖 편집·전체 모델 시각 검토 시간은 미계측이다. [실제 실행 원장](EXECUTION_AND_COST.json)을 제공하며 nested fit/epoch 시간을 다시 합산하지 않았다.

## 다음에 고칠 한 가지

B의 질의 앵커·검색 영역을 실제 target의 유한 물리 edge 대응 계약에 연결하는 작업을 우선한다. 근거는 GT앵커 대입의 큰 회복, 방향만 바꿀 때의 손상, 구멍/다른물체에 의한 복수 전환, 미인증 mask 소속이다. 다음 실험은 이 계약을 자동 관측에서 검증 가능하게 만들고, 구분 불가한 경우에는 검색 범위/불확실성을 유지해야 한다. 현재 실사 결과를 보고 문턱·가중치·seed를 재탐색하거나 no-B/guard를 운영 기본값으로 바꾸지 않는다. 이 작업은 다음 수정 제안이며 본 과제에서 자동 보정 완성을 주장하지 않는다.
'''
    report += '\n별도 CAL46에서도 guard18은39성공을 유지하지만 위치평균2.551→2.614cm, 회전3.445→5.346° 및 R>45° 1→2로 악화했다. no-B18/20은 CAL 성공을2/1장 잃었다. CAL 결과도 문턱·seed 선택에 사용하지 않았다.\n'
    put(public / 'REPORT_KO.md', report, history)
    put(public / 'DECISION_KO.md', '''# 채택 판단

이번 no-B와 H1 guarded switch는 채택하지 않는다. 기존 H1·상태별 처리·N3·SubPix·운영 기본값과 논문을 유지한다.

- no-B: PRIMARY192에서 같은 seed 성공120→113,117→113,121→116, 회복/손상3/10·3/7·3/8. seed19 최대 위치208.22cm를 포함해 모두 보존한다.
- guard: 내부 확인을 통과한 seed18도120→116, 회복2/손상6. 19/20은 확인 실패로 H1에 복귀했다. 미채택 trial의 실사 결과를 근거로 다시 선택하지 않는다.
- P2: 앵커 개선의 GT 진단은 유효하지만 자동 실제 성능이나 물리 소속 인증이 아니다. 원 mask encoding/complete scene/자동 typed 공급기에는 공백이 남는다.
- P4: 생성 가능한 인과적 coverage 공백과 원 입력이 미확정이며, 실제5시도 후 남은4회로 matched6fit을 실행할 수 없어 미실행했다.

다음 수정 한 가지는 **B 질의 앵커·검색 영역을 인증 가능한 target 유한 edge 대응 계약에 연결**하는 것이다. 원 관측이 실제 edge인지/다른 물체인지/복수 전환인지 모르면 위치 범위와 불확실성을 유지한다. 이번 실사로 가중치·문턱·seed를 재탐색하지 않는다.

[전체 근거](REPORT_KO.md) · [네 모집단 전체 표](FULL_RESULTS.md) · [원수치](COMPARISON_ALL_ARMS.csv) · [나쁜 seed 이미지](gallery/extremes/index.html)
''', history)
    put(public / 'RUN_STATUS.json', encoded(dict(schema='OBSERVATION_H1_RUN_STATUS_V1',
        science='COMPLETE_WITH_DOCUMENTED_CONDITIONAL_NOT_RUN',
        P0='EXACT_INPUT_BINDING_PASS', P1='THREE_COMPLETED_FITS_FIVE_ACTUAL_ATTEMPTS',
        P2='GEOMETRY_MASK_PROCEDURAL_TYPED_DIAGNOSTICS_COMPLETE_PHYSICAL_CONTRACT_UNCERTIFIED',
        P3='THREE_SEEDS_CALIBRATED_ONCE_CONFIRMED_SEED18_ACTIVE_SEED19_20_H1_FALLBACK',
        P4='NOT_RUN_INPUT_CAUSAL_COVERAGE_AND_MATCHED_BUDGET_GATES',
        populations={'PRIMARY192':192,'TEST49':49,'REMAINDER143':143,'CAL46':46},
        evaluated_arms=26, original_defaults_and_paper_unchanged=True,
        publication='Packaging and actual remote state are reported by separate audit and receipt.')), history)
    reproduce = '''# 재현 안내

이 디렉터리는 공개 가능한 작은 파생 결과와 원소스 SHA를 담는다. private RGB·전체 후보 bank·GT·weight 파일은 복사하지 않는다. [입력 manifest](INPUT_MANIFEST.json)의 SHA가 맞는 승인된 원 캐시가 있어야 end-to-end 재현할 수 있다. 공개 자료만으로 원 학습/실사 생산을 모두 실행할 수 있다고 주장하지 않는다.

아래 명령은 저장소 root에서 실행한다. 경로는 실행자가 실제 private 입력 위치로 설정한다. 실제 전체 CLI와 종료코드는 [원 실행 명령의 경로 치환본](RECORDED_COMMANDS.json)에 있다. 변수 PYTHON/RUN_DIR/REPO_ROOT/PUBLIC_ROOT/RISK_RUN/D1_RUN과 manifest로 대응한 나머지 private 입력을 제공해야 한다. 재현은 기존 봉인 출력을 변경하지 않는 별도 run 위치에서 수행한다. AdamW를 다시 실행하면 실제 새 fit이므로 단순 검산을 위해 학습 명령을 반복하지 않는다. 실패한 구버전 source와 raw trace는 private에 보존했으므로 현재 source를 실행해 같은 logger 오류를 일부러 재현하지 않는다.

```bash
python -B -m unittest discover -s scripts/research/pallet_observation_contract_h1_v1 -t . -p 'test_*.py'
python -B -m scripts.research.pallet_observation_contract_h1_v1.independent_evaluation --help
python -B -m scripts.research.pallet_observation_contract_h1_v1.independent_no_b --help
python -B -m scripts.research.pallet_observation_contract_h1_v1.independent_policy --help
python -B -m scripts.research.pallet_observation_contract_h1_v1.independent_b_query --help
python -B -m scripts.research.pallet_observation_contract_h1_v1.public_pack --help
```

source 역할은 `preflight.py` 입력 결합, `no_b.py/no_b_train.py` 특징 차단·고정 학습, `b_factorial_batch.py/b_mask_bounds.py/typed_observation.py` 관측 계약 진단, `policy.py/coverage.py` 고정 정책과 합성 분할, `predict.py/evaluate.py` 봉인·평가, `independent_*.py` 별도 계산, `figures.py/gallery.py/gallery_extremes.py` 사후 표시, `report_artifacts.py/public_pack.py` 공개 패키지로 나뉜다. 실제 source 목록·해시는 공개 pack manifest를 기준으로 한다. 구체적인 CLI 인자는 각 모듈 `--help`와 sanitization된 실행 receipt를 참고한다.

`verify_remote_publication.py`는 실제 fetch/ls-remote, 결과 커밋 포함, 두 새 namespace의 전체 파일 집합 및 모든 blob bytes를 manifest SHA·검산된 현재 파일과 대조한다. commit/push를 수행하지 않는 별도 도구이며, offline pack 통과를 원격 검증으로 바꾸지 않는다. 공개 코드만으로 이 조회 로직을 확인할 수 있다.

원 수치를 생산한 `predict.py` bytes는 새 namespace의 `predict_at_prediction_seal.py`에 그대로 보존했다. 현재 `predict.py`에는 봉인 후 누락 점수 fallback 기술 수리가 있으며 actual finite1,428개 출력은 동일하다. 원 실행 source hash까지 재현할 때는 별도 재현 checkout에서 보관본을 `predict.py` 위치로 배치하고, 기존 봉인/원 사용자 checkout은 수정하지 않는다. [source 연결](ARCHIVED_H1_FALLBACK_PUBLIC_SOURCE_BINDING.json)과 [전체 출력 동등성](ARCHIVED_H1_FALLBACK_TECHNICAL_AMENDMENT.json)을 함께 확인한다.

실행 순서는 P0 byte/shape/source binding → 학습 전 계획 고정 → 3seed no-B(기술 실패 포함5시도) → B2×2/mask/shape/typed 진단 → label-free split → 합성 정책 선택·한 번 확인 → 조건부 증강 게이트 → 26경로 예측 봉인 → 실사 참조 open/평가 → 독립 수치 검산 → 그림/갤러리 생성·실제 픽셀 검토 → 공개 transform/링크/누출 검사 → ordinary main 게시다.

원 파일을 그대로 복사한 CSV/선택 자세와 JSONsafe 파생의 original/public SHA·변환 검사는 PUBLIC_PACK_MANIFEST.json에 있다. 게시 영수증 PUBLICATION_RECEIPT.json은 실제 원격 결과 확인 이후에만 추가한다. 영수증의 A는 내용 커밋이며 후속 metadata B 자체를 미리 검증했다고 주장하지 않는다.
'''
    put(public / 'REPRODUCE.md', reproduce, history)
    sources = '''# 방법 근거와 열람 범위

이론은 실사 안전성 보증으로 전용하지 않았다. 아래 공식 초록/출판 페이지를 실제 열람한 범위에서만 연결한다. 모든 PDF 전문을 읽었다고 주장하지 않는다.

- [Safe Policy Improvement with Baseline Bootstrapping, ICML2019](https://proceedings.mlr.press/v97/laroche19a.html): offline 정책 변경에서 baseline과 정보 부족을 분리한다는 배경. 현재 neural score-gap 정책에 논문의 보증이 자동 적용되지 않는다.
- [Regression with Multi-Expert Deferral, ICML2024](https://proceedings.mlr.press/v235/mao24d.html): 고정 예측기와 다른 전문가 사이의 regression deferral 배경이다. 논문의 일관성 결과를 본 합성 split의 실사 안전 보증으로 쓰지 않았다.
- [Two-Player Games for Efficient Non-Convex Constrained Optimization, ALT2019](https://proceedings.mlr.press/v98/cotter19a.html): 비볼록 제약 최적화의 배경. 논문의 게임 기반 최적화를 구현한 것은 아니며 유한 표본 확인과 실제 물리 안전성을 구분한다.
- [Structured Domain Randomization, NVIDIA 공식 페이지](https://research.nvidia.com/labs/amri/publication/prakash2019structured/): 생성 가능한 조건을 근거로 증강을 설계한다는 배경. 이번에는 필요한 인과 조건/원 scene 입력과 fit 예산이 미충족이다.

SRT3D·Blender mask encoding 관련 공식 출처와 실제 열람/미인증 범위는 [관측 계약 문서](OBSERVATION_CONTRACT.md)에 별도 기록했다. 특정 mask의 0/255 의미나 물리 edge 소속은 일반 문헌으로 인증할 수 없다.
'''
    put(public / 'METHOD_AND_SOURCES.md', sources, history)
    readme = '''# B 관측 계약 및 H1 제한적 변경 실험

**결론: 이번 no-B와 guard는 채택하지 않는다.** 기존 H1·N3·SubPix·상태별 처리·운영 기본값과 논문을 보존했다. 전체192장, 원49장, 추가143장과 별도CAL46, 3seed와 실패를 모두 포함한다.

- [한국어 보고서](REPORT_KO.md), [채택 판단](DECISION_KO.md), [26경로×4모집단 전체 표](FULL_RESULTS.md), [평균·분산·P90/P95 원CSV](COMPARISON_ALL_ARMS.csv), [짝지은 회복·손상](PAIRED_RESULTS.md)
- [192장 병렬 갤러리](gallery/index.html), [208.22cm와164.21° 모든 seed 극단 사례](gallery/extremes/index.html)
- [B 앵커/방향·mask·형상·typed 계약](OBSERVATION_CONTRACT.md), [합성 커버리지](SYNTHETIC_COVERAGE.json), [조건부 증강 미실행](AUGMENTATION_GATE.json)
- [재현 안내](REPRODUCE.md), [방법 근거](METHOD_AND_SOURCES.md), [계획](EXPERIMENT_PLAN.json), [입력](INPUT_MANIFEST.json), [검산 수준](VERIFICATION.json), [실제 픽셀 검토](PIXEL_REVIEW.json), [실행 비용](EXECUTION_AND_COST.json), [권리·표시용 가림](RIGHTS_AND_PRIVACY.json)

공개 pack의 파일/원본 해시는 PUBLIC_PACK_MANIFEST.json, 누출·경로·선택 자세 전량 검사 결과는 PUBLIC_PACK_AUDIT.json에 기록한다. 이 파일 자체는 원격 게시의 증거가 아니며 실제 원격 확인 이후 PUBLICATION_RECEIPT.json을 별도로 추가한다.
'''
    put(public / 'README.md', readme, history)
    # The original root README bytes are retained as a suffix, exactly.
    root = repo / 'readme.md'; original = run / 'ROOT_README_BASE_BYTES'
    if not original.exists(): put(original, root.read_bytes(), history)
    pointer = (f'## B 관측 계약·H1 제한적 변경 실험\n\n'
        f'[보고서](_docs/experiments/{NS}/REPORT_KO.md) · '
        f'[전체 표](_docs/experiments/{NS}/FULL_RESULTS.md) · '
        f'[192장 갤러리](_docs/experiments/{NS}/gallery/index.html)\n\n'
        'no-B/guard 미채택. 3seed·전체 분모·실패·조건부 미실행을 보존한 연구 기록입니다.\n\n').encode()
    put(root, pointer + original.read_bytes(), history)
    put(run / 'RAW_PUBLIC_DERIVATIVE_MAP.json', encoded(dict(derivatives=derivatives)), history)
    return dict(status='PREPARED_NOT_YET_PACK_VERIFIED', derivatives=len(derivatives),
        evaluation_rows=104, prediction_whole_poses=238 * 26, actual_reviewed_rasters=36,
        map_sha256=OLD.sha(run / 'RAW_PUBLIC_DERIVATIVE_MAP.json'))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--repo', type=Path, required=True)
    parser.add_argument('--public-root', type=Path, required=True)
    parser.add_argument('--risk-run', type=Path, required=True)
    parser.add_argument('--d1-run', type=Path, required=True)
    args = parser.parse_args(); print(json.dumps(prepare(args), sort_keys=True))


if __name__ == '__main__': main()
