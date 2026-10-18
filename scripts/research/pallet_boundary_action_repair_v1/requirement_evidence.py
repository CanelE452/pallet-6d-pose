"""Receipt-bound final requirement matrix; no reference or score decoding.

Only --requirements-GO permits reading real receipts and writing a new PRIVATE
matrix. --help/--catalog/--proof-template are preparation-only. An omitted proof
is PENDING, never a completion. Prior matrix bytes are archived before any
explicit in-place update. This tool never runs tests, Git, a model, or a solver.

Bindings in --proof-map use {"requirements": {ID: {"checks": [
  {"claim": KEY, "receipt": "RELATIVE.json", "pointer": "/field",
   "op": "eq", "expected": true}]}}}. Every required claim needs a successful
check of a real, hashed receipt. public: and source: prefixes resolve under the
explicit public/source roots. Optional expected_sha256 locks an input receipt.
Do not attest an unperformed check merely to fill a matrix.
"""
from __future__ import annotations
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import re
import shutil
import time

from .registry import ALL_ARMS, EXPECTED_N, L_GROUPS, SEEDS

AXES = ('INPUT_AND_VERSION', 'EXECUTION_COMPLETENESS', 'NUMERIC_REPRODUCTION',
        'GEOMETRIC_SUPPORT', 'LEARNING_ACTION_CONTRACT', 'VISUAL_AND_PUBLICATION')
FIGURES = ('joint_success.png', 'translation_cdf.png', 'rotation_cdf.png',
           'paired_recovery_damage.png', 'actual_action_sources.png',
           'same_bank_2x2_pair.png', 'training_curves_all_seeds.png')
RESULT_SNAPSHOT = 'publication_snapshots/result_A/'
REMOTE_RECEIPT = 'REMOTE_PUBLICATION_RECEIPT.json'
GIT_PROOF = 'PUBLICATION_GIT_PROOF.json'
REVIEW_PASS_STATES = frozenset(('PASS', 'REVIEWED', 'COMPLETED',
    'COMPLETED_MODEL_PIXEL_REVIEW', 'COMPLETED_HUMAN_PIXEL_REVIEW'))
B_KEYS = ('finite_vs_infinite', 'range_and_singular', 'far_shrink', 'infeasible_initial',
          'fixed_universe', 'no_B_input', 'derivatives', 'invariances',
          'status_contract', 'independent_geometry')
P0 = (
    ('작업 트리·프로세스·별도 worktree/branch·최신 main/c69f57e diff 기록', ('worktree_and_processes_checked', 'baseline_and_diff_recorded')),
    ('원 baseline/L checkpoint/분할/후보/RGB/mask 가용성·동일 해시 재사용', ('fixed_inputs_and_reuse_verified',)),
    ('192=49+143/CAL46 별도·합성768/235 동일 ID/asset·독립 test0', ('real_denominators', 'synthetic_split_unchanged', 'independent_real_test_zero')),
    ('config/seed/입력/손실/후보/fallback/모델선택/관측을 새 PRIMARY 전에 고정', ('plan_frozen', 'prediction_plan_bound', 'before_reference_scoring')),
    ('새 CPU/GPU/학습 실제 실행과 과거 재사용 비용 분리', ('actual_branch_costs_bound', 'historical_reuse_separate', 'unknown_cost_not_zero')),
    ('R/t/K/원점/Q/단위/raw-letterbox-crop 계약·합성 GT는 label/eval만 접근', ('geometry_coordinate_contract', 'reference_prediction_separation')),
    ('독립 checker는 생산 성공/집계 함수 import 없이 검산', ('independent_numeric_implementation',)),
)
B_TEXT = (
 '같은 무한 직선·다른 유한 선분에서 실제 지지만 인정',
 'λ범위/양끝점/내부/평행/영길이/depth/off-canvas 분리',
 '원거리 투영 축소가 연장선 우연 교차로 지지를 얻지 않음',
 '초기 pose도 지지 불가하면 임의 복구/정답삽입 없이 실패·보류',
 '고정 RGB/profile/consumed group 수를 후보 이동에 따라 변경하지 않음',
 '무입력 frame은 원 관측 정책 유지; 후보별 임의 누락과 구분',
 '유효 domain analytic/FD와 경계 one-sided 도함수',
 '해상도/Q/full-partial archive 철회/group duplicate 불변',
 'optimizer_success/feasible/finite_support/residual/fallback 별도 저장',
 '독립 코드로 선택된 전체 B 출력 교차·λ·길이·지원 재계산',
)
L_TEXT = (
 ('toy N0/다른후보 good/bad 네 경우·중복·N0미가용', ('toy_action_cases',)),
 ('multi-positive/all-zero 보존·다른 후보 위치/회전 레이블 혼합 금지', ('multi_positive_all_failure_same_pose',)),
 ('H1에 none alias로 동일 N0를 이중 반환하지 않음', ('no_H1_virtual_none_duplicate',)),
 ('candidate 순서/ID 문자열과 semantic 선택·동률 구분', ('semantic_order_ID_invariance',)),
 ('padding/ineligible/missing feature loss/pool·validity 분리', ('loss_pool_validity_contract',)),
 ('원 affine 공통항 상쇄·역할별 affine 동작', ('affine_cancellation_and_role_context',)),
 ('train/val joint/평균/꼬리/accuracy/Brier/source/good부재 진단', ('training_validation_diagnostics', 'legacy_metric_absence_honest')),
 ('같은 code로 D0/D1 state/source/missing/candidate/covariance/feature 분포', ('D0_D1_distribution_diagnostics',)),
 ('동일 새 real bank에서 모든2×2 checkpoint 평가 hash/ID', ('common_real_bank_bytes', 'all_checkpoints_and_IDs',)),
 ('추가 optimizer/seed/architecture 탐색 없이 새학습 최대18·실패 재시도 계수', ('new_training_cap', 'seeds_and_groups_fixed', 'all_attempts_preserved')),
)
PUB = (
 ('작업 전 status/branch/remote/main SHA 및 별도 worktree 기록', ('prework_git_state_recorded',)),
 ('실제 변경 allowlist로만 stage; 사용자 파일을 git add .로 묶지 않음', ('explicit_stage_allowlist',)),
 ('private path/secret/raw RGB/weights/assets/fonts/cache/EXIF 검사', ('public_content_and_privacy_scan',)),
 ('CSV/JSON/README/그림 상대링크·hash·실제 공개 그림 검산', ('public_links_and_hashes',)),
 ('새 README와 기존 docs pointer만 추가·과거 기록/운영/논문 보존', ('index_pointer', 'original_preservation')),
 ('불리한 결과/실패/미검토 상태를 보존한 연구 결과 commit', ('actual_result_commit', 'failures_and_limitations_visible')),
 ('push 직전 fetch·원격 diff/안전 통합·영향 검사', ('prepush_fetch_and_safe_integration',)),
 ('원격 main descendant·보호 준수·일반 fast-forward push', ('normal_fast_forward_push',)),
 ('main 차단 시 branch push/PR와 실제 blocker·사용자 WT 보존', ('branch_PR_fallback_if_needed',)),
 ('실제 remote main/result commit/보고서/대표 그림/gallery/hash 재확인', ('actual_remote_verification', 'remote_artifacts_verified')),
)


def catalog():
    result = []
    def put(ident, section, text, claims):
        result.append(dict(id=ident, section=section, requirement=text, required_claims=list(claims)))
    for i, (text, claims) in enumerate(P0, 1): put(f'P0_{i:02}', '4', text, claims)
    for i, text in enumerate(B_TEXT, 1):
        put(f'B_{i:02}', '5.4', text, ('expected_before_execution', 'technical_case_pass', 'test_source_bound') if i < 10 else ('independent_full238_support', 'production_kernel_not_imported', 'selected_support_checked'))
    for i, (text, claims) in enumerate(L_TEXT, 1): put(f'L_{i:02}', '6.7', text, claims)
    for axis in AXES: put('AXIS_' + axis, '8.1', axis, ('explicit_axis_status', 'bound_axis_evidence'))
    for i, name in enumerate(FIGURES, 1):
        put(f'FIG_{i:02}', '9.2', name, ('actual_PNG_and_data', 'manifest_hash_bound', 'README_direct_embed', 'actual_pixel_review'))
    for i, (text, claims) in enumerate(PUB, 1): put(f'PUB_{i:02}', '10', text, claims)
    put('INVENTORY_37', '7/9', '동일 등록37군·7그룹×3seed·실제 전체238행', ('exact_registered_arms', 'all238_rows', 'group_seed_inventory'))
    for pop, n in EXPECTED_N.items(): put('DENOM_' + pop, '4/7', f'{pop}={n}, 모든 등록군 분모 유지', ('exact_independent_denominator',))
    put('TRAINING_MAX18', '6.6/11', '새 소형학습 최대18회; 모든 실패 시도 포함', ('actual_attempts_at_most18', 'attempt_ledger_bound'))
    put('B_RAW_V1_PRESERVED', '5.4/8.2', '원 v1 산술 FAIL6132 전량·소스 보존; 전체 numeric PASS로 재분류 금지', ('raw_v1_FAIL_retained', 'original6132_accounted', 'raw_functions_and_bounds_unchanged'))
    put('B_SUPPORT_SEPARATE', '5.4/8.1', 'finite support와 raw numeric·physical owner 인증 분리', ('support_predicates_pass', 'raw_status_separate', 'owner_UNVERIFIED'))
    put('GALLERY_192', '9.3', 'PRIMARY192 전체 경량 뷰·상세 최대12·회귀3 별도', ('all192_views', 'detail_limit12', 'regression3_separate'))
    return result


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''): h.update(chunk)
    return h.hexdigest()


def pointer(obj, key):
    if key in ('', '/'): return obj
    if not key.startswith('/'): raise ValueError('Use explicit RFC6901 /pointer')
    for field in key[1:].split('/'):
        field = field.replace('~1', '/').replace('~0', '~')
        obj = obj[int(field)] if isinstance(obj, list) else obj[field]
    return obj


def test(claim, receipt, ptr, expected=None, op='eq', **extra):
    return dict(claim=claim, receipt=receipt, pointer=ptr, op=op, expected=expected, **extra)


def default_proofs():
    proof = {}
    def put(ident, *checks): proof.setdefault(ident, dict(checks=[]))['checks'].extend(checks)
    numeric = 'independent_numeric_v1/INDEPENDENT_CHECK.json'
    action = 'independent_action_full_v1/INDEPENDENT_ACTION_AUDIT.json'
    finite = 'b_finite_independent_v2/INDEPENDENT_FINITE_CHECK.json'
    btests = 'B_GEOMETRY_TESTS_v4.json'
    put('P0_02', test('fixed_inputs_and_reuse_verified', 'P0_BASELINE_INPUT_INVENTORY.json', '/status', 'PASS'))
    for pop, n in EXPECTED_N.items():
        put('P0_03', test('real_denominators', numeric, '/populations/' + pop, n))
        put('DENOM_' + pop, test('exact_independent_denominator', numeric, '/populations/' + pop, n))
    for domain in ('D0', 'D1'):
        for split, n in (('train', 768), ('validation', 235)):
            put('P0_03', test('synthetic_split_unchanged', action, f'/domains/{domain}/counts/{split}~1frames', n))
    put('P0_03', test('synthetic_split_unchanged', action, '/failure_count', 0),
        test('synthetic_split_unchanged', action, '/complete_scope', True))
    put('P0_03', test('independent_real_test_zero', 'EVALUATION_PROTOCOL.json', '/bootstrap/limitation', op='contains', expected='independent real test n=0'))
    put('P0_04', test('plan_frozen', 'PLAN_LOCK.json', '/status', 'FROZEN_BEFORE_NEW_PRIMARY_SCORING'),
        test('prediction_plan_bound', 'PREDICTION_SEAL.json', '/plan_lock_sha256', op='sha256_eq_file', file='PLAN_LOCK.json'),
        test('before_reference_scoring', 'PREDICTION_SEAL.json', '/before_scoring', True))
    for branch in ('B', 'D1', 'L'):
        put('P0_05', test('actual_branch_costs_bound', 'COST_AND_EXECUTION.json', '/branches/' + branch, op='nonempty'))
    put('P0_07', test('independent_numeric_implementation', numeric, '/production_metric_or_aggregate_imports', False),
        test('independent_numeric_implementation', action, '/production_action_input_loss_label_model_function_imports', 0))
    for i, key in enumerate(B_KEYS[:9], 1):
        put(f'B_{i:02}', test('expected_before_execution', btests, '/expected_before_execution/' + key, op='nonempty'),
            test('technical_case_pass', btests, '/status', 'PASS'),
            test('technical_case_pass', btests, '/tests_run', 10),
            test('technical_case_pass', btests, '/failures', op='empty'),
            test('technical_case_pass', btests, '/errors', op='empty'),
            test('test_source_bound', btests, '/source_sha256', op='sha256_eq_file', file='source:test_finite_boundary.py'))
    put('B_10', test('independent_full238_support', finite, '/all238_frames', True),
        test('independent_full238_support', finite, '/geometric_support_status', 'PASS'),
        test('production_kernel_not_imported', finite, '/production_geometry_imports', 0))
    for arm in ('B_FINITE_BANK', 'B_FINITE_FULL', 'B_FINITE_LOCAL', 'EDGE_FINITE_FULL'):
        put('B_10', test('selected_support_checked', finite, '/counts/' + arm + ':selected_independent_invalid_profiles', 0))
    # Existing L suite can be bound by its exact source and actual per-case
    # receipts through --proof-map. Aggregate suite PASS alone does not prove
    # the later real-bank/training-distribution diagnostics in items7--10.
    put('L_07', test('training_validation_diagnostics', action, '/checks/epoch_required_pose_action_diagnostics', 1, 'gte'),
        test('training_validation_diagnostics', action, '/failure_count', 0))
    put('L_09', test('common_real_bank_bytes', 'action_predictions_v1/PREDICTION_SEAL.json', '/features_sha256', op='sha256_eq_file', file='d1_real_candidates_v1/HEAD_FEATURES.jsonl.gz'),
        test('common_real_bank_bytes', 'action_predictions_v1/PREDICTION_SEAL.json', '/all_2x2_same_common_real_features', True),
        test('all_checkpoints_and_IDs', action, '/complete_scope', True),
        test('all_checkpoints_and_IDs', action, '/failure_count', 0),
        test('all_checkpoints_and_IDs', action, '/real/feature_rows', 238),
        test('all_checkpoints_and_IDs', action, '/real/prediction_rows', 238),
        test('all_checkpoints_and_IDs', action, '/real/checkpoint_count', 21))
    put('L_10', test('new_training_cap', 'action_training_v1/TRAINING_MANIFEST.json', '/actual_new_training_attempts', 18, 'lte'),
        test('seeds_and_groups_fixed', 'action_training_v1/TRAINING_MANIFEST.json', '/protocol/seeds', list(SEEDS), 'set_eq'))
    for axis in AXES:
        accepted = ['PASS', 'PASS_WITH_DOCUMENTED_REPRESENTATION_EXCEPTIONS']
        if axis == 'EXECUTION_COMPLETENESS':
            accepted += ['PASS_COMPLETED_OUTPUTS_WITH_DOCUMENTED_TECHNICAL_LOG_GAPS']
        if axis == 'GEOMETRIC_SUPPORT': accepted += ['PASS_WITH_RAW_REPRESENTATION_DIFFERENCES']
        if axis == 'VISUAL_AND_PUBLICATION': accepted = ['PUSHED_MAIN_VERIFIED_WITH_MODEL_PIXEL_REVIEW']
        put('AXIS_' + axis, test('explicit_axis_status', 'VALIDATION.json', '/axes/' + axis + '/status', accepted, 'in'),
            test('bound_axis_evidence', 'VALIDATION.json', '/axes/' + axis + '/evidence', op='nonempty'))
    for key in ('pixel_review', 'privacy_review', 'rights_scope', 'remote_verified'):
        put('AXIS_VISUAL_AND_PUBLICATION', test('explicit_axis_status', 'VALIDATION.json',
            '/axes/VISUAL_AND_PUBLICATION/substatus/' + key, True))
    for i, name in enumerate(FIGURES, 1):
        manifest = 'public:figures/FIGURE_MANIFEST.json'
        data = name[:-4] + '.csv'
        put(f'FIG_{i:02}',
            test('actual_PNG_and_data', manifest, '/artifacts/' + name, op='png_sha256_eq_file', file='public:figures/' + name),
            test('actual_PNG_and_data', manifest, '/artifacts/' + data, op='sha256_eq_file', file='public:figures/' + data),
            test('manifest_hash_bound', manifest, '/evaluation_seal_sha256', op='sha256_eq_file', file='evaluation_v1/EVALUATION_SEAL.json'),
            test('README_direct_embed', manifest, '/artifacts/' + name, op='markdown_embed', file='public:README.md', image='figures/' + name),
            test('actual_pixel_review', 'VISUAL_REVIEW.json', '/', op='reviewed_image', file='public:figures/' + name))
    # A's immutable offline check/manifest are verified after A exists. Later
    # report/matrix metadata never asks a current manifest to hash itself.
    offline, manifest = RESULT_SNAPSHOT + 'PUBLICATION_CHECK.json', RESULT_SNAPSHOT + 'PUBLICATION_MANIFEST.json'
    put('PUB_01', test('prework_git_state_recorded', GIT_PROOF, '/prework_state_recorded', True),
        test('prework_git_state_recorded', GIT_PROOF, '/prework_receipt_sha256/MIDRUN_GIT_STATE.json',
            op='sha256_eq_file', file='MIDRUN_GIT_STATE.json'))
    put('PUB_02', test('explicit_stage_allowlist', GIT_PROOF, '/stage_only_allowlisted', True),
        test('explicit_stage_allowlist', GIT_PROOF, '/explicit_stage_allowlist', op='nonempty'))
    put('PUB_03', test('public_content_and_privacy_scan', offline, '/status', 'PASS'))
    put('PUB_04', test('public_links_and_hashes', offline, '/status', 'PASS'),
        test('public_links_and_hashes', offline, '/checked_local_links', 1, 'gte'),
        test('public_links_and_hashes', offline, '/manifest_sha256', op='sha256_eq_file', file=manifest),
        test('public_links_and_hashes', REMOTE_RECEIPT, '/offline_check_snapshot/sha256', op='sha256_eq_file', file=offline),
        test('public_links_and_hashes', REMOTE_RECEIPT, '/offline_check_snapshot/manifest_sha256', op='sha256_eq_file', file=manifest))
    put('PUB_05', test('index_pointer', offline, '/existing_docs_index_pointers', op='nonempty'),
        test('original_preservation', GIT_PROOF, '/preservation_evidence', op='nonempty'))
    for key in ('original_reports_or_failure_records_changed', 'operating_defaults_changed',
                'papers_changed', 'agent_modified_original_user_worktree'):
        put('PUB_05', test('original_preservation', GIT_PROOF, '/preservation/' + key, False))
    put('PUB_06', test('actual_result_commit', REMOTE_RECEIPT, '/result_commit', op='nonempty'),
        test('actual_result_commit', REMOTE_RECEIPT, '/git_proof_sha256', op='sha256_eq_file', file=GIT_PROOF),
        test('failures_and_limitations_visible', GIT_PROOF, '/failures_and_limitations_visible', True))
    for key, expected in (('fetch_exit_code', 0), ('related_diff_reviewed', True), ('affected_rechecks_passed', True)):
        put('PUB_07', test('prepush_fetch_and_safe_integration', GIT_PROOF, '/prepush/' + key, expected))
    put('PUB_07', test('prepush_fetch_and_safe_integration', GIT_PROOF, '/prepush/remote_main_before', op='nonempty'))
    for key, expected in (('exit_code', 0), ('force_used', False), ('normal_fast_forward', True), ('result_descends_remote_before', True)):
        put('PUB_08', test('normal_fast_forward_push', GIT_PROOF, '/push/' + key, expected))
    for key, expected in (('status', 'PUSHED_MAIN_VERIFIED'), ('network_ls_remote_exit_code', 0),
                          ('network_fetch_exit_code', 0), ('remote_main_contains_result_commit', True)):
        put('PUB_10', test('actual_remote_verification', REMOTE_RECEIPT, '/' + key, expected))
    put('PUB_10', test('remote_artifacts_verified', REMOTE_RECEIPT, '/remote_artifacts_verified', True),
        test('remote_artifacts_verified', REMOTE_RECEIPT, '/verified_artifacts', op='nonempty'))
    proof['PUB_09'] = dict(checks=[], not_applicable_if=dict(
        reason='Actual main push and external verification succeeded; branch/PR fallback was not needed.',
        checks=[test('main_verified_side_condition', REMOTE_RECEIPT, '/status', 'PUSHED_MAIN_VERIFIED'),
            test('main_verified_side_condition', REMOTE_RECEIPT, '/remote_main_contains_result_commit', True),
            test('main_verified_side_condition', REMOTE_RECEIPT, '/remote_artifacts_verified', True)]))
    put('INVENTORY_37', test('exact_registered_arms', 'PREDICTION_SEAL.json', '/arms', list(ALL_ARMS), 'set_eq'),
        test('all238_rows', 'PREDICTION_SEAL.json', '/rows', 238),
        test('group_seed_inventory', 'action_training_v1/TRAINING_MANIFEST.json', '/runs', op='run_inventory'))
    put('TRAINING_MAX18', test('actual_attempts_at_most18', 'action_training_v1/TRAINING_MANIFEST.json', '/actual_new_training_attempts', 18, 'lte'),
        test('actual_attempts_at_most18', 'action_training_v1/TRAINING_MANIFEST.json', '/actual_new_training_attempts', 0, 'gte'),
        test('attempt_ledger_bound', action, '/training/actual_new_training_attempts', 18, 'lte'),
        test('attempt_ledger_bound', action, '/training/manifest_sha256', op='sha256_eq_file', file='action_training_v1/TRAINING_MANIFEST.json'))
    put('B_RAW_V1_PRESERVED', test('raw_v1_FAIL_retained', 'b_finite_independent_v1/INDEPENDENT_FINITE_CHECK.json', '/status', 'FAIL'),
        test('raw_v1_FAIL_retained', 'b_finite_independent_v1/INDEPENDENT_FINITE_CHECK.json', '/failure_count', 6132),
        test('original6132_accounted', finite, '/representation_difference_count', 6132),
        test('original6132_accounted', finite, '/prior_v1_failure_accounting/every_original_arithmetic_failure_accounted_exactly', True),
        test('original6132_accounted', finite, '/prior_v1_failure_accounting/prior_receipt_sha256', op='sha256_eq_file', file='b_finite_independent_v1/INDEPENDENT_FINITE_CHECK.json'),
        test('raw_functions_and_bounds_unchanged', finite, '/model_pose_observation_epsilon_selection_changes', 0),
        test('raw_functions_and_bounds_unchanged', finite, '/prior_v1_failure_accounting/raw_projection_and_predicate_unchanged', True))
    put('B_SUPPORT_SEPARATE', test('support_predicates_pass', finite, '/geometric_support_status', 'PASS'),
        test('raw_status_separate', finite, '/primary_raw_lambda_arithmetic_status', 'FAIL_PRESERVED_AND_EXPLAINED'),
        test('owner_UNVERIFIED', finite, '/original_material_owner', 'UNVERIFIED'))
    put('GALLERY_192', test('all192_views', 'public:gallery/GALLERY_MANIFEST.json', '/generated_thumbnails', 192),
        test('detail_limit12', 'public:gallery/GALLERY_MANIFEST.json', '/detailed_unique_panels', 12, 'lte'),
        test('regression3_separate', 'public:gallery/GALLERY_MANIFEST.json', '/regression_records', 3, 'len_eq'))
    return proof


class Receipts:
    def __init__(self, run, public, source):
        self.roots = dict(run=run.resolve(), public=public.resolve(), source=source.resolve())
        self.cache = {}; self.bindings = {}

    def path(self, name):
        prefix, sep, tail = name.partition(':')
        kind, rel = (prefix, tail) if sep and prefix in ('public', 'source') else ('run', name)
        path = Path(rel)
        if path.is_absolute() or '..' in path.parts:
            raise ValueError('Only explicit receipt-root-relative paths are accepted')
        result = self.roots[kind] / path
        if result.is_symlink() or not result.resolve().is_relative_to(self.roots[kind]):
            raise ValueError('Receipt/file may not escape its declared root')
        return result

    def load(self, name):
        if name not in self.cache:
            path = self.path(name)
            if path.suffix != '.json': raise ValueError('Proof values require JSON receipts, not raw journals or reference files')
            banned = ('FIXED_REFERENCES', 'SCORED_ROWS', 'ACTION_TARGETS', 'ORACLE_ROWS', 'RESULTS.json')
            if any(x in path.name for x in banned): raise ValueError('Raw/reference/scored input is outside requirement-evidence scope')
            before = sha(path); obj = json.loads(path.read_text())
            if sha(path) != before: raise ValueError('Receipt changed during read')
            self.cache[name] = obj; self.bindings[name] = before
        return self.cache[name]

    def check(self, rule):
        result = dict(rule=rule, status='PENDING', observed=None)
        try:
            obj = self.load(rule['receipt'])
            if rule.get('expected_sha256') and self.bindings[rule['receipt']] != rule['expected_sha256']:
                raise ValueError('Expected immutable receipt SHA changed')
            actual = pointer(obj, rule['pointer']); op = rule.get('op', 'eq'); expected = rule.get('expected')
            result['observed'] = actual
            if actual is None:
                result['reason'] = 'Explicit null is unavailable evidence, never substituted with zero'; return result
            if op == 'eq': passed = type(actual) is type(expected) and actual == expected
            elif op == 'in': passed = actual in expected
            elif op == 'gte': passed = isinstance(actual, (int, float)) and not isinstance(actual, bool) and actual >= expected
            elif op == 'lte': passed = isinstance(actual, (int, float)) and not isinstance(actual, bool) and actual <= expected
            elif op == 'len_eq': passed = len(actual) == expected
            elif op == 'set_eq': passed = len(actual) == len(expected) and set(actual) == set(expected)
            elif op == 'empty': passed = isinstance(actual, (list, dict, str)) and len(actual) == 0
            elif op == 'nonempty': passed = isinstance(actual, (list, dict, str)) and len(actual) > 0
            elif op == 'contains': passed = expected in actual
            elif op == 'run_inventory':
                pairs = [(r['arm'], r['seed']) for r in actual]
                passed = len(pairs) == len(L_GROUPS) * len(SEEDS) and set(pairs) == {(g, s) for g in L_GROUPS for s in SEEDS}
            elif op in ('sha256_eq_file', 'png_sha256_eq_file'):
                target = self.path(rule['file']); target_sha = sha(target)
                self.bindings[rule['file']] = target_sha
                passed = actual == target_sha
                if op == 'png_sha256_eq_file':
                    with target.open('rb') as f: passed = passed and f.read(8) == b'\x89PNG\r\n\x1a\n'
            elif op == 'markdown_embed':
                target = self.path(rule['file']); text = target.read_text()
                self.bindings[rule['file']] = sha(target)
                image = re.escape(rule['image'])
                passed = bool(re.search(r'!\[[^\]]*\]\(<?' + image + r'>?(?:\s+[^)]*)?\)', text))
            elif op == 'reviewed_image':
                target = self.path(rule['file']); target_sha = sha(target)
                self.bindings[rule['file']] = target_sha
                if not isinstance(actual, dict): raise TypeError('Visual review must be an explicit object')
                reviewer = actual.get('reviewer_type')
                method = actual.get('actual_review_method', actual.get('review_method'))
                if reviewer not in ('model', 'human') or not isinstance(method, str) or not method.strip():
                    raise KeyError('Explicit reviewer_type and actual_review_method/review_method')
                for key in ('findings', 'corrections', 'recheck_status'):
                    if key not in actual: raise KeyError('Explicit visual review ' + key)
                if not isinstance(actual['findings'], list) or not isinstance(actual['corrections'], list):
                    raise TypeError('Review findings/corrections must be explicit lists, including empty lists')
                rechecked = actual['recheck_status'] in ('NOT_REQUIRED', 'NOT_NEEDED', 'PASS', 'COMPLETED', 'RECHECKED')
                explicit_global = any(actual.get(k) is True for k in
                    ('actual_pixel_review_performed', 'actual_pixels_viewed', 'pixel_review_complete'))
                entries = []
                for key in ('reviewed_files', 'files', 'images', 'reviews', 'records'):
                    value = actual.get(key, [])
                    if isinstance(value, dict):
                        entries += [dict(v, path=k) if isinstance(v, dict) else dict(path=k, sha256=v) for k, v in value.items()]
                    elif isinstance(value, list): entries += [v for v in value if isinstance(v, dict)]
                matching = [v for v in entries if v.get('sha256', v.get('file_sha256')) == target_sha]
                if not matching: raise KeyError('Actual per-image pixel-review hash evidence')
                passed = rechecked and any((explicit_global or v.get('actual_pixels_viewed') is True or v.get('viewed') is True)
                    and v.get('status') in REVIEW_PASS_STATES for v in matching)
                if not passed and not explicit_global and not any('viewed' in v or 'actual_pixels_viewed' in v for v in matching):
                    raise KeyError('Explicit actual pixel viewing flag; file generation is insufficient')
            else: raise ValueError('Unknown proof operator: ' + op)
            result.update(status='PASS' if passed else 'FAIL', receipt_sha256=self.bindings[rule['receipt']])
        except (FileNotFoundError, KeyError, IndexError) as error:
            result['reason'] = 'Missing real receipt or explicit proof field: ' + str(error)
        except (ValueError, TypeError, OSError) as error:
            result.update(status='FAILED_TECHNICAL', reason=str(error))
        return result


def template():
    defaults = default_proofs()
    return dict(schema='REQUIREMENT_EVIDENCE_BINDINGS_V1',
        note='Fill missing claims with existing measured receipts; a generic status string is not a substitute for an unperformed check.',
        requirements={r['id']: dict(required_claims=r['required_claims'], checks=defaults.get(r['id'], {}).get('checks', []),
            **({'not_applicable_if': defaults[r['id']]['not_applicable_if']}
               if 'not_applicable_if' in defaults.get(r['id'], {}) else {})) for r in catalog()})


def validate_not_applicable(ident, spec, current_state):
    """Only the instruction's conditional branch/PR fallback can be waived."""
    if ident != 'PUB_09':
        raise ValueError('NOT_APPLICABLE may not bypass a core requirement: ' + ident)
    if current_state in ('FAIL', 'FAILED_TECHNICAL'):
        raise ValueError('NOT_APPLICABLE may not hide failed evidence: ' + ident)
    if not isinstance(spec.get('reason'), str) or not spec['reason'].strip():
        raise ValueError('Conditional fallback needs an explicit reason')
    needed = {'/status': 'PUSHED_MAIN_VERIFIED', '/remote_main_contains_result_commit': True,
              '/remote_artifacts_verified': True}
    for ptr, expected in needed.items():
        if not any(r.get('receipt') == REMOTE_RECEIPT and r.get('pointer') == ptr and r.get('op', 'eq') == 'eq'
                   and type(r.get('expected')) is type(expected) and r.get('expected') == expected
                   for r in spec.get('checks', [])):
            raise ValueError('Conditional fallback needs actual verified-main proof: ' + ptr)


def reject_self_proof(proof, store, output, proof_map=None):
    forbidden = {output.resolve()}
    if proof_map is not None: forbidden.add(proof_map.resolve())
    for spec in proof.values():
        rules = spec.get('checks', []) + spec.get('not_applicable_if', {}).get('checks', [])
        for rule in rules:
            for key in ('receipt', 'file'):
                if key in rule and store.path(rule[key]).resolve() in forbidden:
                    raise ValueError('Final requirement output/proof map cannot attest or hash itself')


def update(args):
    if not args.requirements_GO: raise PermissionError('Root requirement-evidence GO required; no real receipt read')
    store = Receipts(args.run_dir, args.public_dir, args.source_dir)
    proof = default_proofs(); bindings_sha = None
    if args.proof_map:
        bindings_sha = sha(args.proof_map); supplied = json.loads(args.proof_map.read_text())
        for ident, spec in supplied.get('requirements', {}).items():
            if ident not in {r['id'] for r in catalog()}: raise ValueError('Unregistered requirement: ' + ident)
            proof.setdefault(ident, dict(checks=[]))
            # A supplied claim replaces only that claim's default schema mapping.
            replaced = {x['claim'] for x in spec.get('checks', [])}
            proof[ident]['checks'] = [x for x in proof[ident]['checks'] if x['claim'] not in replaced] + spec.get('checks', [])
            if 'not_applicable_if' in spec: proof[ident]['not_applicable_if'] = spec['not_applicable_if']
    output = args.output or args.run_dir / 'REQUIREMENT_MATRIX_FINAL.json'
    previous = args.previous or args.run_dir / 'REQUIREMENT_MATRIX.json'
    if not output.resolve().is_relative_to(args.run_dir.resolve()) or output.resolve().is_relative_to(args.public_dir.resolve()):
        raise ValueError('Requirement matrix must remain under the PRIVATE run root')
    reject_self_proof(proof, store, output, args.proof_map)
    prior = None
    if previous.is_file():
        prior = dict(file=previous.name, sha256=sha(previous), preserved=True)
        if previous.resolve() == output.resolve():
            archive = previous.with_name(previous.stem + '.before_' + prior['sha256'][:16] + previous.suffix)
            if archive.exists() and sha(archive) != prior['sha256']: raise ValueError('Prior archive collision')
            if not archive.exists(): shutil.copyfile(previous, archive)
            prior['archive'] = archive.name
    records = []
    for item in catalog():
        spec = proof.get(item['id'], {}); checks = [store.check(r) for r in spec.get('checks', [])]
        per_claim = {}
        for claim in item['required_claims']:
            parts = [r for r in checks if r['rule']['claim'] == claim]
            per_claim[claim] = ('PENDING' if not parts or any(r['status'] == 'PENDING' for r in parts)
                else 'FAILED_TECHNICAL' if any(r['status'] == 'FAILED_TECHNICAL' for r in parts)
                else 'FAIL' if any(r['status'] == 'FAIL' for r in parts) else 'PASS')
        states = list(per_claim.values())
        state = ('SATISFIED' if all(s == 'PASS' for s in states) else 'FAILED_TECHNICAL' if 'FAILED_TECHNICAL' in states
            else 'FAIL' if 'FAIL' in states else 'PENDING')
        na = spec.get('not_applicable_if')
        if na:
            validate_not_applicable(item['id'], na, state)
            na_checks = [store.check(r) for r in na.get('checks', [])]
            if na.get('reason') and na_checks and all(r['status'] == 'PASS' for r in na_checks):
                state = 'NOT_APPLICABLE'; checks += na_checks
        records.append(dict(**item, status=state, claim_status=per_claim, evidence_checks=checks,
            not_applicable_reason=na.get('reason') if na and state == 'NOT_APPLICABLE' else None))
    groups = Counter(r['section'] for r in records); counts = Counter(r['status'] for r in records)
    matrix = dict(schema='RECEIPT_BOUND_FINAL_REQUIREMENT_MATRIX_V1', epoch=time.time(),
        generator='pallet_boundary_action_repair_v1.requirement_evidence', source_sha256=sha(__file__),
        instruction_sha256=sha(args.instruction) if args.instruction else None,
        prior_matrix=prior, proof_map_sha256=bindings_sha, receipt_sha256=store.bindings,
        exact_required_counts=dict(P0=7, B_tests=10, L_tests=10, validation_axes=6,
            summary_figures=7, publication_steps=10, registered_arms=37,
            L_groups=7, seeds=3, maximum_new_training_attempts=18,
            denominators=EXPECTED_N),
        records=records, section_counts=dict(groups), status_counts=dict(counts),
        all_applicable_requirements_satisfied=all(r['status'] in ('SATISFIED', 'NOT_APPLICABLE') for r in records),
        no_missing_evidence_marked_complete=True, raw_v1_numeric_failure_not_reclassified=True,
        material_owner='UNVERIFIED', physical_6D_GT='UNVERIFIED', independent_real_test_n=0,
        scope='Compliance proof only. It does not imply performance gain, physical validation, pixel viewing or remote publication without their actual receipts.',
        no_new_GT_or_score_decode=True, no_tests_models_solvers_Git_invoked=True)
    if output.exists() and output.resolve() != previous.resolve():
        raise FileExistsError('Preserve prior final matrix; use a new --output or explicit --previous same as output')
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(output.name + '.new')
    if temporary.exists(): raise FileExistsError('Preserve unfinished matrix output')
    temporary.write_text(json.dumps(matrix, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    temporary.replace(output)
    return dict(output=output.name, sha256=sha(output), counts=dict(counts), previous=prior,
        missing_requirements=[r['id'] for r in records if r['status'] == 'PENDING'],
        all_applicable_requirements_satisfied=matrix['all_applicable_requirements_satisfied'])


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--run-dir', type=Path)
    p.add_argument('--public-dir', type=Path)
    p.add_argument('--source-dir', type=Path, default=Path(__file__).parent)
    p.add_argument('--instruction', type=Path, help='Original instruction file; hashed only, not a new source of execution proof')
    p.add_argument('--proof-map', type=Path, help='Explicit receipt/json-pointer mapping for every missing semantic claim')
    p.add_argument('--previous', type=Path, help='Prior private matrix; preserved, default RUN_DIR/REQUIREMENT_MATRIX.json')
    p.add_argument('--output', type=Path, help='Default RUN_DIR/REQUIREMENT_MATRIX_FINAL.json; never writes public reports')
    p.add_argument('--requirements-GO', action='store_true')
    p.add_argument('--catalog', action='store_true', help='List exact requirements and required proof keys; reads no run data')
    p.add_argument('--proof-template', action='store_true', help='Print explicit mapping template; reads no run data')
    args = p.parse_args()
    if args.catalog: print(json.dumps(catalog(), indent=2, ensure_ascii=False)); return
    if args.proof_template: print(json.dumps(template(), indent=2, ensure_ascii=False)); return
    if not args.run_dir or not args.public_dir: p.error('--run-dir and --public-dir required for actual update')
    print(json.dumps(update(args), indent=2, ensure_ascii=False))


if __name__ == '__main__': main()
