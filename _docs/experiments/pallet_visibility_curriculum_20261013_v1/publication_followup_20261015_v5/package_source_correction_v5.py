"""Publish additive source-provenance correction; preserve old sealed reports."""
import ast
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import shutil
import subprocess

ROOT = Path('/home/minjae/Documents/github/pallet-pose')
WT = Path('/tmp/pallet-visibility-main-publication-20261015-v1')
D = Path('_docs/experiments/pallet_visibility_curriculum_20261013_v1')
B = D/'nonsevere_v1/statewise_joint_ot_20261014_v2/nonsevere_v4_fix/boundary_recovery_v1'
C = D/'nonsevere_v1/remainder_after_test49_v1/repair_campaign_20261015_v1'
P = D/'publication_followup_20261015_v5'
EXPECTED = '95bf3c91a0dac83133e29ac7bcf1b73aa16b2a53'


def sha(p):
    h = hashlib.sha256()
    with p.open('rb') as f:
        for block in iter(lambda: f.read(8*1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def write(p, value):
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(value, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')


def main():
    assert subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=WT, text=True).strip() == EXPECTED
    assert not subprocess.check_output(['git', 'status', '--porcelain'], cwd=WT, text=True).strip()
    assert not (WT/P).exists()
    protected = json.loads((ROOT/C/'ACTIVE_GOAL_PROTOCOL_V1.json').read_text())['protected_user_sha256']
    assert all(sha(ROOT/p) == s for p, s in protected.items())
    audit = B/'frozen_e_zero_edge_native_source_audit_v4_31'
    inventory = audit/'FIRST_CAL_E_ORIGIN_INVENTORY_V1.json'
    closure = audit/'COARSE_Q_GUIDE_SOURCE_CLOSURE_V1.json'
    assert sha(ROOT/inventory) == '4d70fa386b8d883e64a8b46d794b577181df7c0dd7a86598425a0bf38f143c22'
    assert sha(ROOT/closure) == 'c10f6dd26a35b2da960efe3a6533a43af5c8db636f9161b1c3e1652b55ef75ad'
    owner = B/'observed_anchor_native_ray_observations_v4_31/POPULATION_OWNER_COMPLETION_RECEIPT_V31_V1.json'
    assert sha(ROOT/owner) == '9597d47b574a62e93349f39ca4753809672515927edaa957a955b3d284750442'
    stamp = datetime.now(timezone.utc).isoformat()
    gate = dict(schema='ROOT_STRICT_NO_E_POSE_SOURCE_GATE_CORRECTION_V1', UTC=stamp,
        status='FAIL_CONFIRMED_INHERITED_PROPOSAL_AND_EXTRA_GUIDE_SOURCE',
        supersedes_inference_of_provenance_independence_in_old_V29_V30_V31_declarations=True,
        rewrites_old_sealed_artifacts=False,
        evidence={str(inventory): sha(ROOT/inventory), str(closure): sha(ROOT/closure)},
        observed_scope=dict(fixed_first_CAL_id='eval_cad:1778653003088339968',
            proposals=300, E_origin_distinct=50, E_origin_records=112,
            exact_baseline_physical_R_and_t_candidate='CACHED_0', exact_max_difference=0,
            other_actual_frames_inventoried=0),
        source_path='v17 collects normal_stage2/tangent all arms including SAME_FROZEN_E; V29-V31 evaluate all task.proposals and pass same proposals to refiner; extra line pose guides also consume inherited bank',
        interpretation=dict(E_is_prediction_not_GT=True,
            first_CAL_numeric_pose_copy_confirmed=True,
            loss_or_metric_equation_checks_do_not_certify_source_independence=True,
            removing_only_E_origin_tags_or_numeric_equality_filter_is_not_a_clean_fix=True,
            original_coarse_q_S_source_no_downstream_E_copy_found_in_traced_source=True),
        action=dict(V31_full_pose_prepare_predict_score_GO=False,
            V31_actual_process_render=False, campaign_V12_actual_build=False,
            preserve_finished_observation_CAL_model_and_smoke_artifacts=True,
            new_clean_V32_pose_start_source='independent original coarse plus fresh observed-BI SQPNP',
            new_clean_V32_line_source='strict FROZEN17C original / measured BI endpoints only; no inherited pose guide influence',
            V32_actual_experiments_at_this_correction=0,
            automatic_other_object_mask_status='standalone CAL3 supplier implementation; no confirmed occluder or actual NN claim'),
        default='SAME_FROZEN_E', adopted_new_method=False, goal_complete=False,
        protected_user_sha256=protected)
    write(ROOT/C/'SOURCE_PROVENANCE_GATE_CORRECTION_V1.json', gate)
    selected = {C/'SOURCE_PROVENANCE_GATE_CORRECTION_V1.json'}
    folders = ['frozen_e_zero_edge_native_source_audit_v4_31',
        'occluder_mask_supply_source_audit_v1', 'automatic_non_target_instance_mask_feasibility_v1',
        'saved_containment_cost_case_audit_v4_30_v1', 'CAL_geometry_guided_edge_noise_v4_31_checks']
    for folder in folders:
        for path in (ROOT/B/folder).iterdir():
            if path.is_file() and path.suffix in ('.json', '.md', '.py'):
                selected.add(path.relative_to(ROOT))
    obs = B/'observed_anchor_native_ray_observations_v4_31'
    for name in ('FINAL_PROTOCOL.json', 'OBSERVATION_SEAL.json', 'FINAL_COMPLETION_RECEIPT_V31.json',
                 'POPULATION_OWNER_COMPLETION_RECEIPT_V31_V1.json'):
        selected.add(obs/name)
    selected.add(B/'geometry_guided_edge_noise_v4_31_checks/FITTED_MODELS_COMPLETION_V1.json')
    selected.add(B/'geometry_guided_edge_noise_v4_31_checks/FITTED_MODELS_COMPLETION_SEAL_V1.json')
    for path in (ROOT/B/'CAL_geometry_guided_edge_noise_v4_31').rglob('*.json'):
        if 'code_snapshot' not in path.parts:
            selected.add(path.relative_to(ROOT))
    for folder in ('geometry_guided_actual_runtime_smoke_v4_31', 'geometry_guided_fitted_native_independent_smoke_v4_31'):
        for path in (ROOT/B/folder).glob('*.json'):
            if path.name != 'PREDICTION.json':
                selected.add(path.relative_to(ROOT))
    rows = []
    for rel in sorted(selected):
        src, dst = ROOT/rel, WT/rel
        digest = sha(src)
        if dst.exists():
            assert sha(dst) == digest, ('Refuse to rewrite existing main file', str(rel))
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(src, dst)
        assert sha(src) == sha(dst) == digest
        rows.append(dict(path=str(rel), sha256=digest, bytes=src.stat().st_size))
    prefix = '../nonsevere_v1/statewise_joint_ot_20261014_v2/nonsevere_v4_fix/boundary_recovery_v1/'
    report = '''# 후속 공개 V5: 후보 자세 출처 정정

**V29·V30 및 V31의 기존 공통 후보 목록에는 비교 방법 e의 예측 자세가 직접 들어가는 경로가 있었다. 따라서 이 실험들을 “e의 수치 자세를 후보·초기값으로 사용하지 않는 방법”의 검증으로 해석하면 안 된다. 이 점을 이전 보고서에 대한 추가 정정으로 남긴다.** 원래 선언·점수·봉인은 그대로 보존한다. 기존 수학 검산은 이 출처 조건까지 증명한 검사가 아니었다.

고정 첫 CAL 사례의 예측 입력만 열어 확인했다. 300개 후보 중 e 출처 기록이 있는 서로 다른 후보는50개, 출처 기록은112개였다. `CACHED_0`의 물체 회전과 위치는 저장 e와 차이가 정확히0이었다. 다른 후보에도 별도 D/seed alias가 함께 붙어 있어 e 이름만 지우는 것으로는 해결되지 않는다. 이 개수를 나머지237장에도 확인했다고 주장하지 않는다. e는 모델 예측이며 GT가 아니다.

[첫 CAL 실제 후보 출처 목록](PREFIXfrozen_e_zero_edge_native_source_audit_v4_31/FIRST_CAL_E_ORIGIN_INVENTORY_V1.json)과 [초기 coarse/q 및 선 가이드 출처 조사](PREFIXfrozen_e_zero_edge_native_source_audit_v4_31/COARSE_Q_GUIDE_SOURCE_CLOSURE_KO.md), [root의 진행 결정](../nonsevere_v1/remainder_after_test49_v1/repair_campaign_20261015_v1/SOURCE_PROVENANCE_GATE_CORRECTION_V1.json)에 근거를 남겼다. v17이 여러 과거 방법의 후보를 수집하면서 e도 함께 모았고, V29~V31 자세 컨트롤러가 그 목록 전체를 평가와 연속 최적화에 전달했다. 일부 추가 선 탐색 가이드도 이 목록을 사용했다.

## 완료된 것과 보류한 것

| 항목 | 상태 | 해석 |
|---|---|---|
| V30 전체238×5 및 실제 단계별 그림 | 완료·기본값으로 채택하지 않음 | 저장 실행의 점수는 유지하며, 엄격한 no-e 후보 출처 조건의 검증으로 해석하지 않음 |
| V31 선 관측 선택238장 | 완료 | CAL46+TEST49+추가143, 선0개 사례도 보존. 후보 출처 조건 통과와는 별개 |
| V31 CAL46 잡음 모델 | 20개 모델·60개 시작점 적합 완료 | full+4 LOGO, 두 정책. 실제 자세 정확도 개선 결과는 아님 |
| V31 저장 CAL 모델의 별도 계산 재현 | 10,888개 검사·실패0 | 저장 목적함수/미분/바인딩 계산의 일치 |
| V31 첫 CAL 실제 runtime/native 검산 | 53/0, 407/0 | 기존 후보 목록을 사용했으므로 no-e 후보 출처를 입증하지 않음 |
| V31 전체 자세·점수·과정 그림 / V12 집계 | 시작 보류 | 위 출처 문제를 확인한 뒤 전체 계산을 승인하지 않음 |
| V32 새 경로 | 소스·합성 검사 단계 | 실제 CAL/PRIMARY 자세 평가0, 성능 주장 없음 |

[V31 관측 완료 영수증](PREFIXobserved_anchor_native_ray_observations_v4_31/POPULATION_OWNER_COMPLETION_RECEIPT_V31_V1.json), [모델 완료 영수증](PREFIXgeometry_guided_edge_noise_v4_31_checks/FITTED_MODELS_COMPLETION_V1.json), [저장 모델 재현 결과](PREFIXCAL_geometry_guided_edge_noise_v4_31_checks/SAVED_CAL_MODEL_REPLAY_V1.json)를 공개한다. 기본 방법은 `SAME_FROZEN_E`이며 전체192장 모두5cm 이내라는 목표는 아직 달성하지 못했다.

## 이어서 하는 수정

새 V32는 공통 과거 후보 목록을 가져오지 않고, 출처가 확인된 초기 coarse와 실제 관측한 B/I 점으로 새로 푼 PnP 후보만 만든다. 선도 원 초기 기하의 FROZEN17C 질의 또는 실제 B/I 양 끝점에서 출발한 질의만 허용하고, 과거 후보 가이드가 선 점수에 영향을 주는 경로도 차단한다. 초기 coarse/q는 기존 e의 입력에도 쓰인 자료이므로 통계적 독립 관측이라고 부르지 않는다. 조사한 소스에서 downstream e의 수치 자세를 복사하는 경로가 없음을 확인한 범위다. 선택 모집단이 달라지면 CAL만으로 잡음 모델을 다시 적합하고 전체192장에 동일하게 적용한다. S는 영상 관측을 만들지 않고, U는 대략적인 영역 제약만 사용한다.

사용자가 요청한 “예측한 선이 다른 물체에 가려졌는가”의 연결도 조사했다. 현재 SAM 세 대안은 같은 파렛트 box의 분할 대안이며 다른 물체 instance 마스크가 아니다. [마스크 공급 경로 조사](PREFIXoccluder_mask_supply_source_audit_v1/REPORT_KO.md)와 [로컬 일반 객체 검출기 활용 검토](PREFIXautomatic_non_target_instance_mask_feasibility_v1/REPORT_KO.md)를 남겼다. 일반 YOLO 검출 box를 MobileSAM에 넣는 CAL3 진단을 별도로 구현한다. 검출·분할만으로 앞뒤 관계나 외부 가림을 확정하지 않으며, target mask의 바깥도 자동으로 가림 물체로 취급하지 않는다. 이번 정정 시점에는 이 공급기의 실제 NN 실행 결과가 없다.

또한 파란 의자 저장 사례에서 containment 비용이 꺼졌던 것은 아니었다. 전체 B/I 방법은 native392.37 + O/S3.77 + containment23.00 =419.15였지만 선택 자세가 바뀌지 않았다. [저장 비용 합산 확인](PREFIXsaved_containment_cost_case_audit_v4_30_v1/REPORT_KO.md)은 최종 저장 후보의 계산 검사이며 전역 최적화 증명이나 개선 결과는 아니다.

## 실제 이미지와 공개 범위

이미지는 [V4 전체 보고서](../publication_followup_20261015_v4/REPORT_KO.md)와 [실제 두 사례의 단계별 보고서](PREFIXgeometry_guided_process_v4_30_v1/REPORT_KO.md)에 있다. 아래 그림은 기존 V30 실행이며 새 V32 결과나 GT 투영이 아니다.

![기존 V30 파란 의자 사례의 공통 관측과 각 최종 자세](PREFIXgeometry_guided_process_v4_30_v1/case1_montage.png)

이 push는 정정문·출처 증거·닫힌 소형 메타데이터/모델 파일을 추가한다. [복사 manifest](COPIED_MANIFEST.json)와 [검증 영수증](VALIDATION.json)을 남긴다. 원 사용자 수정6개는 보존했고, 과거 봉인 보고서는 고치지 않았다. 1.436GB 관측 저널·행 spool·새 원시 데이터 전체·code snapshots는 이번 main에 포함하지 않는다. 최초 Release도 CAL32/46 시점의 별도 스냅샷이므로 이후 raw238을 포함한다고 주장하지 않는다.
'''.replace('PREFIX', prefix)
    latest = '''# 최신 확인 사항 — 2026-10-15 V5

[후보 자세 출처 정정과 진행 상태](publication_followup_20261015_v5/REPORT_KO.md)를 먼저 확인한다. 과거 공통 후보에 기존 e의 예측 자세가 수집되는 경로를 확인했으므로 V29~V31을 엄격한 no-e 후보 방법의 검증으로 해석하지 않는다. V31 전체 자세 평가는 시작 보류했고, 독립 초기 coarse+새 관측 B/I PnP 후보를 사용하는 V32 소스를 검증하고 있다. 성능 개선 또는 목표 달성은 아직 확인하지 못했다.

[V4 실제 이미지·수치 보고서](publication_followup_20261015_v4/REPORT_KO.md)와 원 봉인은 보존한다. 새 정정문이 후보 출처의 해석을 보완한다.
'''
    for base in (ROOT, WT):
        (base/P).mkdir(parents=True, exist_ok=True)
        (base/P/'REPORT_KO.md').write_text(report, encoding='utf-8')
        assert not (base/D/'LATEST_FINDINGS_KO.md').exists()
        (base/D/'LATEST_FINDINGS_KO.md').write_text(latest, encoding='utf-8')
        write(base/P/'COPIED_MANIFEST.json', dict(UTC=stamp, previous_main=EXPECTED,
            files=rows, count=len(rows), bytes=sum(r['bytes'] for r in rows),
            raw_journals_code_snapshots_and_unfinished32_sources_included=False))
        write(base/P/'VALIDATION.json', {'status': 'PENDING_LOCAL_COPY_AND_LINK_VERIFICATION'})
    shutil.copyfile(Path(__file__), WT/P/Path(__file__).name)
    checks = []
    for row in rows:
        checks.append((row['path'], sha(ROOT/row['path']) == sha(WT/row['path']) == row['sha256']))
    links = []
    for rel in (P/'REPORT_KO.md', D/'LATEST_FINDINGS_KO.md'):
        for target in re.findall(r'!?\[[^\]]*\]\(([^)]+)\)', (WT/rel).read_text()):
            if '://' in target or target.startswith('#'):
                continue
            exists = (WT/rel).parent.joinpath(target.split('#')[0]).exists()
            links.append(dict(source=str(rel), target=target, exists=exists))
            checks.append((str(rel)+':'+target, exists))
    python_count = 0
    for row in rows:
        if row['path'].endswith('.py'):
            ast.parse((WT/row['path']).read_text())
            checks.append((row['path']+':AST', True))
            python_count += 1
    for path, digest in protected.items():
        checks.append((path+':protected', sha(ROOT/path) == digest))
    receipt = dict(UTC=stamp, check_count=len(checks), failures=[n for n, ok in checks if not ok],
        copies_exact=True, links=links, AST_files=python_count,
        protected_user_edits_exact=True, old_sealed_reports_not_rewritten=True,
        numeric_accuracy_or_provenance_independence_not_claimed=True,
        source_gate_status=gate['status'], V31_full_pose_GO=False)
    for base in (ROOT, WT):
        write(base/P/'VALIDATION.json', receipt)
    assert not receipt['failures'], receipt['failures']
    print(json.dumps(dict(copied_files=len(rows), bytes=sum(r['bytes'] for r in rows),
        checks=len(checks), failures=0, report_sha256=sha(ROOT/P/'REPORT_KO.md')), ensure_ascii=False))


if __name__ == '__main__':
    main()
