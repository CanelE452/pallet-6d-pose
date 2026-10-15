"""Copy closed reports/sources only to the isolated main publication checkout."""
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
CAMPAIGN = D/'nonsevere_v1/remainder_after_test49_v1/repair_campaign_20261015_v1'
PUB = D/'publication_followup_20261015_v4'
S = Path('scripts/research/pallet_visibility_curriculum_20261013_v1')
EXPECTED_MAIN = '9f70b1d797e6ad706343d90c2606304fece674f1'


def sha(path):
    result = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8*1024*1024), b''):
            result.update(block)
    return result.hexdigest()


def read(path):
    return json.loads(path.read_text())


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')


def main():
    assert subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=WT, text=True).strip() == EXPECTED_MAIN
    assert not subprocess.check_output(['git', 'status', '--porcelain'], cwd=WT, text=True).strip()
    assert not (WT/PUB).exists()
    protected = read(ROOT/CAMPAIGN/'ACTIVE_GOAL_PROTOCOL_V1.json')['protected_user_sha256']
    assert all(sha(ROOT/path) == digest for path, digest in protected.items())
    folders = '''geometry_guided_edge_eval_v4_30
geometry_guided_edge_association_observations_v4_30
geometry_guided_process_v4_30_v1
independent_geometry_guided_process_review_v4_30
geometry_guided_evaluation_readiness_v4_30
frozen_e_geometry_guided_objective_diagnosis_v4_29b
frozen_actual_support_guard_source_audit_v4_30_v1
independent_campaign_accounting_review_v4_30
observed_anchor_native_ray_association_v4_31_checks
observed_anchor_native_ray_association_peer_review_v4_31
observed_anchor_native_ray_peer_review_v4_31
observed_anchor_native_ray_CAL_plan_v4_31
observed_anchor_native_ray_CAL_cases_v4_31
geometry_guided_population_v4_31_checks
geometry_guided_edges_peer_review_v4_31
geometry_guided_edge_noise_v4_31_checks
geometry_guided_edge_noise_peer_review_v4_31
geometry_guided_native_metadata_peer_review_v4_31
geometry_guided_native_metadata_smoke_plan_v4_31
geometry_guided_native_metadata_smoke_v4_31
observed_anchor_native_ray_method_plan_v4_31
observed_anchor_native_ray_population_source_checks_v4_31
observed_anchor_native_ray_population_peer_review_v4_31
geometry_guided_edge_noise_controller_peer_review_v4_31
geometry_guided_edge_noise_controller_v4_31_checks
geometry_guided_edge_controller_source_checks_v4_31
geometry_guided_edge_controller_peer_review_v4_31
geometry_guided_native_metadata_operator_v4_31b_checks
geometry_guided_native_metadata_operator_peer_review_v4_31b'''.split()
    selected = set()
    absent_optional = []
    for name in folders:
        folder = ROOT/B/name
        if not folder.exists():
            absent_optional.append(name)
            continue
        for path in folder.rglob('*'):
            if (not path.is_file() or any(p in path.parts for p in ('code_snapshot', 'frames', 'graphs', 'spools'))
                    or 'LIVE' in path.name or path.name.startswith('CAL_FRAME')):
                continue
            if path.suffix in ('.json', '.md', '.html', '.png', '.py') or path.name.startswith('prepared_case_'):
                assert path.stat().st_size < 50*1024*1024
                selected.add(path.relative_to(ROOT))
    for name in ('PROGRESS_V11_V30.json', 'RESULTS_V11_V30.html', 'PROGRESS_V11_V30_SEAL.json',
                 'PROGRESS_V11_V30_OWNER_RECEIPT_V1.json'):
        selected.add(CAMPAIGN/name)
    modules = '''observed_anchor_native_ray_association_v4_31
run_observed_anchor_native_ray_CAL_v4_31
geometry_guided_edge_problem_v4_31
geometry_guided_edge_authentication_v4_31
geometry_guided_edge_noise_calibration_v4_31
prepare_geometry_guided_native_evidence_v4_31
smoke_geometry_guided_native_metadata_v4_31
rerank_observed_anchor_native_ray_population_v4_31
prepare_geometry_guided_native_metadata_v4_31b
calibrate_geometry_guided_edge_noise_v4_31
evaluate_geometry_guided_edges_v4_31
evaluate_geometry_guided_edges_v4_30
audit_geometry_guided_edges_v4_30
build_repair_campaign_progress_v11_v30
render_geometry_guided_process_v4_30'''.split()
    for name in modules:
        selected.add(S/(name+'.py'))
        test = S/('test_'+name+'.py')
        if (ROOT/test).exists():
            selected.add(test)
    for name in ('declare_observed_anchor_native_ray_v4_31.py', 'run_observed_bi_edge_REST_v4_30c.py',
                 'diagnose_frozen_e_pose_under_v29_v4_29b.py', 'render_frozen_e_support_failure_v4_29b.py'):
        selected.add(S/name)
    records = []
    for relative in sorted(selected):
        source, target = ROOT/relative, WT/relative
        digest = sha(source)
        old = sha(target) if target.exists() else None
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
        assert sha(source) == sha(target) == digest
        records.append(dict(path=str(relative), bytes=source.stat().st_size, sha256=digest,
            previous_main_sha256=old, changed_from_previous_main=old != digest))
    cutoff = dict(UTC=datetime.now(timezone.utc).isoformat(), previous_main=EXPECTED_MAIN,
        V30='Full238x5 prediction, independent99385/0, fixed score, V11 accounting10148/0, fixed2 process301/0 closed; rejected/default e.',
        V31='Closed pure/source contracts, fixed3CAL rerank and zero-prefit metadata smoke, central plan. Full46/REST/noise/pose experiment is separate ongoing work.',
        unfinished_V31_observation_or_model_artifacts_copied=False,
        large_INPUTS_PREDICTIONS_OBSERVATIONS_or_raw_graphs_copied=False,
        new_independent_metrology_or_fresh_holdout_claim=False,
        protected_user_sha256=protected, absent_optional_folders=absent_optional)
    report = '''# 후속 공개 V4: V30 전체 실패 결과와 실제 단계별 이미지

V30은 전체238장·다섯 방법의 예측과 독립 검산, 동일 기준 점수 계산을 완료했다. 새 방법은 기존49장을 보존하지 못해 **채택하지 않는다**. 기본값은 `SAME_FROZEN_E`다. severe는 0장이고, CAL46은 보정용이며 주 평가는 TEST49+추가143=192장이다. 관측 없음·fallback·대응 불일치를 모두 포함했다.

| 동일 적용 방법 | TEST49 중심≤5cm | 추가143 중심≤5cm | 주192 중심/동시 성공 | 주192 평균/최대 중심 오차(cm) |
|---|---:|---:|---:|---:|
| 기존 e | 49 | 66 | 115/114 | 14.1121/260.4117 |
| 전체 B/I · native | 37 | 62 | 99/92 | 18.2379/244.3000 |
| 전체 B/I · containment | 37 | 61 | 98/95 | 21.1861/247.8790 |
| RGB q 보존 · native | 36 | 74 | 110/102 | 24.4420/1103.7211 |
| RGB q 보존 · containment | 38 | 78 | 116/112 | 19.1772/296.5358 |

동시 성공은 이동≤5cm·회전≤5°다. [정식 전체 보고서](../nonsevere_v1/statewise_joint_ot_20261014_v2/nonsevere_v4_fix/boundary_recovery_v1/geometry_guided_edge_eval_v4_30/REPORT_KO.md)에 회전·fallback·최악 사례·고정 기준·한계를 기록했다. [37개 동일 방법의 비교표](../nonsevere_v1/remainder_after_test49_v1/repair_campaign_20261015_v1/RESULTS_V11_V30.html)는 이전33개 결과를 보존했다. 선택된 자세의 독립 수학 검산99,385/0과 별도 집계 검산10,148/0은 정확도 개선이나 실제 모서리 대응 정답을 보장하지 않는다.

## 실제 이미지와 처리 단계

[고정 두 사례의 9단계 보고서](../nonsevere_v1/statewise_joint_ot_20261014_v2/nonsevere_v4_fix/boundary_recovery_v1/geometry_guided_process_v4_30_v1/REPORT_KO.md)는 공통 관측 네 단계와 다섯 독립 최종 결과다. 중간 optimizer 궤적이나 GT 투영이 아니다. 목재에서 새 활성 선은0개이므로 native 변화의 원인을 새 선으로 주장하지 않는다. 파란 의자에서 전체 B/I 방법은 선3개가 실제 활성화돼도 2.45→237.25cm로 악화했다.

![목재: 공통 관측과 독립 최종 결과](../nonsevere_v1/statewise_joint_ot_20261014_v2/nonsevere_v4_fix/boundary_recovery_v1/geometry_guided_process_v4_30_v1/case0_montage.png)

![파란 의자: 선이 활성화돼도 자세가 악화되는 사례](../nonsevere_v1/statewise_joint_ot_20261014_v2/nonsevere_v4_fix/boundary_recovery_v1/geometry_guided_process_v4_30_v1/case1_montage.png)

## 실패 원인과 다음 실험의 구분

[별도 실제 실패 사진과 선 지지 구간](../nonsevere_v1/statewise_joint_ot_20261014_v2/nonsevere_v4_fix/boundary_recovery_v1/frozen_e_geometry_guided_objective_diagnosis_v4_29b/actual_failure_noapril_1775201416944730624_v1/REPORT_KO.md)에서는 잘못 연결된 선도 원 B/I 점에 대한 법선 거리가0.86/1.83px여서 43px 검사를 통과했다. [query 출처 검사](../nonsevere_v1/statewise_joint_ot_20261014_v2/nonsevere_v4_fix/boundary_recovery_v1/frozen_e_geometry_guided_objective_diagnosis_v4_29b/actual_failure_noapril_1775201416944730624_v1/GUIDE_ANCHOR_DIRECTION_SOURCE_AUDIT_KO_V1.md)는 실제 탐색 앵커가 원 q에서41.08px 이동하고 방향도 native 변과 반대임을 확인했다. 이는 저장 V29 사례의 진단이며 동일 후보가 V30에서 선택됐다는 뜻은 아니다.

[V29 전체49 목적함수 진단](../nonsevere_v1/statewise_joint_ot_20261014_v2/nonsevere_v4_fix/boundary_recovery_v1/frozen_e_geometry_guided_objective_diagnosis_v4_29b/SUMMARY_KO_V1.md)도 함께 공개한다. 저장 e보다 나쁜 자세가 더 낮은 목적함수 값을 얻거나 e가 지지 검사에서 제외되는 사례가 있었다. 초기값 추가만으로 해결된다고 단정하지 않는다.

V31은 실제 query가 원 B/I q에서 시작했는지, 고정 native 방향과 부호가 맞는지를 추가 검사하는 별도 모집단이다. [중앙 선언](../nonsevere_v1/statewise_joint_ot_20261014_v2/nonsevere_v4_fix/boundary_recovery_v1/observed_anchor_native_ray_method_plan_v4_31/PREDECLARATION.json), [고정 CAL3 선 선택 진단](../nonsevere_v1/statewise_joint_ot_20261014_v2/nonsevere_v4_fix/boundary_recovery_v1/observed_anchor_native_ray_CAL_cases_v4_31/REPORT_KO.md), [CAL3 metadata 연결 결과](../nonsevere_v1/statewise_joint_ot_20261014_v2/nonsevere_v4_fix/boundary_recovery_v1/geometry_guided_native_metadata_smoke_v4_31/RESULT.json)를 공개한다. 후보 감소·연결 검사 성공은 정확도 개선 증거가 아니다. 전체 새 CAL 적합과 균일192 평가가 필요하다.

마스크는 기존 후보 추출과 초기 제외에 사용한다. 예상 선의 가시성 분류는 현재 graph/최종 자세 비용에 연결되지 않은 진단 정보다. [소스 감사](../nonsevere_v1/statewise_joint_ot_20261014_v2/nonsevere_v4_fix/boundary_recovery_v1/frozen_actual_support_guard_source_audit_v4_30_v1/REPORT_KO.md)에 한계를 남겼다. target mask 부재만으로 외부 가림을 확정하지 않는다. S의 영상·2D·국소 질의는0이며 기존 bounded 모델 항은 남아 있다. U는 대략적인 범위만 사용한다.

## 이번 push에 포함한 자료

[컷오프](CUTOFF.json), [파일별 복사 manifest](COPIED_MANIFEST.json), [검증 영수증](VALIDATION.json)을 함께 남긴다. 완료된 보고서·그림·점수·독립 진단과 검토가 끝난 새 소스/테스트를 copy-only로 반영했다. 원 작업 폴더의 사용자 수정6개는 그대로 유지했다. 진행 중인 V31 OBS/모델/전체 자세 결과는 넣지 않았다.

큰 INPUTS/PREDICTIONS/OBSERVATIONS 저널·원 raw graph·code snapshots는 이번 main 갱신에 포함하지 않았다. 고정 CAL3의 소형 prepared metadata만 예외다. 따라서 이번 main 자료만으로 원시 픽셀부터 전체 실험을 재현할 수 있다고 주장하지 않는다. [최초 전체 Release](../publication_20261015_v1/PUBLICATION_COMPLETE_KO.md)는 CAL32/46 시점 스냅샷이며 이후 raw238/V30 자료가 그 압축본에 들어 있지 않다. 추가143 기준은 기존 파생 annotation/PnP 참조이며 독립 계측 GT가 아니다. 반복 사용한 탐색적 데이터이고 현재 목표는 미달이다.
'''
    for base in (ROOT, WT):
        (base/PUB).mkdir(parents=True, exist_ok=True)
        write(base/PUB/'CUTOFF.json', cutoff)
        (base/PUB/'REPORT_KO.md').write_text(report, encoding='utf-8')
        write(base/PUB/'COPIED_MANIFEST.json', dict(files=records, file_count=len(records),
            total_bytes=sum(r['bytes'] for r in records), cutoff=cutoff['UTC']))
    shutil.copyfile(Path(__file__), WT/PUB/Path(__file__).name)
    tests = []
    for entry in records:
        tests.append((entry['path'], sha(ROOT/entry['path']) == sha(WT/entry['path']) == entry['sha256']))
    pyfiles = [WT/r['path'] for r in records if r['path'].endswith('.py')]
    for path in pyfiles:
        ast.parse(path.read_text(), filename=str(path))
        tests.append((str(path.relative_to(WT))+':AST', True))
    for base in (ROOT, WT):
        write(base/PUB/'VALIDATION.json', dict(stage='BUILDING_VALIDATION_RECEIPT'))
    mdpaths = [WT/r['path'] for r in records if r['path'].endswith('.md')]+[WT/PUB/'REPORT_KO.md']
    links = []
    for path in mdpaths:
        for target in re.findall(r'!?\[[^\]]*\]\(([^)]+)\)', path.read_text()):
            if '://' in target or target.startswith('#'):
                continue
            target = target.split('#')[0]
            exists = (path.parent/target).exists()
            links.append(dict(source=str(path.relative_to(WT)), target=target, exists=exists))
            tests.append((str(path.relative_to(WT))+':'+target, exists))
    assert all(sha(ROOT/path) == digest for path, digest in protected.items())
    validation = dict(check_count=len(tests), failures=[name for name, ok in tests if not ok],
        AST_files=len(pyfiles), local_links=links, new_numerical_accuracy_tests_claimed=False,
        copies_exact=True, protected_user_edits_exact=True, unfinished_V31_artifacts_excluded=True)
    for base in (ROOT, WT):
        write(base/PUB/'VALIDATION.json', validation)
    if validation['failures']:
        raise ValueError(validation['failures'])
    print(dict(files=len(records), bytes=sum(r['bytes'] for r in records), checks=len(tests), failures=0,
        report_sha256=sha(ROOT/PUB/'REPORT_KO.md')))


if __name__ == '__main__':
    main()
