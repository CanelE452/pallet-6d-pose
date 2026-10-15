"""Copy closed diagnostics and reviewed clean32 code to the isolated main worktree."""
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
S = Path('scripts/research/pallet_visibility_curriculum_20261013_v1')
PUB = D/'publication_followup_20261015_v7'
EXPECTED = '796bf9422a4cc5d5f7b95e87d8c9024e5c7428f3'


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(8*1024*1024), b''):
            h.update(chunk)
    return h.hexdigest()


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')


def main():
    assert subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=WT, text=True).strip() == EXPECTED
    assert not subprocess.check_output(['git', 'status', '--porcelain'], cwd=WT, text=True).strip()
    assert not (WT/PUB).exists()
    protected = json.loads((ROOT/C/'ACTIVE_GOAL_PROTOCOL_V1.json').read_text())['protected_user_sha256']
    assert all(sha(ROOT/p) == h for p, h in protected.items())
    folders = '''class_agnostic_AMG_CAL_case_v1
class_agnostic_AMG_case_checks_v1
class_agnostic_AMG_case_peer_review_v1
class_agnostic_AMG_case_plan_v1
clean_fresh_BI_pose_solver_peer_review_v4_32
clean_fresh_BI_pose_solver_v4_32_checks
initial_only_observed_ray_method_peer_review_v4_32
initial_only_observed_ray_method_source_checks_v4_32
initial_only_observed_ray_method_plan_v4_32
initial_only_observed_ray_population_plan_v4_32
positive_native_CAL_metadata_operator_peer_review_v4_32
positive_native_CAL_metadata_operator_v4_32_checks
positive_native_CAL_metadata_peer_review_v4_32
geometry_guided_edge_noise_peer_review_v4_32
initial_only_edge_noise_controller_v4_32_checks
CAL_geometry_guided_edge_noise_plan_v4_32'''.split()
    selected = set()
    for folder in folders:
        assert (ROOT/B/folder).is_dir(), folder
        for path in (ROOT/B/folder).rglob('*'):
            if not path.is_file() or any(p in path.parts for p in ('code_snapshot', '__pycache__')):
                continue
            if path.suffix in ('.json', '.md', '.py', '.png', '.npz', '.gz'):
                assert path.stat().st_size < 50*1024*1024
                selected.add(path.relative_to(ROOT))
    modules = '''capture_class_agnostic_AMG_case_v1
solve_clean_fresh_BI_poses_v4_32
declare_initial_only_observed_ray_v4_32
prepare_geometry_guided_native_metadata_v4_32
calibrate_geometry_guided_edge_noise_v4_32'''.split()
    for module in modules:
        selected.update((S/(module+'.py'), S/('test_'+module+'.py')))
    obs = B/'initial_only_observed_ray_observations_v4_32'
    for name in '''PROTOCOL.json PREPARATION_SEAL.json CAL_EXECUTION_PROTOCOL.json
CAL_OBSERVATION_SEAL.json CAL_OWNER_COMPLETION_RECEIPT_V32_V1.json
CAL_OWNER_COMPLETION_SEAL_V32_V1.json CAL_ROW_INDEX.json CAL_SUMMARY.json'''.split():
        selected.add(obs/name)
    for name in ('PLAN_EXECUTION_RECEIPT_V1.json', 'PLAN_EXECUTION_SEAL_V1.json'):
        selected.add(B/'geometry_guided_edge_noise_execution_v4_32_checks'/name)
    rows = []
    for rel in sorted(selected):
        src, dst = ROOT/rel, WT/rel
        digest = sha(src)
        if dst.exists():
            assert sha(dst) == digest, ('Refuse to rewrite closed main artifact', str(rel))
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(src, dst)
        assert sha(src) == sha(dst) == digest
        rows.append(dict(path=str(rel), bytes=src.stat().st_size, sha256=digest))
    prefix = '../nonsevere_v1/statewise_joint_ot_20261014_v2/nonsevere_v4_fix/boundary_recovery_v1/'
    report = '''# 후속 공개 V7: 전역 분할 한계와 독립 후보 경로의 CAL46 완료

[V5의 e 후보 출처 정정](../publication_followup_20261015_v5/REPORT_KO.md)은 계속 유효하다. 기존 e는 정답이 아니라 비교할 예측값이며, V32의 수치 후보·시작점·fallback으로 넣지 않는다. [V6의 실제 CAL46 detector→SAM 진단](../publication_followup_20261015_v6/REPORT_KO.md)에 이어 전역 AMG 결과와 검토가 끝난 V32 solver·선언·CAL 운영 코드를 추가한다. **새 V32 전체 자세 정확도는 아직 평가하지 않았다. 기본값은 SAME_FROZEN_E이며 목표는 미달이다.**

## 실제 사진에서 확인한 전역 MobileSAM의 한계

일반 검출기가 파렛트 위 파란 스툴을 놓친 같은 CAL 사진에, detector box 없이 MobileSAM 전역 16×16 균일 grid를 적용했다. 이 한 장은 앞선 누락을 보고 고른 진단 사례이며 대표 성능 표본이 아니다. 특정 KP·S/U/O 좌표·수동 ROI·GT를 prompt로 사용하지 않았다. 로컬 CPU 한 개에서 44.292초가 걸렸으며 Jetson FPS를 나타내지 않는다.

vendor의 고정 quality/stability/NMS 필터를 통과한 42개 mask를 모두 보존했다. 원래 768개의 raw multimask 슬롯 전체를 저장한 것은 아니다. 스툴을 단독으로 분리한 mask는 확인하지 못했다. AMG5는 스툴과 파렛트가 합쳐진 영역이고, AMG6는 주로 파렛트 영역이었다. 이 둘의 target overlap IoU 0.69804·0.97654는 저장 target mask와의 비교이며 물체 정답이나 깊이 정확도가 아니다.

![전역 균일 grid와 모든 후보의 개요](PREFIXclass_agnostic_AMG_CAL_case_v1/overview_candidates.png)

![AMG5의 합쳐진 영역과 AMG6의 파렛트 영역을 포함한 실제 후보](PREFIXclass_agnostic_AMG_CAL_case_v1/gallery_00.png)

[전체 사진·갤러리와 한계](PREFIXclass_agnostic_AMG_case_checks_v1/ACTUAL_FINDINGS_KO_V1.md), [저장 결과 568개 검사·실패0](PREFIXclass_agnostic_AMG_case_checks_v1/ACTUAL_EXECUTION_QA_V1.json), [원 실행 결과](PREFIXclass_agnostic_AMG_CAL_case_v1/RESULT.json)를 함께 공개한다. target과 안 겹치는 40개 CANDIDATE도 실제 앞가림의 확정이 아니다. target과 겹치는 두 후보의 차집합 역시 독립적인 다른 물체 mask로 취급하지 않았다. 이번 진단의 pose consumer 공급은0이며 grid·threshold 변경이나 추가 추론도 하지 않았다.

## 기존 e 후보를 제외한 실제 CAL46 선 선택

중앙 선언을 봉인한 뒤 실제 저장 raw 관측에서 CAL46을 다시 선택했다. 허용 질의는 초기 coarse 모델의 ORIGINAL 질의와 실제 관측 B/I 양 끝점 방향이며, 과거 cached 자세의 가이드·physical VP·외부 pose guide는 쓰지 않는다. 원 q·TLS·raw pixel index는 유지한다. CAL46의 캐시 선택과 최종 검증이 정상 종료됐고 현재 source284·input1237·보호6개와 원 parent63·CAL53 hash closure를 확인했다.

| CAL46 봉인 결과 | 값 |
|---|---:|
| 유지한 canonical 행 | 46 |
| 누락 / 중복 / 초과 | 0 / 0 / 0 |
| 선택된 선 / 선이 0개인 행 | 227 / 0 |
| 새 전체 pose 정확도 점수 | 아직 없음 |

[CAL46 요약](PREFIXinitial_only_observed_ray_observations_v4_32/CAL_SUMMARY.json), [행 목록](PREFIXinitial_only_observed_ray_observations_v4_32/CAL_ROW_INDEX.json), [owner 완료 영수증](PREFIXinitial_only_observed_ray_observations_v4_32/CAL_OWNER_COMPLETION_RECEIPT_V32_V1.json), [중앙 선언](PREFIXinitial_only_observed_ray_method_plan_v4_32/PREDECLARATION.json)을 공개한다. 선택 선 수는 최종 pose 비용의 활성 선 수와 다르며, 선택 완료가 물리적 모서리 대응이나 자세 정확도의 정답을 보장하지 않는다.

## 공개한 방법과 다음 검증

새 순수 solver는 관측 B/I로 새로 만든 SQPnP 후보와 출처를 검증한 초기 coarse를 사용한다. ALL_BI/SOURCE 각각의 native 및 containment 네 방법을 같은 입력 규칙으로 실행한다. e 비교 행은 수치 worker 밖에서 합친다. S는 영상 점을 추론하지 않고 모델/PnP 투영만 사용하며, U는 대략적인 영역 제약만 준다. invalid 초기값·빈 결과·fallback·대응 불일치도 행에서 제외하지 않는다.

solver 합성 통합10개, 중앙 선언18개, positive CAL metadata 운영36개, 새 CAL noise controller29개 검사가 통과했고 각각 독립 검토와 봉인을 보존했다. 이 숫자는 소스별 검사이며 실제 정확도 개선률로 해석하지 않는다. 중앙 선언 V1의 audit SHA 자료형 처리 오류와 초기 합성 fixture 오류도 별도 기록으로 남겼다. 현재 원 자료를 바꾸거나 과거 모델로 대체하지 않는다.

CAL46의 새 선 모집단으로 잡음 모델을 다시 적합하고, 나머지 TEST49+추가143에도 같은 선 선택을 적용한다. 이후 새 후보 경로·기하 제약·선 비용을 연결한 전체238 예측을 봉인한 다음 CAL46과 주 평가192의 오차를 따로 계산한다. severe는0이며, 주 평가는 모든192장과 기존49장 유지 조건을 함께 확인한다. 어느 방법도 아직 새 기본값으로 채택하지 않았다.

[복사 manifest](COPIED_MANIFEST.json)와 [로컬 검증](VALIDATION.json)에 공개 범위를 남겼다. 실제 AMG42 NPZ·5개 PNG·MD 보고서·QA, 닫힌 source/test/선언/CAL46 완료 metadata를 담았다. 진행 중인 REST192·새 noise 적합 결과·미검토 projector/controller는 이번 복사 대상이 아니다. 큰 raw 관측 저널·spool·반복 code snapshot·checkpoint도 main에 추가하지 않아 main만으로 모든 raw 픽셀 계산을 재현한다고 주장하지 않는다. 기존 대용량 snapshot은 [최초 Release](https://github.com/CanelE452/pallet-6d-pose/releases/tag/visibility-curriculum-snapshot-20261015-v1)에 있으며 이후 V30–V32의 전체 raw 자료를 포함하지 않는다. 원 사용자 수정6개와 기존 보고서·봉인은 유지했다.
'''.replace('PREFIX', prefix)
    latest = '''# 최신 확인 사항 — 2026-10-15 V7

[실제 전역 MobileSAM·V32 CAL46 완료 보고서](publication_followup_20261015_v7/REPORT_KO.md)를 추가했다. 전역 분할도 파란 스툴을 단독 instance로 분리하지 못했으므로 가림 확정 mask로 연결하지 않았다. 기존 e 후보 출처를 제외한 새 선 선택은 CAL46 전수에서 완료됐다(46행·227선). 새 V32 전체 자세 오차는 아직 평가하지 않았고 기본값은 SAME_FROZEN_E, 목표는 미달이다.

[V5 출처 정정](publication_followup_20261015_v5/REPORT_KO.md), [V6 검출기 진단·고정 CAL3](publication_followup_20261015_v6/REPORT_KO.md), [V4 실제 과정·기존 점수](publication_followup_20261015_v4/REPORT_KO.md)는 각각 원 봉인과 함께 보존한다.
'''
    stamp = datetime.now(timezone.utc).isoformat()
    old_pointer = sha(WT/D/'LATEST_FINDINGS_KO.md')
    for base in (ROOT, WT):
        (base/PUB).mkdir(parents=True, exist_ok=True)
        (base/PUB/'REPORT_KO.md').write_text(report, encoding='utf-8')
        (base/D/'LATEST_FINDINGS_KO.md').write_text(latest, encoding='utf-8')
        write(base/PUB/'COPIED_MANIFEST.json', dict(UTC=stamp, previous_main=EXPECTED,
            count=len(rows), bytes=sum(r['bytes'] for r in rows), files=rows,
            changed_existing_pointer_previous_sha256=old_pointer,
            live_observation_or_models_or_unreviewed_sources_copied=False))
    shutil.copyfile(Path(__file__), WT/PUB/Path(__file__).name)
    for base in (ROOT, WT):
        write(base/PUB/'VALIDATION.json', dict(stage='LOCAL_COPY_VERIFICATION'))
    checks = [(row['path'], sha(ROOT/row['path']) == sha(WT/row['path']) == row['sha256']) for row in rows]
    py_count = 0
    for row in rows:
        if row['path'].endswith('.py'):
            ast.parse((WT/row['path']).read_text())
            py_count += 1
            checks.append((row['path']+':AST', True))
    links = []
    for rel in (PUB/'REPORT_KO.md', D/'LATEST_FINDINGS_KO.md', B/'class_agnostic_AMG_case_checks_v1/ACTUAL_FINDINGS_KO_V1.md'):
        for target in re.findall(r'!?\[[^\]]*\]\(([^)]+)\)', (WT/rel).read_text()):
            if '://' in target or target.startswith('#'):
                continue
            exists = (WT/rel).parent.joinpath(target.split('#')[0]).exists()
            links.append(dict(source=str(rel), target=target, exists=exists))
            checks.append((str(rel)+':'+target, exists))
    for rel, digest in protected.items():
        checks.append((rel+':protected', sha(ROOT/rel) == digest))
    validation = dict(UTC=stamp, checks=len(checks), failures=[n for n, ok in checks if not ok],
        copies_exact=True, AST_files=py_count, local_links=links, protected6_exact=True,
        only_existing_main_change='LATEST_FINDINGS_KO.md unsealed pointer',
        new_full_pose_or_primary_accuracy_claim=False)
    for base in (ROOT, WT):
        write(base/PUB/'VALIDATION.json', validation)
    assert not validation['failures'], validation['failures']
    print(dict(files=len(rows), bytes=sum(r['bytes'] for r in rows), checks=len(checks), failures=0,
        report_sha256=sha(ROOT/PUB/'REPORT_KO.md')))


if __name__ == '__main__':
    main()
