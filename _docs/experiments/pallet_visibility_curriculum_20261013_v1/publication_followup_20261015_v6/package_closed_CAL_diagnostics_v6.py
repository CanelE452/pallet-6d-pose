"""Publish closed CAL masks, raw-query diagnosis and reviewed pure V32 code."""
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
PUB = D/'publication_followup_20261015_v6'
EXPECTED = '23490406b7cf176aadb96be3a5ec93c23dba1ddd'


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
    assert subprocess.check_output(['git','rev-parse','HEAD'],cwd=WT,text=True).strip() == EXPECTED
    assert not subprocess.check_output(['git','status','--porcelain'],cwd=WT,text=True).strip()
    assert not (WT/PUB).exists()
    protected = json.loads((ROOT/C/'ACTIVE_GOAL_PROTOCOL_V1.json').read_text())['protected_user_sha256']
    assert all(sha(ROOT/p) == h for p,h in protected.items())
    folders = '''clean_fresh_BI_eval_design_v4_32
clean_fresh_BI_runtime_peer_review_v4_32
clean_fresh_BI_runtime_v4_32_checks
clean_native_evidence_factory_v4_32_checks
fresh_observed_BI_pnp_proposals_peer_review_v4_32
fresh_observed_BI_pnp_proposals_v4_32_checks
generic_instance_mask_candidates_CAL3_v1
generic_instance_mask_candidates_CAL46_v1
generic_instance_mask_candidates_checks_v1
generic_instance_mask_candidates_peer_review_v1
generic_instance_mask_candidates_plan_v1
generic_instance_mask_population_CAL_checks_v1
generic_instance_mask_population_CAL_peer_review_v1
generic_instance_mask_population_CAL_plan_v1
geometry_guided_edges_peer_review_v4_32
initial_only_geometry_guided_edges_v4_32_checks
initial_only_observed_ray_CAL_actual_checks_v4_32
initial_only_observed_ray_CAL_cases_v4_32
initial_only_observed_ray_CAL_operator_checks_v4_32
initial_only_observed_ray_CAL_operator_peer_review_v4_32
initial_only_observed_ray_CAL_plan_v4_32
initial_only_observed_ray_association_peer_review_v4_32
initial_only_observed_ray_association_v4_32_checks
initial_only_observed_ray_observation_peer_review_v4_32
initial_only_observed_ray_population_source_checks_v4_32
initial_only_observed_ray_population_peer_review_v4_32
initial_only_raw_query_source_audit_v4_32
positive_native_evidence_factory_peer_review_v4_32'''.split()
    selected = set()
    for folder in folders:
        for path in (ROOT/B/folder).rglob('*'):
            if not path.is_file() or any(p in path.parts for p in ('code_snapshot','__pycache__')):
                continue
            if path.suffix in ('.json','.md','.py','.png','.npz','.gz'):
                assert path.stat().st_size < 50*1024*1024
                selected.add(path.relative_to(ROOT))
    modules = '''fresh_observed_BI_pnp_proposals_v4_32
initial_only_observed_ray_association_v4_32
clean_fresh_BI_runtime_v4_32
initial_only_observed_ray_observation_v4_32
run_initial_only_observed_ray_CAL_v4_32
rerank_initial_only_observed_ray_population_v4_32
geometry_guided_edge_problem_v4_32
geometry_guided_edge_authentication_v4_32
geometry_guided_edge_noise_calibration_v4_32
prepare_geometry_guided_native_evidence_v4_32
capture_generic_instance_mask_candidates_v1
capture_generic_instance_mask_population_CAL_v1'''.split()
    for module in modules:
        selected.update((S/(module+'.py'), S/('test_'+module+'.py')))
    rows = []
    for rel in sorted(selected):
        src,dst = ROOT/rel,WT/rel
        digest = sha(src)
        if dst.exists():
            assert sha(dst) == digest, ('Refuse to rewrite closed main artifact',str(rel))
        dst.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(src,dst)
        assert sha(src) == sha(dst) == digest
        rows.append(dict(path=str(rel),bytes=src.stat().st_size,sha256=digest))
    prefix = '../nonsevere_v1/statewise_joint_ot_20261014_v2/nonsevere_v4_fix/boundary_recovery_v1/'
    report = '''# 후속 공개 V6: 실제 SAM 후보와 V32 선 출처 진단

[V5 후보 자세 출처 정정](../publication_followup_20261015_v5/REPORT_KO.md)은 계속 유효하다. 기존 e의 수치 자세를 후보로 가져오는 경로를 제거하는 V32 순수 모듈을 검토했고, 실제 저장 CAL3에서 허용된 선을 다시 선택했다. **새 V32 전체 자세·잡음 적합·192장 정확도 평가는 아직 없다.** 기본값은 `SAME_FROZEN_E`이며 목표는 미달이다.

## 실제 다른 물체 마스크 공급 진단

일반 COCO80 YOLO26n 검출 box를 MobileSAM에 넣는 공급기를 같은 설정(conf0.25/imgsz640/max20)으로 CAL46 전수에 적용했다. KP·S/U/O 점 prompt, GT, 평가 오차로 사례를 고르는 경로는 없다. 프레임당 SAM image encoder는 최대1회이며 검출 box마다 세 분할 대안과 점수를 모두 남겼다. 46행·모든 빈 결과를 보존했다. 이는 로컬 CPU 실제 실행이고 Jetson FPS 추정 근거가 아니다.

| CAL46 진단 | 결과 |
|---|---:|
| 후보가 나온 영상 / 검출 없음 | 27 / 19 |
| detector forward / SAM encoder / box decoder | 46 / 27 / 67 |
| 저장한 분할 대안 | 201 |
| target 겹침 의심 / 미겹침 후보 | 22 / 45 |
| other-object·앞가림 확정 / 현재 pose 방법에 공급 | 0 / 0 |

`CANDIDATE`는 파렛트와 겹치지 않는다는 진단이며, 그 물체가 파렛트보다 앞에 있다는 뜻이 아니다. `suspected_target`도 진단 표기다. [전체46장 실제 그림과 결과](PREFIXgeneric_instance_mask_candidates_CAL46_v1/REPORT_KO.md), [1,849개 검사·실패0 영수증](PREFIXgeneric_instance_mask_population_CAL_checks_v1/ACTUAL_EXECUTION_QA_V1.json), [구체적 누락 사진 설명](PREFIXgeneric_instance_mask_population_CAL_checks_v1/ACTUAL_FINDINGS_KO_V1.md)을 공개한다. 최초 CAL3 별도 실행도 [그림](PREFIXgeneric_instance_mask_candidates_CAL3_v1/REPORT_KO.md)과 222/0 검사로 남겼으며 중복 세 사례의 예측은 전수 실행과 일치했다.

파렛트 자체를 bench로 검출해 거의 같은 마스크를 내거나, 정작 필요한 파란 스툴을 못 찾는 문제가 있었다. 아래 CAL 사진에서는 파렛트 위 스툴을 놓치고 배경 사람·우산만 검출했다. 배경 물체 마스크가 많아도 가려진 선을 해결했다고 볼 수 없다. 검출되지 않은 스툴을 “가림 없음”으로 처리하면 안 된다.

![파란 스툴을 놓치고 배경 물체만 검출한 실제 CAL 영상](PREFIXgeneric_instance_mask_candidates_CAL46_v1/case21_candidates.png)

그래서 이 공급기를 가림 hard veto로 연결하지 않았다. 동일 CAL 사진의 종류에 의존하지 않는 MobileSAM 전역 grid 분할은 별도 소스·선언을 준비하는 단계이며, 이번 공개 시점의 실제 AMG 결과는 없다. 최초 CAL3 실행의 임시 runtime 설정이 `/tmp/Ultralytics`로 저장된 운영상 차이도 영수증에 남겼다. CAL46 실행은 새 출력 폴더의 설정 디렉터리를 미리 만들어 해당 차이를 해소했으며 검출·분할 수학은 그대로다.

## e 후보 출처를 차단한 선 선택

원 raw chain의 실제 질의가 초기 기하에서 나온 `FROZEN17C_ORIGINAL_MODEL_SEED` 또는 실제 관측 B/I 양 끝점의 `MEASURED_BI_ENDPOINT_DIRECTION`일 때만 허용했다. cached 자세의 가이드와 physical VP 질의는 제외하고, 그래프 점수에도 외부 pose guide를 넘기지 않는다. 원 q·covariance·TLS·픽셀·raw index는 이동하거나 새로 만들지 않았다. [질의별 측정 상한/가지치기 소스 조사](PREFIXinitial_only_raw_query_source_audit_v4_32/AUDIT_V1.json)는 원 raw 질의 생성이 extra cached 질의와 전역 quota를 공유하지 않음을 조사한 범위다.

| 동일한 고정 CAL 사례 | 허용 후 유일 선 후보 | 최종 선택 선 |
|---|---:|---:|
| wood_184309:000925 | 75 | 6 |
| wood_184309:000804 | 275 | 8 |
| eval_cad:1778653017736058368 | 5 | 3 |

[실제 저장 선 진단](PREFIXinitial_only_observed_ray_CAL_cases_v4_32/REPORT_KO.md)과 [독립 식 재현333/0](PREFIXinitial_only_observed_ray_CAL_actual_checks_v4_32/AUDIT_V1.json)을 공개한다. 후자는 원 raw ref·원 q 앵커·고정 방향·법선43px 범위의 계산 검사다. 실제 물리 모서리 대응의 정답, 그래프 전역 최적, 새로운 pose 정확도를 보장하지 않는다. 선 선택 수와 최종 pose 비용에 활성화되는 선 수 역시 구분해야 한다.

## 새 자세 경로와 공개 범위

V32 새 후보 생성기는 관측 B/I의 full 및 고정3/4점 부분집합 SQPnP와 출처가 확인된 초기 coarse만 받는다. 원 과거 후보 목록·e·old initialization 인자는 없다. 초기 coarse가 유효하지 않으면 행을 유지하고 pose 없음으로 남긴다. 새 core/auth/noise, clean family/start/refine adapter, positive physics factory, 전체238 관측 운영 소스까지 순수·소스 검토를 마쳤다. 각각의 봉인·독립 검토와 초기 합성 fixture 실패 기록도 보존했다. 개별 합성검사 통과를 실제 전체 성능 개선으로 합산하거나 해석하지 않는다.

다음 단계는 새 모집단의 CAL46 잡음 모델을 별도로 적합하고, 새 B/I PnP 후보와 모든 기존 기하 제약을 연결해 TEST49+추가143 전체에 동일하게 평가하는 것이다. S는 영상/2D 관측을 만들지 않고 모델/PnP 투영만, U는 대략적인 영역만 사용한다. severe는0이며, 빈 결과·fallback·대응 불일치를 제외하지 않는다. e는 별도 비교 행으로만 보존한다.

[복사 manifest](COPIED_MANIFEST.json)와 [검증 영수증](VALIDATION.json)에 이번 공개 파일을 기록한다. 닫힌 실제 CAL 마스크 NPZ·PNG·보고서·소형 결과·소스/테스트를 추가한다. 진행 중인 AMG·전체32 pose solver/controller·중앙선언은 이번 복사 대상이 아니다. 큰 원 raw238 저널/spool/code snapshots와 모델 checkpoint도 이번 main에 넣지 않았다. 따라서 이번 main만으로 raw 픽셀부터 전체 실험을 완전히 재현한다고 주장하지 않는다. 원 사용자 수정6개와 기존 봉인은 보존했다.
'''.replace('PREFIX',prefix)
    latest = '''# 최신 확인 사항 — 2026-10-15 V6

[실제 CAL46 분할·V32 선 선택 보고서](publication_followup_20261015_v6/REPORT_KO.md)를 추가했다. 파란 스툴을 놓치는 일반 검출기의 한계를 실제 사진으로 확인했으며 이 후보를 가림 정답이나 hard veto로 쓰지 않았다. e 출처를 차단한 CAL3 선 선택은 완료했고, 새 V32 전체 자세 정확도는 아직 평가하지 않았다. 기본값은 기존 e이며 목표는 미달이다.

[V5 출처 정정](publication_followup_20261015_v5/REPORT_KO.md)은 계속 유효하다. [V4 실제 과정·점수](publication_followup_20261015_v4/REPORT_KO.md)도 원 봉인과 함께 보존한다.
'''
    stamp = datetime.now(timezone.utc).isoformat()
    old_pointer = sha(WT/D/'LATEST_FINDINGS_KO.md')
    for base in (ROOT,WT):
        (base/PUB).mkdir(parents=True,exist_ok=True)
        (base/PUB/'REPORT_KO.md').write_text(report,encoding='utf-8')
        (base/D/'LATEST_FINDINGS_KO.md').write_text(latest,encoding='utf-8')
        write(base/PUB/'COPIED_MANIFEST.json',dict(UTC=stamp,previous_main=EXPECTED,
            count=len(rows),bytes=sum(r['bytes'] for r in rows),files=rows,
            changed_existing_pointer_previous_sha256=old_pointer,
            full32_pose_AMG_or_unreviewed_sources_copied=False))
        write(base/PUB/'VALIDATION.json',dict(stage='LOCAL_COPY_VERIFICATION'))
    shutil.copyfile(Path(__file__),WT/PUB/Path(__file__).name)
    checks = [(row['path'],sha(ROOT/row['path']) == sha(WT/row['path']) == row['sha256']) for row in rows]
    py_count = 0
    for row in rows:
        if row['path'].endswith('.py'):
            ast.parse((WT/row['path']).read_text())
            py_count += 1
            checks.append((row['path']+':AST',True))
    links = []
    for rel in (PUB/'REPORT_KO.md',D/'LATEST_FINDINGS_KO.md'):
        for target in re.findall(r'!?\[[^\]]*\]\(([^)]+)\)',(WT/rel).read_text()):
            if '://' in target or target.startswith('#'):
                continue
            exists = (WT/rel).parent.joinpath(target.split('#')[0]).exists()
            links.append(dict(source=str(rel),target=target,exists=exists))
            checks.append((str(rel)+':'+target,exists))
    for rel,digest in protected.items():
        checks.append((rel+':protected',sha(ROOT/rel) == digest))
    validation = dict(UTC=stamp,checks=len(checks),failures=[n for n,ok in checks if not ok],
        copies_exact=True,AST_files=py_count,local_links=links,protected6_exact=True,
        only_existing_main_change='LATEST_FINDINGS_KO.md unsealed pointer',
        full_pose_or_primary_accuracy_claim=False)
    for base in (ROOT,WT):
        write(base/PUB/'VALIDATION.json',validation)
    assert not validation['failures'],validation['failures']
    print(dict(files=len(rows),bytes=sum(r['bytes']for r in rows),checks=len(checks),failures=0,
        report_sha256=sha(ROOT/PUB/'REPORT_KO.md')))


if __name__ == '__main__':
    main()
