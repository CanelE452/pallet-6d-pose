"""Prepare a curated V8 manifest/report only; never copy into Git or commit.

The manifest covers closed repository-relative payloads and the generated
report, packaging code and staging lists. It excludes itself and VALIDATION
to avoid a self-hash cycle. Raw observation journals/spools and every file in
the live R2 output are excluded. No pose/reference evaluation occurs here.
"""
import ast
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import subprocess

ROOT=Path('/home/minjae/Documents/github/pallet-pose')
D=Path('_docs/experiments/pallet_visibility_curriculum_20261013_v1')
B=D/'nonsevere_v1/statewise_joint_ot_20261014_v2/nonsevere_v4_fix/boundary_recovery_v1'
S=Path('scripts/research/pallet_visibility_curriculum_20261013_v1')
PUB=D/'publication_followup_20261015_v8'
PROTECTED=D/'nonsevere_v1/remainder_after_test49_v1/repair_campaign_20261015_v1/ACTIVE_GOAL_PROTOCOL_V1.json'
PINS={
 B/'CAL_geometry_guided_edge_noise_v4_32/MODEL_SEAL.json':'4b7d20df18f62269360a873e530ee29f7535c9d3d4ec4c6b6f72e49b7be48ab7',
 B/'initial_only_observed_ray_observations_v4_32/OBSERVATION_SEAL.json':'9d6b68bc50137fdc6356b70091aa81c783756a3ffec1f39408480812e186e280',
 B/'clean_fresh_BI_actual_runtime_smoke_v4_32/SEAL.json':'a8341769d6733debe770bae6954da63ec4ab3b9ea3978d9ee242cc3b700dd193',
 B/'clean_fresh_BI_actual_runtime_process_CAL_v4_32_v1/display_v2/RENDER_SEAL.json':'32a989232c4a1208e22a5ab4709737eaa681bc1a23709b8acc69dbd56ffb599d',
 B/'clean_fresh_BI_actual_runtime_process_CAL_v4_32_v1/ROOT_DISPLAY_V2_REVIEW_V1.json':'b58ec8a306b9762017014cac305abc32f14d937ff504018565603b3c2836a3a9',
 B/'clean_fresh_BI_pose_audit_root_review_v4_32/SOURCE_REVIEW_V1.json':'41df5bb9e9116a86d595b3328eea9bd97df555c8d4a8b81bdca5b5d1910b4e8d',
 B/'clean_fresh_BI_score_root_review_v4_32/SOURCE_REVIEW_V1.json':'7aae67528fbd0528e91dc32e9e1fcc16deb01da4ab60aaec0e5fc2db1b346d68',
 B/'clean_fresh_BI_score_peer_review_v4_32/SOURCE_REVIEW_V1.json':'474df223c6dd2db27eac147b9fdf884b9ff1db7d68bcd23c62a52a530c23d2b6',
 S/'clean_fresh_BI_input_projection_v4_32.py':'5dee94f2d4975a50a97564f1a4f7795c8c3994f69bb80db3f7e89f9c8d2c6026',
 S/'evaluate_clean_fresh_BI_poses_v4_32.py':'9fe1701fcad50119fe5937a4394149d99132237db8d4857b033920dc228e90ec',
 S/'evaluate_clean_fresh_BI_poses_v4_32_r2.py':'fa9ebe1d327972c585ee5f1f750a50f341151ec421c0e8ad5124a9a2e6b38187',
 S/'score_clean_fresh_BI_poses_v4_32.py':'ed00f94eccef5f55b5f0985a47e3ba53207e5809063c4c53d292565921849f1f',
 S/'audit_clean_fresh_BI_poses_v4_32.py':'a8ad3335c5750d961c8845769cbaa1cf433d1bc3e3bdaabb7a414155bf7b1203',
}


def sha(path):
    h=hashlib.sha256()
    with path.open('rb')as stream:
        for block in iter(lambda:stream.read(4*1024*1024),b''):h.update(block)
    return h.hexdigest()


def read(path):return json.loads((ROOT/path).read_text())


def write(path,value):
    (ROOT/path).write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n')


def allowed(rel):
    if rel.is_absolute()or'..'in rel.parts or not(rel.is_relative_to(D)or rel.is_relative_to(S)):
        raise ValueError('Only experiment/source paths allowed: '+str(rel))
    bad={'code_snapshot','source_snapshot','source_snapshot_v1','source_snapshot_v3','__pycache__','.pytest_cache','row_spool'}
    if any(part in bad for part in rel.parts):return False
    if any(part.startswith('clean_fresh_BI_pose_eval_v4_32_r2')for part in rel.parts):return False
    if rel.suffix in('.pyc','.pending')or'LIVE_PROGRESS'in rel.name:return False
    if rel.name in('OBSERVATIONS.jsonl.gz','CAL_OBSERVATIONS.jsonl.gz','REST_OBSERVATIONS.jsonl.gz',
                    'CAL_PREPARED_GROUPS.jsonl.gz','CAL_AUTHENTICATED_INPUTS.jsonl.gz','INPUTS.jsonl.gz',
                    'PREDICTIONS.jsonl.gz','E_COMPARISON.jsonl.gz'):return False
    if any(part in('.git','.ssh')or part=='.env'for part in rel.parts):raise ValueError('Machine path forbidden')
    return True


def select():
    selected=set()
    folders='''CAL_geometry_guided_edge_noise_v4_32
CAL_geometry_guided_edge_noise_v4_32_checks
geometry_guided_edge_noise_execution_v4_32_checks
clean_fresh_BI_input_projection_v4_32_checks
clean_fresh_BI_input_projection_peer_review_v4_32
clean_fresh_BI_pose_controller_source_checks_v4_32
clean_fresh_BI_pose_controller_source_checks_v4_32_r2
clean_fresh_BI_pose_controller_root_review_v4_32
clean_fresh_BI_pose_controller_peer_review_v4_32
clean_fresh_BI_pose_controller_peer_review_v4_32_r2
clean_fresh_BI_pose_eval_plan_v4_32
clean_fresh_BI_pose_eval_plan_v4_32_r2
clean_fresh_BI_actual_runtime_smoke_source_checks_v4_32
clean_fresh_BI_runtime_smoke_peer_review_v4_32
clean_fresh_BI_actual_runtime_smoke_plan_v4_32
clean_fresh_BI_actual_runtime_smoke_execution_v4_32_checks
clean_fresh_BI_actual_runtime_smoke_v4_32
clean_native_runtime_equation_audit_v4_32
clean_native_runtime_equation_peer_review_v4_32
clean_native_runtime_equation_driver_peer_review_v4_32
clean_fresh_BI_score_source_checks_v4_32
clean_fresh_BI_score_root_review_v4_32
clean_fresh_BI_score_peer_review_v4_32
clean_fresh_BI_pose_audit_source_checks_v4_32
clean_fresh_BI_pose_audit_root_review_v4_32'''.split()
    for folder in folders:
        parent=ROOT/B/folder
        if not parent.is_dir():raise FileNotFoundError(parent)
        selected.update(p.relative_to(ROOT)for p in parent.rglob('*')if p.is_file()and allowed(p.relative_to(ROOT)))
    for name in('PREPARATION_SEAL.json','CAL_PREFLIGHT_V1.json','CAL_PREFLIGHT_SEAL_V1.json'):
        selected.add(B/'geometry_guided_native_evidence_v4_32_checks'/name)
    obs=B/'initial_only_observed_ray_observations_v4_32'
    for name in '''PROTOCOL.json PREPARATION_SEAL.json CAL_EXECUTION_PROTOCOL.json CAL_OBSERVATION_SEAL.json
CAL_OWNER_COMPLETION_RECEIPT_V32_V1.json CAL_OWNER_COMPLETION_SEAL_V32_V1.json CAL_ROW_INDEX.json CAL_SUMMARY.json
REST_EXECUTION_PROTOCOL.json REST_OBSERVATION_SEAL.json REST_PREPARATION_SEAL.json REST_PROTOCOL.json REST_ROW_INDEX.json REST_SUMMARY.json
FINAL_PROTOCOL.json OBSERVATION_SEAL.json FINAL_COMPLETION_RECEIPT_V32.json
POPULATION_OWNER_COMPLETION_RECEIPT_V32_V2.json POPULATION_OWNER_COMPLETION_SEAL_V32_V2.json OWNER_POSTCHECK_ATTEMPT_V1_FAILURE.json
owner_verify_completion_v32_v1.py owner_verify_completion_v32_v2.py'''.split():selected.add(obs/name)
    for name in('OWNER_PREPARATION_FAILURE_RECEIPT_V1.json','PREPARATION_FAILURE_V1.json'):
        selected.add(B/'clean_fresh_BI_pose_eval_v4_32'/name)
    process=B/'clean_fresh_BI_actual_runtime_process_CAL_v4_32_v1'
    for name in('LATEST_REPORT_V2.md','OWNER_COMPLETION_RECEIPT_V2.json','OWNER_COMPLETION_SEAL_V2.json',
                 'RENDER_SEAL.json','SOURCE_PLAN_V1.json','ROOT_DISPLAY_V2_REVIEW_V1.json'):selected.add(process/name)
    selected.update(p.relative_to(ROOT)for p in(ROOT/process/'display_v2').rglob('*')if p.is_file()and allowed(p.relative_to(ROOT)))
    for name in('clean_fresh_BI_input_projection_v4_32','evaluate_clean_fresh_BI_poses_v4_32',
                 'evaluate_clean_fresh_BI_poses_v4_32_r2','smoke_clean_fresh_BI_runtime_v4_32',
                 'score_clean_fresh_BI_poses_v4_32','audit_clean_fresh_BI_poses_v4_32'):
        selected.update((S/(name+'.py'),S/('test_'+name+'.py')))
    for version in('v1','v2'):selected.add(S/('render_clean_fresh_BI_actual_runtime_process_CAL_v4_32_'+version+'.py'))
    previous={}
    for pub in sorted((ROOT/D).glob('publication_followup_20261015_v[1-7]')):
        manifest=pub/'COPIED_MANIFEST.json'
        if not manifest.exists():continue
        document=json.loads(manifest.read_text())
        values=document.get('files',document.get('entries',[]))
        if isinstance(values,dict):
            values=[dict(path=k,sha256=v if isinstance(v,str)else v['sha256'])for k,v in values.items()]
        for row in values:
            old=previous.get(row['path'])
            if old is not None and old!=row['sha256']:raise ValueError('Earlier publication path changed')
            previous[row['path']]=row['sha256']
    skipped=[]
    for rel in sorted(selected.copy()):
        if str(rel)in previous:
            if sha(ROOT/rel)!=previous[str(rel)]:raise ValueError('Do not overwrite an earlier published frozen file: '+str(rel))
            selected.remove(rel);skipped.append(str(rel))
    return selected,skipped,previous


def main():
    if (ROOT/PUB/'COPIED_MANIFEST.json').exists():raise FileExistsError('Preserve the closed V8 package')
    stamp=datetime.now(timezone.utc).isoformat()
    protected=read(PROTECTED)['protected_user_sha256']
    if len(protected)!=6:raise ValueError('Exact original protected6')
    for rel,value in protected.items():
        if sha(ROOT/rel)!=value:raise ValueError('Protected original changed')
    for rel,value in PINS.items():
        if sha(ROOT/rel)!=value:raise ValueError('Pinned closed artifact changed: '+str(rel))
    selected,skipped,previous=select()
    modelseal=read(B/'CAL_geometry_guided_edge_noise_v4_32/MODEL_SEAL.json')
    for name,value in modelseal['assets'].items():
        rel=B/'CAL_geometry_guided_edge_noise_v4_32'/name
        if rel not in selected or sha(ROOT/rel)!=value:raise ValueError('All seven sealed model assets required')
    fit=read(B/'geometry_guided_edge_noise_execution_v4_32_checks/FIT_POSTCHECK_RECEIPT_V1.json')
    if fit['model_count']!=20 or fit['optimizer_start_count']!=60 or fit['failures'] or fit['unavailable_models']or fit['saved_optimizer_failures']:
        raise ValueError('Closed successful20/60 CAL fit expected')
    replay=read(B/'CAL_geometry_guided_edge_noise_v4_32_checks/SAVED_CAL_MODEL_REPLAY_V1.json')
    native=read(B/'clean_native_runtime_equation_audit_v4_32/ACTUAL_AUDIT_OWNER_RECEIPT_V1.json')
    if replay['failure_count']or native['failures']or native['checks']!=1179:raise ValueError('Closed independent checks expected')
    a=fit['summary']['ALL_BI_REPLACE'];z=fit['summary']['RGB_NONFALLBACK_PRESERVE']
    prefix='../nonsevere_v1/statewise_joint_ot_20261014_v2/nonsevere_v4_fix/boundary_recovery_v1/'
    report=f'''# 후속 공개 V8: 새 CAL 모델과 독립 입력 경로의 정지 연결 검증

[V7의 CAL46 선 선택·전역 분할 한계](../publication_followup_20261015_v7/REPORT_KO.md)에 이어, 검증과 봉인이 끝난 V32 CAL 모델·positive 입력 경계·정지 PnP 연결을 공개한다. **전체 V32 자세 정확도 결과는 아직 없다. 기본값은 SAME_FROZEN_E이고 목표는 미달이다.** 진행 중인 R2 예측이나 그 입력·출력은 이 묶음에 넣지 않았다.

## 첫 CAL 실제 영상에서 확인한 연결

`eval_cad:1778653003088339968`은 고정 CAL 목록의 첫 사례다. 오차나 GT로 고르지 않았다. 상태는 `I0, U1, B2, U3, B4, B5, S6, B7`이며 외부 O/T가 없다. 아래는 수치 연결 진단이지 가림 효과나 정확도 개선의 증거가 아니다.

![원본·예측 모델·실제 유한 선 지지](PREFIXclean_fresh_BI_actual_runtime_process_CAL_v4_32_v1/display_v2/actual_process_4panel.png)

보라색 점선은 초기 독립 coarse의 모델 예측이다. B/I 점은 보관된 영상 관측이고 U1/U3는 넓은 타원 범위만 표시했다. U 중심과 S6 관측점은 그리지 않았다. 청록색 유한 TLS와 노란 실제 표본은 새 정밀 KP나 GT가 아니다. ALL 비용은 4개 물리변·5개 owner, SOURCE는 원 RGB 관측을 보존하면서 2개 물리변·2개 owner를 사용했다. 첫 표시본에서 설명상자가 지지를 가린 문제는 첫 봉인을 남긴 뒤 `display_v2`로 고쳤고 수치는 바꾸지 않았다.

![연속 최적화 전 독립 coarse와 새 시작 자세](PREFIXclean_fresh_BI_actual_runtime_process_CAL_v4_32_v1/display_v2/static_native_starts_2panel.png)

두 wire 모두 예측이며 ORIGINAL_WD의 저장된 정지 시작 자세를 그렸다. SOURCE의 coarse 비용이 더 낮아도 hardguard를 통과하지 않는다. 이 그림은 연속 최적화나 최종 방법 선택 결과가 아니다. [그림 설명·전체 원본·provenance](PREFIXclean_fresh_BI_actual_runtime_process_CAL_v4_32_v1/LATEST_REPORT_V2.md), [실제 smoke 봉인](PREFIXclean_fresh_BI_actual_runtime_smoke_v4_32/SEAL.json)을 보존했다.

관측 B/I와 독립 coarse에서 fresh SQPnP를 32회 호출했고, 두 W/D별로 fresh22+자체 coarse1인 23후보를 만들었다. 과거 e 숫자·cached bank는 수치 worker에 전달하지 않았다. 원 B/I q·source·covariance·weight와 full guard archive를 보존했다. 이 정지 smoke에서 기존 evaluator/refiner wrapper·연속 optimizer 호출은0이다.

## 새 CAL 잡음 모델과 독립 수치 검산

CAL inventory46과 기존 compatible44를 유지해 두 source 정책 각각 full+4LOGO의 선·접선 모델을 적합했다. 20모델·60시작점이 모두 수렴 성공했고 unavailable이나 과거 모델 재명명은 없었다. per-chain floor는 한 번만 더하며 접선축은 선택된 실제 측정선에 고정한다.

| CAL full 모델 | ALL_BI_REPLACE | RGB_NONFALLBACK_PRESERVE |
|---|---:|---:|
| 선 법선 σ [px] | {a['line']['sigma_normal_px']:.8f} | {z['line']['sigma_normal_px']:.8f} |
| 선 sin-angle σ | {a['line']['sigma_sine']:.8f} | {z['line']['sigma_sine']:.8f} |
| 접선 γ | {a['tangent']['gamma']:.8f} | {z['tangent']['gamma']:.8f} |

이 σ/γ는 새 선택 모집단의 CAL 비용 계수이며 물리변 정체·독립 coverage·정확도의 보증이 아니다. [모델·full+4LOGO 봉인](PREFIXCAL_geometry_guided_edge_noise_v4_32/MODEL_SEAL.json), [적합 owner](PREFIXgeometry_guided_edge_noise_execution_v4_32_checks/FIT_POSTCHECK_RECEIPT_V1.json), [runtime export](PREFIXgeometry_guided_edge_noise_execution_v4_32_checks/RUNTIME_PARAMETER_EXPORT_V1.json)을 함께 둔다.

저장 CAL errors만 사용한 독립 재현은 **{replay['checks']:,}개 검사·실패0**, 최대 차이 {replay['maximum_numeric_difference']:.6g}이었다. [독립 saved-model 재현](PREFIXCAL_geometry_guided_edge_noise_v4_32_checks/SAVED_CAL_MODEL_REPLAY_V1.json)은 새 fit이나 참조 재개방을 하지 않았다.

첫 CAL의 저장 입력·시작점·자세·비용은 별도 native scalar 식으로 다시 확인했다. **1,179는 검사 수이며 정확도 성공 장수가 아니다.** 자체 coarse와 시작점 16개 중8개가 현재 제약을 만족했다는 것도 정답 여부를 뜻하지 않는다. native scalar 최대 차이 {native['maximum_scalar_difference']:.6g}, gradient 성분 최대 차이 {native['maximum_gradient_component_difference']:.6g}였다. [독립 native 결과와 범위](PREFIXclean_native_runtime_equation_audit_v4_32/ACTUAL_FINDINGS_KO_V1.md)는 O/S/C·전체 selector·물리 대응·전역 최적이나 전체 정확도를 인증하지 않는다.

## 관측 완료와 새 컨트롤러의 준비 범위

CAL46+TEST49+REMAINDER143의 저장 raw 관측에 같은 initial-only 선택 규칙을 적용해 canonical238을 봉인했다. CAL227선·REST913선은 선택된 제안 수이며 최종 비용의 활성 선 수와 같지 않다. zero-selected 한 행도 남겼다. [전체 관측 완료 영수증](PREFIXinitial_only_observed_ray_observations_v4_32/POPULATION_OWNER_COMPLETION_RECEIPT_V32_V2.json), [최종 봉인](PREFIXinitial_only_observed_ray_observations_v4_32/OBSERVATION_SEAL.json)을 공개한다. 이는 전체 pose 정확도 평가의 완료가 아니다.

positive projector는 수치 worker에 명시한 양성 필드만 넘긴다. e는 부모 비교 stream에만 남기고 baseline·옛 proposals·initializations·외부 guide를 전달하지 않는다. native 원점 이동 `native_t`, 모델 중심 `R_cf @ mean(XYZ) + native_t`, 물리 회전 `R_cf @ Q`를 구분한다.

최초 컨트롤러 `v4_32`의 전체 입력 준비는 ndarray의 저장용 digest 직렬화에서 18행 뒤 실패했다. [실패 영수증](PREFIXclean_fresh_BI_pose_eval_v4_32/OWNER_PREPARATION_FAILURE_RECEIPT_V1.json)을 남기고 별도 R2에서 `G._digest(C.finite(packet))`로 고쳤다. 수치 helper28개의 AST와 그 외 변경 범위는 독립 검사로 확인했고 모델·관측은 다시 적합하거나 관측하지 않았다. 원 실패 출력은 덮어쓰지 않는다. [R2 소스 검토](PREFIXclean_fresh_BI_pose_controller_peer_review_v4_32_r2/SOURCE_REVIEW_V1.json), [입력 경계 검토](PREFIXclean_fresh_BI_input_projection_peer_review_v4_32/SOURCE_REVIEW_V1.json)를 보존한다. R2 전체 예측은 별도 진행 중이며 이 공개에는 그 live 폴더를 전부 제외한다.

새 all238 auditor와 스코어러는 소스·합성 검토만 닫혔다. auditor21개, scorer20개 검사가 통과했지만 실제 전체 감사·채점은 아직 수행하지 않았다. [auditor 검토](PREFIXclean_fresh_BI_pose_audit_root_review_v4_32/SOURCE_REVIEW_V1.json), [scorer ROOT 검토](PREFIXclean_fresh_BI_score_root_review_v4_32/SOURCE_REVIEW_V1.json), [scorer 독립 검토](PREFIXclean_fresh_BI_score_peer_review_v4_32/SOURCE_REVIEW_V1.json)를 공개한다. 향후 완전한238×5 예측과 독립 감사 실패0이 봉인된 뒤 별도 승인으로만 고정 참조를 읽는다. 자세 없는 행은 전체 분모에 실패로 남고 유한 fallback은 채점한다. headline은 `≤5cm·≤5°`이며 예전 strict `<5` boolean은 별도로 표시한다. 점수로 기본 방법을 바꾸거나 행을 누락하지 않는다.

## 공개 범위와 재현 한계

[정확한 경로·크기·SHA manifest](COPIED_MANIFEST.json), [검증](VALIDATION.json), [게시 파일 목록](FILES_TO_PUBLISH.txt), [필요한 force-add 목록](FORCE_ADD_FILES.txt)을 제공한다. 패키징은 commit·push·index 변경을 하지 않았으며 사용자 보호6개를 보존했다. V1–V7에 이미 공개한 동일 파일은 재선정하지 않았다.

이 묶음은 작은 CAL 모델 자산·완료 metadata·영수증·검토된 코드·그림을 담는다. 큰 raw OBS/CAL/REST 저널·spool·반복 code snapshot, live R2 output·positive INPUTS·E_COMPARISON·PRED 및 새 전체 점수는 포함하지 않는다. 작은 CAL saved-error 두 gzip은 모델 봉인의 자산으로 포함하며 원시 RGB/OBS 저널과 구분한다. 원본 CAL RGB 한 장은 승인된 설명그림 provenance로 바이트 그대로 포함했다. main만으로 전체 raw 재실행을 할 수 있다고 주장하지 않는다. [최초 대용량 Release](https://github.com/CanelE452/pallet-6d-pose/releases/tag/visibility-curriculum-snapshot-20261015-v1)는 이후 V32 raw를 포함하지 않는다. 새 정확도·목표달성·기본 채택을 선언하지 않는다.
'''.replace('PREFIX',prefix)
    (ROOT/PUB/'REPORT_KO.md').write_text(report)
    generated={'package_closed_clean32_runtime_v8.py','REPORT_KO.md','FILES_TO_PUBLISH.txt','FORCE_ADD_FILES.txt','COPIED_MANIFEST.json','VALIDATION.json',
               'initial_package_attempt_v1/package_closed_clean32_runtime_v8.py','initial_package_attempt_v1/FAILURE.json'}
    allpaths=selected|{PUB/name for name in generated}
    (ROOT/PUB/'FILES_TO_PUBLISH.txt').write_text(''.join(str(p)+'\n'for p in sorted(allpaths)))
    result=subprocess.run(['git','check-ignore','--stdin'],cwd=ROOT,input=''.join(str(p)+'\n'for p in sorted(allpaths)),text=True,capture_output=True)
    if result.returncode not in(0,1):raise ValueError(result.stderr)
    ignored=sorted(result.stdout.splitlines())
    (ROOT/PUB/'FORCE_ADD_FILES.txt').write_text(''.join(p+'\n'for p in ignored))
    manifested=selected|{PUB/name for name in generated-{'COPIED_MANIFEST.json','VALIDATION.json'}}
    rows=[]
    for rel in sorted(manifested):
        path=ROOT/rel
        if not allowed(rel)or path.is_symlink()or not path.is_file():raise ValueError('Unsafe selection: '+str(rel))
        if path.stat().st_size>=50*1024*1024:raise ValueError('Large payload forbidden: '+str(rel))
        rows.append(dict(path=str(rel),bytes=path.stat().st_size,sha256=sha(path)))
    write(PUB/'COPIED_MANIFEST.json',dict(schema='CURATED_CLOSED_V32_RUNTIME_PUBLICATION_MANIFEST_V8',UTC=stamp,
        count=len(rows),bytes=sum(r['bytes']for r in rows),files=rows,previous_publication_paths_skipped=skipped,
        self_reference_exclusions=['COPIED_MANIFEST.json','VALIDATION.json'],
        complete_file_list='FILES_TO_PUBLISH.txt',complete_file_count=len(allpaths),
        exclude_live_R2_output=True,raw_observation_journals_or_spools_included=False,
        current_primary_or_prediction_metrics_included=False,protected_user_sha256=protected,
        git_commit_push_or_index_changes_by_packager=False))
    checks=[];links=[];ast_count=0
    for row in rows:
        p=ROOT/row['path'];checks.append((row['path']+':SHA',sha(p)==row['sha256']))
        if p.suffix=='.py':ast.parse(p.read_text());ast_count+=1
        if p.suffix=='.md':
            for target in re.findall(r'!?\[[^\]]*\]\(([^)]+)\)',p.read_text()):
                if '://'in target or target.startswith('#'):continue
                resolved=(p.parent/target.split('#')[0]).resolve()
                rel=str(resolved.relative_to(ROOT))if resolved.is_relative_to(ROOT)else None
                exists=resolved.exists()
                covered=rel in{r['path']for r in rows}or rel in previous or rel in{str(PUB/x)for x in generated}
                # Earlier publication metadata (reports/receipts) intentionally
                # sit outside their own payload manifests; Git tracks them.
                if exists and not covered and rel is not None:
                    covered=subprocess.run(['git','ls-files','--error-unmatch','--',rel],cwd=ROOT,capture_output=True).returncode==0
                links.append(dict(source=row['path'],target=target,exists=exists,published_or_included=covered))
                checks.append((row['path']+':'+target,exists and covered))
    for rel,value in protected.items():checks.append((rel+':protected',sha(ROOT/rel)==value))
    for rel,value in PINS.items():checks.append((str(rel)+':pin',sha(ROOT/rel)==value))
    actualpub={str(p.relative_to(ROOT/PUB))for p in(ROOT/PUB).rglob('*')if p.is_file()}|{'VALIDATION.json'}
    checks.append(('publication metadata coverage',actualpub==generated))
    validation=dict(schema='CURATED_V8_PUBLICATION_PREPARATION_VALIDATION',UTC=stamp,
        manifest_sha256=sha(ROOT/PUB/'COPIED_MANIFEST.json'),checks=len(checks),failures=[n for n,ok in checks if not ok],
        AST_files=ast_count,local_links=links,protected6_exact=True,pinned_closed_assets_exact=True,
        payload_files=len(rows),payload_bytes=sum(r['bytes']for r in rows),all_paths_to_publish=len(allpaths),
        manifest_coverage_excludes_only_itself_and_validation=True,maximum_payload_file_bytes=max(r['bytes']for r in rows),
        force_add_count=len(ignored),live_data_folders_copied=False,actual_new_reference_prediction_image_or_model_inference=0,
        original_index_or_source_modified=False,publication_status='PREPARED_FOR_ROOT_REVIEW_NOT_COMMITTED_OR_PUSHED')
    write(PUB/'VALIDATION.json',validation)
    print(json.dumps(dict(payload_files=len(rows),bytes=validation['payload_bytes'],checks=len(checks),failures=validation['failures'],
        source_sha256=sha(Path(__file__)),report_sha256=sha(ROOT/PUB/'REPORT_KO.md'),manifest_sha256=sha(ROOT/PUB/'COPIED_MANIFEST.json'),
        validation_sha256=sha(ROOT/PUB/'VALIDATION.json'),force_add_count=len(ignored)),ensure_ascii=False))
    if validation['failures']:raise ValueError('Preserve failed V8 validation; append-only correction required')


if __name__=='__main__':main()
