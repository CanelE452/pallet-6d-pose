"""Small public projections of completed private evidence; never modifies runs."""
from __future__ import annotations
import argparse
from collections import Counter
import gzip
import json
from pathlib import Path
import re
import shutil
import time
from scripts.research.pallet_risk_information_v1.common import read,rows,save_rows,sha,write
from .registry import ARMS,LEARNED,SEEDS,POPULATIONS


def clean(value):
    if isinstance(value,dict):
        return {((Path(k).name if k.startswith('/') else k)):clean(v) for k,v in value.items()}
    if isinstance(value,list): return [clean(v) for v in value]
    if isinstance(value,str):
        if value.startswith(('/home/','/tmp/','/usr/','/opt/')): return '<private>/'+Path(value).name
        return re.sub(r'/(?:home|tmp|usr|opt)/[^\s\"\'<>]+', '<private-path>', value)
    return value


def number(v,digits=3):return 'NA' if v is None else f'{v:.{digits}f}'


def table(summary,pop):
    lines=[f'분모: {pop}. 위치 cm, 회전 degree. 표준편차는 ddof=1.', '',
        '| 경로 | N | 성공 | 위치 평균±SD | 회전 평균±SD | T>50/100 | R>45 | no-pose | fallback |',
        '|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for a,v in summary[pop].items():
        t,r=v['translation_cm'],v['rotation_deg'];fallback=v['status_counts']['fallback']['true_n']
        lines.append(f'| {a} | {v["denominator"]} | {v["successes"]} | {number(t["mean"])} ± {number(t["sample_std"])} | {number(r["mean"])} ± {number(r["sample_std"])} | {v["T_gt50cm"]}/{v["T_gt100cm"]} | {v["R_gt45deg"]} | {v["no_pose"]} | {fallback} |')
    return '\n'.join(lines)


def pair_table(pairs,pop,main_only=False):
    lines=['| before → after | 회복 | 손상 | 공동성공 | 공동실패 | Δ성공 %p | Δ위치 cm | Δ회전 deg | 성공차 recording CI95 %p |',
           '|---|---:|---:|---:|---:|---:|---:|---:|---|']
    for name,v in pairs[pop].items():
        if main_only and not v['before'].startswith('OBS_COMMON'): continue
        c=v['counts'];ci=v['deltas']['success_rate']['recording_cluster_CI']['CI95']
        ci='NA' if ci is None else f'[{number(100*ci[0])}, {number(100*ci[1])}]'
        lines.append(f'| {v["before"]} → {v["after"]} | {c["RECOVERY"]} | {c["DAMAGE"]} | {c["BOTH_SUCCESS"]} | {c["BOTH_FAILURE"]} | {number(v["success_difference_percentage_points"])} | {number(v["deltas"]["translation_cm"]["mean"])} | {number(v["deltas"]["rotation_deg"]["mean"])} | {ci} |')
    return '\n'.join(lines)


def run(root,public):
    root,public=Path(root),Path(public);public.mkdir(parents=True,exist_ok=True)
    summary=read(root/'evaluation/SUMMARY.json');pairs=read(root/'evaluation/PAIRED.json');obs=read(root/'evaluation/OBSERVATION_SUMMARY.json')
    plan=read(root/'EXPERIMENT_PLAN.json');train=read(root/'TRAINING_MANIFEST.json');pred=read(root/'PREDICTION_MANIFEST.json')
    for name in ('COMPARISON_ALL_ARMS.csv','ROW_METRICS.csv.gz','OBSERVATION_ROWS.csv.gz','SUMMARY.json','PAIRED.json','OBSERVATION_SUMMARY.json'):
        shutil.copyfile(root/'evaluation'/name,public/name)
    if clean(plan) != plan: raise ValueError('Canonical plan unexpectedly contains private paths')
    # This plan contains source/data hashes and scientific settings only.
    # Publish its exact pre-fit bytes rather than inventing a new plan hash.
    shutil.copyfile(root/'EXPERIMENT_PLAN.json',public/'EXPERIMENT_PLAN.json')
    write(public/'TRAINING_MANIFEST.json',clean(dict(train,private_manifest_sha256=sha(root/'TRAINING_MANIFEST.json'),
        independent_training_audit=read(root/'INDEPENDENT_TRAINING_AUDIT.json'))))
    write(public/'INPUT_AND_FEATURE_CONTRACT.json',clean(dict(schema='PUBLIC_N0_COMMON_INPUT_AND_FEATURE_CONTRACT_V1',
        P0_status=read(root/'P0_INPUT_CONTRACT.json')['status'],P0_sha256=sha(root/'P0_INPUT_CONTRACT.json'),
        inputs=dict(real=238,PRIMARY=192,CAL=46,synthetic_train=768,synthetic_validation=235),
        common_bank_manifest_sha256=sha(root/'common_bank/COMMON_BANK_MANIFEST.json'),
        extraction_lock=read(root/'EXTRACTION_LOCK_TECHNICAL_1.json'),producer_lock=read(root/'BANK_PRODUCER_SOURCE_LOCK_TECHNICAL_2.json'),
        feature_contract=plan['feature_contract'],solver_contract=plan['solver_contract'],fixed_common_sources=plan['common_sources'],
        reference_types=dict(training='synthetic virtual cuboid projection only',real_pose='existing reconstructed physical-pose/cuboid convention',
            physical_material_edge='Only separately authenticated original annotation provenance can establish this; no mask-derived labels are certified'))))
    selected=[]
    for row in rows(root/'PREDICTIONS.jsonl.gz'):
        selected.append({**{k:row[k] for k in ('id','partition','session','cluster_id')},'predictions':{
            arm:dict(actual_pose=e['actual_pose'],status=e['status'],accepted=e.get('accepted'),fallback=e.get('fallback'),
                     observation_signature=e.get('observation_signature'),solver_origin_arm=e.get('solver_origin_arm'))
            for arm,e in row['predictions'].items()}})
    save_rows(public/'SELECTED_POSES.jsonl.gz',selected)
    write(public/'PREDICTION_SEAL_PUBLIC.json',clean({k:pred[k] for k in (
        'status','complete','before_scoring','real_reference_inputs','experiment_plan_sha256','rows','arms','ordered_ids','artifacts','checkpoint_bindings',
        'source_sha256','backend','actual_new_pose_optimizer_calls','unique_observation_packets','exact_observation_cache_reuses',
        'cached_numeric_model_choice_cost','fresh_solver_cost','pose_status_counts','seal_epoch','wall_seconds','CPU_seconds','timing_scope')}))
    training_receipt=read(root/'execution/six_fixed_scientific_fits/RECEIPT.json')
    costs=dict(schema='BI_LOCAL_ACTUAL_EXECUTION_COST_V1',GPU_new_job_seconds=0,new_detector_forward_calls=0,new_H1_fits=0,
        completed_scientific_fits=6,actual_optimizer_constructions=6,failed_scientific_fits=0,actual_optimizer_steps=sum(r['optimizer_steps_this_attempt'] for r in train['runs']),
        six_fit_process=training_receipt,
        observation_batches=[read(root/'common_bank'/name) for name in ('synthetic_all_all_RUN.json','real_all_all_RUN.json')],
        blind_prediction_manifest={k:pred[k] for k in ('wall_seconds','CPU_seconds','actual_new_pose_optimizer_calls','unique_observation_packets','exact_observation_cache_reuses','cached_numeric_model_choice_cost','fresh_solver_cost')},
        technical_attempts=[read(p) for p in sorted((root/'execution').glob('*/RECEIPT.json'))],
        timing_caveat='Each named process/phase is actual measurement. Parent and nested subset times are not additive. Cached masks/N0 loading is not complete RGB backbone inference. Original extraction durations retained in cache rows are not new elapsed cost.',
        budget=plan['resource_bounds'])
    # Full bank RUN contains only aggregate bindings plus private paths; retain
    # costs/configuration but remove full cache indexing from public projection.
    costs['observation_batches']=[{k:v for k,v in x.items() if k not in ('frame_seals','records','inventory','ordered_ids','frames_detail','binding')} for x in costs['observation_batches']]
    write(public/'EXECUTION_AND_COST.json',clean(costs))
    paired_sections=['# 짝지은 회복·손상', '', '주 대비는 같은 seed의 COMMON→BI다. 세 seed는 같은192장의 반복 학습이며576개 독립 영상으로 세지 않는다. Δ는 after−before다. recording cluster bootstrap seed20260917,10,000 draws; 별도 session 감도·연속오차 CI·tail 발생/해소·모든 ID는 [PAIRED.json](PAIRED.json)에 보존했다. 반복 개발 자료에 대한 기술적 CI이며 독립 검증이나 동등성을 뜻하지 않는다.']
    for pop in POPULATIONS:paired_sections+=['',f'## {pop}','',pair_table(pairs,pop)]
    (public/'PAIRED_RESULTS.md').write_text('\n'.join(paired_sections)+'\n')
    full=['# 모든 경로·전체 분모 결과','', '공동 성공: 위치≤5cm AND 원 C2 회전≤5°. 위치·회전의 가용 n, 표본 분산·P90·P95·최대값은 [원수치 CSV](COMPARISON_ALL_ARMS.csv)에 있다. 극단값 제외0건.']
    for pop in POPULATIONS:full+=['',f'## {pop}','',table(summary,pop)]
    (public/'FULL_RESULTS.md').write_text('\n'.join(full)+'\n')
    observations=['# 관측 수준', '', '모든 B/I 표시는 고정 자동 상태다. 아래 오차 참조는 등록 직육면체의 가상 투영 코너이며 실제 재료 경계의 정답이 아니다. 부분선은 full2D 점 오차를 계산하지 않는다. target N은 부분선·유지를 포함한다. full N의 적용 전후는 같은 참조 subset이고, accepted subset을 전체 개선으로 대신하지 않는다.', '',
        '| population / arm / auto state | target N | full N | partial N | q0 keep | actual changed | alternatives N | before px mean±SD | after px mean±SD | better/worse/same corners |',
        '|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    for v in obs:
        observations.append(f'| {v["population"]} / {v["arm"]} / {v["automatic_state"]} | {v["target_corners"]} | {v["full_point_n"]} | {v["partial_n"]} | {v["q0_keep_n"]} | {v["changed_n"]} | {v["valid_alternative_corners"]} | {number(v["before"]["mean"])} ± {number(v["before"]["sample_std"])} | {number(v["after"]["mean"])} ± {number(v["after"]["sample_std"])} | {v["recovered_corners"]}/{v["damaged_corners"]}/{v["same_corners"]} |')
    (public/'OBSERVATION_RESULTS.md').write_text('\n'.join(observations)+'\n')
    report=['# B·I 조건부 국소 관측 보정 결과', '', '결과 판정과 남은 병목은 아래 해석 문단을 따른다. 원 계획·소스·전체 분모·실패를 보존했다.', '',
        '## 같은 seed의 추가 효과', '', pair_table(pairs,'PRIMARY192',main_only=True), '',
        '주 비교는 동일 N0·자동 V·상태·마스크·두 cue의 합집합 후보·181개 numeric+181개 validity·27,585 parameters·공통 자세 솔버에서 점수망 c만 바꾸었다. COMMON은(0.5,0.5), BI는 자동B/I onehot다. COMMON도 기하/마스크에서 역할을 간접 추론할 수 있고, 공통 후단의 B/I CAL covariance·reliability는 유지했다. 파이프라인 전체의 B/I 제거 효과가 아니다.', '',
        '## 실제 자세 결과', '',table(summary,'PRIMARY192'), '',
        '원49장·추가143장·별도CAL46과 전체 분산·P90/P95·최대값·가용 n은 [FULL_RESULTS](FULL_RESULTS.md)와 [CSV](COMPARISON_ALL_ARMS.csv)를 따른다. CAL은 PRIMARY에 포함하지 않는다. N0108/E0114/H1대표120/PoseFix112·111은 historical complete systems이며 새 matched solver의 성능으로 복사하지 않았다.', '',
        '## 실행과 공정성', '',
        '기존 합성 train768/validation235, 원 asset/run split과 C1/C2를 유지해2모델×3seed=6fit을 완료했다. 각 seed의 초기 tensor·optimizer·프레임 순서 prefix를 독립 검산했다. 새 실사 지도학습/백본 forward/H1 fit/렌더/튜닝은0이다. 기존 상태 분류기와 CAL 교정의 실사 감독 이력은 유지하므로 전체 시스템 실사 감독0이라고 하지 않는다.', '',
        '원 v2의 두 cue를 같은15×49 픽셀에서 계산했다. 최대6선·12서로 다른 incident-neighbor 교점+q0를 보존하고, 전체 교점 일치 veto만 양군에서 공통 제거했다. target은 raw 교점이 아닌 blend 후 실제 적용좌표의 log1p(error/2px)다. loss는 candidate→corner→frame 평균 SmoothL1이며 argmin 실제 후보 하나를 쓴다.', '',
        '완전 후보가 없을 때만 원 FACE 부분선을 사용하고 해당 full2D 우도는 폐기했다. 부분선은 normal만 구속하며 tangent endpoint 일치를 강제하지 않는다. 알려진 동일 픽셀 source의 중복은 억제하지만 서로 다른 픽셀에서 나온 같은 물리 변의 독립성까지 인증하지 못한다. O/T/U 영역·H1·R_CONT·전역 outline 비용은 추가하지 않은 공통 가시 점/부분선 경로이며 E0 재현이 아니다.', '',
        'RAW_OBSERVATION_MATCHED는 원 가시q0만 같은 새 솔버에 넣은 기준이다. LEGACY_RULE_MATCHED는 기존 D1 frozen_BI_fit+CAL/recover stages와 같은N0 FACE 관측을 넣었다. LEGACY와 새 모델 차이는 후보 생성과 선택을 함께 바꾸므로 c 하나의 효과로 귀속하지 않는다.', '',
        '새 관측→exact packet→fresh pose가 모두 연결됐다. 각 W/D 표현에 원N0 시작 하나씩, 최대2회 최적화, 같은 cost/guard/fallback을 적용했다. 정확히 같은 전체 관측·covariance·state·N0·config만 캐시 재사용했다. [독립 검산](VERIFICATION.json), [관측 전체표](OBSERVATION_RESULTS.md), [짝지은 전체표](PAIRED_RESULTS.md), [진단](DIAGNOSTIC_RESULTS.md)을 함께 본다.', '',
        '## 비용·오류·시각화', '',
        f'새 실제 optimizer 생성6회, 완료6회, 학습 실패0회, 총{costs["actual_optimizer_steps"]}steps. 전체6fit process wall {number(training_receipt["wall_seconds"])}s. 실제 관측/자세 계산 비용과 모든 실행 실패는 [EXECUTION_AND_COST](EXECUTION_AND_COST.json)에 남겼다. 초기 SciPy Qhull 임시파일이 I/O guard 밖으로 나가던 실행 오류는 run-local tempdir로 수정했고, 실제 실사 자세 저장 전 실패한 로그를 보존했다. 추출 pilot의 metadata/validation 기술 수정도 원 소스와 출력 동등성 증거를 남겼다. 학습 과학 설정은 바꾸지 않았다.', '',
        '[전체192장 경량 갤러리](gallery/index.html)는 같은 이미지의 COMMON·BI와 모든 시스템 numeric row를 연결한다. 자동 회복/손상/공동성공/공동실패의 고정 ID 예시, 최대 악화, B/I crop 및 6fit 실제 곡선을 포함한다. S는 직접 영상 관측이 아닌 pose projection이다. 모델 pixel view와 사람/browser 열람은 [VISUAL_REVIEW](VISUAL_REVIEW.json)에서 구분한다.', '',
        '## 한계와 채택', '',
        '세 seed는 한192장 데이터셋을 반복 평가한 결과다. 원 recording cluster bootstrap과 session 감도는 기술적 불확실성으로, 과거 반복 선택·domain gap·적은 recording 수를 보정하지 않는다. 물리 재료 모서리 소속은 가상 코너 오차만으로 증명하지 않는다. 결과를 근거로 운영 기본값이나 논문을 자동 수정하지 않았다. 추가 캠페인은 시작하지 않는다.', '']
    (public/'REPORT_KO.md').write_text('\n'.join(report))
    (public/'README.md').write_text('# B·I 조건부 국소 관측 보정\n\n[한국어 보고서](REPORT_KO.md) · [모든 결과](FULL_RESULTS.md) · [전체192장 갤러리](gallery/index.html) · [전체 수치 CSV](COMPARISON_ALL_ARMS.csv) · [paired](PAIRED_RESULTS.md) · [관측](OBSERVATION_RESULTS.md) · [진단](DIAGNOSTIC_RESULTS.md) · [재현](REPRODUCE.md)\n\n같은 정보·후단에서 점수망의 명시적 B/I 조건 추가 효과를 검사한 고정6fit 실험이다. 성공할 때까지 설정을 바꾸지 않았다. N0·기존 E0/H1·두 PoseFix는 별도 frozen 전체 시스템 비교다.\n\n![paired transitions](figures/paired_success_transitions.png)\n')
    return dict(status='ARTIFACT_BASE_COMPLETE',public_files=len(list(public.rglob('*'))))


def main():
    p=argparse.ArgumentParser();p.add_argument('--run-dir',type=Path,required=True);p.add_argument('--public',type=Path,required=True);a=p.parse_args();print(run(a.run_dir,a.public))


if __name__=='__main__':main()
