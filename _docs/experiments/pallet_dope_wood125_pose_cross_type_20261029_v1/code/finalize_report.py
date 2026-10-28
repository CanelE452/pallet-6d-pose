"""Create the requested README and immutable after-hash report."""
from pathlib import Path
import json
import csv
import hashlib

OUT=Path(__file__).resolve().parents[1]
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
    return h.hexdigest()
def read(p):return json.loads(p.read_text())
def rows(p):
    with p.open(newline='') as f:return list(csv.DictReader(f))
def save(p,v):
    with p.open('x') as f:json.dump(v,f,ensure_ascii=False,indent=2);f.write('\n')

def main():
    assert read(OUT/'gates/REPRODUCTION_PASS.json')['passed']
    assert read(OUT/'gates/B_SCORE_COMPLETE.json')['passed']
    assert read(OUT/'gates/INDEPENDENT_CODE_QA.json')['passed']
    assert read(OUT/'gates/INDEPENDENT_NUMERICAL_QA.json')['passed']
    before=read(OUT/'reproducibility/INPUT_HASHES_BEFORE.json')['files']
    after={p:sha(p) for p in before};bad=[p for p in before if before[p]!=after[p]]
    assert not bad,bad
    save(OUT/'reproducibility/INPUT_HASHES_AFTER.json',dict(passed=True,checked_count=len(after),mismatches=[],files=after))
    aa=rows(OUT/'results/A_WOOD125_POSE_PAIRED_CI.csv');bb=rows(OUT/'results/B_CROSS_TYPE_WOOD125.csv')
    cp=read(OUT/'reproducibility/CHECKPOINTS.json');bindings=read(OUT/'reproducibility/COMMIT_INPUTS.json')
    t=('# 목재125장 자세 차이 및 교차 종류 평가\n\n'
        '[확인] 기존 `7a865ce`의 목재125장(기존45+승인80), 평가 기준점963개를 그대로 사용했다. '
        '추가 학습·PnP·체크포인트 선택·임계값 변경은 없었다. 신규 추론은 PLASTIC_R2와 GREEN_R2에 승인된80장씩 총160회이며, '
        '기존45장과 WOOD_R2 전체125장은 저장 예측을 재사용했다.\n\n'
        '## 게이트와 재현 확인\n\n'
        '[확인] 지정 커밋·파일·열·모델 이름이 존재하고 로컬 파생 CSV와 커밋 blob이 바이트 단위로 일치한다 '
        '(`gates/PREFLIGHT.json`, `reproducibility/COMMIT_INPUTS.json`). 요청한 새 WOOD125 자세 CI와 교차 종류125장 결과는 '
        '지정 커밋/통합 산출물 및 pallet-research의 기존 paired-CI/cross-type 파일 후보 검사에서 발견되지 않았다. '
        '기존45장 교차 종류 결과는 재현 기준으로 사용했다.\n\n'
        '[확인] 원 `b_pose_metrics.bootstrap_comparison` 함수로 직사각형·정사각형의 R1/R2 및 목재 기존45장의 R1/R2 '
        '총24행을 재계산했다. 표본 n·ID 순서 SHA·복원추출 가중치 SHA·판정이 정확히 일치했고 수치 최대절대차는 '
        '`1.7763568394002505e-15`였다. 수치 검사 허용치는 사전에 `1e-12`로 고정했고 '
        'ID·가중치·판정에는 허용치를 적용하지 않았다 (`gates/A_REPRODUCTION.csv`).\n\n'
        '[확인] WOOD_R2의 재채점은 **600/963 = 0.6230529595015576**으로 커밋 값과 정확히 일치했다. '
        '원45장에서도 WOOD_R2 208/346, PLASTIC_R2 166/346, GREEN_R2 172/346을 재현했다 '
        '(`gates/B_WOOD125_REPRODUCTION.json`, `gates/B_ORIGINAL45_REPRODUCTION.csv`). '
        'WOOD_R1/R2는 이전 a4c8904 선언 SHA와 같고 실제 가중치 파일 SHA도 대조했다.\n\n'
        '## A. 목재125장 자세 오차: 자기학습 − 합성 전용 대조군\n\n'
        '각 지표에서 두 모델 모두 값이 존재하는 같은 영상만 사용한다. 음수는 자기학습 오차가 작다는 뜻이다. '
        '원 함수·원 지표별 공통표본/행 순서를 유지한 대응 영상 bootstrap 10,000회, PCG64 seed20261026, '
        '선형 백분위95% CI이며 다중 비교 보정은 없다.\n\n'
        '| 차수 | 지표 | n | 평균 차이 | 95% CI | 판정 |\n|---|---|---:|---:|---|---|\n')
    labels={'translation_error_m':'이동 오차 (m)','relative_translation_error':'상대 이동 오차 (무차원)',
        'rotation_indexed_deg':'인덱스 회전 오차 (°)','rotation_symmetry_deg':'대칭 회전 오차 (°)'}
    for r in aa:
        rnd=r['candidate_role'][-1];fmt='.6f' if r['metric'] in ['translation_error_m','relative_translation_error'] else '.4f'
        d=float(r['mean_difference_candidate_minus_comparator']);lo=float(r['ci95_low']);hi=float(r['ci95_high'])
        t+=f"| R{rnd} | {labels[r['metric']]} | {r['n_common']} | {d:+{fmt}} | [{lo:{fmt}}, {hi:{fmt}}] | {r['decision']} |\n"
        if r['metric']=='translation_error_m':
            t+=f"| R{rnd} | 이동 오차 (cm, 위 행 단위 변환) | {r['n_common']} | {float(r['mean_difference_cm']):+.4f} | [{float(r['ci95_low_cm']):.4f}, {float(r['ci95_high_cm']):.4f}] | {r['decision']} |\n"
    t+=('\n[확인] 두 차수 모두 이동·상대 이동 오차의 차이는 불명확했다. 회전 오차는 R1·R2에서 자기학습이 작았다. '
        '이는 공통 성공 표본 조건부 결과이며, 회전 가설이 다르면 NA로 유지했다. '
        '기존2D 기준점·등록 치수·K에서 재구성한 참조에 대한 **보조 자세 평가**로, 독립 실측6D 정답에 대한 정확도로 해석하지 않는다. '
        '추가80장의 K quality UNKNOWN, signed pose UNCONFIRMED 및 녹화 중복 한계도 그대로 유지된다.\n\n'
        '## B. 다른 종류로 학습한 R2 모델의 목재125장 평가\n\n'
        '| 학습 종류 / 모델 | 평가 영상 | 기준점 | PCK@10 (%) | 검출률 (%) | 조건부 정밀도@10 (%) | 정답점 / 관측점 |\n'
        '|---|---:|---:|---:|---:|---:|---|\n')
    baseline=read(OUT/'gates/B_WOOD125_REPRODUCTION.json')['summary']
    for r in [baseline,*bb]:
        t+=f"| {r['model']} | {r['n_DEV']} | {r['n_reference_points']} | {100*float(r['PCK10']):.4f} | {100*float(r['coverage']):.4f} | {100*float(r['conditional_precision10']):.4f} | {r['n_correct10']} / {r['n_observed_reference_predictions']} |\n"
    t+=('\n[확인] PCK는 원본 좌표에서 오차≤10px인 점/전체963점이며, 누락은 실패다. 검출률은 평가 기준점 중 관측된 점의 비율이다. '
        '의미 채널0–7만 사용하고 중심점과 대칭 재정렬은 제외했다. 원 Gaussian σ2 국소최대 `sm > 0.3`, 반경5 및 '
        '0.4395 오프셋 디코더를 그대로 사용하며, 저장된 raw confidence에 추가 임계값을 적용하지 않았다. '
        'K·영상 목록·정답·좌표·threshold를 바꾸지 않았다. B의 CI는 요청대로 계산하지 않았다.\n\n'
        '## 입력·체크포인트 SHA 및 코드\n\n'
        '| 모델 | 실제 체크포인트 SHA256 |\n|---|---|\n')
    for name,x in cp.items():t+=f"| {name} | `{x['sha256']}` |\n"
    t+='\n정확한 로컬 체크포인트 경로는 `reproducibility/CHECKPOINTS.json`에 있다.\n\n'
    t+='| 커밋 입력 | commit | SHA256 |\n|---|---|---|\n'
    for name,x in bindings.items():t+=f"| {name} | `{x['commit']}` | `{x['sha256']}` |\n"
    t+=('\n원 자세 CI 함수: `/home/minjae/Documents/pallet-research/dope-thesis-analysis-batch2-v1/code/b_pose_metrics.py:135–185`. '
        '원 채점: `/home/minjae/Documents/pallet-research/dope-wood640-eval80-20261027-v1/code/combined_metrics.py:114–169`. '
        '원 추론 절차: 같은 code의 `evaluate80.py:185–208`, '
        '`/home/minjae/Documents/github/pallet-pose/scripts/research/pallet_dope_refiner_20261001_v1/dope_adapter.py:68,140–161`. '
        '이번 래퍼는 `code/run_analysis.py`이며 원 함수들을 직접 import했다. 원본 코드는 수정하지 않았다.\n\n'
        '## 실행 이력·보존 확인\n\n'
        '[확인] 첫 B 실행은 base Python에 cv2가 없어 import 단계에서 멈췄다. 당시 새 forward는0이고 기존 입력109개는 그대로였다 '
        '(`GATE_STOP.json`). 원 과거 실행로그 `MERGED_COMMANDS.jsonl:2`에서 실제 사용한 '
        '`/home/minjae/anaconda3/envs/pallet-pose/bin/python`을 확인해 동일 래퍼를 그 환경에서 실행했다. '
        '원 환경에서도 WOOD125 600/963을 재확인했다. 설치·코드·조건·입력 변경 없이 실행 환경만 복구했으며 '
        '이것은 과학적 재현 게이트 실패 후의 재시도가 아니다 (`gates/RUNTIME_RECOVERY.json`). '
        '독립 코드·수치 검증 기록도 함께 공개한다.\n\n'
        f'[확인] 기존 입력 **{len(after)}개**의 전후 SHA가 모두 같고 불일치0이다. '
        '체크포인트6개·승인80 RGB·예측/점수 캐시·메타데이터·원 코드를 포함한다. '
        '정답 좌표는 `score_predictions → combined_metrics.added_row` 채점에만 사용했다. '
        'forward 함수에는 정답 입력이 없으며, 임계값·체크포인트·사례 선택에도 사용하지 않았다. '
        '새 PnP·학습0, 원시 정답 열람0, 승인80 외 보호FINAL 열람0이다. '
        '입력·샘플·규칙 변경 여부는 SHA 및 원 함수 재사용으로 확인했고, 결과에 대한 원인 추정은 추가하지 않았다.\n\n'
        '입력 파일별 전체 SHA는 `reproducibility/INPUT_HASHES_BEFORE.json`과 `INPUT_HASHES_AFTER.json`, '
        '명령·seed·코드SHA와160회forward는 `logs/RUN_LOG.jsonl`에 있다. '
        '새 예측 좌표와 Git 발행 보조 파일은 `.private/`·`.publication/`에 로컬 보존하고 커밋에서 제외한다.\n')
    with (OUT/'README.md').open('x') as f:f.write(t)
    with (OUT/'.gitignore').open('x') as f:f.write('/.private/\n/.publication/\n')
    save(OUT/'COMPLETE.json',dict(complete=True,A_rows=8,B_models=2,B_frames_per_model=125,B_reference_points=963,
        input_files_unchanged=len(after),input_mismatches=0,all_scientific_gates_passed=True,
        code_QA=True,numerical_QA=True,training_updates=0,new_PnP_fits=0,new_NN_forwards=160,
        preserved_pre_forward_environment_failure=True,independent_6D_ground_truth=False))
    print('FINAL_REPORT_COMPLETE',len(after),flush=True)

if __name__=='__main__':main()
