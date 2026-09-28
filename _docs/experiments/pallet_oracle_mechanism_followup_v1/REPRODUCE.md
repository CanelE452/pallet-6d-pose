# 재현·재개 안내

## 실행 환경과 범위

기준 checkout은 `cf52dc624fb5c83e305ff3b951f3c362b9ca1842`, branch `main`이다. 실제 실행은 Ubuntu, RTX3080 10GB, torch2.1.1+cu118의 기존 `pallet-yolo26` 환경을 사용했다. sandbox의 CUDA 장치 접근 실패와 host GPU 상태를 구분했고, host 실행 권한을 받아 사용했다. 재부팅·드라이버 변경·다른 프로세스 종료는 하지 않았다.

공개 Git에는 코드·aggregate 결과·기존 공개 사례 이미지·hash binding만 있다. 원본 RGB·개별 좌표·K/pose 배열·새 checkpoint는 `data/pallet/results/pallet_oracle_mechanism_followup_v1/`에 비공개로 남는다. 따라서 공개 저장소만으로 모든 numerical 결과를 처음부터 다시 만들 수 있다고 주장하지 않는다. 아래 재현에는 해당 권한 있는 원자료와 기존 환경이 필요하다. 다운로드나 새 설치를 자동 수행하지 않는다.

기존 결과를 삭제하여 재학습을 유도하지 않는다. 결과가 있는 경우 checkpoint/입력 hash를 확인하고 재사용한다. 미완료 fit/log가 있으면 원인을 확인하고 같은 명세의 resume를 별도 처리하며 다른 seed로 조용히 재시작하지 않는다. 완성된 학습을 재개 때 다시 돌리지 않는다. GPU fit은 한 번에 하나만 실행한다.

## 읽기·감사 우선

저장소 root에서 기존 Python 환경을 활성화한 뒤:

```bash
python -m scripts.research.pallet_oracle_mechanism_followup_v1.prepare
python -m unittest discover -s scripts/research/pallet_oracle_mechanism_followup_v1 -t . -p 'test_*.py' -v
python -m pytest scripts/research/pallet_material_selftrain_closure_v1/test_pair.py scripts/research/pallet_type_selftrain_v1/test_contract.py -q
python -m scripts.research.pallet_oracle_mechanism_followup_v1.final_audit
```

`prepare`는 입력 lock이 있으면 hash 검증만 한다. `final_audit`는 기존 baseline/원고/GT 보존과 현재 산출물을 확인한다. 공개 데이터만 있는 환경에서는 private artifact 관련 검사가 실행 불가능할 수 있으며 PASS로 바꾸지 않는다. 실제 실행 결과는 `AUDIT.json` 및 `RESOURCE_LEDGER.json`을 읽는다.

## 처음 실행할 때의 의존 순서

이미 완료된 이 checkout에서는 다음 명령을 일괄 재실행할 필요가 없다. 단계별 guard와 파일/해시를 먼저 확인한다. oracle freeze와 scoring을 분리한 이유는 GT가 선택 규칙으로 들어가지 않게 하기 위해서다.

```bash
# Frozen native TRAIN prediction: Wood only; Plastic existing cache reused.
python -m scripts.research.pallet_oracle_mechanism_followup_v1.train_follow infer
python -m scripts.research.pallet_oracle_mechanism_followup_v1.train_follow analyze

# CPU frozen candidate diagnostics: no student fit.
python -m scripts.research.pallet_oracle_mechanism_followup_v1.pose_oracle freeze
python -m scripts.research.pallet_oracle_mechanism_followup_v1.pose_oracle score
python -m scripts.research.pallet_oracle_mechanism_followup_v1.pose_oracle sanity
python -m scripts.research.pallet_oracle_mechanism_followup_v1.pose_cues
python -m scripts.research.pallet_oracle_mechanism_followup_v1.pose_sanity_detail
python -m scripts.research.pallet_oracle_mechanism_followup_v1.coordinate_oracle freeze
python -m scripts.research.pallet_oracle_mechanism_followup_v1.coordinate_oracle score

# Frozen models, optimizer step 0; GPU.
python -m scripts.research.pallet_oracle_mechanism_followup_v1.signal_diagnostic PLASTIC
python -m scripts.research.pallet_oracle_mechanism_followup_v1.signal_diagnostic WOOD

# C1: same-candidate robust score, CPU, zero fits.
python -m scripts.research.pallet_oracle_mechanism_followup_v1.cycle_pose

# C2: requires its locked SPEC before preflight. Four fits total, not per invocation.
python -m scripts.research.pallet_oracle_mechanism_followup_v1.cycle_affine preflight
python -m scripts.research.pallet_oracle_mechanism_followup_v1.cycle_affine train-all
python -m scripts.research.pallet_oracle_mechanism_followup_v1.cycle_affine_eval infer
python -m scripts.research.pallet_oracle_mechanism_followup_v1.cycle_affine_eval score
python -m scripts.research.pallet_oracle_mechanism_followup_v1.cycle_affine_box_audit

# C3: same existing manual support, exactly two fits. Explicit GPU opt-in.
python -m scripts.research.pallet_oracle_mechanism_followup_v1.cycle_manual prepare
python -m scripts.research.pallet_oracle_mechanism_followup_v1.cycle_manual preflight
python -m scripts.research.pallet_oracle_mechanism_followup_v1.cycle_manual train-all --allow-gpu
python -m scripts.research.pallet_oracle_mechanism_followup_v1.cycle_manual infer --allow-gpu
python -m scripts.research.pallet_oracle_mechanism_followup_v1.cycle_manual score

# Measured graphics; existing publication IDs only, no generated imagery.
python -m scripts.research.pallet_oracle_mechanism_followup_v1.figures
python -m scripts.research.pallet_oracle_mechanism_followup_v1.final_summary
```

C3의 원자료 export는 기존 direct-click source 필드와 teacher 당시 annotation hash를 검증한다. RAW9도 manual support 위치를 공유한다. 같은 수동38점의 직접 사용 경로를 완전 무감독으로 표기하지 않는다. 실제 augment support parity가 어긋나면 학습을 강행하지 않는다.

C3 `score`는 최초 기본 보고서도 생성한다. 최종 한국어 설명·보조표는 저장 JSON/CSV를 확인해 덧붙였으므로 완료 보고서를 `report`로 덮어쓰지 않는다. 원시 수치와 동일하게 자동 생성하는 최종 비교표는 `final_summary`에 있다. ledger는 실행자가 실제 완료 단계만 누적하며 GPU 실패를 fit으로 바꾸거나 이미 완료한 fit을 다시 합산하지 않는다.

## 집계와 정보 계약

- native2D 평가: 기존 고정 모집단·전체 코너 분모·matching·whole-object symmetry·실패 penalty 유지. fixed-ID oracle/verified66은 별도 열이다.
- AUC: ADDsym/object-diameter, threshold0–0.1을1001개 점으로 사다리꼴 적분한 기존 normalized AUC. 결측을 제외하지 않는다.
- 실제 D9 solver는 corner0..7로 SQPnP+LM을 풀고 residual에는 center를 포함한9점을 사용한다. 이름만 보고9점 solver라고 재해석하지 않는다.
- native TRAIN 잔차는 pseudo-target imitation이며 물리적 GT 오차가 아니다. 원영상과 실제 augmented input은 별도 진단이다.
- 모든 oracle/평가 참조는 scoring 또는 diagnostic-only 경로다. GT로 선택한 출력·가중치·후보를 production 학습에 쓰지 않는다.
- 과거 논문 main과 PDF는 보존한다. 새 `CLAIM_IMPACT.md`는 사후 수정 제안이며 원고를 조용히 고치거나 재빌드하지 않는다.

## 자원·기술 실패·공개

공통 상한은3사이클/12fits/7,680updates/GPU6h/wall10h다. GPU seconds는 각 계산 함수의 계측 사용 구간이며 전체 연구 elapsed wall과 구분한다. import/startup과 계측되지 않은 기술 실패는0으로 꾸미지 않고 NA로 남긴다. 실패 원인·재실행은 `EXPERIMENT_LOG.md`, 실제 checkpoint/trace는 각 cycle 결과와 ledger에 연결한다.

그림은 `FIGURE_MANIFEST.json`의 기존 공개 ID와 RGB SHA를 확인한다. 새 originalRGB/좌표/가중치를 stage하지 않는다. 관련 namespace만 `git diff --cached --stat` 및 실제 diff를 검토하여 커밋하고, `git push origin HEAD` 후 로컬/원격 SHA를 확인한다. 사용자의 기존 미추적 파일은 그대로 둔다.
